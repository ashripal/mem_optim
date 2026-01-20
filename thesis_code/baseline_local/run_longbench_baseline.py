"""
Baseline tiered runner (No memory optimizations)
Tier 2 (Disk): LongBench JSONL + output logs
Tier 1 (RAM): simple in-process cache of loaded examples
Tier 0 (Compute): MPS (Apple GPU) if available, else CPU

This baseline intentionally does not do:
- retrieval indexing
- hierarchical promotion / eviction policies
- summarization / compression

Establishes tier boundary and log metrics
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Tier 1: RAM cache (very basic)
@dataclass
class RAMCache:
    # Small capped cache to show Tier 1 explicitly
    max_size: int = 256
    _store: Dict[str, Any] = None
    _order: List[str] = None

    def __post_init__(self) -> None:
        self._store = {}
        self._order = []

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            # bump recency
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            # bump recency
            self._order.remove(key)
        self._store[key] = value
        self._order.append(key)
        if len(self._store) > self.max_size:
            evict = self._order.pop(0)
            self._store.pop(evict, None)

def pick_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def rss_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def find_jsonl_files(longbench_repo_dir: Path) -> List[Path]:
    # scan for *.jsonl files
    files = sorted(longbench_repo_dir.glob("*.jsonl"))
    return files

def load_jsonl_as_dataset(path: Path):
    dataset = load_dataset("json", data_files=str(path), split="train")
    return dataset

def build_prompt(example: Dict[str, Any]) -> str:
    # LongBench standardized fields:
    # - example["context"] (long)
    # - example["input"] (question/instruction)
    # - example["answers"] (list)
    # See LongBench dataset card for format. :contentReference[oaicite:3]{index=3}
    context = example.get("context", "")
    question = example.get("input", "")
    # baseline template
    return (
        "Use the following context to answer the question.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer (be concise):"
    )

def generate_answer(
        model, 
        tokenizer,
        device: torch.device,
        prompt: str,
        max_new_tokens: int,
        max_input_tokens
) -> Tuple[str, Dict[str, Any]]:
    t0 = time.perf_counter()
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    # input_len = int(inputs["input_ids"].shape[-1])
    
    # tokenize with truncation guard
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    input_len = int(inputs["input_ids"].shape[-1])

    # for k in inputs:
    #     inputs[k] = inputs[k].to(device)
    model_device = next(model.parameters()).device
    for k in inputs:
        inputs[k] = inputs[k].to(model_device)

    # Approx TTFT: first token emission isn't directly exposed in non-streaming generation
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=input_len + max_new_tokens,
            do_sample=False,
            # temperature=0.0,
            use_cache=True
        )
    t1 = time.perf_counter()

    # decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    # answer = decoded[len(prompt):].strip()
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    answer = full_text[len(prompt_text):].strip()

    output_len = int(out.shape[-1] - input_len)
    dt = t1 - t0
    toks_per_sec = (output_len / dt) if dt > 0 else None

    metrics = {
        "input_tokens": input_len,
        "output_tokens": output_len,
        "latency_s": dt, 
        "tokens_per_sec": toks_per_sec,
        "truncated_to_max_input": input_len >= max_input_tokens,
        "device_used": str(model_device)
    }
    return answer, metrics

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier2_repo", default="tier2_disk/longbench_repo", help="Tier 2 LongBench repo folder")
    ap.add_argument("--out_dir", default="tier2_disk/runs", help="Tier 2 output logs folder")
    ap.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model ID")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_examples", type=int, default=20)
    ap.add_argument("--task_glob", default="", help="Optional filter for dataset files")
    ap.add_argument("--ram_cache_items", type=int, default=256)
    ap.add_argument("--max_input_tokens", type=int, default=2048, help="Clamp input length to avoid MPS OOM")
    ap.add_argument("--cpu_fallback_on_long", action="store_true", help="If input too long, run on CPU instead of truncating")
    args = ap.parse_args()

    tier2_repo = Path(args.tier2_repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"[device] Using device: {device}")
    print(f"[tier2] Repo dir: {tier2_repo}")
    print(f"[tier2] Out dir: {out_dir}")

    # ---- Tier 2: discover dataset files (disk) ----
    jsonl_files = find_jsonl_files(tier2_repo)
    if args.task_glob:
        jsonl_files = [f for f in jsonl_files if args.task_glob.lower() in f.name.lower()]

    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found under {tier2_repo}."
            "Inspect the folder or re-run download_longbench.py"
        )
    
    print(f"[tier2] Found {len(jsonl_files)} JSONL files after filtering.")
    for f in jsonl_files[:5]:
        # print(" -", f.relative_to(tier2_repo))
        print(" -", f.name)
    
    # ---- Tier 0: Load model to compute device ----
    print(f"[model ] loading model {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    # MPS works well with float16, CPU use float32
    dtype = torch.float16 if device.type == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype)
    model.to(device)
    model.eval()
    print(f"[model ] loaded.")

    # guardrail for max input tokens
    model_max = getattr(model.config, "max_position_embeddings", None)
    #effective_max = min(args.max_input_tokens, model_max) if model_max else args.max_input_tokens
    effective_max = args.max_input_tokens
    print(f"[cfg ] model max_position_embeddings: {model_max} | max_input_tokens set to: {effective_max}")

    ram_cache = RAMCache(max_size=args.ram_cache_items)
    # ram_cache = RAMCache()

    # ---- Output file (tier 2) ----
    run_id = int(time.time())
    out_path = out_dir / f"baseline_longbench_{run_id}.jsonl"
    print(f"[write ] {out_path}")

    total_seen = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        # iterate over JSONL tasks (Tier 2)
        for task_file in jsonl_files:
            if total_seen >= args.max_examples:
                break

            # Tier 1 cache key = filename
            cache_key = str(task_file)
            ds = ram_cache.get(cache_key)
            if ds is None:
                ds = load_jsonl_as_dataset(task_file)
                ram_cache.put(cache_key, ds)

            # run few examples from this task
            for example in ds:
                if total_seen >= args.max_examples:
                    break

                prompt = build_prompt(example)

                prompt_tokens = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False, truncation=False)
                approx_len = len(prompt_tokens["input_ids"])

                device_used = device
                if args.cpu_fallback_on_long and approx_len > effective_max:
                    device_used = torch.device("cpu")
                    print(f"[info ] example input tokens {approx_len} exceeds max {effective_max}, running on CPU")

                mem_before = rss_mb()
                status = "ok"
                try:
                    if device_used.type != next(model.parameters()).device.type:
                        model.to(device_used)

                    answer, gen_metrics = generate_answer(
                        model = model,
                        tokenizer = tokenizer,
                        device = device,
                        prompt = prompt,
                        max_new_tokens = args.max_new_tokens,
                        max_input_tokens = effective_max
                    )
                    # status = "ok"
                except RuntimeError as e:
                    status = f"runtime_error: {type(e).__name__}"
                    answer, gen_metrics = "", {"device_used": str(device_used)}
                mem_after = rss_mb()

                record = {
                    "task_file": str(task_file.relative_to(tier2_repo)),
                    "dataset_name": example.get("dataset", None),
                    "length_meta": example.get("length", None),
                    "prompt_chars": len(prompt),
                    "answers_gold": example.get("answers", None),
                    "answer_pred": answer,
                    "ram_rss_mb_before": mem_before,
                    "ram_rss_mb_after": mem_after,
                    "approx_prompt_tokens": approx_len,
                    **gen_metrics
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                total_seen += 1
    print(f"[done ] wrote {total_seen} examples to {out_path}")

if __name__ == "__main__":
    main()

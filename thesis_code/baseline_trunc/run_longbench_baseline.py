"""
Baseline tiered runner (No memory optimizations)

Tier 2 (Disk): LongBench JSONL + output logs
Tier 1 (RAM): in-process dataset cache (LRU-ish)
Tier 0 (Compute): MPS if available, else CPU

This baseline intentionally does not do:
- retrieval indexing
- hierarchical promotion / eviction
- summarization / compression

Purpose: establish tier plumbing + produce reproducible metrics + show failure modes
"""

from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import psutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tier 1: RAM cache
@dataclass
class RAMCache:
    max_size: int = 64
    _store: Dict[str, Any] = None
    _order: list[str] = None

    def __post_init__(self) -> None:
        self._store = {}
        self._order = []

    def get(self, key: str) -> Optional[Any]:
        if key in self._store:
            # Move to end to mark as recently used
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        if key in self._store:
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
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)

def find_jsonl_files(repo_dir: Path) -> list[Path]:
    return sorted(repo_dir.glob("*.jsonl"))

def load_jsonl_as_dataset(path: Path):
    return load_dataset("json", data_files=str(path), split="train")

def build_prompt(example: Dict[str, Any]) -> str:
    context = example.get("context", "")
    question = example.get("input", "")
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n"
        "Answer (be concise):"
    )

def generate_answer(
    model,
    tokenizer, 
    prompt: str,
    max_new_tokens: int, 
    max_input_tokens: int
) -> Tuple[str, Dict[str, Any]]:
    """
    Generation that is safe for MPS/CPU:
    - tokenizes with truncation to max_input_tokens
    - moves inputs to *model device* to avoid mismatch warnings
    """
    t0 = time.perf_counter()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )
    input_len = int(inputs["input_ids"].shape[-1])
    #truncated = input_len >= max_input_tokens
    hit_cap = (input_len == max_input_tokens)

    model_device = next(model.parameters()).device
    for k in inputs:
        inputs[k] = inputs[k].to(model_device)
    
    with torch.no_grad():
        model.generation_config.max_length = None
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            #max_length=int(inputs["input_ids"].shape[-1] + max_new_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    t1 = time.perf_counter()

    # full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    # answer = full_text[len(prompt_text):].strip()
    out_ids = outputs[0].detach().to("cpu")
    gen_ids = out_ids[input_len:]  # only new tokens
    answer = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()


    output_len = int(outputs.shape[-1] - input_len)
    dt = t1 - t0
    tokens_per_sec = (output_len / dt) if dt > 0 else None

    metrics = {
        "input_tokens": input_len,
        "output_tokens": output_len,
        "latency_s": dt,
        "tokens_per_sec": tokens_per_sec,
        "device_used": str(model_device),
        "hit_max_input_cap": hit_cap
    }
    return answer, metrics

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier2_repo", required=True, help="Path to LongBench dataset repo (Tier 2)")
    ap.add_argument("--out_dir", default="tier2_disk/runs", help="Output folder for results (Tier 2)")
    ap.add_argument("--model_id", default="microsoft/Phi-3-mini-128k-instruct", help="HuggingFace model ID")
    ap.add_argument("--task_glob", default="", help="Substring filter, e.g. trec")
    ap.add_argument("--max_examples", type=int, default=25)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_cache_items", type=int, default=64)
    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.add_argument("--cpu_fallback_on_long", action="store_true", help="If True, use CPU when MPS OOM occurs")
    args = ap.parse_args()

    tier2_repo = Path(args.tier2_repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"[device] Using device: {device}")
    print(f"[tier2] Repo: {tier2_repo}")
    print(f"[tier2] Output dir: {out_dir}")

    jsonl_files = find_jsonl_files(tier2_repo)
    if args.task_glob:
        jsonl_files = [f for f in jsonl_files if args.task_glob.lower() in f.name.lower()]

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {tier2_repo} matching '{args.task_glob}'")

    print(f"[tier2] Found {len(jsonl_files)} JSONL files after filtering.")
    for f in jsonl_files[:5]:
        print(f" - {f.name}")

    print(f"[model] Loading model '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    dtype = torch.float16 if device.type == "mps" else torch.float32
    # model_mps = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype, device_map=None).to("mps").eval()
    # model_cpu = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype, device_map=None).to("cpu").eval()
    # model = model_cpu if (args.cpu_fallback_on_long and prompt_tokens_used >= effective_max) else model_mps
    # model.to(device)
    # model.eval()
    # print(f"[model] Loaded model to {device} with dtype {dtype}.")
    # print(f"[model] Loading model '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    device = pick_device()
    dtype_mps = torch.float16 if device.type == "mps" else torch.float32

    # Load ONE model to MPS by default (fast iteration baseline)
    model_mps = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype_mps,
        device_map=None
    ).to(device).eval()

    # Optional CPU model (only if flag set)
    model_cpu = None
    if args.cpu_fallback_on_long:
        model_cpu = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.float32,
            device_map=None
        ).to("cpu").eval()

    model_max = getattr(model_mps.config, "max_position_embeddings", None)
    effective_max = min(args.max_input_tokens, model_max) if model_max else args.max_input_tokens
    print(f"[model] Loaded model to {device} with dtype {dtype_mps}.")
    print(f"[model] Effective max input tokens: {effective_max}")

    ram_cache = RAMCache(max_size=args.max_cache_items)

    run_id = int(time.time())
    out_path = out_dir / f"baseline_longbench_{run_id}.jsonl"
    print(f"[write] Output path: {out_path}")

    total_seen = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for task_file in jsonl_files:
            if total_seen >= args.max_examples:
                break

            cache_key = str(task_file)
            ds = ram_cache.get(cache_key)
            if ds is None:
                ds = load_jsonl_as_dataset(task_file)
                ram_cache.put(cache_key, ds)

            for example in ds:
                if total_seen >= args.max_examples:
                    break

                # prompt = build_prompt(example)
                messages = [{"role": "user", "content": build_prompt(example)}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                # approx prompt length (for routing + logs)
                # token_ids = tokenizer(
                #     prompt,
                #     add_special_tokens=True,
                #     return_attention_mask=False,
                #     truncation=False
                # )["input_ids"]
                # approx_len = len(tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=effective_max+1)["input_ids"])
                # too_long = approx_len > effective_max
                total_cap = 20000
                prompt_tokens_total = len(tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=total_cap, return_attention_mask=False)["input_ids"])

                prompt_tokens_used = len(tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=effective_max, return_attention_mask=False)["input_ids"])

                was_truncated = (prompt_tokens_total > prompt_tokens_used) or (prompt_tokens_used == effective_max)
                
                # device_used = device

                # if args.cpu_fallback_on_long and prompt_tokens_used > effective_max:
                #     device_used = torch.device("cpu")
                #     print(f"[info] example input tokens exceed model max ({approx_len} > {model_max}), using CPU")

                # if device_used.type != next(model.parameters()).device.type:
                #     model.to(device_used)
                #     print(f"[info] moved model to {device_used} for this example")
                model = model_mps
                if args.cpu_fallback_on_long and (prompt_tokens_used >= effective_max) and (model_cpu is not None):
                    model = model_cpu
                    print(f"[info] switched model to CPU for this example")

                mem_before = rss_mb()
                status = "ok"
                try:
                    answer, gen_metrics = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        max_input_tokens=effective_max
                    )
                except RuntimeError as e:
                    # status = f"runtime_error:{type(e).__name__}"
                    # answer, gen_metrics = "", {"device_used": str(next(model.parameters()).device)}
                    status = "runtime_error"
                    answer = ""
                    gen_metrics = {
                        "device_used": str(next(model.parameters()).device),
                        "error": repr(e)[:500]
                    }

                mem_after = rss_mb()

                record = {
                    "status": status,
                    "task_file": task_file.name,
                    # "approx_prompt_tokens": approx_len,
                    "prompt_tokens_total_capped": prompt_tokens_total,
                    "prompt_tokens_used": prompt_tokens_used,
                    "was_truncated": was_truncated,
                    "prompt_chars": len(prompt),
                    "answers_gold": example.get("answers", None),
                    "answer_pred": answer,
                    "ram_rss_mb_before": mem_before,
                    "ram_rss_mb_after": mem_after,
                    **gen_metrics
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                total_seen += 1
    
    print(f"[done] Wrote {total_seen} records to {out_path}")

if __name__ == "__main__":
    main()
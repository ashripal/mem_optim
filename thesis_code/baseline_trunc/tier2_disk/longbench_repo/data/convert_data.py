"""
One-time dataset conversion script to convert the original LongBench JSONL files into a format that is more suitable for our experiments. This includes:
- reading LongBench JSONL files
- creating files with the original format, exact repeated questions that test cache hits, and transformer-generated paraphrases

Output JSONL (one line per turn):
{
  "session_id": str,
  "turn_id": int,
  "variant_type": "original" | "repeat" | "paraphrase",
  "task_file": str,
  "example_index": int,
  "context": str,
  "input": str,
  "answers": list[str],
  "source_input": str
}

Example run: 
  python prepare_longbench_sessions_transformers.py \
    --longbench_dir tier2_disk/longbench_repo/data/data \
    --task_glob trec \
    --out_jsonl tier2_disk/sessions/trec_sessions.jsonl \
    --max_examples 200 \
    --repeats 2 \
    --paraphrases 2 \
    --model_id google/flan-t5-large \
    --device cuda
"""
from __future__ import annotations
import argparse
import json
import re
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig

# finds all JSONL files in the given directory, optionally filtering by a glob pattern (e.g., "trec" to only include files with "trec" in the name)
def find_jsonl_files(longbench_dir: Path, task_glob: str) -> List[Path]:
    files = sorted(longbench_dir.glob("*.jsonl"))
    if task_glob:
        files = [f for f in files if task_glob.lower() in f.name.lower()]
    return files

# loads a JSONL file using HuggingFace Datasets, returning a Dataset object
def load_jsonl(path: Path):
    return load_dataset("json", data_files=str(path), split="train")

# safely creates the parent directory for the given output path if it doesn't already exist
def safe_mkdir(out_dir: Path):
    out_dir.parent.mkdir(parents=True, exist_ok=True)

def soft_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+([?.!,;:])", r"\1", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s



# normalizes a question string by stripping leading/trailing whitespace and collapsing multiple internal spaces into a single space
def normalize_question(question: str) -> str:
    q = question.strip()
    q = re.sub(r"\s+", " ", q)
    # remove spaces before punctuation like " ?"
    q = re.sub(r"\s+([?.!,;:])", r"\1", q)
    # ensure ends with ? or .
    if q and q[-1] not in ["?", "."]:
        q += "?"
    return q

# extracts lines from a given text string, removing leading numbering (e.g., "1) ...", "- ...", "• ...") and stripping whitespace. Returns a list of cleaned lines.
# def extract_lines(text: str) -> List[str]:
#     if not text:
#         return []
#     lines = []
#     for line in text.splitlines():
#         line = line.strip()
#         line = re.sub(r"^\d+[\)\.\-]\s*", "", line) # "1) ..."
#         line = re.sub(r"^[\-\u2022]\s*", "", line)
#         line = line.strip()
#         if line:
#             lines.append(line)
#     return lines
def extract_lines(text: str) -> List[str]:
    if not text:
        return []
    
    # normalize separators the model might use
    text = text.replace(";", "\n")
    text = text.replace(" | ", "\n")

    # if the model returned everything in one line, try to split by "?"
    if "\n" not in text and text.count("?") >= 2:
        parts = [p.strip() for p in text.split("?") if p.strip()]
        text = "\n".join([p + "?" for p in parts])
    
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[\)\.\-]\s*", "", line) # "1) ..."
        line = re.sub(r"^[\-\u2022]\s*", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    return lines

class Paraphraser:
    """
    Uses a seq2seq model (e.g., T5) to generate paraphrases of input questions. 
    """
    def __init__(self, model_id: str, device: str, max_new_tokens: int, seed: int) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens

        random.seed(seed)
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cfg = getattr(self.tokenizer, "model_input_names", None)
        conf = AutoConfig.from_pretrained(model_id)
        is_seq2seq = conf.is_encoder_decoder

        if is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but not available.")
        
        self.model.to(torch.device(device))
        self.model.eval()

    @torch.no_grad()
    def paraphrase(self, question: str, n: int) -> List[str]:
        q = normalize_question(question)
        # prompt = (
        #     f"Generate {n} paraphrases of the following question. The paraphrases should have the same meaning but different wording. Avoid simply reordering words or making minor punctuation changes.\n\n"
        #     f"Keep the meaning the same, but try to use different words and structure. "
        #     f"Return each paraphrase on a new line, without numbering or bullet points.\n\n"
        #     f"Question:\n{q}\n\n"
        # )
        prompt = (
            f"Generate {n} meaning-preserving paraphrases of the following question.\n"
            f"Each paraphrase must use different wording and structure.\n"
            f"Do NOT copy the original question.\n"
            f"Output one paraphrase per line.\n\n"
            f"{q}"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=self.max_new_tokens,
        #     do_sample=False,
        #     num_beams=4,
        #     early_stopping=True
        # )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            num_beams=max(8, n*4),
            # num_beam_groups=n,
            temperature=0.9,
            top_p=0.9,
            # diversity_penalty=0.8,
            top_k=50,
            num_return_sequences=n,
            repetition_penalty=1.2
        )

        # text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # text = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        # # candidates = extract_lines(text)
        # candidates = []
        # for t in text:
        #     candidates.extend(extract_lines(t))
        # texts = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        texts = []
        for o in outputs:
            decoded = self.tokenizer.decode(o, skip_special_tokens=True)
            # if causal model, strip prompt prefix
            if not getattr(self.model.config, "is_encoder_decoder", False):
                prompt_decoded = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                if decoded.startswith(prompt_decoded):
                    decoded = decoded[len(prompt_decoded):].strip()

            texts.append(decoded)

        candidates = []
        for t in texts:
            candidates.extend(extract_lines(t))

        # clean + deduplicate + remove identical questions
        cleaned: List[str] = []
        seen = set()
        for c in candidates:
            c_norm = normalize_question(c)
            if not c_norm:
                continue
            if soft_key(c_norm) == soft_key(q):
                continue
            # if c_norm in seen:
            #     continue
            # seen.add(c_norm)
            c_key = soft_key(c_norm)
            if c_key in seen:
                continue
            seen.add(c_key)
            cleaned.append(c_norm)
            if len(cleaned) >= n:
                break
        fallbacks = [
            "In other words, {q}",
            "Can you answer this question: {q}",
            "Please explain: {q}",
            "I want to know: {q}"
        ]
        i = 0
        while len(cleaned) < n:
            f = normalize_question(fallbacks[i % len(fallbacks)].format(q=q))
            if soft_key(f) != soft_key(q) and soft_key(f) not in {soft_key(x) for x in cleaned}:
                cleaned.append(f)
            i += 1
        return cleaned[:n]
    
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--longbench_dir", required=True, help="Path to LongBench data/data directory containing *.jsonl")
    ap.add_argument("--task_glob", default="", help="Substring filter (e.g., trec, qasper)")
    ap.add_argument("--out_jsonl", required=True, help="Output sessions JSONL")
    ap.add_argument("--max_examples", type=int, default=200, help="Max examples per task file to expand")
    ap.add_argument("--repeats", type=int, default=2, help="Exact repeats after original")
    ap.add_argument("--paraphrases", type=int, default=2, help="Paraphrase turns per example")
    ap.add_argument("--seed", type=int, default=0)

    # Transformers paraphraser config
    ap.add_argument("--model_id", default="google/flan-t5-large", help="Seq2seq model for paraphrasing (e.g., google/flan-t5-large)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for paraphrasing model (e.g., cuda or cpu)")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate for each paraphrase")

    args = ap.parse_args()
    random.seed(args.seed)

    longbench_dir = Path(args.longbench_dir).resolve()
    out_path = Path(args.out_jsonl).resolve()
    safe_mkdir(out_path)

    files = find_jsonl_files(longbench_dir, args.task_glob)
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {longbench_dir} matching glob '{args.task_glob}'")
    
    paraphraser = Paraphraser(args.model_id, args.device, args.max_new_tokens, args.seed)

    total_sessions = 0
    total_turns = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for task_file in files:
            ds = load_jsonl(task_file)

            max_n = min(args.max_examples, len(ds))
            indices = list(range(len(ds)))
            random.shuffle(indices)
            indices = indices[:max_n]

            for idx in tqdm(indices, desc=f"task={task_file.name}"):
                ex = ds[idx]
                context = ex.get("context", "")
                raw_q = ex.get("input", "") or ""

                # Strip known TREC formatting like "Question: ... Type: ..."
                raw_q = re.sub(r"(?i)^\s*question:\s*", "", raw_q).strip()
                raw_q = re.sub(r"(?i)\s*type:\s*.*$", "", raw_q).strip()

                # Fix common spacing artifacts
                raw_q = re.sub(r"\s+([?.!,;:])", r"\1", raw_q)
                raw_q = re.sub(r"\s+,", ",", raw_q)

                source_q = normalize_question(raw_q)

                answers = ex.get("answers", None)

                # stable session id across runs given same seed
                session_id = f"{task_file.stem}_{idx}_seed{args.seed}"
                turn_id = 0

                def emit(turn_q: str, variant: str) -> None:
                    nonlocal total_turns
                    rec = {
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "variant_type": variant,
                        "task_file": task_file.name,
                        "example_index": idx,
                        "context": context,
                        "input": normalize_question(turn_q),
                        "answers": answers,
                        "source_input": source_q
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_turns += 1

                # turn 0: original question
                emit(source_q, "original")
                total_sessions += 1

                # exact repeats
                for _ in range(args.repeats):
                    turn_id += 1
                    emit(source_q, "repeat")

                # paraphrases
                if args.paraphrases > 0 and paraphraser is not None:
                    paras = paraphraser.paraphrase(source_q, args.paraphrases)
                    for p in paras:
                        turn_id += 1
                        emit(p, "paraphrase")
    print(f"Done! Wrote {total_turns} turns across {total_sessions} sessions to {out_path}")
    # f_out.close()

if __name__ == "__main__":
    main()



        
                      

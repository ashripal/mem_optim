#!/usr/bin/env python3
"""
run_longbench_baseline.py

A clean, single-file runner that supports **three** evaluation modes on LongBench:

1) baseline_llm
   - Always call the LLM (no memory).
2) baseline_context_only
   - Never call the LLM; return dataset-provided answer(s) only.
3) memory_qacache
   - Build a lightweight RAM/Disk “memory”:
     - Embed each (question) into a vector
     - Search for similar past questions
     - If similarity >= threshold: return cached answer (bypass LLM)
     - Else: call LLM, then store (q, a, embedding) into SQLite + RAM index

This aligns with your goal: show that “similar questions” can reduce latency by hitting memory.

------------------------------------------------------------
Expected input (LongBench JSONL rows):
- context: str
- input: str
- answers: list[str] (gold answers)
------------------------------------------------------------
Output:
Writes JSONL records to out_dir/runs_*.jsonl with per-example metrics and routing.

------------------------------------------------------------
Install deps (conda/pip):
  pip install transformers datasets psutil numpy

Optional (recommended for faster embedding):
  pip install torch

Notes:
- Embeddings are computed with a small encoder model via Transformers (mean pooling).
- LLM generation uses Transformers causal LM.
- We do NOT cache the dataset itself in RAM anymore (only QA/embeddings).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import psutil
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

# -------------------------
# Utilities
# -------------------------

def pick_device(prefer: str = "mps") -> torch.device:
    if prefer == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def rss_mb() -> float:
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)

def normalize_ws(s: str) -> str:
    return " ".join((s or "").strip().split())

def soft_key(s: str) -> str:
    """
    Soft normalization for exact-match / near-exact-match keys.
    """
    s = (s or "").lower().strip()
    s = " ".join(s.split())
    # remove spaces before punctuation
    for ch in ["?", ".", ",", "!", ":", ";"]:
        s = s.replace(f" {ch}", ch)
    return s

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def find_jsonl_files(longbench_dir: Path, task_glob: str) -> List[Path]:
    files = sorted(longbench_dir.glob("*.jsonl"))
    if task_glob:
        files = [f for f in files if task_glob.lower() in f.name.lower()]
    return files

def iter_dataset_rows(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    # Using HF datasets to read jsonl reliably
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")
    for row in ds:
        yield row

# -------------------------
# Disk memory (SQLite)
# -------------------------

class MemoryDB:
    """
    SQLite-backed QA memory.

    Stores:
      - q_key (soft key)
      - question
      - answer
      - embedding (BLOB float32)
      - created_ts

    Also stores a simple hash index to speed up exact matches.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_memory (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              q_hash TEXT NOT NULL,
              q_key  TEXT NOT NULL,
              question TEXT NOT NULL,
              answer   TEXT NOT NULL,
              emb BLOB NOT NULL,
              dim INTEGER NOT NULL,
              created_ts REAL NOT NULL
            );
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_q_hash ON qa_memory(q_hash);")
        self.conn.commit()

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def _from_blob(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != dim:
            # Defensive; return best-effort
            return arr.astype(np.float32)
        return arr

    def insert(self, question: str, answer: str, emb: np.ndarray) -> int:
        qk = soft_key(question)
        qh = sha1_hex(qk)
        blob = self._to_blob(emb)
        dim = int(emb.shape[0])
        ts = time.time()
        cur = self.conn.execute(
            "INSERT INTO qa_memory(q_hash, q_key, question, answer, emb, dim, created_ts) VALUES (?,?,?,?,?,?,?)",
            (qh, qk, question, answer, blob, dim, ts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_exact(self, question: str) -> Optional[Tuple[int, str]]:
        """
        Exact match via q_hash + q_key.
        Returns (id, answer) if found.
        """
        qk = soft_key(question)
        qh = sha1_hex(qk)
        cur = self.conn.execute(
            "SELECT id, answer FROM qa_memory WHERE q_hash=? AND q_key=? ORDER BY id DESC LIMIT 1",
            (qh, qk),
        )
        row = cur.fetchone()
        if not row:
            return None
        return int(row[0]), str(row[1])

    def load_all_embeddings(self) -> Tuple[List[int], np.ndarray, List[str]]:
        """
        Load all embeddings for in-RAM similarity search.
        Returns (ids, matrix [N,D], answers)
        """
        cur = self.conn.execute("SELECT id, emb, dim, answer FROM qa_memory ORDER BY id ASC")
        ids: List[int] = []
        answers: List[str] = []
        vecs: List[np.ndarray] = []
        for rid, blob, dim, ans in cur.fetchall():
            v = self._from_blob(blob, int(dim))
            ids.append(int(rid))
            answers.append(str(ans))
            vecs.append(v)
        if not vecs:
            return [], np.zeros((0, 0), dtype=np.float32), []
        mat = np.vstack([v.reshape(1, -1) for v in vecs]).astype(np.float32)
        return ids, mat, answers

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

# -------------------------
# RAM index (embeddings + answers)
# -------------------------

@dataclass
class RAMIndex:
    """
    Holds embeddings and answers in RAM for fast cosine similarity.
    This is the "RAM tier" for QA + embedding cache.
    """
    ids: List[int]
    emb_matrix: np.ndarray  # shape [N, D], float32
    answers: List[str]

    @staticmethod
    def empty() -> "RAMIndex":
        return RAMIndex(ids=[], emb_matrix=np.zeros((0, 0), dtype=np.float32), answers=[])

    def add(self, rid: int, emb: np.ndarray, answer: str) -> None:
        emb = emb.astype(np.float32).reshape(1, -1)
        if self.emb_matrix.size == 0:
            self.emb_matrix = emb
        else:
            self.emb_matrix = np.vstack([self.emb_matrix, emb])
        self.ids.append(rid)
        self.answers.append(answer)

    def search(self, q_emb: np.ndarray, top_k: int = 1) -> List[Tuple[float, int, str]]:
        """
        Returns list of (score, id, answer), sorted desc by score.
        """
        if self.emb_matrix.size == 0:
            return []
        q = q_emb.astype(np.float32).reshape(1, -1)
        # cosine similarity
        denom = (np.linalg.norm(self.emb_matrix, axis=1) * (np.linalg.norm(q) + 1e-8)) + 1e-8
        sims = (self.emb_matrix @ q.T).reshape(-1) / denom
        k = min(top_k, sims.shape[0])
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        ranked = sorted([(float(sims[i]), self.ids[i], self.answers[i]) for i in idx], key=lambda x: -x[0])
        return ranked

# -------------------------
# Embedding model (Transformers)
# -------------------------

class HFEmbedder:
    """
    Mean-pooling encoder using Transformers AutoModel.
    Default: intfloat/e5-small-v2 (fast, solid).
    """
    def __init__(self, model_id: str, device: torch.device):
        self.model_id = model_id
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_text(self, text: str, max_length: int = 256) -> np.ndarray:
        # For E5-style models, prefix helps:
        # "query: ..." for queries, "passage: ..." for passages.
        # We'll keep it simple here and do "query:" for question embeddings.
        text = "query: " + normalize_ws(text)
        inputs = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # last_hidden_state: [B, T, H]
        hid = out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(hid.size()).float()
        summed = torch.sum(hid * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)
        mean = summed / counts
        vec = mean[0].detach().cpu().float().numpy()
        # L2 normalize for cosine search stability
        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).astype(np.float32)

# -------------------------
# LLM generation (Transformers)
# -------------------------

def build_prompt(context: str, question: str) -> str:
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n"
        "Answer:"
    )

@torch.no_grad()
def generate_llm(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    max_input_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Safe generation:
    - truncates input to max_input_tokens
    - decodes only generated tokens (avoid huge decode / overflow issues)
    """
    t0 = time.perf_counter()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        padding=True,
    )
    input_len = int(inputs["input_ids"].shape[-1])
    hit_cap = (input_len == max_input_tokens)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    t1 = time.perf_counter()

    # Decode only the generated continuation
    gen_ids = out[0, input_len:].detach().cpu().tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    out_tokens = int(out.shape[-1] - input_len)
    dt = t1 - t0
    tps = (out_tokens / dt) if dt > 0 else None

    metrics = {
        "input_tokens": input_len,
        "output_tokens": out_tokens,
        "latency_llm_s": dt,
        "tokens_per_sec": tps,
        "truncated_to_max_input": hit_cap,
        "device_used": str(device),
    }
    return answer, metrics

# -------------------------
# Runner
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--tier2_repo", required=True, help="Path to LongBench data directory containing *.jsonl")
    ap.add_argument("--task_glob", default="", help="Filter tasks (e.g., trec)")

    ap.add_argument("--out_dir", default="runs", help="Output directory for run artifacts")
    ap.add_argument("--max_examples", type=int, default=50)

    # Modes / baselines
    ap.add_argument("--mode", choices=["baseline_llm", "baseline_context_only", "memory_qacache"], default="baseline_llm")

    # LLM settings
    ap.add_argument("--llm_model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--llm_device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_input_tokens", type=int, default=2048)

    # Memory settings (used by memory_qacache)
    ap.add_argument("--mem_db", default="memory/qa_memory.sqlite3")
    ap.add_argument("--embed_model_id", default="intfloat/e5-small-v2")
    ap.add_argument("--embed_device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--similarity_threshold", type=float, default=0.92)
    ap.add_argument("--top_k", type=int, default=1)

    args = ap.parse_args()

    tier2_repo = Path(args.tier2_repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = find_jsonl_files(tier2_repo, args.task_glob)
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {tier2_repo} matching '{args.task_glob}'")

    run_id = int(time.time())
    out_path = out_dir / f"{args.mode}_{run_id}.jsonl"

    # Devices
    llm_device = pick_device(args.llm_device)
    embed_device = pick_device(args.embed_device)

    # Load LLM if needed
    llm_tok = None
    llm_model = None
    if args.mode in ("baseline_llm", "memory_qacache"):
        llm_tok = AutoTokenizer.from_pretrained(args.llm_model_id, use_fast=True)

        if llm_tok.pad_token is None:
            llm_tok.pad_token = llm_tok.eos_token

        dtype = torch.float16 if llm_device.type in ("mps", "cuda") else torch.float32
        llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_id, torch_dtype=dtype)
        llm_model.to(llm_device)
        llm_model.eval()

        llm_model.config.pad_token_id = llm_tok.pad_token_id

        # if model config provides a smaller limit, respect it
        model_max = getattr(llm_model.config, "max_position_embeddings", None)
        if model_max:
            args.max_input_tokens = min(args.max_input_tokens, int(model_max))

    # Load memory components if needed
    memdb = None
    ram_index = RAMIndex.empty()
    embedder = None
    if args.mode == "memory_qacache":
        memdb = MemoryDB(Path(args.mem_db).resolve())
        ids, mat, answers = memdb.load_all_embeddings()
        ram_index = RAMIndex(ids=ids, emb_matrix=mat, answers=answers)
        embedder = HFEmbedder(args.embed_model_id, embed_device)

    print(f"[tier2] repo={tier2_repo}")
    print(f"[tier2] tasks={len(jsonl_files)} files | filter='{args.task_glob}'")
    print(f"[run ] mode={args.mode} -> {out_path}")
    if llm_model is not None:
        print(f"[llm ] {args.llm_model_id} on {llm_device} | max_input_tokens={args.max_input_tokens}")
    if embedder is not None:
        print(f"[emb ] {args.embed_model_id} on {embed_device} | threshold={args.similarity_threshold} | top_k={args.top_k}")
        print(f"[mem ] sqlite={Path(args.mem_db).resolve()} | ram_entries={len(ram_index.ids)}")

    total = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for task_file in jsonl_files:
            for i, ex in enumerate(iter_dataset_rows(task_file)):
                if total >= args.max_examples:
                    break

                context = ex.get("context", "") or ""
                q = ex.get("input", "") or ""
                gold = ex.get("answers", None)

                record: Dict[str, Any] = {
                    "status": "ok",
                    "mode": args.mode,
                    "task_file": task_file.name,
                    "example_index": i,
                    "question": q,
                    "answers_gold": gold,
                    "ram_rss_mb_before": rss_mb(),
                }

                t_start = time.perf_counter()
                route = "none"
                cache_hit = False
                sim_score = None
                answer = ""

                try:
                    if args.mode == "baseline_context_only":
                        # “Context baseline”: use dataset-provided answer(s) only
                        route = "context_only"
                        if isinstance(gold, list) and gold:
                            answer = str(gold[0])
                        else:
                            answer = ""

                    elif args.mode == "baseline_llm":
                        route = "llm_only"
                        prompt = build_prompt(context, q)
                        answer, m = generate_llm(
                            model=llm_model,
                            tokenizer=llm_tok,
                            device=llm_device,
                            prompt=prompt,
                            max_new_tokens=args.max_new_tokens,
                            max_input_tokens=args.max_input_tokens,
                        )
                        record.update(m)

                    else:
                        # memory_qacache
                        # 1) exact match check
                        exact = memdb.get_exact(q)
                        if exact is not None:
                            rid, ans = exact
                            route = "memory_exact"
                            cache_hit = True
                            answer = ans
                        else:
                            # 2) embed + similarity search
                            t_emb0 = time.perf_counter()
                            q_emb = embedder.embed_text(q)
                            t_emb1 = time.perf_counter()
                            record["latency_embed_s"] = (t_emb1 - t_emb0)

                            hits = ram_index.search(q_emb, top_k=args.top_k)
                            if hits:
                                sim_score, rid, ans = hits[0]
                                if sim_score >= args.similarity_threshold:
                                    route = "memory_similar_hit"
                                    cache_hit = True
                                    answer = ans

                            # 3) miss -> call LLM, then store
                            if not cache_hit:
                                route = "llm_miss_store"
                                prompt = build_prompt(context, q)
                                answer, m = generate_llm(
                                    model=llm_model,
                                    tokenizer=llm_tok,
                                    device=llm_device,
                                    prompt=prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    max_input_tokens=args.max_input_tokens,
                                )
                                record.update(m)
                                new_id = memdb.insert(q, answer, q_emb)
                                ram_index.add(new_id, q_emb, answer)

                except RuntimeError as e:
                    record["status"] = "runtime_error"
                    record["error"] = repr(e)[:600]
                except Exception as e:
                    record["status"] = "error"
                    record["error"] = repr(e)[:600]

                t_end = time.perf_counter()

                record.update({
                    "route": route,
                    "cache_hit": cache_hit,
                    "similarity_score": sim_score,
                    "answer_pred": answer,
                    "latency_total_s": (t_end - t_start),
                    "ram_rss_mb_after": rss_mb(),
                })

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                total += 1

            if total >= args.max_examples:
                break

    if memdb is not None:
        memdb.close()

    print(f"[done] wrote {total} records -> {out_path}")

if __name__ == "__main__":
    main()
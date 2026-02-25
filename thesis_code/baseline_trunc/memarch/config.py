
"""
config.py

Central configuration for the RAM/Disk prototype on a MacBook.

Design goals:
- One place to control paths, model ids, cache sizes, thresholds, and evaluation knobs.
- Runner + policies read from this config (no scattered constants).
- Supports three policies:
    1) baseline_llm      : always call LLM (no memory reuse)
    2) baseline_context  : "oracle-ish" baseline that answers using dataset context only
    3) proposed_memory   : embed -> similarity -> reuse answer or augment prompt -> LLM

This file should NOT import heavy ML libraries (torch/transformers) to keep startup fast.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # ------------------------
    # Experiment identity
    # ------------------------
    run_name: str = "run"
    seed: int = 0

    # ------------------------
    # Policy selection
    # ------------------------
    policy: str = "baseline_llm"  # baseline_llm | baseline_context | proposed_memory

    # ------------------------
    # Data
    # ------------------------
    # Either point to LongBench task JSONL(s) OR to your converted session JSONL.
    # For memory-latency evaluation, sessions_jsonl is preferred.
    longbench_dir: Optional[Path] = None      # directory containing *.jsonl
    task_glob: str = ""                       # e.g., "trec"
    sessions_jsonl: Optional[Path] = None     # e.g., sessions/trec_sessions.jsonl
    max_turns: int = 200                      # total turns processed across sessions
    max_sessions: int = 0                     # 0 = unlimited
    shuffle_sessions: bool = False            # keep deterministic by default

    # ------------------------
    # Output
    # ------------------------
    out_dir: Path = Path("runs")
    out_jsonl: Optional[Path] = None          # if None, runner will create under out_dir

    # ------------------------
    # Models (IDs are HF model names)
    # ------------------------
    generator_model_id: str = "microsoft/Phi-3-mini-128k-instruct"
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ------------------------
    # Device hints (Mac)
    # ------------------------
    # device="auto" lets model wrapper choose mps if available, else cpu
    device: str = "auto"                      # auto | cpu | mps

    # ------------------------
    # Prompt/token budgeting
    # ------------------------
    max_input_tokens: int = 8192              # runner-level cap (and/or model cap)
    max_new_tokens: int = 64

    # ------------------------
    # RAM caches (Tier 1)
    # ------------------------
    qa_cache_max_items: int = 2048
    embed_cache_max_items: int = 8192

    # ------------------------
    # Similarity + memory reuse
    # ------------------------
    top_k: int = 3
    similarity_threshold: float = 0.90
    bypass_llm_on_hit: bool = True            # if True: reuse cached answer directly
    include_hits_in_prompt: bool = True       # if not bypassing, inject retrieved QAs

    # ------------------------
    # Disk store (Tier 2)
    # ------------------------
    # SQLite recommended (single-file, fast enough, easy to inspect)
    db_path: Path = Path("tier2_disk/memory.sqlite3")
    # Optional: keep vectors also persisted so “memory” survives restarts
    persist_embeddings: bool = True
    persist_qa: bool = True

    # ------------------------
    # Baseline-context behavior
    # ------------------------
    # baseline_context is meant to show an upper bound using dataset context only.
    # For LongBench, the "answers" field is gold; baseline_context should NOT simply return gold.
    # Recommended: answer by extracting from context with a small heuristic,
    # or with a tiny non-LLM method (e.g., regex/keyword span) to illustrate limitations.
    context_answer_max_chars: int = 256

    def to_json(self) -> str:
        d = asdict(self)
        # Convert Paths to strings for JSON
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                d[k] = v
        return json.dumps(d, indent=2)

    @staticmethod
    def from_args(argv: Optional[list[str]] = None) -> "Config":
        ap = argparse.ArgumentParser()

        # Identity / policy
        ap.add_argument("--run_name", default="run")
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--policy", default="baseline_llm",
                        choices=["baseline_llm", "baseline_context", "proposed_memory"])

        # Data sources
        ap.add_argument("--sessions_jsonl", default="", help="Converted session-turns JSONL (preferred)")
        ap.add_argument("--longbench_dir", default="", help="Directory containing LongBench *.jsonl files")
        ap.add_argument("--task_glob", default="", help="Substring filter for task files, e.g. trec")
        ap.add_argument("--max_turns", type=int, default=200)
        ap.add_argument("--max_sessions", type=int, default=0)
        ap.add_argument("--shuffle_sessions", action="store_true")

        # Output
        ap.add_argument("--out_dir", default="runs")
        ap.add_argument("--out_jsonl", default="")

        # Models
        ap.add_argument("--generator_model_id", default="microsoft/Phi-3-mini-128k-instruct")
        ap.add_argument("--embed_model_id", default="sentence-transformers/all-MiniLM-L6-v2")

        # Device
        ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])

        # Token budgeting
        ap.add_argument("--max_input_tokens", type=int, default=8192)
        ap.add_argument("--max_new_tokens", type=int, default=64)

        # RAM caches
        ap.add_argument("--qa_cache_max_items", type=int, default=2048)
        ap.add_argument("--embed_cache_max_items", type=int, default=8192)

        # Similarity
        ap.add_argument("--top_k", type=int, default=3)
        ap.add_argument("--similarity_threshold", type=float, default=0.90)
        ap.add_argument("--bypass_llm_on_hit", action="store_true")
        ap.add_argument("--no_bypass_llm_on_hit", action="store_true")
        ap.add_argument("--include_hits_in_prompt", action="store_true")
        ap.add_argument("--no_include_hits_in_prompt", action="store_true")

        # Disk store
        ap.add_argument("--db_path", default="tier2_disk/memory.sqlite3")
        ap.add_argument("--persist_embeddings", action="store_true")
        ap.add_argument("--no_persist_embeddings", action="store_true")
        ap.add_argument("--persist_qa", action="store_true")
        ap.add_argument("--no_persist_qa", action="store_true")

        args = ap.parse_args(argv)

        cfg = Config(
            run_name=args.run_name,
            seed=args.seed,
            policy=args.policy,

            longbench_dir=Path(args.longbench_dir).resolve() if args.longbench_dir else None,
            task_glob=args.task_glob,
            sessions_jsonl=Path(args.sessions_jsonl).resolve() if args.sessions_jsonl else None,
            max_turns=args.max_turns,
            max_sessions=args.max_sessions,
            shuffle_sessions=args.shuffle_sessions,

            out_dir=Path(args.out_dir).resolve(),
            out_jsonl=Path(args.out_jsonl).resolve() if args.out_jsonl else None,

            generator_model_id=args.generator_model_id,
            embed_model_id=args.embed_model_id,

            device=args.device,

            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,

            qa_cache_max_items=args.qa_cache_max_items,
            embed_cache_max_items=args.embed_cache_max_items,

            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,

            # default True unless explicitly disabled
            bypass_llm_on_hit=True,
            include_hits_in_prompt=True,

            db_path=Path(args.db_path).resolve(),

            persist_embeddings=True,
            persist_qa=True,
        )

        if args.no_bypass_llm_on_hit:
            cfg.bypass_llm_on_hit = False
        elif args.bypass_llm_on_hit:
            cfg.bypass_llm_on_hit = True

        if args.no_include_hits_in_prompt:
            cfg.include_hits_in_prompt = False
        elif args.include_hits_in_prompt:
            cfg.include_hits_in_prompt = True

        if args.no_persist_embeddings:
            cfg.persist_embeddings = False
        elif args.persist_embeddings:
            cfg.persist_embeddings = True

        if args.no_persist_qa:
            cfg.persist_qa = False
        elif args.persist_qa:
            cfg.persist_qa = True

        return cfg
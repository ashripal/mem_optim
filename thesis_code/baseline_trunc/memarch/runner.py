
"""
runner.py

High-level orchestration:
- Load data turns (prefer sessions_jsonl)
- Route each turn through a chosen policy (baseline_llm / baseline_context / proposed_memory)
- Log per-turn JSONL records + summary metrics

Important: runner.py should not implement "memory policy" logic.
That belongs in memory/policy.py so swapping baselines is trivial and fair.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator, Optional, Tuple

from tbaseline_mem.config import Config

# Data I/O
from tbaseline_mem.data.longbench_io import iter_longbench_turns, iter_session_turns

# Memory + models are owned by the policy (but runner wires them up)
from tbaseline_mem.memory.disk_store import DiskStore
from tbaseline_mem.memory.qa_cache import QACache
from tbaseline_mem.memory.embed_cache import EmbedCache

from tbaseline_mem.models.generator import Generator
from tbaseline_mem.models.embedder import Embedder

from tbaseline_mem.memory.policy import (
    BaselineLLMPolicy,
    BaselineContextPolicy,
    ProposedMemoryPolicy,
)

from tbaseline_mem.utils.metrics import rss_mb, Timer, MetricsAggregator


def _default_out_path(cfg: Config) -> Path:
    ts = int(time.time())
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg.out_dir / f"{cfg.run_name}__{cfg.policy}__{ts}.jsonl"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # optional
        np.random.seed(seed)
    except Exception:
        pass


def _choose_turn_stream(cfg: Config) -> Iterator[Dict[str, Any]]:
    """
    Returns an iterator of "turn records" with at least:
      session_id, turn_id, context, input, answers, variant_type
    """
    if cfg.sessions_jsonl:
        return iter_session_turns(cfg.sessions_jsonl, max_turns=cfg.max_turns, max_sessions=cfg.max_sessions,
                                  shuffle_sessions=cfg.shuffle_sessions, seed=cfg.seed)

    if cfg.longbench_dir:
        return iter_longbench_turns(cfg.longbench_dir, task_glob=cfg.task_glob, max_turns=cfg.max_turns, seed=cfg.seed)

    raise ValueError("Provide either --sessions_jsonl OR --longbench_dir")


def _build_policy(cfg: Config) -> Tuple[object, Dict[str, Any]]:
    """
    Construct storage + caches + models once, then build the selected policy.
    Returns (policy, init_info) where init_info can be logged into the run header.
    """
    # Tier 2 (Disk store)
    store = DiskStore(cfg.db_path)
    store.init_schema()

    # Tier 1 (RAM caches) — store QA + embeddings, NOT datasets
    qa_cache = QACache(max_items=cfg.qa_cache_max_items)
    embed_cache = EmbedCache(max_items=cfg.embed_cache_max_items)

    # Models
    generator = Generator(
        model_id=cfg.generator_model_id,
        device=cfg.device,
        max_input_tokens=cfg.max_input_tokens,
        max_new_tokens=cfg.max_new_tokens,
    )
    embedder = Embedder(
        model_id=cfg.embed_model_id,
        device=cfg.device,
    )

    if cfg.policy == "baseline_llm":
        policy = BaselineLLMPolicy(
            cfg=cfg,
            generator=generator,
        )
    elif cfg.policy == "baseline_context":
        policy = BaselineContextPolicy(
            cfg=cfg,
        )
    elif cfg.policy == "proposed_memory":
        policy = ProposedMemoryPolicy(
            cfg=cfg,
            store=store,
            qa_cache=qa_cache,
            embed_cache=embed_cache,
            generator=generator,
            embedder=embedder,
        )
    else:
        raise ValueError(f"Unknown policy: {cfg.policy}")

    init_info = {
        "db_path": str(cfg.db_path),
        "qa_cache_max_items": cfg.qa_cache_max_items,
        "embed_cache_max_items": cfg.embed_cache_max_items,
        "generator_model_id": cfg.generator_model_id,
        "embed_model_id": cfg.embed_model_id,
        "device": cfg.device,
        "max_input_tokens": cfg.max_input_tokens,
        "max_new_tokens": cfg.max_new_tokens,
    }
    return policy, init_info


def main(argv: Optional[list[str]] = None) -> None:
    cfg = Config.from_args(argv)
    _seed_everything(cfg.seed)

    out_path = cfg.out_jsonl or _default_out_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build policy + dependencies once
    policy, init_info = _build_policy(cfg)

    # Metrics aggregator (summary printed at end + optionally written)
    agg = MetricsAggregator()

    # Write header record (so every run is self-describing)
    with out_path.open("w", encoding="utf-8") as f:
        header = {
            "record_type": "run_header",
            "timestamp_unix": int(time.time()),
            "config": json.loads(cfg.to_json()),
            "init": init_info,
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        # Iterate turns
        turns = _choose_turn_stream(cfg)
        seen = 0

        for turn in turns:
            if seen >= cfg.max_turns:
                break

            mem_before = rss_mb()

            with Timer() as t:
                result = policy.handle_turn(turn)  # <- core interface
            mem_after = rss_mb()

            # Normalize the output record structure for easy plotting
            record = {
                "record_type": "turn",
                "status": result.get("status", "ok"),
                "policy": cfg.policy,

                # Identity
                "session_id": turn.get("session_id"),
                "turn_id": turn.get("turn_id"),
                "variant_type": turn.get("variant_type"),
                "task_file": turn.get("task_file"),
                "example_index": turn.get("example_index"),

                # Inputs
                "question": turn.get("input"),
                "source_input": turn.get("source_input"),
                "context_chars": len(turn.get("context", "") or ""),

                # Outputs
                "answer_pred": result.get("answer_pred", ""),
                "answers_gold": turn.get("answers"),

                # Memory signals
                "cache_hit": result.get("cache_hit", False),
                "similarity": result.get("similarity", None),
                "hit_key": result.get("hit_key", None),
                "retrieved_k": result.get("retrieved_k", 0),

                # Timing + resources
                "latency_s": t.seconds,
                "ram_rss_mb_before": mem_before,
                "ram_rss_mb_after": mem_after,
            }

            # Merge any extra metrics the policy produces (tokens, device, truncation, errors…)
            for k, v in result.items():
                if k in record:
                    continue
                record[k] = v

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            agg.add(record)
            seen += 1

    # Print summary
    summary = agg.summary()
    print(f"[done] wrote: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
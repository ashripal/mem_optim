# pipeline/runner.py
"""
Experiment runner for the LongBench baseline.

This module is intentionally *thin*: it orchestrates the evaluation run by wiring together:
- Tier 2 (Disk) loader
- Tier 1 (RAM) cache
- Tier 0 (Compute) engine
- Evaluator + JSONL logger

It should not contain model loading details, truncation logic, plotting, or summarization logic.
Those belong in their respective modules.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

from baseline.pipeline.evaluator import evaluate_example
from baseline.pipeline.logging import JSONLLogger
from baseline.tiers.tier0_compute import ComputeEngine
from baseline.tiers.tier1_cache import LRUCache
from baseline.tiers.tier2_disk import DiskLoader


def _make_run_id(prefix: str = "baseline_longbench") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def run_experiment(cfg: Any) -> str:
    """
    Run a full LongBench evaluation and write per-example results to a JSONL file.

    Args:
        cfg: A config object (typically a dataclass) with fields like:
            - tier2_repo: str
            - out_dir: str
            - model_id: str
            - task_glob: str
            - max_examples: int
            - max_new_tokens: int
            - max_cache_items: int
            - max_input_tokens: int
            - device: str
            - dtype: str
            - cpu_fallback_on_long: bool

    Returns:
        Absolute path to the generated JSONL run file.
    """
    # --- Create output run path ---
    out_dir = getattr(cfg, "out_dir", "tier2_disk/runs")
    os.makedirs(out_dir, exist_ok=True)

    run_id = _make_run_id()
    run_path = os.path.join(out_dir, f"{run_id}.jsonl")

    # --- Initialize tiers ---
    disk = DiskLoader(
        repo_dir=getattr(cfg, "tier2_repo"),
        task_glob=getattr(cfg, "task_glob", ""),
        max_examples=getattr(cfg, "max_examples", 25),
    )
    cache = LRUCache(capacity=getattr(cfg, "max_cache_items", 64))
    compute = ComputeEngine(cfg)

    # --- Logger ---
    logger = JSONLLogger(run_path)

    # Write a small "run header" record (handy for provenance)
    header: Dict[str, Any] = {
        "type": "run_header",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(vars(cfg)),
    }
    logger.write(header)

    # --- Main loop ---
    n_total = 0
    n_ok = 0
    n_err = 0

    for ex in disk.iter_examples():
        n_total += 1

        try:
            record = evaluate_example(
                example=ex,
                cache=cache,
                compute=compute,
                cfg=cfg,
            )
            n_ok += 1
        except Exception as e:  # keep run going; record the failure
            n_err += 1
            record = {
                "type": "example_result",
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "task": ex.get("task"),
                "example_id": ex.get("example_id"),
            }

        logger.write(record)

    # --- Footer ---
    footer: Dict[str, Any] = {
        "type": "run_footer",
        "run_id": run_id,
        "finished_at": datetime.now().isoformat(),
        "counts": {"total": n_total, "ok": n_ok, "err": n_err},
    }
    logger.write(footer)
    logger.close()

    return os.path.abspath(run_path)
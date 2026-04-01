"""
Experiment runner for the LongBench baseline.

TRUE BASELINE:
- NO Tier 1 cache
- NO storage or reuse of prior outputs
- ONLY Tier 0 (compute) + Tier 2 (data)

This module orchestrates:
- Tier 2 (Disk) loader
- Tier 0 (Compute) engine
- Evaluator + JSONL logger
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

import torch

from baseline.pipeline.evaluator import evaluate_example
from baseline.pipeline.logging import JSONLLogger
from baseline.tiers.tier0_compute import ComputeEngine
from baseline.tiers.tier2_disk import DiskLoader
from baseline.utils.system import get_system_info


def _make_run_id(prefix: str = "baseline_longbench") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def run_experiment(cfg: Any) -> str:
    """
    Run a full LongBench evaluation and write per-example results to a JSONL file.

    Baseline behavior:
    - Each example is processed independently.
    - No answer caching or reuse is performed.
    """

    out_dir = getattr(cfg, "out_dir", "tier2_disk/runs")
    os.makedirs(out_dir, exist_ok=True)

    run_id = _make_run_id()
    run_path = os.path.join(out_dir, f"{run_id}.jsonl")

    disk = DiskLoader(
        repo_dir=getattr(cfg, "tier2_repo"),
        task_glob=getattr(cfg, "task_glob", ""),
        max_examples=getattr(cfg, "max_examples", 25),
    )
    compute = ComputeEngine(cfg)

    logger = JSONLLogger(run_path)

    try:
        header: Dict[str, Any] = {
            "type": "run_header",
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(vars(cfg)),
            "system_info": get_system_info(),
        }
        logger.write(header)

        n_total = 0
        n_ok = 0
        n_err = 0

        use_cuda = torch.cuda.is_available() and getattr(cfg, "device", "auto") in ("auto", "cuda")

        with torch.no_grad():
            for ex in disk.iter_examples():
                n_total += 1

                try:
                    record = evaluate_example(
                        example=ex,
                        cache=None,  # kept explicit to show the baseline does not use cache
                        compute=compute,
                        cfg=cfg,
                    )
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    record = {
                        "type": "example_result",
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "task": ex.get("task"),
                        "example_id": ex.get("example_id"),
                    }

                logger.write(record)

                if n_total % 10 == 0:
                    print(f"[progress] {n_total} examples processed (ok={n_ok}, err={n_err})")

                if use_cuda and n_total % 20 == 0:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        footer: Dict[str, Any] = {
            "type": "run_footer",
            "run_id": run_id,
            "finished_at": datetime.now().isoformat(),
            "counts": {"total": n_total, "ok": n_ok, "err": n_err},
        }
        logger.write(footer)

    finally:
        logger.close()

    return os.path.abspath(run_path)
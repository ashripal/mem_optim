# baseline/benchmarks/execute.py
"""
Benchmark execution for the baseline LongBench pipeline.

Responsibilities:
- Build a workload using baseline.benchmarks.workload
- Run examples through baseline pipeline (cache + compute)
- Log JSONL benchmark records
- Save workload manifest
- Optionally write summary

Design goals:
- Stay thin (reuse existing baseline pipeline logic)
- Preserve compatibility with run_experiment-style flow
- Add benchmark structure (workload + artifacts)
- Support device/dtype (Jetson-ready)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from baseline.benchmarks.configs import BenchmarkConfig
from baseline.benchmarks.workload import prepare_workload, build_workload_manifest
from baseline.pipeline.evaluator import evaluate_example
from baseline.pipeline.logging import JSONLLogger
from baseline.tiers.tier0_compute import ComputeEngine
from baseline.tiers.tier2_disk import DiskLoader

try:
    from baseline.analysis.summarize import summarize_run  # type: ignore
except Exception:
    summarize_run = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _make_run_id(prefix: str = "baseline_benchmark") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def _safe_cfg_dict(cfg: BenchmarkConfig) -> Dict[str, Any]:
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    return dict(vars(cfg))


def _write_json(path: str, payload: Dict[str, Any]) -> str:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(p)


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------

def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, str]:
    """
    Execute a baseline benchmark run.

    Returns:
        {
            "run_jsonl": ...,
            "workload_manifest_json": ...,
            "summary_json": ... (optional)
        }
    """
    cfg.validate()

    # -----------------------------
    # Resolve output paths
    # -----------------------------
    run_dir = Path(cfg.resolved_out_dir()).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    run_id = _make_run_id(prefix=cfg.benchmark_name)
    run_jsonl = run_dir / f"{run_id}.jsonl"
    manifest_json = run_dir / f"{run_id}.manifest.json"
    summary_json = run_dir / f"{run_id}.summary.json"

    # -----------------------------
    # Build workload
    # -----------------------------
    base_disk = DiskLoader(
        repo_dir=cfg.tier2_repo,
        task_glob=cfg.task_glob,
        max_examples=cfg.max_examples,
    )

    base_examples = list(base_disk.iter_examples())
    workload = prepare_workload(cfg, base_examples)
    workload_manifest = build_workload_manifest(cfg, workload)

    # -----------------------------
    # Initialize pipeline
    # -----------------------------
    cache = LRUCache(capacity=cfg.max_cache_items)
    compute = ComputeEngine(cfg)

    resolved_device = getattr(compute, "active_device", None)
    resolved_dtype_obj = getattr(compute, "model_dtype", None)
    resolved_dtype = (
        str(resolved_dtype_obj).replace("torch.", "")
        if resolved_dtype_obj is not None
        else "none"
    )
    generation_backend = "manual_greedy"

    # -----------------------------
    # Run loop
    # -----------------------------
    n_total = 0
    n_ok = 0
    n_err = 0
    n_cache_hits = 0
    n_generated = 0
    total_latency_s_acc = 0.0

    with JSONLLogger(str(run_jsonl)) as logger:
        logger.write(
            {
                "type": "run_header",
                "run_id": run_id,
                "created_at": datetime.now().isoformat(),
                "benchmark_name": cfg.benchmark_name,
                "notes": cfg.notes,
                "config": _safe_cfg_dict(cfg),
                "resolved_out_dir": str(run_dir),
                "resolved_runtime": {
                    "device": resolved_device,
                    "dtype": resolved_dtype,
                    "generation_backend": generation_backend,
                },
                "workload_manifest_preview": {
                    "mode": workload_manifest.get("workload_mode"),
                    "total_examples": workload_manifest.get("total_workload_examples"),
                },
            }
        )

        for ex in workload:
            n_total += 1

            try:
                record = evaluate_example(
                    example=ex,
                    cache=cache,
                    compute=compute,
                    cfg=cfg,
                )
                n_ok += 1
            except Exception as e:
                record = {
                    "type": "example_result",
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "task": ex.get("task"),
                    "example_id": ex.get("example_id"),
                }
                n_err += 1

            if record.get("cache_hit"):
                n_cache_hits += 1
            if record.get("served_from") == "tier0_compute":
                n_generated += 1
            if record.get("latency_s") is not None:
                total_latency_s_acc += float(record["latency_s"])

            logger.write(record)

        logger.write(
            {
                "type": "run_footer",
                "run_id": run_id,
                "finished_at": datetime.now().isoformat(),
                "counts": {
                    "total": n_total,
                    "ok": n_ok,
                    "err": n_err,
                },
                "aggregate_metrics": {
                    "mean_latency_s": (total_latency_s_acc / n_total) if n_total else None,
                    "n_generated": n_generated,
                    "n_cache_hits": n_cache_hits,
                    "cache_hit_rate": (n_cache_hits / n_total) if n_total else 0.0,
                },
                "resolved_runtime": {
                    "device": resolved_device,
                    "dtype": resolved_dtype,
                    "generation_backend": generation_backend,
                },
            }
        )

    # -----------------------------
    # Write artifacts
    # -----------------------------
    _write_json(str(manifest_json), workload_manifest)

    artifacts: Dict[str, str] = {
        "run_jsonl": str(run_jsonl.resolve()),
        "workload_manifest_json": str(manifest_json.resolve()),
    }

    if cfg.output.write_summary_json and summarize_run is not None:
        summary = summarize_run(str(run_jsonl))
        _write_json(str(summary_json), summary)
        artifacts["summary_json"] = str(summary_json.resolve())

    return artifacts
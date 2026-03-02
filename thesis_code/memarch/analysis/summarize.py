# memarch/analysis/summarize.py
"""
Summarize a memarch run JSONL log into aggregate metrics.

Input: JSONL file produced by memarch.pipeline.logging.JsonlLogger
Output: a dict of metrics (and optional pretty text)

This module is intentionally dependency-light (no pandas) so it runs on Jetson.
If you want richer reporting, you can create a separate script that uses pandas.

Typical usage (from scripts/ or memarch/main.py):
  summary = summarize_run("artifacts/runs/run_001/log.jsonl")
  print(format_summary(summary))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from memarch.utils.metrics import compute_run_metrics, RunMetrics


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


@dataclass(frozen=True)
class RunSummary:
    run_id: Optional[str]
    metrics: RunMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "metrics": self.metrics.to_dict(),
        }


def summarize_run(log_path: str) -> RunSummary:
    """
    Read a JSONL log and compute aggregate run metrics.

    We use only "example" events for metrics.
    """
    run_id: Optional[str] = None
    example_events: List[Dict[str, Any]] = []

    for evt in _read_jsonl(log_path):
        if evt.get("type") == "run_start":
            run_id = evt.get("run_id") or run_id
        if evt.get("type") == "example":
            example_events.append(evt)

    metrics = compute_run_metrics(example_events)
    return RunSummary(run_id=run_id, metrics=metrics)


def format_summary(summary: RunSummary) -> str:
    """
    Create a human-readable summary text (for CLI printing).
    """
    m = summary.metrics
    lines = []
    lines.append(f"Run ID: {summary.run_id or 'unknown'}")
    lines.append(f"Examples: {m.num_examples}")
    lines.append(f"Memory hit rate: {m.hit_rate:.3f} ({m.num_used_memory}/{m.num_examples})")
    lines.append("")
    lines.append("Latency (ms):")
    lines.append(f"  total: mean={m.total.mean_ms:.2f}  p50={m.total.p50_ms:.2f}  p95={m.total.p95_ms:.2f}  p99={m.total.p99_ms:.2f}")
    lines.append(f"  memory_lookup: mean={m.memory_lookup.mean_ms:.2f}  p50={m.memory_lookup.p50_ms:.2f}  p95={m.memory_lookup.p95_ms:.2f}  p99={m.memory_lookup.p99_ms:.2f}")
    lines.append(f"  generation_est: mean={m.generation_est.mean_ms:.2f}  p50={m.generation_est.p50_ms:.2f}  p95={m.generation_est.p95_ms:.2f}  p99={m.generation_est.p99_ms:.2f}")
    return "\n".join(lines)


def write_summary_json(summary: RunSummary, out_path: str) -> None:
    """
    Write summary to a JSON file.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
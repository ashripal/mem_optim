# analysis/summarize.py

"""
Summarize a LongBench baseline run JSONL into aggregated statistics.

TRUE BASELINE:
- Stateless execution
- NO cache
- NO memory
- ALL requests go to the LLM

This module is pure analysis.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# IO
# -----------------------------

def load_run_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"run_jsonl not found: {p}")

    records = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_example_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("type") == "example_result"]


# -----------------------------
# Stats helpers
# -----------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _collect_numeric(records: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(r[key]) for r in records if _is_number(r.get(key))]


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _percentile(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    r = (p / 100.0) * (len(ys) - 1)
    lo, hi = int(r), min(int(r) + 1, len(ys) - 1)
    w = r - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def _summary(xs):
    return {
        "count": len(xs),
        "mean": _mean(xs),
        "p50": _percentile(xs, 50),
        "p95": _percentile(xs, 95),
        "p99": _percentile(xs, 99),
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
    }


# -----------------------------
# Summaries
# -----------------------------

def summarize_counts(records):
    total = len(records)
    ok = sum(1 for r in records if r.get("ok"))
    return {
        "total": total,
        "ok": ok,
        "err": total - ok,
        "ok_rate": ok / total if total else None,
    }


def summarize_latency(records):
    ok = [r for r in records if r.get("ok")]
    return _summary(_collect_numeric(ok, "latency_s"))


def summarize_tokens(records):
    ok = [r for r in records if r.get("ok")]

    return {
        "input_tokens": _summary(_collect_numeric(ok, "input_tokens")),
        "output_tokens": _summary(_collect_numeric(ok, "output_tokens")),
        "tokens_per_second": _summary(_collect_numeric(ok, "tokens_per_second")),
    }


def summarize_devices(records):
    ok = [r for r in records if r.get("ok")]
    counts = {}
    for r in ok:
        d = str(r.get("device", "unknown"))
        counts[d] = counts.get(d, 0) + 1
    return {"counts": counts}


def summarize_quality(records):
    ok = [r for r in records if r.get("ok")]

    def avg(key):
        vals = _collect_numeric(ok, key)
        return _mean(vals)

    return {
        "exact_match": avg("exact_match"),
        "token_f1": avg("token_f1"),
        "char_f1": avg("char_f1"),
    }


# -----------------------------
# Main API
# -----------------------------

def summarize_run(run_jsonl_path: str) -> Dict[str, Any]:
    records = load_run_jsonl(run_jsonl_path)
    ex = extract_example_records(records)

    summary = {
        "counts": summarize_counts(ex),
        "latency": summarize_latency(ex),
        "tokens": summarize_tokens(ex),
        "devices": summarize_devices(ex),
        "quality": summarize_quality(ex),
    }

    return summary
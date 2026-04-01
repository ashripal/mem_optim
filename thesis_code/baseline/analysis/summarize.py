"""
Summarize a LongBench baseline run JSONL into aggregated statistics.

TRUE BASELINE:
- Stateless execution
- NO cache reuse
- Every request is processed independently by the LLM

This module is intentionally pure analysis:
- Reads a run JSONL (produced by pipeline/logging.py)
- Extracts per-example records (type == "example_result")
- Computes aggregate stats for latency/tokens/memory/device/quality
- Optionally writes a flattened single-row CSV summary

It does NOT:
- Plot (see analysis/plot.py)
- Parse CLI args
- Depend on the dataset loader or compute modules
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

    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {p} line {line_no}: {e}") from e
    return records


def extract_example_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("type") == "example_result"]


# -----------------------------
# Stats helpers
# -----------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (
        isinstance(x, float) and math.isnan(x)
    )


def _collect_numeric(records: List[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for r in records:
        v = r.get(key, None)
        if _is_number(v):
            vals.append(float(v))
    return vals


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _min(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(min(xs))


def _max(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(max(xs))


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return float((ys[mid - 1] + ys[mid]) / 2.0)


def _percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    if p <= 0:
        return float(min(xs))
    if p >= 100:
        return float(max(xs))

    ys = sorted(xs)
    n = len(ys)
    r = (p / 100.0) * (n - 1)
    lo = int(math.floor(r))
    hi = int(math.ceil(r))
    if lo == hi:
        return float(ys[lo])
    w = r - lo
    return float((1.0 - w) * ys[lo] + w * ys[hi])


def _summary_stats(xs: List[float]) -> Dict[str, Any]:
    return {
        "count": len(xs),
        "mean": _mean(xs),
        "median": _median(xs),
        "p90": _percentile(xs, 90.0),
        "p95": _percentile(xs, 95.0),
        "p99": _percentile(xs, 99.0),
        "min": _min(xs),
        "max": _max(xs),
    }


def _count_where(records: List[Dict[str, Any]], key: str, truthy: bool = True) -> int:
    n = 0
    for r in records:
        v = r.get(key, None)
        if truthy:
            if bool(v):
                n += 1
        else:
            if not bool(v):
                n += 1
    return n


# -----------------------------
# Summarizers
# -----------------------------

def summarize_counts(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    ok = sum(1 for r in records if bool(r.get("ok", False)))
    err = total - ok
    return {"total": total, "ok": ok, "err": err, "ok_rate": (ok / total) if total else None}


def summarize_latency(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_records = [r for r in records if bool(r.get("ok", False))]
    lat = _collect_numeric(ok_records, "latency_s")
    return _summary_stats(lat)


def summarize_tokens(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_records = [r for r in records if bool(r.get("ok", False))]
    in_tok = _collect_numeric(ok_records, "input_tokens")
    out_tok = _collect_numeric(ok_records, "output_tokens")
    tps = _collect_numeric(ok_records, "tokens_per_second")

    truncated_count = _count_where(ok_records, "truncated", truthy=True)
    truncated_rate = (truncated_count / len(ok_records)) if ok_records else None

    return {
        "input_tokens": _summary_stats(in_tok),
        "output_tokens": _summary_stats(out_tok),
        "tokens_per_second": _summary_stats(tps),
        "truncated_count": truncated_count,
        "truncated_rate": truncated_rate,
    }


def summarize_memory(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_records = [r for r in records if bool(r.get("ok", False))]
    rss_before = _collect_numeric(ok_records, "rss_before_mb")
    rss_after = _collect_numeric(ok_records, "rss_after_mb")
    rss_delta = _collect_numeric(ok_records, "rss_delta_mb")
    return {
        "rss_before_mb": _summary_stats(rss_before),
        "rss_after_mb": _summary_stats(rss_after),
        "rss_delta_mb": _summary_stats(rss_delta),
    }


def summarize_devices(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_records = [r for r in records if bool(r.get("ok", False))]
    counts: Dict[str, int] = {}
    for r in ok_records:
        d = r.get("device", None)
        if d is None:
            d = "unknown"
        counts[str(d)] = counts.get(str(d), 0) + 1

    total = len(ok_records)
    shares = {k: (v / total) if total else None for k, v in counts.items()}
    return {"counts": counts, "shares": shares}


def summarize_cache(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_records = [r for r in records if bool(r.get("ok", False))]
    hits = _count_where(ok_records, "cache_hit", truthy=True)
    misses = _count_where(ok_records, "cache_hit", truthy=False)
    total = len(ok_records)
    hit_rate = (hits / total) if total else None
    return {"hits": hits, "misses": misses, "hit_rate": hit_rate}


def summarize_quality(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_records = [r for r in records if bool(r.get("ok", False))]

    exact = _collect_numeric(ok_records, "exact_match")
    contains = _collect_numeric(ok_records, "contains_answer")
    token_f1 = _collect_numeric(ok_records, "token_f1")
    char_f1 = _collect_numeric(ok_records, "char_f1")

    return {
        "exact_match": _summary_stats(exact),
        "contains_answer": _summary_stats(contains),
        "token_f1": _summary_stats(token_f1),
        "char_f1": _summary_stats(char_f1),
    }


# -----------------------------
# Main API
# -----------------------------

def summarize_run(run_jsonl_path: str) -> Dict[str, Any]:
    records = load_run_jsonl(run_jsonl_path)
    ex = extract_example_records(records)

    summary: Dict[str, Any] = {
        "run_jsonl": str(Path(run_jsonl_path).expanduser().resolve()),
        "counts": summarize_counts(ex),
        "latency": summarize_latency(ex),
        "tokens": summarize_tokens(ex),
        "memory": summarize_memory(ex),
        "devices": summarize_devices(ex),
        "cache": summarize_cache(ex),
        "quality": summarize_quality(ex),
    }

    header = next((r for r in records if r.get("type") == "run_header"), None)
    if header:
        summary["run_id"] = header.get("run_id")
        summary["created_at"] = header.get("created_at")
        summary["config"] = header.get("config")
        if "system_info" in header:
            summary["system_info"] = header.get("system_info")

    footer = next((r for r in records if r.get("type") == "run_footer"), None)
    if footer:
        summary["finished_at"] = footer.get("finished_at")

    return summary


# -----------------------------
# CSV Output
# -----------------------------

def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def write_summary_csv(summary: Dict[str, Any], out_csv: str) -> str:
    out_path = Path(out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat = _flatten_dict(summary)

    safe: Dict[str, Any] = {}
    for k, v in flat.items():
        if isinstance(v, (dict, list, tuple)):
            safe[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe[k] = v

    fieldnames = sorted(safe.keys())

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(safe)

    return str(out_path)
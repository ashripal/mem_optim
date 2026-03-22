# memarch/analysis/summarize.py
"""
Summarize a memarch benchmark JSONL log into aggregate metrics.

Input:
  JSONL file produced by memarch.benchmarks.execute.run_benchmark

Output:
  A lightweight dict-like summary object and optional human-readable text.

This module is intentionally dependency-light (no pandas) so it runs on
resource-constrained systems as well.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterator, List, Optional


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


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    idx = (len(sorted_vals) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])

    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _numeric_stats(values: List[Any]) -> Dict[str, Any]:
    nums = [float(v) for v in values if _is_number(v)]
    if not nums:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }

    s = sorted(nums)
    return {
        "count": len(nums),
        "mean": mean(nums),
        "median": median(nums),
        "p90": _percentile(s, 0.90),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "min": min(nums),
        "max": max(nums),
    }


@dataclass(frozen=True)
class RunSummary:
    run_id: Optional[str]
    created_at: Optional[str]
    finished_at: Optional[str]
    config: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "metrics": self.metrics,
        }
        if self.config is not None:
            out["config"] = self.config
        return out


def summarize_run(log_path: str) -> RunSummary:
    """
    Read a memarch benchmark JSONL log and compute aggregate run metrics.

    We use only:
      - run_header
      - example_result
      - run_footer
    """
    run_id: Optional[str] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    example_events: List[Dict[str, Any]] = []

    for evt in _read_jsonl(log_path):
        evt_type = evt.get("type")

        if evt_type == "run_header":
            run_id = evt.get("run_id") or run_id
            created_at = evt.get("created_at") or created_at
            config = evt.get("config") or config

        elif evt_type == "example_result":
            example_events.append(evt)

        elif evt_type == "run_footer":
            run_id = evt.get("run_id") or run_id
            finished_at = evt.get("finished_at") or finished_at

    ok_events = [e for e in example_events if bool(e.get("ok"))]
    err_events = [e for e in example_events if not bool(e.get("ok"))]

    total_ok = len(ok_events)

    used_memory_count = sum(1 for e in ok_events if bool(e.get("used_memory")))
    generated_count = sum(1 for e in ok_events if bool(e.get("generated")))
    ram_hits = sum(1 for e in ok_events if e.get("source_tier") == "ram")
    disk_hits = sum(1 for e in ok_events if e.get("source_tier") == "disk")
    compute_served = sum(1 for e in ok_events if e.get("source_tier") == "compute")
    promoted_to_ram = sum(1 for e in ok_events if bool(e.get("promoted_to_ram")))
    llm_bypassed = sum(1 for e in ok_events if bool(e.get("llm_bypassed")))
    stored_count = sum(1 for e in ok_events if bool(e.get("stored")))

    exact_hits = sum(1 for e in ok_events if e.get("match_type") == "exact")
    semantic_served_directly = sum(
        1 for e in ok_events
        if e.get("match_type") == "semantic" and bool(e.get("used_memory"))
    )
    semantic_used = sum(1 for e in ok_events if bool(e.get("semantic_used")))
    semantic_bypassed = sum(1 for e in ok_events if bool(e.get("semantic_bypassed")))

    match_type_counts: Dict[str, int] = {}
    for e in ok_events:
        mt = e.get("match_type")
        if mt is None:
            mt = "none"
        match_type_counts[str(mt)] = match_type_counts.get(str(mt), 0) + 1

    devices: Dict[str, int] = {}
    for e in ok_events:
        d = e.get("device")
        if d:
            devices[d] = devices.get(d, 0) + 1

    device_shares = {
        k: (v / total_ok if total_ok > 0 else 0.0)
        for k, v in devices.items()
    }

    semantic_scores = [
        e.get("semantic_score")
        for e in ok_events
        if _is_number(e.get("semantic_score"))
    ]

    semantic_ranks = [
        e.get("semantic_candidate_rank")
        for e in ok_events
        if _is_number(e.get("semantic_candidate_rank"))
    ]

    rows = ok_events
    latency_values = [e.get("latency_s") for e in rows]
    latency_stats = _numeric_stats(latency_values)

    n_examples = len(rows)
    n_generated = sum(1 for r in rows if bool(r.get("generated", False)))
    n_memory_hits = sum(1 for r in rows if bool(r.get("used_memory", False)))
    n_exact_hits = sum(1 for r in rows if r.get("match_type") == "exact")
    n_semantic_hits = sum(
        1 for r in rows
        if (r.get("hit_before_generate") or {}).get("match_type") == "semantic"
        or r.get("match_type") == "semantic"
    )
    n_semantic_context_used = sum(1 for r in rows if bool(r.get("semantic_used", False)))
    n_semantic_bypass = sum(1 for r in rows if bool(r.get("semantic_bypassed", False)))
    memory_hit_rate = n_memory_hits / len(rows) if rows else 0.0
    llm_calls_saved = len(rows) - n_generated

    metrics: Dict[str, Any] = {
        "n_examples": n_examples,
        "mean_latency_s": latency_stats["mean"],
        "p50_latency_s": latency_stats["median"],
        "p95_latency_s": latency_stats["p95"],
        "n_generated": n_generated,
        "n_memory_hits": n_memory_hits,
        "memory_hit_rate": memory_hit_rate,
        "n_exact_hits": n_exact_hits,
        "n_semantic_hits": n_semantic_hits,
        "n_semantic_context_used": n_semantic_context_used,
        "n_semantic_bypass": n_semantic_bypass,
        "llm_calls_saved": llm_calls_saved,
        "counts": {
            "total": len(example_events),
            "ok": total_ok,
            "err": len(err_events),
            "ok_rate": (total_ok / len(example_events)) if example_events else 0.0,
        },
        "memory": {
            "used_memory_count": used_memory_count,
            "memory_hit_rate": (used_memory_count / total_ok) if total_ok > 0 else 0.0,
            "total_hit_rate": (used_memory_count / total_ok) if total_ok > 0 else 0.0,
            "generated_count": generated_count,
            "generation_rate": (generated_count / total_ok) if total_ok > 0 else 0.0,
            "ram_hits": ram_hits,
            "disk_hits": disk_hits,
            "compute_served": compute_served,
            "promoted_to_ram": promoted_to_ram,
            "llm_bypassed": llm_bypassed,
            "stored_count": stored_count,
        },
        "retrieval": {
            "exact_hits": exact_hits,
            "exact_hit_rate": (exact_hits / total_ok) if total_ok > 0 else 0.0,
            "semantic_used_count": semantic_used,
            "semantic_used_rate": (semantic_used / total_ok) if total_ok > 0 else 0.0,
            "semantic_bypassed_count": semantic_bypassed,
            "semantic_bypassed_rate": (semantic_bypassed / total_ok) if total_ok > 0 else 0.0,
            "semantic_served_directly": semantic_served_directly,
            "match_type_counts": match_type_counts,
            "semantic_score": _numeric_stats(semantic_scores),
            "semantic_candidate_rank": _numeric_stats(semantic_ranks),
        },
        "latency": {
            "total_s": latency_stats,
            "memory_lookup_ms": _numeric_stats([e.get("memory_lookup_ms") for e in ok_events]),
            "generation_ms_est": _numeric_stats([e.get("generation_ms_est") for e in ok_events]),
        },
        "tokens": {
            "input_tokens": _numeric_stats([e.get("input_tokens") for e in ok_events]),
            "output_tokens": _numeric_stats([e.get("output_tokens") for e in ok_events]),
            "tokens_per_second": _numeric_stats([e.get("tokens_per_second") for e in ok_events]),
            "truncated_count": sum(1 for e in ok_events if bool(e.get("truncated"))),
            "truncated_rate": (
                sum(1 for e in ok_events if bool(e.get("truncated"))) / total_ok
                if total_ok > 0 else 0.0
            ),
        },
        "memory_usage": {
            "rss_before_mb": _numeric_stats([e.get("rss_before_mb") for e in ok_events]),
            "rss_after_mb": _numeric_stats([e.get("rss_after_mb") for e in ok_events]),
            "rss_delta_mb": _numeric_stats([e.get("rss_delta_mb") for e in ok_events]),
        },
        "devices": {
            "counts": devices,
            "shares": device_shares,
        },
        "quality": {
            "exact_match": _numeric_stats([e.get("exact_match") for e in ok_events]),
            "contains_answer": _numeric_stats([e.get("contains_answer") for e in ok_events]),
            "token_f1": _numeric_stats([e.get("token_f1") for e in ok_events]),
            "char_f1": _numeric_stats([e.get("char_f1") for e in ok_events]),
        },
        "tiers": {
            "source_tier_counts": {
                "ram": ram_hits,
                "disk": disk_hits,
                "compute": compute_served,
            }
        },
    }

    return RunSummary(
        run_id=run_id,
        created_at=created_at,
        finished_at=finished_at,
        config=config,
        metrics=metrics,
    )


def format_summary(summary: RunSummary) -> str:
    """
    Create a human-readable summary text for CLI printing.
    """
    m = summary.metrics
    counts = m["counts"]
    mem = m["memory"]
    retrieval = m["retrieval"]
    lat = m["latency"]

    lines = []
    lines.append(f"Run ID: {summary.run_id or 'unknown'}")
    lines.append(f"Examples: total={counts['total']} ok={counts['ok']} err={counts['err']}")
    lines.append(
        f"Memory hit rate: {mem['memory_hit_rate']:.3f} "
        f"({mem['used_memory_count']}/{counts['ok'] if counts['ok'] else 0})"
    )
    lines.append(
        f"Exact hits: {retrieval['exact_hits']} "
        f"({retrieval['exact_hit_rate']:.3f})"
    )
    lines.append(
        f"Semantic used: {retrieval['semantic_used_count']} "
        f"({retrieval['semantic_used_rate']:.3f})"
    )
    lines.append(
        f"Semantic bypassed: {retrieval['semantic_bypassed_count']} "
        f"({retrieval['semantic_bypassed_rate']:.3f})"
    )
    lines.append(
        f"Tier usage: ram={mem['ram_hits']} disk={mem['disk_hits']} compute={mem['compute_served']}"
    )
    lines.append("")
    lines.append("Latency:")
    lines.append(
        "  total_s: "
        f"mean={lat['total_s']['mean'] if lat['total_s']['mean'] is not None else 'n/a'}  "
        f"p50={lat['total_s']['median'] if lat['total_s']['median'] is not None else 'n/a'}  "
        f"p95={lat['total_s']['p95'] if lat['total_s']['p95'] is not None else 'n/a'}  "
        f"p99={lat['total_s']['p99'] if lat['total_s']['p99'] is not None else 'n/a'}"
    )
    lines.append(
        "  memory_lookup_ms: "
        f"mean={lat['memory_lookup_ms']['mean'] if lat['memory_lookup_ms']['mean'] is not None else 'n/a'}  "
        f"p50={lat['memory_lookup_ms']['median'] if lat['memory_lookup_ms']['median'] is not None else 'n/a'}  "
        f"p95={lat['memory_lookup_ms']['p95'] if lat['memory_lookup_ms']['p95'] is not None else 'n/a'}  "
        f"p99={lat['memory_lookup_ms']['p99'] if lat['memory_lookup_ms']['p99'] is not None else 'n/a'}"
    )
    lines.append(
        "  generation_ms_est: "
        f"mean={lat['generation_ms_est']['mean'] if lat['generation_ms_est']['mean'] is not None else 'n/a'}  "
        f"p50={lat['generation_ms_est']['median'] if lat['generation_ms_est']['median'] is not None else 'n/a'}  "
        f"p95={lat['generation_ms_est']['p95'] if lat['generation_ms_est']['p95'] is not None else 'n/a'}  "
        f"p99={lat['generation_ms_est']['p99'] if lat['generation_ms_est']['p99'] is not None else 'n/a'}"
    )
    return "\n".join(lines)


def write_summary_json(summary: RunSummary, out_path: str) -> None:
    """
    Write summary to a JSON file.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
# memarch/utils/metrics.py
"""
Metrics aggregation utilities for memarch.

Goals:
- Lightweight, dependency-free (pure Python)
- Useful for both online summaries (runner) and offline analysis (analysis/summarize.py)
- Deterministic behavior

We avoid pandas here to keep core package small and portable.
(You can use pandas in scripts/analysis if you want.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: List[float], p: float) -> float:
    """
    Compute percentile p in [0,100] using linear interpolation.

    Deterministic, no numpy dependency.
    """
    if not xs:
        return 0.0
    if p <= 0:
        return float(min(xs))
    if p >= 100:
        return float(max(xs))

    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return float(ys[f])
    d0 = ys[f] * (c - k)
    d1 = ys[c] * (k - f)
    return float(d0 + d1)


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @staticmethod
    def from_samples(samples_ms: List[float]) -> "LatencyStats":
        return LatencyStats(
            count=len(samples_ms),
            mean_ms=mean(samples_ms),
            p50_ms=percentile(samples_ms, 50),
            p95_ms=percentile(samples_ms, 95),
            p99_ms=percentile(samples_ms, 99),
        )


@dataclass
class RunMetrics:
    num_examples: int
    num_used_memory: int
    hit_rate: float

    total: LatencyStats
    memory_lookup: LatencyStats
    generation_est: LatencyStats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_examples": self.num_examples,
            "num_used_memory": self.num_used_memory,
            "hit_rate": self.hit_rate,
            "total": self.total.__dict__,
            "memory_lookup": self.memory_lookup.__dict__,
            "generation_est": self.generation_est.__dict__,
        }


def compute_run_metrics(example_events: Iterable[Dict[str, Any]]) -> RunMetrics:
    """
    Compute aggregate metrics from per-example log events.

    Expected event structure (from pipeline/logging.py):
      {
        "type": "example",
        "meta": {... "used_memory": bool ...},
        "timings_ms": {"total_ms": ..., "memory_lookup_ms": ..., "generation_ms_est": ...}
      }
    """
    total_ms: List[float] = []
    mem_ms: List[float] = []
    gen_ms: List[float] = []
    used_mem = 0
    n = 0

    for evt in example_events:
        if evt.get("type") != "example":
            continue
        n += 1
        meta = evt.get("meta") or {}
        if bool(meta.get("used_memory")):
            used_mem += 1
        t = evt.get("timings_ms") or {}
        total_ms.append(float(t.get("total_ms", 0.0)))
        mem_ms.append(float(t.get("memory_lookup_ms", 0.0)))
        gen_ms.append(float(t.get("generation_ms_est", 0.0)))

    hit_rate = (used_mem / n) if n else 0.0

    return RunMetrics(
        num_examples=n,
        num_used_memory=used_mem,
        hit_rate=float(hit_rate),
        total=LatencyStats.from_samples(total_ms),
        memory_lookup=LatencyStats.from_samples(mem_ms),
        generation_est=LatencyStats.from_samples(gen_ms),
    )
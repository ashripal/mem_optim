# memarch/analysis/plot.py
"""
Plotting utilities for memarch run logs.

This module reads JSONL logs and generates simple matplotlib plots:
- Total latency over examples
- Memory lookup latency over examples
- Memory hit/miss as a timeline (optional)
- Histogram of total latency (optional)

Design goals:
- Minimal dependencies beyond matplotlib
- No seaborn (per project constraints)
- One plot per figure (no subplots)
- No explicit color styling unless requested

Usage:
  from memarch.analysis.plot import plot_latency_series, plot_latency_histogram
  plot_latency_series("artifacts/runs/run_001/log.jsonl", "artifacts/runs/run_001/latency_total.png")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt


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


def _extract_series(log_path: str) -> Tuple[List[int], List[float], List[float], List[float], List[int]]:
    """
    Returns:
      idxs, total_ms, memory_lookup_ms, generation_ms_est, used_memory_flags(0/1)
    """
    idxs: List[int] = []
    total_ms: List[float] = []
    mem_ms: List[float] = []
    gen_ms: List[float] = []
    used_mem: List[int] = []

    i = 0
    for evt in _read_jsonl(log_path):
        if evt.get("type") != "example":
            continue
        i += 1
        t = evt.get("timings_ms") or {}
        m = evt.get("meta") or {}

        idxs.append(i)
        total_ms.append(float(t.get("total_ms", 0.0)))
        mem_ms.append(float(t.get("memory_lookup_ms", 0.0)))
        gen_ms.append(float(t.get("generation_ms_est", 0.0)))
        used_mem.append(1 if bool(m.get("used_memory")) else 0)

    return idxs, total_ms, mem_ms, gen_ms, used_mem


def _ensure_parent(out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)


def plot_latency_series(
    log_path: str,
    out_path: str,
    *,
    which: str = "total",
    title: Optional[str] = None,
) -> None:
    """
    Plot a latency series over example index.

    which: "total" | "memory" | "generation"
    """
    idxs, total_ms, mem_ms, gen_ms, _ = _extract_series(log_path)

    if which == "total":
        ys = total_ms
        ylabel = "Total latency (ms)"
        default_title = "Total latency over examples"
    elif which == "memory":
        ys = mem_ms
        ylabel = "Memory lookup latency (ms)"
        default_title = "Memory lookup latency over examples"
    elif which == "generation":
        ys = gen_ms
        ylabel = "Generation latency (ms) (estimated)"
        default_title = "Generation latency over examples (estimated)"
    else:
        raise ValueError("which must be one of: total, memory, generation")

    fig = plt.figure()
    plt.plot(idxs, ys)
    plt.xlabel("Example index")
    plt.ylabel(ylabel)
    plt.title(title or default_title)
    plt.tight_layout()

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_latency_histogram(
    log_path: str,
    out_path: str,
    *,
    which: str = "total",
    bins: int = 40,
    title: Optional[str] = None,
) -> None:
    """
    Plot histogram of latency values.

    which: "total" | "memory" | "generation"
    """
    _, total_ms, mem_ms, gen_ms, _ = _extract_series(log_path)

    if which == "total":
        xs = total_ms
        xlabel = "Total latency (ms)"
        default_title = "Total latency histogram"
    elif which == "memory":
        xs = mem_ms
        xlabel = "Memory lookup latency (ms)"
        default_title = "Memory lookup latency histogram"
    elif which == "generation":
        xs = gen_ms
        xlabel = "Generation latency (ms) (estimated)"
        default_title = "Generation latency histogram (estimated)"
    else:
        raise ValueError("which must be one of: total, memory, generation")

    fig = plt.figure()
    plt.hist(xs, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title or default_title)
    plt.tight_layout()

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_memory_hit_timeline(
    log_path: str,
    out_path: str,
    *,
    title: Optional[str] = None,
) -> None:
    """
    Plot a simple 0/1 timeline for memory hits.

    Useful to visualize warm-up (more hits later) vs steady miss patterns.
    """
    idxs, _, _, _, used_mem = _extract_series(log_path)

    fig = plt.figure()
    plt.plot(idxs, used_mem)
    plt.xlabel("Example index")
    plt.ylabel("Used memory (1=yes, 0=no)")
    plt.title(title or "Memory hit timeline")
    plt.yticks([0, 1])
    plt.tight_layout()

    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
# analysis/plot.py

"""
Plotting utilities for LongBench baseline runs.

TRUE BASELINE:
- Stateless LLM execution
- NO cache / NO memory
- Every request is independent

Focus:
- Latency behavior
- Throughput behavior
- Stability across requests
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# IO
# -----------------------------

def load_run_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"run_jsonl not found: {p}")

    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_example_records(records, ok_only=True):
    ex = [r for r in records if r.get("type") == "example_result"]
    if ok_only:
        ex = [r for r in ex if r.get("ok")]
    return ex


def _finite(x):
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def _xy(records, xk, yk):
    xs, ys = [], []
    for r in records:
        if _finite(r.get(xk)) and _finite(r.get(yk)):
            xs.append(float(r[xk]))
            ys.append(float(r[yk]))
    return xs, ys


# -----------------------------
# Core plots
# -----------------------------

def scatter(records, xk, yk, title, xlabel, ylabel, out):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    xs, ys = _xy(records, xk, yk)

    plt.figure()
    plt.scatter(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    return str(out)


def histogram(records, key, title, xlabel, out, bins=40):
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    vals = [float(r[key]) for r in records if _finite(r.get(key))]

    plt.figure()
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    return str(out)


def plot_latency_timeline(records, out):
    """
    MOST IMPORTANT BASELINE PLOT:
    latency vs request index (shows no caching effects)
    """
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    ys = [r["latency_s"] for r in records if _finite(r.get("latency_s"))]
    xs = list(range(len(ys)))

    plt.figure()
    plt.plot(xs, ys)
    plt.title("Latency vs Request Index")
    plt.xlabel("Request index")
    plt.ylabel("Latency (s)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    return str(out)


# -----------------------------
# Public API
# -----------------------------

def plot_suite(run_jsonl: str, out_dir: str, prefix="baseline"):
    records = extract_example_records(load_run_jsonl(run_jsonl))

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    return {
        "latency_vs_tokens": scatter(
            records,
            "input_tokens",
            "latency_s",
            "Latency vs Input Tokens",
            "Input tokens",
            "Latency (s)",
            out / f"{prefix}_latency_vs_tokens.png",
        ),
        "tps_vs_tokens": scatter(
            records,
            "input_tokens",
            "tokens_per_second",
            "Tokens/sec vs Input Tokens",
            "Input tokens",
            "Tokens/sec",
            out / f"{prefix}_tps_vs_tokens.png",
        ),
        "latency_hist": histogram(
            records,
            "latency_s",
            "Latency Distribution",
            "Latency (s)",
            out / f"{prefix}_latency_hist.png",
        ),
        "latency_timeline": plot_latency_timeline(
            records,
            out / f"{prefix}_latency_timeline.png",
        ),
    }
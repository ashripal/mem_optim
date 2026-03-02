# analysis/plot.py
"""
Plotting utilities for LongBench run JSONL files.

This module is intentionally focused on visualization (matplotlib only):
- Reads a run JSONL (produced by pipeline/logging.py)
- Extracts per-example records (type == "example_result")
- Generates plots commonly used in the baseline:
    1) Latency vs input tokens (split by device)
    2) Tokens/sec vs input tokens (split by device)
    3) RSS delta vs input tokens (split by device)
    4) Latency histogram
    5) RSS delta histogram

Notes / constraints:
- Uses matplotlib (no seaborn).
- Does not set explicit colors; matplotlib defaults are used.
- No CLI parsing here (keep wrappers thin elsewhere).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# IO / extraction
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


def extract_example_records(records: List[Dict[str, Any]], *, ok_only: bool = True) -> List[Dict[str, Any]]:
    ex = [r for r in records if r.get("type") == "example_result"]
    if ok_only:
        ex = [r for r in ex if bool(r.get("ok", False))]
    return ex


def _finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def _group_by_device(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        d = r.get("device", "unknown")
        d = str(d) if d is not None else "unknown"
        groups.setdefault(d, []).append(r)
    return groups


def _xy_from_records(records: List[Dict[str, Any]], x_key: str, y_key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for r in records:
        x = r.get(x_key)
        y = r.get(y_key)
        if _finite_number(x) and _finite_number(y):
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


# -----------------------------
# Core plot functions
# -----------------------------

def plot_scatter_by_device(
    records: List[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    out_path: str,
    alpha: float = 0.6,
    s: float = 16.0,
) -> str:
    """
    Generic scatter plot split by device. Uses matplotlib default colors.

    Returns:
      Absolute path to the saved PNG.
    """
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    groups = _group_by_device(records)

    plt.figure()
    for device, recs in sorted(groups.items(), key=lambda kv: kv[0]):
        xs, ys = _xy_from_records(recs, x_key, y_key)
        if not xs:
            continue
        # No explicit color set; matplotlib default cycle applies per device call
        plt.scatter(xs, ys, label=device, alpha=alpha, s=s)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    return str(out)


def plot_histogram(
    records: List[Dict[str, Any]],
    *,
    key: str,
    title: str,
    x_label: str,
    out_path: str,
    bins: int = 40,
) -> str:
    """
    Single histogram for a numeric field across ok records.
    Uses matplotlib default styling/colors.

    Returns:
      Absolute path to the saved PNG.
    """
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    vals: List[float] = []
    for r in records:
        v = r.get(key)
        if _finite_number(v):
            vals.append(float(v))

    plt.figure()
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    return str(out)


# -----------------------------
# Convenience wrappers (baseline defaults)
# -----------------------------

def plot_latency_vs_input_tokens(run_jsonl: str, out_png: str) -> str:
    records = extract_example_records(load_run_jsonl(run_jsonl), ok_only=True)
    return plot_scatter_by_device(
        records,
        x_key="input_tokens",
        y_key="latency_s",
        title="Latency vs Input Tokens",
        x_label="Input tokens",
        y_label="Latency (s)",
        out_path=out_png,
    )


def plot_tps_vs_input_tokens(run_jsonl: str, out_png: str) -> str:
    records = extract_example_records(load_run_jsonl(run_jsonl), ok_only=True)
    return plot_scatter_by_device(
        records,
        x_key="input_tokens",
        y_key="tokens_per_second",
        title="Tokens/sec vs Input Tokens",
        x_label="Input tokens",
        y_label="Tokens per second",
        out_path=out_png,
    )


def plot_rss_delta_vs_input_tokens(run_jsonl: str, out_png: str) -> str:
    records = extract_example_records(load_run_jsonl(run_jsonl), ok_only=True)
    return plot_scatter_by_device(
        records,
        x_key="input_tokens",
        y_key="rss_delta_mb",
        title="RSS Delta vs Input Tokens",
        x_label="Input tokens",
        y_label="RSS delta (MB)",
        out_path=out_png,
    )


def plot_latency_hist(run_jsonl: str, out_png: str, bins: int = 40) -> str:
    records = extract_example_records(load_run_jsonl(run_jsonl), ok_only=True)
    return plot_histogram(
        records,
        key="latency_s",
        title="Latency Distribution",
        x_label="Latency (s)",
        out_path=out_png,
        bins=bins,
    )


def plot_rss_delta_hist(run_jsonl: str, out_png: str, bins: int = 40) -> str:
    records = extract_example_records(load_run_jsonl(run_jsonl), ok_only=True)
    return plot_histogram(
        records,
        key="rss_delta_mb",
        title="RSS Delta Distribution",
        x_label="RSS delta (MB)",
        out_path=out_png,
        bins=bins,
    )


def plot_suite(
    run_jsonl: str,
    out_dir: str,
    *,
    prefix: str = "baseline",
) -> Dict[str, str]:
    """
    Produce a small suite of baseline plots into out_dir and return paths.
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    paths["latency_vs_tokens"] = plot_latency_vs_input_tokens(
        run_jsonl, str(out / f"{prefix}_latency_vs_input_tokens.png")
    )
    paths["tps_vs_tokens"] = plot_tps_vs_input_tokens(
        run_jsonl, str(out / f"{prefix}_tps_vs_input_tokens.png")
    )
    paths["rss_delta_vs_tokens"] = plot_rss_delta_vs_input_tokens(
        run_jsonl, str(out / f"{prefix}_rss_delta_vs_input_tokens.png")
    )
    paths["latency_hist"] = plot_latency_hist(
        run_jsonl, str(out / f"{prefix}_latency_hist.png")
    )
    paths["rss_delta_hist"] = plot_rss_delta_hist(
        run_jsonl, str(out / f"{prefix}_rss_delta_hist.png")
    )
    return paths
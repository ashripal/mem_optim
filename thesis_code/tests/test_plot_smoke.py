# tests/test_plot_smoke.py
"""
Smoke tests for analysis/plot.py.

We do NOT verify pixel-perfect correctness. We only ensure:
- Plot functions run without error on a tiny synthetic run JSONL
- Output PNG files are created and non-empty

This keeps tests fast and CI-friendly.

Run:
  pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path

from baseline.analysis.plot import (
    plot_latency_vs_input_tokens,
    plot_rss_delta_hist,
    plot_rss_delta_vs_input_tokens,
    plot_suite,
    plot_tps_vs_input_tokens,
)


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _tiny_run_records():
    return [
        {"type": "run_header", "run_id": "r_plot", "created_at": "2026-02-25T00:00:00", "config": {}},
        {
            "type": "example_result",
            "ok": True,
            "device": "cpu",
            "cache_hit": False,
            "latency_s": 1.0,
            "input_tokens": 100,
            "output_tokens": 10,
            "tokens_per_second": 10.0,
            "rss_delta_mb": 5.0,
            "rss_before_mb": 100.0,
            "rss_after_mb": 105.0,
        },
        {
            "type": "example_result",
            "ok": True,
            "device": "mps",
            "cache_hit": False,
            "latency_s": 0.5,
            "input_tokens": 200,
            "output_tokens": 20,
            "tokens_per_second": 40.0,
            "rss_delta_mb": 2.0,
            "rss_before_mb": 120.0,
            "rss_after_mb": 122.0,
        },
        {"type": "run_footer", "run_id": "r_plot", "finished_at": "2026-02-25T00:01:00", "counts": {"total": 2, "ok": 2, "err": 0}},
    ]


def _assert_png_exists_nonempty(p: Path):
    assert p.exists(), f"Expected PNG to exist: {p}"
    assert p.stat().st_size > 0, f"Expected PNG to be non-empty: {p}"


def test_individual_plot_functions_create_pngs(tmp_path: Path):
    run_jsonl = tmp_path / "run.jsonl"
    _write_jsonl(run_jsonl, _tiny_run_records())

    p1 = Path(tmp_path / "latency_vs_tokens.png")
    p2 = Path(tmp_path / "tps_vs_tokens.png")
    p3 = Path(tmp_path / "rss_delta_vs_tokens.png")
    p4 = Path(tmp_path / "rss_delta_hist.png")

    plot_latency_vs_input_tokens(str(run_jsonl), str(p1))
    plot_tps_vs_input_tokens(str(run_jsonl), str(p2))
    plot_rss_delta_vs_input_tokens(str(run_jsonl), str(p3))
    plot_rss_delta_hist(str(run_jsonl), str(p4), bins=10)

    _assert_png_exists_nonempty(p1)
    _assert_png_exists_nonempty(p2)
    _assert_png_exists_nonempty(p3)
    _assert_png_exists_nonempty(p4)


def test_plot_suite_creates_all_outputs(tmp_path: Path):
    run_jsonl = tmp_path / "run.jsonl"
    _write_jsonl(run_jsonl, _tiny_run_records())

    out_dir = tmp_path / "plots"
    paths = plot_suite(str(run_jsonl), str(out_dir), prefix="smoke")

    # Ensure expected keys exist
    expected_keys = {
        "latency_vs_tokens",
        "tps_vs_tokens",
        "rss_delta_vs_tokens",
        "latency_hist",
        "rss_delta_hist",
    }
    assert expected_keys.issubset(set(paths.keys()))

    for _, path_str in paths.items():
        _assert_png_exists_nonempty(Path(path_str))
# tests/test_summarize.py
"""
Unit tests for analysis/summarize.py.

Creates a tiny synthetic run JSONL and verifies:
- example_result extraction
- counts (ok/err)
- latency mean/median
- device breakdown
- cache hit rate
- truncation rate
- CSV writing produces a file

Run:
  pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline.analysis.summarize import summarize_run, write_summary_csv


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_summarize_run_basic(tmp_path: Path):
    run_path = tmp_path / "run.jsonl"

    records = [
        {"type": "run_header", "run_id": "r1", "created_at": "2026-02-25T00:00:00", "config": {"x": 1}},
        # ok record (cpu)
        {
            "type": "example_result",
            "ok": True,
            "device": "cpu",
            "cache_hit": False,
            "truncated": False,
            "latency_s": 2.0,
            "input_tokens": 100,
            "output_tokens": 10,
            "tokens_per_second": 5.0,
            "rss_before_mb": 100.0,
            "rss_after_mb": 110.0,
            "rss_delta_mb": 10.0,
        },
        # ok record (mps) + cache hit
        {
            "type": "example_result",
            "ok": True,
            "device": "mps",
            "cache_hit": True,
            "truncated": True,
            "latency_s": 1.0,
            "input_tokens": 200,
            "output_tokens": 20,
            "tokens_per_second": 20.0,
            "rss_before_mb": 120.0,
            "rss_after_mb": 121.0,
            "rss_delta_mb": 1.0,
        },
        # error record
        {"type": "example_result", "ok": False, "device": "cpu", "cache_hit": False, "error": "boom"},
        {"type": "run_footer", "run_id": "r1", "finished_at": "2026-02-25T00:01:00", "counts": {"total": 3, "ok": 2, "err": 1}},
    ]

    _write_jsonl(run_path, records)

    summary = summarize_run(str(run_path))

    # Provenance
    assert summary["run_id"] == "r1"
    assert summary["created_at"] == "2026-02-25T00:00:00"
    assert summary["finished_at"] == "2026-02-25T00:01:00"

    # Counts
    assert summary["counts"]["total"] == 3
    assert summary["counts"]["ok"] == 2
    assert summary["counts"]["err"] == 1
    assert summary["counts"]["ok_rate"] == pytest.approx(2 / 3)

    # Latency stats computed on ok records only: [2.0, 1.0]
    assert summary["latency"]["count"] == 2
    assert summary["latency"]["mean"] == pytest.approx(1.5)
    assert summary["latency"]["median"] == pytest.approx(1.5)
    assert summary["latency"]["min"] == pytest.approx(1.0)
    assert summary["latency"]["max"] == pytest.approx(2.0)

    # Device breakdown on ok records only
    assert summary["devices"]["counts"]["cpu"] == 1
    assert summary["devices"]["counts"]["mps"] == 1
    assert summary["devices"]["shares"]["cpu"] == pytest.approx(0.5)
    assert summary["devices"]["shares"]["mps"] == pytest.approx(0.5)

    # Cache hit rate on ok records only: 1 hit out of 2
    assert summary["cache"]["hits"] == 1
    assert summary["cache"]["misses"] == 1
    assert summary["cache"]["hit_rate"] == pytest.approx(0.5)

    # Truncation rate on ok records only: 1 truncated out of 2
    assert summary["tokens"]["truncated_count"] == 1
    assert summary["tokens"]["truncated_rate"] == pytest.approx(0.5)

    # Tokens/sec stats exist and count ok-only records
    assert summary["tokens"]["tokens_per_second"]["count"] == 2
    assert summary["tokens"]["tokens_per_second"]["mean"] == pytest.approx((5.0 + 20.0) / 2)


def test_write_summary_csv(tmp_path: Path):
    run_path = tmp_path / "run.jsonl"
    _write_jsonl(
        run_path,
        [
            {"type": "run_header", "run_id": "r2", "created_at": "x", "config": {}},
            {"type": "example_result", "ok": True, "device": "cpu", "cache_hit": False, "latency_s": 1.0},
            {"type": "run_footer", "run_id": "r2", "finished_at": "y", "counts": {"total": 1, "ok": 1, "err": 0}},
        ],
    )

    summary = summarize_run(str(run_path))
    out_csv = tmp_path / "summary.csv"
    csv_path = write_summary_csv(summary, str(out_csv))

    p = Path(csv_path)
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    # Should contain headers and at least the run_id
    assert "run_id" in text
    assert "r2" in text
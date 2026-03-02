# tests/test_logging.py
"""
Unit tests for pipeline/logging.py (JSONLLogger).

Validates:
- Writes one JSON object per line
- File is created and readable
- Handles non-JSON-serializable objects via fallback
- Close prevents further writes

Run:
  pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline.pipeline.logging import JSONLLogger


def _read_lines(p: Path):
    return p.read_text(encoding="utf-8").splitlines()


def test_writes_jsonl_lines(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    logger = JSONLLogger(str(out))

    logger.write({"type": "run_header", "run_id": "abc"})
    logger.write({"type": "example_result", "ok": True, "latency_s": 1.23})
    logger.close()

    assert out.exists()
    lines = _read_lines(out)
    assert len(lines) == 2

    r0 = json.loads(lines[0])
    r1 = json.loads(lines[1])

    assert r0["type"] == "run_header"
    assert r1["type"] == "example_result"
    assert r1["ok"] is True
    assert r1["latency_s"] == 1.23


def test_fallback_for_non_serializable_values(tmp_path: Path):
    out = tmp_path / "run.jsonl"

    class NotSerializable:
        def __str__(self):
            return "NOT_SERIALIZABLE_OK"

    logger = JSONLLogger(str(out))
    logger.write({"obj": NotSerializable()})
    logger.close()

    lines = _read_lines(out)
    assert len(lines) == 1

    rec = json.loads(lines[0])
    # Should be stringified by fallback
    assert rec["obj"] == "NOT_SERIALIZABLE_OK"


def test_context_manager_closes_file(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with JSONLLogger(str(out)) as logger:
        logger.write({"x": 1})

    # After context exit, file should exist and contain the record
    lines = _read_lines(out)
    assert len(lines) == 1
    assert json.loads(lines[0])["x"] == 1


def test_write_after_close_raises(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    logger = JSONLLogger(str(out))
    logger.write({"x": 1})
    logger.close()

    with pytest.raises(RuntimeError):
        logger.write({"y": 2})
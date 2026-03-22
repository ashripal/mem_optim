# tests/test_runner.py
"""
Unit tests for pipeline/runner.py orchestration.

We avoid loading real HF models by monkeypatching the runner module to use fakes for:
- DiskLoader
- ComputeEngine

We keep:
- real LRUCache
- real JSONLLogger (writes to tmp_path)

Validates:
- run_experiment returns a path that exists
- JSONL includes run_header and run_footer
- number of example_result records matches yielded examples
- footer counts are correct

Run:
  pytest -q
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

import baseline.pipeline.runner as runner_mod


@dataclass
class _Cfg:
    tier2_repo: str
    out_dir: str
    model_id: str = "fake/model"
    task_glob: str = ""
    max_examples: int = 25
    max_input_tokens: int = 128
    max_new_tokens: int = 16
    max_cache_items: int = 8
    device: str = "auto"
    dtype: str = "auto"
    cpu_fallback_on_long: bool = False


class FakeDiskLoader:
    def __init__(self, repo_dir: str, task_glob: str = "", max_examples: int = 25):
        self.repo_dir = repo_dir
        self.task_glob = task_glob
        self.max_examples = max_examples

    def iter_examples(self) -> Iterator[Dict[str, Any]]:
        # Yield a tiny deterministic stream
        yield {"context": "c1", "question": "q1", "task": "t1", "example_id": 0}
        yield {"context": "c2", "question": "q2", "task": "t1", "example_id": 1}


class FakeComputeEngine:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.calls = 0

    def generate(self, *, prompt: str, max_input_tokens: int, max_new_tokens: int) -> Dict[str, Any]:
        self.calls += 1
        return {
            "ok": True,
            "device": "cpu",
            "dtype": "float32",
            "truncated": False,
            "input_tokens": 10,
            "output_tokens": 5,
            "gen_time_s": 0.1,
            "prompt_used": prompt,
            "output_text": "FAKE_OUTPUT",
        }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def test_runner_writes_header_examples_footer(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "runs"
    cfg = _Cfg(tier2_repo=str(tmp_path), out_dir=str(out_dir))

    # Patch runner's imports (DiskLoader and ComputeEngine)
    monkeypatch.setattr(runner_mod, "DiskLoader", FakeDiskLoader)
    monkeypatch.setattr(runner_mod, "ComputeEngine", FakeComputeEngine)

    run_path = runner_mod.run_experiment(cfg)
    p = Path(run_path)
    assert p.exists()

    recs = _read_jsonl(p)
    assert recs[0]["type"] == "run_header"
    assert recs[-1]["type"] == "run_footer"

    example_recs = [r for r in recs if r.get("type") == "example_result"]
    assert len(example_recs) == 2

    footer = recs[-1]
    assert footer["counts"]["total"] == 2
    assert footer["counts"]["ok"] == 2
    assert footer["counts"]["err"] == 0


def test_runner_continues_on_evaluator_error(tmp_path: Path, monkeypatch):
    """
    Force evaluate_example to raise for the first example and ensure runner:
    - writes an error record
    - continues to the next example
    - footer counts reflect err/ok
    """
    out_dir = tmp_path / "runs"
    cfg = _Cfg(tier2_repo=str(tmp_path), out_dir=str(out_dir))

    monkeypatch.setattr(runner_mod, "DiskLoader", FakeDiskLoader)
    monkeypatch.setattr(runner_mod, "ComputeEngine", FakeComputeEngine)

    # Patch evaluate_example inside runner module
    calls = {"n": 0}

    def boom_once(*, example, cache, compute, cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        # Return a minimal valid record
        return {
            "type": "example_result",
            "ok": True,
            "task": example.get("task"),
            "example_id": example.get("example_id"),
            "cache_hit": False,
            "latency_s": 0.5,
            "rss_before_mb": 1.0,
            "rss_after_mb": 1.5,
            "rss_delta_mb": 0.5,
            "input_tokens": 10,
            "output_tokens": 5,
            "device": "cpu",
            "dtype": "float32",
            "truncated": False,
            "tokens_per_second": 10.0,
        }

    monkeypatch.setattr(runner_mod, "evaluate_example", boom_once)

    run_path = runner_mod.run_experiment(cfg)
    recs = _read_jsonl(Path(run_path))

    example_recs = [r for r in recs if r.get("type") == "example_result"]
    assert len(example_recs) == 2

    # First should be an error record (runner's except block)
    assert example_recs[0]["ok"] is False
    assert "error" in example_recs[0]

    # Second should be ok
    assert example_recs[1]["ok"] is True

    footer = recs[-1]
    assert footer["counts"]["total"] == 2
    assert footer["counts"]["ok"] == 1
    assert footer["counts"]["err"] == 1
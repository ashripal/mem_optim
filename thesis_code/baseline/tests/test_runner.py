from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import baseline.pipeline.runner as runner_mod


@dataclass
class DummyConfig:
    tier2_repo: str
    out_dir: str
    model_id: str = "dummy/model"
    task_glob: str = ""
    max_examples: int = 2
    max_input_tokens: int = 128
    max_new_tokens: int = 16
    device: str = "cpu"
    dtype: str = "fp32"
    cpu_fallback_on_long: bool = False
    use_fast_tokenizer: bool = True
    attn_implementation: str | None = None
    trust_remote_code: bool = False


class FakeDiskLoader:
    def __init__(self, repo_dir: str, task_glob: str = "", max_examples: int = 25):
        self.repo_dir = repo_dir
        self.task_glob = task_glob
        self.max_examples = max_examples

    def iter_examples(self) -> Iterator[Dict[str, Any]]:
        yield {
            "context": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "task": "qa",
            "example_id": 0,
        }
        yield {
            "context": "The sky is blue.",
            "question": "What color is the sky?",
            "answer": "blue",
            "task": "qa",
            "example_id": 1,
        }


class FakeComputeEngine:
    init_count = 0

    def __init__(self, cfg: Any):
        type(self).init_count += 1
        self.cfg = cfg

    def generate(self, *, prompt: str, max_input_tokens: int, max_new_tokens: int) -> Dict[str, Any]:
        return {
            "ok": True,
            "output_text": "dummy answer",
            "input_tokens": 10,
            "output_tokens": 3,
            "device": "cpu",
            "dtype": "fp32",
            "gen_time_s": 0.01,
            "tokenize_time_s": 0.002,
            "decode_time_s": 0.003,
            "generation_backend": "fake_generate",
            "truncated": False,
        }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _minimal_ok_record(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "example_result",
        "ok": True,
        "task": example.get("task"),
        "example_id": example.get("example_id"),
        "llm_bypassed": False,
        "latency_s": 0.1,
        "compute_latency_s": 0.1,
        "rss_before_mb": 100.0,
        "rss_after_mb": 101.0,
        "rss_delta_mb": 1.0,
        "input_tokens": 10,
        "output_tokens": 3,
        "device": "cpu",
        "dtype": "fp32",
        "generation_backend": "fake_generate",
        "truncated": False,
        "output_text": "dummy answer",
        "ref_text": "dummy ref",
        "exact_match": 0.0,
        "contains_answer": 0.0,
        "token_f1": 0.0,
        "char_f1": 0.0,
        "tokens_per_second": 30.0,
    }


def test_runner_writes_header_examples_footer(tmp_path, monkeypatch):
    FakeComputeEngine.init_count = 0

    monkeypatch.setattr(runner_mod, "DiskLoader", FakeDiskLoader)
    monkeypatch.setattr(runner_mod, "ComputeEngine", FakeComputeEngine)

    cfg = DummyConfig(
        tier2_repo=str(tmp_path),
        out_dir=str(tmp_path / "runs"),
    )

    run_path = runner_mod.run_experiment(cfg)
    records = _read_jsonl(Path(run_path))

    assert Path(run_path).exists()
    assert records[0]["type"] == "run_header"
    assert records[-1]["type"] == "run_footer"

    example_records = [r for r in records if r.get("type") == "example_result"]
    assert len(example_records) == 2

    footer = records[-1]
    assert footer["counts"]["total"] == 2
    assert footer["counts"]["ok"] == 2
    assert footer["counts"]["err"] == 0

    # Baseline runner should build compute once.
    assert FakeComputeEngine.init_count == 1

    # Provenance should be present in header.
    assert "config" in records[0]
    assert "system_info" in records[0]


def test_runner_is_stateless_no_cache_path(tmp_path, monkeypatch):
    FakeComputeEngine.init_count = 0

    monkeypatch.setattr(runner_mod, "DiskLoader", FakeDiskLoader)
    monkeypatch.setattr(runner_mod, "ComputeEngine", FakeComputeEngine)

    called = {"n": 0}

    def fake_evaluate_example(*, example, cache, compute, cfg):
        called["n"] += 1

        # True baseline: runner should not pass a cache object.
        assert cache is None

        return _minimal_ok_record(example)

    monkeypatch.setattr(runner_mod, "evaluate_example", fake_evaluate_example)

    cfg = DummyConfig(
        tier2_repo=str(tmp_path),
        out_dir=str(tmp_path / "runs"),
    )

    run_path = runner_mod.run_experiment(cfg)
    records = _read_jsonl(Path(run_path))

    example_records = [r for r in records if r.get("type") == "example_result"]
    assert len(example_records) == 2
    assert called["n"] == 2

    for rec in example_records:
        assert rec["ok"] is True
        assert rec["llm_bypassed"] is False
        assert rec["device"] == "cpu"
        assert rec["output_text"] == "dummy answer"


def test_runner_continues_on_evaluator_error(tmp_path, monkeypatch):
    FakeComputeEngine.init_count = 0

    monkeypatch.setattr(runner_mod, "DiskLoader", FakeDiskLoader)
    monkeypatch.setattr(runner_mod, "ComputeEngine", FakeComputeEngine)

    calls = {"n": 0}

    def fake_evaluate_example(*, example, cache, compute, cfg):
        calls["n"] += 1
        assert cache is None

        if calls["n"] == 1:
            raise RuntimeError("boom")

        return _minimal_ok_record(example)

    monkeypatch.setattr(runner_mod, "evaluate_example", fake_evaluate_example)

    cfg = DummyConfig(
        tier2_repo=str(tmp_path),
        out_dir=str(tmp_path / "runs"),
    )

    run_path = runner_mod.run_experiment(cfg)
    records = _read_jsonl(Path(run_path))

    example_records = [r for r in records if r.get("type") == "example_result"]
    assert len(example_records) == 2

    assert example_records[0]["ok"] is False
    assert "error" in example_records[0]

    assert example_records[1]["ok"] is True

    footer = records[-1]
    assert footer["counts"]["total"] == 2
    assert footer["counts"]["ok"] == 1
    assert footer["counts"]["err"] == 1
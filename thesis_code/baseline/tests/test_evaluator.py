from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest


class DummyCompute:
    """
    Fake compute engine that simulates LLM generation.
    Tracks whether generate() is called.
    """

    def __init__(self):
        self.called = False
        self.last_prompt = None

    def generate(self, *, prompt: str, max_input_tokens: int, max_new_tokens: int) -> Dict[str, Any]:
        self.called = True
        self.last_prompt = prompt

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
            "generation_backend": "dummy_generate",
            "truncated": False,
        }


class DummyCache:
    """
    If this is ever used, the test should fail.
    """

    def get(self, *args, **kwargs):
        raise AssertionError("Cache should NOT be used in baseline")

    def put(self, *args, **kwargs):
        raise AssertionError("Cache should NOT be used in baseline")


@dataclass
class DummyConfig:
    max_input_tokens: int = 128
    max_new_tokens: int = 16


def make_example() -> Dict[str, Any]:
    return {
        "context": "Paris is the capital of France.",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "task": "qa",
        "example_id": 0,
    }


def test_evaluator_calls_llm():
    from baseline.pipeline.evaluator import evaluate_example

    compute = DummyCompute()
    cfg = DummyConfig()
    example = make_example()

    result = evaluate_example(example, None, compute, cfg)

    assert compute.called is True
    assert compute.last_prompt is not None
    assert "Paris is the capital of France." in compute.last_prompt
    assert "What is the capital of France?" in compute.last_prompt

    assert result["ok"] is True
    assert "output_text" in result
    assert result["llm_bypassed"] is False


def test_evaluator_no_cache_usage():
    from baseline.pipeline.evaluator import evaluate_example

    compute = DummyCompute()
    cache = DummyCache()
    cfg = DummyConfig()
    example = make_example()

    # Should NOT raise AssertionError from DummyCache
    result = evaluate_example(example, cache, compute, cfg)

    assert result["ok"] is True
    assert compute.called is True


def test_output_structure_keys():
    from baseline.pipeline.evaluator import evaluate_example

    compute = DummyCompute()
    cfg = DummyConfig()
    example = make_example()

    result = evaluate_example(example, None, compute, cfg)

    expected_keys = [
        "ok",
        "output_text",
        "latency_s",
        "compute_latency_s",
        "rss_before_mb",
        "rss_after_mb",
        "rss_delta_mb",
        "input_tokens",
        "output_tokens",
        "device",
        "dtype",
        "exact_match",
        "token_f1",
        "char_f1",
        "tokens_per_second",
    ]

    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_metrics_computation():
    from baseline.pipeline.evaluator import evaluate_example

    compute = DummyCompute()
    cfg = DummyConfig()
    example = make_example()

    result = evaluate_example(example, None, compute, cfg)

    assert "exact_match" in result
    assert "token_f1" in result
    assert "char_f1" in result
    assert "contains_answer" in result

    assert isinstance(result["exact_match"], float)
    assert isinstance(result["token_f1"], float)
    assert isinstance(result["char_f1"], float)


def test_evaluator_reports_stateless_baseline_flags():
    from baseline.pipeline.evaluator import evaluate_example

    compute = DummyCompute()
    cfg = DummyConfig()
    example = make_example()

    result = evaluate_example(example, None, compute, cfg)

    assert result["llm_bypassed"] is False
    assert result["served_from"] == "tier0_compute"
    assert result["source_tier"] == "tier0_compute"
    assert result["tier_path"] == ["tier0_compute"]

    # Keep this assertion only if the evaluator still emits cache_hit.
    if "cache_hit" in result:
        assert result["cache_hit"] is False
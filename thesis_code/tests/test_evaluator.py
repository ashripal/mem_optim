# tests/test_evaluator.py
"""
Unit tests for baseline/pipeline/evaluator.py using a fake compute engine.

These tests verify evaluator features without loading any real HF model:
- cache miss -> compute called, cache populated
- cache hit -> compute NOT called, latency_s == 0
- rss fields exist and are numeric
- tokens_per_second computation behavior
- metadata passthrough (device/truncated/input_tokens/output_tokens)

Run:
  pytest -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from baseline.pipeline.evaluator import evaluate_example
from baseline.tiers.tier1_cache import LRUCache


@dataclass
class _Cfg:
    max_input_tokens: int = 128
    max_new_tokens: int = 16


class FakeComputeEngine:
    """
    Deterministic fake for ComputeEngine.generate().
    Tracks call count and returns stable metadata.
    """

    def __init__(self):
        self.calls = 0
        self.last_prompt = None

    def generate(self, *, prompt: str, max_input_tokens: int, max_new_tokens: int) -> Dict[str, Any]:
        self.calls += 1
        self.last_prompt = prompt

        # Make token counts depend on prompt length to emulate variability (still deterministic)
        in_tokens = min(max_input_tokens, max(1, len(prompt) // 5))
        out_tokens = max(1, min(max_new_tokens, 7))

        return {
            "ok": True,
            "device": "cpu",
            "truncated": len(prompt) > 2000,  # arbitrary, deterministic condition
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "gen_time_s": 0.25,
            "prompt_used": prompt[:100],
            "output_text": "FAKE_OUTPUT",
        }


def test_evaluator_cache_miss_then_hit_behavior():
    cfg = _Cfg(max_input_tokens=64, max_new_tokens=8)
    cache = LRUCache(capacity=8)
    compute = FakeComputeEngine()

    example = {"context": "ctx", "question": "what is x?", "answer": "y", "task": "trec", "example_id": 0}

    # First pass: miss -> compute called
    r1 = evaluate_example(example=example, cache=cache, compute=compute, cfg=cfg)
    assert r1["ok"] is True
    assert r1["cache_hit"] is False
    assert compute.calls == 1
    assert r1["device"] == "cpu"
    assert r1["input_tokens"] is not None
    assert r1["output_tokens"] is not None
    assert isinstance(r1["latency_s"], float)
    assert r1["latency_s"] > 0.0

    # Second pass: hit -> compute NOT called, latency should be 0
    r2 = evaluate_example(example=example, cache=cache, compute=compute, cfg=cfg)
    assert r2["ok"] is True
    assert r2["cache_hit"] is True
    assert compute.calls == 1  # unchanged
    assert r2["latency_s"] == 0.0

    # Tokens/sec should be None on cache hit (since latency=0)
    assert r2["tokens_per_second"] is None


def test_evaluator_tokens_per_second_computed_on_miss(monkeypatch):
    """
    Ensure tokens_per_second is computed as output_tokens / latency_s when latency_s > 0.
    We monkeypatch time.time to make latency deterministic.
    """
    cfg = _Cfg(max_input_tokens=64, max_new_tokens=8)
    cache = LRUCache(capacity=8)
    compute = FakeComputeEngine()

    example = {"context": "ctx", "question": "q", "task": "trec", "example_id": 0}

    # Patch time.time used inside evaluator module
    import baseline.pipeline.evaluator as evaluator_mod

    times = iter([100.0, 101.0])  # latency = 1.0 second
    monkeypatch.setattr(evaluator_mod.time, "time", lambda: next(times))

    r = evaluate_example(example=example, cache=cache, compute=compute, cfg=cfg)

    assert r["cache_hit"] is False
    assert r["latency_s"] == pytest.approx(1.0)
    assert r["output_tokens"] == 7  # from FakeComputeEngine
    assert r["tokens_per_second"] == pytest.approx(7.0)


def test_evaluator_includes_memory_fields():
    cfg = _Cfg()
    cache = LRUCache(capacity=8)
    compute = FakeComputeEngine()

    example = {"context": "ctx", "question": "q", "task": "trec", "example_id": 0}

    r = evaluate_example(example=example, cache=cache, compute=compute, cfg=cfg)

    for k in ("rss_before_mb", "rss_after_mb", "rss_delta_mb"):
        assert k in r
        assert isinstance(r[k], float)

    assert r["rss_delta_mb"] == pytest.approx(r["rss_after_mb"] - r["rss_before_mb"], rel=1e-6, abs=1e-6)


def test_evaluator_passthrough_flags_from_compute():
    cfg = _Cfg(max_input_tokens=64, max_new_tokens=8)
    cache = LRUCache(capacity=8)
    compute = FakeComputeEngine()

    long_context = "x" * 5000
    example = {"context": long_context, "question": "q", "task": "trec", "example_id": 0}

    r = evaluate_example(example=example, cache=cache, compute=compute, cfg=cfg)

    assert r["device"] == "cpu"
    assert isinstance(r["truncated"], bool)
    assert r["truncated"] is True
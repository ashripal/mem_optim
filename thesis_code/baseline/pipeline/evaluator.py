# pipeline/evaluator.py
"""
Per-example evaluation logic for the LongBench baseline.

This module is responsible for:
- Extracting input text from the dataset example
- Handling Tier 1 (RAM) cache lookup/insert
- Invoking Tier 0 (compute) for tokenization + generation
- Measuring latency + memory usage
- Returning a structured result dictionary

It should NOT:
- Load models
- Open JSONL files
- Handle plotting or aggregation
"""

from __future__ import annotations

import time
from typing import Any, Dict

from baseline.utils.metrics import compute_basic_metrics
from baseline.utils.system import get_rss_mb


def _build_prompt(example: Dict[str, Any]) -> str:
    """
    Construct the input prompt from a LongBench-style example.
    Adjust this if your dataset schema changes.
    """
    context = example.get("context", "")
    question = example.get("question", "")
    return f"{context}\n\n{question}".strip()


def evaluate_example(
    example: Dict[str, Any],
    cache: Any,
    compute: Any,
    cfg: Any,
) -> Dict[str, Any]:
    """
    Evaluate a single example.

    Args:
        example: A dict representing one dataset example.
        cache: Tier 1 cache object (e.g., LRUCache).
        compute: Tier 0 compute engine.
        cfg: Global config object.

    Returns:
        A structured result dictionary suitable for JSONL logging.
    """

    # ------------------------------
    # Build prompt
    # ------------------------------
    prompt = _build_prompt(example)
    cache_key = hash(prompt)

    # ------------------------------
    # Memory before
    # ------------------------------
    rss_before = get_rss_mb()

    # ------------------------------
    # Total timer starts here
    # ------------------------------
    total_t0 = time.time()

    # ------------------------------
    # Tier 1 lookup timing
    # ------------------------------
    lookup_t0 = time.time()
    cached = cache.get(cache_key)
    lookup_latency_s = time.time() - lookup_t0

    cache_hit = cached is not None

    if cache_hit:
        output_text = cached["output_text"]
        meta = cached["meta"]

        compute_latency_s = 0.0
        served_from = "tier1_ram"
        tier_path = ["tier1_ram"]
        llm_bypassed = True

    else:
        # ------------------------------
        # Tier 0 compute
        # ------------------------------
        compute_t0 = time.time()

        generation = compute.generate(
            prompt=prompt,
            max_input_tokens=getattr(cfg, "max_input_tokens", 8192),
            max_new_tokens=getattr(cfg, "max_new_tokens", 64),
        )

        compute_latency_s = time.time() - compute_t0

        if not generation.get("ok", False):
            raise RuntimeError(
                f"Tier0 generation failed on device={generation.get('device')}: "
                f"{generation.get('error', 'unknown error')}"
            )

        if "output_text" not in generation:
            raise RuntimeError(
                f"Tier0 generation returned ok=True but no output_text. "
                f"Keys={sorted(generation.keys())}"
            )

        output_text = generation["output_text"]
        meta = generation

        cache.put(
            cache_key,
            {
                "output_text": output_text,
                "meta": meta,
            },
        )

        served_from = "tier0_compute"
        tier_path = ["tier1_ram", "tier0_compute"]
        llm_bypassed = False

    total_latency_s = time.time() - total_t0

    # ------------------------------
    # Memory after
    # ------------------------------
    rss_after = get_rss_mb()

    # ------------------------------
    # Quality metrics
    # ------------------------------
    quality = compute_basic_metrics(output_text, example)

    # ------------------------------
    # Build result record
    # ------------------------------
    result: Dict[str, Any] = {
        "type": "example_result",
        "ok": True,
        "task": example.get("task"),
        "example_id": example.get("example_id"),

        # Cache / serving behavior
        "cache_hit": cache_hit,
        "served_from": served_from,
        "source_tier": served_from,  # kept for compatibility with benchmark executor
        "tier_path": tier_path,
        "llm_bypassed": llm_bypassed,

        # Latency breakdown
        "latency_s": total_latency_s,
        "lookup_latency_s": lookup_latency_s,
        "compute_latency_s": compute_latency_s,
        "gen_time_s": meta.get("gen_time_s"),

        # Memory
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,

        # Generation metadata
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "device": meta.get("device"),
        "truncated": meta.get("truncated"),
        "fallback_from": meta.get("fallback_from"),
        "fallback_reason": meta.get("fallback_reason"),
        "device_switch_persisted": meta.get("device_switch_persisted"),

        # Output / scoring
        "output_text": output_text,
        "ref_text": quality.get("ref_text"),
        "exact_match": quality.get("exact_match"),
        "contains_answer": quality.get("contains_answer"),
        "token_f1": quality.get("token_f1"),
        "char_f1": quality.get("char_f1"),

        # Throughput
        "tokens_per_second": (
            meta.get("output_tokens", 0) / compute_latency_s
            if compute_latency_s and compute_latency_s > 0
            else None
        ),
    }

    return result
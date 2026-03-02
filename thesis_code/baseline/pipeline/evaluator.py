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
    # Tier 1 lookup
    # ------------------------------
    cached = cache.get(cache_key)
    cache_hit = cached is not None

    if cache_hit:
        output_text = cached["output_text"]
        meta = cached["meta"]
        latency_s = 0.0
    else:
        # ------------------------------
        # Tier 0 compute
        # ------------------------------
        t0 = time.time()

        generation = compute.generate(
            prompt=prompt,
            max_input_tokens=getattr(cfg, "max_input_tokens", 8192),
            max_new_tokens=getattr(cfg, "max_new_tokens", 64),
        )

        latency_s = time.time() - t0

        output_text = generation["output_text"]
        meta = generation

        # Store in cache
        cache.put(
            cache_key,
            {
                "output_text": output_text,
                "meta": meta,
            },
        )

    # ------------------------------
    # Memory after
    # ------------------------------
    rss_after = get_rss_mb()

    # ------------------------------
    # Build result record
    # ------------------------------
    result: Dict[str, Any] = {
        "type": "example_result",
        "ok": True,
        "task": example.get("task"),
        "example_id": example.get("example_id"),
        "cache_hit": cache_hit,
        "latency_s": latency_s,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "device": meta.get("device"),
        "truncated": meta.get("truncated"),
        "tokens_per_second": (
            meta.get("output_tokens", 0) / latency_s if latency_s > 0 else None
        ),
    }

    return result
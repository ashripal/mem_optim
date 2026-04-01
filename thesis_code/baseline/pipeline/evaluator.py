"""
Per-example evaluation logic for the LongBench baseline.

TRUE BASELINE BEHAVIOR:
- NO caching
- NO storage or reuse of prior outputs
- ALWAYS call the LLM
- prompt -> compute.generate -> output

This module is responsible for:
- Extracting input text from the dataset example
- Invoking Tier 0 (compute) for tokenization + generation
- Measuring latency + memory usage
- Returning a structured result dictionary

It should NOT:
- Load models
- Open JSONL files
- Handle plotting or aggregation
- Perform cache lookup / insertion
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from baseline.utils.metrics import compute_basic_metrics
from baseline.utils.system import get_rss_mb


def _build_prompt(example: Dict[str, Any]) -> str:
    """
    Construct the input prompt from a LongBench-style example.

    Baseline rule:
    - Use only the current example's provided fields.
    - Do not incorporate any stored or reused prior state.
    """
    context = example.get("context", "")
    question = example.get("question", "")
    return f"{context}\n\n{question}".strip()


def evaluate_example(
    example: Dict[str, Any],
    cache: Optional[Any],  # kept only for temporary API compatibility; intentionally unused
    compute: Any,
    cfg: Any,
) -> Dict[str, Any]:
    """
    Evaluate a single example with a stateless baseline.

    Args:
        example: A dict representing one dataset example.
        cache: Unused. Present only for temporary compatibility with older callers.
        compute: Tier 0 compute engine.
        cfg: Global config object.

    Returns:
        A structured result dictionary suitable for JSONL logging.
    """
    del cache  # make the no-cache baseline behavior explicit

    prompt = _build_prompt(example)

    rss_before = get_rss_mb()
    total_t0 = time.perf_counter()

    # TRUE BASELINE: always call the LLM
    compute_t0 = time.perf_counter()
    generation = compute.generate(
        prompt=prompt,
        max_input_tokens=getattr(cfg, "max_input_tokens", 8192),
        max_new_tokens=getattr(cfg, "max_new_tokens", 64),
    )
    compute_latency_s = time.perf_counter() - compute_t0

    if not generation.get("ok", False):
        raise RuntimeError(
            f"Tier0 generation failed on device={generation.get('device')}: "
            f"{generation.get('error', 'unknown error')}"
        )

    if "output_text" not in generation:
        raise RuntimeError(
            "Tier0 generation returned ok=True but no output_text. "
            f"Keys={sorted(generation.keys())}"
        )

    output_text = generation["output_text"]
    meta = generation

    total_latency_s = time.perf_counter() - total_t0
    rss_after = get_rss_mb()

    quality = compute_basic_metrics(output_text, example)

    result: Dict[str, Any] = {
        "type": "example_result",
        "ok": True,
        "task": example.get("task"),
        "example_id": example.get("example_id"),

        # Baseline: no memory / no cache
        "cache_hit": False,
        "served_from": "tier0_compute",
        "source_tier": "tier0_compute",
        "tier_path": ["tier0_compute"],
        "llm_bypassed": False,

        # Latency
        "latency_s": total_latency_s,
        "lookup_latency_s": 0.0,
        "compute_latency_s": compute_latency_s,
        "gen_time_s": meta.get("gen_time_s"),
        "tokenize_time_s": meta.get("tokenize_time_s"),
        "decode_time_s": meta.get("decode_time_s"),

        # Memory
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,

        # Generation metadata
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "device": meta.get("device"),
        "dtype": meta.get("dtype"),
        "generation_backend": meta.get("generation_backend"),
        "truncated": meta.get("truncated"),
        "fallback_from": meta.get("fallback_from"),
        "fallback_reason": meta.get("fallback_reason"),
        "device_switch_persisted": meta.get("device_switch_persisted"),

        # Optional CUDA telemetry
        "cuda_device_name": meta.get("cuda_device_name"),
        "gpu_mem_allocated_mb": meta.get("gpu_mem_allocated_mb"),
        "gpu_mem_reserved_mb": meta.get("gpu_mem_reserved_mb"),
        "gpu_max_mem_allocated_mb": meta.get("gpu_max_mem_allocated_mb"),
        "gpu_max_mem_reserved_mb": meta.get("gpu_max_mem_reserved_mb"),

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
            if compute_latency_s > 0
            else None
        ),
    }

    return result
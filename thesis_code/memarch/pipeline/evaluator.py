# memarch/pipeline/evaluator.py
"""
Evaluator for memarch.

Responsibilities:
- Measure per-example latency breakdown (total / memory vs generation when available)
- Record memory hit/miss metadata returned by MemoryManager.answer()
- Collect lightweight resource snapshots (RSS MB; optional GPU later)
- Emit JSONL logs via memarch.pipeline.logging.JsonlLogger

Design goals:
- Works on Mac (Apple silicon) and Jetson Orin with no code changes
- Unit-test friendly: you can pass fake generator/manager and a temp logger
- Preserve resolved runtime metadata so baseline and memarch runs are comparable

Current retrieval-aware behavior:
- Supports exact / lexical / semantic retrieval metadata
- Distinguishes direct memory reuse from context-assisted generation
- Preserves retrieval-stage debug info for downstream analysis
- Keeps evaluator-side logic thin and deterministic

Notes:
- We use time.perf_counter() for timing.
- RSS measurement uses psutil if installed; otherwise we degrade gracefully.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Optional, Protocol, Tuple

from memarch.memory.schema import MemoryQuery
from memarch.pipeline.logging import JsonlLogger


# -------------------------
# Optional resource probing
# -------------------------

def _get_rss_mb() -> Optional[float]:
    """
    Return current process RSS in MB, if possible.
    Uses psutil if available; otherwise returns None.
    """
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        rss = float(proc.memory_info().rss) / (1024.0 * 1024.0)
        return rss
    except Exception:
        return None


# -------------------------
# Protocols (to keep evaluator decoupled)
# -------------------------

class MemoryManagerLike(Protocol):
    def answer(self, mq: MemoryQuery, generator: Any) -> Tuple[str, Dict[str, Any]]: ...
    def retrieve(self, mq: MemoryQuery, return_meta: bool = False) -> Any: ...
    def stats(self) -> Dict[str, Any]: ...


class GeneratorLike(Protocol):
    def generate(self, mq: MemoryQuery, retrieved: Optional[Any] = None) -> Tuple[str, Any, Any]: ...
    def info(self) -> Dict[str, Any]: ...


# -------------------------
# Data structures
# -------------------------

@dataclass(frozen=True)
class EvalResult:
    example_id: str
    task: str
    answer_text: str
    meta: Dict[str, Any]
    timings_ms: Dict[str, float]
    resources: Dict[str, Any]


# -------------------------
# Evaluator
# -------------------------

class Evaluator:
    def __init__(
        self,
        *,
        manager: MemoryManagerLike,
        generator: GeneratorLike,
        logger: Optional[JsonlLogger] = None,
    ) -> None:
        self._manager = manager
        self._generator = generator
        self._logger = logger

    def _retrieve_with_debug(self, mq: MemoryQuery) -> Tuple[Any, Dict[str, Any], float]:
        """
        Time retrieval separately and return:
          (hit, retrieve_debug, retrieval_probe_ms)

        Supports both:
        - retrieve(mq, return_meta=True) -> (hit, debug)
        - older retrieve(mq) -> hit
        """
        t0 = time.perf_counter()
        try:
            retrieve_result = self._manager.retrieve(mq, return_meta=True)
        except TypeError:
            retrieve_result = self._manager.retrieve(mq)
        t1 = time.perf_counter()
        retrieval_probe_ms = (t1 - t0) * 1000.0

        if isinstance(retrieve_result, tuple) and len(retrieve_result) == 2:
            hit, retrieve_dbg = retrieve_result
        else:
            hit, retrieve_dbg = retrieve_result, {}

        return hit, dict(retrieve_dbg or {}), float(retrieval_probe_ms)

    def evaluate_one(
        self,
        *,
        example_id: str,
        task: str,
        mq: MemoryQuery,
        log_query_text: bool = True,
    ) -> EvalResult:
        """
        Evaluate a single MemoryQuery.

        Timing breakdown approach:
        - We measure retrieval-only time by calling manager.retrieve() once first.
        - We then measure full answer() latency separately.
        - If a hit existed and answer() ultimately served directly from memory,
          generation_ms is 0.
        - Otherwise generation_ms is approximated as total - memory_lookup_ms.

        This keeps the evaluator simple and deterministic while preserving the
        exact / lexical / semantic metadata returned by MemoryManager.answer().
        """
        # 1) time retrieval separately (for breakdown + debugging)
        hit, retrieve_dbg, retrieval_probe_ms = self._retrieve_with_debug(mq)
        memory_lookup_ms = float(
            retrieve_dbg.get("memory_lookup_ms", retrieval_probe_ms) or retrieval_probe_ms
        )

        # 2) time full answer path
        t2 = time.perf_counter()
        answer_text, meta = self._manager.answer(mq, self._generator)
        t3 = time.perf_counter()
        total_ms = (t3 - t2) * 1000.0

        meta = dict(meta or {})

        used_memory = bool(meta.get("used_memory", False))
        generated = bool(meta.get("generated", False))
        source_tier = meta.get("source_tier")
        match_type = meta.get("match_type")
        retrieval_stage = meta.get("retrieval_stage", retrieve_dbg.get("retrieval_stage"))

        # exact / lexical / semantic normalized flags
        lexical_used = bool(meta.get("lexical_used", False))
        lexical_bypassed = bool(meta.get("lexical_bypassed", False))
        lexical_context_used = bool(meta.get("lexical_context_used", False))

        semantic_used = bool(meta.get("semantic_used", False))
        semantic_bypassed = bool(meta.get("semantic_bypassed", False))
        semantic_context_used = semantic_used and generated and not semantic_bypassed

        # 3) estimate generation_ms
        if hit is not None and used_memory:
            generation_ms = 0.0
        else:
            generation_ms = float(
                meta.get("generation_ms_est", max(0.0, total_ms - memory_lookup_ms))
            )

        timings_ms = {
            "memory_lookup_ms": float(memory_lookup_ms),
            "generation_ms_est": float(generation_ms),
            "total_ms": float(total_ms),
        }

        resources: Dict[str, Any] = {}
        rss = _get_rss_mb()
        if rss is not None:
            resources["rss_mb"] = rss

        # Include store stats snapshot if available
        try:
            resources["store_stats"] = self._manager.stats()
        except Exception:
            pass

        # Add generator info if available
        try:
            resources["generator_info"] = self._generator.info()
        except Exception:
            pass

        # Capture generator-side runtime metadata if available
        generator_meta = dict(getattr(self._generator, "last_generation_meta", {}) or {})

        retrieved_memory = meta.get("retrieved_memory")
        hit_debug = dict(getattr(hit, "debug", {}) or {}) if hit is not None else {}

        eval_meta: Dict[str, Any] = {
            **meta,

            # retrieval probe metadata
            "retrieval_hit_present": hit is not None,
            "retrieval_hit_match_type": getattr(getattr(hit, "match_type", None), "value", None)
            if hit is not None else None,
            "retrieval_hit_source_tier": getattr(getattr(hit, "source_tier", None), "value", None)
            if hit is not None else None,
            "retrieval_probe_memory_lookup_ms": float(retrieval_probe_ms),
            "retrieval_stage": retrieval_stage,
            "namespaces_checked": meta.get(
                "namespaces_checked",
                retrieve_dbg.get("namespaces_checked", []),
            ),

            # normalized serving fields
            "used_memory": used_memory,
            "generated": generated,
            "source_tier": source_tier,
            "match_type": match_type,
            "score": meta.get("score"),

            # lexical retrieval metadata
            "lexical_used": lexical_used,
            "lexical_bypassed": lexical_bypassed,
            "lexical_context_used": lexical_context_used,
            "lexical_reason": meta.get(
                "lexical_reason",
                retrieve_dbg.get("reason") if retrieval_stage == "lexical" else None,
            ),
            "lexical_candidate_count": meta.get(
                "lexical_candidate_count",
                retrieve_dbg.get("candidate_count") if retrieval_stage == "lexical" else None,
            ),
            "lexical_top_score": meta.get(
                "lexical_top_score",
                retrieve_dbg.get("top_score") if retrieval_stage == "lexical" else None,
            ),
            "lexical_top_rank": meta.get(
                "lexical_top_rank",
                retrieve_dbg.get("top_rank") if retrieval_stage == "lexical" else None,
            ),
            "lexical_enabled_debug": meta.get(
                "lexical_enabled_debug",
                retrieve_dbg.get("lexical_enabled") if "lexical_enabled" in retrieve_dbg else None,
            ),
            "lexical_match_type": meta.get(
                "lexical_match_type",
                retrieve_dbg.get("lexical_match_type") if retrieval_stage == "lexical" else None,
            ),
            "lexical_same_source": meta.get(
                "lexical_same_source",
                retrieve_dbg.get("same_source") if retrieval_stage == "lexical" else None,
            ),

            # semantic retrieval metadata
            "semantic_used": semantic_used,
            "semantic_context_used": semantic_context_used,
            "semantic_bypassed": semantic_bypassed,
            "semantic_candidate_rank": meta.get("semantic_candidate_rank"),
            "semantic_score": meta.get(
                "semantic_score",
                meta.get("score") if match_type == "semantic" else None,
            ),
            "semantic_reason": meta.get(
                "semantic_reason",
                retrieve_dbg.get("reason") if retrieval_stage == "semantic" else None,
            ),
            "semantic_candidate_count": meta.get(
                "semantic_candidate_count",
                retrieve_dbg.get("candidate_count") if retrieval_stage == "semantic" else None,
            ),
            "semantic_top_score": meta.get(
                "semantic_top_score",
                retrieve_dbg.get("top_score") if retrieval_stage == "semantic" else None,
            ),
            "semantic_top_rank": meta.get(
                "semantic_top_rank",
                retrieve_dbg.get("top_rank") if retrieval_stage == "semantic" else None,
            ),
            "semantic_enabled_debug": meta.get(
                "semantic_enabled_debug",
                retrieve_dbg.get("semantic_enabled") if "semantic_enabled" in retrieve_dbg else None,
            ),

            # retrieved-hit summary before generation
            "retrieved_memory": retrieved_memory,
            "hit_before_generate": meta.get("hit_before_generate"),
            "retrieval_hit_debug": hit_debug if hit_debug else None,

            # query/source/evidence metadata
            "doc_signature": meta.get("doc_signature"),
            "source_file": meta.get("source_file"),
            "chunk_index": meta.get("chunk_index"),
            "chunk_id": meta.get("chunk_id"),
            "question_type": meta.get("question_type"),
            "query_evidence_text": meta.get("query_evidence_text"),
            "query_evidence_chars": meta.get("query_evidence_chars"),
            "stored_evidence_text": meta.get("stored_evidence_text"),
            "stored_evidence_chars": meta.get("stored_evidence_chars"),
            "normalized_answer_for_storage": meta.get("normalized_answer_for_storage"),

            # storage metadata
            "stored": meta.get("stored"),
            "stored_scopes": meta.get("stored_scopes", []),
            "store_skipped": meta.get("store_skipped", []),

            # generator/runtime metadata
            "device": generator_meta.get("device"),
            "dtype": generator_meta.get("dtype"),
            "generation_backend": generator_meta.get("generation_backend"),
            "decoding_mode": generator_meta.get("decoding_mode"),
            "num_beams": generator_meta.get("num_beams"),
            "do_sample": generator_meta.get("do_sample"),
            "temperature": generator_meta.get("temperature"),
            "top_p": generator_meta.get("top_p"),
            "input_tokens": generator_meta.get("input_tokens"),
            "output_tokens": generator_meta.get("output_tokens"),
            "truncated": generator_meta.get("truncated"),
            "tokenize_time_s": generator_meta.get("tokenize_time_s"),
            "gen_time_s": generator_meta.get("gen_time_s"),
            "decode_time_s": generator_meta.get("decode_time_s"),
            "cuda_device_name": generator_meta.get("cuda_device_name"),
            "gpu_mem_allocated_mb": generator_meta.get("gpu_mem_allocated_mb"),
            "gpu_mem_reserved_mb": generator_meta.get("gpu_mem_reserved_mb"),

            # retrieved-context details from generator path
            "used_retrieved_context": generator_meta.get("used_retrieved_context"),
            "retrieved_match_type": generator_meta.get("retrieved_match_type"),
            "retrieved_source_tier": generator_meta.get("retrieved_source_tier"),
            "retrieved_score": generator_meta.get("retrieved_score"),
            "retrieved_doc_signature_match": generator_meta.get("retrieved_doc_signature_match"),
            "retrieved_evidence_chars": generator_meta.get("retrieved_evidence_chars"),
            "reduced_context_used": generator_meta.get("reduced_context_used"),
            "full_context_chars": generator_meta.get("full_context_chars"),
            "final_context_chars": generator_meta.get("final_context_chars"),
        }

        result = EvalResult(
            example_id=example_id,
            task=task,
            answer_text=answer_text,
            meta=eval_meta,
            timings_ms=timings_ms,
            resources=resources,
        )

        if self._logger is not None:
            self._logger.log_example(
                example_id=example_id,
                task=task,
                query=mq.raw_query if log_query_text else "",
                meta=eval_meta,
                timings_ms=timings_ms,
                resources=resources,
            )

        return result

    def evaluate_many(
        self,
        *,
        examples: Iterable[Tuple[str, str, MemoryQuery]],
        log_query_text: bool = True,
    ) -> Iterator[EvalResult]:
        """
        Evaluate multiple examples.

        Input format:
          examples yields (example_id, task, MemoryQuery)
        """
        for example_id, task, mq in examples:
            yield self.evaluate_one(
                example_id=example_id,
                task=task,
                mq=mq,
                log_query_text=log_query_text,
            )
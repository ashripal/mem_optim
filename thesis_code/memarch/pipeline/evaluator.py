# memarch/pipeline/evaluator.py
"""
Evaluator for memarch.

Responsibilities:
- Measure per-example latency breakdown (total / memory vs generation when available)
- Record memory hit/miss metadata returned by MemoryManager.answer()
- Collect lightweight resource snapshots (RSS MB; optional GPU later)
- Emit JSONL logs via memarch.pipeline.logging.JsonlLogger

Design goals:
- Works on Mac (Apple silicon) and Jetson Orin Nano with no code changes
- Unit-test friendly: you can pass fake generator/manager and a temp logger

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
    def retrieve(self, mq: MemoryQuery) -> Any: ...
    def stats(self) -> Dict[str, Any]: ...


class GeneratorLike(Protocol):
    def generate(self, mq: MemoryQuery, retrieved: Optional[Any] = None) -> Tuple[str, Any, Any]: ...


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
        - We measure total time around manager.answer().
        - We also measure retrieval-only time by calling manager.retrieve() once first.
          This adds a small overhead but gives a consistent "memory_lookup_ms".
        - If retrieval returns a hit and the manager returns directly, generation_ms is 0.
          Otherwise generation_ms is approximated as total - memory_lookup_ms.
        """
        # 1) time retrieval separately (for breakdown)
        t0 = time.perf_counter()
        hit = self._manager.retrieve(mq)
        t1 = time.perf_counter()
        memory_lookup_ms = (t1 - t0) * 1000.0

        # 2) time full answer path
        t2 = time.perf_counter()
        answer_text, meta = self._manager.answer(mq, self._generator)
        t3 = time.perf_counter()
        total_ms = (t3 - t2) * 1000.0

        # 3) estimate generation_ms
        # If a hit existed and manager returns directly, generation cost is effectively 0 here.
        used_memory = bool(meta.get("used_memory"))
        if hit is not None and used_memory:
            generation_ms = 0.0
        else:
            # Approximate; note retrieval was measured separately, so this is a coarse proxy.
            generation_ms = max(0.0, total_ms - memory_lookup_ms)

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

        result = EvalResult(
            example_id=example_id,
            task=task,
            answer_text=answer_text,
            meta=meta,
            timings_ms=timings_ms,
            resources=resources,
        )

        if self._logger is not None:
            self._logger.log_example(
                example_id=example_id,
                task=task,
                query=mq.raw_query if log_query_text else "",
                meta={
                    **meta,
                    # Helpful for debugging: did we have a hit before generating?
                    "retrieval_hit_present": hit is not None,
                },
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
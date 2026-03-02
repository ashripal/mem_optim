# memarch/pipeline/runner.py
"""
Runner: orchestrates loading examples, running evaluation, and writing logs.

Phase 1 goals:
- Keep runner thin and deterministic
- Accept an iterator of MemoryQuery objects (from LongBench or other datasets)
- Use Evaluator to measure and log
- Produce a small in-memory summary for convenience (optional)

This module should not embed dataset-specific logic; that belongs in memarch/data/*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple

from memarch.memory.schema import MemoryQuery
from memarch.pipeline.evaluator import Evaluator, EvalResult
from memarch.pipeline.logging import JsonlLogger, RunInfo


class ExampleSource(Protocol):
    """
    Dataset adapter protocol.

    Implementations in memarch/data/* should yield:
      (example_id, task, MemoryQuery)
    """
    def __iter__(self) -> Iterator[Tuple[str, str, MemoryQuery]]: ...


@dataclass(frozen=True)
class RunSummary:
    num_examples: int
    num_used_memory: int
    avg_total_ms: float
    avg_memory_lookup_ms: float
    avg_generation_ms_est: float


def _safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class Runner:
    def __init__(
        self,
        *,
        evaluator: Evaluator,
        logger: Optional[JsonlLogger] = None,
    ) -> None:
        self._evaluator = evaluator
        self._logger = logger

    def run(
        self,
        *,
        examples: Iterable[Tuple[str, str, MemoryQuery]],
        log_query_text: bool = True,
        max_examples: Optional[int] = None,
    ) -> RunSummary:
        """
        Run evaluation over examples and return a small summary.

        Args:
          examples: iterable yielding (example_id, task, MemoryQuery)
          log_query_text: whether to include raw query text in logs
          max_examples: optional cap
        """
        total_ms: List[float] = []
        mem_ms: List[float] = []
        gen_ms: List[float] = []
        used_memory_count = 0
        n = 0

        for example_id, task, mq in examples:
            if max_examples is not None and n >= max_examples:
                break

            res: EvalResult = self._evaluator.evaluate_one(
                example_id=example_id,
                task=task,
                mq=mq,
                log_query_text=log_query_text,
            )

            n += 1
            if bool(res.meta.get("used_memory")):
                used_memory_count += 1

            total_ms.append(float(res.timings_ms.get("total_ms", 0.0)))
            mem_ms.append(float(res.timings_ms.get("memory_lookup_ms", 0.0)))
            gen_ms.append(float(res.timings_ms.get("generation_ms_est", 0.0)))

        return RunSummary(
            num_examples=n,
            num_used_memory=used_memory_count,
            avg_total_ms=_safe_mean(total_ms),
            avg_memory_lookup_ms=_safe_mean(mem_ms),
            avg_generation_ms_est=_safe_mean(gen_ms),
        )


def make_default_logger(log_path: str, notes: Optional[str] = None) -> JsonlLogger:
    """
    Convenience for scripts: create a run logger with run_start/run_end events.
    """
    run_info = RunInfo.create(notes=notes)
    return JsonlLogger(log_path, run_info=run_info)
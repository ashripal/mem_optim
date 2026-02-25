# utils/metrics.py
"""
metrics.py

Lightweight metrics + logging helpers for the baseline memory architecture.

Goals:
- Measure end-to-end latency per turn (and optionally sub-stage latencies).
- Track memory usage (RSS) in a portable way (Mac/Linux).
- Write JSONL records suitable for plotting later.

No external dependencies beyond:
- psutil (recommended)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def now_s() -> float:
    """Wall-clock seconds (perf counter for timing)."""
    return time.perf_counter()


def rss_mb() -> Optional[float]:
    """Resident set size in MB (process memory). Returns None if psutil unavailable."""
    if psutil is None:
        return None
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)


@dataclass
class Timer:
    """Context manager timer for a named stage."""
    name: str
    t0: float = field(default=0.0, init=False)
    dt: float = field(default=0.0, init=False)

    def __enter__(self) -> "Timer":
        self.t0 = now_s()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.dt = now_s() - self.t0


@dataclass
class TurnMetrics:
    """
    A single 'turn' record. Keep this flat-ish so it writes cleanly to JSONL.
    """
    session_id: str
    turn_id: int
    variant_type: str  # original | repeat | paraphrase
    task_file: str
    example_index: int

    # Prompt sizing
    prompt_chars: int
    prompt_tokens: Optional[int] = None
    truncated: Optional[bool] = None
    max_input_tokens: Optional[int] = None

    # Cache behavior
    qa_cache_hit: Optional[bool] = None
    embed_cache_hit: Optional[bool] = None
    retrieved_neighbors: Optional[int] = None

    # Outputs
    answer_pred: str = ""
    answers_gold: Optional[list[str]] = None

    # Performance
    latency_s: Optional[float] = None
    tokens_out: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    device_used: Optional[str] = None

    # Memory
    rss_mb_before: Optional[float] = None
    rss_mb_after: Optional[float] = None

    # Errors
    status: str = "ok"
    error: Optional[str] = None

    # Optional stage breakdowns
    stage_s: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "status": self.status,
            "error": self.error,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "variant_type": self.variant_type,
            "task_file": self.task_file,
            "example_index": self.example_index,
            "prompt_chars": self.prompt_chars,
            "prompt_tokens": self.prompt_tokens,
            "truncated": self.truncated,
            "max_input_tokens": self.max_input_tokens,
            "qa_cache_hit": self.qa_cache_hit,
            "embed_cache_hit": self.embed_cache_hit,
            "retrieved_neighbors": self.retrieved_neighbors,
            "answers_gold": self.answers_gold,
            "answer_pred": self.answer_pred,
            "latency_s": self.latency_s,
            "tokens_out": self.tokens_out,
            "tokens_per_sec": self.tokens_per_sec,
            "device_used": self.device_used,
            "rss_mb_before": self.rss_mb_before,
            "rss_mb_after": self.rss_mb_after,
            "stage_s": self.stage_s,
        }
        return d


class JsonlLogger:
    """Append-only JSONL logger (safe for long runs)."""

    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open("w", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
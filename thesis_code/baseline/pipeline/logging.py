"""
JSONL logging utilities for experiment runs.

Optimized for:
- Edge devices (Jetson AGX Orin)
- Low I/O overhead
- Reliable but not over-synchronized writes

Design goals:
- Append-only JSONL output
- Buffered writes with periodic flush
- Minimal impact on latency measurements
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, TextIO


def _json_fallback(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


class JSONLLogger:
    """
    Efficient JSONL writer with batched flushing.

    Args:
        path: Output file path
        ensure_dir: Create directory if needed
        flush_every: Flush after N writes (default optimized for edge)
        fsync: Force OS-level sync (slow, disabled by default)
    """

    def __init__(
        self,
        path: str,
        *,
        ensure_dir: bool = True,
        flush_every: int = 20,   # 🔥 changed from 1 → 20
        fsync: bool = False,     # 🔥 new (optional durability)
    ):
        self.path = os.path.abspath(path)
        self.flush_every = max(1, int(flush_every))
        self.fsync = fsync
        self._n_written = 0
        self._fh: Optional[TextIO] = None

        if ensure_dir:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # Use moderate buffering instead of line-buffering
        self._fh = open(self.path, "w", encoding="utf-8")

    @property
    def n_written(self) -> int:
        return self._n_written

    def write(self, record: Dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JSONLLogger is closed.")

        line = json.dumps(record, ensure_ascii=False, default=_json_fallback)
        self._fh.write(line + "\n")

        self._n_written += 1

        if (self._n_written % self.flush_every) == 0:
            self._flush()

    def _flush(self) -> None:
        if self._fh is None:
            return

        self._fh.flush()

        if self.fsync:
            try:
                os.fsync(self._fh.fileno())
            except Exception:
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._flush()
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
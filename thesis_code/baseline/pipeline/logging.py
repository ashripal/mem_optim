# pipeline/logging.py
"""
JSONL logging utilities for experiment runs.

This module is responsible for writing newline-delimited JSON records safely and
consistently. It is intentionally small and reusable across baselines/variants.

Design goals:
- Append-only JSONL output
- Flush often for robustness (runs can be long)
- Handle non-JSON-serializable values gracefully (best-effort)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, TextIO


def _json_fallback(obj: Any) -> str:
    """
    Best-effort fallback for objects that aren't JSON serializable.
    """
    try:
        return str(obj)
    except Exception:
        return repr(obj)


class JSONLLogger:
    """
    Simple JSONL writer.

    Usage:
        logger = JSONLLogger("path/to/run.jsonl")
        logger.write({"type": "run_header", ...})
        logger.write({"type": "example_result", ...})
        logger.close()
    """

    def __init__(self, path: str, *, ensure_dir: bool = True, flush_every: int = 1):
        self.path = os.path.abspath(path)
        self.flush_every = max(1, int(flush_every))
        self._n_written = 0
        self._fh: Optional[TextIO] = None

        if ensure_dir:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # Use line buffering where available; also manually flush per flush_every.
        self._fh = open(self.path, "w", encoding="utf-8", buffering=1)

    @property
    def n_written(self) -> int:
        return self._n_written

    def write(self, record: Dict[str, Any]) -> None:
        """
        Write one JSON object as a single line.

        Args:
            record: Dict that will be serialized to JSON.
        """
        if self._fh is None:
            raise RuntimeError("JSONLLogger is closed.")

        line = json.dumps(record, ensure_ascii=False, default=_json_fallback)
        self._fh.write(line + "\n")

        self._n_written += 1
        if (self._n_written % self.flush_every) == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
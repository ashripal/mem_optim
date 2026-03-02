# memarch/pipeline/logging.py
"""
Structured logging for memarch runs.

Goals:
- Append-only JSONL logs (easy to stream, analyze, and summarize)
- Portable across macOS / Jetson / other Linux devices
- Deterministic schema (committee-friendly)

What we log per example:
- identifiers (run_id, example_id, task)
- query + minimal context preview (optional)
- timings (total, memory, generation)
- memory decision metadata (hit/miss, scope, tier)
- resource snapshots (rss_mb, gpu mem if available later)
- store outcomes (what was written)

This module intentionally avoids any dependency on the rest of the pipeline, so it can be
used in unit tests and by scripts/analysis later.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    created_at_utc: str
    host: str
    notes: Optional[str] = None

    @staticmethod
    def create(notes: Optional[str] = None) -> "RunInfo":
        return RunInfo(
            run_id=str(uuid.uuid4()),
            created_at_utc=utc_now_iso(),
            host=os.uname().nodename if hasattr(os, "uname") else "unknown",
            notes=notes,
        )


class JsonlLogger:
    """
    Append-only JSONL writer.

    Usage:
      logger = JsonlLogger("artifacts/runs/run_001/log.jsonl", run_info=RunInfo.create())
      logger.write_event({...})
      logger.close()
    """

    def __init__(self, path: str, *, run_info: Optional[RunInfo] = None) -> None:
        if not path:
            raise ValueError("path must be non-empty")
        self.path = path
        ensure_parent_dir(self.path)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._closed = False

        self.run_info = run_info
        if self.run_info is not None:
            # Write a run header as the first event (idempotent-ish if repeated).
            self.write_event(
                {
                    "type": "run_start",
                    "run_id": self.run_info.run_id,
                    "created_at_utc": self.run_info.created_at_utc,
                    "host": self.run_info.host,
                    "notes": self.run_info.notes,
                }
            )

    def write_event(self, event: Dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("logger is closed")
        if not isinstance(event, dict):
            raise TypeError("event must be a dict")

        # Ensure every event has a timestamp
        if "ts_utc" not in event:
            event["ts_utc"] = utc_now_iso()

        line = _stable_json_dumps(event)
        self._fh.write(line + "\n")
        self._fh.flush()

    def log_example(
        self,
        *,
        example_id: str,
        task: str,
        query: str,
        meta: Dict[str, Any],
        timings_ms: Dict[str, float],
        resources: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Convenience wrapper for a standard per-example event.
        """
        evt: Dict[str, Any] = {
            "type": "example",
            "run_id": self.run_info.run_id if self.run_info else None,
            "example_id": example_id,
            "task": task,
            "query": query,
            "meta": meta,
            "timings_ms": timings_ms,
        }
        if resources is not None:
            evt["resources"] = resources
        self.write_event(evt)

    def close(self) -> None:
        if self._closed:
            return
        if self.run_info is not None:
            self.write_event(
                {
                    "type": "run_end",
                    "run_id": self.run_info.run_id,
                }
            )
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
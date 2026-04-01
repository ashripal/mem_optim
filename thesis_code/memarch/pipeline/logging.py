# memarch/pipeline/logging.py

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
    def __init__(self, path: str, *, run_info: Optional[RunInfo] = None) -> None:
        if not path:
            raise ValueError("path must be non-empty")
        self.path = path
        ensure_parent_dir(self.path)

        # enable OS-level buffering
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self._closed = False

        # ✅ NEW: flush batching
        self._write_count = 0
        self._flush_every = 50  # flush every 50 writes

        self.run_info = run_info
        if self.run_info is not None:
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

        if "ts_utc" not in event:
            event["ts_utc"] = utc_now_iso()

        line = _stable_json_dumps(event)
        self._fh.write(line + "\n")

        # ✅ buffered flush
        self._write_count += 1
        if self._write_count % self._flush_every == 0:
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
        meta = dict(meta or {})
        timings_ms = dict(timings_ms or {})

        evt: Dict[str, Any] = {
            "type": "example",
            "run_id": self.run_info.run_id if self.run_info else None,
            "example_id": example_id,
            "task": task,
            "query": query,

            "used_memory": bool(meta.get("used_memory", False)),
            "generated": bool(meta.get("generated", False)),
            "source_tier": meta.get("source_tier"),
            "match_type": meta.get("match_type"),
            "score": meta.get("score"),

            "semantic_used": bool(meta.get("semantic_used", False)),
            "semantic_bypassed": bool(meta.get("semantic_bypassed", False)),
            "semantic_candidate_rank": meta.get("semantic_candidate_rank"),

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
        self._fh.flush()  # final flush
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
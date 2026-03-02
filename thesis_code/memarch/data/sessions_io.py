# memarch/data/sessions_io.py
"""
Session dataset I/O utilities.

This module supports a simple, portable JSONL format for "conversation sessions"
that you can generate from LongBench or other sources.

Primary goal:
- Provide a consistent adapter that yields:
    (example_id, task, MemoryQuery)

Recommended JSONL schema (per line):
{
  "id": "ex_0001",                 # optional; fallback created if missing
  "task": "trec",                  # optional; fallback to provided default_task
  "user_id": "u123",               # optional
  "session_id": "s456",            # optional
  "cohort_id": "c789",             # optional
  "query": "....",                 # REQUIRED (or 'question'/'input')
  "dataset_context": "....",       # optional (LongBench passage)
  "context": {...},                # optional extra structured context (JSON object)
  "meta": {...}                    # optional metadata (JSON object)
}

Notes:
- We keep dataset_context separate and store it into MemoryQuery.context["dataset_context"].
- We also preserve any additional context in MemoryQuery.context["extra_context"].
- Do NOT store massive raw docs in context unless you intend to pay the cost in signatures/keys.

This file intentionally does not depend on baseline code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from memarch.memory.schema import MemoryQuery


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sessions file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def _coerce_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _first_present(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _example_id(record: Dict[str, Any], idx: int) -> str:
    rid = _first_present(record, ("id", "example_id", "qid", "uid"))
    return _coerce_str(rid) if rid is not None else f"ex_{idx:08d}"


def _extract_query(record: Dict[str, Any]) -> str:
    q = _first_present(record, ("query", "question", "input", "prompt", "instruction"))
    return _coerce_str(q)


def iter_sessions_jsonl(
    path: str,
    *,
    default_task: str = "default",
    model_id: str = "mistral-7b-instruct",
    prompt_version: str = "v1",
    max_examples: Optional[int] = None,
) -> Iterator[Tuple[str, str, MemoryQuery]]:
    """
    Iterate a sessions JSONL file and yield (example_id, task, MemoryQuery).

    Args:
      path: sessions JSONL file
      default_task: used when a record doesn't provide task
      model_id/prompt_version: written into MemoryQuery for version gating
      max_examples: optional cap
    """
    n = 0
    for idx, record in enumerate(_read_jsonl(path), start=1):
        ex_id = _example_id(record, idx)
        task = _coerce_str(record.get("task") or default_task) or "default"

        raw_query = _extract_query(record)
        if not raw_query.strip():
            raise ValueError(f"Record {ex_id} missing query field (query/question/input).")

        # Pull LongBench-like context if present
        dataset_context = _coerce_str(
            _first_present(record, ("dataset_context", "context_text", "passage", "document", "context"))
        )

        extra_ctx = record.get("context")
        if extra_ctx is not None and not isinstance(extra_ctx, dict):
            raise ValueError(f"Record {ex_id} field 'context' must be an object/dict if provided.")

        meta = record.get("meta")
        if meta is not None and not isinstance(meta, dict):
            raise ValueError(f"Record {ex_id} field 'meta' must be an object/dict if provided.")

        ctx: Dict[str, Any] = {
            "dataset_context": dataset_context,
            "session_meta": meta or {},
        }
        if extra_ctx:
            ctx["extra_context"] = extra_ctx

        mq = MemoryQuery(
            raw_query=raw_query,
            user_id=_coerce_str(record.get("user_id")) or None,
            session_id=_coerce_str(record.get("session_id")) or None,
            cohort_id=_coerce_str(record.get("cohort_id")) or None,
            task=task,
            context=ctx,
            prompt_version=prompt_version,
            model_id=model_id,
        )

        yield ex_id, task, mq

        n += 1
        if max_examples is not None and n >= max_examples:
            break


def write_sessions_jsonl(path: str, records: Iterator[Dict[str, Any]]) -> None:
    """
    Write a sessions JSONL file from an iterator of dict records.

    This is useful if you create session datasets from LongBench in scripts/.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
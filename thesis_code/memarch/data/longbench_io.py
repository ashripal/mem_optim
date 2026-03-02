# memarch/data/longbench_io.py
"""
LongBench dataset I/O utilities.

Goal (per your requirement):
Ensure the LLM can use *dataset-provided context* by putting it into MemoryQuery.context.

LongBench comes in a few slightly different JSON/JSONL shapes depending on where you got it
(Hugging Face datasets, local preprocessing scripts, etc.). This loader is designed to be:
- Flexible (handles common field names)
- Deterministic
- Minimal assumptions (no task-specific prompt parsing unless needed)

Output format:
  Iterator[(example_id, task, MemoryQuery)]

Where MemoryQuery contains:
  - raw_query: the question/instruction portion (best-effort extraction)
  - context: includes dataset context under a stable key:
      context["dataset_context"] = <the passage/document/context>
    and also stores the raw record metadata under context["longbench_meta"].

IMPORTANT:
- For fair measurement, keep dataset_context separate from personalization memory.
- Your generator prompt template should always include mq.context["dataset_context"] when present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from memarch.memory.schema import MemoryQuery


# -------------------------
# Utilities
# -------------------------

def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LongBench file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def _first_present(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _coerce_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


# -------------------------
# Extraction heuristics
# -------------------------

_QUERY_KEYS = ("question", "query", "instruction", "prompt", "input")
_CONTEXT_KEYS = ("context", "passage", "document", "background", "article", "source", "text")


def extract_query_and_context(record: Dict[str, Any]) -> Tuple[str, str]:
    """
    Best-effort extraction of:
      - raw_query: the "question/instruction" part
      - dataset_context: the passage/document/context part (if present)

    If only a single 'input' exists and no explicit context field exists,
    we treat:
      raw_query = input
      dataset_context = ""  (unknown / not separable)

    Rationale: don't invent parsing logic that might be wrong for certain tasks.
    If you want task-specific parsing, do it via a configurable adapter later.
    """
    # 1) If explicit context exists, take it.
    ctx = _first_present(record, _CONTEXT_KEYS)
    dataset_context = _coerce_str(ctx)

    # 2) Query: prefer explicit question fields; fall back to input/prompt.
    q = _first_present(record, _QUERY_KEYS)
    raw_query = _coerce_str(q)

    # If query is empty but input is present, use input
    if not raw_query:
        raw_query = _coerce_str(record.get("input", ""))

    # If context is missing but sometimes the record has "document" nested:
    if not dataset_context:
        nested = record.get("doc") or record.get("docs")
        if isinstance(nested, dict):
            dataset_context = _coerce_str(_first_present(nested, _CONTEXT_KEYS))
        elif isinstance(nested, list) and nested:
            # join a few docs conservatively
            parts = []
            for d in nested[:3]:
                if isinstance(d, dict):
                    parts.append(_coerce_str(_first_present(d, _CONTEXT_KEYS)))
                else:
                    parts.append(_coerce_str(d))
            dataset_context = "\n\n".join([p for p in parts if p])

    return raw_query, dataset_context


def extract_example_id(record: Dict[str, Any], fallback_index: int) -> str:
    """
    Determine a stable-ish example id.
    """
    rid = _first_present(record, ("id", "example_id", "qid", "uid"))
    if rid is None:
        return f"ex_{fallback_index:08d}"
    return _coerce_str(rid)


# -------------------------
# Public API
# -------------------------

def iter_longbench_jsonl(
    path: str,
    *,
    task: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    cohort_id: Optional[str] = None,
    model_id: str = "mistral-7b-instruct",
    prompt_version: str = "v1",
    include_raw_record_in_context: bool = False,
) -> Iterator[Tuple[str, str, MemoryQuery]]:
    """
    Iterate a LongBench JSONL file and yield (example_id, task, MemoryQuery).

    Args:
      path: JSONL file
      task: task/domain label (used for GLOBAL namespace scoping)
      user_id/session_id/cohort_id: optional personalization identifiers
      model_id/prompt_version: stored in MemoryQuery for version scoping
      include_raw_record_in_context: if True, stores the entire record in context (can be large)

    Yields:
      (example_id, task, MemoryQuery)
    """
    for idx, record in enumerate(_read_jsonl(path), start=1):
        example_id = extract_example_id(record, idx)
        raw_query, dataset_context = extract_query_and_context(record)

        # Minimal metadata for debugging/eval without bloating memory keys.
        # NOTE: Do NOT include raw_record in the signature unless you intend it to affect cache keys.
        longbench_meta = {
            "example_id": example_id,
            "task": task,
            # common fields that might exist:
            "dataset": record.get("dataset"),
            "subset": record.get("subset"),
        }

        ctx: Dict[str, Any] = {
            # This is the key your generator should look for in the prompt template:
            "dataset_context": dataset_context,
            "longbench_meta": longbench_meta,
        }
        if include_raw_record_in_context:
            # Warning: can be huge; keep False by default.
            ctx["longbench_raw"] = record

        mq = MemoryQuery(
            raw_query=raw_query,
            user_id=user_id,
            session_id=session_id,
            cohort_id=cohort_id,
            task=task,
            context=ctx,
            prompt_version=prompt_version,
            model_id=model_id,
        )

        yield example_id, task, mq


def load_longbench_examples(
    path: str,
    *,
    task: str,
    max_examples: Optional[int] = None,
    **kwargs: Any,
) -> Iterator[Tuple[str, str, MemoryQuery]]:
    """
    Convenience wrapper around iter_longbench_jsonl with optional max_examples cap.
    """
    n = 0
    for ex in iter_longbench_jsonl(path, task=task, **kwargs):
        yield ex
        n += 1
        if max_examples is not None and n >= max_examples:
            break
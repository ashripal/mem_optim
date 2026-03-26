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

Normalized internal record schema:
  - query_text         <- input | question | query | instruction | prompt
  - context_text       <- context | passage | document | background | article | source | text
  - reference_answers  <- answers | answer | output | target | reference
  - task               <- dataset | caller-provided task
  - source_id          <- _id | id | example_id | qid | uid | source_file

Output format:
  Iterator[(example_id, task, MemoryQuery)]

Where MemoryQuery contains:
  - raw_query: the normalized query_text
  - context: includes dataset context under a stable key:
      context["dataset_context"] = <the passage/document/context>
    and also stores normalized LongBench metadata under context["longbench_meta"].

IMPORTANT:
- For fair measurement, keep dataset_context separate from personalization memory.
- Your generator prompt template should always include mq.context["dataset_context"] when present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


def _coerce_str_list(x: Any) -> List[str]:
    """
    Normalize answer-like fields to a list[str].

    Handles:
      - str -> [str]
      - list/tuple -> [str, ...]
      - None -> []
      - everything else -> [str(x)]
    """
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (list, tuple)):
        out: List[str] = []
        for item in x:
            s = _coerce_str(item).strip()
            if s:
                out.append(s)
        return out
    s = _coerce_str(x).strip()
    return [s] if s else []


def _infer_task_from_path(path: str) -> str:
    """
    Fallback task inference from the JSONL filename.
    Example:
      /some/path/trec.jsonl -> trec
    """
    return Path(path).stem


# -------------------------
# Extraction heuristics
# -------------------------

# Prefer LongBench-standardized field names first, then common local variants.
_QUERY_KEYS = ("input", "question", "query", "instruction", "prompt")
_CONTEXT_KEYS = ("context", "passage", "document", "background", "article", "source", "text")
_ANSWER_KEYS = ("answers", "answer", "output", "target", "reference")
_ID_KEYS = ("_id", "id", "example_id", "qid", "uid")
_TASK_KEYS = ("dataset", "task", "subset")


def extract_query_and_context(record: Dict[str, Any]) -> Tuple[str, str]:
    """
    Best-effort extraction of:
      - query_text: the "question/instruction" part
      - context_text: the passage/document/context part (if present)

    If only a single 'input' exists and no explicit context field exists,
    we treat:
      query_text = input
      context_text = ""  (unknown / not separable)

    Rationale: don't invent parsing logic that might be wrong for certain tasks.
    If you want task-specific parsing, do it via a configurable adapter later.
    """
    # 1) If explicit context exists, take it.
    ctx = _first_present(record, _CONTEXT_KEYS)
    context_text = _coerce_str(ctx)

    # 2) Query: prefer standardized LongBench "input"; then fall back.
    q = _first_present(record, _QUERY_KEYS)
    query_text = _coerce_str(q)

    # 3) If context is missing but the record has nested doc/docs, extract conservatively.
    if not context_text:
        nested = record.get("doc") or record.get("docs")
        if isinstance(nested, dict):
            context_text = _coerce_str(_first_present(nested, _CONTEXT_KEYS))
        elif isinstance(nested, list) and nested:
            parts: List[str] = []
            for d in nested[:3]:
                if isinstance(d, dict):
                    parts.append(_coerce_str(_first_present(d, _CONTEXT_KEYS)))
                else:
                    parts.append(_coerce_str(d))
            context_text = "\n\n".join([p for p in parts if p])

    return query_text, context_text


def extract_reference_answers(record: Dict[str, Any]) -> List[str]:
    """
    Normalize answer/reference fields to a list[str].

    LongBench standardized schema uses:
      - answers: List[str]

    But local/preprocessed variants may use:
      - answer: str
      - output / target / reference
    """
    raw = _first_present(record, _ANSWER_KEYS)
    return _coerce_str_list(raw)


def extract_example_id(record: Dict[str, Any], fallback_index: int) -> str:
    """
    Determine a stable-ish example id.

    LongBench standardized schema uses:
      - _id

    Fallbacks:
      - id, example_id, qid, uid
      - generated ex_<index>
    """
    rid = _first_present(record, _ID_KEYS)
    if rid is None:
        return f"ex_{fallback_index:08d}"
    return _coerce_str(rid)


def normalize_longbench_record(
    record: Dict[str, Any],
    *,
    task: Optional[str] = None,
    source_file: Optional[str] = None,
    fallback_index: int = 0,
) -> Dict[str, Any]:
    """
    Convert a raw LongBench/local record into one normalized internal schema.

    Returned keys:
      - example_id
      - query_text
      - context_text
      - reference_answers
      - task
      - source_id
      - source_file
      - dataset
      - language
      - length
      - all_classes
      - subset
      - raw_record
    """
    query_text, context_text = extract_query_and_context(record)
    reference_answers = extract_reference_answers(record)
    example_id = extract_example_id(record, fallback_index)

    dataset_name = _coerce_str(_first_present(record, ("dataset",)))
    task_name = _coerce_str(task or dataset_name or _infer_task_from_path(source_file or "")).strip()
    if not task_name:
        task_name = "unknown"

    source_file_str = _coerce_str(source_file)
    source_id = _coerce_str(
        _first_present(record, ("_id", "id", "example_id", "qid", "uid", "source_file"))
    )
    if not source_id:
        # Fall back to the normalized example id so every record has one stable identifier.
        source_id = example_id

    normalized: Dict[str, Any] = {
        "example_id": example_id,
        "query_text": query_text,
        "context_text": context_text,
        "reference_answers": reference_answers,
        "task": task_name,
        "source_id": source_id,
        "source_file": source_file_str,
        "dataset": dataset_name or task_name,
        "language": _coerce_str(record.get("language")),
        "length": record.get("length"),
        "all_classes": record.get("all_classes"),
        "subset": _coerce_str(record.get("subset")),
        "raw_record": record,
    }
    return normalized


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
        norm = normalize_longbench_record(
            record,
            task=task,
            source_file=path,
            fallback_index=idx,
        )

        example_id = norm["example_id"]
        task_name = norm["task"]

        # Minimal normalized metadata for debugging/eval without bloating memory keys.
        longbench_meta: Dict[str, Any] = {
            "example_id": example_id,
            "task": task_name,
            "dataset": norm["dataset"],
            "subset": norm["subset"],
            "source_id": norm["source_id"],
            "source_file": norm["source_file"],
            "language": norm["language"],
            "length": norm["length"],
            "all_classes": norm["all_classes"],
            "reference_answers": norm["reference_answers"],
            "query_text": norm["query_text"],
            "context_text": norm["context_text"],
        }

        ctx: Dict[str, Any] = {
            # This is the key your generator should look for in the prompt template.
            "dataset_context": norm["context_text"],
            "longbench_meta": longbench_meta,
        }
        if include_raw_record_in_context:
            # Warning: can be huge; keep False by default.
            ctx["longbench_raw"] = norm["raw_record"]

        mq = MemoryQuery(
            raw_query=norm["query_text"],
            user_id=user_id,
            session_id=session_id,
            cohort_id=cohort_id,
            task=task_name,
            context=ctx,
            prompt_version=prompt_version,
            model_id=model_id,
        )

        yield example_id, task_name, mq


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
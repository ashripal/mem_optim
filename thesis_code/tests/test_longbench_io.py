# tests/test_longbench_io.py
#
# These tests validate that LongBench examples are correctly converted into MemoryQuery
# objects and, critically, that dataset-provided context is placed into:
#
#   mq.context["dataset_context"]
#
# This matters because the thesis claim depends on:
#   - Using dataset context for answering (LongBench passage/document)
#   - Keeping personalization memory separate from dataset context
#
# We intentionally test a few common JSONL shapes because LongBench data can come from
# different pipelines with different field names.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pytest

from memarch.data.longbench_io import iter_longbench_jsonl, extract_query_and_context
from memarch.memory.schema import MemoryQuery


def _write_jsonl(tmp_path: Path, rows: list[Dict[str, Any]], name: str = "lb.jsonl") -> str:
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(p)


def test_extract_query_and_context_prefers_explicit_fields():
    record = {
        "question": "What is the answer?",
        "context": "PASSAGE_ABC",
        "id": "ex1",
    }
    q, ctx = extract_query_and_context(record)
    assert q == "What is the answer?"
    assert ctx == "PASSAGE_ABC"


def test_extract_query_and_context_falls_back_to_input_when_needed():
    record = {
        "input": "Explain X.",
        "id": "ex1",
    }
    q, ctx = extract_query_and_context(record)
    assert q == "Explain X."
    assert ctx == ""  # no explicit context field present


def test_iter_longbench_jsonl_yields_memoryquery_with_dataset_context(tmp_path):
    """
    Core invariant: mq.context must contain dataset_context and longbench_meta.
    """
    rows = [
        {
            "id": "lb_001",
            "question": "Q1?",
            "context": "CONTEXT_1",
        },
        {
            "id": "lb_002",
            "question": "Q2?",
            "context": "CONTEXT_2",
        },
    ]
    path = _write_jsonl(tmp_path, rows)

    out = list(
        iter_longbench_jsonl(
            path,
            task="trec",
            user_id="u1",
            session_id="s1",
            cohort_id="c1",
            model_id="mistral-7b-instruct",
            prompt_version="v1",
        )
    )

    assert len(out) == 2

    ex_id, task, mq = out[0]
    assert ex_id == "lb_001"
    assert task == "trec"
    assert isinstance(mq, MemoryQuery)

    assert mq.raw_query == "Q1?"
    assert mq.task == "trec"
    assert mq.user_id == "u1"
    assert mq.session_id == "s1"
    assert mq.cohort_id == "c1"

    # Dataset context must be present
    assert "dataset_context" in mq.context
    assert mq.context["dataset_context"] == "CONTEXT_1"

    # Metadata must exist for debugging
    assert "longbench_meta" in mq.context
    assert mq.context["longbench_meta"]["example_id"] == "lb_001"
    assert mq.context["longbench_meta"]["task"] == "trec"


def test_iter_longbench_jsonl_handles_alt_field_names(tmp_path):
    """
    Ensure we can parse common variants:
      - query/passage
      - instruction/document
      - input/article
    """
    rows = [
        {"id": "a", "query": "Q?", "passage": "P1"},
        {"id": "b", "instruction": "Do X", "document": "DOC"},
        {"id": "c", "input": "Explain", "article": "ART"},
    ]
    path = _write_jsonl(tmp_path, rows)

    out = list(iter_longbench_jsonl(path, task="trec"))
    assert len(out) == 3

    assert out[0][2].raw_query == "Q?"
    assert out[0][2].context["dataset_context"] == "P1"

    assert out[1][2].raw_query == "Do X"
    assert out[1][2].context["dataset_context"] == "DOC"

    assert out[2][2].raw_query == "Explain"
    assert out[2][2].context["dataset_context"] == "ART"


def test_iter_longbench_jsonl_generates_fallback_example_ids(tmp_path):
    """
    If the record has no explicit id, we create a deterministic fallback: ex_00000001, etc.
    """
    rows = [
        {"question": "Q1", "context": "C1"},
        {"question": "Q2", "context": "C2"},
    ]
    path = _write_jsonl(tmp_path, rows)

    out = list(iter_longbench_jsonl(path, task="trec"))
    assert out[0][0] == "ex_00000001"
    assert out[1][0] == "ex_00000002"


def test_iter_longbench_jsonl_includes_dataset_context_key_even_if_missing(tmp_path):
    """
    Even if context is absent, we still include dataset_context with an empty string.
    This keeps prompt-building logic simple and avoids KeyError.
    """
    rows = [
        {"id": "x1", "question": "Q?", "context": None},
    ]
    path = _write_jsonl(tmp_path, rows)

    out = list(iter_longbench_jsonl(path, task="trec"))
    mq = out[0][2]

    assert "dataset_context" in mq.context
    assert mq.context["dataset_context"] == ""


def test_iter_longbench_jsonl_optional_raw_record_inclusion(tmp_path):
    """
    If include_raw_record_in_context=True, we store the entire record under context["longbench_raw"].
    This is off by default because it can be large.
    """
    rows = [{"id": "x1", "question": "Q?", "context": "C"}]
    path = _write_jsonl(tmp_path, rows)

    out = list(iter_longbench_jsonl(path, task="trec", include_raw_record_in_context=True))
    mq = out[0][2]

    assert "longbench_raw" in mq.context
    assert mq.context["longbench_raw"]["id"] == "x1"
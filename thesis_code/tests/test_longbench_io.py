# tests/test_longbench_io.py
"""
Unit tests for memarch/data/longbench_io.py.

These tests validate:
- LongBench-style field normalization
- Support for alternate/local field names
- Answer normalization to list[str]
- Example ID extraction and fallback behavior
- MemoryQuery construction with dataset context injected into mq.context
- Optional inclusion of raw records in context
- max_examples limiting
- invalid JSON handling

Run:
  pytest -q tests/test_longbench_io.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memarch.data.longbench_io import (
    extract_example_id,
    extract_query_and_context,
    extract_reference_answers,
    iter_longbench_jsonl,
    load_longbench_examples,
    normalize_longbench_record,
)
from memarch.memory.schema import MemoryQuery


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_extract_query_and_context_from_standard_longbench_fields():
    record = {
        "input": "What is the capital of France?",
        "context": "France is a country in Europe. Paris is its capital.",
    }

    query_text, context_text = extract_query_and_context(record)

    assert query_text == "What is the capital of France?"
    assert context_text == "France is a country in Europe. Paris is its capital."


def test_extract_query_and_context_from_local_variant_fields():
    record = {
        "question": "Who wrote Hamlet?",
        "article": "William Shakespeare wrote Hamlet.",
    }

    query_text, context_text = extract_query_and_context(record)

    assert query_text == "Who wrote Hamlet?"
    assert context_text == "William Shakespeare wrote Hamlet."


def test_extract_query_and_context_uses_nested_docs_when_context_missing():
    record = {
        "input": "Summarize the documents.",
        "docs": [
            {"document": "Doc one."},
            {"text": "Doc two."},
            "Doc three.",
            {"context": "Doc four should be ignored because only first three are used."},
        ],
    }

    query_text, context_text = extract_query_and_context(record)

    assert query_text == "Summarize the documents."
    assert context_text == "Doc one.\n\nDoc two.\n\nDoc three."


def test_extract_reference_answers_from_string():
    record = {"answer": "Paris"}

    answers = extract_reference_answers(record)

    assert answers == ["Paris"]


def test_extract_reference_answers_from_list():
    record = {"answers": ["Paris", "The capital is Paris"]}

    answers = extract_reference_answers(record)

    assert answers == ["Paris", "The capital is Paris"]


def test_extract_reference_answers_handles_none_and_non_string_values():
    assert extract_reference_answers({"answers": None}) == []
    assert extract_reference_answers({"answer": 42}) == ["42"]
    assert extract_reference_answers({"answers": ["Paris", None, 7, ""]}) == ["Paris", "7"]


def test_extract_example_id_prefers_standardized_id_fields():
    assert extract_example_id({"_id": "abc123"}, fallback_index=1) == "abc123"
    assert extract_example_id({"id": "legacy-id"}, fallback_index=1) == "legacy-id"
    assert extract_example_id({"example_id": 7}, fallback_index=1) == "7"
    assert extract_example_id({"qid": "q-9"}, fallback_index=1) == "q-9"
    assert extract_example_id({"uid": "u-1"}, fallback_index=1) == "u-1"


def test_extract_example_id_falls_back_to_generated_id():
    ex_id = extract_example_id({}, fallback_index=12)
    assert ex_id == "ex_00000012"


def test_normalize_longbench_record_with_standard_fields(tmp_path: Path):
    source_file = tmp_path / "trec.jsonl"

    record = {
        "_id": "rec-1",
        "input": "What is the capital of France?",
        "context": "France is a country in Europe. Paris is its capital.",
        "answers": ["Paris"],
        "dataset": "trec",
        "language": "en",
        "length": 1234,
        "all_classes": ["LOC", "HUM", "NUM"],
    }

    norm = normalize_longbench_record(
        record,
        task=None,
        source_file=str(source_file),
        fallback_index=1,
    )

    assert norm["example_id"] == "rec-1"
    assert norm["query_text"] == "What is the capital of France?"
    assert norm["context_text"] == "France is a country in Europe. Paris is its capital."
    assert norm["reference_answers"] == ["Paris"]
    assert norm["task"] == "trec"
    assert norm["source_id"] == "rec-1"
    assert norm["source_file"] == str(source_file)
    assert norm["dataset"] == "trec"
    assert norm["language"] == "en"
    assert norm["length"] == 1234
    assert norm["all_classes"] == ["LOC", "HUM", "NUM"]
    assert norm["raw_record"] == record


def test_normalize_longbench_record_with_local_variant_fields(tmp_path: Path):
    source_file = tmp_path / "hotpotqa.jsonl"

    record = {
        "question": "Who wrote Hamlet?",
        "document": "William Shakespeare wrote Hamlet.",
        "answer": "William Shakespeare",
        "id": "legacy-7",
    }

    norm = normalize_longbench_record(
        record,
        task="hotpotqa",
        source_file=str(source_file),
        fallback_index=3,
    )

    assert norm["example_id"] == "legacy-7"
    assert norm["query_text"] == "Who wrote Hamlet?"
    assert norm["context_text"] == "William Shakespeare wrote Hamlet."
    assert norm["reference_answers"] == ["William Shakespeare"]
    assert norm["task"] == "hotpotqa"
    assert norm["source_id"] == "legacy-7"
    assert norm["dataset"] == "hotpotqa"


def test_normalize_longbench_record_infers_task_from_source_file_when_missing(tmp_path: Path):
    source_file = tmp_path / "narrativeqa.jsonl"

    record = {
        "input": "What happened?",
        "context": "A long story here.",
        "answers": ["Something happened."],
    }

    norm = normalize_longbench_record(
        record,
        task=None,
        source_file=str(source_file),
        fallback_index=5,
    )

    assert norm["task"] == "narrativeqa"
    assert norm["dataset"] == "narrativeqa"
    assert norm["source_file"] == str(source_file)


def test_iter_longbench_jsonl_builds_memory_query_and_injects_dataset_context(tmp_path: Path):
    path = tmp_path / "trec.jsonl"
    records = [
        {
            "_id": "lb-1",
            "input": "What is the capital of France?",
            "context": "France is in Europe. Paris is the capital.",
            "answers": ["Paris"],
            "dataset": "trec",
            "subset": "test",
        }
    ]
    _write_jsonl(path, records)

    items = list(
        iter_longbench_jsonl(
            str(path),
            task="trec",
            user_id="user_a",
            session_id="session_a",
            cohort_id="cohort_a",
            model_id="fake-model",
            prompt_version="v2",
        )
    )

    assert len(items) == 1

    example_id, task, mq = items[0]
    assert example_id == "lb-1"
    assert task == "trec"
    assert isinstance(mq, MemoryQuery)

    assert mq.raw_query == "What is the capital of France?"
    assert mq.user_id == "user_a"
    assert mq.session_id == "session_a"
    assert mq.cohort_id == "cohort_a"
    assert mq.task == "trec"
    assert mq.model_id == "fake-model"
    assert mq.prompt_version == "v2"

    assert mq.context["dataset_context"] == "France is in Europe. Paris is the capital."
    assert "longbench_meta" in mq.context

    meta = mq.context["longbench_meta"]
    assert meta["example_id"] == "lb-1"
    assert meta["task"] == "trec"
    assert meta["dataset"] == "trec"
    assert meta["subset"] == "test"
    assert meta["source_id"] == "lb-1"
    assert meta["source_file"] == str(path)
    assert meta["reference_answers"] == ["Paris"]
    assert meta["query_text"] == "What is the capital of France?"
    assert meta["context_text"] == "France is in Europe. Paris is the capital."


def test_iter_longbench_jsonl_optionally_includes_raw_record(tmp_path: Path):
    path = tmp_path / "trec.jsonl"
    record = {
        "_id": "lb-2",
        "input": "Question?",
        "context": "Context.",
        "answers": ["Answer"],
        "dataset": "trec",
    }
    _write_jsonl(path, [record])

    items = list(
        iter_longbench_jsonl(
            str(path),
            task="trec",
            include_raw_record_in_context=True,
        )
    )

    _, _, mq = items[0]
    assert "longbench_raw" in mq.context
    assert mq.context["longbench_raw"] == record


def test_load_longbench_examples_respects_max_examples(tmp_path: Path):
    path = tmp_path / "trec.jsonl"
    _write_jsonl(
        path,
        [
            {"_id": "1", "input": "q1", "context": "c1", "answers": ["a1"]},
            {"_id": "2", "input": "q2", "context": "c2", "answers": ["a2"]},
            {"_id": "3", "input": "q3", "context": "c3", "answers": ["a3"]},
        ],
    )

    items = list(
        load_longbench_examples(
            str(path),
            task="trec",
            max_examples=2,
        )
    )

    assert len(items) == 2
    assert items[0][0] == "1"
    assert items[1][0] == "2"


def test_iter_longbench_jsonl_uses_task_argument_but_preserves_dataset_metadata(tmp_path: Path):
    path = tmp_path / "custom_task.jsonl"
    _write_jsonl(
        path,
        [
            {
                "_id": "x1",
                "input": "Who wrote Hamlet?",
                "context": "William Shakespeare wrote Hamlet.",
                "answers": ["William Shakespeare"],
                "dataset": "original_dataset_name",
            }
        ],
    )

    items = list(iter_longbench_jsonl(str(path), task="override_task"))

    example_id, task, mq = items[0]
    meta = mq.context["longbench_meta"]

    assert example_id == "x1"
    assert task == "override_task"
    assert mq.task == "override_task"
    assert meta["task"] == "override_task"
    assert meta["dataset"] == "original_dataset_name"


def test_iter_longbench_jsonl_generates_fallback_ids_when_missing(tmp_path: Path):
    path = tmp_path / "trec.jsonl"
    _write_jsonl(
        path,
        [
            {"input": "q1", "context": "c1", "answers": ["a1"]},
            {"input": "q2", "context": "c2", "answers": ["a2"]},
        ],
    )

    items = list(iter_longbench_jsonl(str(path), task="trec"))

    assert items[0][0] == "ex_00000001"
    assert items[1][0] == "ex_00000002"

    assert items[0][2].context["longbench_meta"]["source_id"] == "ex_00000001"
    assert items[1][2].context["longbench_meta"]["source_id"] == "ex_00000002"


def test_invalid_json_line_raises_value_error(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input": "ok", "context": "ctx"}\n{not valid json}\n', encoding="utf-8")

    it = iter_longbench_jsonl(str(path), task="trec")

    first = next(it)
    assert first[2].raw_query == "ok"

    with pytest.raises(ValueError):
        next(it)


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        list(iter_longbench_jsonl("does_not_exist.jsonl", task="trec"))
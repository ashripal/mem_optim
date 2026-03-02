# tests/test_tier2_disk.py
"""
Unit tests for Tier 2 (Disk) loader.

These tests create small temporary JSONL task files and validate:
- file discovery
- task_glob filtering
- streaming JSONL records
- max_examples global cap
- injected metadata fields (task, example_id, source_file)
- invalid JSON handling

Run:
  pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline.tiers.tier2_disk import DiskLoader


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_discovers_jsonl_files_and_lists_tasks(tmp_path: Path):
    # Create two task files
    _write_jsonl(tmp_path / "trec.jsonl", [{"context": "c1", "question": "q1", "answer": "a1"}])
    _write_jsonl(tmp_path / "narrativeqa.jsonl", [{"context": "c2", "question": "q2", "answer": "a2"}])

    loader = DiskLoader(repo_dir=str(tmp_path), task_glob="", max_examples=10)
    tasks = loader.list_tasks()

    assert tasks == ["narrativeqa", "trec"]  # sorted by filename
    ex = list(loader.iter_examples())
    assert len(ex) == 2


def test_task_glob_filters_files(tmp_path: Path):
    _write_jsonl(tmp_path / "trec.jsonl", [{"context": "c1", "question": "q1"}])
    _write_jsonl(tmp_path / "hotpotqa.jsonl", [{"context": "c2", "question": "q2"}])

    loader = DiskLoader(repo_dir=str(tmp_path), task_glob="trec", max_examples=10)
    assert loader.list_tasks() == ["trec"]

    ex = list(loader.iter_examples())
    assert len(ex) == 1
    assert ex[0]["task"] == "trec"


def test_injects_metadata_fields(tmp_path: Path):
    _write_jsonl(
        tmp_path / "trec.jsonl",
        [
            {"context": "c1", "question": "q1", "answer": "a1"},
            {"context": "c2", "question": "q2", "answer": "a2"},
        ],
    )

    loader = DiskLoader(repo_dir=str(tmp_path), task_glob="", max_examples=10)
    ex = list(loader.iter_examples())

    assert ex[0]["task"] == "trec"
    assert ex[0]["example_id"] == 0
    assert "source_file" in ex[0]
    assert ex[0]["source_file"].endswith("trec.jsonl")

    assert ex[1]["task"] == "trec"
    assert ex[1]["example_id"] == 1
    assert ex[1]["source_file"].endswith("trec.jsonl")


def test_max_examples_is_global_cap_across_tasks(tmp_path: Path):
    _write_jsonl(tmp_path / "a_task.jsonl", [{"x": 1}, {"x": 2}, {"x": 3}])
    _write_jsonl(tmp_path / "b_task.jsonl", [{"y": 1}, {"y": 2}, {"y": 3}])

    loader = DiskLoader(repo_dir=str(tmp_path), task_glob="", max_examples=4)
    ex = list(loader.iter_examples())

    assert len(ex) == 4
    # Deterministic order: a_task.jsonl then b_task.jsonl (sorted filenames)
    assert ex[0]["task"] == "a_task"
    assert ex[1]["task"] == "a_task"
    assert ex[2]["task"] == "a_task"
    assert ex[3]["task"] == "b_task"

    assert ex[0]["example_id"] == 0
    assert ex[3]["example_id"] == 3


def test_raises_if_repo_dir_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        DiskLoader(repo_dir=str(missing), task_glob="", max_examples=1)


def test_raises_if_no_jsonl_files(tmp_path: Path):
    # Empty directory
    with pytest.raises(FileNotFoundError):
        DiskLoader(repo_dir=str(tmp_path), task_glob="", max_examples=1)


def test_raises_if_task_glob_matches_nothing(tmp_path: Path):
    _write_jsonl(tmp_path / "trec.jsonl", [{"x": 1}])
    with pytest.raises(FileNotFoundError):
        DiskLoader(repo_dir=str(tmp_path), task_glob="nope", max_examples=1)


def test_invalid_json_line_raises_value_error(tmp_path: Path):
    # Create a jsonl with an invalid line
    p = tmp_path / "trec.jsonl"
    p.write_text('{"ok": 1}\n{this is not json}\n', encoding="utf-8")

    loader = DiskLoader(repo_dir=str(tmp_path), task_glob="", max_examples=10)

    it = loader.iter_examples()
    first = next(it)
    assert first["task"] == "trec"

    with pytest.raises(ValueError):
        next(it)
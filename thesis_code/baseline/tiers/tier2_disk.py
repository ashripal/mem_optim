# tiers/tier2_disk.py
"""
Tier 2 (Disk): LongBench dataset discovery + loading.

Responsibilities:
- Locate task JSONL files on disk (Tier 2 repository path)
- Optionally filter tasks via substring match (task_glob)
- Stream examples from JSONL (do not load everything into RAM)
- Attach minimal metadata (task name, example_id, source file)

This module should NOT:
- Do model inference (Tier 0)
- Cache items in RAM (Tier 1)
- Write run logs (pipeline/logging.py)

Expected layout (common):
  <repo_dir>/
    <task1>.jsonl
    <task2>.jsonl
or
  <repo_dir>/data/
    <task>.jsonl

So: you should pass cfg.tier2_repo as the directory that *contains* the JSONL files.
(Your prepare script can output that directory.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class TaskFile:
    task: str
    path: Path


def _list_jsonl_files(repo_dir: Path) -> List[Path]:
    """
    Return all .jsonl files directly in repo_dir (non-recursive),
    sorted for determinism.
    """
    return sorted([p for p in repo_dir.glob("*.jsonl") if p.is_file()])


def _infer_task_name(path: Path) -> str:
    """
    Task name is typically the filename without extension.
    """
    return path.stem


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Stream JSONL records from a file, skipping empty lines.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {line_no}: {e}") from e


class DiskLoader:
    """
    Streams LongBench examples from disk.

    Each yielded example is a dict that includes:
      - task: str
      - example_id: int (sequential within the stream)
      - source_file: str (path to jsonl)
      - plus the original JSON fields (e.g., context/question/answer/etc.)
    """

    def __init__(self, repo_dir: str, task_glob: str = "", max_examples: int = 25):
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"Tier2 repo_dir not found: {self.repo_dir}")
        if not self.repo_dir.is_dir():
            raise NotADirectoryError(f"Tier2 repo_dir must be a directory: {self.repo_dir}")

        self.task_glob = (task_glob or "").strip()
        self.max_examples = int(max_examples) if max_examples is not None else 25

        # Discover files once for deterministic iteration
        files = _list_jsonl_files(self.repo_dir)
        if not files:
            raise FileNotFoundError(
                f"No .jsonl files found in: {self.repo_dir}\n"
                f"Tip: pass the directory that contains the extracted LongBench JSONLs."
            )

        task_files: List[TaskFile] = []
        for p in files:
            task = _infer_task_name(p)
            if self.task_glob and (self.task_glob not in task):
                continue
            task_files.append(TaskFile(task=task, path=p))

        if not task_files:
            raise FileNotFoundError(
                f"No task files matched task_glob='{self.task_glob}' in {self.repo_dir}"
            )

        self.task_files = task_files

    def list_tasks(self) -> List[str]:
        """
        Return the ordered list of task names that will be iterated.
        """
        return [tf.task for tf in self.task_files]

    def iter_examples(self) -> Iterator[Dict[str, Any]]:
        """
        Stream examples from selected task files, up to max_examples total.

        Note:
        - This baseline keeps selection simple: it streams in file order.
        - max_examples applies across *all* tasks combined (global cap),
          matching typical quick baseline runs.
        """
        n_yielded = 0
        global_id = 0

        for tf in self.task_files:
            for record in _iter_jsonl(tf.path):
                if self.max_examples is not None and n_yielded >= self.max_examples:
                    return

                example: Dict[str, Any] = dict(record)  # copy to avoid side effects
                example["task"] = tf.task
                example["example_id"] = global_id
                example["source_file"] = str(tf.path)

                yield example

                n_yielded += 1
                global_id += 1
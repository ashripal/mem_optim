from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class LongBenchExample:
    task_file: str
    example_index: int
    context: str
    question: str
    answers: Optional[List[str]]


def find_task_files(longbench_dir: Path, task_glob: str = "") -> List[Path]:
    """
    Finds LongBench JSONL task files in a directory.
    longbench_dir: path to folder containing *.jsonl (e.g., .../LongBench/data)
    task_glob: substring filter (e.g., "trec" -> trec.jsonl, trec_e.jsonl)
    """
    files = sorted(longbench_dir.glob("*.jsonl"))
    if task_glob:
        tg = task_glob.lower()
        files = [p for p in files if tg in p.name.lower()]
    return files


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_longbench_examples(
    longbench_dir: Path,
    task_glob: str = "",
    max_examples_per_file: Optional[int] = None,
) -> Iterator[LongBenchExample]:
    """
    Streams examples from LongBench task JSONL(s) without loading all into RAM.
    """
    task_files = find_task_files(longbench_dir, task_glob)
    if not task_files:
        raise FileNotFoundError(f"No *.jsonl found under {longbench_dir} matching '{task_glob}'")

    for task_path in task_files:
        count = 0
        for idx, ex in enumerate(iter_jsonl(task_path)):
            # LongBench commonly uses "context", "input", "answers"
            context = str(ex.get("context", ""))
            question = str(ex.get("input", ""))

            answers = ex.get("answers", None)
            if answers is not None:
                # normalize to list[str]
                if isinstance(answers, list):
                    answers = [str(a) for a in answers]
                else:
                    answers = [str(answers)]

            yield LongBenchExample(
                task_file=task_path.name,
                example_index=idx,
                context=context,
                question=question,
                answers=answers,
            )

            count += 1
            if max_examples_per_file is not None and count >= max_examples_per_file:
                break
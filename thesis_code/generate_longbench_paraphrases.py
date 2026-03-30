#!/usr/bin/env python3
"""
generate_longbench_paraphrases.py

Generate paraphrased LongBench question files using the OpenAI API.

This script creates family-aware paraphrase evaluation data for the current
thesis pipeline. It preserves the original LongBench JSONL row format and only
rewrites the `input` field (the question), while keeping `context`, `answers`,
and all other metadata intact.

Updated family-aware behavior:
- every source row gets a stable family_id
- optionally includes the original question as the family anchor (A0)
- each paraphrase becomes a family variant (A1, A2, ...)
- output rows carry metadata needed by workload builders:
    * family_id
    * variant_id
    * variant_kind
    * family_size
    * base_input
    * original_row_id

Typical flow:
1. Resolve the LongBench repo directory.
2. Ensure the LongBench JSONL files are extracted from data.zip if needed.
3. Read each JSONL task file.
4. For each selected row, ask an OpenAI model for N paraphrases.
5. Write output JSONL files in the same row-oriented format as the inputs.

Recommended default output mode is `expand_rows`, where each original question
becomes multiple rows with paraphrased `input` values.

Recommended for family-aware benchmarking:
- use --include_original so each family contains A0 + paraphrases
- use --mode expand_rows
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "The 'openai' package is required. Install it with: pip install openai"
    ) from exc


# -----------------------------
# Constants and small utilities
# -----------------------------
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_API_ENV = "OPENAI_API_KEY"
DEFAULT_REQUESTS_PER_MINUTE = 60
DEFAULT_TEMPERATURE = None
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_REQUEST_TIMEOUT_S = 60.0
DEFAULT_CONNECT_TIMEOUT_S = 20.0
DEFAULT_RETRY_BACKOFF_BASE_S = 2.0
DEFAULT_PROGRESS_EVERY = 1


@dataclass
class RunStats:
    """Collect simple run statistics for logging and reproducibility."""

    files_seen: int = 0
    files_written: int = 0
    rows_seen: int = 0
    rows_written: int = 0
    rows_skipped: int = 0
    rows_failed: int = 0
    api_calls: int = 0
    api_retries: int = 0
    paraphrases_written: int = 0
    original_rows_written: int = 0
    elapsed_s: float = 0.0


@dataclass
class RateLimiter:
    """
    Simple per-process rate limiter.

    This keeps the script from sending requests too quickly when many rows are
    processed in sequence.
    """

    requests_per_minute: int
    _last_request_ts: float = 0.0

    def wait(self) -> None:
        if self.requests_per_minute <= 0:
            return
        min_interval = 60.0 / float(self.requests_per_minute)
        now = time.time()
        delta = now - self._last_request_ts
        if delta < min_interval:
            time.sleep(min_interval - delta)
        self._last_request_ts = time.time()


def log_info(message: str) -> None:
    print(message, flush=True)


def log_warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# -----------------------------
# LongBench file discovery
# -----------------------------

def extract_longbench_if_needed(repo_dir: Path, out_dir: Optional[Path] = None) -> Path:
    """
    Reuse the current LongBench preparation assumption:
    - dataset repo contains `data.zip`
    - extraction produces `data/*.jsonl`

    Returns the directory containing the extracted JSONL files.
    """
    repo_dir = repo_dir.resolve()
    out_dir = (out_dir or repo_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_jsonl_dir = out_dir / "data"
    if candidate_jsonl_dir.exists() and any(candidate_jsonl_dir.glob("*.jsonl")):
        return candidate_jsonl_dir

    zip_path = repo_dir / "data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not find extracted JSONL files or {zip_path}. "
            "Run your LongBench download step first."
        )

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    if candidate_jsonl_dir.exists() and any(candidate_jsonl_dir.glob("*.jsonl")):
        return candidate_jsonl_dir

    if any(out_dir.glob("*.jsonl")):
        return out_dir

    raise FileNotFoundError(f"No JSONL files found after extracting {zip_path}.")


def find_task_files(jsonl_dir: Path, task_glob: str = "") -> List[Path]:
    """Find LongBench task files, optionally filtering by substring."""
    files = sorted(jsonl_dir.glob("*.jsonl"))
    if task_glob:
        needle = task_glob.lower()
        files = [path for path in files if needle in path.name.lower()]
    return files


# -----------------------------
# JSONL helpers
# -----------------------------

def iter_jsonl_rows(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield `(line_number, parsed_row)` pairs from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSON object as a single JSONL line."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# Prompting and validation
# -----------------------------

def build_paraphrase_messages(
    *,
    task_name: str,
    original_question: str,
    num_paraphrases: int,
) -> List[Dict[str, str]]:
    """
    Build a structured prompt for paraphrase generation.

    We keep the instruction strict because this dataset is for evaluation.
    The model should rewrite the question without changing meaning.
    """
    developer_text = (
        "You paraphrase benchmark questions for evaluation. "
        "Preserve the original meaning exactly. Do not answer the question. "
        "Do not add or remove facts. Do not change named entities, dates, "
        "numbers, labels, or constraints except by equivalent rewording. "
        "Keep the output in the same language as the input. "
        "Return valid JSON only that matches the requested schema."
    )
    user_text = (
        f"Task file: {task_name}\n"
        f"Original question: {original_question}\n"
        f"Requested paraphrase count: {num_paraphrases}\n\n"
        "Return a JSON object with exactly one key named 'paraphrases'. "
        "That value must be a JSON array of distinct paraphrased questions."
    )
    return [
        {"role": "developer", "content": developer_text},
        {"role": "user", "content": user_text},
    ]


def normalize_text(text: str) -> str:
    """Lightweight normalization for duplicate and equality checks."""
    return " ".join((text or "").strip().lower().split())


def looks_like_question(text: str) -> bool:
    """
    Best-effort guard to reject obviously malformed outputs.

    We allow both punctuated and unpunctuated questions because datasets vary.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    return True


def validate_paraphrases(
    *,
    original_question: str,
    paraphrases: Sequence[str],
    expected_count: int,
) -> List[str]:
    """
    Validate and clean the model output.

    Rules:
    - non-empty strings only
    - remove duplicates
    - remove paraphrases identical to the original after normalization
    - keep the order stable
    """
    cleaned: List[str] = []
    seen: set[str] = set()
    original_norm = normalize_text(original_question)

    for item in paraphrases:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        if not looks_like_question(text):
            continue
        norm = normalize_text(text)
        if norm == original_norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(text)

    if len(cleaned) > expected_count:
        cleaned = cleaned[:expected_count]
    return cleaned


# -----------------------------
# OpenAI API call
# -----------------------------

def call_openai_for_paraphrases(
    *,
    client: OpenAI,
    model: str,
    task_name: str,
    original_question: str,
    num_paraphrases: int,
    temperature: Optional[float],
    max_output_tokens: int,
    request_timeout_s: float,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Request paraphrases from the OpenAI API.

    This uses structured JSON output via `response_format`. We still validate the
    result locally because model outputs should never be trusted blindly.
    """
    messages = build_paraphrase_messages(
        task_name=task_name,
        original_question=original_question,
        num_paraphrases=num_paraphrases,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_output_tokens,
        timeout=request_timeout_s,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "longbench_paraphrases",
                "schema": {
                    "type": "object",
                    "properties": {
                        "paraphrases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": num_paraphrases,
                            "maxItems": num_paraphrases,
                        }
                    },
                    "required": ["paraphrases"],
                    "additionalProperties": False,
                },
            },
        },
    )

    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    paraphrases = payload.get("paraphrases", [])

    usage: Dict[str, Any] = {}
    if getattr(response, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    return paraphrases, usage


# -----------------------------
# Family metadata helpers
# -----------------------------

def _family_id(file_name: str, original_line_no: int) -> str:
    return f"{file_name}:{original_line_no}"


def _variant_id(file_name: str, original_line_no: int, variant_index: int) -> str:
    return f"{file_name}:{original_line_no}:v{variant_index}"


def _base_row_metadata(
    *,
    row: Dict[str, Any],
    file_name: str,
    original_line_no: int,
    model: str,
    prompt_version: str,
    family_size: int,
) -> Dict[str, Any]:
    """
    Metadata shared by original and paraphrase variants.
    """
    base_input = row.get("input", "")
    family_id = _family_id(file_name, original_line_no)

    return {
        "family_id": family_id,
        "family_size": family_size,
        "base_input": base_input,
        "paraphrase_of": base_input,
        "paraphrase_model": model,
        "prompt_version": prompt_version,
        "original_row_id": family_id,
    }


# -----------------------------
# Row transformation
# -----------------------------

def make_output_rows(
    *,
    row: Dict[str, Any],
    original_line_no: int,
    file_name: str,
    paraphrases: Sequence[str],
    model: str,
    prompt_version: str,
    mode: str,
    include_original: bool,
) -> List[Dict[str, Any]]:
    """
    Convert one original row and generated paraphrases into output rows.

    Modes:
    - expand_rows:
        one output row per paraphrase, plus optional original A0
    - replace_input:
        one output row using only the first paraphrase, plus optional original A0

    Family-aware behavior:
    - original row (if included) becomes variant_kind="original", paraphrase_index=-1
    - paraphrases become variant_kind="semantic", paraphrase_index=0..N-1
    """
    rows: List[Dict[str, Any]] = []
    family_size = len(paraphrases) + (1 if include_original else 0)
    shared_meta = _base_row_metadata(
        row=row,
        file_name=file_name,
        original_line_no=original_line_no,
        model=model,
        prompt_version=prompt_version,
        family_size=family_size,
    )
    base_input = row.get("input", "")
    file_line_family_id = shared_meta["family_id"]

    if include_original:
        original_row = dict(row)
        original_row.update(shared_meta)
        original_row["variant_id"] = _variant_id(file_name, original_line_no, 0)
        original_row["variant_kind"] = "original"
        original_row["paraphrase_index"] = -1
        original_row["input"] = base_input
        rows.append(original_row)

    if mode == "replace_input":
        if not paraphrases:
            return rows
        new_row = dict(row)
        new_row.update(shared_meta)
        new_row["variant_id"] = _variant_id(
            file_name,
            original_line_no,
            1 if include_original else 0,
        )
        new_row["variant_kind"] = "semantic"
        new_row["paraphrase_index"] = 0
        new_row["paraphrase_type"] = "llm_semantic_rewrite"
        new_row["input"] = paraphrases[0]
        rows.append(new_row)
        return rows

    start_variant_index = 1 if include_original else 0
    for idx, question in enumerate(paraphrases):
        new_row = dict(row)
        new_row.update(shared_meta)
        new_row["variant_id"] = _variant_id(
            file_name,
            original_line_no,
            start_variant_index + idx,
        )
        new_row["variant_kind"] = "semantic"
        new_row["paraphrase_index"] = idx
        new_row["paraphrase_type"] = "llm_semantic_rewrite"
        new_row["input"] = question
        rows.append(new_row)

    return rows


# -----------------------------
# Main file processor
# -----------------------------

def process_file(
    *,
    input_path: Path,
    output_path: Path,
    client: OpenAI,
    limiter: RateLimiter,
    model: str,
    paraphrases_per_question: int,
    max_examples: int,
    temperature: Optional[float],
    max_output_tokens: int,
    mode: str,
    prompt_version: str,
    retries: int,
    include_original: bool,
    dry_run: bool,
    stats: RunStats,
    request_timeout_s: float,
    retry_backoff_base_s: float,
    progress_every: int,
) -> None:
    """Process one LongBench JSONL file and write its paraphrased output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    processed = 0
    progress_every = max(1, int(progress_every))

    for line_no, row in iter_jsonl_rows(input_path):
        stats.rows_seen += 1

        if max_examples > 0 and processed >= max_examples:
            break

        original_question = row.get("input")
        if not isinstance(original_question, str) or not original_question.strip():
            stats.rows_skipped += 1
            continue

        if dry_run:
            processed += 1
            if processed % progress_every == 0:
                log_info(
                    f"[dry-run] {input_path.name} row={processed} line={line_no}"
                )
            continue

        log_info(
            f"[info] {input_path.name} row={processed + 1}"
            f"{f'/{max_examples}' if max_examples > 0 else ''}"
            f" line={line_no} requesting paraphrases..."
        )

        paraphrases: List[str] = []
        usage_info: Dict[str, Any] = {}
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                limiter.wait()
                stats.api_calls += 1
                if attempt > 0:
                    stats.api_retries += 1

                raw_paraphrases, usage_info = call_openai_for_paraphrases(
                    client=client,
                    model=model,
                    task_name=input_path.stem,
                    original_question=original_question,
                    num_paraphrases=paraphrases_per_question,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    request_timeout_s=request_timeout_s,
                )

                paraphrases = validate_paraphrases(
                    original_question=original_question,
                    paraphrases=raw_paraphrases,
                    expected_count=paraphrases_per_question,
                )

                if len(paraphrases) >= 1:
                    break

                if attempt < retries:
                    backoff_s = retry_backoff_base_s * (2 ** attempt)
                    log_warn(
                        f"[warn] {input_path.name}:{line_no} returned no usable paraphrases "
                        f"on attempt {attempt + 1}/{retries + 1}; retrying in {backoff_s:.1f}s"
                    )
                    time.sleep(backoff_s)

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # pragma: no cover - API/network dependent
                last_error = exc
                if attempt < retries:
                    backoff_s = retry_backoff_base_s * (2 ** attempt)
                    log_warn(
                        f"[warn] Failed {input_path.name}:{line_no} "
                        f"attempt {attempt + 1}/{retries + 1}: {exc} "
                        f"(retrying in {backoff_s:.1f}s)"
                    )
                    time.sleep(backoff_s)

        if not paraphrases:
            stats.rows_failed += 1
            if last_error is not None:
                log_warn(
                    f"[warn] Failed to paraphrase {input_path.name}:{line_no}: {last_error}"
                )
            else:
                log_warn(
                    f"[warn] Failed to paraphrase {input_path.name}:{line_no}: "
                    "model returned no usable paraphrases"
                )
            processed += 1
            continue

        out_rows = make_output_rows(
            row=row,
            original_line_no=line_no,
            file_name=input_path.name,
            paraphrases=paraphrases,
            model=model,
            prompt_version=prompt_version,
            mode=mode,
            include_original=include_original,
        )

        for out_row in out_rows:
            if usage_info:
                out_row["paraphrase_usage"] = usage_info
            write_jsonl_row(output_path, out_row)
            stats.rows_written += 1
            if out_row.get("variant_kind") == "original":
                stats.original_rows_written += 1
            elif out_row.get("variant_kind") == "semantic":
                stats.paraphrases_written += 1

        processed += 1
        if processed % progress_every == 0:
            log_info(
                f"[info] {input_path.name} completed row={processed}"
                f"{f'/{max_examples}' if max_examples > 0 else ''}; "
                f"usable_paraphrases={len(paraphrases)} total_rows_written={stats.rows_written}"
            )


# -----------------------------
# Summary writing
# -----------------------------

def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    """Write a human- and machine-readable summary JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paraphrased LongBench question files using the OpenAI API."
    )
    parser.add_argument(
        "--repo_dir",
        default="tier2_disk/longbench_repo",
        help="Path to the downloaded LongBench dataset repo.",
    )
    parser.add_argument(
        "--data_dir",
        default="",
        help="Optional direct path to extracted JSONL files. If omitted, resolve from repo_dir.",
    )
    parser.add_argument(
        "--out_dir",
        default="tier2_disk/longbench_repo/paraphrased_data",
        help="Output directory for paraphrased JSONL files.",
    )
    parser.add_argument(
        "--task_glob",
        default="",
        help="Optional substring filter, for example 'trec' or 'hotpotqa'.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="Maximum number of source questions to process per input file. Use <=0 for all rows.",
    )
    parser.add_argument(
        "--paraphrases_per_question",
        type=int,
        default=2,
        help="Number of paraphrases to request for each original question.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--api_key_env",
        default=DEFAULT_API_ENV,
        help="Environment variable that stores the OpenAI API key.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum completion tokens for each paraphrase generation call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for paraphrase generation.",
    )
    parser.add_argument(
        "--requests_per_minute",
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Best-effort client-side rate limit.",
    )
    parser.add_argument(
        "--mode",
        choices=["expand_rows", "replace_input"],
        default="expand_rows",
        help="Whether to expand each source row into multiple rows or replace input with one paraphrase.",
    )
    parser.add_argument(
        "--include_original",
        action="store_true",
        help="Include the original question as variant_kind='original' in each family.",
    )
    parser.add_argument(
        "--prompt_version",
        default=DEFAULT_PROMPT_VERSION,
        help="String tag stored in output metadata for reproducibility.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries after the first failed paraphrase attempt.",
    )
    parser.add_argument(
        "--request_timeout_s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Per-request timeout in seconds for the OpenAI API call.",
    )
    parser.add_argument(
        "--connect_timeout_s",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_S,
        help="Client connection timeout in seconds.",
    )
    parser.add_argument(
        "--retry_backoff_base_s",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_BASE_S,
        help="Base seconds for exponential retry backoff.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print progress every N processed source rows.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover files and count rows without making OpenAI API calls or writing output.",
    )
    return parser


# -----------------------------
# Entrypoint
# -----------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    start_ts = time.time()

    repo_dir = Path(args.repo_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if args.data_dir:
        jsonl_dir = Path(args.data_dir).resolve()
        if not jsonl_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {jsonl_dir}")
    else:
        jsonl_dir = extract_longbench_if_needed(repo_dir)

    task_files = find_task_files(jsonl_dir, args.task_glob)
    if not task_files:
        raise FileNotFoundError(
            f"No JSONL task files found in {jsonl_dir} matching task_glob={args.task_glob!r}."
        )

    log_info(f"[info] JSONL dir: {jsonl_dir}")
    log_info(f"[info] Output dir: {out_dir}")
    log_info(f"[info] Files selected: {len(task_files)}")

    if args.dry_run:
        for path in task_files:
            log_info(f"[dry-run] {path.name}")
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set the environment variable {args.api_key_env}."
        )

    client = OpenAI(
        api_key=api_key,
        timeout=args.connect_timeout_s,
    )
    limiter = RateLimiter(requests_per_minute=args.requests_per_minute)
    stats = RunStats()

    for input_path in task_files:
        stats.files_seen += 1
        output_path = out_dir / input_path.name
        log_info(f"[info] Processing {input_path.name} -> {output_path}")
        process_file(
            input_path=input_path,
            output_path=output_path,
            client=client,
            limiter=limiter,
            model=args.model,
            paraphrases_per_question=args.paraphrases_per_question,
            max_examples=args.max_examples,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            mode=args.mode,
            prompt_version=args.prompt_version,
            retries=args.retries,
            include_original=args.include_original,
            dry_run=args.dry_run,
            stats=stats,
            request_timeout_s=args.request_timeout_s,
            retry_backoff_base_s=args.retry_backoff_base_s,
            progress_every=args.progress_every,
        )
        stats.files_written += 1

    stats.elapsed_s = time.time() - start_ts

    summary = {
        "script": "generate_longbench_paraphrases.py",
        "repo_dir": str(repo_dir),
        "jsonl_dir": str(jsonl_dir),
        "out_dir": str(out_dir),
        "task_glob": args.task_glob,
        "max_examples": args.max_examples,
        "paraphrases_per_question": args.paraphrases_per_question,
        "mode": args.mode,
        "include_original": args.include_original,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "requests_per_minute": args.requests_per_minute,
        "prompt_version": args.prompt_version,
        "request_timeout_s": args.request_timeout_s,
        "connect_timeout_s": args.connect_timeout_s,
        "retry_backoff_base_s": args.retry_backoff_base_s,
        "progress_every": args.progress_every,
        "stats": {
            "files_seen": stats.files_seen,
            "files_written": stats.files_written,
            "rows_seen": stats.rows_seen,
            "rows_written": stats.rows_written,
            "rows_skipped": stats.rows_skipped,
            "rows_failed": stats.rows_failed,
            "api_calls": stats.api_calls,
            "api_retries": stats.api_retries,
            "paraphrases_written": stats.paraphrases_written,
            "original_rows_written": stats.original_rows_written,
            "elapsed_s": stats.elapsed_s,
        },
    }
    write_summary(out_dir / "paraphrase_run_summary.json", summary)
    log_info(f"[done] Wrote paraphrased data to {out_dir}")
    log_info(f"[done] Summary: {out_dir / 'paraphrase_run_summary.json'}")


if __name__ == "__main__":
    main()
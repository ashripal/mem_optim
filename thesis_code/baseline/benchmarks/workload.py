# baseline/benchmarks/workload.py
"""
Workload construction utilities for baseline LongBench benchmarking.

This module sits *above* Tier 2 disk loading.

Responsibilities:
- Load the base LongBench example set selected by BenchmarkConfig
- Transform that base set into a benchmark workload sequence
- Make replay / cache-pressure behavior explicit
- Attach benchmark-specific metadata to each workload item

This module does NOT:
- Run model inference
- Manage caches
- Write logs
- Summarize results

Why this module exists:
A plain pass over LongBench mostly measures cold misses. To study cache hits,
misses, and evictions, we need controlled replay behavior and explicit workload
ordering.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from baseline.benchmarks.configs import BenchmarkConfig, WorkloadConfig
from baseline.tiers.tier2_disk import DiskLoader


Example = Dict[str, Any]


def load_base_examples(cfg: BenchmarkConfig) -> List[Example]:
    """
    Load the base LongBench example set using the existing Tier 2 DiskLoader.

    Notes:
    - task_glob="" means "all task files"
    - max_examples applies to the base selection only
    - returned examples are materialized into a list so workload shaping
      can replay / reorder them deterministically
    """
    cfg.validate()

    disk = DiskLoader(
        repo_dir=cfg.tier2_repo,
        task_glob=cfg.task_glob,
        max_examples=cfg.max_examples,
    )
    return [dict(ex) for ex in disk.iter_examples()]


def build_workload_sequence(
    examples: List[Example],
    workload_cfg: WorkloadConfig,
) -> List[Example]:
    """
    Build the benchmark sequence from the base example set.

    Supported modes:
    - cold:
        Each selected example appears exactly once.
    - replay_once:
        Sequence is repeated one additional time.
        Example: [A, B, C, A, B, C]
    - replay_k:
        Sequence is repeated replay_k total times.
        Example with replay_k=3: [A, B, C, A, B, C, A, B, C]
    - cache_pressure:
        The first pass is preserved, then examples are revisited in a
        delayed/interleaved pattern that is more likely to cause LRU eviction
        when cache capacity is small relative to working set size.

    Shuffle behavior:
    - Applied to the base example list before mode expansion.
    - Controlled by workload_cfg.shuffle and workload_cfg.seed.
    """
    workload_cfg.validate()

    base = [dict(ex) for ex in examples]
    if not base:
        return []

    if workload_cfg.shuffle:
        rng = random.Random(workload_cfg.seed)
        rng.shuffle(base)

    if workload_cfg.mode == "cold":
        return _build_cold(base)

    if workload_cfg.mode == "replay_once":
        return _build_replay_once(base)

    if workload_cfg.mode == "replay_k":
        return _build_replay_k(base, replay_k=workload_cfg.replay_k)

    if workload_cfg.mode == "cache_pressure":
        return _build_cache_pressure(base)

    if workload_cfg.mode == "mixed_reuse":
        return _build_mixed_reuse(
            base,
            total_requests=workload_cfg.total_requests,
            repeat_fraction=workload_cfg.repeat_fraction,
            seed=workload_cfg.seed,
        )

    raise ValueError(f"Unsupported workload mode: {workload_cfg.mode!r}")


def annotate_workload_positions(
    sequence: List[Example],
    workload_cfg: WorkloadConfig,
) -> List[Example]:
    """
    Attach benchmark-specific metadata to each workload item.

    Added fields:
    - workload_mode: benchmark mode name
    - workload_pos: absolute position in the benchmark stream
    - workload_repeat_index: 0 for first appearance of a base example, then 1, 2, ...
    - workload_pass: same as repeat index for now; kept for readability in logs
    - base_example_id: original example_id from DiskLoader
    - base_task: original task from DiskLoader
    - base_source_file: original source_file from DiskLoader

    Important:
    - This keeps the original example_id unchanged for compatibility
    - The benchmark metadata is additive
    """
    annotated: List[Example] = []
    seen_counts: Dict[Any, int] = {}

    for pos, ex in enumerate(sequence):
        item = dict(ex)

        base_example_id = item.get("example_id")
        repeat_index = seen_counts.get(base_example_id, 0)
        seen_counts[base_example_id] = repeat_index + 1

        item["workload_mode"] = workload_cfg.mode
        item["workload_pos"] = pos
        item["workload_repeat_index"] = repeat_index
        item["workload_pass"] = repeat_index

        item["base_example_id"] = base_example_id
        item["base_task"] = item.get("task")
        item["base_source_file"] = item.get("source_file")

        annotated.append(item)

    return annotated


def prepare_workload(
    cfg: BenchmarkConfig,
    base_examples: Optional[List[Example]] = None,
) -> List[Example]:
    """
    End-to-end helper:
    1) load base examples from Tier 2, unless already provided
    2) shape them into the requested benchmark sequence
    3) attach workload metadata

    This supports both usage styles:
    - prepare_workload(cfg)
    - prepare_workload(cfg, base_examples=preloaded_examples)
    """
    if base_examples is None:
        base_examples = load_base_examples(cfg)

    sequence = build_workload_sequence(base_examples, cfg.workload)
    annotated = annotate_workload_positions(sequence, cfg.workload)
    return annotated


def build_workload_manifest(
    cfg: BenchmarkConfig,
    workload: List[Example],
) -> Dict[str, Any]:
    """
    Build a small manifest describing the benchmark workload.

    Useful for saving alongside a run JSONL so you can later verify:
    - which mode was used
    - how many unique base examples were selected
    - how many total accesses were executed
    - which tasks are represented
    """
    unique_ids = sorted({ex.get("base_example_id", ex.get("example_id")) for ex in workload})
    tasks = sorted({ex.get("task") for ex in workload if ex.get("task") is not None})
    source_files = sorted({ex.get("source_file") for ex in workload if ex.get("source_file")})

    return {
        "benchmark_name": cfg.benchmark_name,
        "notes": cfg.notes,
        "tier2_repo": cfg.tier2_repo,
        "workload_mode": cfg.workload.mode,
        "task_glob": cfg.task_glob,
        "base_max_examples": cfg.max_examples,
        "total_workload_examples": len(workload),
        "unique_base_examples": len(unique_ids),
        "tasks": tasks,
        "source_files": source_files,
        "shuffle": cfg.workload.shuffle,
        "seed": cfg.workload.seed,
        "replay_k": cfg.workload.replay_k,
        "max_cache_items": cfg.max_cache_items,
        "max_input_tokens": cfg.max_input_tokens,
        "max_new_tokens": cfg.max_new_tokens,
        "model_id": cfg.model_id,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "total_requests": cfg.workload.total_requests,
        "repeat_fraction": cfg.workload.repeat_fraction,
        "observed_repeat_requests": max(0, len(workload) - len(unique_ids)),
        "observed_repeat_rate": (
            max(0, len(workload) - len(unique_ids)) / len(workload)
            if workload else 0.0
        )
    }


# ---------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------


def _clone_example(ex: Example) -> Example:
    """
    Deep-copy one example so benchmark metadata added later does not alias
    across repeated references.
    """
    return copy.deepcopy(ex)


def _build_cold(base: List[Example]) -> List[Example]:
    """
    One pass only. Best for cold-cache latency and truncation behavior.
    """
    return [_clone_example(ex) for ex in base]


def _build_replay_once(base: List[Example]) -> List[Example]:
    """
    Two total passes:
    first pass tends to miss, second pass should hit if cache capacity allows.
    """
    return _build_replay_k(base, replay_k=2)


def _build_replay_k(base: List[Example], replay_k: int) -> List[Example]:
    """
    Repeat the full sequence replay_k total times.

    Example:
      base = [A, B, C], replay_k=3
      result = [A, B, C, A, B, C, A, B, C]

    Why whole-sequence replay instead of immediate duplicates like [A, A, B, B]?
    - It better resembles a user returning to related queries later
    - It gives a controllable reuse distance
    - It makes cache size matter more
    """
    if replay_k <= 0:
        raise ValueError("replay_k must be > 0")

    out: List[Example] = []
    for _ in range(replay_k):
        for ex in base:
            out.append(_clone_example(ex))
    return out


def _build_cache_pressure(base: List[Example]) -> List[Example]:
    """
    Build a delayed-reuse sequence intended to stress an LRU cache.

    Strategy:
    - Pass 1: include all base examples once
    - Pass 2: revisit even-indexed examples
    - Pass 3: revisit odd-indexed examples

    Example for [A, B, C, D, E, F]:
      [A, B, C, D, E, F, A, C, E, B, D, F]

    Why this helps:
    - There is guaranteed reuse
    - Reuse is delayed rather than immediate
    - With a small cache, early items may be evicted before replay
    - With a larger cache, more of the second/third pass accesses should hit

    This gives you a simple first cache-pressure workload without needing
    cache-capacity-aware planning in this module.
    """
    out: List[Example] = []

    for ex in base:
        out.append(_clone_example(ex))

    evens = [base[i] for i in range(0, len(base), 2)]
    odds = [base[i] for i in range(1, len(base), 2)]

    for ex in evens:
        out.append(_clone_example(ex))
    for ex in odds:
        out.append(_clone_example(ex))

    return out

def _build_mixed_reuse(
    base: List[Example],
    *,
    total_requests: int,
    repeat_fraction: float,
    seed: int,
) -> List[Example]:
    """
    Build a mixed workload with a controlled repeat ratio.

    Example:
      total_requests = 100
      repeat_fraction = 0.30
      len(base) = 70

    Result:
      - 70 first-seen requests
      - 30 repeated requests sampled from those 70
      - final order shuffled deterministically
    """
    if not base:
        return []

    if total_requests <= 0:
        raise ValueError("total_requests must be > 0")
    if not (0.0 <= repeat_fraction < 1.0):
        raise ValueError("repeat_fraction must be in [0.0, 1.0)")

    rng = random.Random(seed)

    unique_pool = [_clone_example(ex) for ex in base]
    n_unique = len(unique_pool)

    if total_requests < n_unique:
        raise ValueError(
            f"total_requests ({total_requests}) must be >= number of selected base examples ({n_unique})"
        )

    n_repeats = total_requests - n_unique

    out: List[Example] = [_clone_example(ex) for ex in unique_pool]

    for _ in range(n_repeats):
        src = rng.choice(base)
        out.append(_clone_example(src))

    rng.shuffle(out)
    return out
# memarch/benchmarks/workload.py
"""
Workload construction utilities for memarch LongBench benchmarking.

This module sits above dataset loading and below benchmark execution.

Responsibilities:
- Load the base LongBench example set selected by BenchmarkConfig
- Transform that base set into a benchmark workload sequence
- Make replay / cache-pressure behavior explicit
- Attach benchmark-specific metadata to each workload item

This module does NOT:
- Run generation
- Query memory tiers
- Write logs
- Summarize results

Design goal:
Keep workload construction as parallel as possible to baseline so benchmark
comparisons isolate architecture differences rather than workload differences.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List

from memarch.benchmarks.configs import BenchmarkConfig, WorkloadConfig
from baseline.tiers.tier2_disk import DiskLoader


Example = Dict[str, Any]


def load_base_examples(cfg: BenchmarkConfig) -> List[Example]:
    """
    Load the base LongBench example set.

    Notes:
    - task_glob="" means "all task files"
    - max_examples applies to the base selection only
    - examples are materialized into a list so replay / reordering is deterministic

    We intentionally reuse the baseline DiskLoader here so baseline and memarch
    are benchmarked on the exact same dataset stream semantics.
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
        delayed/interleaved pattern that is more likely to cause RAM eviction
        and expose disk-tier reuse.

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
    
    if workload_cfg.mode == "exact_interleaved":
        return _build_exact_interleaved(base)

    if workload_cfg.mode == "approx_interleaved":
        return _build_approx_interleaved(base)

    if workload_cfg.mode == "family_clustered":
        return _build_family_clustered(base)

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
    - workload_mode
    - workload_pos
    - workload_repeat_index
    - workload_pass
    - base_example_id
    - base_task
    - base_source_file

    Important:
    - The original example_id is preserved
    - Benchmark metadata is additive
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


def attach_namespace_metadata(
    sequence: List[Example],
    cfg: BenchmarkConfig,
) -> List[Example]:
    """
    Attach memarch identity / namespace information to every workload item.
    """
    out: List[Example] = []
    for ex in sequence:
        item = dict(ex)
        item["namespace_user_id"] = cfg.namespaces.user_id
        item["namespace_session_id"] = cfg.namespaces.session_id
        item["namespace_cohort_id"] = cfg.namespaces.cohort_id
        out.append(item)
    return out


def prepare_workload(cfg: BenchmarkConfig) -> List[Example]:
    """
    End-to-end helper:
    1) load base examples
    2) shape them into the requested benchmark sequence
    3) attach workload metadata
    4) attach memarch namespace metadata
    """
    base_examples = load_base_examples(cfg)
    sequence = build_workload_sequence(base_examples, cfg.workload)
    annotated = annotate_workload_positions(sequence, cfg.workload)
    annotated = attach_namespace_metadata(annotated, cfg)
    return annotated


def build_workload_manifest(
    cfg: BenchmarkConfig,
    workload: List[Example],
) -> Dict[str, Any]:
    """
    Build a small manifest describing the benchmark workload and namespace scope.
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
        "ram_capacity_items": cfg.memory.ram_capacity_items,
        "disk_store_path": cfg.resolved_disk_store_path(),
        # "similarity_threshold": cfg.memory.similarity_threshold,
        "retrieval_mode": cfg.memory.retrieval_mode,
        "semantic_enabled": cfg.memory.semantic_enabled,
        "semantic_threshold_context": cfg.memory.semantic_threshold_context,
        "semantic_threshold_bypass": cfg.memory.semantic_threshold_bypass,
        "max_semantic_candidates": cfg.memory.max_semantic_candidates,
        "embedding_model_id": cfg.memory.embedding_model_id,
        "promote_disk_hits_to_ram": cfg.memory.promote_disk_hits_to_ram,
        "return_memory_directly": cfg.memory.return_memory_directly,
        "enable_storage": cfg.memory.enable_storage,
        "store_in_ram": cfg.memory.store_in_ram,
        "store_on_disk": cfg.memory.store_on_disk,
        "max_input_tokens": cfg.max_input_tokens,
        "max_new_tokens": cfg.max_new_tokens,
        "model_id": cfg.model_id,
        "namespaces": {
            "user_id": cfg.namespaces.user_id,
            "session_id": cfg.namespaces.session_id,
            "cohort_id": cfg.namespaces.cohort_id,
        },
        "total_requests": cfg.workload.total_requests,
        "repeat_fraction": cfg.workload.repeat_fraction,
        "observed_repeat_requests": max(0, len(workload) - len(unique_ids)),
        "observed_repeat_rate": (
            max(0, len(workload) - len(unique_ids)) / len(workload)
            if workload else 0.0
        ),
    }


def group_workload_by_base_id(workload: List[Example]) -> Dict[Any, List[Example]]:
    """
    Convenience helper for analysis/debugging:
    group workload items by base example id.
    """
    grouped: Dict[Any, List[Example]] = {}
    for ex in workload:
        key = ex.get("base_example_id", ex.get("example_id"))
        grouped.setdefault(key, []).append(ex)
    return grouped


# ---------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------


def _clone_example(ex: Example) -> Example:
    return copy.deepcopy(ex)


def _build_cold(base: List[Example]) -> List[Example]:
    return [_clone_example(ex) for ex in base]


def _build_replay_once(base: List[Example]) -> List[Example]:
    return _build_replay_k(base, replay_k=2)


def _build_replay_k(base: List[Example], replay_k: int) -> List[Example]:
    if replay_k <= 0:
        raise ValueError("replay_k must be > 0")

    out: List[Example] = []
    for _ in range(replay_k):
        for ex in base:
            out.append(_clone_example(ex))
    return out


def _build_cache_pressure(base: List[Example]) -> List[Example]:
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

def _group_by_family(examples: List[Example]) -> Dict[str, List[Example]]:
    groups: Dict[str, List[Example]] = {}

    for ex in examples:
        fid = ex.get("family_id") or ex.get("original_row_id") or ex.get("example_id")
        groups.setdefault(fid, []).append(ex)

    return groups

def _build_exact_interleaved(base: List[Example]) -> List[Example]:
    out: List[Example] = []

    for ex in base:
        out.append(_clone_example(ex))

    for ex in base:
        out.append(_clone_example(ex))

    return out

def _build_approx_interleaved(base: List[Example]) -> List[Example]:
    groups = _group_by_family(base)
    out: List[Example] = []

    originals = []
    variants = []

    for fam in groups.values():
        # sort so original comes first
        fam_sorted = sorted(fam, key=lambda x: x.get("paraphrase_index", -1))

        if not fam_sorted:
            continue

        originals.append(fam_sorted[0])

        if len(fam_sorted) > 1:
            variants.append(fam_sorted[1])  # take first paraphrase

    # First pass: originals
    for ex in originals:
        out.append(_clone_example(ex))

    # Second pass: variants
    for ex in variants:
        out.append(_clone_example(ex))

    return out

def _build_family_clustered(base: List[Example]) -> List[Example]:
    groups = _group_by_family(base)
    out: List[Example] = []

    for fam in groups.values():
        fam_sorted = sorted(fam, key=lambda x: x.get("paraphrase_index", -1))

        for ex in fam_sorted:
            out.append(_clone_example(ex))

    return out
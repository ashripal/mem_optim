# memarch/benchmarks/workload.py
"""
Workload construction utilities for memarch LongBench benchmarking.

This module sits above dataset loading and below benchmark execution.

Responsibilities:
- Load the base LongBench example set selected by BenchmarkConfig
- Transform that base set into a benchmark workload sequence
- Make replay / cache-pressure / paraphrase-family behavior explicit
- Attach benchmark-specific metadata to each workload item

This module does NOT:
- Run generation
- Query memory tiers
- Write logs
- Summarize results

Design goal:
Keep workload construction as parallel as possible to baseline so benchmark
comparisons isolate architecture differences rather than workload differences.

Verified paraphrase reuse relevance:
- approx_interleaved is the main workload for testing second-pass paraphrases
- family_clustered is useful for family-level consistency analysis
- workload metadata should preserve family/base relationships so later analysis
  can measure repeated-query and paraphrase-family behavior

Temporary debug additions:
- family-level prints inside _build_approx_interleaved()
- assertions that enforce:
    * pass 0 = originals only
    * pass 1 = variants only
These are intentionally loud while debugging workload correctness.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from memarch.benchmarks.configs import BenchmarkConfig, WorkloadConfig
from baseline.tiers.tier2_disk import DiskLoader


Example = Dict[str, Any]


# =============================================================================
# Small helpers
# =============================================================================

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _first_nonempty(*values: Any) -> str:
    for v in values:
        s = _safe_str(v)
        if s:
            return s
    return ""


def _is_paraphrase_mode(mode: str) -> bool:
    """
    Modes where family-aware semantics matter most.
    """
    return str(mode or "").strip() in {"approx_interleaved", "family_clustered"}


def _family_id_for_example(ex: Example) -> str:
    """
    Stable family identifier for paraphrase-family workloads.
    """
    return _first_nonempty(
        ex.get("family_id"),
        ex.get("original_row_id"),
        ex.get("base_example_id"),
        ex.get("example_id"),
    )


def _variant_label(ex: Example) -> str:
    return _safe_str(ex.get("variant")).lower()


def _example_id_text(ex: Example) -> str:
    """
    Stable string form of example_id for id-pattern heuristics.
    """
    return _safe_str(ex.get("example_id"))


def _is_original_variant(ex: Example) -> bool:
    """
    Stronger heuristic for identifying the original/base row within a paraphrase family.

    Why this change:
    - The previous version relied too heavily on paraphrase_index / missing fields
    - In workload_paraphrase-style files, the safest original indicators are often:
        * example_id == family_id
        * example_id == original_row_id
        * example_id does NOT contain a paraphrase suffix
    - We also explicitly reject rows that look like paraphrases by id/variant/query fields
    """
    variant = _variant_label(ex)
    ex_id = _example_id_text(ex)
    family_id = _safe_str(ex.get("family_id"))
    original_row_id = _safe_str(ex.get("original_row_id"))

    # -------------------------------------------------------------------------
    # Strong positive signals for original/base rows
    # -------------------------------------------------------------------------
    if variant == "original":
        return True

    if ex_id and family_id and ex_id == family_id:
        return True

    if ex_id and original_row_id and ex_id == original_row_id:
        return True

    # -------------------------------------------------------------------------
    # Strong negative signals for paraphrase rows
    # -------------------------------------------------------------------------
    if ex.get("paraphrase_index") is not None:
        return False

    if variant.startswith("para_") or variant in {"paraphrase", "rewrite", "rephrased"}:
        return False

    # Common id pattern in these workload files, e.g. squad_000083_para_0
    if "_para_" in ex_id or ex_id.endswith(("_p0", "_p1", "_p2", "_p3")):
        return False

    # If any paraphrase-specific surface-form field is populated, treat as non-original.
    if any(
        _safe_str(ex.get(k))
        for k in (
            "paraphrase",
            "paraphrase_text",
            "paraphrased_question",
            "question_paraphrase",
            "query_paraphrase",
            "variant_query",
            "rewrite",
            "rephrased_question",
        )
    ):
        return False

    # -------------------------------------------------------------------------
    # Fallback rule:
    # If this row belongs to a family and does not carry paraphrase signals,
    # treat it as original.
    # -------------------------------------------------------------------------
    if family_id or original_row_id:
        return True

    # Standalone non-family rows are treated as originals.
    return True


def _variant_sort_key(ex: Example) -> tuple[int, int, str]:
    """
    Sort original first, then paraphrases by paraphrase index if available.
    """
    if _is_original_variant(ex):
        return (0, -1, "original")

    pidx = ex.get("paraphrase_index")
    if pidx is not None:
        try:
            return (1, int(pidx), _variant_label(ex))
        except (TypeError, ValueError):
            pass

    variant = _variant_label(ex)
    if variant.startswith("para_"):
        suffix = variant.split("_", 1)[1]
        try:
            return (1, int(suffix), variant)
        except (TypeError, ValueError):
            return (1, 999999, variant)

    return (2, 999999, variant)


def _normalize_query_fields(ex: Example) -> Example:
    """
    Ensure the benchmark sees the intended retrieval/generation query text.

    For paraphrase-family datasets, many rows preserve the original 'question'
    while placing the paraphrased surface form in another field. MemArch later
    extracts query text by checking query_text/question/input/query/prompt/raw_query,
    so we normalize into query_text here.
    """
    item = dict(ex)

    is_original = _is_original_variant(item)

    original_query = _first_nonempty(
        item.get("query_text"),
        item.get("question"),
        item.get("input"),
        item.get("query"),
        item.get("prompt"),
        item.get("raw_query"),
    )

    paraphrase_query = _first_nonempty(
        item.get("paraphrase"),
        item.get("paraphrase_text"),
        item.get("paraphrased_question"),
        item.get("question_paraphrase"),
        item.get("query_paraphrase"),
        item.get("variant_query"),
        item.get("rewrite"),
        item.get("rephrased_question"),
    )

    chosen_query = original_query if is_original or not paraphrase_query else paraphrase_query

    if chosen_query:
        item["query_text"] = chosen_query
        item.setdefault("raw_query", chosen_query)

    # Keep the original question field when present for dataset fidelity, but if the
    # row is a paraphrase and question is blank, also fill it for easier debugging.
    if chosen_query and not _safe_str(item.get("question")):
        item["question"] = chosen_query

    return item


def _group_by_family(examples: List[Example]) -> Dict[str, List[Example]]:
    groups: Dict[str, List[Example]] = {}
    for ex in examples:
        fid = _family_id_for_example(ex)
        groups.setdefault(fid, []).append(ex)
    return groups


def _limit_base_examples_for_mode(
    examples: List[Example],
    workload_cfg: WorkloadConfig,
) -> List[Example]:
    """
    Apply workload.max_examples after loading, with family-aware semantics for
    paraphrase/family modes.
    """
    max_examples = workload_cfg.max_examples
    if max_examples is None or max_examples <= 0:
        return [dict(ex) for ex in examples]

    if _is_paraphrase_mode(workload_cfg.mode):
        groups = _group_by_family(examples)
        kept_family_ids = list(groups.keys())[:max_examples]
        out: List[Example] = []
        for fid in kept_family_ids:
            fam_sorted = sorted(groups[fid], key=_variant_sort_key)
            for ex in fam_sorted:
                out.append(dict(ex))
        return out

    return [dict(ex) for ex in examples[:max_examples]]


# =============================================================================
# Loading
# =============================================================================

def _load_examples_from_jsonl(path: Path) -> List[Example]:
    """
    Load examples directly from a single JSONL file.

    We intentionally load the full file first and apply max_examples later in a
    family-aware way. This avoids cutting paraphrase families in half.
    """
    rows: List[Example] = []

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)
            if not isinstance(ex, dict):
                continue

            ex = dict(ex)
            ex.setdefault("example_id", idx)
            ex.setdefault("task", ex.get("task") or path.stem)
            ex.setdefault("source_file", str(path.resolve()))
            ex.setdefault("source_id", ex.get("dataset_id") or ex.get("id") or ex.get("example_id"))

            ex = _normalize_query_fields(ex)
            rows.append(ex)

    return rows


def load_base_examples(cfg: BenchmarkConfig) -> List[Example]:
    """
    Load the base example set.

    Behavior:
    - If workload.task_glob points to a real .jsonl file, load that file directly.
    - Otherwise, treat workload.task_glob as a pattern for DiskLoader.
    - For paraphrase-aware modes, apply max_examples after loading in a
      family-aware way.
    """
    cfg.validate()

    task_glob = str(cfg.workload.task_glob or "").strip()

    if task_glob:
        task_path = Path(task_glob).expanduser()
        if task_path.suffix.lower() == ".jsonl" and task_path.exists() and task_path.is_file():
            rows = _load_examples_from_jsonl(task_path.resolve())
            return _limit_base_examples_for_mode(rows, cfg.workload)

    disk = DiskLoader(
        repo_dir=cfg.tier2_repo,
        task_glob=task_glob,
        max_examples=None,
    )
    rows = [_normalize_query_fields(dict(ex)) for ex in disk.iter_examples()]
    return _limit_base_examples_for_mode(rows, cfg.workload)


# =============================================================================
# Workload sequence construction
# =============================================================================

def build_workload_sequence(
    examples: List[Example],
    workload_cfg: WorkloadConfig,
) -> List[Example]:
    """
    Build the benchmark sequence from the base example set.
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


# =============================================================================
# Metadata annotation
# =============================================================================

def annotate_workload_positions(
    sequence: List[Example],
    workload_cfg: WorkloadConfig,
) -> List[Example]:
    """
    Attach benchmark-specific metadata to each workload item.
    """
    annotated: List[Example] = []
    seen_counts: Dict[Any, int] = {}

    for pos, ex in enumerate(sequence):
        item = dict(ex)

        if _is_paraphrase_mode(workload_cfg.mode):
            anchor_id = _family_id_for_example(item)
        else:
            anchor_id = item.get("example_id")

        repeat_index = seen_counts.get(anchor_id, 0)
        seen_counts[anchor_id] = repeat_index + 1

        item["workload_mode"] = workload_cfg.mode
        item["workload_pos"] = pos
        item["workload_repeat_index"] = repeat_index
        item["workload_pass"] = repeat_index

        item["base_example_id"] = anchor_id
        item["base_task"] = item.get("task")
        item["base_source_file"] = item.get("source_file")

        item["family_id"] = _family_id_for_example(item)
        item["variant_label"] = _variant_label(item) or ("original" if _is_original_variant(item) else "variant")
        item["is_original_variant"] = bool(_is_original_variant(item))

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


# =============================================================================
# Manifest helpers
# =============================================================================

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
    family_ids = sorted({ex.get("family_id") for ex in workload if ex.get("family_id")})
    variant_rows = [ex for ex in workload if not bool(ex.get("is_original_variant", False))]

    observed_repeat_requests = max(0, len(workload) - len(unique_ids))
    observed_repeat_rate = (
        observed_repeat_requests / len(workload)
        if workload else 0.0
    )

    return {
        "benchmark_name": cfg.benchmark_name,
        "notes": cfg.notes,
        "tier2_repo": cfg.tier2_repo,
        "workload_mode": cfg.workload.mode,
        "task_glob": cfg.workload.task_glob,
        "base_max_examples": cfg.workload.max_examples,
        "total_workload_examples": len(workload),
        "unique_base_examples": len(unique_ids),
        "unique_families": len(family_ids),
        "variant_rows": len(variant_rows),
        "tasks": tasks,
        "source_files": source_files,
        "shuffle": cfg.workload.shuffle,
        "seed": cfg.workload.seed,
        "replay_k": cfg.workload.replay_k,
        "ram_capacity_items": cfg.memory.ram_capacity_items,
        "disk_store_path": cfg.resolved_disk_store_path(),
        "retrieval_mode": cfg.memory.retrieval_mode,
        "lexical_enabled": cfg.memory.lexical_enabled,
        "lexical_threshold_context": cfg.memory.lexical_threshold_context,
        "lexical_threshold_bypass": cfg.memory.lexical_threshold_bypass,
        "lexical_top_k": cfg.memory.lexical_top_k,
        "prefer_same_source": cfg.memory.prefer_same_source,
        "safe_direct_reuse_tasks": list(cfg.memory.safe_direct_reuse_tasks),
        "semantic_enabled": cfg.memory.semantic_enabled,
        "semantic_threshold_context": cfg.memory.semantic_threshold_context,
        "semantic_threshold_bypass": cfg.memory.semantic_threshold_bypass,
        "max_semantic_candidates": cfg.memory.max_semantic_candidates,
        "allow_semantic_bypass": getattr(cfg.memory, "allow_semantic_bypass", True),
        "require_same_document_for_semantic_bypass": getattr(
            cfg.memory, "require_same_document_for_semantic_bypass", True
        ),
        "semantic_bypass_min_margin": getattr(cfg.memory, "semantic_bypass_min_margin", 0.02),
        "require_evidence_support_for_semantic_bypass": getattr(
            cfg.memory, "require_evidence_support_for_semantic_bypass", True
        ),
        "semantic_direct_reuse_tasks": list(
            getattr(cfg.memory, "semantic_direct_reuse_tasks", ["squad", "extractive_qa", "qa", "trec"])
        ),
        "semantic_bypass_max_answer_words": getattr(
            cfg.memory, "semantic_bypass_max_answer_words", 12
        ),
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
        "observed_repeat_requests": observed_repeat_requests,
        "observed_repeat_rate": observed_repeat_rate,
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


# =============================================================================
# Internal builders
# =============================================================================

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
    """
    First pass, then delayed even/odd revisits to apply RAM pressure.
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
    """
    if not base:
        return []

    if total_requests is None or total_requests <= 0:
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

    target_repeats = int(math.floor(float(total_requests) * float(repeat_fraction)))
    max_possible_repeats = total_requests - n_unique
    n_repeats = min(target_repeats, max_possible_repeats)

    out: List[Example] = [_clone_example(ex) for ex in unique_pool]

    for _ in range(n_repeats):
        src = rng.choice(base)
        out.append(_clone_example(src))

    while len(out) < total_requests:
        src = rng.choice(base)
        out.append(_clone_example(src))

    rng.shuffle(out)
    return out


def _build_exact_interleaved(base: List[Example]) -> List[Example]:
    """
    Build [all originals] + [same exact questions again].
    """
    out: List[Example] = []

    for ex in base:
        out.append(_clone_example(ex))

    for ex in base:
        out.append(_clone_example(ex))

    return out


def _build_approx_interleaved(base: List[Example]) -> List[Example]:
    """
    Build an interleaved approximate-reuse workload:
    [orig A, para A, orig B, para B, ...]

    This is the key workload for approximate reuse testing.

    Critical behavior:
    - each original must appear before its paired paraphrase
    - each paraphrase should be close enough to its original for reuse
    - never silently use a paraphrase as the family's "original"
    """
    groups = _group_by_family(base)
    out: List[Example] = []

    originals: List[Example] = []
    variants: List[Example] = []

    # Deterministic family order helps debugging and reproducibility.
    for fid in sorted(groups.keys()):
        fam = groups[fid]
        fam_sorted = sorted(fam, key=_variant_sort_key)
        if not fam_sorted:
            continue

        # ---------------------------------------------------------------------
        # DEBUG: print family composition and original classification
        # ---------------------------------------------------------------------
        print(f"[DEBUG] FAMILY {fid}")
        for ex in fam_sorted:
            print(
                "  ID:", _example_id_text(ex),
                "| is_original:", _is_original_variant(ex),
                "| variant:", _variant_label(ex),
                "| paraphrase_index:", ex.get("paraphrase_index"),
                "| family_id:", _safe_str(ex.get("family_id")),
                "| original_row_id:", _safe_str(ex.get("original_row_id")),
            )

        # ---------------------------------------------------------------------
        # Pick the original/base row conservatively.
        # Do NOT fall back to fam_sorted[0], because that can accidentally be a
        # paraphrase when family labeling is imperfect.
        # ---------------------------------------------------------------------
        original_candidates = [ex for ex in fam_sorted if _is_original_variant(ex)]

        if not original_candidates:
            # Try one last id-based rescue before skipping the family.
            original_candidates = [
                ex for ex in fam_sorted
                if _example_id_text(ex) == fid
                or _example_id_text(ex) == _safe_str(ex.get("original_row_id"))
            ]

        if not original_candidates:
            print(f"[DEBUG] SKIP FAMILY {fid}: no original candidate found")
            continue

        original = original_candidates[0]
        original_q = _safe_str(_normalize_query_fields(original).get("query_text"))

        # ---------------------------------------------------------------------
        # Pick a true non-original variant whose query surface differs from the
        # original query.
        # ---------------------------------------------------------------------
        non_originals: List[Example] = []
        for ex in fam_sorted:
            if ex is original:
                continue
            if _is_original_variant(ex):
                continue

            norm_ex = _normalize_query_fields(ex)
            ex_q = _safe_str(norm_ex.get("query_text"))

            # Skip duplicates of the original surface form.
            if ex_q and original_q and ex_q == original_q:
                continue

            non_originals.append(ex)

        if not non_originals:
            print(f"[DEBUG] SKIP FAMILY {fid}: no usable paraphrase variant found")
            continue

        variant = sorted(non_originals, key=_variant_sort_key)[0]

        orig_item = _normalize_query_fields(_clone_example(original))
        var_item = _normalize_query_fields(_clone_example(variant))

        orig_item["approx_family_id"] = fid
        orig_item["approx_role"] = "original"
        orig_item["is_original_variant"] = True

        var_item["approx_family_id"] = fid
        var_item["approx_role"] = "variant"
        var_item["is_original_variant"] = False

        # ---------------------------------------------------------------------
        # DEBUG assertions for each selected pair
        # ---------------------------------------------------------------------
        assert orig_item["approx_role"] == "original"
        assert var_item["approx_role"] == "variant"

        print(
            f"[DEBUG] SELECT FAMILY {fid}: "
            f"original={_example_id_text(orig_item)} "
            f"variant={_example_id_text(var_item)}"
        )

        originals.append(orig_item)
        variants.append(var_item)

    for orig, var in zip(originals, variants):
        out.append(orig)
        out.append(var)

    # -------------------------------------------------------------------------
    # DEBUG assertions across the final workload shape
    # -------------------------------------------------------------------------
    # for i, ex in enumerate(out[:len(originals)]):
    #     assert ex["approx_role"] == "original", f"Bad pass0 at {i}: {ex.get('example_id')}"

    # for i, ex in enumerate(out[len(originals):]):
    #     assert ex["approx_role"] == "variant", f"Bad pass1 at {i}: {ex.get('example_id')}"
    for i in range(0, len(out), 2):
        assert out[i]["approx_role"] == "original", (
            f"Bad interleaved original at {i}: {out[i].get('example_id')}"
        )
        if i + 1 < len(out):
            assert out[i + 1]["approx_role"] == "variant", (
                f"Bad interleaved variant at {i+1}: {out[i+1].get('example_id')}"
            )
            assert out[i]["approx_family_id"] == out[i + 1]["approx_family_id"], (
                f"Family mismatch at pair starting {i}: "
                f"{out[i].get('approx_family_id')} vs {out[i+1].get('approx_family_id')}"
            )

    print(
        f"[DEBUG] APPROX_INTERLEAVED BUILT: "
        f"{len(originals)} interleaved original/variant pairs"
    )

    for i in range(0, min(len(out), 12), 2):
        if i + 1 < len(out):
            print(
                "[ORDER CHECK]",
                i,
                _example_id_text(out[i]),
                out[i].get("approx_role"),
                "->",
                _example_id_text(out[i + 1]),
                out[i + 1].get("approx_role"),
                "| family:",
                out[i].get("approx_family_id"),
                flush=True,
            )

    return out


def _build_family_clustered(base: List[Example]) -> List[Example]:
    """
    Keep each family grouped together in stable family order.
    """
    groups = _group_by_family(base)
    out: List[Example] = []

    for family_id in sorted(groups.keys()):
        fam_sorted = sorted(groups[family_id], key=_variant_sort_key)
        for ex in fam_sorted:
            item = _clone_example(ex)
            item["approx_family_id"] = family_id
            item["approx_role"] = "original" if _is_original_variant(ex) else "variant"
            item = _normalize_query_fields(item)
            out.append(item)

    return out
# memarch/benchmarks/execute.py
"""
Benchmark execution for the memarch LongBench pipeline.

Responsibilities:
- Build a workload sequence using memarch.benchmarks.workload
- Initialize RAM store, disk store, memory manager, embedder, and generator
- Run the workload through memarch retrieval/generation
- Log benchmark-aware records to JSONL
- Save a workload manifest next to the run
- Optionally write a summary JSON after completion

Design goal:
Keep the benchmark protocol parallel to baseline while exposing richer
multi-tier behavior:
- exact RAM hit vs exact disk hit vs semantic-assisted generation vs compute miss
- whether generation was bypassed
- whether semantic retrieval was used
- whether disk hits were promoted to RAM
- storage decisions
- timing breakdowns
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from baseline.utils.metrics import compute_basic_metrics
from memarch.benchmarks.configs import BenchmarkConfig
from memarch.benchmarks.workload import build_workload_manifest, prepare_workload
from memarch.memory.disk_store import DiskStoreSQLite
from memarch.memory.manager import MemoryManager, MemoryManagerConfig
from memarch.memory.policy import RetrievalPolicy
from memarch.memory.ram_store import RamStoreLRU
from memarch.memory.schema import MemoryQuery, Scope
from memarch.models.embedder import Embedder, EmbedderConfig
from memarch.models.generator import GeneratorConfig, HFGenerator

try:
    from memarch.analysis.summarize import summarize_run  # type: ignore
except Exception:
    summarize_run = None  # optional


# ---------------------------------------------------------------------
# Small local utilities
# ---------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _safe_cfg_dict(cfg: BenchmarkConfig) -> Dict[str, Any]:
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    return dict(vars(cfg))


def _get_rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _get_system_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available(),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def _make_run_id(prefix: str = "memarch_benchmark") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def _write_json(path: str, payload: Dict[str, Any]) -> str:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return str(out)


def _maybe_clear_disk_store_before_run(cfg: BenchmarkConfig) -> Optional[str]:
    """
    Delete the persistent disk store before the run when configured.

    Returns the resolved path that was removed, or None if nothing was removed.
    """
    if not bool(getattr(cfg.memory, "clear_disk_store_before_run", False)):
        return None

    disk_path = Path(cfg.resolved_disk_store_path()).expanduser().resolve()
    if disk_path.exists():
        disk_path.unlink()
        return str(disk_path)
    return None


def _extract_semantic_debug(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize semantic retrieval debug fields for logging.

    Supports both:
    - direct debug fields attached at top level of manager meta
    - nested retrieval_debug / semantic dicts
    """
    retrieval_debug = meta.get("retrieval_debug")
    if not isinstance(retrieval_debug, dict):
        retrieval_debug = {}

    semantic_block = retrieval_debug.get("semantic")
    if not isinstance(semantic_block, dict):
        semantic_block = {}

    retrieval_stage = meta.get("retrieval_stage", retrieval_debug.get("retrieval_stage"))

    # Prefer explicit top-level fields if present, otherwise fall back to nested debug.
    semantic_reason = meta.get("semantic_reason")
    if semantic_reason is None:
        semantic_reason = retrieval_debug.get("reason", semantic_block.get("reason"))

    semantic_candidate_count = meta.get("semantic_candidate_count")
    if semantic_candidate_count is None:
        semantic_candidate_count = retrieval_debug.get("candidate_count", semantic_block.get("candidate_count"))

    semantic_top_score = meta.get("semantic_top_score")
    if semantic_top_score is None:
        semantic_top_score = retrieval_debug.get("top_score", semantic_block.get("top_score"))

    semantic_top_rank = meta.get("semantic_top_rank")
    if semantic_top_rank is None:
        semantic_top_rank = retrieval_debug.get("top_rank", semantic_block.get("top_rank"))

    semantic_enabled_debug = meta.get("semantic_enabled_debug")
    if semantic_enabled_debug is None:
        semantic_enabled_debug = retrieval_debug.get("semantic_enabled", semantic_block.get("semantic_enabled"))

    return {
        "retrieval_stage": retrieval_stage,
        "semantic_reason": semantic_reason,
        "semantic_candidate_count": semantic_candidate_count,
        "semantic_top_score": semantic_top_score,
        "semantic_top_rank": semantic_top_rank,
        "semantic_enabled_debug": semantic_enabled_debug,
        "retrieval_debug": retrieval_debug if retrieval_debug else None,
    }


class JSONLLogger(AbstractContextManager):
    """
    Minimal JSONL logger to keep this executor self-contained.
    """

    def __init__(self, path: str):
        self.path = str(Path(path).expanduser().resolve())
        self._fh = None

    def __enter__(self) -> "JSONLLogger":
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, record: Dict[str, Any]) -> None:
        assert self._fh is not None, "Logger file handle is not open"
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
        self._fh = None


def _stats_to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _stats_to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return {k: _stats_to_jsonable(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "__dict__"):
        return {k: _stats_to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _ram_stats(ram: Any) -> Any:
    if ram is None:
        return None
    for fn_name in ("stats", "snapshot_stats"):
        fn = getattr(ram, fn_name, None)
        if callable(fn):
            try:
                return _stats_to_jsonable(fn())
            except Exception:
                pass
    return None


def _disk_stats(disk: Any) -> Any:
    if disk is None:
        return None
    fn = getattr(disk, "stats", None)
    if callable(fn):
        try:
            return _stats_to_jsonable(fn())
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------


def _extract_query_text(example: Dict[str, Any]) -> str:
    candidates = [
        example.get("question"),
        example.get("input"),
        example.get("query"),
        example.get("prompt"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_context_text(example: Dict[str, Any]) -> str:
    candidates = [
        example.get("context"),
        example.get("passage"),
        example.get("document"),
        example.get("article"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_memory_query(example: Dict[str, Any], cfg: BenchmarkConfig) -> MemoryQuery:
    query_text = _extract_query_text(example)
    context_text = _extract_context_text(example)

    if not query_text:
        raise ValueError(
            f"Could not build non-empty raw_query for task={example.get('task')} "
            f"example_id={example.get('example_id')}. Keys={sorted(example.keys())}"
        )

    doc_signature = _sha256_text(
        f"{example.get('task', '')}||{example.get('source_file', '')}||{context_text}"
    )

    allow_semantic = bool(cfg.memory.semantic_enabled) and cfg.memory.retrieval_mode in {
        "semantic_context",
        "semantic_bypass",
    }

    return MemoryQuery(
        raw_query=query_text,
        user_id=cfg.namespaces.user_id,
        session_id=cfg.namespaces.session_id,
        cohort_id=cfg.namespaces.cohort_id,
        task=example.get("task") or "longbench_qa",
        model_id=cfg.model_id,
        prompt_version="longbench_v1",
        allow_semantic=allow_semantic,
        context={
            "dataset_context": context_text,
            "doc_signature": doc_signature,
            "source_file": example.get("source_file"),
            "task": example.get("task"),
            "example_id": example.get("example_id"),
        },
    )


def _init_ram_store(cfg: BenchmarkConfig) -> Any:
    """
    Initialize RAM store with both:
    - a generous byte budget
    - an explicit item-count cap for benchmark control
    """
    return RamStoreLRU(
        max_mb=64,
        max_items=int(cfg.memory.ram_capacity_items),
    )


def _init_embedder(cfg: BenchmarkConfig) -> Optional[Embedder]:
    if not bool(cfg.memory.semantic_enabled):
        return None

    emb_cfg = EmbedderConfig(
        model_id=cfg.memory.embedding_model_id,
        device=cfg.memory.embedding_device,
        local_files_only=cfg.memory.embedding_local_files_only,
    )
    return Embedder(emb_cfg)


def _init_generator(cfg: BenchmarkConfig) -> HFGenerator:
    gen_cfg = GeneratorConfig(
        model_id=cfg.model_id,
        device=getattr(cfg, "device", "auto"),
        max_input_length=int(getattr(cfg, "max_input_tokens", 2048)),
        max_new_tokens=int(getattr(cfg, "max_new_tokens", 64)),
        do_sample=False,
        local_files_only=bool(getattr(cfg, "local_files_only", False)),
        torch_dtype=str(getattr(cfg, "dtype", "auto")),
    )
    return HFGenerator(gen_cfg)


def _init_retrieval_policy(cfg: BenchmarkConfig) -> RetrievalPolicy:
    retrieval_mode = str(cfg.memory.retrieval_mode)

    semantic_enabled = bool(cfg.memory.semantic_enabled) and retrieval_mode in {
        "semantic_context",
        "semantic_bypass",
    }

    semantic_threshold_bypass = float(cfg.memory.semantic_threshold_bypass)
    if retrieval_mode == "semantic_context":
        semantic_threshold_bypass = 1.01

    return RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=semantic_enabled,
        semantic_threshold_context=float(cfg.memory.semantic_threshold_context),
        semantic_threshold_bypass=semantic_threshold_bypass,
        max_semantic_candidates=int(cfg.memory.max_semantic_candidates),
    )


def _init_manager(cfg: BenchmarkConfig, ram: Any, disk: Any, embedder: Optional[Embedder]) -> Any:
    mm_cfg = MemoryManagerConfig(
        retrieval_policy=_init_retrieval_policy(cfg),
        promote_disk_hits_to_ram=bool(cfg.memory.promote_disk_hits_to_ram),
        return_memory_directly=bool(cfg.memory.return_memory_directly),
        embedder=embedder,
    )
    return MemoryManager(ram=ram, disk=disk, cfg=mm_cfg)


# ---------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------


def _build_ok_record(
    *,
    example: Dict[str, Any],
    answer: str,
    meta: Dict[str, Any],
    quality: Dict[str, Any],
    rss_before: Optional[float],
    rss_after: Optional[float],
    total_latency_s: float,
    generator: HFGenerator,
    ram: Any,
    disk: Any,
    cfg: BenchmarkConfig,
) -> Dict[str, Any]:
    timings_ms = dict(meta.get("timings_ms") or {})
    raw_gen_meta = dict(getattr(generator, "last_generation_meta", {}) or {})

    memory_lookup_ms = float(
        meta.get("memory_lookup_ms", timings_ms.get("memory_lookup_ms", 0.0) or 0.0)
    )
    generation_ms_est = float(
        meta.get("generation_ms_est", timings_ms.get("generation_ms_est", 0.0) or 0.0)
    )

    generated = bool(meta.get("generated", False))
    used_memory = bool(meta.get("used_memory", False))

    if generation_ms_est <= 0.0:
        if generated:
            generation_ms_est = float(raw_gen_meta.get("gen_time_s", 0.0) or 0.0) * 1000.0
        else:
            generation_ms_est = 0.0

    timings_ms.setdefault("memory_lookup_ms", memory_lookup_ms)
    timings_ms.setdefault("generation_ms_est", generation_ms_est)
    timings_ms.setdefault("total_ms", total_latency_s * 1000.0)

    source_tier = str(meta.get("source_tier", "compute" if generated else "unknown"))
    llm_bypassed = bool(meta.get("semantic_bypassed", False)) or (
        used_memory and bool(cfg.memory.return_memory_directly)
    )

    if generated:
        gen_meta = raw_gen_meta
    else:
        gen_meta = {
            "device": None,
            "dtype": None,
            "generation_backend": None,
            "input_tokens": None,
            "output_tokens": None,
            "truncated": None,
            "tokenize_time_s": None,
            "gen_time_s": None,
            "decode_time_s": None,
            "cuda_device_name": None,
            "gpu_mem_allocated_mb": None,
            "gpu_mem_reserved_mb": None,
            "used_retrieved_context": None,
            "retrieved_match_type": None,
            "retrieved_source_tier": None,
            "retrieved_score": None,
        }

    semantic_debug = _extract_semantic_debug(meta)

    return {
        "type": "example_result",
        "ok": True,
        "task": example.get("task"),
        "example_id": example.get("example_id"),
        "source_file": example.get("source_file"),

        # workload provenance
        "workload_mode": example.get("workload_mode"),
        "workload_pos": example.get("workload_pos"),
        "workload_repeat_index": example.get("workload_repeat_index"),
        "workload_pass": example.get("workload_pass"),
        "base_example_id": example.get("base_example_id"),
        "base_task": example.get("base_task"),
        "base_source_file": example.get("base_source_file"),

        # namespaces
        "namespace_user_id": example.get("namespace_user_id"),
        "namespace_session_id": example.get("namespace_session_id"),
        "namespace_cohort_id": example.get("namespace_cohort_id"),
        "namespaces_checked": meta.get("namespaces_checked", []),

        # memory / serving path
        "used_memory": used_memory,
        "source_tier": source_tier,
        "served_from": source_tier,
        "llm_bypassed": llm_bypassed,
        "generated": generated,
        "promoted_to_ram": bool(meta.get("promoted_to_ram", False)),
        "stored": meta.get("stored"),
        "stored_scopes": meta.get("stored_scopes", []),

        # semantic retrieval metadata
        "semantic_used": bool(meta.get("semantic_used", False)),
        "semantic_bypassed": bool(meta.get("semantic_bypassed", False)),
        "semantic_candidate_rank": meta.get("semantic_candidate_rank"),
        "semantic_score": meta.get("score")
        if meta.get("match_type") == "semantic"
        else meta.get("hit_before_generate", {}).get("score")
        if bool(meta.get("semantic_used", False))
        else None,
        "semantic_reason": semantic_debug["semantic_reason"],
        "semantic_candidate_count": semantic_debug["semantic_candidate_count"],
        "semantic_top_score": semantic_debug["semantic_top_score"],
        "semantic_top_rank": semantic_debug["semantic_top_rank"],
        "semantic_enabled_debug": semantic_debug["semantic_enabled_debug"],

        # retrieval / match metadata
        "retrieval_stage": semantic_debug["retrieval_stage"],
        "match_type": meta.get("match_type"),
        "hit_before_generate": meta.get("hit_before_generate"),
        "retrieval_debug": semantic_debug["retrieval_debug"],

        # latency breakdown
        "latency_s": total_latency_s,
        "memory_lookup_ms": memory_lookup_ms,
        "generation_ms_est": generation_ms_est,
        "timings_ms": timings_ms,

        # memory usage
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": (
            (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None
        ),

        # generator / model metadata
        "device": gen_meta.get("device"),
        "dtype": gen_meta.get("dtype"),
        "generation_backend": gen_meta.get("generation_backend"),
        "input_tokens": gen_meta.get("input_tokens"),
        "output_tokens": gen_meta.get("output_tokens"),
        "truncated": gen_meta.get("truncated"),
        "tokenize_time_s": gen_meta.get("tokenize_time_s"),
        "gen_time_s": gen_meta.get("gen_time_s"),
        "decode_time_s": gen_meta.get("decode_time_s"),
        "cuda_device_name": gen_meta.get("cuda_device_name"),
        "gpu_mem_allocated_mb": gen_meta.get("gpu_mem_allocated_mb"),
        "gpu_mem_reserved_mb": gen_meta.get("gpu_mem_reserved_mb"),
        "used_retrieved_context": gen_meta.get("used_retrieved_context"),
        "retrieved_match_type": gen_meta.get("retrieved_match_type"),
        "retrieved_source_tier": gen_meta.get("retrieved_source_tier"),
        "retrieved_score": gen_meta.get("retrieved_score"),

        # answer + quality
        "answer": answer,
        "output_text": answer,
        "ref_text": quality.get("ref_text"),
        "exact_match": quality.get("exact_match"),
        "contains_answer": quality.get("contains_answer"),
        "token_f1": quality.get("token_f1"),
        "char_f1": quality.get("char_f1"),

        # throughput only meaningful on compute path
        "tokens_per_second": (
            (gen_meta.get("output_tokens", 0) / float(gen_meta.get("gen_time_s", 0.0)))
            if generated and float(gen_meta.get("gen_time_s", 0.0) or 0.0) > 0
            else None
        ),

        # live store stats
        "ram_stats_after": _ram_stats(ram),
        "disk_stats_after": _disk_stats(disk),
    }


def _build_error_record(
    *,
    example: Dict[str, Any],
    error: Exception,
    rss_before: Optional[float],
    rss_after: Optional[float],
    total_latency_s: float,
    ram: Any,
    disk: Any,
) -> Dict[str, Any]:
    return {
        "type": "example_result",
        "ok": False,
        "error": f"{type(error).__name__}: {error}",
        "task": example.get("task"),
        "example_id": example.get("example_id"),
        "source_file": example.get("source_file"),
        "workload_mode": example.get("workload_mode"),
        "workload_pos": example.get("workload_pos"),
        "workload_repeat_index": example.get("workload_repeat_index"),
        "workload_pass": example.get("workload_pass"),
        "base_example_id": example.get("base_example_id"),
        "base_task": example.get("base_task"),
        "base_source_file": example.get("base_source_file"),
        "namespace_user_id": example.get("namespace_user_id"),
        "namespace_session_id": example.get("namespace_session_id"),
        "namespace_cohort_id": example.get("namespace_cohort_id"),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": (
            (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None
        ),
        "latency_s": total_latency_s,
        "ram_stats_after": _ram_stats(ram),
        "disk_stats_after": _disk_stats(disk),
        "example_keys": sorted(example.keys()),
        "raw_query_candidate": (
            example.get("question")
            or example.get("input")
            or example.get("query")
            or example.get("prompt")
        ),
    }


# ---------------------------------------------------------------------
# Main benchmark execution
# ---------------------------------------------------------------------


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, str]:
    """
    Execute one memarch benchmark run and write artifacts.

    Returns a dict with artifact paths:
      - run_jsonl
      - workload_manifest_json
      - summary_json (optional)
    """
    cfg.validate()

    run_dir = Path(cfg.resolved_out_dir()).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    disk_store_parent = Path(cfg.resolved_disk_store_path()).expanduser().resolve().parent
    disk_store_parent.mkdir(parents=True, exist_ok=True)

    cleared_disk_store_path = _maybe_clear_disk_store_before_run(cfg)

    run_id = _make_run_id(prefix=cfg.benchmark_name)
    run_jsonl = run_dir / f"{run_id}.jsonl"
    manifest_json = run_dir / f"{run_id}.manifest.json"
    summary_json = run_dir / f"{run_id}.summary.json"

    workload = prepare_workload(cfg)
    workload_manifest = build_workload_manifest(cfg, workload)

    # Initialize memarch runtime
    ram = _init_ram_store(cfg)
    disk = DiskStoreSQLite(cfg.resolved_disk_store_path())
    embedder = _init_embedder(cfg)
    manager = _init_manager(cfg, ram=ram, disk=disk, embedder=embedder)
    generator = _init_generator(cfg)

    system_info = _get_system_info()
    generator_info = {}
    try:
        generator_info = generator.info()
    except Exception:
        generator_info = {}

    embedder_info = {}
    if embedder is not None:
        try:
            embedder_info = embedder.info()
        except Exception:
            embedder_info = {}

    n_total = 0
    n_ok = 0
    n_err = 0
    n_generated = 0
    n_memory_hits = 0
    n_exact_hits = 0
    n_semantic_context = 0
    n_semantic_bypass = 0
    total_latency_s_acc = 0.0

    with JSONLLogger(str(run_jsonl)) as logger:
        logger.write(
            {
                "type": "run_header",
                "run_id": run_id,
                "created_at": _utc_now_iso(),
                "benchmark_name": cfg.benchmark_name,
                "notes": cfg.notes,
                "config": _safe_cfg_dict(cfg),
                "resolved_out_dir": str(run_dir),
                "disk_store_path": cfg.resolved_disk_store_path(),
                "cleared_disk_store_before_run": cleared_disk_store_path,
                "system_info": system_info,
                "resolved_runtime": {
                    "generator": generator_info,
                    "embedder": embedder_info,
                },
                "workload_manifest_preview": {
                    "workload_mode": workload_manifest.get("workload_mode"),
                    "base_max_examples": workload_manifest.get("base_max_examples"),
                    "unique_base_examples": workload_manifest.get("unique_base_examples"),
                    "total_workload_examples": workload_manifest.get("total_workload_examples"),
                    "tasks": workload_manifest.get("tasks"),
                    "namespaces": workload_manifest.get("namespaces"),
                },
            }
        )

        for ex in workload:
            n_total += 1
            rss_before = _get_rss_mb()
            t0 = time.time()

            try:
                mq = _build_memory_query(ex, cfg)

                gen_meta_before = dict(getattr(generator, "last_generation_meta", {}) or {})
                answer, meta = manager.answer(mq, generator)
                total_latency_s = time.time() - t0
                rss_after = _get_rss_mb()

                meta = dict(meta or {})
                timings_ms = dict(meta.get("timings_ms") or {})
                gen_meta_after = dict(getattr(generator, "last_generation_meta", {}) or {})

                generation_happened = bool(meta.get("generated", False))
                if not generation_happened:
                    generation_happened = gen_meta_after != gen_meta_before and bool(gen_meta_after)

                gen_time_s = float(gen_meta_after.get("gen_time_s", 0.0) or 0.0)

                if "generation_ms_est" not in meta:
                    meta["generation_ms_est"] = gen_time_s * 1000.0 if generation_happened else 0.0

                if "memory_lookup_ms" not in meta:
                    lookup_ms = max(0.0, (total_latency_s * 1000.0) - meta["generation_ms_est"])
                    meta["memory_lookup_ms"] = lookup_ms

                timings_ms.setdefault("generation_ms_est", meta["generation_ms_est"])
                timings_ms.setdefault("memory_lookup_ms", meta["memory_lookup_ms"])
                timings_ms.setdefault("total_ms", total_latency_s * 1000.0)
                meta["timings_ms"] = timings_ms

                if "generated" not in meta:
                    meta["generated"] = generation_happened

                quality = compute_basic_metrics(answer, ex)

                record = _build_ok_record(
                    example=ex,
                    answer=answer,
                    meta=meta,
                    quality=quality,
                    rss_before=rss_before,
                    rss_after=rss_after,
                    total_latency_s=total_latency_s,
                    generator=generator,
                    ram=ram,
                    disk=disk,
                    cfg=cfg,
                )
                n_ok += 1
            except Exception as e:
                total_latency_s = time.time() - t0
                rss_after = _get_rss_mb()
                record = _build_error_record(
                    example=ex,
                    error=e,
                    rss_before=rss_before,
                    rss_after=rss_after,
                    total_latency_s=total_latency_s,
                    ram=ram,
                    disk=disk,
                )
                n_err += 1

            if bool(record.get("generated", False)):
                n_generated += 1
            if bool(record.get("used_memory", False)):
                n_memory_hits += 1
            if record.get("match_type") == "exact":
                n_exact_hits += 1
            if bool(record.get("semantic_used", False)):
                n_semantic_context += 1
            if bool(record.get("semantic_bypassed", False)):
                n_semantic_bypass += 1
            if record.get("latency_s") is not None:
                total_latency_s_acc += float(record["latency_s"])

            logger.write(record)

        logger.write(
            {
                "type": "run_footer",
                "run_id": run_id,
                "finished_at": _utc_now_iso(),
                "counts": {
                    "total": n_total,
                    "ok": n_ok,
                    "err": n_err,
                },
                "aggregate_metrics": {
                    "mean_latency_s": (total_latency_s_acc / n_total) if n_total else None,
                    "n_generated": n_generated,
                    "n_memory_hits": n_memory_hits,
                    "memory_hit_rate": (n_memory_hits / n_total) if n_total else 0.0,
                    "n_exact_hits": n_exact_hits,
                    "n_semantic_context_used": n_semantic_context,
                    "n_semantic_bypass": n_semantic_bypass,
                    "llm_calls_saved": n_total - n_generated,
                },
                "ram_stats_final": _ram_stats(ram),
                "disk_stats_final": _disk_stats(disk),
                "system_info": system_info,
                "resolved_runtime": {
                    "generator": generator_info,
                    "embedder": embedder_info,
                },
            }
        )

    _write_json(str(manifest_json), workload_manifest)

    artifacts: Dict[str, str] = {
        "run_jsonl": str(run_jsonl.resolve()),
        "workload_manifest_json": str(manifest_json.resolve()),
    }

    if bool(cfg.output.write_summary_json) and summarize_run is not None:
        summary = summarize_run(str(run_jsonl))
        _write_json(str(summary_json), summary)
        artifacts["summary_json"] = str(summary_json.resolve())

    return artifacts
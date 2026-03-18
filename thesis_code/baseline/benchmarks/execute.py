# memarch/benchmarks/execute.py
"""
Benchmark execution for the memarch LongBench pipeline.

Responsibilities:
- Build a workload sequence using memarch.benchmarks.workload
- Initialize RAM store, disk store, memory manager, and generator
- Run the workload through memarch retrieval/generation
- Log benchmark-aware records to JSONL
- Save a workload manifest next to the run
- Optionally write a summary JSON after completion

Design goal:
Keep the benchmark protocol parallel to baseline while exposing richer
multi-tier behavior:
- RAM hit vs disk hit vs compute miss
- whether generation was bypassed
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
from typing import Any, Dict, Optional, Tuple

import torch

from memarch.benchmarks.configs import BenchmarkConfig
from memarch.benchmarks.workload import build_workload_manifest, prepare_workload
from memarch.memory.disk_store import DiskStoreSQLite
from memarch.memory.manager import MemoryManager, MemoryManagerConfig
from memarch.memory.ram_store import RamStoreLRU
from memarch.memory.schema import MemoryHit, MemoryQuery, Provenance, QualitySignals

# Reuse your already-working HF compute backend for real generation
from baseline.tiers.tier0_compute import ComputeEngine
from baseline.utils.metrics import compute_basic_metrics

try:
    from memarch.analysis.summarize import summarize_run  # type: ignore
except Exception:
    summarize_run = None


# ---------------------------------------------------------------------
# Small local utilities
# ---------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()


def _compose_prompt(context_text: str, query_text: str) -> str:
    """
    Safely compose a prompt from context + query.

    Handles missing pieces gracefully.
    """
    context_text = context_text or ""
    query_text = query_text or ""

    if context_text and query_text:
        return f"{context_text}\n\n{query_text}".strip()
    if query_text:
        return query_text.strip()
    return context_text.strip()


def _extract_query_text(example: Dict[str, Any]) -> str:
    """
    Extract the user query from a LongBench-style example.

    Supports multiple possible field names because task schemas vary.
    """
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
    """
    Extract the supporting context/document text from a LongBench-style example.

    Supports multiple possible field names because task schemas vary.
    """
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
# Generator adapter: MemoryManager expects a generator-like object
# ---------------------------------------------------------------------


class HFComputeGeneratorAdapter:
    """
    Adapter that lets memarch's MemoryManager use the already-working HF compute
    path from the baseline.

    Expected interface:
        generate(mq: MemoryQuery, retrieved: Optional[MemoryHit]) -> (answer, prov, qual)
    """

    def __init__(self, compute: ComputeEngine, cfg: BenchmarkConfig):
        self.compute = compute
        self.cfg = cfg
        self.call_count = 0
        self.last_prompt: Optional[str] = None
        self.last_generation_meta: Optional[Dict[str, Any]] = None

    def _build_prompt(self, mq: MemoryQuery) -> str:
        dataset_ctx = (mq.context or {}).get("dataset_context", "") or ""
        return _compose_prompt(dataset_ctx, mq.raw_query)

    def generate(
        self,
        mq: MemoryQuery,
        retrieved: Optional[MemoryHit] = None,
    ) -> Tuple[str, Provenance, QualitySignals]:
        self.call_count += 1

        prompt = self._build_prompt(mq)
        self.last_prompt = prompt

        generation = self.compute.generate(
            prompt=prompt,
            max_input_tokens=int(getattr(self.cfg, "max_input_tokens", 8192)),
            max_new_tokens=int(getattr(self.cfg, "max_new_tokens", 64)),
        )

        if not generation.get("ok", False):
            raise RuntimeError(
                f"HF generator failed on device={generation.get('device')}: "
                f"{generation.get('error', 'unknown error')}"
            )

        self.last_generation_meta = dict(generation)
        answer = str(generation.get("output_text", ""))

        prov = Provenance(
            model_id=mq.model_id,
            prompt_version=mq.prompt_version,
            generated_at_utc=datetime.now(timezone.utc),
            generator_backend="hf_compute_adapter",
            quantization="unknown",
            context_window=int(getattr(self.cfg, "max_input_tokens", 8192)),
        )

        qual = QualitySignals(
            score=1.0 if answer else 0.0,
            success=bool(answer),
            metrics={
                "input_tokens": generation.get("input_tokens"),
                "output_tokens": generation.get("output_tokens"),
                "gen_time_s": generation.get("gen_time_s"),
            },
        )
        return answer, prov, qual


# ---------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------


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

    return MemoryQuery(
        raw_query=query_text,
        user_id=cfg.namespaces.user_id,
        session_id=cfg.namespaces.session_id,
        cohort_id=cfg.namespaces.cohort_id,
        task=example.get("task") or "longbench_qa",
        model_id=cfg.model_id,
        prompt_version="longbench_v1",
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
    Be tolerant to small API differences while keeping one execute.py.

    The demo app uses RamStoreLRU(max_mb=...).
    If your implementation instead supports item-count capacity, this tries that too.
    """
    candidates = [
        {"max_items": int(cfg.memory.ram_capacity_items)},
        {"capacity": int(cfg.memory.ram_capacity_items)},
        {"max_mb": int(cfg.memory.ram_capacity_items)},
    ]

    last_err: Optional[Exception] = None
    for kwargs in candidates:
        try:
            return RamStoreLRU(**kwargs)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not initialize RamStoreLRU with any known signature: {last_err}")


def _init_manager(cfg: BenchmarkConfig, ram: Any, disk: Any) -> Any:
    mm_cfg = MemoryManagerConfig(
        promote_disk_hits_to_ram=bool(cfg.memory.promote_disk_hits_to_ram),
        return_memory_directly=bool(cfg.memory.return_memory_directly),
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
    generator: HFComputeGeneratorAdapter,
    ram: Any,
    disk: Any,
    cfg: BenchmarkConfig,
) -> Dict[str, Any]:
    timings_ms = dict(meta.get("timings_ms") or {})
    memory_lookup_ms = float(meta.get("memory_lookup_ms", timings_ms.get("memory_lookup_ms", 0.0) or 0.0))
    generation_ms_est = float(meta.get("generation_ms_est", timings_ms.get("generation_ms_est", 0.0) or 0.0))
    timings_ms.setdefault("memory_lookup_ms", memory_lookup_ms)
    timings_ms.setdefault("generation_ms_est", generation_ms_est)
    timings_ms.setdefault("total_ms", total_latency_s * 1000.0)

    source_tier = str(meta.get("source_tier", "compute" if meta.get("generated") else "unknown"))
    used_memory = bool(meta.get("used_memory"))
    llm_bypassed = used_memory and bool(cfg.memory.return_memory_directly)

    gen_meta = generator.last_generation_meta or {}

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
        "generated": bool(meta.get("generated", not used_memory)),
        "promoted_to_ram": bool(meta.get("promoted_to_ram", False)),
        "stored": meta.get("stored"),
        "stored_scopes": meta.get("stored_scopes", []),

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
        "input_tokens": gen_meta.get("input_tokens"),
        "output_tokens": gen_meta.get("output_tokens"),
        "truncated": gen_meta.get("truncated"),
        "gen_time_s": gen_meta.get("gen_time_s"),
        "fallback_from": gen_meta.get("fallback_from"),
        "fallback_reason": gen_meta.get("fallback_reason"),
        "device_switch_persisted": gen_meta.get("device_switch_persisted"),

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
            if float(gen_meta.get("gen_time_s", 0.0) or 0.0) > 0
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
        "example_keys": sorted(example.keys()),
        "raw_query_candidate": (
            example.get("question")
            or example.get("input")
            or example.get("query")
            or example.get("prompt")
        ),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": (
            (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None
        ),
        "latency_s": total_latency_s,
        "ram_stats_after": _ram_stats(ram),
        "disk_stats_after": _disk_stats(disk),
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

    Path(cfg.resolved_disk_store_path()).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    run_id = _make_run_id(prefix=cfg.benchmark_name)
    run_jsonl = run_dir / f"{run_id}.jsonl"
    manifest_json = run_dir / f"{run_id}.manifest.json"
    summary_json = run_dir / f"{run_id}.summary.json"

    workload = prepare_workload(cfg)
    workload_manifest = build_workload_manifest(cfg, workload)

    ram = _init_ram_store(cfg)
    disk = DiskStoreSQLite(cfg.resolved_disk_store_path())
    manager = _init_manager(cfg, ram=ram, disk=disk)

    compute = ComputeEngine(cfg)
    generator = HFComputeGeneratorAdapter(compute=compute, cfg=cfg)

    system_info = _get_system_info()

    n_total = 0
    n_ok = 0
    n_err = 0

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
                "system_info": system_info,
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
                answer, meta = manager.answer(mq, generator)
                total_latency_s = time.time() - t0
                rss_after = _get_rss_mb()

                quality = compute_basic_metrics(answer, ex)

                record = _build_ok_record(
                    example=ex,
                    answer=answer,
                    meta=dict(meta or {}),
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
                "ram_stats_final": _ram_stats(ram),
                "disk_stats_final": _disk_stats(disk),
                "generator_call_count": generator.call_count,
                "system_info": system_info,
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
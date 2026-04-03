from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from memarch.utils.metrics import compute_basic_metrics
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


_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

_TREC_LABELS = {"ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"}


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
    info: Dict[str, Any] = {
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

    if torch.cuda.is_available():
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["cuda_total_memory_mb"] = round(float(props.total_memory) / (1024.0 * 1024.0), 3)
            info["cuda_capability"] = f"{props.major}.{props.minor}"
            info["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 3)
            info["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024.0 * 1024.0), 3)
        except Exception:
            pass

    return info


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


def _construct_with_supported_kwargs(cls: Any, **kwargs: Any) -> Any:
    """
    Construct an object using only kwargs supported by the target signature.
    """
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**kwargs)

    supported = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return cls(**filtered)


def _safe_component_info(component: Any) -> Dict[str, Any]:
    if component is None:
        return {}
    info_fn = getattr(component, "info", None)
    if callable(info_fn):
        try:
            out = info_fn()
            return dict(out) if isinstance(out, dict) else {"info": out}
        except Exception as e:
            return {"info_error": f"{type(e).__name__}: {e}"}
    return {}


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
    stage_is_semantic = retrieval_stage == "semantic"

    semantic_reason = meta.get("semantic_reason")
    if semantic_reason is None:
        semantic_reason = semantic_block.get("reason") if not stage_is_semantic else retrieval_debug.get(
            "reason", semantic_block.get("reason")
        )

    semantic_candidate_count = meta.get("semantic_candidate_count")
    if semantic_candidate_count is None:
        semantic_candidate_count = semantic_block.get("candidate_count") if not stage_is_semantic else retrieval_debug.get(
            "candidate_count", semantic_block.get("candidate_count")
        )

    semantic_top_score = meta.get("semantic_top_score")
    if semantic_top_score is None:
        semantic_top_score = semantic_block.get("top_score") if not stage_is_semantic else retrieval_debug.get(
            "top_score", semantic_block.get("top_score")
        )

    semantic_top_rank = meta.get("semantic_top_rank")
    if semantic_top_rank is None:
        semantic_top_rank = semantic_block.get("top_rank") if not stage_is_semantic else retrieval_debug.get(
            "top_rank", semantic_block.get("top_rank")
        )

    semantic_enabled_debug = meta.get("semantic_enabled_debug")
    if semantic_enabled_debug is None:
        semantic_enabled_debug = semantic_block.get("semantic_enabled") if not stage_is_semantic else retrieval_debug.get(
            "semantic_enabled", semantic_block.get("semantic_enabled")
        )

    return {
        "retrieval_stage": retrieval_stage,
        "semantic_reason": semantic_reason,
        "semantic_candidate_count": semantic_candidate_count,
        "semantic_top_score": semantic_top_score,
        "semantic_top_rank": semantic_top_rank,
        "semantic_enabled_debug": semantic_enabled_debug,
        "retrieval_debug": retrieval_debug if retrieval_debug else None,
    }


def _extract_lexical_debug(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize lexical retrieval debug fields for logging.
    """
    retrieval_debug = meta.get("retrieval_debug")
    if not isinstance(retrieval_debug, dict):
        retrieval_debug = {}

    lexical_block = retrieval_debug.get("lexical")
    if not isinstance(lexical_block, dict):
        lexical_block = {}

    retrieval_stage = meta.get("retrieval_stage", retrieval_debug.get("retrieval_stage"))
    stage_is_lexical = retrieval_stage == "lexical"

    lexical_reason = meta.get("lexical_reason")
    if lexical_reason is None:
        lexical_reason = lexical_block.get("reason") if not stage_is_lexical else retrieval_debug.get(
            "reason", lexical_block.get("reason")
        )

    lexical_candidate_count = meta.get("lexical_candidate_count")
    if lexical_candidate_count is None:
        lexical_candidate_count = lexical_block.get("candidate_count") if not stage_is_lexical else retrieval_debug.get(
            "candidate_count", lexical_block.get("candidate_count")
        )

    lexical_top_score = meta.get("lexical_top_score")
    if lexical_top_score is None:
        lexical_top_score = lexical_block.get("top_score") if not stage_is_lexical else retrieval_debug.get(
            "top_score", lexical_block.get("top_score")
        )

    lexical_top_rank = meta.get("lexical_top_rank")
    if lexical_top_rank is None:
        lexical_top_rank = lexical_block.get("top_rank") if not stage_is_lexical else retrieval_debug.get(
            "top_rank", lexical_block.get("top_rank")
        )

    lexical_enabled_debug = meta.get("lexical_enabled_debug")
    if lexical_enabled_debug is None:
        lexical_enabled_debug = lexical_block.get("lexical_enabled") if not stage_is_lexical else retrieval_debug.get(
            "lexical_enabled", lexical_block.get("lexical_enabled")
        )

    lexical_match_type = meta.get("lexical_match_type")
    if lexical_match_type is None:
        lexical_match_type = lexical_block.get("lexical_match_type") if not stage_is_lexical else retrieval_debug.get(
            "lexical_match_type", lexical_block.get("lexical_match_type")
        )

    lexical_same_source = meta.get("lexical_same_source")
    if lexical_same_source is None:
        lexical_same_source = lexical_block.get("same_source") if not stage_is_lexical else retrieval_debug.get(
            "same_source", lexical_block.get("same_source")
        )

    return {
        "retrieval_stage": retrieval_stage,
        "lexical_reason": lexical_reason,
        "lexical_candidate_count": lexical_candidate_count,
        "lexical_top_score": lexical_top_score,
        "lexical_top_rank": lexical_top_rank,
        "lexical_enabled_debug": lexical_enabled_debug,
        "lexical_match_type": lexical_match_type,
        "lexical_same_source": lexical_same_source,
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


def _normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _normalize_for_matching(text: str) -> str:
    return _normalize_ws(text).lower()


def _extract_word_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(_normalize_for_matching(text))


def _extract_query_text(example: Dict[str, Any]) -> str:
    candidates = [
        example.get("query_text"),
        example.get("question"),
        example.get("input"),
        example.get("query"),
        example.get("prompt"),
        example.get("raw_query"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_context_text(example: Dict[str, Any]) -> str:
    candidates = [
        example.get("context_text"),
        example.get("dataset_context"),
        example.get("context"),
        example.get("passage"),
        example.get("document"),
        example.get("article"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_doc_signature(example: Dict[str, Any], *, context_text: str) -> str:
    """
    Build a stable document signature for same-document reuse.

    Priority order:
    1. explicit doc_signature if already present
    2. paraphrase-family identifiers
    3. stable source/document ids
    4. deterministic fallback hash over task + source file + context text
    """
    existing = example.get("doc_signature")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    for key in (
        "family_id",
        "original_row_id",
        "base_example_id",
        "source_id",
        "_id",
        "id",
        "doc_id",
        "document_id",
        "article_id",
        "passage_id",
        "title",
    ):
        value = example.get(key)
        if value is None:
            continue
        sval = str(value).strip()
        if sval:
            return _sha256_text(f"docsig::{example.get('task', '')}::{key}::{sval}")

    return _sha256_text(
        f"{example.get('task', '')}||{example.get('source_file', '')}||{context_text}"
    )


def _extract_chunk_index(example: Dict[str, Any]) -> Optional[int]:
    candidates = [
        example.get("chunk_index"),
        example.get("chunk_id_numeric"),
        example.get("source_record_index"),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            idx = int(value)
            if idx >= 0:
                return idx
        except (TypeError, ValueError):
            continue
    return None


def _extract_chunk_id(example: Dict[str, Any]) -> Optional[str]:
    candidates = [
        example.get("chunk_id"),
        example.get("passage_id"),
        example.get("doc_chunk_id"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_source_id(example: Dict[str, Any]) -> Optional[str]:
    candidates = [
        example.get("source_id"),
        example.get("_id"),
        example.get("id"),
        example.get("example_id"),
    ]
    for value in candidates:
        if value is None:
            continue
        sval = str(value).strip()
        if sval:
            return sval
    return None


def _infer_question_type(example: Dict[str, Any], query_text: str) -> str:
    task = str(example.get("task") or "").strip().lower()
    q = _normalize_for_matching(query_text)

    if task == "trec":
        return "classification"

    if q.startswith(("who ", "what ", "when ", "where ", "why ", "how ")):
        return "qa"

    if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ")):
        return "boolean_qa"

    if "classify" in q or "label" in q:
        return "classification"

    if "summarize" in q or "summary" in q:
        return "summarization"

    return "unknown"


def _build_answer_canonical(example: Dict[str, Any]) -> Optional[str]:
    task = str(example.get("task") or "").strip().lower()

    if task == "trec":
        answers = example.get("answers")
        if isinstance(answers, list) and answers:
            first = str(answers[0]).strip().upper()
            if first in _TREC_LABELS:
                return first

        target = example.get("target")
        if isinstance(target, str) and target.strip().upper() in _TREC_LABELS:
            return target.strip().upper()

    return None


def _pick_best_overlap_window(query_text: str, context_text: str, *, window_chars: int = 320) -> str:
    ctx = _normalize_ws(context_text)
    if not ctx:
        return ""

    q_tokens = set(_extract_word_tokens(query_text))
    if not q_tokens:
        return ctx[:window_chars].strip()

    raw_lines = [line.strip() for line in context_text.splitlines() if line.strip()]
    if len(raw_lines) >= 2:
        best_line = ""
        best_score = -1
        for line in raw_lines:
            ltoks = set(_extract_word_tokens(line))
            score = len(q_tokens & ltoks)
            if score > best_score:
                best_score = score
                best_line = line
        best_line = _normalize_ws(best_line)
        if best_score > 0 and best_line:
            return best_line[:window_chars].strip()

    best_start = 0
    best_score = -1
    stride = max(64, window_chars // 2)
    for start in range(0, len(ctx), stride):
        window = ctx[start:start + window_chars]
        if not window:
            continue
        wtoks = set(_extract_word_tokens(window))
        score = len(q_tokens & wtoks)
        if score > best_score:
            best_score = score
            best_start = start

    snippet = ctx[best_start:best_start + window_chars].strip()
    if snippet:
        return snippet

    return ctx[:window_chars].strip()


def _build_memory_query(example: Dict[str, Any], cfg: BenchmarkConfig) -> MemoryQuery:
    query_text = _extract_query_text(example)
    context_text = _extract_context_text(example)

    if not query_text:
        raise ValueError(
            f"Could not build non-empty raw_query for task={example.get('task')} "
            f"example_id={example.get('example_id')}. Keys={sorted(example.keys())}"
        )

    doc_signature = _extract_doc_signature(example, context_text=context_text)
    source_file = example.get("source_file")
    source_id = _extract_source_id(example)
    chunk_index = _extract_chunk_index(example)
    chunk_id = _extract_chunk_id(example)
    question_type = _infer_question_type(example, query_text)
    evidence_text = _pick_best_overlap_window(query_text, context_text)
    answer_canonical = _build_answer_canonical(example)

    print(
        "[EXECUTE DOCSIG]",
        {
            "example_id": example.get("example_id"),
            "family_id": example.get("family_id"),
            "original_row_id": example.get("original_row_id"),
            "base_example_id": example.get("base_example_id"),
            "source_id": example.get("source_id"),
            "doc_signature": doc_signature,
        },
        flush=True,
    )

    retrieval_mode = str(getattr(cfg.memory, "retrieval_mode", "exact_only")).strip()

    allow_semantic = bool(getattr(cfg.memory, "semantic_enabled", False)) and retrieval_mode in {
        "semantic_context",
        "semantic_bypass",
        "lexical_semantic_context",
        "lexical_gated_direct_semantic_context",
    }

    mq_kwargs: Dict[str, Any] = {
        "raw_query": query_text,
        "user_id": cfg.namespaces.user_id,
        "session_id": cfg.namespaces.session_id,
        "cohort_id": cfg.namespaces.cohort_id,
        "task": example.get("task") or "longbench_qa",
        "model_id": cfg.model_id,
        "prompt_version": "longbench_v1",
        "allow_semantic": allow_semantic,
        "doc_signature": doc_signature,
        "source_file": source_file,
        "source_id": source_id,
        "chunk_index": chunk_index,
        "chunk_id": chunk_id,
        "question_type": question_type,
        "evidence_text": evidence_text or None,
        "answer_canonical": answer_canonical,
        "context": {
            "dataset_context": context_text,
            "doc_signature": doc_signature,
            "source_file": source_file,
            "source_id": source_id,
            "task": example.get("task"),
            "example_id": example.get("example_id"),
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "question_type": question_type,
            "evidence_text": evidence_text or None,
            "answer_canonical": answer_canonical,
        },
    }

    return _construct_with_supported_kwargs(MemoryQuery, **mq_kwargs)


def _init_ram_store(cfg: BenchmarkConfig) -> Any:
    return RamStoreLRU(
        max_mb=int(getattr(cfg.memory, "ram_max_mb", 64)),
        max_items=int(cfg.memory.ram_capacity_items),
    )


def _init_embedder(cfg: BenchmarkConfig) -> Optional[Embedder]:
    if not bool(cfg.memory.semantic_enabled):
        return None

    emb_cfg = EmbedderConfig(
        model_id=cfg.memory.embedding_model_id,
        device=getattr(cfg.memory, "embedding_device", "auto"),
        max_length=int(getattr(cfg.memory, "embedding_max_length", 512)),
        batch_size=int(getattr(cfg.memory, "embedding_batch_size", 16)),
        normalize=bool(getattr(cfg.memory, "embedding_normalize", True)),
        local_files_only=bool(getattr(cfg.memory, "embedding_local_files_only", False)),
        use_fast_tokenizer=bool(getattr(cfg.memory, "embedding_use_fast_tokenizer", False)),
        torch_dtype=str(getattr(cfg.memory, "embedding_dtype", "auto")),
        low_cpu_mem_usage=bool(getattr(cfg.memory, "embedding_low_cpu_mem_usage", True)),
        use_safetensors=bool(getattr(cfg.memory, "embedding_use_safetensors", True)),
        trust_remote_code=bool(getattr(cfg.memory, "embedding_trust_remote_code", False)),
        attn_implementation=str(getattr(cfg.memory, "embedding_attn_implementation", "auto")),
        cpu_fallback_on_failure=bool(getattr(cfg.memory, "embedding_cpu_fallback_on_failure", True)),
    )
    return Embedder(emb_cfg)


def _init_generator(cfg: BenchmarkConfig) -> HFGenerator:
    gen_cfg = GeneratorConfig(
        model_id=cfg.model_id,
        device=getattr(cfg, "device", "auto"),
        max_input_length=int(getattr(cfg, "max_input_tokens", 2048)),
        max_new_tokens=int(getattr(cfg, "max_new_tokens", 64)),
        decoding_mode=str(getattr(cfg, "decoding_mode", "greedy")),
        num_beams=int(getattr(cfg, "num_beams", 1)),
        temperature=float(getattr(cfg, "temperature", 0.2)),
        top_p=float(getattr(cfg, "top_p", 0.95)),
        do_sample=bool(getattr(cfg, "do_sample", False)),
        local_files_only=bool(getattr(cfg, "local_files_only", False)),
        torch_dtype=str(getattr(cfg, "dtype", "auto")),
        use_fast_tokenizer=bool(getattr(cfg, "use_fast_tokenizer", False)),
        cpu_fallback_on_failure=bool(getattr(cfg, "cpu_fallback_on_failure", True)),
        low_cpu_mem_usage=bool(getattr(cfg, "low_cpu_mem_usage", True)),
        use_safetensors=bool(getattr(cfg, "use_safetensors", True)),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
        attn_implementation=str(getattr(cfg, "attn_implementation", "auto")),
        use_kv_cache=bool(getattr(cfg, "use_kv_cache", True)),
        include_retrieved_memory_context=bool(getattr(cfg, "include_retrieved_memory_context", True)),
        include_dataset_context=bool(getattr(cfg, "include_dataset_context", True)),
        include_doc_signature=bool(getattr(cfg, "include_doc_signature", True)),
        prefer_retrieved_evidence_context=bool(getattr(cfg, "prefer_retrieved_evidence_context", True)),
        reduce_context_on_semantic_hit=bool(getattr(cfg, "reduce_context_on_semantic_hit", True)),
        prefer_local_context_for_qa=bool(getattr(cfg, "prefer_local_context_for_qa", True)),
        trec_use_few_shot=bool(getattr(cfg, "trec_use_few_shot", False)),
        skip_special_tokens=bool(getattr(cfg, "skip_special_tokens", True)),
    )
    return HFGenerator(gen_cfg)


def _init_retrieval_policy(cfg: BenchmarkConfig) -> RetrievalPolicy:
    retrieval_mode = str(getattr(cfg.memory, "retrieval_mode", "exact_only")).strip()

    lexical_enabled = bool(getattr(cfg.memory, "lexical_enabled", False)) and retrieval_mode in {
        "lexical_context",
        "lexical_gated_direct",
        "lexical_semantic_context",
        "lexical_gated_direct_semantic_context",
    }

    semantic_enabled = bool(getattr(cfg.memory, "semantic_enabled", False)) and retrieval_mode in {
        "semantic_context",
        "semantic_bypass",
        "lexical_semantic_context",
        "lexical_gated_direct_semantic_context",
    }

    semantic_allow_bypass = semantic_enabled and retrieval_mode == "semantic_bypass"

    lexical_threshold_bypass = float(
        getattr(
            cfg.memory,
            "lexical_threshold_bypass",
            getattr(cfg.memory, "lexical_direct_threshold", 0.90),
        )
    )
    if retrieval_mode in {"lexical_context", "lexical_semantic_context"}:
        lexical_threshold_bypass = 1.01

    policy_kwargs: Dict[str, Any] = {
        "scope_order": [Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        "lexical_enabled": lexical_enabled,
        "lexical_threshold_context": float(
            getattr(cfg.memory, "lexical_threshold_context", 0.55)
        ),
        "lexical_threshold_bypass": lexical_threshold_bypass,
        "lexical_top_k": int(getattr(cfg.memory, "lexical_top_k", 3)),
        "prefer_same_source": bool(getattr(cfg.memory, "prefer_same_source", True)),
        "safe_direct_reuse_tasks": list(
            getattr(cfg.memory, "safe_direct_reuse_tasks", ["trec"])
        ),
        "semantic_enabled": semantic_enabled,
        "semantic_threshold_context": float(
            getattr(cfg.memory, "semantic_threshold_context", 0.85)
        ),
        "semantic_threshold_bypass": float(
            getattr(cfg.memory, "semantic_threshold_bypass", 1.01)
        ),
        "max_semantic_candidates": int(
            getattr(cfg.memory, "max_semantic_candidates", 5)
        ),
        "allow_semantic_bypass": semantic_allow_bypass,
    }

    return _construct_with_supported_kwargs(RetrievalPolicy, **policy_kwargs)


def _init_manager(cfg: BenchmarkConfig, ram: Any, disk: Any, embedder: Optional[Embedder]) -> Any:
    retrieval_policy = _init_retrieval_policy(cfg)

    lexical_enabled = bool(getattr(cfg.memory, "lexical_enabled", False)) and str(
        getattr(cfg.memory, "retrieval_mode", "exact_only")
    ).strip() in {
        "lexical_context",
        "lexical_gated_direct",
        "lexical_semantic_context",
        "lexical_gated_direct_semantic_context",
    }

    manager_kwargs: Dict[str, Any] = {
        "retrieval_policy": retrieval_policy,
        "promote_disk_hits_to_ram": bool(cfg.memory.promote_disk_hits_to_ram),
        "return_memory_directly": bool(cfg.memory.return_memory_directly),
        "embedder": embedder,
        "lexical_enabled": lexical_enabled,
        "lexical_context_threshold": float(
            getattr(cfg.memory, "lexical_threshold_context", 0.55)
        ),
        "lexical_direct_threshold": float(
            getattr(
                cfg.memory,
                "lexical_threshold_bypass",
                getattr(cfg.memory, "lexical_direct_threshold", 0.90),
            )
        ),
        "lexical_top_k": int(getattr(cfg.memory, "lexical_top_k", 3)),
        "prefer_same_source": bool(getattr(cfg.memory, "prefer_same_source", True)),
        "safe_direct_reuse_tasks": list(
            getattr(cfg.memory, "safe_direct_reuse_tasks", ["trec"])
        ),
        "enable_storage": bool(getattr(cfg.memory, "enable_storage", True)),
        "store_in_ram": bool(getattr(cfg.memory, "store_in_ram", True)),
        "store_on_disk": bool(getattr(cfg.memory, "store_on_disk", True)),
    }

    mm_cfg = _construct_with_supported_kwargs(MemoryManagerConfig, **manager_kwargs)
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
    match_type = meta.get("match_type")

    llm_bypassed = bool(used_memory and not generated)

    if generated:
        gen_meta = raw_gen_meta
    else:
        gen_meta = {
            "device": None,
            "dtype": None,
            "generation_backend": None,
            "decoding_mode": None,
            "num_beams": None,
            "do_sample": None,
            "temperature": None,
            "top_p": None,
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
            "retrieved_doc_signature_match": None,
            "retrieved_evidence_chars": None,
            "reduced_context_used": None,
            "full_context_chars": None,
            "final_context_chars": None,
            "use_kv_cache": None,
            "fallback_used": None,
            "fallback_from": None,
            "fallback_reason": None,
        }

    semantic_debug = _extract_semantic_debug(meta)
    lexical_debug = _extract_lexical_debug(meta)

    retrieval_score = (
        meta.get("score")
        if meta.get("score") is not None
        else meta.get("retrieved_score")
        if meta.get("retrieved_score") is not None
        else lexical_debug["lexical_top_score"]
        if lexical_debug["lexical_top_score"] is not None
        else semantic_debug["semantic_top_score"]
    )

    lexical_used = bool(meta.get("lexical_used", False)) or (str(match_type or "").lower() == "lexical")
    lexical_bypassed = bool(meta.get("lexical_bypassed", False)) and not generated
    lexical_context_used = bool(meta.get("lexical_context_used", False)) or (
        lexical_used and generated and not lexical_bypassed
    )

    semantic_used = bool(meta.get("semantic_used", False)) or (str(match_type or "").lower() == "semantic")
    semantic_bypassed = bool(meta.get("semantic_bypassed", False)) and not generated
    semantic_context_used = semantic_used and generated and not semantic_bypassed

    return {
        "type": "example_result",
        "ok": True,
        "task": example.get("task"),
        "example_id": example.get("example_id"),
        "source_file": example.get("source_file"),
        "source_id": example.get("source_id"),

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
        "namespaces_checked": meta.get("namespaces_checked", []),

        "used_memory": used_memory,
        "source_tier": source_tier,
        "served_from": source_tier,
        "llm_bypassed": llm_bypassed,
        "generated": generated,
        "promoted_to_ram": bool(meta.get("promoted_to_ram", False)),
        "stored": meta.get("stored"),
        "stored_scopes": meta.get("stored_scopes", []),
        "store_debug": meta.get("store"),
        "store_skipped": meta.get("store_skipped"),

        "doc_signature": example.get("doc_signature") or meta.get("doc_signature"),
        "query_question_type": meta.get("question_type") or example.get("question_type"),
        "query_chunk_index": meta.get("chunk_index"),
        "query_chunk_id": meta.get("chunk_id"),
        "query_evidence_chars": (
            len(str(meta.get("query_evidence_text", "")))
            if meta.get("query_evidence_text") is not None
            else None
        ),
        "stored_evidence_chars": (
            len(str(meta.get("stored_evidence_text", "")))
            if meta.get("stored_evidence_text") is not None
            else None
        ),

        "retrieval_mode_config": getattr(cfg.memory, "retrieval_mode", None),
        "retrieval_stage": semantic_debug["retrieval_stage"] or lexical_debug["retrieval_stage"],
        "match_type": match_type,
        "retrieval_score": retrieval_score,
        "hit_before_generate": meta.get("hit_before_generate"),
        "retrieval_debug": semantic_debug["retrieval_debug"] or lexical_debug["retrieval_debug"],

        "lexical_used": lexical_used,
        "lexical_bypassed": lexical_bypassed,
        "lexical_context_used": lexical_context_used,
        "lexical_reason": lexical_debug["lexical_reason"],
        "lexical_candidate_count": lexical_debug["lexical_candidate_count"],
        "lexical_top_score": lexical_debug["lexical_top_score"],
        "lexical_top_rank": lexical_debug["lexical_top_rank"],
        "lexical_enabled_debug": lexical_debug["lexical_enabled_debug"],
        "lexical_match_type": lexical_debug["lexical_match_type"],
        "lexical_same_source": lexical_debug["lexical_same_source"],

        "semantic_used": semantic_used,
        "semantic_context_used": semantic_context_used,
        "semantic_bypassed": semantic_bypassed,
        "semantic_candidate_rank": meta.get("semantic_candidate_rank"),
        "semantic_score": meta.get("score")
        if match_type == "semantic"
        else meta.get("hit_before_generate", {}).get("score")
        if semantic_used
        else None,
        "semantic_reason": semantic_debug["semantic_reason"],
        "semantic_candidate_count": semantic_debug["semantic_candidate_count"],
        "semantic_top_score": semantic_debug["semantic_top_score"],
        "semantic_top_rank": semantic_debug["semantic_top_rank"],
        "semantic_enabled_debug": semantic_debug["semantic_enabled_debug"],

        "latency_s": total_latency_s,
        "memory_lookup_ms": memory_lookup_ms,
        "generation_ms_est": generation_ms_est,
        "timings_ms": timings_ms,

        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": (
            (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None
        ),

        "device": gen_meta.get("device"),
        "dtype": gen_meta.get("dtype"),
        "generation_backend": gen_meta.get("generation_backend"),
        "decoding_mode": gen_meta.get("decoding_mode"),
        "num_beams": gen_meta.get("num_beams"),
        "do_sample": gen_meta.get("do_sample"),
        "temperature": gen_meta.get("temperature"),
        "top_p": gen_meta.get("top_p"),
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
        "retrieved_doc_signature_match": gen_meta.get("retrieved_doc_signature_match"),
        "retrieved_evidence_chars": gen_meta.get("retrieved_evidence_chars"),
        "reduced_context_used": gen_meta.get("reduced_context_used"),
        "full_context_chars": gen_meta.get("full_context_chars"),
        "final_context_chars": gen_meta.get("final_context_chars"),
        "use_kv_cache": gen_meta.get("use_kv_cache"),
        "fallback_used": gen_meta.get("fallback_used"),
        "fallback_from": gen_meta.get("fallback_from"),
        "fallback_reason": gen_meta.get("fallback_reason"),

        "answer": answer,
        "output_text": answer,
        "ref_text": quality.get("ref_text"),
        "exact_match": quality.get("exact_match"),
        "contains_answer": quality.get("contains_answer"),
        "token_f1": quality.get("token_f1"),
        "char_f1": quality.get("char_f1"),

        "tokens_per_second": (
            (gen_meta.get("output_tokens", 0) / float(gen_meta.get("gen_time_s", 0.0)))
            if generated and float(gen_meta.get("gen_time_s", 0.0) or 0.0) > 0
            else None
        ),

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
        "source_id": example.get("source_id"),
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
            example.get("query_text")
            or example.get("question")
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

    ram = _init_ram_store(cfg)
    disk = DiskStoreSQLite(cfg.resolved_disk_store_path())
    embedder = _init_embedder(cfg)
    manager = _init_manager(cfg, ram=ram, disk=disk, embedder=embedder)
    generator = _init_generator(cfg)

    system_info = _get_system_info()
    generator_info = _safe_component_info(generator)
    embedder_info = _safe_component_info(embedder)

    n_total = 0
    n_ok = 0
    n_err = 0
    n_generated = 0
    n_memory_hits = 0
    n_exact_hits = 0
    n_lexical_context = 0
    n_lexical_bypass = 0
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
                "retrieval_config": {
                    "retrieval_mode": cfg.memory.retrieval_mode,
                    "return_memory_directly": bool(cfg.memory.return_memory_directly),
                    "promote_disk_hits_to_ram": bool(cfg.memory.promote_disk_hits_to_ram),
                    "lexical_enabled": bool(getattr(cfg.memory, "lexical_enabled", False)),
                    "lexical_context_threshold": float(getattr(cfg.memory, "lexical_threshold_context", 0.55)),
                    "lexical_direct_threshold": float(
                        getattr(
                            cfg.memory,
                            "lexical_threshold_bypass",
                            getattr(cfg.memory, "lexical_direct_threshold", 0.90),
                        )
                    ),
                    "lexical_top_k": int(getattr(cfg.memory, "lexical_top_k", 3)),
                    "prefer_same_source": bool(getattr(cfg.memory, "prefer_same_source", True)),
                    "safe_direct_reuse_tasks": list(getattr(cfg.memory, "safe_direct_reuse_tasks", ["trec"])),
                    "semantic_enabled": bool(cfg.memory.semantic_enabled),
                    "semantic_threshold_context": float(cfg.memory.semantic_threshold_context),
                    "semantic_threshold_bypass": float(cfg.memory.semantic_threshold_bypass),
                    "max_semantic_candidates": int(cfg.memory.max_semantic_candidates),
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

                # Make storage behavior explicit at executor level.
                print(
                    "[EXECUTE ANSWER META]",
                    {
                        "example_id": ex.get("example_id"),
                        "raw_query": getattr(mq, "raw_query", None),
                        "generated": meta.get("generated"),
                        "used_memory": meta.get("used_memory"),
                        "stored": meta.get("stored"),
                        "stored_scopes": meta.get("stored_scopes"),
                        "retrieval_stage": meta.get("retrieval_stage"),
                        "doc_signature": meta.get("doc_signature") or getattr(mq, "doc_signature", None),
                        "store_present": "store" in meta,
                        "store_skipped_present": "store_skipped" in meta,
                    },
                    flush=True,
                )

                meta.setdefault("doc_signature", getattr(mq, "doc_signature", None))
                meta.setdefault("source_file", getattr(mq, "source_file", None))
                meta.setdefault("source_id", getattr(mq, "source_id", None))
                meta.setdefault("chunk_index", getattr(mq, "chunk_index", None))
                meta.setdefault("chunk_id", getattr(mq, "chunk_id", None))
                meta.setdefault("question_type", getattr(mq, "question_type", None))
                meta.setdefault("query_evidence_text", getattr(mq, "evidence_text", None))

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
            if bool(record.get("lexical_context_used", False)):
                n_lexical_context += 1
            if bool(record.get("lexical_bypassed", False)):
                n_lexical_bypass += 1
            if bool(record.get("semantic_context_used", False)):
                n_semantic_context += 1
            if bool(record.get("semantic_bypassed", False)):
                n_semantic_bypass += 1
            if record.get("latency_s") is not None:
                total_latency_s_acc += float(record["latency_s"])

            logger.write(record)

        generator_info_final = _safe_component_info(generator)
        embedder_info_final = _safe_component_info(embedder)

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
                    "n_lexical_context_used": n_lexical_context,
                    "n_lexical_bypass": n_lexical_bypass,
                    "n_semantic_context_used": n_semantic_context,
                    "n_semantic_bypass": n_semantic_bypass,
                    "llm_calls_saved": n_total - n_generated,
                },
                "ram_stats_final": _ram_stats(ram),
                "disk_stats_final": _disk_stats(disk),
                "system_info": _get_system_info(),
                "resolved_runtime": {
                    "generator": generator_info_final,
                    "embedder": embedder_info_final,
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
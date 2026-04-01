# memarch/config.py
"""
Central configuration for memarch.

Design goals:
- Single place to control memory budgets, model settings, and paths
- Portable across Mac Apple Silicon and Jetson Orin
- No heavy dependencies
- Explicit defaults for reproducibility

This module is intentionally lightweight. It does not import heavy runtime
dependencies like transformers or torch. Runtime modules should consume these
configs and decide how to apply them.

This file is aligned with the current HF-based memarch stack:
- memarch.models.generator.HFGenerator
- memarch.models.embedder.HFEmbedder
- memarch.benchmarks.execute.run_benchmark
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_flag(name: str, default: str = "0") -> bool:
    return bool(int(os.environ.get(name, default)))


# -------------------------
# Paths
# -------------------------

@dataclass(frozen=True)
class PathConfig:
    base_dir: str
    artifacts_dir: str
    benchmark_runs_dir: str
    demo_dir: str
    disk_store_path: str

    @staticmethod
    def default() -> "PathConfig":
        base = os.environ.get("MEMARCH_BASE_DIR", str(Path.cwd()))
        artifacts = os.path.join(base, "artifacts")
        benchmark_runs = os.path.join(artifacts, "benchmark_runs", "memarch")
        demo_dir = os.path.join(artifacts, "demo")
        disk_store = os.path.join(artifacts, "memory_store.sqlite")

        return PathConfig(
            base_dir=base,
            artifacts_dir=artifacts,
            benchmark_runs_dir=benchmark_runs,
            demo_dir=demo_dir,
            disk_store_path=disk_store,
        )


# -------------------------
# Memory configuration
# -------------------------

@dataclass(frozen=True)
class MemoryConfig:
    """
    Controls RAM, disk persistence, and retrieval behavior.
    """

    # RAM store sizing
    ram_max_mb: int = int(os.environ.get("MEMARCH_RAM_MAX_MB", "512"))
    ram_capacity_items: int = int(os.environ.get("MEMARCH_RAM_CAPACITY_ITEMS", "512"))

    # Disk persistence
    store_on_disk: bool = _env_flag("MEMARCH_STORE_ON_DISK", "1")
    store_in_ram: bool = _env_flag("MEMARCH_STORE_IN_RAM", "1")
    enable_storage: bool = _env_flag("MEMARCH_ENABLE_STORAGE", "1")
    clear_disk_store_before_run: bool = _env_flag("MEMARCH_CLEAR_DISK_STORE_BEFORE_RUN", "0")

    # Promotion / bypass behavior
    promote_disk_hits_to_ram: bool = _env_flag("MEMARCH_PROMOTE_DISK_HITS", "1")
    return_memory_directly: bool = _env_flag("MEMARCH_RETURN_MEMORY_DIRECTLY", "1")

    # Retrieval mode
    # expected values include:
    #   exact_only
    #   lexical_context
    #   lexical_gated_direct
    #   semantic_context
    #   semantic_bypass
    #   lexical_semantic_context
    #   lexical_gated_direct_semantic_context
    retrieval_mode: str = os.environ.get("MEMARCH_RETRIEVAL_MODE", "exact_only").strip()

    # Lexical retrieval
    lexical_enabled: bool = _env_flag("MEMARCH_LEXICAL_ENABLED", "0")
    lexical_threshold_context: float = float(os.environ.get("MEMARCH_LEXICAL_THRESHOLD_CONTEXT", "0.55"))
    lexical_threshold_bypass: float = float(os.environ.get("MEMARCH_LEXICAL_THRESHOLD_BYPASS", "0.90"))
    lexical_top_k: int = int(os.environ.get("MEMARCH_LEXICAL_TOP_K", "3"))
    prefer_same_source: bool = _env_flag("MEMARCH_PREFER_SAME_SOURCE", "1")

    # Semantic retrieval
    semantic_enabled: bool = _env_flag("MEMARCH_SEMANTIC_ENABLED", "0")
    semantic_threshold_context: float = float(os.environ.get("MEMARCH_SEMANTIC_THRESHOLD_CONTEXT", "0.85"))
    semantic_threshold_bypass: float = float(os.environ.get("MEMARCH_SEMANTIC_THRESHOLD_BYPASS", "1.01"))
    max_semantic_candidates: int = int(os.environ.get("MEMARCH_MAX_SEMANTIC_CANDIDATES", "5"))

    # Embedding index
    embed_index_max_entries: int = int(os.environ.get("MEMARCH_EMBED_INDEX_SIZE", "10000"))

    # Safe direct reuse
    safe_direct_reuse_tasks_raw: str = os.environ.get("MEMARCH_SAFE_DIRECT_REUSE_TASKS", "trec")

    @property
    def safe_direct_reuse_tasks(self) -> list[str]:
        raw = self.safe_direct_reuse_tasks_raw.strip()
        if not raw:
            return []
        return [part.strip() for part in raw.split(",") if part.strip()]


# -------------------------
# Generator / model configuration
# -------------------------

@dataclass(frozen=True)
class ModelConfig:
    """
    Controls the HF generator runtime.
    """

    model_id: str = os.environ.get("MEMARCH_MODEL_ID", "distilgpt2")
    model_path: Optional[str] = os.environ.get("MEMARCH_MODEL_PATH")

    # Runtime selection
    device: str = os.environ.get("MEMARCH_DEVICE", "auto")
    dtype: str = os.environ.get("MEMARCH_DTYPE", "auto")  # auto | float16 | bfloat16 | float32

    # Input/output limits
    max_input_tokens: int = int(os.environ.get("MEMARCH_MAX_INPUT_TOKENS", "2048"))
    max_new_tokens: int = int(os.environ.get("MEMARCH_MAX_NEW_TOKENS", "256"))

    # Decoding
    decoding_mode: str = os.environ.get("MEMARCH_DECODING_MODE", "greedy")
    num_beams: int = int(os.environ.get("MEMARCH_NUM_BEAMS", "1"))
    do_sample: bool = _env_flag("MEMARCH_DO_SAMPLE", "0")
    temperature: float = float(os.environ.get("MEMARCH_TEMPERATURE", "0.2"))
    top_p: float = float(os.environ.get("MEMARCH_TOP_P", "0.95"))

    # Loading behavior
    local_files_only: bool = _env_flag("MEMARCH_LOCAL_FILES_ONLY", "0")
    use_fast_tokenizer: bool = _env_flag("MEMARCH_USE_FAST_TOKENIZER", "0")
    low_cpu_mem_usage: bool = _env_flag("MEMARCH_LOW_CPU_MEM_USAGE", "1")
    use_safetensors: bool = _env_flag("MEMARCH_USE_SAFETENSORS", "1")
    trust_remote_code: bool = _env_flag("MEMARCH_TRUST_REMOTE_CODE", "0")
    attn_implementation: str = os.environ.get("MEMARCH_ATTN_IMPLEMENTATION", "auto")
    use_kv_cache: bool = _env_flag("MEMARCH_USE_KV_CACHE", "1")

    # Fallback behavior
    cpu_fallback_on_failure: bool = _env_flag("MEMARCH_CPU_FALLBACK_ON_FAILURE", "1")

    # Prompt construction / context reduction
    include_retrieved_memory_context: bool = _env_flag("MEMARCH_INCLUDE_RETRIEVED_MEMORY_CONTEXT", "1")
    include_dataset_context: bool = _env_flag("MEMARCH_INCLUDE_DATASET_CONTEXT", "1")
    include_doc_signature: bool = _env_flag("MEMARCH_INCLUDE_DOC_SIGNATURE", "0")

    prefer_retrieved_evidence_context: bool = _env_flag("MEMARCH_PREFER_RETRIEVED_EVIDENCE_CONTEXT", "1")
    reduce_context_on_semantic_hit: bool = _env_flag("MEMARCH_REDUCE_CONTEXT_ON_SEMANTIC_HIT", "1")
    max_evidence_chars: int = int(os.environ.get("MEMARCH_MAX_EVIDENCE_CHARS", "400"))
    max_local_context_chars: int = int(os.environ.get("MEMARCH_MAX_LOCAL_CONTEXT_CHARS", "260"))
    max_full_context_chars: int = int(os.environ.get("MEMARCH_MAX_FULL_CONTEXT_CHARS", "1200"))

    prefer_local_context_for_qa: bool = _env_flag("MEMARCH_PREFER_LOCAL_CONTEXT_FOR_QA", "1")
    qa_max_output_words: int = int(os.environ.get("MEMARCH_QA_MAX_OUTPUT_WORDS", "6"))
    trec_use_few_shot: bool = _env_flag("MEMARCH_TREC_USE_FEW_SHOT", "0")
    skip_special_tokens: bool = _env_flag("MEMARCH_SKIP_SPECIAL_TOKENS", "1")

    @property
    def resolved_model_id(self) -> str:
        """
        Prefer an explicit local model path when provided.
        """
        if self.model_path:
            return self.model_path
        return self.model_id


# -------------------------
# Embedding configuration
# -------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Controls the semantic embedder runtime.
    """

    model_id: str = os.environ.get(
        "MEMARCH_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    model_path: Optional[str] = os.environ.get("MEMARCH_EMBED_MODEL_PATH")

    device: str = os.environ.get("MEMARCH_EMBED_DEVICE", "auto")
    dtype: str = os.environ.get("MEMARCH_EMBED_DTYPE", "auto")

    max_length: int = int(os.environ.get("MEMARCH_EMBED_MAX_LENGTH", "512"))
    batch_size: int = int(os.environ.get("MEMARCH_EMBED_BATCH_SIZE", "16"))
    normalize: bool = _env_flag("MEMARCH_EMBED_NORMALIZE", "1")

    local_files_only: bool = _env_flag("MEMARCH_EMBED_LOCAL_FILES_ONLY", "0")
    use_fast_tokenizer: bool = _env_flag("MEMARCH_EMBED_USE_FAST_TOKENIZER", "0")
    low_cpu_mem_usage: bool = _env_flag("MEMARCH_EMBED_LOW_CPU_MEM_USAGE", "1")
    use_safetensors: bool = _env_flag("MEMARCH_EMBED_USE_SAFETENSORS", "1")
    trust_remote_code: bool = _env_flag("MEMARCH_EMBED_TRUST_REMOTE_CODE", "0")
    attn_implementation: str = os.environ.get("MEMARCH_EMBED_ATTN_IMPLEMENTATION", "auto")

    cpu_fallback_on_failure: bool = _env_flag("MEMARCH_EMBED_CPU_FALLBACK_ON_FAILURE", "1")

    # For logging / validation only
    embedding_dim: int = int(os.environ.get("MEMARCH_EMBED_DIM", "384"))

    @property
    def resolved_model_id(self) -> str:
        if self.model_path:
            return self.model_path
        return self.model_id


# -------------------------
# Evaluation / logging configuration
# -------------------------

@dataclass(frozen=True)
class EvalConfig:
    log_query_text: bool = _env_flag("MEMARCH_LOG_QUERY_TEXT", "1")
    max_examples: Optional[int] = (
        int(os.environ["MEMARCH_MAX_EXAMPLES"])
        if "MEMARCH_MAX_EXAMPLES" in os.environ
        else None
    )

    log_memory_hits: bool = _env_flag("MEMARCH_LOG_MEMORY_HITS", "1")
    log_cache_promotions: bool = _env_flag("MEMARCH_LOG_CACHE_PROMOTIONS", "1")
    write_summary_json: bool = _env_flag("MEMARCH_WRITE_SUMMARY_JSON", "1")


# -------------------------
# Feature flags
# -------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """
    Enables/disables major architectural components.
    """

    enable_tier1: bool = _env_flag("MEMARCH_ENABLE_TIER1", "1")
    enable_tier2: bool = _env_flag("MEMARCH_ENABLE_TIER2", "1")
    enable_semantic: bool = _env_flag("MEMARCH_ENABLE_SEMANTIC", "0")
    enable_reranker: bool = _env_flag("MEMARCH_ENABLE_RERANKER", "0")


# -------------------------
# Top-level config
# -------------------------

@dataclass(frozen=True)
class MemArchConfig:
    paths: PathConfig
    memory: MemoryConfig
    model: ModelConfig
    embedding: EmbeddingConfig
    eval: EvalConfig
    features: FeatureConfig

    @staticmethod
    def default() -> "MemArchConfig":
        return MemArchConfig(
            paths=PathConfig.default(),
            memory=MemoryConfig(),
            model=ModelConfig(),
            embedding=EmbeddingConfig(),
            eval=EvalConfig(),
            features=FeatureConfig(),
        )
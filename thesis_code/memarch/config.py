# memarch/config.py
"""
Central configuration for memarch.

Design goals:
- Single place to control memory budgets, model settings, and paths
- Portable across Mac Apple Silicon and Jetson Orin
- No heavy dependencies
- Explicit defaults for reproducibility (committee-friendly)

This config does NOT automatically detect hardware-specific optimal values.
You can override via environment variables or CLI in scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# -------------------------
# Paths
# -------------------------

@dataclass(frozen=True)
class PathConfig:
    base_dir: str
    artifacts_dir: str
    runs_dir: str
    disk_store_path: str

    @staticmethod
    def default() -> "PathConfig":
        base = os.environ.get("MEMARCH_BASE_DIR", str(Path.cwd()))
        artifacts = os.path.join(base, "artifacts")
        runs = os.path.join(artifacts, "runs")
        disk_store = os.path.join(artifacts, "memory_store.sqlite")

        return PathConfig(
            base_dir=base,
            artifacts_dir=artifacts,
            runs_dir=runs,
            disk_store_path=disk_store,
        )


# -------------------------
# Memory configuration
# -------------------------

@dataclass(frozen=True)
class MemoryConfig:
    """
    Controls Tier1 (RAM), Tier2 (disk), and Phase2 embedding index.
    """

    # Tier 1 RAM budget (MB)
    ram_max_mb: int = int(os.environ.get("MEMARCH_RAM_MAX_MB", "512"))

    # Promote disk hits to RAM cache
    promote_disk_hits_to_ram: bool = True

    # Phase 1 behavior: return memory hits directly (bypass LLM)
    return_memory_directly: bool = True

    # Phase 2: semantic retrieval enable
    enable_semantic_retrieval: bool = bool(
        int(os.environ.get("MEMARCH_ENABLE_SEMANTIC", "0"))
    )

    # Embedding index size (Phase 2)
    embed_index_max_entries: int = int(
        os.environ.get("MEMARCH_EMBED_INDEX_SIZE", "10000")
    )

    # Similarity threshold for semantic hits
    semantic_similarity_threshold: float = float(
        os.environ.get("MEMARCH_SIM_THRESHOLD", "0.85")
    )


# -------------------------
# Model configuration
# -------------------------

@dataclass(frozen=True)
class ModelConfig:
    """
    Phase 1 target:
      - Mistral 7B Instruct (quantized)
      - llama.cpp backend by default
    """

    # Logical model identifier
    model_id: str = "mistral-7b-instruct"

    # Backend used to run the model
    backend: str = "llama_cpp"  # options: llama_cpp, mlx, remote_http

    # Path to local model (GGUF for llama.cpp)
    model_path: Optional[str] = None

    # Generation parameters
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    context_window: int = 4096

    # Quantization descriptor (for logging only)
    quantization: str = "Q4_K_M"

    # Runtime resolution (filled at runtime, not user-set)
    device: Optional[str] = None
    dtype: Optional[str] = None

    @staticmethod
    def from_env() -> "ModelConfig":
        return ModelConfig(
            model_id=os.environ.get("MEMARCH_MODEL_ID", "mistral-7b-instruct"),
            backend=os.environ.get("MEMARCH_BACKEND", "llama_cpp"),
            model_path=os.environ.get("MEMARCH_MODEL_PATH"),
            temperature=float(os.environ.get("MEMARCH_TEMPERATURE", "0.2")),
            top_p=float(os.environ.get("MEMARCH_TOP_P", "0.95")),
            max_tokens=int(os.environ.get("MEMARCH_MAX_TOKENS", "512")),
            context_window=int(os.environ.get("MEMARCH_CONTEXT_WINDOW", "4096")),
            quantization=os.environ.get("MEMARCH_QUANT", "Q4_K_M"),
        )


# -------------------------
# Embedding configuration (Phase 2)
# -------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Controls semantic retrieval (Phase 2).
    """

    model_name: str = os.environ.get(
        "MEMARCH_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    embedding_dim: int = int(os.environ.get("MEMARCH_EMBED_DIM", "384"))

    normalize: bool = True

    backend: str = os.environ.get(
        "MEMARCH_EMBED_BACKEND",
        "sentence_transformers",  # future: "onnx", "tensorrt"
    )


# -------------------------
# Evaluation configuration
# -------------------------

@dataclass(frozen=True)
class EvalConfig:
    log_query_text: bool = True
    max_examples: Optional[int] = None

    # Logging granularity
    log_memory_hits: bool = True
    log_cache_promotions: bool = True


# -------------------------
# Feature flags
# -------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """
    Enables/disables major architectural components.
    """

    enable_tier1: bool = True
    enable_tier2: bool = True
    enable_semantic: bool = bool(
        int(os.environ.get("MEMARCH_ENABLE_SEMANTIC", "0"))
    )

    # Future: reranking / hybrid retrieval
    enable_reranker: bool = False


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
            model=ModelConfig.from_env(),
            embedding=EmbeddingConfig(),
            eval=EvalConfig(),
            features=FeatureConfig(),
        )
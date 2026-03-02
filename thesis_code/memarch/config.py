# memarch/config.py
"""
Central configuration for memarch.

Design goals:
- Single place to control memory budgets, model settings, and paths
- Portable across Mac Apple Silicon and Jetson Orin Nano
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
    # Tier 1 RAM budget (MB)
    ram_max_mb: int = 512  # reasonable for Mac; tune lower for Jetson if needed

    # Tier 2 disk path is handled in PathConfig
    promote_disk_hits_to_ram: bool = True

    # Whether to return memory hits directly (Phase 1: True)
    return_memory_directly: bool = True

    # Embedding index (Phase 2)
    embed_index_max_entries: int = 10_000


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
    model_id: str = "mistral-7b-instruct"
    backend: str = "llama_cpp"  # or "mlx", "remote_http"
    model_path: Optional[str] = None  # path to GGUF file

    # Generation parameters (conservative defaults)
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    context_window: int = 4096

    # Quantization descriptor (informational/logging)
    quantization: str = "Q4_K_M"

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
# Evaluation configuration
# -------------------------

@dataclass(frozen=True)
class EvalConfig:
    log_query_text: bool = True
    max_examples: Optional[int] = None


# -------------------------
# Top-level config
# -------------------------

@dataclass(frozen=True)
class MemArchConfig:
    paths: PathConfig
    memory: MemoryConfig
    model: ModelConfig
    eval: EvalConfig

    @staticmethod
    def default() -> "MemArchConfig":
        return MemArchConfig(
            paths=PathConfig.default(),
            memory=MemoryConfig(),
            model=ModelConfig.from_env(),
            eval=EvalConfig(),
        )
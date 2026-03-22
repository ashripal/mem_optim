# memarch/models/embedder.py
"""
Embedding backend for memarch using Hugging Face Transformers.

Phase 1 semantic retrieval goals:
- provide a stable embedding interface for semantic memory lookup
- keep embedding logic separate from memory policy / storage logic
- support constrained and heterogeneous devices:
    - MacBook (CPU / MPS if available)
    - Jetson Orin Nano / AGX Orin (CPU / CUDA if available)
    - other Linux edge devices

Design choices:
- Uses AutoTokenizer + AutoModel from transformers
- Uses mean pooling over the last hidden state
- Supports optional L2 normalization
- Returns plain Python lists for portability with SQLite / JSON-backed storage

Notes:
- This file does NOT build a vector index. That belongs in:
    memarch/memory/embed_index.py
- This file does NOT implement caching. That belongs in:
    memarch/memory/manager.py or a higher-level embedding cache if needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


Vector = List[float]


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _select_device(device: str = "auto") -> str:
    """
    Device policy for embeddings.

    Priority when device='auto':
      1. CUDA
      2. MPS
      3. CPU
    """
    device = (device or "auto").lower().strip()

    if device == "auto":
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if device == "cuda":
        if not _cuda_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available.")
        return "cuda"

    if device == "mps":
        if not _mps_available():
            raise RuntimeError("Requested device='mps' but MPS is not available.")
        return "mps"

    if device == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported device: {device}")


@dataclass(frozen=True)
class EmbedderConfig:
    """
    Configuration for the embedding backend.

    Default model:
      sentence-transformers/all-MiniLM-L6-v2

    This is a practical default because it is:
    - widely used
    - small enough for constrained devices
    - effective for semantic similarity tasks
    """
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"

    # Tokenization / batching
    max_length: int = 512
    batch_size: int = 16

    # Output behavior
    normalize: bool = True

    # Loading behavior
    local_files_only: bool = False
    use_fast_tokenizer: bool = False

    # Runtime behavior
    cpu_fallback_on_failure: bool = True


class HFEmbedder:
    """
    Lightweight embedding wrapper around a Hugging Face encoder model.

    Public API:
      - embed(text) -> List[float]
      - embed_batch(texts) -> List[List[float]]

    Backward-compatible aliases:
      - encode(text)
      - encode_batch(texts)
    """

    def __init__(self, cfg: Optional[EmbedderConfig] = None) -> None:
        self.cfg = cfg or EmbedderConfig()
        self.device = _select_device(self.cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            use_fast=self.cfg.use_fast_tokenizer,
            local_files_only=self.cfg.local_files_only,
        )

        self.model = AutoModel.from_pretrained(
            self.cfg.model_id,
            local_files_only=self.cfg.local_files_only,
        )
        self.model.eval()
        self.model.to(self.device)

        self.last_batch_meta: Optional[Dict[str, object]] = None

    @staticmethod
    def _sanitize_text(text: Optional[str]) -> str:
        """
        Convert inputs into deterministic strings for embedding.

        This keeps behavior stable across pipeline components and prevents
        accidental None handling issues.
        """
        if text is None:
            return ""
        return str(text).strip()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling over non-padding tokens.

        Args:
            last_hidden_state: [batch, seq_len, hidden_dim]
            attention_mask:    [batch, seq_len]

        Returns:
            pooled: [batch, hidden_dim]
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeddings = last_hidden_state * mask
        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _record_batch_meta(
        self,
        *,
        batch_size: int,
        truncated_count: int,
        tokenize_time_s: float,
        embed_time_s: float,
        backend_used: str,
    ) -> None:
        meta: Dict[str, object] = {
            "device": self.device,
            "batch_size": batch_size,
            "truncated_count": truncated_count,
            "tokenize_time_s": tokenize_time_s,
            "embed_time_s": embed_time_s,
            "normalize": self.cfg.normalize,
            "embedding_dim": self.embedding_dim(),
            "backend_used": backend_used,
        }

        if self.device == "cuda":
            try:
                meta["cuda_device_name"] = torch.cuda.get_device_name(0)
                meta["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 3)
                meta["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 3)
            except Exception:
                pass

        self.last_batch_meta = meta

    def _embed_batch_once(self, texts: Sequence[str]) -> List[Vector]:
        cleaned_texts = [self._sanitize_text(t) for t in texts]

        tok_t0 = time.time()
        enc = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        tokenize_time_s = time.time() - tok_t0

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        truncated_count = int((attention_mask.sum(dim=1) >= self.cfg.max_length).sum().item())

        embed_t0 = time.time()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            last_hidden_state = outputs.last_hidden_state
            pooled = self._mean_pool(last_hidden_state, attention_mask)

            if self.cfg.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            batch_vectors = pooled.detach().cpu().tolist()
        embed_time_s = time.time() - embed_t0

        self._record_batch_meta(
            batch_size=len(cleaned_texts),
            truncated_count=truncated_count,
            tokenize_time_s=tokenize_time_s,
            embed_time_s=embed_time_s,
            backend_used="hf_forward",
        )

        return batch_vectors

    def embed(self, text: str) -> Vector:
        """
        Embed a single text into one vector.
        """
        out = self.embed_batch([text])
        return out[0]

    def embed_batch(self, texts: Sequence[str]) -> List[Vector]:
        """
        Embed a batch of texts.

        Returns:
            List of embeddings as plain Python lists.
        """
        if texts is None:
            raise ValueError("texts must not be None")
        if len(texts) == 0:
            return []

        cleaned_texts = [self._sanitize_text(t) for t in texts]
        all_vectors: List[Vector] = []

        for start in range(0, len(cleaned_texts), self.cfg.batch_size):
            batch = cleaned_texts[start : start + self.cfg.batch_size]

            try:
                batch_vectors = self._embed_batch_once(batch)
            except RuntimeError as e:
                if self.device in {"cuda", "mps"} and self.cfg.cpu_fallback_on_failure:
                    self.model.to("cpu")
                    self.device = "cpu"
                    batch_vectors = self._embed_batch_once(batch)
                else:
                    raise RuntimeError(
                        f"HFEmbedder failed on device={self.device}: {type(e).__name__}: {e}"
                    ) from e

            all_vectors.extend(batch_vectors)

        return all_vectors

    # Backward-compatible aliases
    def encode(self, text: str) -> Vector:
        return self.embed(text)

    def encode_batch(self, texts: Sequence[str]) -> List[Vector]:
        return self.embed_batch(texts)

    def embedding_dim(self) -> int:
        """
        Return the embedding dimension for this model.

        Uses the model config hidden size, which is standard for encoder outputs.
        """
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Could not determine embedding dimension from model config.")
        return int(hidden_size)

    @staticmethod
    def embedding_norm(vector: Sequence[float]) -> float:
        """
        Compute the L2 norm of a vector.

        Useful for:
        - optional storage/debugging in MemoryItem.embedding_norm
        - validation if normalization settings change
        """
        if vector is None:
            raise ValueError("vector must not be None")
        return float(sqrt(sum(float(x) * float(x) for x in vector)))

    def info(self) -> Dict[str, object]:
        """
        Lightweight metadata for logging/debugging.
        """
        return {
            "model_id": self.cfg.model_id,
            "device": self.device,
            "max_length": self.cfg.max_length,
            "batch_size": self.cfg.batch_size,
            "normalize": self.cfg.normalize,
            "embedding_dim": self.embedding_dim(),
            "use_fast_tokenizer": self.cfg.use_fast_tokenizer,
            "cpu_fallback_on_failure": self.cfg.cpu_fallback_on_failure,
            "last_batch_meta": self.last_batch_meta,
        }


# Simple canonical interface for the rest of memarch
Embedder = HFEmbedder
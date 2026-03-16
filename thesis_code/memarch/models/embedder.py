# memarch/models/embedder.py
"""
Embedding backend for memarch using Hugging Face Transformers.

Why this exists:
- Phase 1 demo does not require semantic retrieval, but the architecture should already
  have a stable embedding interface for Phase 2.
- This wrapper keeps embedding logic separate from memory policy / storage logic.
- It is designed to work on:
    - MacBook (CPU / MPS if available)
    - Jetson Orin Nano (CPU / CUDA if available)
    - Other Linux edge devices

Design choices:
- Uses AutoTokenizer + AutoModel from transformers
- Uses mean pooling over the last hidden state
- Supports optional L2 normalization
- Returns plain Python lists for maximum portability with the rest of memarch

Notes:
- This file intentionally does NOT build a vector index. That belongs in:
    memarch/memory/embed_index.py
- This file intentionally does NOT implement caching. That belongs in:
    memarch/memory/manager.py or a higher-level embedding cache if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


Vector = List[float]


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _select_device(device: str = "auto") -> str:
    """
    Device policy for embeddings.

    Priority when device='auto':
      1. CUDA
      2. MPS
      3. CPU

    This keeps behavior portable and predictable.
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
    - easy to use with transformers directly
    """
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"

    # Tokenization / batching
    max_length: int = 512
    batch_size: int = 16

    # Output behavior
    normalize: bool = True

    # Local files only mode is useful in constrained/offline environments.
    local_files_only: bool = False


class HFEmbedder:
    """
    Lightweight embedding wrapper around a Hugging Face encoder model.

    Public API:
      - encode(text) -> List[float]
      - encode_batch(texts) -> List[List[float]]

    Example:
        embedder = HFEmbedder()
        v = embedder.encode("What is systems engineering?")
    """

    def __init__(self, cfg: Optional[EmbedderConfig] = None) -> None:
        self.cfg = cfg or EmbedderConfig()
        self.device = _select_device(self.cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            use_fast=True,
            local_files_only=self.cfg.local_files_only,
        )

        self.model = AutoModel.from_pretrained(
            self.cfg.model_id,
            local_files_only=self.cfg.local_files_only,
        )
        self.model.eval()
        self.model.to(self.device)

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

    def encode(self, text: str) -> Vector:
        """
        Encode a single text into one embedding vector.
        """
        if text is None:
            raise ValueError("text must not be None")
        out = self.encode_batch([text])
        return out[0]

    def encode_batch(self, texts: Sequence[str]) -> List[Vector]:
        """
        Encode a batch of texts.

        Returns:
            List of embeddings as plain Python lists.
        """
        if texts is None:
            raise ValueError("texts must not be None")
        if len(texts) == 0:
            return []

        # Convert to strings defensively
        texts = ["" if t is None else str(t) for t in texts]

        all_vectors: List[Vector] = []

        for start in range(0, len(texts), self.cfg.batch_size):
            batch = texts[start : start + self.cfg.batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Standard transformer encoder output
                last_hidden_state = outputs.last_hidden_state
                pooled = self._mean_pool(last_hidden_state, attention_mask)

                if self.cfg.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)

                # Move to CPU and convert to Python-native lists
                batch_vectors = pooled.detach().cpu().tolist()
                all_vectors.extend(batch_vectors)

        return all_vectors

    def embedding_dim(self) -> int:
        """
        Return the embedding dimension for this model.

        Uses the model config hidden size, which is standard for encoder outputs.
        """
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Could not determine embedding dimension from model config.")
        return int(hidden_size)

    def info(self) -> dict:
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
        }
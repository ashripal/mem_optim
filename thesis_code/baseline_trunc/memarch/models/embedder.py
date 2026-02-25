# tbaseline_mem/models/embedder.py
"""
Embedder for "memory architecture" experiments.

Goal:
- Produce embeddings for:
  (a) user queries (questions)
  (b) (optional) stored QA keys like "question" or "question+short_context"
- Used by similarity routing to decide:
  - cache hit (reuse answer)
  - cache-augmented generation (retrieve prior QAs as support)
  - full generation (no relevant memory)

Design notes:
- Uses SentenceTransformers when available (recommended).
- Falls back to a plain Transformers mean-pooling encoder if needed.
- Supports MPS on Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch is required for embedder.py") from e


@dataclass
class EmbedderConfig:
    # Best default for quality/speed on CPU/MPS:
    # - "sentence-transformers/all-MiniLM-L6-v2" is fast, compact, strong for similarity
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "mps"  # "mps" | "cpu" | "cuda"
    batch_size: int = 16
    normalize: bool = True  # cosine similarity becomes dot product if normalized


class Embedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.device = self._pick_device(cfg.device)

        self._st_model = None
        self._hf_tokenizer = None
        self._hf_model = None

        # Prefer SentenceTransformers if installed
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(cfg.model_id, device=str(self.device))
        except Exception:
            # Fallback: plain HF encoder
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            self._hf_tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
            self._hf_model = AutoModel.from_pretrained(cfg.model_id).to(self.device)
            self._hf_model.eval()

    @staticmethod
    def _pick_device(requested: str) -> torch.device:
        requested = (requested or "").lower()
        if requested == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        """
        Returns: float32 array of shape [N, D]
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self._st_model is not None:
            vecs = self._st_model.encode(
                list(texts),
                batch_size=self.cfg.batch_size,
                normalize_embeddings=self.cfg.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vecs.astype(np.float32)

        # HF fallback mean-pooling
        assert self._hf_model is not None and self._hf_tokenizer is not None

        all_vecs: List[np.ndarray] = []
        bs = max(1, int(self.cfg.batch_size))
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = list(texts[i : i + bs])
                tok = self._hf_tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                tok = {k: v.to(self.device) for k, v in tok.items()}
                out = self._hf_model(**tok)
                last_hidden = out.last_hidden_state  # [B, T, H]
                attn = tok["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                masked = last_hidden * attn
                summed = masked.sum(dim=1)  # [B, H]
                counts = attn.sum(dim=1).clamp(min=1)  # [B, 1]
                mean = summed / counts  # [B, H]

                vec = mean.detach().float().cpu().numpy()
                if self.cfg.normalize:
                    norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
                    vec = vec / norms
                all_vecs.append(vec.astype(np.float32))

        return np.vstack(all_vecs)
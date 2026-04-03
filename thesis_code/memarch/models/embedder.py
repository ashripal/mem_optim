from __future__ import annotations

"""
Embedding backend for memarch using Hugging Face Transformers.

Role in the verified paraphrase reuse system:
- provide stable embeddings for semantic candidate retrieval
- stay separate from memory policy / storage / verification logic
- remain lightweight and portable across:
    - MacBook (CPU / MPS)
    - Jetson Orin Nano / AGX Orin (CPU / CUDA)
    - other Linux edge devices

Design choices:
- Uses AutoTokenizer + AutoModel from transformers
- Uses mean pooling over the last hidden state
- Supports optional L2 normalization
- Returns plain Python lists for portability with SQLite / JSON-backed storage

Important non-goals:
- This file does NOT build a vector index
- This file does NOT decide whether a semantic hit is safe to bypass
- This file does NOT cache embeddings beyond the loaded model/tokenizer
"""

import os
import time
from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


Vector = List[float]


# =============================================================================
# Device helpers
# =============================================================================

def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _looks_like_local_model_path(model_id: str) -> bool:
    """
    Heuristic for distinguishing a local model path from a HF repo id.
    """
    model_id = str(model_id or "").strip()
    if not model_id:
        return False
    if os.path.isabs(model_id):
        return True
    if model_id.startswith(".") or model_id.startswith("~"):
        return True
    if os.path.sep in model_id:
        return True
    return os.path.exists(os.path.expanduser(model_id))


def _resolve_model_source(model_id: str) -> str:
    """
    Resolve the configured model source to either:
    - absolute local path
    - or the original HF model id string
    """
    if _looks_like_local_model_path(model_id):
        return os.path.abspath(os.path.expanduser(model_id))
    return str(model_id)


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


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class EmbedderConfig:
    """
    Configuration for the embedding backend.

    Default model:
      sentence-transformers/all-MiniLM-L6-v2

    This remains a practical default because it is:
    - widely used
    - small enough for constrained devices
    - effective for semantic similarity tasks

    Notes for paraphrase reuse:
    - normalized embeddings are recommended because semantic scoring in the
      index/retrieval layer often assumes cosine-like similarity behavior
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
    torch_dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    trust_remote_code: bool = False
    attn_implementation: str = "auto"  # auto | eager | sdpa | flash_attention_2

    # Runtime behavior
    cpu_fallback_on_failure: bool = True


# =============================================================================
# HF embedder
# =============================================================================

class HFEmbedder:
    """
    Lightweight embedding wrapper around a Hugging Face encoder model.

    Public API:
      - embed(text) -> List[float]
      - embed_batch(texts) -> List[List[float]]

    Backward-compatible aliases:
      - encode(text)
      - encode_batch(texts)

    Design note:
    This class should stay focused on representation learning only.
    The verified paraphrase reuse strategy is implemented later by:
    - candidate retrieval
    - policy verification
    - manager-level decision logic
    """

    def __init__(self, cfg: Optional[EmbedderConfig] = None) -> None:
        self.cfg = cfg or EmbedderConfig()
        self._validate_config()

        self.model_source = _resolve_model_source(self.cfg.model_id)
        self.device = _select_device(self.cfg.device)
        self.model_dtype = self._resolve_torch_dtype(self.cfg.torch_dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            use_fast=self.cfg.use_fast_tokenizer,
            local_files_only=self.cfg.local_files_only,
            trust_remote_code=self.cfg.trust_remote_code,
        )

        self.model = self._load_model(self.model_source, self.device, self.model_dtype)

        # Metadata from the most recent batch. Helpful for tests/debugging.
        self.last_batch_meta: Optional[Dict[str, object]] = None

    # -------------------------------------------------------------------------
    # Validation / model loading
    # -------------------------------------------------------------------------

    def _validate_config(self) -> None:
        if int(self.cfg.max_length) <= 0:
            raise ValueError("max_length must be > 0")
        if int(self.cfg.batch_size) <= 0:
            raise ValueError("batch_size must be > 0")

        attn_impl = (self.cfg.attn_implementation or "auto").strip()
        if attn_impl not in {"auto", "eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "attn_implementation must be one of: auto, eager, sdpa, flash_attention_2"
            )

        dtype_name = (self.cfg.torch_dtype or "auto").lower().strip()
        if dtype_name not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("torch_dtype must be one of: auto, float16, bfloat16, float32")

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str, device: str):
        """
        Resolve the torch dtype for the current device.

        Current default behavior:
        - CUDA: float16 by default
        - MPS: float16 by default
        - CPU: float32 by default
        """
        dtype_name = (dtype_name or "auto").lower().strip()

        if dtype_name == "auto":
            if device == "cuda":
                return torch.float16
            if device == "mps":
                return torch.float16
            return torch.float32

        if dtype_name == "float16":
            return torch.float32 if device == "cpu" else torch.float16
        if dtype_name == "bfloat16":
            if device == "cuda":
                return torch.bfloat16
            return torch.float32
        if dtype_name == "float32":
            return torch.float32

        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")

    def _model_load_kwargs(self, device: str, dtype):
        """
        Collect model loading kwargs in one place for easier maintenance.
        """
        kwargs = {
            "local_files_only": self.cfg.local_files_only,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": bool(self.cfg.low_cpu_mem_usage),
            "trust_remote_code": bool(self.cfg.trust_remote_code),
        }

        if bool(self.cfg.use_safetensors):
            kwargs["use_safetensors"] = True

        attn_impl = (self.cfg.attn_implementation or "auto").strip()
        if attn_impl != "auto":
            kwargs["attn_implementation"] = attn_impl

        return kwargs

    @staticmethod
    def _move_model_to_device(model, device: str, dtype) -> None:
        """
        Move a model to device as defensively as possible.

        Real HF models usually accept .to(device=..., dtype=...).
        Some fakes used in unit tests only accept .to(device) or .to("cpu").
        """
        try:
            model.to(device=device, dtype=dtype)
            return
        except TypeError:
            pass

        try:
            model.to(device)
            return
        except TypeError:
            pass

        try:
            model.to(device=device)
            return
        except TypeError:
            pass

        if dtype is not None:
            try:
                model.to(dtype=dtype)
                return
            except TypeError:
                pass

        model.to(device)

    @staticmethod
    def _move_tensor_to_device(t: torch.Tensor, device: str) -> torch.Tensor:
        """
        Move tensors defensively across device types.

        non_blocking is useful on CUDA but some backends / test doubles may not
        like extra kwargs, so we degrade cleanly.
        """
        try:
            return t.to(device, non_blocking=True)
        except TypeError:
            return t.to(device)

    def _load_model(self, model_source: str, device: str, dtype):
        """
        Load the encoder model and move it to the selected device.
        """
        model = AutoModel.from_pretrained(
            model_source,
            **self._model_load_kwargs(device, dtype),
        )
        model.eval()
        self._move_model_to_device(model, device, dtype)
        return model

    def _reload_model_for_device(self, device: str) -> None:
        """
        Reload the model on a different device, mainly for CPU fallback after
        CUDA/MPS runtime failure.
        """
        self.model_dtype = self._resolve_torch_dtype(self.cfg.torch_dtype, device)

        try:
            del self.model
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self.model = self._load_model(self.model_source, device, self.model_dtype)
        self.device = device

    # -------------------------------------------------------------------------
    # Text / pooling helpers
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Metadata helpers
    # -------------------------------------------------------------------------

    def _record_batch_meta(
        self,
        *,
        batch_size: int,
        truncated_count: int,
        tokenize_time_s: float,
        embed_time_s: float,
        backend_used: str,
        fallback_used: bool = False,
        fallback_from: Optional[str] = None,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """
        Record batch-level runtime metadata for debugging and analysis.
        """
        meta: Dict[str, object] = {
            "device": self.device,
            "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
            "batch_size": batch_size,
            "truncated_count": truncated_count,
            "tokenize_time_s": tokenize_time_s,
            "embed_time_s": embed_time_s,
            "normalize": self.cfg.normalize,
            "embedding_dim": self.embedding_dim(),
            "backend_used": backend_used,
            "fallback_used": bool(fallback_used),
            "fallback_from": fallback_from,
            "fallback_reason": fallback_reason,
        }

        if self.device == "cuda":
            try:
                meta["cuda_device_name"] = torch.cuda.get_device_name(0)
                meta["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 3)
                meta["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 3)
            except Exception:
                pass

        self.last_batch_meta = meta

    # -------------------------------------------------------------------------
    # Core embedding methods
    # -------------------------------------------------------------------------

    def _embed_batch_once(self, texts: Sequence[str]) -> List[Vector]:
        """
        Embed one batch on the current device.

        Returns plain Python lists for portability with:
        - RAM store
        - SQLite disk store
        - JSON logs
        """
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

        input_ids = self._move_tensor_to_device(enc["input_ids"], self.device)
        attention_mask = self._move_tensor_to_device(enc["attention_mask"], self.device)

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
                    fallback_from = self.device
                    fallback_reason = f"{type(e).__name__}: {e}"

                    self._reload_model_for_device("cpu")
                    batch_vectors = self._embed_batch_once(batch)

                    if isinstance(self.last_batch_meta, dict):
                        self.last_batch_meta["fallback_used"] = True
                        self.last_batch_meta["fallback_from"] = fallback_from
                        self.last_batch_meta["fallback_reason"] = fallback_reason
                else:
                    raise RuntimeError(
                        f"HFEmbedder failed on device={self.device}: {type(e).__name__}: {e}"
                    ) from e

            all_vectors.extend(batch_vectors)

        return all_vectors

    # -------------------------------------------------------------------------
    # Backward-compatible aliases
    # -------------------------------------------------------------------------

    def encode(self, text: str) -> Vector:
        return self.embed(text)

    def encode_batch(self, texts: Sequence[str]) -> List[Vector]:
        return self.embed_batch(texts)

    # -------------------------------------------------------------------------
    # Optional future helper
    # -------------------------------------------------------------------------

    def embed_query_text(self, raw_query: str, canonical_query: Optional[str] = None) -> Vector:
        """
        Helper for future migration if you want to embed canonical query text
        instead of raw query text.

        Current behavior:
        - if canonical_query is provided and non-empty, embed it
        - otherwise embed raw_query

        This keeps the embedder ready for a future canonical-intent layer
        without forcing that change today.
        """
        text = self._sanitize_text(canonical_query) or self._sanitize_text(raw_query)
        return self.embed(text)

    # -------------------------------------------------------------------------
    # Introspection / math helpers
    # -------------------------------------------------------------------------

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
            "model_source": self.model_source,
            "device": self.device,
            "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
            "max_length": self.cfg.max_length,
            "batch_size": self.cfg.batch_size,
            "normalize": self.cfg.normalize,
            "embedding_dim": self.embedding_dim(),
            "use_fast_tokenizer": self.cfg.use_fast_tokenizer,
            "local_files_only": self.cfg.local_files_only,
            "low_cpu_mem_usage": self.cfg.low_cpu_mem_usage,
            "use_safetensors": self.cfg.use_safetensors,
            "attn_implementation": self.cfg.attn_implementation,
            "cpu_fallback_on_failure": self.cfg.cpu_fallback_on_failure,
            "last_batch_meta": self.last_batch_meta,
        }


# Backward-compatible alias used elsewhere in the project.
Embedder = HFEmbedder
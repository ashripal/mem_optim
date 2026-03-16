# memarch/models/generator.py
"""
Generation backend for memarch using Hugging Face Transformers.

Purpose:
- Provide a stable generation interface for MemoryManager
- Make it easy to switch from the fake demo generator to a real local model
- Keep prompt construction explicit so we can prove:
    1) dataset context is used
    2) optional retrieved memory can be injected
    3) behavior is portable across Mac / Jetson / other constrained devices

Phase 1 design:
- Deterministic-ish prompt builder
- Local text generation with transformers
- Returns:
    (answer_text, Provenance, QualitySignals)

Important:
- This module does NOT implement caching or routing.
- MemoryManager decides whether generation is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memarch.memory.schema import MemoryHit, MemoryQuery, Provenance, QualitySignals


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _select_device(device: str = "auto") -> str:
    """
    Device selection policy for generation.

    device='auto' priority:
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
class GeneratorConfig:
    """
    Configuration for the generation backend.

    Default model is intentionally a placeholder-compatible setting.
    For your thesis setup, you will likely override this with a real local model path
    or a small instruct model while iterating.

    Example overrides:
      model_id="mistralai/Mistral-7B-Instruct-v0.2"
      model_id="/path/to/local/model"
    """
    model_id: str = "distilgpt2"
    device: str = "auto"

    # Generation behavior
    max_input_length: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = False

    # Loading behavior
    local_files_only: bool = False
    torch_dtype: str = "auto"   # "auto" | "float16" | "bfloat16" | "float32"

    # Prompt behavior
    include_retrieved_memory_context: bool = True
    include_dataset_context: bool = True

    # Cleanup
    skip_special_tokens: bool = True


class HFGenerator:
    """
    Hugging Face text generation wrapper.

    Public API:
      generate(mq: MemoryQuery, retrieved: Optional[MemoryHit]) -> (text, provenance, quality)

    This matches what MemoryManager expects.
    """

    def __init__(self, cfg: Optional[GeneratorConfig] = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self.device = _select_device(self.cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            use_fast=True,
            local_files_only=self.cfg.local_files_only,
        )

        # Many causal models need a pad token for batched generation
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = self._resolve_torch_dtype(self.cfg.torch_dtype, self.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            local_files_only=self.cfg.local_files_only,
            torch_dtype=dtype,
        )
        self.model.eval()
        self.model.to(self.device)

        # Stores last prompt for debugging / demo proof of context injection
        self.last_prompt: Optional[str] = None

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str, device: str):
        dtype_name = (dtype_name or "auto").lower().strip()

        if dtype_name == "auto":
            if device == "cuda":
                return torch.float16
            if device == "mps":
                return torch.float16
            return torch.float32

        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float32":
            return torch.float32

        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")

    def build_prompt(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> str:
        """
        Build the final prompt passed to the model.

        Prompt sections:
        - Optional retrieved memory context (Phase 1: if manager chooses to still generate)
        - Optional dataset context (LongBench / PDF chunk context)
        - User question

        The structure is intentionally explicit for debugability and demo clarity.
        """
        parts = []

        parts.append("You are a helpful assistant. Answer the user's question using the provided context when relevant.")

        if self.cfg.include_retrieved_memory_context and retrieved is not None:
            parts.append(
                "PREVIOUSLY USEFUL MEMORY:\n"
                f"{retrieved.item.answer_text}"
            )

        if self.cfg.include_dataset_context:
            dataset_ctx = (mq.context or {}).get("dataset_context", "") or ""
            if dataset_ctx:
                parts.append(f"DATASET CONTEXT:\n{dataset_ctx}")

        # Optional document signature, useful for debugging / reproducibility
        doc_sig = (mq.context or {}).get("doc_signature")
        if doc_sig:
            parts.append(f"DOCUMENT SIGNATURE: {doc_sig}")

        parts.append(f"QUESTION:\n{mq.raw_query}")
        parts.append("ANSWER:")

        prompt = "\n\n".join(parts)
        self.last_prompt = prompt
        return prompt

    def generate(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> Tuple[str, Provenance, QualitySignals]:
        """
        Generate an answer for the given MemoryQuery.

        Returns:
            answer_text, provenance, quality_signals
        """
        prompt = self.build_prompt(mq, retrieved=retrieved)

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_input_length,
            padding=False,
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature if self.cfg.do_sample else None,
                top_p=self.cfg.top_p if self.cfg.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # We only want the newly generated portion, not the echoed prompt
        generated_ids = output_ids[0][input_ids.shape[1] :]
        answer_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=self.cfg.skip_special_tokens,
        ).strip()

        # Fallback: if the model returns empty text, return a safe placeholder
        if not answer_text:
            answer_text = "(No answer generated.)"

        provenance = Provenance(
            model_id=mq.model_id,
            prompt_version=mq.prompt_version,
            generated_at_utc=datetime.now(timezone.utc),
            generator_backend="transformers",
            quantization=None,
            context_window=self.cfg.max_input_length,
        )

        # Phase 1 quality signal is intentionally simple.
        # Real task metrics or user feedback can populate this later.
        quality = QualitySignals(
            score=None,
            success=True if answer_text and answer_text != "(No answer generated.)" else False,
            metrics={},
        )

        return answer_text, provenance, quality

    def info(self) -> dict:
        """
        Lightweight metadata for logging/debugging.
        """
        return {
            "model_id": self.cfg.model_id,
            "device": self.device,
            "max_input_length": self.cfg.max_input_length,
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "do_sample": self.cfg.do_sample,
        }
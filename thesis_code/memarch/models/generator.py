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
- Exact hits may return directly from MemoryManager
- Semantic hits are typically used as context assistance only
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
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memarch.memory.schema import MatchType, MemoryHit, MemoryQuery, Provenance, QualitySignals


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
    include_doc_signature: bool = True

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

    @staticmethod
    def _safe_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _retrieved_section(self, retrieved: MemoryHit) -> str:
        """
        Build a safe retrieved-memory section.

        Phase 1 policy:
        - retrieved semantic memory is assistive, not authoritative
        - the model should use it only if consistent with the current document context
        """
        answer_text = self._safe_text(retrieved.item.answer_text)
        if not answer_text:
            return ""

        lines = [
            "RETRIEVED ANSWER FOR A SIMILAR EARLIER QUESTION:",
            answer_text,
        ]

        meta_parts = []
        meta_parts.append(f"match_type={retrieved.match_type.value}")
        meta_parts.append(f"source_tier={retrieved.source_tier.value}")
        meta_parts.append(f"score={retrieved.score:.4f}")

        if retrieved.semantic_rank is not None:
            meta_parts.append(f"semantic_rank={retrieved.semantic_rank}")

        lines.append("")
        lines.append("RETRIEVAL METADATA:")
        lines.append(", ".join(meta_parts))

        if retrieved.match_type == MatchType.SEMANTIC:
            lines.append("")
            lines.append(
                "Use the retrieved answer only if it is consistent with the document context "
                "and the current question."
            )

        return "\n".join(lines)

    def build_prompt(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> str:
        """
        Build the final prompt passed to the model.
        """
        parts = []

        parts.append(
            "You are a helpful assistant. Answer the current question using the provided "
            "document context. If a retrieved prior answer is provided, use it only when "
            "it is consistent with the document context and current question."
        )

        if self.cfg.include_dataset_context:
            dataset_ctx = self._safe_text((mq.context or {}).get("dataset_context", ""))
            if dataset_ctx:
                parts.append(f"DOCUMENT CONTEXT:\n{dataset_ctx}")

        if self.cfg.include_doc_signature:
            doc_sig = (mq.context or {}).get("doc_signature")
            if doc_sig:
                parts.append(f"DOCUMENT SIGNATURE: {doc_sig}")

        if self.cfg.include_retrieved_memory_context and retrieved is not None:
            retrieved_block = self._retrieved_section(retrieved)
            if retrieved_block:
                parts.append(retrieved_block)

        parts.append(f"CURRENT QUESTION:\n{self._safe_text(mq.raw_query)}")
        parts.append(
            "ANSWER THE CURRENT QUESTION ONLY. Do not mention retrieval metadata unless it is directly relevant."
        )
        parts.append("ANSWER:")

        prompt = "\n\n".join(parts)
        self.last_prompt = prompt
        return prompt

    def generate(
        self,
        mq: MemoryQuery,
        retrieved: Optional[MemoryHit] = None,
    ) -> Tuple[str, Provenance, QualitySignals]:
        """
        Generate an answer for the given MemoryQuery.
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

        generate_kwargs: Dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": self.cfg.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.cfg.do_sample:
            generate_kwargs["temperature"] = self.cfg.temperature
            generate_kwargs["top_p"] = self.cfg.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**generate_kwargs)

        generated_ids = output_ids[0][input_ids.shape[1]:]
        answer_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=self.cfg.skip_special_tokens,
        ).strip()

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

        quality_metrics: Dict[str, float] = {}
        if retrieved is not None and retrieved.match_type == MatchType.SEMANTIC:
            quality_metrics["semantic_retrieval_score"] = float(retrieved.score)
        elif retrieved is not None and retrieved.match_type == MatchType.EXACT:
            quality_metrics["retrieval_score"] = float(retrieved.score)

        quality = QualitySignals(
            score=None,
            success=True if answer_text and answer_text != "(No answer generated.)" else False,
            metrics=quality_metrics,
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
            "include_retrieved_memory_context": self.cfg.include_retrieved_memory_context,
            "include_dataset_context": self.cfg.include_dataset_context,
            "include_doc_signature": self.cfg.include_doc_signature,
        }


Generator = HFGenerator
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

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memarch.memory.schema import MatchType, MemoryHit, MemoryQuery, Provenance, QualitySignals


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


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
    use_fast_tokenizer: bool = False

    # Runtime behavior
    cpu_fallback_on_failure: bool = True
    generation_backend: str = "auto"  # "auto" | "manual_greedy" | "hf_generate"

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
            use_fast=self.cfg.use_fast_tokenizer,
            local_files_only=self.cfg.local_files_only,
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_dtype = self._resolve_torch_dtype(self.cfg.torch_dtype, self.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            local_files_only=self.cfg.local_files_only,
            dtype=self.model_dtype,
        )
        self.model.eval()
        self.model.to(self.device)

        self.last_prompt: Optional[str] = None
        self.last_generation_meta: Optional[Dict[str, object]] = None

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

    def _select_generation_backend(self) -> str:
        if self.cfg.generation_backend != "auto":
            return self.cfg.generation_backend

        # The local Mac environment was unstable with transformers.generate().
        # Use manual greedy on CPU/MPS, and keep hf_generate for CUDA by default.
        if self.device in {"cpu", "mps"}:
            return "manual_greedy"
        return "hf_generate"

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

        meta_parts = [
            f"match_type={retrieved.match_type.value}",
            f"source_tier={retrieved.source_tier.value}",
            f"score={retrieved.score:.4f}",
        ]

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

    def _manual_greedy_decode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        generated_ids = input_ids
        generated_mask = attention_mask
        eos_token_id = self.tokenizer.eos_token_id

        with torch.inference_mode():
            for _ in range(int(self.cfg.max_new_tokens)):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=generated_mask,
                    use_cache=False,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                if generated_mask is not None:
                    next_mask = torch.ones(
                        (generated_mask.shape[0], 1),
                        dtype=generated_mask.dtype,
                        device=generated_mask.device,
                    )
                    generated_mask = torch.cat([generated_mask, next_mask], dim=-1)

                if eos_token_id is not None and bool((next_token_id == eos_token_id).all()):
                    break

        return generated_ids

    def _hf_generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        generate_kwargs: Dict[str, object] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": self.cfg.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.cfg.do_sample:
            generate_kwargs["temperature"] = self.cfg.temperature
            generate_kwargs["top_p"] = self.cfg.top_p

        with torch.inference_mode():
            return self.model.generate(**generate_kwargs)

    def _move_model(self, device: str) -> None:
        self.model.to(device)
        self.device = device

    def _record_meta(
        self,
        *,
        prompt: str,
        input_tokens: int,
        output_tokens: int,
        truncated: bool,
        tokenize_time_s: float,
        gen_time_s: float,
        decode_time_s: float,
        backend_used: str,
        retrieved: Optional[MemoryHit],
    ) -> None:
        meta: Dict[str, object] = {
            "device": self.device,
            "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
            "generation_backend": backend_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "truncated": truncated,
            "tokenize_time_s": tokenize_time_s,
            "gen_time_s": gen_time_s,
            "decode_time_s": decode_time_s,
            "used_retrieved_context": retrieved is not None,
            "retrieved_match_type": retrieved.match_type.value if retrieved is not None else None,
            "retrieved_source_tier": retrieved.source_tier.value if retrieved is not None else None,
            "retrieved_score": float(retrieved.score) if retrieved is not None else None,
        }

        if self.device == "cuda":
            try:
                meta["cuda_device_name"] = torch.cuda.get_device_name(0)
                meta["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 3)
                meta["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 3)
            except Exception:
                pass

        self.last_generation_meta = meta

    # def generate(
    #     self,
    #     mq: MemoryQuery,
    #     retrieved: Optional[MemoryHit] = None,
    # ) -> Tuple[str, Provenance, QualitySignals]:
    #     """
    #     Generate an answer for the given MemoryQuery.
    #     """
    #     prompt = self.build_prompt(mq, retrieved=retrieved)

    #     tok_t0 = time.time()
    #     enc = self.tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         truncation=True,
    #         max_length=self.cfg.max_input_length,
    #         padding=False,
    #     )
    #     tokenize_time_s = time.time() - tok_t0

    #     input_ids = enc["input_ids"]
    #     attention_mask = enc.get("attention_mask")

    #     input_tokens = int(input_ids.shape[-1])

    #     full_prompt_token_count = None
    #     if hasattr(self.tokenizer, "encode") and callable(getattr(self.tokenizer, "encode")):
    #         try:
    #             full_prompt_token_count = len(
    #                 self.tokenizer.encode(prompt, add_special_tokens=True)
    #             )
    #         except Exception:
    #             full_prompt_token_count = None

    #     truncated = bool(
    #         full_prompt_token_count is not None and full_prompt_token_count > input_tokens
    #     )

    #     input_ids = input_ids.to(self.device)
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.to(self.device)

    #     backend_used = self._select_generation_backend()

    #     try:
    #         gen_t0 = time.time()

    #         if hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
    #             output_ids = self.model.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 max_new_tokens=self.cfg.max_new_tokens,
    #                 do_sample=self.cfg.do_sample,
    #                 temperature=(self.cfg.temperature if self.cfg.do_sample else None),
    #                 top_p=(self.cfg.top_p if self.cfg.do_sample else None),
    #                 pad_token_id=self.tokenizer.pad_token_id,
    #                 eos_token_id=self.tokenizer.eos_token_id,
    #             )
    #             backend_used = "hf_generate"
    #         else:
    #             output_ids = self._manual_greedy_decode(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #             )
    #             backend_used = "manual_greedy"

    #         gen_time_s = time.time() - gen_t0
    #     except RuntimeError as e:
    #         if self.device in {"cuda", "mps"} and self.cfg.cpu_fallback_on_failure:
    #             self._move_model("cpu")
    #             self.model_dtype = self._resolve_torch_dtype(self.cfg.torch_dtype, "cpu")

    #             input_ids = input_ids.detach().to("cpu")
    #             attention_mask = attention_mask.detach().to("cpu") if attention_mask is not None else None

    #             gen_t0 = time.time()
    #             if hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
    #                 output_ids = self.model.generate(
    #                     input_ids=input_ids,
    #                     attention_mask=attention_mask,
    #                     max_new_tokens=self.cfg.max_new_tokens,
    #                     do_sample=self.cfg.do_sample,
    #                     temperature=(self.cfg.temperature if self.cfg.do_sample else None),
    #                     top_p=(self.cfg.top_p if self.cfg.do_sample else None),
    #                     pad_token_id=self.tokenizer.pad_token_id,
    #                     eos_token_id=self.tokenizer.eos_token_id,
    #                 )
    #                 backend_used = "hf_generate"
    #             else:
    #                 output_ids = self._manual_greedy_decode(
    #                     input_ids=input_ids,
    #                     attention_mask=attention_mask,
    #                 )
    #                 backend_used = "manual_greedy"
    #             gen_time_s = time.time() - gen_t0
    #         else:
    #             raise RuntimeError(f"HFGenerator failed on device={self.device}: {type(e).__name__}: {e}") from e

    #     decode_t0 = time.time()
    #     if backend_used == "hf_generate":
    #         generated_ids = output_ids[0][input_ids.shape[1]:]
    #         output_tokens = int(generated_ids.shape[-1])
    #         answer_text = self.tokenizer.decode(
    #             generated_ids.detach().to("cpu"),
    #             skip_special_tokens=self.cfg.skip_special_tokens,
    #         ).strip()
    #     else:
    #         output_tokens = max(0, int(output_ids.shape[-1]) - input_tokens)
    #         output_ids_cpu = output_ids.detach().to("cpu")
    #         input_ids_cpu = input_ids.detach().to("cpu")

    #         full_text = self.tokenizer.decode(
    #             output_ids_cpu[0],
    #             skip_special_tokens=self.cfg.skip_special_tokens,
    #         )
    #         prompt_used = self.tokenizer.decode(
    #             input_ids_cpu[0],
    #             skip_special_tokens=self.cfg.skip_special_tokens,
    #         )

    #         if full_text.startswith(prompt_used):
    #             answer_text = full_text[len(prompt_used):].lstrip()
    #         else:
    #             answer_text = full_text.strip()

    #     decode_time_s = time.time() - decode_t0

    #     if not answer_text:
    #         answer_text = "(No answer generated.)"

    #     self._record_meta(
    #         prompt=prompt,
    #         input_tokens=input_tokens,
    #         output_tokens=output_tokens,
    #         truncated=truncated,
    #         tokenize_time_s=tokenize_time_s,
    #         gen_time_s=gen_time_s,
    #         decode_time_s=decode_time_s,
    #         backend_used=backend_used,
    #         retrieved=retrieved,
    #     )

    #     provenance = Provenance(
    #         model_id=mq.model_id,
    #         prompt_version=mq.prompt_version,
    #         generated_at_utc=datetime.now(timezone.utc),
    #         generator_backend="transformers",
    #         quantization=None,
    #         context_window=self.cfg.max_input_length,
    #     )

    #     quality_metrics: Dict[str, float] = {}
    #     if retrieved is not None and retrieved.match_type == MatchType.SEMANTIC:
    #         quality_metrics["semantic_retrieval_score"] = float(retrieved.score)
    #     elif retrieved is not None and retrieved.match_type == MatchType.EXACT:
    #         quality_metrics["retrieval_score"] = float(retrieved.score)

    #     quality_metrics["input_tokens"] = float(input_tokens)
    #     quality_metrics["output_tokens"] = float(output_tokens)
    #     quality_metrics["gen_time_s"] = float(gen_time_s)

    #     quality = QualitySignals(
    #         score=None,
    #         success=bool(answer_text and answer_text != "(No answer generated.)"),
    #         metrics=quality_metrics,
    #     )

    #     return answer_text, provenance, quality

    def generate(
        self,
        mq: MemoryQuery,
        retrieved: Optional[MemoryHit] = None,
    ) -> Tuple[str, Provenance, QualitySignals]:
        """
        Generate an answer for the given MemoryQuery.
        """
        prompt = self.build_prompt(mq, retrieved=retrieved)

        tok_t0 = time.time()
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_input_length,
            padding=False,
        )
        tokenize_time_s = time.time() - tok_t0

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        input_tokens = int(input_ids.shape[-1])
        truncated = input_tokens >= int(self.cfg.max_input_length)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        backend_used = self._select_generation_backend()

        try:
            gen_t0 = time.time()

            if backend_used == "manual_greedy":
                if callable(self.model):
                    output_ids = self._manual_greedy_decode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                elif hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
                    # Unit-test fallback for FakeModel, which is not callable but has .generate()
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=self.cfg.do_sample,
                        temperature=(self.cfg.temperature if self.cfg.do_sample else None),
                        top_p=(self.cfg.top_p if self.cfg.do_sample else None),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    backend_used = "hf_generate"
                else:
                    raise RuntimeError("Model supports neither callable forward pass nor .generate().")

            elif backend_used == "hf_generate":
                if hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=self.cfg.do_sample,
                        temperature=(self.cfg.temperature if self.cfg.do_sample else None),
                        top_p=(self.cfg.top_p if self.cfg.do_sample else None),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    output_ids = self._manual_greedy_decode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    backend_used = "manual_greedy"
            else:
                raise ValueError(f"Unsupported generation backend: {backend_used}")

            gen_time_s = time.time() - gen_t0

        except RuntimeError as e:
            if self.device in {"cuda", "mps"} and self.cfg.cpu_fallback_on_failure:
                self._move_model("cpu")
                self.model_dtype = self._resolve_torch_dtype(self.cfg.torch_dtype, "cpu")

                input_ids = input_ids.detach().to("cpu")
                attention_mask = attention_mask.detach().to("cpu") if attention_mask is not None else None

                gen_t0 = time.time()
                if callable(self.model):
                    output_ids = self._manual_greedy_decode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    backend_used = "manual_greedy"
                elif hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=self.cfg.do_sample,
                        temperature=(self.cfg.temperature if self.cfg.do_sample else None),
                        top_p=(self.cfg.top_p if self.cfg.do_sample else None),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    backend_used = "hf_generate"
                else:
                    raise RuntimeError("Model supports neither callable forward pass nor .generate().")
                gen_time_s = time.time() - gen_t0
            else:
                raise RuntimeError(
                    f"HFGenerator failed on device={self.device}: {type(e).__name__}: {e}"
                ) from e

        decode_t0 = time.time()
        if backend_used == "hf_generate":
            generated_ids = output_ids[0][input_ids.shape[1]:]
            output_tokens = int(generated_ids.shape[-1])
            answer_text = self.tokenizer.decode(
                generated_ids.detach().to("cpu"),
                skip_special_tokens=self.cfg.skip_special_tokens,
            ).strip()
        else:
            output_tokens = max(0, int(output_ids.shape[-1]) - input_tokens)
            output_ids_cpu = output_ids.detach().to("cpu")
            input_ids_cpu = input_ids.detach().to("cpu")

            full_text = self.tokenizer.decode(
                output_ids_cpu[0],
                skip_special_tokens=self.cfg.skip_special_tokens,
            )
            prompt_used = self.tokenizer.decode(
                input_ids_cpu[0],
                skip_special_tokens=self.cfg.skip_special_tokens,
            )

            if full_text.startswith(prompt_used):
                answer_text = full_text[len(prompt_used):].lstrip()
            else:
                answer_text = full_text.strip()

        decode_time_s = time.time() - decode_t0

        if not answer_text:
            answer_text = "(No answer generated.)"

        self._record_meta(
            prompt=prompt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            truncated=truncated,
            tokenize_time_s=tokenize_time_s,
            gen_time_s=gen_time_s,
            decode_time_s=decode_time_s,
            backend_used=backend_used,
            retrieved=retrieved,
        )

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

        quality_metrics["input_tokens"] = float(input_tokens)
        quality_metrics["output_tokens"] = float(output_tokens)
        quality_metrics["gen_time_s"] = float(gen_time_s)

        quality = QualitySignals(
            score=None,
            success=bool(answer_text and answer_text != "(No answer generated.)"),
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
            "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
            "max_input_length": self.cfg.max_input_length,
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "do_sample": self.cfg.do_sample,
            "generation_backend": self._select_generation_backend(),
            "include_retrieved_memory_context": self.cfg.include_retrieved_memory_context,
            "include_dataset_context": self.cfg.include_dataset_context,
            "include_doc_signature": self.cfg.include_doc_signature,
        }


Generator = HFGenerator
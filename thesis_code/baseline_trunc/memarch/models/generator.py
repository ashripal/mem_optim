# tbaseline_mem/models/generator.py
"""
Text generator for baseline + memory-augmented runs.

Key requirements for our thesis demo:
- Accept (context, question, optional retrieved memory snippets) and generate an answer.
- Measure latency and token counts.
- Safe defaults for Apple M1/MPS:
  - truncation to a token budget
  - optional CPU fallback on RuntimeError (OOM / MPS issues)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import time

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch is required for generator.py") from e

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


@dataclass
class GeneratorConfig:
    model_id: str = "microsoft/Phi-3-mini-128k-instruct"
    device: str = "mps"  # "mps" | "cpu" | "cuda"
    dtype: str = "auto"  # "auto" | "float16" | "float32"
    max_input_tokens: int = 4096
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 0.9
    cpu_fallback_on_error: bool = True


class Generator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.device = self._pick_device(cfg.device)

        tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

        # Some models don't have pad token set; safe for generation
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        dtype = self._pick_dtype(cfg.dtype, self.device)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype)
        model.to(self.device)
        model.eval()

        self.tokenizer = tok
        self.model = model

        # For budgeting: model's supported context length (if present)
        self.model_max_positions = getattr(model.config, "max_position_embeddings", None)

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

    @staticmethod
    def _pick_dtype(dtype: str, device: torch.device):
        dtype = (dtype or "auto").lower()
        if dtype == "float32":
            return torch.float32
        if dtype == "float16":
            return torch.float16
        # auto
        if device.type in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    def build_prompt(
        self,
        context: str,
        question: str,
        retrieved_memory: Optional[List[str]] = None,
    ) -> str:
        """
        Build a single prompt string. Keep it simple and stable for baselines.

        retrieved_memory: optional list of short snippets from prior QA memory.
        """
        mem_block = ""
        if retrieved_memory:
            joined = "\n".join(f"- {m}" for m in retrieved_memory)
            mem_block = (
                "\n\nRelevant prior experience (may help):\n"
                f"{joined}\n"
            )

        user_content = (
            "Use the context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
            f"{mem_block}\n"
            "Answer concisely:"
        )

        # Prefer chat template if present
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_content}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return user_content

    def generate(
        self,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns: (answer_text, metrics)
        metrics include token counts, latency, truncation flag, device.
        """
        effective_max_in = self.cfg.max_input_tokens
        if self.model_max_positions:
            effective_max_in = min(effective_max_in, int(self.model_max_positions))

        t0 = time.perf_counter()

        # Tokenize with truncation budget
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=effective_max_in,
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        truncated = (input_tokens >= effective_max_in)

        # Move to model device
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.do_sample,
                    temperature=self.cfg.temperature if self.cfg.do_sample else 0.0,
                    top_p=self.cfg.top_p if self.cfg.do_sample else 1.0,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        except RuntimeError as e:
            if not self.cfg.cpu_fallback_on_error:
                raise
            # CPU fallback (slow but robust)
            self.model.to("cpu")
            model_device = torch.device("cpu")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.do_sample,
                    temperature=self.cfg.temperature if self.cfg.do_sample else 0.0,
                    top_p=self.cfg.top_p if self.cfg.do_sample else 1.0,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        t1 = time.perf_counter()

        # Decode answer only (avoid prompt duplication issues)
        # Decode full output and then strip prompt_text
        full_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        answer = full_text[len(prompt_text):].strip()

        output_tokens = int(out.shape[-1] - inputs["input_ids"].shape[-1])
        dt = t1 - t0
        tps = (output_tokens / dt) if dt > 0 else None

        metrics = {
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": output_tokens,
            "latency_s": dt,
            "tokens_per_sec": tps,
            "device_used": str(model_device),
            "truncated_to_max_input": truncated,
            "max_input_tokens": effective_max_in,
        }
        return answer, metrics
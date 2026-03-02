# tiers/tier0_compute.py
"""
Tier 0 (Compute): model + tokenizer + device selection + generation.

Responsibilities:
- Load HuggingFace tokenizer + causal LM once
- Choose device (prefer MPS if available, else CPU)
- Apply input token truncation to respect max_input_tokens
- Run generation and return standardized metadata
- Optional CPU fallback when MPS hits long-sequence/runtime issues

This module should NOT:
- Touch dataset files (Tier 2)
- Implement caching (Tier 1)
- Write JSONL logs (pipeline/logging.py)
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _select_preferred_device(cfg: Any) -> str:
    """
    Baseline device policy:
      - Prefer MPS if available (Apple Silicon), else CPU.
    """
    if _mps_available():
        return "mps"
    return "cpu"


def _truncate_input_ids(
    input_ids: torch.Tensor, max_input_tokens: int
) -> Tuple[torch.Tensor, bool]:
    """
    Truncate tokenized input to max_input_tokens.

    Strategy (baseline, simple & reproducible):
      - Keep the *last* max_input_tokens tokens (tail truncation).
      - This is commonly used for long-context prompts.

    Returns:
      (possibly truncated input_ids, truncated_flag)
    """
    if max_input_tokens is None or max_input_tokens <= 0:
        return input_ids, False

    # input_ids shape: [1, seq_len]
    seq_len = int(input_ids.shape[-1])
    if seq_len <= max_input_tokens:
        return input_ids, False

    truncated_ids = input_ids[:, -max_input_tokens:]
    return truncated_ids, True


class ComputeEngine:
    """
    Encapsulates Tier 0 compute.

    Public API:
      - generate(prompt, max_input_tokens, max_new_tokens) -> dict
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model_id: str = getattr(cfg, "model_id")
        self.cpu_fallback_on_long: bool = bool(getattr(cfg, "cpu_fallback_on_long", False))

        # Select device and load artifacts
        self.preferred_device: str = _select_preferred_device(cfg)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        # Ensure pad token exists for generation if needed
        if self.tokenizer.pad_token_id is None:
            # Common fallback: use EOS as PAD for causal LMs
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model (baseline: full precision; keep simple and explicit)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=None,  # baseline simplicity; rely on HF defaults
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Move to preferred device
        self._move_model(self.preferred_device)

    def _move_model(self, device: str) -> None:
        """
        Move model to device. Kept separate to support CPU fallback retry.
        """
        if device == "mps":
            self.model.to("mps")
        else:
            self.model.to("cpu")

    def _generate_once(
        self,
        *,
        prompt: str,
        device: str,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """
        Perform a single generation attempt on the specified device.
        """
        # Tokenize on CPU first (tokenizer is CPU anyway)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"]  # [1, seq_len]

        input_ids, truncated = _truncate_input_ids(input_ids, max_input_tokens)
        input_tokens = int(input_ids.shape[-1])

        # Move inputs to device
        if device == "mps":
            input_ids = input_ids.to("mps")
        else:
            input_ids = input_ids.to("cpu")

        # Generation config (baseline, deterministic)
        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        t0 = time.time()
        with torch.inference_mode():
            out_ids = self.model.generate(input_ids=input_ids, **gen_kwargs)
        gen_time_s = time.time() - t0

        # out_ids includes the prompt + generated continuation
        out_total_tokens = int(out_ids.shape[-1])
        output_tokens = max(0, out_total_tokens - input_tokens)

        # Decode full output, then strip prompt prefix as best-effort
        full_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Best-effort to isolate the generated continuation:
        # Decode prompt_used from input_ids to handle truncation properly.
        prompt_used = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if full_text.startswith(prompt_used):
            output_text = full_text[len(prompt_used) :].lstrip()
        else:
            # Fallback: return full text if matching fails
            output_text = full_text

        return {
            "ok": True,
            "device": device,
            "truncated": truncated,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "gen_time_s": gen_time_s,
            "prompt_used": prompt_used,
            "output_text": output_text,
        }

    def generate(
        self,
        *,
        prompt: str,
        max_input_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a completion for the given prompt.

        Returns a dict containing:
          - output_text
          - device used
          - token counts
          - truncation flag
          - timing
        """
        max_input_tokens = int(max_input_tokens or getattr(self.cfg, "max_input_tokens", 8192))
        max_new_tokens = int(max_new_tokens or getattr(self.cfg, "max_new_tokens", 64))

        # Attempt on preferred device
        device = self.preferred_device
        try:
            return self._generate_once(
                prompt=prompt,
                device=device,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )
        except RuntimeError as e:
            # Optional CPU fallback (primarily for MPS long-sequence instability)
            if device == "mps" and self.cpu_fallback_on_long:
                # Move model to CPU and retry once
                self._move_model("cpu")
                try:
                    out = self._generate_once(
                        prompt=prompt,
                        device="cpu",
                        max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens,
                    )
                    out["fallback_from"] = "mps"
                    out["fallback_reason"] = f"{type(e).__name__}: {e}"
                    return out
                except Exception as e2:
                    return {
                        "ok": False,
                        "device": "cpu",
                        "error": f"CPU fallback also failed: {type(e2).__name__}: {e2}",
                        "fallback_from": "mps",
                        "fallback_reason": f"{type(e).__name__}: {e}",
                    }

            # No fallback or not eligible
            return {
                "ok": False,
                "device": device,
                "error": f"{type(e).__name__}: {e}",
            }
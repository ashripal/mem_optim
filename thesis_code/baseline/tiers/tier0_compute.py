# tiers/tier0_compute.py
"""
Tier 0 (Compute): model + tokenizer + device selection + generation.

Responsibilities:
- Load HuggingFace tokenizer + causal LM once
- Choose device (prefer CUDA if available, else MPS if available, else CPU)
- Apply input token truncation to respect max_input_tokens
- Run generation and return standardized metadata
- Optional CPU fallback when accelerator hits long-sequence/runtime issues

This module should NOT:
- Touch dataset files (Tier 2)
- Implement caching (Tier 1)
- Write JSONL logs (pipeline/logging.py)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _select_preferred_device(cfg: Any) -> str:
    """
    Device policy:
      - Prefer CUDA if available
      - Else prefer MPS if available
      - Else CPU

    Optional config override:
      cfg.device in {"auto", "cuda", "mps", "cpu"}
    """
    requested = str(getattr(cfg, "device", "auto")).strip().lower()

    if requested == "cuda":
        return "cuda" if _cuda_available() else "cpu"
    if requested == "mps":
        return "mps" if _mps_available() else "cpu"
    if requested == "cpu":
        return "cpu"

    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(cfg: Any, device: str) -> Optional[torch.dtype]:
    """
    Resolve torch dtype from config.

    Supported cfg.dtype values:
      - "auto" / None
      - "fp16" / "float16"
      - "bf16" / "bfloat16"
      - "fp32" / "float32"

    Default behavior:
      - CUDA -> float16
      - MPS  -> float16
      - CPU  -> float32
    """
    raw_dtype = getattr(cfg, "dtype", "auto")
    dtype_name = "auto" if raw_dtype is None else str(raw_dtype).strip().lower()

    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32

    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _truncate_tensor_tail(
    tensor: torch.Tensor,
    max_input_tokens: int,
) -> Tuple[torch.Tensor, bool]:
    """
    Tail-truncate a rank-2 token-aligned tensor on the sequence dimension.
    """
    if max_input_tokens is None or max_input_tokens <= 0:
        return tensor, False

    seq_len = int(tensor.shape[-1])
    if seq_len <= max_input_tokens:
        return tensor, False

    return tensor[:, -max_input_tokens:], True


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

        self.preferred_device: str = _select_preferred_device(cfg)
        self.active_device: str = self.preferred_device
        self.model_dtype: Optional[torch.dtype] = _resolve_torch_dtype(cfg, self.active_device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.model_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._move_model(self.active_device)

    def _move_model(self, device: str) -> None:
        """
        Move model to the target device and update active_device.
        """
        if device == "cuda":
            self.model.to("cuda")
        elif device == "mps":
            self.model.to("mps")
        else:
            self.model.to("cpu")
        self.active_device = device

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

        Notes:
        - Uses tokenizer-side truncation first to avoid materializing oversized inputs.
        - Uses a manual greedy decoding loop instead of transformers.generate(),
          because generate() is crashing in the current local Mac environment.
        """
        tok_t0 = time.time()

        enc_full = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        full_input_ids = enc_full["input_ids"]
        full_input_tokens = int(full_input_ids.shape[-1])

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=int(max_input_tokens),
        )
        tok_time_s = time.time() - tok_t0

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        input_ids, truncated_ids = _truncate_tensor_tail(input_ids, max_input_tokens)

        truncated_mask = False
        if attention_mask is not None:
            attention_mask, truncated_mask = _truncate_tensor_tail(attention_mask, max_input_tokens)

        input_tokens = int(input_ids.shape[-1])
        tokenizer_truncated = full_input_tokens > input_tokens
        truncated = bool(tokenizer_truncated or truncated_ids or truncated_mask)

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        gen_t0 = time.time()

        generated_ids = input_ids
        generated_mask = attention_mask

        eos_token_id = self.tokenizer.eos_token_id

        with torch.inference_mode():
            for _ in range(int(max_new_tokens)):
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

        gen_time_s = time.time() - gen_t0

        decode_t0 = time.time()

        out_total_tokens = int(generated_ids.shape[-1])
        output_tokens = max(0, out_total_tokens - input_tokens)

        generated_ids_cpu = generated_ids.detach().to("cpu")
        input_ids_cpu = input_ids.detach().to("cpu")

        full_text = self.tokenizer.decode(generated_ids_cpu[0], skip_special_tokens=True)
        prompt_used = self.tokenizer.decode(input_ids_cpu[0], skip_special_tokens=True)
        decode_time_s = time.time() - decode_t0

        if full_text.startswith(prompt_used):
            output_text = full_text[len(prompt_used):].lstrip()
        else:
            output_text = full_text

        meta: Dict[str, Any] = {
            "ok": True,
            "device": device,
            "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
            "truncated": truncated,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "gen_time_s": gen_time_s,
            "tokenize_time_s": tok_time_s,
            "decode_time_s": decode_time_s,
            "generation_backend": "manual_greedy",
            "prompt_used": prompt_used,
            "output_text": output_text,
        }

        if device == "cuda":
            try:
                meta["cuda_device_name"] = torch.cuda.get_device_name(0)
                meta["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 3)
                meta["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 3)
            except Exception:
                pass

        return meta

    def generate(
        self,
        *,
        prompt: str,
        max_input_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a completion for the given prompt.
        """
        max_input_tokens = int(max_input_tokens or getattr(self.cfg, "max_input_tokens", 8192))
        max_new_tokens = int(max_new_tokens or getattr(self.cfg, "max_new_tokens", 64))

        device = self.active_device

        try:
            return self._generate_once(
                prompt=prompt,
                device=device,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )
        except RuntimeError as e:
            if device in {"cuda", "mps"} and self.cpu_fallback_on_long:
                try:
                    self._move_model("cpu")
                    self.preferred_device = "cpu"
                    self.model_dtype = _resolve_torch_dtype(self.cfg, "cpu")
                    self.model.to("cpu")

                    out = self._generate_once(
                        prompt=prompt,
                        device="cpu",
                        max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens,
                    )
                    out["fallback_from"] = device
                    out["fallback_reason"] = f"{type(e).__name__}: {e}"
                    out["device_switch_persisted"] = True
                    return out
                except Exception as e2:
                    return {
                        "ok": False,
                        "device": "cpu",
                        "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
                        "error": f"CPU fallback also failed: {type(e2).__name__}: {e2}",
                        "fallback_from": device,
                        "fallback_reason": f"{type(e).__name__}: {e}",
                        "device_switch_persisted": True,
                    }

            return {
                "ok": False,
                "device": device,
                "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
                "error": f"{type(e).__name__}: {e}",
            }
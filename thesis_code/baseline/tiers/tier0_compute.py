# (FULL FILE — copy/paste replaces existing)

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
    try:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    except Exception:
        return False


def _select_preferred_device(cfg: Any) -> str:
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


def _cuda_bf16_supported() -> bool:
    try:
        return _cuda_available() and bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _resolve_torch_dtype(cfg: Any, device: str) -> Optional[torch.dtype]:
    raw_dtype = getattr(cfg, "dtype", "auto")
    dtype_name = "auto" if raw_dtype is None else str(raw_dtype).strip().lower()

    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32

    if device == "cuda":
        return torch.bfloat16 if _cuda_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _truncate_tensor_tail(
    tensor: Optional[torch.Tensor],
    max_input_tokens: int,
) -> Tuple[Optional[torch.Tensor], bool]:
    if tensor is None:
        return None, False

    if max_input_tokens is None or max_input_tokens <= 0:
        return tensor, False

    seq_len = int(tensor.shape[-1])
    if seq_len <= max_input_tokens:
        return tensor, False

    return tensor[:, -max_input_tokens:], True


def _dtype_name(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return "none"
    return str(dtype).replace("torch.", "")


class ComputeEngine:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model_id: str = getattr(cfg, "model_id")
        self.cpu_fallback_on_long: bool = bool(getattr(cfg, "cpu_fallback_on_long", False))

        self.preferred_device: str = _select_preferred_device(cfg)
        self.active_device: str = self.preferred_device
        self.model_dtype: Optional[torch.dtype] = _resolve_torch_dtype(cfg, self.active_device)

        # 🔥 Enable TF32 on Jetson (Ampere)
        if self.active_device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        use_fast = bool(getattr(cfg, "use_fast_tokenizer", True))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=use_fast)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "torch_dtype": self.model_dtype,
        }

        attn_impl = getattr(cfg, "attn_implementation", None)
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        model_kwargs["trust_remote_code"] = bool(getattr(cfg, "trust_remote_code", False))

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()
        self._move_model(self.active_device)

        # 🔥 Optional compile (safe fallback)
        if self.active_device == "cuda":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        # 🔥 Warmup (VERY important on Jetson)
        self._warmup()

    def _warmup(self):
        try:
            dummy = self.tokenizer("Hello", return_tensors="pt").input_ids.to(self.active_device)
            with torch.inference_mode():
                _ = self.model.generate(dummy, max_new_tokens=2)
        except Exception:
            pass

    def _move_model(self, device: str) -> None:
        if device == "cuda":
            self.model.to("cuda")
        elif device == "mps":
            self.model.to("mps")
        else:
            self.model.to("cpu")
            try:
                self.model.float()
            except Exception:
                pass

        self.active_device = device

    def _sync_device(self, device: str) -> None:
        try:
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
        except Exception:
            pass

    def _generate_once(
        self,
        *,
        prompt: str,
        device: str,
        max_input_tokens: int,
        max_new_tokens: int,
    ) -> Dict[str, Any]:

        tok_t0 = time.time()
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        tok_time_s = time.time() - tok_t0

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        input_ids, _ = _truncate_tensor_tail(input_ids, max_input_tokens)
        if attention_mask is not None:
            attention_mask, _ = _truncate_tensor_tail(attention_mask, max_input_tokens)

        # 🔥 Non-blocking transfer (Jetson boost)
        input_ids = input_ids.to(device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        self._sync_device(device)
        t0 = time.time()

        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        self._sync_device(device)
        gen_time_s = time.time() - t0

        continuation_ids = out_ids[:, input_ids.shape[-1]:]
        output_text = self.tokenizer.decode(continuation_ids[0], skip_special_tokens=True)

        return {
            "ok": True,
            "device": device,
            "dtype": _dtype_name(self.model_dtype),
            "gen_time_s": gen_time_s,
            "output_text": output_text,
        }

    def generate(self, *, prompt: str, max_input_tokens=None, max_new_tokens=None):
        return self._generate_once(
            prompt=prompt,
            device=self.active_device,
            max_input_tokens=max_input_tokens or 2048,
            max_new_tokens=max_new_tokens or 64,
        )
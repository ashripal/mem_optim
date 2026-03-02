# scripts/test_tier0_compute.py
"""
Basic smoke-test script for Tier 0 (Compute).

Why this exists:
- Unit tests should NOT download or run large HF models.
- This script is a *manual / developer* check to validate Tier0 behavior end-to-end:
  - model/tokenizer loading
  - device selection (MPS vs CPU)
  - input truncation behavior
  - generation works
  - (optional) CPU fallback flag is wired

Recommended usage:
- Use a small model for quick checks.
- Run on CPU unless you specifically want to test MPS.

Examples:

# Quick CPU smoke test with a small model
python scripts/test_tier0_compute.py --model_id gpt2 --device cpu --max_input_tokens 128 --max_new_tokens 32

# Test truncation with a long prompt
python scripts/test_tier0_compute.py --model_id gpt2 --device cpu --long_prompt --max_input_tokens 64 --max_new_tokens 16

# Prefer MPS if available (falls back to CPU if MPS is unavailable)
python scripts/test_tier0_compute.py --model_id gpt2 --device auto

Notes:
- This script will download the model from HuggingFace if not cached.
- Keep it out of CI unless you have a dedicated cache + allowances.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from typing import Any, Optional

from baseline.tiers.tier0_compute import ComputeEngine


@dataclass
class _Cfg:
    """
    Minimal config object compatible with ComputeEngine.
    ComputeEngine reads:
      - model_id
      - max_input_tokens
      - max_new_tokens
      - cpu_fallback_on_long
    """
    model_id: str
    max_input_tokens: int
    max_new_tokens: int
    cpu_fallback_on_long: bool


def _build_prompt(long_prompt: bool) -> str:
    base = (
        "You are a helpful assistant. "
        "Summarize the following in one sentence:\n\n"
        "LongBench evaluates long-context reasoning and retrieval.\n"
    )
    if not long_prompt:
        return base

    # Create an intentionally long prompt to trigger truncation.
    filler = " ".join(["token"] * 5000)
    return base + "\n\nFILLER:\n" + filler + "\n\nQuestion: What is LongBench?"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_id",
        type=str,
        default="gpt2",
        help="HuggingFace model id. Use a small model for quick checks (default: gpt2).",
    )
    ap.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Requested device preference. 'auto' uses Tier0 default (prefers MPS if available).",
    )
    ap.add_argument("--max_input_tokens", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument(
        "--cpu_fallback_on_long",
        action="store_true",
        help="Enable CPU fallback on MPS runtime errors (used in Tier0).",
    )
    ap.add_argument(
        "--long_prompt",
        action="store_true",
        help="Use an intentionally long prompt to test truncation.",
    )
    ap.add_argument(
        "--print_prompt_used",
        action="store_true",
        help="Print the prompt actually used after truncation.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = _Cfg(
        model_id=args.model_id,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        cpu_fallback_on_long=args.cpu_fallback_on_long,
    )

    print("========================================")
    print(" Tier0 Compute Smoke Test")
    print("========================================")
    print(f"Model             : {cfg.model_id}")
    print(f"Requested device  : {args.device}")
    print(f"Max input tokens  : {cfg.max_input_tokens}")
    print(f"Max new tokens    : {cfg.max_new_tokens}")
    print(f"CPU fallback      : {cfg.cpu_fallback_on_long}")
    print(f"Long prompt       : {args.long_prompt}")
    print("========================================")

    prompt = _build_prompt(args.long_prompt)

    # Instantiate Tier0
    engine = ComputeEngine(cfg)

    # Optionally override device preference (manual)
    if args.device != "auto":
        # Force preferred device and move the model there.
        # This keeps behavior explicit for manual testing.
        engine.preferred_device = args.device
        engine._move_model(args.device)

    # Run generation
    out = engine.generate(
        prompt=prompt,
        max_input_tokens=cfg.max_input_tokens,
        max_new_tokens=cfg.max_new_tokens,
    )

    print("\n--- Tier0 Output (metadata) ---")
    # Print key metadata deterministically
    keys = [
        "ok",
        "device",
        "fallback_from",
        "fallback_reason",
        "truncated",
        "input_tokens",
        "output_tokens",
        "gen_time_s",
    ]
    for k in keys:
        if k in out:
            print(f"{k:>15}: {out.get(k)}")

    if args.print_prompt_used and "prompt_used" in out:
        print("\n--- Prompt Used (post-truncation) ---")
        print(out["prompt_used"][:2000])
        if len(out["prompt_used"]) > 2000:
            print("... [truncated for display]")

    print("\n--- Generated Text ---")
    text = out.get("output_text", "")
    print(textwrap.fill(text, width=100) if isinstance(text, str) else text)

    # Basic sanity checks (exit code behavior is nice for automation)
    if not out.get("ok", False):
        raise SystemExit(2)

    if args.long_prompt and not out.get("truncated", False):
        print("\n[warn] long_prompt was set but 'truncated' is False. "
              "This may happen if the tokenizer produced fewer tokens than expected.")
    print("\n[ok] Tier0 smoke test completed.")


if __name__ == "__main__":
    main()
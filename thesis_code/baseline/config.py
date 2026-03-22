# config.py
"""
Central configuration for the LongBench baseline.

Responsibilities:
- Parse CLI arguments
- Store configuration in a typed dataclass
- Keep ALL experiment knobs in one place

This allows:
- Reproducible experiments
- Easy baseline vs optimized comparisons
- Clean logging of config into run_header
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Config:
    # Tier 2 (Disk)
    tier2_repo: str
    task_glob: str

    # Output
    out_dir: str

    # Model
    model_id: str

    # Run limits
    max_examples: int
    max_input_tokens: int
    max_new_tokens: int

    # Tier 1 (RAM)
    max_cache_items: int

    # Device behavior
    device: str
    dtype: str
    cpu_fallback_on_long: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongBench Baseline Runner")

    # -----------------------------
    # Tier 2
    # -----------------------------
    parser.add_argument(
        "--tier2_repo",
        type=str,
        required=True,
        help="Directory containing LongBench JSONL files (Tier 2 Disk).",
    )
    parser.add_argument(
        "--task_glob",
        type=str,
        default="",
        help="Substring filter for task filenames (e.g., 'trec').",
    )

    # -----------------------------
    # Output
    # -----------------------------
    parser.add_argument(
        "--out_dir",
        type=str,
        default="tier2_disk/runs",
        help="Directory where run JSONL will be written.",
    )

    # -----------------------------
    # Model
    # -----------------------------
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Phi-3-mini-128k-instruct",
        help="HuggingFace model ID.",
    )

    # -----------------------------
    # Run parameters
    # -----------------------------
    parser.add_argument(
        "--max_examples",
        type=int,
        default=25,
        help="Maximum total examples to evaluate.",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=8192,
        help="Maximum input tokens before truncation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate.",
    )

    # -----------------------------
    # Tier 1
    # -----------------------------
    parser.add_argument(
        "--max_cache_items",
        type=int,
        default=64,
        help="Maximum number of items in RAM cache.",
    )

    # -----------------------------
    # Device / precision behavior
    # -----------------------------
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Execution device preference. 'auto' selects cuda -> mps -> cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
        help="Model dtype policy. 'auto' uses accelerator-friendly defaults.",
    )
    parser.add_argument(
        "--cpu_fallback_on_long",
        action="store_true",
        help="Retry generation on CPU if CUDA/MPS fails for long sequences.",
    )

    return parser


def get_config() -> Config:
    parser = build_parser()
    args = parser.parse_args()

    return Config(
        tier2_repo=args.tier2_repo,
        task_glob=args.task_glob,
        out_dir=args.out_dir,
        model_id=args.model_id,
        max_examples=args.max_examples,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_cache_items=args.max_cache_items,
        device=args.device,
        dtype=args.dtype,
        cpu_fallback_on_long=args.cpu_fallback_on_long,
    )
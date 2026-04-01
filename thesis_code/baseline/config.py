"""
Central configuration for the LongBench baseline.

TRUE BASELINE:
- Stateless LLM execution
- NO caching
- NO memory reuse
- Each query is processed independently

Responsibilities:
- Parse CLI arguments
- Store configuration in a typed dataclass
- Keep ALL experiment knobs in one place

This allows:
- Reproducible experiments
- Clean baseline vs memarch comparison
- Transparent logging into run_header
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

    # Device / precision behavior
    device: str
    dtype: str
    cpu_fallback_on_long: bool

    # Tokenizer / model loading behavior
    use_fast_tokenizer: bool
    attn_implementation: str | None
    trust_remote_code: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongBench TRUE Baseline Runner (Stateless)")

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
        default="fp16",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
        help="Model dtype policy. fp16 recommended for Jetson.",
    )
    parser.add_argument(
        "--cpu_fallback_on_long",
        action="store_true",
        help="Retry generation on CPU if CUDA/MPS fails for long sequences.",
    )

    # -----------------------------
    # Tokenizer / model loading behavior
    # -----------------------------
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        default=True,
        help="Use the fast tokenizer implementation when available.",
    )
    parser.add_argument(
        "--no_use_fast_tokenizer",
        action="store_false",
        dest="use_fast_tokenizer",
        help="Disable the fast tokenizer implementation.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "sdpa", "flash_attention_2"],
        help=(
            "Optional attention backend passed to the model loader. "
            "Use only if supported by the selected model/runtime."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow Hugging Face models with custom remote code.",
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
        device=args.device,
        dtype=args.dtype,
        cpu_fallback_on_long=args.cpu_fallback_on_long,
        use_fast_tokenizer=args.use_fast_tokenizer,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
"""
Entry point for running the LongBench baseline experiment.

TRUE BASELINE:
- Stateless LLM execution
- NO caching
- NO memory reuse
- Each query is processed independently

Responsibilities:
- Load config
- Invoke experiment runner
- Print run path
- Exit cleanly

This file intentionally contains NO logic.
"""

from __future__ import annotations

from baseline.config import get_config
from baseline.pipeline.runner import run_experiment


def main() -> None:
    cfg = get_config()

    print("========================================")
    print(" LongBench TRUE Baseline Run (Stateless)")
    print("========================================")
    print(f"Tier2 repo        : {cfg.tier2_repo}")
    print(f"Model             : {cfg.model_id}")
    print(f"Device            : {cfg.device}")
    print(f"Dtype             : {cfg.dtype}")
    print(f"Max examples      : {cfg.max_examples}")
    print(f"Max input tokens  : {cfg.max_input_tokens}")
    print(f"Max new tokens    : {cfg.max_new_tokens}")
    print(f"CPU fallback      : {cfg.cpu_fallback_on_long}")

    # --- Edge-relevant debug knobs ---
    print("----------------------------------------")
    print("Tokenizer / Model Settings")
    print("----------------------------------------")
    print(f"Use fast tokenizer : {cfg.use_fast_tokenizer}")
    print(f"Attention impl     : {cfg.attn_implementation}")
    print(f"Trust remote code  : {cfg.trust_remote_code}")

    print("========================================")

    run_path = run_experiment(cfg)

    print("Run complete.")
    print(f"Results written to: {run_path}")


if __name__ == "__main__":
    main()
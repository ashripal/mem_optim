from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

from baseline.benchmarks.configs import BenchmarkConfig, OutputConfig, WorkloadConfig
from baseline.benchmarks.execute import run_benchmark


def _normalize_dtype(dtype: str) -> str:
    mapping = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
        "auto": "auto",
    }
    return mapping[str(dtype).strip().lower()]

def _normalize_workload_mode(mode: str) -> str:
    """
    Translate memarch-style workload names into the subset currently supported
    by the baseline benchmark config.
    """
    mode = str(mode).strip().lower()

    mapping = {
        "cold": "cold",
        "replay_once": "replay_once",
        "replay_k": "replay_k",
        "mixed_reuse": "mixed_reuse",
        "exact_interleaved": "mixed_reuse",
        "approx_interleaved": "mixed_reuse",
        "family_clustered": "mixed_reuse",
        "cache_pressure": "replay_k",
    }

    if mode not in mapping:
        raise ValueError(
            f"Unsupported CLI mode for baseline runner: {mode!r}. "
            f"Expected one of: {sorted(mapping.keys())}"
        )

    normalized = mapping[mode]
    if normalized != mode:
        print(f"[info] Baseline runner remapping workload mode {mode!r} -> {normalized!r}")
    return normalized

def _normalize_total_requests(
    original_mode: str,
    normalized_mode: str,
    total_requests: int | None,
    max_examples: int,
    replay_k: int,
) -> int | None:
    """
    Fill in required total_requests when a richer memarch-style mode is remapped
    onto a baseline mode that requires it.
    """
    if total_requests is not None:
        return int(total_requests)

    original_mode = str(original_mode).strip().lower()
    normalized_mode = str(normalized_mode).strip().lower()

    if normalized_mode == "mixed_reuse":
        if original_mode in {"exact_interleaved", "approx_interleaved", "family_clustered"}:
            inferred = int(max_examples)
            if inferred <= 0:
                raise ValueError(
                    f"Could not infer total_requests for mode={original_mode!r}; "
                    f"max_examples must be > 0."
                )
            print(
                f"[info] Baseline runner inferring total_requests={inferred} "
                f"for remapped mode {original_mode!r} -> 'mixed_reuse'"
            )
            return inferred

    if normalized_mode == "replay_k":
        if original_mode == "cache_pressure":
            inferred = max(1, int(max_examples))
            print(
                f"[info] Baseline runner inferring total_requests={inferred} "
                f"for remapped mode 'cache_pressure' -> 'replay_k'"
            )
            return inferred

    return total_requests

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a baseline LongBench benchmark with configurable workload and "
            "generation settings. This runner mirrors the memarch CLI surface "
            "as closely as possible, excluding retrieval-specific memory options."
        )
    )

    parser.add_argument("--tier2_repo", type=str, required=True)
    parser.add_argument("--benchmark_name", type=str, default="baseline_longbench_benchmark")
    parser.add_argument("--out_root", type=str, default="artifacts/benchmark_runs/baseline")
    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--task_glob", type=str, default="")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Direct path to a JSONL workload file (overrides task_glob).",
    )

    parser.add_argument("--max_examples", type=int, default=25)
    parser.add_argument(
        "--mode",
        type=str,
        default="cold",
        choices=[
            "cold",
            "replay_once",
            "replay_k",
            "cache_pressure",
            "mixed_reuse",
            "exact_interleaved",
            "approx_interleaved",
            "family_clustered",
        ],
    )
    parser.add_argument("--replay_k", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_requests", type=int, default=None)
    parser.add_argument("--repeat_fraction", type=float, default=0.0)

    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-128k-instruct")
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument(
        "--decoding_mode",
        type=str,
        default="greedy",
        choices=["greedy", "beam", "sample"],
        help="Generation decoding mode. Accepted for CLI parity with memarch.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam count. Must be >1 only when decoding_mode=beam.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cpu_fallback_on_long", action="store_true")

    parser.add_argument("--jetson_safe_mode", action="store_true")

    parser.add_argument("--write_workload_manifest", action="store_true", default=True)
    parser.add_argument(
        "--no_write_workload_manifest",
        dest="write_workload_manifest",
        action="store_false",
    )
    parser.add_argument("--write_summary_json", action="store_true")

    return parser


def args_to_config(args: argparse.Namespace) -> BenchmarkConfig:
    effective_task_glob = str(args.task_glob or "").strip()
    if str(args.input_path or "").strip():
        effective_task_glob = Path(args.input_path).stem

    max_input_tokens = int(args.max_input_tokens)
    if args.jetson_safe_mode:
        max_input_tokens = min(max_input_tokens, 2048)

    normalized_mode = _normalize_workload_mode(args.mode)
    normalized_total_requests = _normalize_total_requests(
        original_mode=args.mode,
        normalized_mode=normalized_mode,
        total_requests=args.total_requests,
        max_examples=args.max_examples,
        replay_k=args.replay_k,
    )

    workload = WorkloadConfig(
        task_glob=effective_task_glob,
        max_examples=args.max_examples,
        mode=normalized_mode,
        replay_k=args.replay_k,
        shuffle=args.shuffle,
        seed=args.seed,
        total_requests=normalized_total_requests,
        repeat_fraction=args.repeat_fraction,
    )

    output = OutputConfig(
        root_dir=args.out_root,
        write_workload_manifest=args.write_workload_manifest,
        write_summary_json=args.write_summary_json,
    )

    cfg = BenchmarkConfig(
        benchmark_name=args.benchmark_name,
        notes=args.notes,
        tier2_repo=args.tier2_repo,
        out_dir=args.out_root,
        model_id=args.model_id,
        max_input_tokens=max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=_normalize_dtype(args.dtype),
        cpu_fallback_on_long=args.cpu_fallback_on_long,
        workload=workload,
        output=output,
    )

    if args.decoding_mode != "greedy":
        print("[warn] Non-greedy decoding flags were provided for baseline CLI parity.")
        print("[warn] Ensure the underlying baseline generator/config consumes them if intended.")

    if args.do_sample:
        print("[warn] --do_sample was provided for baseline CLI parity.")
        print("[warn] Ensure the underlying baseline generator/config consumes it if intended.")

    if args.local_files_only:
        print("[warn] --local_files_only was provided for baseline CLI parity.")
        print("[warn] Ensure the underlying baseline loader/config consumes it if intended.")

    cfg.validate()
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = args_to_config(args)

    print("========================================")
    print(" Baseline LongBench Benchmark")
    print("========================================")
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    print("========================================")

    artifacts = run_benchmark(cfg)

    print("Run complete.")
    print("Artifacts:")
    if isinstance(artifacts, dict):
        for key, value in artifacts.items():
            print(f"- {key}: {value}")
    else:
        print(artifacts)


if __name__ == "__main__":
    main()
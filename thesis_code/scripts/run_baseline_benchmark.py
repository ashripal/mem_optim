# scripts/run_baseline_benchmark.py
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a baseline LongBench benchmark with configurable workload replay/caching."
    )

    # Required / core paths
    parser.add_argument(
        "--tier2_repo",
        type=str,
        required=True,
        help="Path to the directory containing LongBench task .jsonl files.",
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="baseline_longbench_benchmark",
        help="Logical name for this benchmark run.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="artifacts/benchmark_runs/baseline",
        help="Root directory for benchmark outputs.",
    )

    # Workload selection / shaping
    parser.add_argument(
        "--task_glob",
        type=str,
        default="",
        help="Substring filter for LongBench task filenames. Empty means all tasks.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=25,
        help="Maximum number of base examples to load before replay expansion.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cold",
        choices=["cold", "replay_once", "replay_k", "cache_pressure"],
        help="Workload mode controlling reuse behavior.",
    )
    parser.add_argument(
        "--replay_k",
        type=int,
        default=2,
        help="Total number of passes when mode=replay_k.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the base example list before workload expansion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --shuffle is enabled.",
    )

    # System / model knobs
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Phi-3-mini-128k-instruct",
        help="Hugging Face model id for Tier 0 generation.",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=8192,
        help="Maximum input tokens sent to the model.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--max_cache_items",
        type=int,
        default=64,
        help="Tier 1 RAM cache capacity in items.",
    )
    parser.add_argument(
        "--cpu_fallback_on_long",
        action="store_true",
        help="Enable CPU fallback when long inputs exceed device constraints.",
    )

    # Output behavior
    parser.add_argument(
        "--write_workload_manifest",
        action="store_true",
        default=True,
        help="Write a workload manifest JSON next to the run JSONL.",
    )
    parser.add_argument(
        "--no_write_workload_manifest",
        action="store_true",
        help="Disable workload manifest writing.",
    )
    parser.add_argument(
        "--write_summary_json",
        action="store_true",
        help="Also write a summary JSON after the run completes.",
    )

    # Provenance
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional freeform notes stored in the run config.",
    )

    return parser


def args_to_config(args: argparse.Namespace) -> BenchmarkConfig:
    write_workload_manifest = True
    if args.no_write_workload_manifest:
        write_workload_manifest = False
    elif args.write_workload_manifest:
        write_workload_manifest = True

    workload = WorkloadConfig(
        task_glob=args.task_glob,
        max_examples=args.max_examples,
        mode=args.mode,
        replay_k=args.replay_k,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    output = OutputConfig(
        root_dir=args.out_root,
        write_workload_manifest=write_workload_manifest,
        write_summary_json=args.write_summary_json,
    )

    cfg = BenchmarkConfig(
        benchmark_name=args.benchmark_name,
        notes=args.notes,
        tier2_repo=args.tier2_repo,
        out_dir=args.out_root,
        model_id=args.model_id,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_cache_items=args.max_cache_items,
        cpu_fallback_on_long=args.cpu_fallback_on_long,
        workload=workload,
        output=output,
    )
    cfg.validate()
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = args_to_config(args)
    artifacts = run_benchmark(cfg)

    print("\nBaseline benchmark completed.\n")
    print("Resolved configuration:")
    print(json.dumps(cfg.to_dict(), indent=2, default=str))

    print("\nArtifacts:")
    for key, value in artifacts.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
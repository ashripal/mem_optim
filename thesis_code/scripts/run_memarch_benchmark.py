from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

from memarch.benchmarks.configs import (
    BenchmarkConfig,
    MemoryConfig,
    NamespaceConfig,
    OutputConfig,
    WorkloadConfig,
)
from memarch.benchmarks.execute import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a memarch LongBench benchmark with configurable workload and multi-tier memory settings."
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
        default="memarch_longbench_benchmark",
        help="Logical name for this benchmark run.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="artifacts/benchmark_runs/memarch",
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

    # Model / generation knobs
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/Phi-3-mini-128k-instruct",
        help="Hugging Face model id for generation.",
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
        "--cpu_fallback_on_long",
        action="store_true",
        help="Enable CPU fallback when long inputs exceed device constraints.",
    )

    # Namespace / identity knobs
    parser.add_argument(
        "--user_id",
        type=str,
        default="user_a",
        help="User namespace id for memarch retrieval/storage.",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        default="session_a",
        help="Session namespace id for memarch retrieval/storage.",
    )
    parser.add_argument(
        "--cohort_id",
        type=str,
        default=None,
        help="Optional cohort namespace id.",
    )

    # Memory knobs
    parser.add_argument(
        "--ram_capacity_items",
        type=int,
        default=64,
        help="RAM memory capacity for memarch.",
    )
    parser.add_argument(
        "--disk_store_path",
        type=str,
        default="artifacts/benchmark_runs/memarch/memory/memarch_benchmark.sqlite",
        help="SQLite path for persistent memarch disk store.",
    )
    parser.add_argument(
        "--clear_disk_store_before_run",
        action="store_true",
        help="Delete the existing persistent memarch disk store before this run to force a cold start.",
    )

    # Retrieval mode / semantic knobs
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="exact_only",
        choices=["exact_only", "semantic_context", "semantic_bypass"],
        help="Retrieval mode for the benchmark run.",
    )
    parser.add_argument(
        "--semantic_enabled",
        action="store_true",
        help="Enable semantic retrieval support.",
    )
    parser.add_argument(
        "--semantic_threshold_context",
        type=float,
        default=0.85,
        help="Minimum similarity score required to use a semantic hit as generation context.",
    )
    parser.add_argument(
        "--semantic_threshold_bypass",
        type=float,
        default=1.01,
        help="Minimum similarity score required to bypass generation and return a semantic hit directly.",
    )
    parser.add_argument(
        "--max_semantic_candidates",
        type=int,
        default=5,
        help="Maximum number of semantic candidates to rank after filtering.",
    )

    # Embedder knobs
    parser.add_argument(
        "--embedding_model_id",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model id used for semantic retrieval.",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device used for embedding generation.",
    )
    parser.add_argument(
        "--embedding_local_files_only",
        action="store_true",
        help="Load embedding model only from local files.",
    )

    parser.add_argument(
        "--promote_disk_hits_to_ram",
        action="store_true",
        default=True,
        help="Promote disk hits into RAM after retrieval.",
    )
    parser.add_argument(
        "--no_promote_disk_hits_to_ram",
        action="store_true",
        help="Disable promotion of disk hits into RAM.",
    )
    parser.add_argument(
        "--return_memory_directly",
        action="store_true",
        default=True,
        help="Return reusable memory hits directly when policy allows.",
    )
    parser.add_argument(
        "--no_return_memory_directly",
        action="store_true",
        help="Disable direct return of memory hits.",
    )
    parser.add_argument(
        "--enable_storage",
        action="store_true",
        default=True,
        help="Enable storing generated outputs into memory.",
    )
    parser.add_argument(
        "--no_enable_storage",
        action="store_true",
        help="Disable storing generated outputs into memory.",
    )
    parser.add_argument(
        "--store_in_ram",
        action="store_true",
        default=True,
        help="Store admitted entries in RAM.",
    )
    parser.add_argument(
        "--no_store_in_ram",
        action="store_true",
        help="Disable RAM storage.",
    )
    parser.add_argument(
        "--store_on_disk",
        action="store_true",
        default=True,
        help="Store admitted entries on disk.",
    )
    parser.add_argument(
        "--no_store_on_disk",
        action="store_true",
        help="Disable disk storage.",
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
    write_workload_manifest = not args.no_write_workload_manifest
    promote_disk_hits_to_ram = not args.no_promote_disk_hits_to_ram
    return_memory_directly = not args.no_return_memory_directly
    enable_storage = not args.no_enable_storage
    store_in_ram = not args.no_store_in_ram
    store_on_disk = not args.no_store_on_disk

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

    namespaces = NamespaceConfig(
        user_id=args.user_id,
        session_id=args.session_id,
        cohort_id=args.cohort_id,
    )

    memory = MemoryConfig(
        ram_capacity_items=args.ram_capacity_items,
        disk_store_path=args.disk_store_path,
        clear_disk_store_before_run=args.clear_disk_store_before_run,
        retrieval_mode=args.retrieval_mode,
        semantic_enabled=args.semantic_enabled,
        semantic_threshold_context=args.semantic_threshold_context,
        semantic_threshold_bypass=args.semantic_threshold_bypass,
        max_semantic_candidates=args.max_semantic_candidates,
        embedding_model_id=args.embedding_model_id,
        embedding_device=args.embedding_device,
        embedding_local_files_only=args.embedding_local_files_only,
        promote_disk_hits_to_ram=promote_disk_hits_to_ram,
        return_memory_directly=return_memory_directly,
        enable_storage=enable_storage,
        store_in_ram=store_in_ram,
        store_on_disk=store_on_disk,
    )

    cfg = BenchmarkConfig(
        benchmark_name=args.benchmark_name,
        notes=args.notes,
        tier2_repo=args.tier2_repo,
        out_dir=args.out_root,
        model_id=args.model_id,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        cpu_fallback_on_long=args.cpu_fallback_on_long,
        workload=workload,
        output=output,
        namespaces=namespaces,
        memory=memory,
    )
    cfg.validate()
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = args_to_config(args)
    artifacts = run_benchmark(cfg)

    print("\nMemarch benchmark completed.\n")
    print("Resolved configuration:")
    print(json.dumps(cfg.to_dict(), indent=2, default=str))

    print("\nArtifacts:")
    for key, value in artifacts.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
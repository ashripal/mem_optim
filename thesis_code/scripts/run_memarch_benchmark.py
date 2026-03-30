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
        description=(
            "Run a memarch LongBench benchmark with configurable workload and "
            "evidence-guided multi-tier memory settings."
        )
    )

    parser.add_argument("--tier2_repo", type=str, required=True)
    parser.add_argument("--benchmark_name", type=str, default="memarch_longbench_benchmark")
    parser.add_argument("--out_root", type=str, default="artifacts/benchmark_runs/memarch")
    parser.add_argument("--notes", type=str, default="")

    parser.add_argument("--task_glob", type=str, default="")
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
        help="Generation decoding mode.",
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
        default="float32",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cpu_fallback_on_long", action="store_true")

    # Keep this flag for convenience/documentation, but do not pass it into BenchmarkConfig
    # unless your local BenchmarkConfig actually defines a jetson_safe_mode field.
    parser.add_argument("--jetson_safe_mode", action="store_true")

    parser.add_argument("--user_id", type=str, default="user_a")
    parser.add_argument("--session_id", type=str, default="session_a")
    parser.add_argument("--cohort_id", type=str, default=None)

    parser.add_argument("--ram_capacity_items", type=int, default=64)
    parser.add_argument(
        "--disk_store_path",
        type=str,
        default="artifacts/benchmark_runs/memarch/memory/memarch_benchmark.sqlite",
    )
    parser.add_argument("--clear_disk_store_before_run", action="store_true")

    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="lexical_gated_direct_semantic_context",
        choices=[
            "exact_only",
            "lexical_context",
            "lexical_gated_direct",
            "semantic_context",
            "semantic_bypass",
            "lexical_semantic_context",
            "lexical_gated_direct_semantic_context",
        ],
    )
    parser.add_argument("--promote_disk_hits_to_ram", action="store_true", default=True)
    parser.add_argument("--no_promote_disk_hits_to_ram", dest="promote_disk_hits_to_ram", action="store_false")
    parser.add_argument("--return_memory_directly", action="store_true", default=True)
    parser.add_argument("--no_return_memory_directly", dest="return_memory_directly", action="store_false")

    # Lexical controls
    parser.add_argument("--lexical_enabled", action="store_true")
    parser.add_argument("--lexical_context_threshold", type=float, default=0.55)
    parser.add_argument("--lexical_direct_threshold", type=float, default=0.75)
    parser.add_argument("--lexical_top_k", type=int, default=3)
    parser.add_argument("--prefer_same_source", action="store_true", default=True)
    parser.add_argument("--no_prefer_same_source", dest="prefer_same_source", action="store_false")
    parser.add_argument(
        "--safe_direct_reuse_tasks",
        type=str,
        default="trec",
        help="Comma-separated task list allowed for lexical direct reuse.",
    )

    # Semantic controls
    parser.add_argument("--semantic_enabled", action="store_true")
    parser.add_argument("--semantic_threshold_context", type=float, default=0.75)
    parser.add_argument("--semantic_threshold_bypass", type=float, default=1.01)
    parser.add_argument("--max_semantic_candidates", type=int, default=5)

    parser.add_argument(
        "--embedding_model_id",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--embedding_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--embedding_local_files_only", action="store_true")

    parser.add_argument("--disable_storage", action="store_true")
    parser.add_argument("--disable_store_in_ram", action="store_true")
    parser.add_argument("--disable_store_on_disk", action="store_true")

    parser.add_argument("--write_workload_manifest", action="store_true", default=True)
    parser.add_argument("--no_write_workload_manifest", dest="write_workload_manifest", action="store_false")
    parser.add_argument("--write_summary_json", action="store_true")

    return parser


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


def _parse_task_list(text: str) -> list[str]:
    if not str(text or "").strip():
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def args_to_config(args: argparse.Namespace) -> BenchmarkConfig:
    workload = WorkloadConfig(
        task_glob=args.task_glob,
        max_examples=args.max_examples,
        mode=args.mode,
        replay_k=args.replay_k,
        shuffle=args.shuffle,
        seed=args.seed,
        total_requests=args.total_requests,
        repeat_fraction=args.repeat_fraction,
    )

    output = OutputConfig(
        root_dir=args.out_root,
        write_workload_manifest=args.write_workload_manifest,
        write_summary_json=args.write_summary_json,
    )

    namespaces = NamespaceConfig(
        user_id=args.user_id,
        session_id=args.session_id,
        cohort_id=args.cohort_id,
    )

    retrieval_mode = str(args.retrieval_mode).strip()

    lexical_mode_requested = retrieval_mode in {
        "lexical_context",
        "lexical_gated_direct",
        "lexical_semantic_context",
        "lexical_gated_direct_semantic_context",
    }
    semantic_mode_requested = retrieval_mode in {
        "semantic_context",
        "semantic_bypass",
        "lexical_semantic_context",
        "lexical_gated_direct_semantic_context",
    }

    memory = MemoryConfig(
        ram_capacity_items=args.ram_capacity_items,
        disk_store_path=args.disk_store_path,
        clear_disk_store_before_run=args.clear_disk_store_before_run,
        retrieval_mode=retrieval_mode,
        promote_disk_hits_to_ram=args.promote_disk_hits_to_ram,
        return_memory_directly=args.return_memory_directly,

        lexical_enabled=bool(args.lexical_enabled or lexical_mode_requested),
        lexical_threshold_context=args.lexical_context_threshold,
        lexical_threshold_bypass=args.lexical_direct_threshold,
        lexical_top_k=args.lexical_top_k,
        prefer_same_source=args.prefer_same_source,
        safe_direct_reuse_tasks=_parse_task_list(args.safe_direct_reuse_tasks),

        semantic_enabled=bool(args.semantic_enabled or semantic_mode_requested),
        semantic_threshold_context=args.semantic_threshold_context,
        semantic_threshold_bypass=args.semantic_threshold_bypass,
        max_semantic_candidates=args.max_semantic_candidates,

        embedding_model_id=args.embedding_model_id,
        embedding_device=args.embedding_device,
        embedding_local_files_only=args.embedding_local_files_only,

        enable_storage=not args.disable_storage,
        store_in_ram=not args.disable_store_in_ram,
        store_on_disk=not args.disable_store_on_disk,
    )

    cfg = BenchmarkConfig(
        benchmark_name=args.benchmark_name,
        notes=args.notes,
        tier2_repo=args.tier2_repo,
        out_dir=args.out_root,
        model_id=args.model_id,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        decoding_mode=args.decoding_mode,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device=args.device,
        dtype=_normalize_dtype(args.dtype),
        local_files_only=args.local_files_only,
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

    print("========================================")
    print(" MemArch LongBench Benchmark")
    print("========================================")
    print(json.dumps(cfg.to_dict(), indent=2))
    print("========================================")

    run_path = run_benchmark(cfg)

    print("Run complete.")
    print(f"Results written to: {run_path}")


if __name__ == "__main__":
    main()
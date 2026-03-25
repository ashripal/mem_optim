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
        description="Run a memarch LongBench benchmark with configurable workload and evidence-guided multi-tier memory settings."
    )

    parser.add_argument("--tier2_repo", type=str, required=True)
    parser.add_argument("--benchmark_name", type=str, default="memarch_longbench_benchmark")
    parser.add_argument("--out_root", type=str, default="artifacts/benchmark_runs/memarch")

    parser.add_argument("--task_glob", type=str, default="")
    parser.add_argument("--max_examples", type=int, default=25)
    parser.add_argument(
        "--mode",
        type=str,
        default="cold",
        choices=["cold", "replay_once", "replay_k", "cache_pressure", "mixed_reuse"],
    )
    parser.add_argument("--replay_k", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_requests", type=int, default=None)
    parser.add_argument("--repeat_fraction", type=float, default=0.0)

    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-128k-instruct")
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    # Greedy rollback defaults
    parser.add_argument(
        "--decoding_mode",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Generation decoding mode.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam count. Keep this at 1 for greedy decoding.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cpu_fallback_on_long", action="store_true")
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
        default="exact_only",
        choices=["exact_only", "semantic_context"],
    )
    parser.add_argument("--semantic_enabled", action="store_true")
    parser.add_argument("--semantic_threshold_context", type=float, default=0.85)
    parser.add_argument("--semantic_threshold_bypass", type=float, default=1.01)
    parser.add_argument("--max_semantic_candidates", type=int, default=5)

    parser.add_argument(
        "--embedding_model_id",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding_device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--embedding_local_files_only", action="store_true")

    parser.add_argument("--promote_disk_hits_to_ram", action="store_true", default=True)
    parser.add_argument("--no_promote_disk_hits_to_ram", action="store_true")
    parser.add_argument("--return_memory_directly", action="store_true", default=True)
    parser.add_argument("--no_return_memory_directly", action="store_true")
    parser.add_argument("--enable_storage", action="store_true", default=True)
    parser.add_argument("--no_enable_storage", action="store_true")
    parser.add_argument("--store_in_ram", action="store_true", default=True)
    parser.add_argument("--no_store_in_ram", action="store_true")
    parser.add_argument("--store_on_disk", action="store_true", default=True)
    parser.add_argument("--no_store_on_disk", action="store_true")

    parser.add_argument("--write_workload_manifest", action="store_true", default=True)
    parser.add_argument("--no_write_workload_manifest", action="store_true")
    parser.add_argument("--write_summary_json", action="store_true")

    parser.add_argument("--notes", type=str, default="")

    return parser


def _normalize_dtype(dtype: str) -> str:
    d = (dtype or "auto").lower().strip()
    mapping = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }
    return mapping.get(d, d)


def args_to_config(args: argparse.Namespace) -> BenchmarkConfig:
    write_workload_manifest = not args.no_write_workload_manifest
    promote_disk_hits_to_ram = not args.no_promote_disk_hits_to_ram
    return_memory_directly = not args.no_return_memory_directly
    enable_storage = not args.no_enable_storage
    store_in_ram = not args.no_store_in_ram
    store_on_disk = not args.no_store_on_disk

    max_input_tokens = int(args.max_input_tokens)
    if args.jetson_safe_mode:
        max_input_tokens = min(max_input_tokens, 2048)

    retrieval_mode = str(args.retrieval_mode).strip()
    semantic_enabled = retrieval_mode == "semantic_context" or bool(args.semantic_enabled)

    semantic_threshold_bypass = 1.01
    if retrieval_mode == "exact_only":
        semantic_enabled = False

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
        retrieval_mode=retrieval_mode,
        semantic_enabled=semantic_enabled,
        semantic_threshold_context=args.semantic_threshold_context,
        semantic_threshold_bypass=semantic_threshold_bypass,
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
        max_input_tokens=max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        decoding_mode=args.decoding_mode,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=bool(args.do_sample),
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
    artifacts = run_benchmark(cfg)

    print("\nMemarch benchmark completed.\n")
    print("Resolved configuration:")
    print(json.dumps(cfg.to_dict(), indent=2, default=str))

    print("\nArtifacts:")
    for key, value in artifacts.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
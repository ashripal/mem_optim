# memarch/benchmarks/configs.py
"""
Typed configuration objects for memarch benchmark runs.

Goals:
- Mirror baseline benchmark config structure as closely as possible
- Keep workload definition separate from memarch-specific memory controls
- Make runs easy to serialize, compare, and reproduce
- Preserve familiar top-level fields like:
    tier2_repo, out_dir, model_id, max_input_tokens, max_new_tokens

This module does NOT:
- Parse CLI arguments
- Load dataset examples
- Execute benchmarks
- Create plots or summaries
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Workload configuration
# ---------------------------------------------------------------------

@dataclass
class WorkloadConfig:
    """
    Controls how raw LongBench examples are selected and replayed.

    Supported modes:
      - "cold": each selected example appears once
      - "replay_once": selected examples are replayed one additional time
      - "replay_k": selected examples are repeated replay_k total times
      - "cache_pressure": delayed reuse pattern intended to stress memory tiers
    """
    task_glob: str = ""
    max_examples: Optional[int] = 25

    mode: str = "cold"
    replay_k: int = 2

    shuffle: bool = False
    seed: int = 0

    def validate(self) -> None:
        valid_modes = {"cold", "replay_once", "replay_k", "cache_pressure"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid workload mode: {self.mode!r}. "
                f"Expected one of: {sorted(valid_modes)}"
            )

        if self.max_examples is not None and int(self.max_examples) <= 0:
            raise ValueError("max_examples must be > 0 when provided")

        if int(self.replay_k) <= 0:
            raise ValueError("replay_k must be > 0")

        if self.mode == "replay_once" and self.replay_k != 2:
            raise ValueError(
                "For mode='replay_once', replay_k must remain 2. "
                "Use mode='replay_k' for other repeat counts."
            )


# ---------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------

@dataclass
class OutputConfig:
    """
    Controls where benchmark artifacts are written.
    """
    root_dir: str = "artifacts/benchmark_runs/memarch"
    write_workload_manifest: bool = True
    write_summary_json: bool = False

    def validate(self) -> None:
        if not str(self.root_dir).strip():
            raise ValueError("root_dir must be non-empty")

    def resolve_run_dir(self, benchmark_name: str, workload_mode: str) -> Path:
        return Path(self.root_dir).expanduser().resolve() / workload_mode / benchmark_name


# ---------------------------------------------------------------------
# Namespace / identity configuration
# ---------------------------------------------------------------------

@dataclass
class NamespaceConfig:
    """
    Controls identity scoping for memarch retrieval/storage.

    These fields are intentionally simple strings so benchmark scripts can
    assign stable identities for controlled experiments.
    """
    user_id: str = "user_a"
    session_id: str = "session_a"
    cohort_id: Optional[str] = None

    def validate(self) -> None:
        if not str(self.user_id).strip():
            raise ValueError("user_id must be non-empty")
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty")


# ---------------------------------------------------------------------
# Memory configuration
# ---------------------------------------------------------------------

@dataclass
class MemoryConfig:
    """
    Memarch-specific memory and retrieval controls.

    This config now supports explicit benchmark modes for:
    - exact-only retrieval
    - semantic retrieval used as generation context
    - semantic retrieval with direct bypass
    """
    ram_capacity_items: int = 64

    # Disk-backed persistent store for memarch benchmark runs
    disk_store_path: str = "artifacts/benchmark_runs/memarch/memory/memarch_benchmark.sqlite"

    # When True, delete any existing persistent disk store before the run starts.
    clear_disk_store_before_run: bool = False

    # Retrieval mode for benchmark comparison
    retrieval_mode: str = "exact_only"
    # Allowed:
    #   - "exact_only"
    #   - "semantic_context"
    #   - "semantic_bypass"

    # Exact / direct-return behavior
    promote_disk_hits_to_ram: bool = True
    return_memory_directly: bool = True

    # Semantic retrieval controls
    semantic_enabled: bool = False
    semantic_threshold_context: float = 0.85
    semantic_threshold_bypass: float = 1.01
    max_semantic_candidates: int = 5

    # Embedder controls
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "auto"
    embedding_local_files_only: bool = False

    # Storage / admission toggles
    enable_storage: bool = True
    store_in_ram: bool = True
    store_on_disk: bool = True

    def validate(self) -> None:
        if int(self.ram_capacity_items) <= 0:
            raise ValueError("ram_capacity_items must be > 0")

        if not str(self.disk_store_path).strip():
            raise ValueError("disk_store_path must be non-empty")

        valid_modes = {"exact_only", "semantic_context", "semantic_bypass"}
        if self.retrieval_mode not in valid_modes:
            raise ValueError(
                f"Invalid retrieval_mode: {self.retrieval_mode!r}. "
                f"Expected one of: {sorted(valid_modes)}"
            )

        if not (0.0 <= float(self.semantic_threshold_context) <= 1.0):
            raise ValueError("semantic_threshold_context must be in [0.0, 1.0]")

        if float(self.semantic_threshold_bypass) < 0.0:
            raise ValueError("semantic_threshold_bypass must be >= 0.0")

        if float(self.semantic_threshold_bypass) < float(self.semantic_threshold_context):
            raise ValueError(
                "semantic_threshold_bypass must be >= semantic_threshold_context"
            )

        if int(self.max_semantic_candidates) <= 0:
            raise ValueError("max_semantic_candidates must be > 0")

        if not str(self.embedding_model_id).strip():
            raise ValueError("embedding_model_id must be non-empty")

    def effective_semantic_enabled(self) -> bool:
        return self.retrieval_mode in {"semantic_context", "semantic_bypass"} and self.semantic_enabled

    def effective_bypass_enabled(self) -> bool:
        return self.retrieval_mode == "semantic_bypass" and self.semantic_enabled


# ---------------------------------------------------------------------
# Full benchmark configuration
# ---------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """
    Full configuration for one memarch benchmark run.

    Compatibility goal:
    Keep the most important baseline-style top-level fields available so
    benchmark orchestration can remain parallel across baseline and memarch.
    """

    # -----------------------------
    # Provenance / naming
    # -----------------------------
    benchmark_name: str = "memarch_longbench_benchmark"
    notes: str = ""

    # -----------------------------
    # Tier 2 (dataset / disk repo)
    # -----------------------------
    tier2_repo: str = ""

    # -----------------------------
    # Output
    # -----------------------------
    out_dir: str = "artifacts/benchmark_runs/memarch"

    # -----------------------------
    # Model
    # -----------------------------
    model_id: str = "microsoft/Phi-3-mini-128k-instruct"

    # -----------------------------
    # Run / generation parameters
    # -----------------------------
    max_input_tokens: int = 8192
    max_new_tokens: int = 64

    # -----------------------------
    # Device behavior
    # -----------------------------
    cpu_fallback_on_long: bool = False

    # -----------------------------
    # Benchmark-specific structure
    # -----------------------------
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    namespaces: NamespaceConfig = field(default_factory=NamespaceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @property
    def task_glob(self) -> str:
        return self.workload.task_glob

    @property
    def max_examples(self) -> Optional[int]:
        return self.workload.max_examples

    @property
    def max_cache_items(self) -> int:
        """
        Baseline-compatibility alias for RAM capacity.
        """
        return self.memory.ram_capacity_items

    def validate(self) -> None:
        if not self.tier2_repo or not str(self.tier2_repo).strip():
            raise ValueError("tier2_repo must be a non-empty path string")

        if int(self.max_input_tokens) <= 0:
            raise ValueError("max_input_tokens must be > 0")

        if int(self.max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")

        self.workload.validate()
        self.output.validate()
        self.namespaces.validate()
        self.memory.validate()

    def resolved_out_dir(self) -> str:
        run_dir = self.output.resolve_run_dir(
            benchmark_name=self.benchmark_name,
            workload_mode=self.workload.mode,
        )
        return str(run_dir)

    def resolved_disk_store_path(self) -> str:
        return str(Path(self.memory.disk_store_path).expanduser().resolve())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def for_all_tasks(
        cls,
        *,
        tier2_repo: str,
        benchmark_name: str,
        mode: str = "cold",
        max_examples: Optional[int] = 25,
        model_id: str = "microsoft/Phi-3-mini-128k-instruct",
        max_input_tokens: int = 8192,
        max_new_tokens: int = 64,
        ram_capacity_items: int = 64,
        disk_store_path: str = "artifacts/benchmark_runs/memarch/memory/memarch_benchmark.sqlite",
        clear_disk_store_before_run: bool = False,
        retrieval_mode: str = "exact_only",
        semantic_enabled: bool = False,
        semantic_threshold_context: float = 0.85,
        semantic_threshold_bypass: float = 1.01,
        max_semantic_candidates: int = 5,
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "auto",
        embedding_local_files_only: bool = False,
        promote_disk_hits_to_ram: bool = True,
        return_memory_directly: bool = True,
        enable_storage: bool = True,
        store_in_ram: bool = True,
        store_on_disk: bool = True,
        cpu_fallback_on_long: bool = False,
        out_root: str = "artifacts/benchmark_runs/memarch",
        user_id: str = "user_a",
        session_id: str = "session_a",
        cohort_id: Optional[str] = None,
        notes: str = "",
    ) -> "BenchmarkConfig":
        cfg = cls(
            benchmark_name=benchmark_name,
            notes=notes,
            tier2_repo=tier2_repo,
            out_dir=out_root,
            model_id=model_id,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            cpu_fallback_on_long=cpu_fallback_on_long,
            workload=WorkloadConfig(
                task_glob="",
                max_examples=max_examples,
                mode=mode,
            ),
            output=OutputConfig(root_dir=out_root),
            namespaces=NamespaceConfig(
                user_id=user_id,
                session_id=session_id,
                cohort_id=cohort_id,
            ),
            memory=MemoryConfig(
                ram_capacity_items=ram_capacity_items,
                disk_store_path=disk_store_path,
                clear_disk_store_before_run=clear_disk_store_before_run,
                retrieval_mode=retrieval_mode,
                semantic_enabled=semantic_enabled,
                semantic_threshold_context=semantic_threshold_context,
                semantic_threshold_bypass=semantic_threshold_bypass,
                max_semantic_candidates=max_semantic_candidates,
                embedding_model_id=embedding_model_id,
                embedding_device=embedding_device,
                embedding_local_files_only=embedding_local_files_only,
                promote_disk_hits_to_ram=promote_disk_hits_to_ram,
                return_memory_directly=return_memory_directly,
                enable_storage=enable_storage,
                store_in_ram=store_in_ram,
                store_on_disk=store_on_disk,
            ),
        )
        cfg.validate()
        return cfg
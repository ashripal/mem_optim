from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# Workload config
# =============================================================================

@dataclass
class WorkloadConfig:
    """
    Controls how raw LongBench examples are selected and replayed.

    Notes:
    - approx_interleaved is the key workload for paraphrase reuse evaluation
    - family_clustered can also be useful when testing paraphrase families
    """
    task_glob: str = ""
    max_examples: Optional[int] = 25

    mode: str = "cold"
    replay_k: int = 2

    shuffle: bool = False
    seed: int = 0

    total_requests: Optional[int] = None
    repeat_fraction: float = 0.0

    def validate(self) -> None:
        valid_modes = {
            "cold",
            "replay_once",
            "replay_k",
            "cache_pressure",
            "mixed_reuse",
            "exact_interleaved",
            "approx_interleaved",
            "family_clustered",
        }
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

        if self.mode == "mixed_reuse":
            if self.total_requests is None or int(self.total_requests) <= 0:
                raise ValueError("For mode='mixed_reuse', total_requests must be > 0")
            if not (0.0 <= float(self.repeat_fraction) < 1.0):
                raise ValueError("repeat_fraction must be in [0.0, 1.0)")
            if self.max_examples is None or int(self.max_examples) <= 0:
                raise ValueError("For mode='mixed_reuse', max_examples must be > 0")
            if int(self.total_requests) < int(self.max_examples):
                raise ValueError(
                    "For mode='mixed_reuse', total_requests must be >= max_examples"
                )

        if self.mode in {"approx_interleaved", "family_clustered"}:
            if self.max_examples is None or int(self.max_examples) <= 0:
                raise ValueError(
                    f"For mode={self.mode!r}, max_examples must be > 0"
                )


# =============================================================================
# Output config
# =============================================================================

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


# =============================================================================
# Namespace config
# =============================================================================

@dataclass
class NamespaceConfig:
    """
    Controls identity scoping for memarch retrieval/storage.
    """
    user_id: str = "user_a"
    session_id: str = "session_a"
    cohort_id: Optional[str] = None

    def validate(self) -> None:
        if not str(self.user_id).strip():
            raise ValueError("user_id must be non-empty")
        if not str(self.session_id).strip():
            raise ValueError("session_id must be non-empty")


# =============================================================================
# Memory / retrieval config
# =============================================================================

@dataclass
class MemoryConfig:
    """
    Memarch-specific memory and retrieval controls.

    Retrieval ladder intended by the verified paraphrase reuse spec:
      exact reuse
      -> lexical direct/context (optional)
      -> verified semantic direct reuse
      -> semantic context-only
      -> generation

    This config does not itself implement the logic; it only exposes the knobs
    used later by manager.py and policy.py.
    """
    ram_capacity_items: int = 64
    disk_store_path: str = "artifacts/benchmark_runs/memarch/memory/memarch_benchmark.sqlite"
    clear_disk_store_before_run: bool = False

    # Default to the intended stack for paraphrased workloads.
    retrieval_mode: str = "lexical_gated_direct_semantic_context"

    promote_disk_hits_to_ram: bool = True
    return_memory_directly: bool = True

    # -------------------------------------------------------------------------
    # Lexical retrieval
    # -------------------------------------------------------------------------
    lexical_enabled: bool = True
    lexical_threshold_context: float = 0.55
    lexical_threshold_bypass: float = 0.75
    lexical_top_k: int = 3
    prefer_same_source: bool = True
    safe_direct_reuse_tasks: list[str] = field(default_factory=lambda: ["trec"])

    # -------------------------------------------------------------------------
    # Semantic retrieval
    # -------------------------------------------------------------------------
    semantic_enabled: bool = True

    # Context threshold should be lower than direct-reuse threshold.
    semantic_threshold_context: float = 0.85

    # Verified semantic bypass threshold from the spec.
    semantic_threshold_bypass: float = 0.95

    max_semantic_candidates: int = 5

    # -------------------------------------------------------------------------
    # Verified semantic direct reuse gates
    # -------------------------------------------------------------------------
    allow_semantic_bypass: bool = True
    require_same_document_for_semantic_bypass: bool = True
    semantic_bypass_min_margin: float = 0.02
    require_evidence_support_for_semantic_bypass: bool = True
    semantic_direct_reuse_tasks: list[str] = field(
        default_factory=lambda: ["squad", "extractive_qa", "qa", "trec"]
    )
    semantic_bypass_max_answer_words: int = 12

    # -------------------------------------------------------------------------
    # Embedding model config
    # -------------------------------------------------------------------------
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "auto"
    embedding_local_files_only: bool = False

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------
    enable_storage: bool = True
    store_in_ram: bool = True
    store_on_disk: bool = True

    def validate(self) -> None:
        if int(self.ram_capacity_items) <= 0:
            raise ValueError("ram_capacity_items must be > 0")

        if not str(self.disk_store_path).strip():
            raise ValueError("disk_store_path must be non-empty")

        valid_modes = {
            "exact_only",
            "lexical_context",
            "lexical_gated_direct",
            "semantic_context",
            "semantic_bypass",
            "lexical_semantic_context",
            "lexical_gated_direct_semantic_context",
        }
        if self.retrieval_mode not in valid_modes:
            raise ValueError(
                f"Invalid retrieval_mode: {self.retrieval_mode!r}. "
                f"Expected one of: {sorted(valid_modes)}"
            )

        # ------------------------------
        # Lexical validation
        # ------------------------------
        if not (0.0 <= float(self.lexical_threshold_context) <= 1.0):
            raise ValueError("lexical_threshold_context must be in [0.0, 1.0]")

        if float(self.lexical_threshold_bypass) < 0.0:
            raise ValueError("lexical_threshold_bypass must be >= 0.0")

        if float(self.lexical_threshold_bypass) < float(self.lexical_threshold_context):
            raise ValueError(
                "lexical_threshold_bypass must be >= lexical_threshold_context"
            )

        if int(self.lexical_top_k) <= 0:
            raise ValueError("lexical_top_k must be > 0")

        # ------------------------------
        # Semantic validation
        # ------------------------------
        if not (0.0 <= float(self.semantic_threshold_context) <= 1.0):
            raise ValueError("semantic_threshold_context must be in [0.0, 1.0]")

        if not (0.0 <= float(self.semantic_threshold_bypass) <= 1.0):
            raise ValueError("semantic_threshold_bypass must be in [0.0, 1.0]")

        if float(self.semantic_threshold_bypass) < float(self.semantic_threshold_context):
            raise ValueError(
                "semantic_threshold_bypass must be >= semantic_threshold_context"
            )

        if int(self.max_semantic_candidates) <= 0:
            raise ValueError("max_semantic_candidates must be > 0")

        # ------------------------------
        # Verified semantic bypass gates
        # ------------------------------
        if float(self.semantic_bypass_min_margin) < 0.0:
            raise ValueError("semantic_bypass_min_margin must be >= 0.0")

        if int(self.semantic_bypass_max_answer_words) <= 0:
            raise ValueError("semantic_bypass_max_answer_words must be > 0")

        for task in self.safe_direct_reuse_tasks:
            if not str(task).strip():
                raise ValueError("safe_direct_reuse_tasks must not contain empty task names")

        for task in self.semantic_direct_reuse_tasks:
            if not str(task).strip():
                raise ValueError("semantic_direct_reuse_tasks must not contain empty task names")

        # ------------------------------
        # Embedding validation
        # ------------------------------
        if not str(self.embedding_model_id).strip():
            raise ValueError("embedding_model_id must be non-empty")

        valid_embed_devices = {"auto", "cuda", "mps", "cpu"}
        if str(self.embedding_device).strip().lower() not in valid_embed_devices:
            raise ValueError(
                f"embedding_device must be one of {sorted(valid_embed_devices)}"
            )

        # ------------------------------
        # Storage validation
        # ------------------------------
        if not self.enable_storage and (self.store_in_ram or self.store_on_disk):
            raise ValueError(
                "store_in_ram/store_on_disk cannot be True when enable_storage is False"
            )

        # exact_only means only exact reuse should be active; we allow the other
        # fields to remain populated for convenience across experiments.
        if self.retrieval_mode == "exact_only":
            pass

        # semantic_bypass mode is the narrow explicit benchmark mode for testing
        # verified semantic direct reuse. It should not run with semantic disabled.
        if self.retrieval_mode == "semantic_bypass" and not self.semantic_enabled:
            raise ValueError(
                "semantic_enabled must be True when retrieval_mode='semantic_bypass'"
            )

    # -------------------------------------------------------------------------
    # Effective feature flags
    # -------------------------------------------------------------------------

    def effective_lexical_enabled(self) -> bool:
        return self.retrieval_mode in {
            "lexical_context",
            "lexical_gated_direct",
            "lexical_semantic_context",
            "lexical_gated_direct_semantic_context",
        } and self.lexical_enabled

    def effective_lexical_direct_enabled(self) -> bool:
        return self.retrieval_mode in {
            "lexical_gated_direct",
            "lexical_gated_direct_semantic_context",
        } and self.lexical_enabled

    def effective_semantic_enabled(self) -> bool:
        return self.retrieval_mode in {
            "semantic_context",
            "semantic_bypass",
            "lexical_semantic_context",
            "lexical_gated_direct_semantic_context",
        } and self.semantic_enabled

    def effective_bypass_enabled(self) -> bool:
        """
        Whether verified semantic direct reuse should be enabled.

        This replaces the old looser notion of 'semantic_bypass'. The actual
        bypass remains gated by policy-time verification.
        """
        return (
            self.retrieval_mode in {"semantic_bypass"}
            and self.semantic_enabled
            and self.allow_semantic_bypass
        )


# =============================================================================
# Full benchmark config
# =============================================================================

@dataclass
class BenchmarkConfig:
    """
    Full configuration for one memarch benchmark run.
    """
    benchmark_name: str = "memarch_longbench_benchmark"
    notes: str = ""

    tier2_repo: str = ""
    out_dir: str = "artifacts/benchmark_runs/memarch"

    model_id: str = "microsoft/Phi-3-mini-128k-instruct"

    max_input_tokens: int = 8192
    max_new_tokens: int = 64

    decoding_mode: str = "greedy"  # "greedy" | "beam" | "sample"
    num_beams: int = 1
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = False

    device: str = "auto"
    dtype: str = "float32"
    local_files_only: bool = False
    cpu_fallback_on_long: bool = False

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
        return self.memory.ram_capacity_items

    def validate(self) -> None:
        if not self.tier2_repo or not str(self.tier2_repo).strip():
            raise ValueError("tier2_repo must be a non-empty path string")

        if int(self.max_input_tokens) <= 0:
            raise ValueError("max_input_tokens must be > 0")

        if int(self.max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")

        if self.decoding_mode not in {"greedy", "beam", "sample"}:
            raise ValueError("decoding_mode must be 'greedy', 'beam', or 'sample'")

        if int(self.num_beams) <= 0:
            raise ValueError("num_beams must be > 0")

        valid_devices = {"auto", "cuda", "mps", "cpu"}
        if str(self.device).strip().lower() not in valid_devices:
            raise ValueError(
                f"device must be one of {sorted(valid_devices)}, got {self.device!r}"
            )

        valid_dtypes = {"auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"}
        if str(self.dtype).strip().lower() not in valid_dtypes:
            raise ValueError(
                f"dtype must be one of {sorted(valid_dtypes)}, got {self.dtype!r}"
            )

        self.workload.validate()
        self.output.validate()
        self.namespaces.validate()
        self.memory.validate()

        if self.decoding_mode == "beam" and int(self.num_beams) <= 1:
            raise ValueError("num_beams must be > 1 when decoding_mode='beam'")

        if self.decoding_mode != "beam" and int(self.num_beams) != 1:
            raise ValueError("num_beams should be 1 unless decoding_mode='beam'")

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
        decoding_mode: str = "greedy",
        num_beams: int = 1,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: bool = False,
        device: str = "auto",
        dtype: str = "float32",
        local_files_only: bool = False,
        ram_capacity_items: int = 64,
        disk_store_path: str = "artifacts/benchmark_runs/memarch/memory/memarch_benchmark.sqlite",
        clear_disk_store_before_run: bool = False,
        retrieval_mode: str = "lexical_gated_direct_semantic_context",
        lexical_enabled: bool = True,
        lexical_context_threshold: float = 0.55,
        lexical_direct_threshold: float = 0.75,
        lexical_top_k: int = 3,
        prefer_same_source: bool = True,
        safe_direct_reuse_tasks: Optional[list[str]] = None,
        semantic_enabled: bool = True,
        semantic_threshold_context: float = 0.85,
        semantic_threshold_bypass: float = 0.95,
        max_semantic_candidates: int = 5,
        allow_semantic_bypass: bool = True,
        require_same_document_for_semantic_bypass: bool = True,
        semantic_bypass_min_margin: float = 0.02,
        require_evidence_support_for_semantic_bypass: bool = True,
        semantic_direct_reuse_tasks: Optional[list[str]] = None,
        semantic_bypass_max_answer_words: int = 12,
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
        replay_k: int = 2,
        shuffle: bool = False,
        seed: int = 0,
        total_requests: Optional[int] = None,
        repeat_fraction: float = 0.0,
    ) -> "BenchmarkConfig":
        """
        Convenience constructor used by benchmark scripts.
        """
        cfg = cls(
            benchmark_name=benchmark_name,
            notes=notes,
            tier2_repo=tier2_repo,
            out_dir=out_root,
            model_id=model_id,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            decoding_mode=decoding_mode,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            cpu_fallback_on_long=cpu_fallback_on_long,
            workload=WorkloadConfig(
                task_glob="",
                max_examples=max_examples,
                mode=mode,
                replay_k=replay_k,
                shuffle=shuffle,
                seed=seed,
                total_requests=total_requests,
                repeat_fraction=repeat_fraction,
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
                lexical_enabled=lexical_enabled,
                lexical_threshold_context=lexical_context_threshold,
                lexical_threshold_bypass=lexical_direct_threshold,
                lexical_top_k=lexical_top_k,
                prefer_same_source=prefer_same_source,
                safe_direct_reuse_tasks=safe_direct_reuse_tasks or ["trec"],
                semantic_enabled=semantic_enabled,
                semantic_threshold_context=semantic_threshold_context,
                semantic_threshold_bypass=semantic_threshold_bypass,
                max_semantic_candidates=max_semantic_candidates,
                allow_semantic_bypass=allow_semantic_bypass,
                require_same_document_for_semantic_bypass=require_same_document_for_semantic_bypass,
                semantic_bypass_min_margin=semantic_bypass_min_margin,
                require_evidence_support_for_semantic_bypass=require_evidence_support_for_semantic_bypass,
                semantic_direct_reuse_tasks=semantic_direct_reuse_tasks or ["squad", "extractive_qa", "qa", "trec"],
                semantic_bypass_max_answer_words=semantic_bypass_max_answer_words,
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
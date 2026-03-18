# baseline/benchmarks/configs.py
"""
Typed configuration objects for baseline benchmark runs.

Purpose:
- Keep benchmark settings explicit and serializable
- Separate workload-shaping knobs from model/cache/runtime knobs
- Remain compatible with the existing baseline runner/evaluator/tiers

This module does NOT:
- Parse CLI arguments
- Load data
- Run benchmarks
- Write logs

Those responsibilities belong elsewhere.
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
    Controls how raw LongBench examples are shaped into a benchmark stream.

    Notes:
    - task_glob is passed through to DiskLoader. Empty string means "all tasks".
    - max_examples applies to the *base* set before replay/expansion.
    - mode determines how the selected examples are replayed to create
      cache hits, misses, and eviction pressure.

    Supported modes (initial plan):
      - "cold": each selected example appears once
      - "replay_once": selected examples are replayed one additional time
      - "replay_k": selected examples are repeated replay_k total times
      - "cache_pressure": reuse is delayed to create possible evictions
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
            # Keep semantics explicit and less confusing.
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

    Suggested layout:
      artifacts/benchmark_runs/baseline/<mode>/
    """
    root_dir: str = "artifacts/benchmark_runs/baseline"
    write_workload_manifest: bool = True
    write_summary_json: bool = False

    def resolve_run_dir(self, benchmark_name: str, workload_mode: str) -> Path:
        """
        Resolve the directory under which a specific benchmark run should be written.
        """
        return Path(self.root_dir).expanduser().resolve() / workload_mode / benchmark_name


# ---------------------------------------------------------------------
# Full benchmark configuration
# ---------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """
    Full configuration for one baseline benchmark run.

    Compatibility goal:
    These field names intentionally mirror the existing baseline Config where useful,
    so existing components can read them via getattr(...) without modification.

    Fields like:
      - tier2_repo
      - task_glob
      - out_dir
      - model_id
      - max_examples
      - max_input_tokens
      - max_new_tokens
      - max_cache_items
      - cpu_fallback_on_long
    are preserved because the current baseline code expects them.
    """

    # -----------------------------
    # Provenance / naming
    # -----------------------------
    benchmark_name: str = "baseline_longbench_benchmark"
    notes: str = ""

    # -----------------------------
    # Tier 2 (Disk)
    # -----------------------------
    tier2_repo: str = ""

    # -----------------------------
    # Output
    # -----------------------------
    out_dir: str = "artifacts/benchmark_runs/baseline"

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
    # Tier 1 (RAM)
    # -----------------------------
    max_cache_items: int = 64

    # -----------------------------
    # Device behavior
    # -----------------------------
    cpu_fallback_on_long: bool = False

    # -----------------------------
    # Benchmark-specific structure
    # -----------------------------
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -----------------------------------------------------------------
    # Compatibility properties
    # -----------------------------------------------------------------

    @property
    def task_glob(self) -> str:
        """
        Compatibility property for DiskLoader / existing runner interfaces.
        """
        return self.workload.task_glob

    @property
    def max_examples(self) -> Optional[int]:
        """
        Compatibility property for DiskLoader / existing runner interfaces.

        Important:
        This refers to the base dataset selection before workload expansion.
        """
        return self.workload.max_examples

    # -----------------------------------------------------------------
    # Validation / serialization helpers
    # -----------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate that the config is internally consistent before execution.
        """
        if not self.tier2_repo or not str(self.tier2_repo).strip():
            raise ValueError("tier2_repo must be a non-empty path string")

        if int(self.max_input_tokens) <= 0:
            raise ValueError("max_input_tokens must be > 0")

        if int(self.max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")

        if int(self.max_cache_items) <= 0:
            raise ValueError("max_cache_items must be > 0")

        self.workload.validate()

    def resolved_out_dir(self) -> str:
        """
        Resolve the directory that should contain artifacts for this benchmark.

        This is what benchmark execution code should use when deciding where
        to write the run JSONL and related manifests.
        """
        run_dir = self.output.resolve_run_dir(
            benchmark_name=self.benchmark_name,
            workload_mode=self.workload.mode,
        )
        return str(run_dir)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a plain dict for JSON logging / manifests.
        """
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
        max_cache_items: int = 64,
        cpu_fallback_on_long: bool = False,
        out_root: str = "artifacts/benchmark_runs/baseline",
        notes: str = "",
    ) -> "BenchmarkConfig":
        """
        Convenience constructor for the common case:
        run across all LongBench task files.

        task_glob="" means the DiskLoader should include every discovered JSONL task.
        """
        cfg = cls(
            benchmark_name=benchmark_name,
            notes=notes,
            tier2_repo=tier2_repo,
            out_dir=out_root,
            model_id=model_id,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            max_cache_items=max_cache_items,
            cpu_fallback_on_long=cpu_fallback_on_long,
            workload=WorkloadConfig(
                task_glob="",
                max_examples=max_examples,
                mode=mode,
            ),
            output=OutputConfig(root_dir=out_root),
        )
        cfg.validate()
        return cfg
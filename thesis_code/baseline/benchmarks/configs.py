# baseline/benchmarks/configs.py
"""
Typed configuration objects for stateless baseline benchmark runs.

Purpose:
- Keep benchmark settings explicit and serializable
- Separate workload-shaping knobs from model/runtime knobs
- Remain compatible with the current stateless baseline benchmark flow

TRUE BASELINE:
- No RAM cache
- No memory reuse
- Every request goes directly to Tier 0 compute

This module does NOT:
- Parse CLI arguments
- Load data
- Run benchmarks
- Write logs
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
    - max_examples applies to the base set before replay/expansion.
    - mode determines how selected examples are repeated in the request stream.

    Supported modes:
      - "cold": each selected example appears once
      - "replay_once": selected examples are replayed one additional time
      - "replay_k": selected examples are repeated replay_k total times
      - "mixed_reuse": synthetic mix of first-seen and repeated requests

    Important:
    These are workload-shaping modes only. They do NOT imply system-side caching.
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
        valid_modes = {"cold", "replay_once", "replay_k", "mixed_reuse"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid workload mode: {self.mode!r}. "
                f"Expected one of: {sorted(valid_modes)}"
            )

        if self.max_examples is not None and int(self.max_examples) <= 0:
            raise ValueError("max_examples must be > 0 when provided")

        if int(self.replay_k) <= 0:
            raise ValueError("replay_k must be > 0")

        if self.mode == "replay_once" and int(self.replay_k) != 2:
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


# ---------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------

@dataclass
class OutputConfig:
    """
    Controls where benchmark artifacts are written.

    Suggested layout:
      artifacts/benchmark_runs/baseline/<mode>/<benchmark_name>/
    """
    root_dir: str = "artifacts/benchmark_runs/baseline"
    write_workload_manifest: bool = True
    write_summary_json: bool = False

    def resolve_run_dir(self, benchmark_name: str, workload_mode: str) -> Path:
        return Path(self.root_dir).expanduser().resolve() / workload_mode / benchmark_name


# ---------------------------------------------------------------------
# Full benchmark configuration
# ---------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """
    Full configuration for one stateless baseline benchmark run.

    Compatibility goal:
    These field names intentionally mirror the current stateless baseline runner
    where useful, so existing components can read them via getattr(...) without
    extra translation.

    Preserved fields:
      - tier2_repo
      - task_glob (via property)
      - out_dir
      - model_id
      - max_examples (via property)
      - max_input_tokens
      - max_new_tokens
      - device
      - dtype
      - cpu_fallback_on_long
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
    # Device behavior
    # -----------------------------
    device: str = "auto"
    dtype: str = "auto"
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
        if not self.tier2_repo or not str(self.tier2_repo).strip():
            raise ValueError("tier2_repo must be a non-empty path string")

        if int(self.max_input_tokens) <= 0:
            raise ValueError("max_input_tokens must be > 0")

        if int(self.max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")

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

    def resolved_out_dir(self) -> str:
        """
        Resolve the directory that should contain artifacts for this benchmark.
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
        device: str = "auto",
        dtype: str = "auto",
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
            device=device,
            dtype=dtype,
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
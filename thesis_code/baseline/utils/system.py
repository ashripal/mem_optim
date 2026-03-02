# utils/system.py
"""
System utilities for the LongBench baseline.

Provides lightweight, dependency-minimal helpers for:
- Process RSS measurement (MB)
- Simple timing context manager
- Basic environment/device introspection (optional but helpful for provenance)

This module should NOT:
- Load models
- Read datasets
- Write JSONL logs
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import psutil


def get_rss_mb(pid: Optional[int] = None) -> float:
    """
    Return resident set size (RSS) memory for the given process in MB.
    Defaults to the current process.
    """
    if pid is None:
        pid = os.getpid()
    proc = psutil.Process(pid)
    rss_bytes = proc.memory_info().rss
    return rss_bytes / (1024.0 * 1024.0)


@dataclass
class Timer:
    """
    Simple timing context manager.

    Usage:
        with Timer() as t:
            ...
        print(t.elapsed_s)
    """
    start_s: float = 0.0
    end_s: float = 0.0
    elapsed_s: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_s = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end_s = time.time()
        self.elapsed_s = self.end_s - self.start_s


def get_system_info() -> Dict[str, Any]:
    """
    Return a small set of system/provenance info for logging.
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pid": os.getpid(),
    }

    # Optional torch device visibility (avoid hard dependency elsewhere)
    try:
        import torch  # type: ignore

        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        info["cuda_device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
    except Exception:
        info["torch_version"] = None
        info["cuda_available"] = None
        info["mps_available"] = None
        info["cuda_device_count"] = None

    return info
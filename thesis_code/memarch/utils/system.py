# memarch/utils/system.py
"""
System utilities for memarch.

Goals:
- Portable across macOS (Apple silicon) and Jetson Orin Nano (Linux)
- No hard dependency on optional libraries (psutil, pynvml, jetson-stats)
- Provide small helpers useful for logging and evaluation

This module is intentionally lightweight and safe to import anywhere.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DeviceInfo:
    os: str
    platform: str
    machine: str
    processor: str
    python_version: str
    hostname: str


def get_device_info() -> DeviceInfo:
    return DeviceInfo(
        os=platform.system(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_version(platform.python_version()),
        hostname=os.uname().nodename if hasattr(os, "uname") else "unknown",
    )


def get_rss_mb() -> Optional[float]:
    """
    Return current process RSS in MB using psutil if available.

    Returns None if psutil isn't installed or an error occurs.
    """
    try:
        import psutil  # type: ignore
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def get_cpu_count() -> int:
    """Return logical CPU count (best effort)."""
    try:
        c = os.cpu_count()
        return int(c) if c is not None else 1
    except Exception:
        return 1


def get_gpu_snapshot() -> Optional[Dict[str, Any]]:
    """
    Best-effort GPU snapshot for NVIDIA GPUs using NVML (pynvml).

    On Jetson Orin Nano, NVML may be available depending on environment.
    Returns None if unavailable.
    """
    try:
        from pynvml import (  # type: ignore
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetName,
        )

        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        mem = nvmlDeviceGetMemoryInfo(h)
        util = nvmlDeviceGetUtilizationRates(h)
        name = nvmlDeviceGetName(h)
        # nvml returns bytes
        return {
            "gpu_name": name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name),
            "gpu_mem_total_mb": float(mem.total) / (1024.0 * 1024.0),
            "gpu_mem_used_mb": float(mem.used) / (1024.0 * 1024.0),
            "gpu_mem_free_mb": float(mem.free) / (1024.0 * 1024.0),
            "gpu_util_percent": int(util.gpu),
            "gpu_mem_util_percent": int(util.memory),
        }
    except Exception:
        return None


class Timer:
    """
    Simple context manager timer using perf_counter.

    Usage:
      with Timer() as t:
          ...
      elapsed_ms = t.elapsed_ms
    """
    def __init__(self) -> None:
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        self._t1 = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._t1 = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        if self._t0 is None:
            return 0.0
        t1 = self._t1 if self._t1 is not None else time.perf_counter()
        return float(t1 - self._t0)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0
# memarch/main.py
"""
memarch main entrypoints.

This module provides small, script-friendly functions to:
- run a memarch evaluation given a dataset iterator
- summarize a run log
- generate basic plots

Keep this file thin: it wires together config + stores + manager + pipeline.

NOTE:
- We do NOT implement the actual model backend here. The generator is passed in or built
  from memarch.models.generator (to be implemented next).
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from memarch.config import MemArchConfig
from memarch.memory.disk_store import DiskStoreSQLite
from memarch.memory.ram_store import RamStoreLRU
from memarch.memory.manager import MemoryManager, MemoryManagerConfig
from memarch.pipeline.evaluator import Evaluator
from memarch.pipeline.logging import JsonlLogger, RunInfo
from memarch.pipeline.runner import Runner, RunSummary
from memarch.analysis.summarize import summarize_run, format_summary
from memarch.analysis.plot import plot_latency_series, plot_latency_histogram, plot_memory_hit_timeline
from memarch.memory.schema import MemoryQuery


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def make_run_log_path(cfg: MemArchConfig, run_id: str) -> str:
    run_dir = os.path.join(cfg.paths.runs_dir, run_id)
    _ensure_dir(run_dir)
    return os.path.join(run_dir, "log.jsonl")


def build_manager(cfg: MemArchConfig) -> MemoryManager:
    """
    Construct RAM/Disk stores + MemoryManager from config.
    """
    ram = RamStoreLRU(max_mb=cfg.memory.ram_max_mb)
    disk = DiskStoreSQLite(cfg.paths.disk_store_path)

    mm_cfg = MemoryManagerConfig(
        promote_disk_hits_to_ram=cfg.memory.promote_disk_hits_to_ram,
        return_memory_directly=cfg.memory.return_memory_directly,
    )

    return MemoryManager(ram=ram, disk=disk, cfg=mm_cfg)


def run(
    *,
    examples: Iterable[Tuple[str, str, MemoryQuery]],
    generator: Any,
    cfg: Optional[MemArchConfig] = None,
    notes: Optional[str] = None,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run memarch over an example iterator and return a dict with:
      - run_info
      - run_summary (simple averages)
      - log_path

    Args:
      examples: iterable yielding (example_id, task, MemoryQuery)
      generator: object implementing GeneratorLike (see pipeline/evaluator.py)
      cfg: optional MemArchConfig
      notes: optional run notes stored in log header
      log_path: optional override for JSONL log path
    """
    cfg = cfg or MemArchConfig.default()

    # Create run info and log path
    run_info = RunInfo.create(notes=notes)
    log_path = log_path or make_run_log_path(cfg, run_info.run_id)

    manager = build_manager(cfg)

    with JsonlLogger(log_path, run_info=run_info) as logger:
        evaluator = Evaluator(manager=manager, generator=generator, logger=logger)
        runner = Runner(evaluator=evaluator, logger=logger)

        summary: RunSummary = runner.run(
            examples=examples,
            log_query_text=cfg.eval.log_query_text,
            max_examples=cfg.eval.max_examples,
        )

    return {
        "run_info": asdict(run_info),
        "run_summary": asdict(summary),
        "log_path": log_path,
    }


def summarize(log_path: str) -> Dict[str, Any]:
    """
    Summarize a run log into aggregate metrics and return as dict.
    """
    s = summarize_run(log_path)
    return s.to_dict()


def summarize_text(log_path: str) -> str:
    """
    Summarize a run log and return a human-readable text summary.
    """
    s = summarize_run(log_path)
    return format_summary(s)


def plot_all(log_path: str, out_dir: str) -> Dict[str, str]:
    """
    Generate a small set of standard plots for a run.

    Saves:
      - latency_total.png
      - latency_memory.png
      - latency_generation.png
      - latency_total_hist.png
      - memory_hit_timeline.png

    Returns: dict of plot_name -> file path
    """
    _ensure_dir(out_dir)

    paths: Dict[str, str] = {}

    p_total = os.path.join(out_dir, "latency_total.png")
    plot_latency_series(log_path, p_total, which="total")
    paths["latency_total"] = p_total

    p_mem = os.path.join(out_dir, "latency_memory.png")
    plot_latency_series(log_path, p_mem, which="memory")
    paths["latency_memory"] = p_mem

    p_gen = os.path.join(out_dir, "latency_generation.png")
    plot_latency_series(log_path, p_gen, which="generation")
    paths["latency_generation"] = p_gen

    p_hist = os.path.join(out_dir, "latency_total_hist.png")
    plot_latency_histogram(log_path, p_hist, which="total")
    paths["latency_total_hist"] = p_hist

    p_hit = os.path.join(out_dir, "memory_hit_timeline.png")
    plot_memory_hit_timeline(log_path, p_hit)
    paths["memory_hit_timeline"] = p_hit

    return paths
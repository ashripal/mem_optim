# demo/state.py
"""
demo/state.py

Small helpers for Streamlit session_state:
- Centralize the keys used across the demo
- Provide deterministic defaults
- Provide safe getters/setters that avoid KeyError
- Keep runtime-only objects (manager/ram/disk/generator) in a consistent place

This file is intentionally lightweight for the initial demo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------------------
# Session-state keys (single source of truth)
# --------------------------------------------------------------------------------------

K_CFG = "cfg"
K_IDENTITY = "identity"
K_DOC = "doc"
K_MEM = "mem"
K_TURNS = "turns"


# --------------------------------------------------------------------------------------
# Default configuration blocks
# --------------------------------------------------------------------------------------

def default_cfg() -> Dict[str, Any]:
    """
    Demo configuration defaults.

    NOTE: These defaults are intentionally conservative and designed to work well on:
      - MacBook (CPU)
      - Jetson Orin Nano
    """
    return {
        "ram_max_mb": 256,
        "promote_disk_hits_to_ram": True,
        "return_memory_directly": True,
        "enable_global_writes": False,
        "model_mode": "fake",  # "fake" | "llama_cpp" (future)
        "show_prompt_preview": True,
        "selector_top_k": 2,
    }


def default_identity() -> Dict[str, Any]:
    """
    Identity defaults used to demonstrate namespace isolation:
      - session:user ordering (session wins)
      - changing user/session changes namespaces (no leakage)
    """
    return {
        "user_id": "user_a",
        "session_id": "session_a",
        "cohort_id": "",
        "task": "pdf_qa",
        "model_id": "mistral-7b-instruct",
        "prompt_version": "v1",
    }


def default_doc() -> Optional[Dict[str, Any]]:
    """
    No document loaded initially.
    Once loaded, this becomes a dict with fields like:
      pdf_name, doc_signature, chunks, ingest_stats, ...
    """
    return None


def default_turns() -> List[Dict[str, Any]]:
    """
    Turn history.
    Each element should be a fully renderable dict (see app.py turn schema).
    """
    return []


def default_disk_path(project_root: Optional[str] = None) -> str:
    """
    Determine a stable disk path for the demo's SQLite store.

    By default, uses:
      <cwd>/artifacts/demo/demo_memory.sqlite

    You can pass project_root if you prefer:
      <project_root>/artifacts/demo/demo_memory.sqlite
    """
    base = project_root or os.getcwd()
    artifacts = os.path.join(base, "artifacts", "demo")
    os.makedirs(artifacts, exist_ok=True)
    return os.path.join(artifacts, "demo_memory.sqlite")


def default_mem(project_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Memory runtime objects.

    IMPORTANT:
    - ram/disk/manager/generator are runtime-only Python objects
      (not serializable; they reset on browser refresh)
    - disk_path is a string and persists across sessions/runs
    """
    return {
        "disk_path": default_disk_path(project_root=project_root),
        "ram": None,
        "disk": None,
        "manager": None,
        "run_id": None,
        "generator": None,
    }


# --------------------------------------------------------------------------------------
# Typed views (optional ergonomic layer)
# --------------------------------------------------------------------------------------

@dataclass
class DemoState:
    """
    A small typed wrapper around session_state-like dict.

    This is optional but useful to reduce typos and keep the codebase consistent.
    """
    cfg: Dict[str, Any]
    identity: Dict[str, Any]
    doc: Optional[Dict[str, Any]]
    mem: Dict[str, Any]
    turns: List[Dict[str, Any]]


def ensure_state(ss: Dict[str, Any], *, project_root: Optional[str] = None) -> DemoState:
    """
    Ensure session_state has all required keys populated with defaults.

    Parameters:
      ss: streamlit.session_state (dict-like)
      project_root: optional path to ensure disk_path is stable

    Returns:
      DemoState wrapper
    """
    if K_CFG not in ss:
        ss[K_CFG] = default_cfg()

    if K_IDENTITY not in ss:
        ss[K_IDENTITY] = default_identity()

    if K_DOC not in ss:
        ss[K_DOC] = default_doc()

    if K_MEM not in ss:
        ss[K_MEM] = default_mem(project_root=project_root)

    if K_TURNS not in ss:
        ss[K_TURNS] = default_turns()

    return DemoState(
        cfg=ss[K_CFG],
        identity=ss[K_IDENTITY],
        doc=ss[K_DOC],
        mem=ss[K_MEM],
        turns=ss[K_TURNS],
    )


# --------------------------------------------------------------------------------------
# Convenience helpers
# --------------------------------------------------------------------------------------

def reset_turns(ss: Dict[str, Any]) -> None:
    """Clear turn history."""
    ss[K_TURNS] = []


def set_doc(ss: Dict[str, Any], doc: Optional[Dict[str, Any]], *, clear_turns: bool = True) -> None:
    """
    Set current document.
    Typically you clear turn history when the document changes for a clean demo.
    """
    ss[K_DOC] = doc
    if clear_turns:
        reset_turns(ss)


def clear_ram_only(ss: Dict[str, Any]) -> None:
    """
    Simulate a restart by clearing RAM but leaving Disk intact.

    This helper assumes ss[K_MEM]["ram"] has a .clear() method (RamStoreLRU).
    """
    mem = ss.get(K_MEM) or {}
    ram = mem.get("ram")
    if ram is not None and hasattr(ram, "clear"):
        ram.clear()
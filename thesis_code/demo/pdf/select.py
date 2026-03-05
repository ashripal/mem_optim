# demo/pdf/select.py
"""
demo/pdf/select.py

Chunk selection for PDF context injection (Phase 1).

IMPORTANT:
- This is *not* a semantic retrieval system (no embeddings, no vector DB).
- For the initial demo, we use deterministic lexical scoring so the system:
    - works offline
    - works on MacBook + Jetson
    - is simple to explain to a thesis committee
- The novelty you are demoing is the multi-tier memory + LLM bypass + persistence,
  not a retrieval algorithm.

Selection strategies implemented:
- lexical_overlap: token intersection count
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """
    Simple deterministic tokenization.
    """
    return [t.lower() for t in _WORD_RE.findall(text or "")]


@dataclass(frozen=True)
class SelectionConfig:
    """
    Configuration for context selection.
    """
    method: str = "lexical_overlap"  # currently only supported method
    top_k: int = 2                  # number of chunks to include
    min_overlap: int = 1            # minimum overlap to be considered relevant


def _score_lexical_overlap(question: str, chunk_text: str) -> int:
    q_tokens = set(tokenize(question))
    c_tokens = set(tokenize(chunk_text))
    if not q_tokens or not c_tokens:
        return 0
    return len(q_tokens.intersection(c_tokens))


def select_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    cfg: Optional[SelectionConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Select the top-k chunks for a question.

    Returns:
      selected_chunks: list of chunk dicts (same schema as doc["chunks"])
      meta: dict with selection_ms + debug info

    Determinism:
      - tie-breaks are resolved by chunk_id (lexicographic) to keep demos reproducible
    """
    cfg = cfg or SelectionConfig()

    t0 = time.perf_counter()

    method = (cfg.method or "").lower().strip()
    if method != "lexical_overlap":
        raise ValueError(f"Unsupported selection method '{cfg.method}'. Supported: lexical_overlap")

    q = (question or "").strip()
    if not q or not chunks:
        return [], {"selection_ms": (time.perf_counter() - t0) * 1000.0, "method": method, "reason": "empty_input"}

    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for ch in chunks:
        ch_text = ch.get("text", "") or ""
        ch_id = ch.get("chunk_id", "") or ""
        score = _score_lexical_overlap(q, ch_text)
        scored.append((score, ch_id, ch))

    # Sort: best score first, then stable chunk_id
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Filter by min_overlap to avoid injecting irrelevant chunks
    selected = [x[2] for x in scored if x[0] >= int(cfg.min_overlap)][: max(1, int(cfg.top_k))]

    selection_ms = (time.perf_counter() - t0) * 1000.0

    meta = {
        "selection_ms": selection_ms,
        "method": method,
        "top_k": int(cfg.top_k),
        "min_overlap": int(cfg.min_overlap),
        "selected_chunk_ids": [c.get("chunk_id", "") for c in selected],
    }
    return selected, meta


def build_dataset_context(selected_chunks: List[Dict[str, Any]]) -> str:
    """
    Join selected chunks into a single dataset_context string.
    """
    if not selected_chunks:
        return ""
    return "\n\n".join([(c.get("text", "") or "") for c in selected_chunks])
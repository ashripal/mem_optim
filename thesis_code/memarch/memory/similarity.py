# memarch/memory/similarity.py
"""
Similarity utilities for semantic retrieval (Phase 2).

Phase 1 uses exact-match only, but we include these functions now so:
- unit tests can validate scoring determinism
- later semantic retrieval can be added without reorganizing modules

This module should stay dependency-light (pure Python).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


Vector = Sequence[float]


def l2_norm(v: Vector) -> float:
    """Compute L2 norm."""
    return math.sqrt(sum(float(x) * float(x) for x in v))


def dot(a: Vector, b: Vector) -> float:
    """Compute dot product (assumes same length)."""
    return sum(float(x) * float(y) for x, y in zip(a, b))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """
    Cosine similarity in [-1, 1] (typically [0,1] for embedding vectors depending on model).

    Returns 0.0 if either vector has zero norm.
    """
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def top_k_cosine(
    query: Vector,
    candidates: Iterable[Tuple[str, Vector]],
    *,
    k: int = 5,
    min_score: float = 0.0,
) -> List[Tuple[str, float]]:
    """
    Compute cosine similarities and return top-k keys by score.

    Args:
      query: query embedding
      candidates: iterable of (key, vector)
      k: number of results
      min_score: filter threshold

    Returns:
      list of (key, score) sorted by score desc
    """
    if k <= 0:
        return []
    q = list(query)
    if len(q) == 0:
        return []

    scored: List[Tuple[str, float]] = []
    for key, vec in candidates:
        v = list(vec)
        if len(v) != len(q):
            continue
        s = float(cosine_similarity(q, v))
        if s >= min_score:
            scored.append((key, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0,1] via min-max scaling.

    Useful if you later combine heterogeneous signals.
    If all scores equal, returns all 1.0.
    """
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]
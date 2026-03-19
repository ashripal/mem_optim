# memarch/memory/similarity.py
"""
Similarity utilities for semantic retrieval.

Phase 1 usage:
- exact-match retrieval remains primary
- semantic retrieval uses brute-force cosine similarity over stored embeddings
- this module stays dependency-light and pure Python for easy testing

This file intentionally does not know about storage tiers, MemoryItem, or indexing.
It only provides vector scoring and ranking helpers.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, TypeVar


Vector = Sequence[float]
T = TypeVar("T")


def _as_float_list(v: Vector) -> List[float]:
    """Convert a vector-like input to a concrete list of floats."""
    return [float(x) for x in v]


def l2_norm(v: Vector) -> float:
    """Compute L2 norm."""
    vec = _as_float_list(v)
    return math.sqrt(sum(x * x for x in vec))


def dot(a: Vector, b: Vector) -> float:
    """Compute dot product. Raises if lengths differ."""
    va = _as_float_list(a)
    vb = _as_float_list(b)
    if len(va) != len(vb):
        raise ValueError(f"Vector length mismatch: {len(va)} != {len(vb)}")
    return sum(x * y for x, y in zip(va, vb))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """
    Compute cosine similarity.

    Returns:
      value in [-1, 1]

    Notes:
    - Returns 0.0 if either vector has zero norm
    - Raises ValueError if vector lengths differ
    """
    va = _as_float_list(a)
    vb = _as_float_list(b)

    if len(va) != len(vb):
        raise ValueError(f"Vector length mismatch: {len(va)} != {len(vb)}")

    if len(va) == 0:
        return 0.0

    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na == 0.0 or nb == 0.0:
        return 0.0

    score = sum(x * y for x, y in zip(va, vb)) / (na * nb)

    # Numerical safety: floating point can rarely drift slightly outside bounds.
    return max(-1.0, min(1.0, float(score)))


def top_k_similar(
    query: Vector,
    candidates: Iterable[Tuple[Vector, T]],
    *,
    k: int = 5,
    min_score: float = 0.0,
) -> List[Tuple[float, T]]:
    """
    Compute cosine similarity between a query vector and candidate vectors.

    Args:
      query:
        Query embedding vector
      candidates:
        Iterable of (candidate_vector, payload)
      k:
        Number of top results to return
      min_score:
        Minimum cosine similarity required to keep a candidate

    Returns:
      List of (score, payload), sorted by descending score

    Behavior:
    - candidates with mismatched vector dimensions are skipped
    - zero-length query returns []
    - k <= 0 returns []
    """
    if k <= 0:
        return []

    q = _as_float_list(query)
    if len(q) == 0:
        return []

    scored: List[Tuple[float, T]] = []

    for vec, payload in candidates:
        try:
            score = float(cosine_similarity(q, vec))
        except ValueError:
            # Skip dimension-mismatched candidates rather than crashing retrieval.
            continue

        if score >= min_score:
            scored.append((score, payload))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def top_k_cosine(
    query: Vector,
    candidates: Iterable[Tuple[str, Vector]],
    *,
    k: int = 5,
    min_score: float = 0.0,
) -> List[Tuple[str, float]]:
    """
    Backward-compatible helper for key-based ranking.

    Args:
      query:
        Query embedding vector
      candidates:
        Iterable of (key, vector)
      k:
        Number of results
      min_score:
        Minimum cosine similarity required to keep a candidate

    Returns:
      List of (key, score), sorted by descending score
    """
    ranked = top_k_similar(
        query,
        ((vec, key) for key, vec in candidates),
        k=k,
        min_score=min_score,
    )
    return [(key, score) for score, key in ranked]


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0,1] via min-max scaling.

    Useful if you later combine heterogeneous signals.
    If all scores are equal, returns all 1.0.
    """
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(float(s) - mn) / (mx - mn) for s in scores]
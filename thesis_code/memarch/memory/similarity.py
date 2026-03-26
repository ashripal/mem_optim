# memarch/memory/similarity.py
"""
Similarity utilities for retrieval.

Phase 1 / current usage:
- exact-match retrieval remains primary
- semantic retrieval can use brute-force cosine similarity over stored embeddings
- lexical retrieval provides a lightweight approximate matching signal
- this module stays dependency-light and pure Python for easy testing

This file intentionally does not know about storage tiers, MemoryItem, or indexing.
It only provides vector scoring, lexical scoring, and ranking helpers.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Sequence, Tuple, TypeVar


Vector = Sequence[float]
T = TypeVar("T")


# -----------------------------------------------------------------------------
# Vector helpers (existing semantic retrieval support)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Lexical retrieval helpers
# -----------------------------------------------------------------------------

def _as_token_list(tokens: Sequence[str] | None) -> List[str]:
    """
    Normalize token-like input to a concrete list[str], dropping empty values.
    """
    if not tokens:
        return []
    out: List[str] = []
    for t in tokens:
        s = str(t).strip()
        if s:
            out.append(s)
    return out


def jaccard_score(a_tokens: Sequence[str] | None, b_tokens: Sequence[str] | None) -> float:
    """
    Compute Jaccard similarity over token sets.

    Returns:
      value in [0, 1]

    Behavior:
    - both empty -> 1.0
    - one empty -> 0.0
    """
    a = set(_as_token_list(a_tokens))
    b = set(_as_token_list(b_tokens))

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return float(inter / union)


def token_f1_score(a_tokens: Sequence[str] | None, b_tokens: Sequence[str] | None) -> float:
    """
    Compute token-level F1 using multiset overlap.

    Returns:
      value in [0, 1]

    Behavior:
    - both empty -> 1.0
    - one empty -> 0.0
    """
    a = _as_token_list(a_tokens)
    b = _as_token_list(b_tokens)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    ca = Counter(a)
    cb = Counter(b)
    overlap = sum((ca & cb).values())

    precision = overlap / max(1, len(a))
    recall = overlap / max(1, len(b))
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def lexical_score(
    query_norm: str,
    query_tokens: Sequence[str] | None,
    item_norm: str,
    item_tokens: Sequence[str] | None,
    *,
    same_source: bool = False,
    exact_bonus: float = 0.10,
    same_source_bonus: float = 0.10,
) -> float:
    """
    Compute a lightweight lexical similarity score in [0, 1].

    Signals combined:
    - token F1 (primary)
    - Jaccard overlap
    - exact-normalized match bonus
    - same-source bonus

    Suggested usage:
    - exact canonical lookup first
    - lexical_score for approximate candidate ranking
    - high threshold for direct reuse
    - lower threshold for context-assisted generation

    Args:
      query_norm:
        Normalized query text
      query_tokens:
        Tokenized normalized query
      item_norm:
        Normalized candidate item text
      item_tokens:
        Tokenized normalized candidate item text
      same_source:
        Whether query and candidate come from the same source/document/file
      exact_bonus:
        Bonus added if normalized texts match exactly
      same_source_bonus:
        Bonus added if same_source is True

    Returns:
      float in [0, 1]
    """
    qn = (query_norm or "").strip()
    inorm = (item_norm or "").strip()
    qt = _as_token_list(query_tokens)
    it = _as_token_list(item_tokens)

    if not qn and not inorm:
        return 1.0
    if not qn or not inorm:
        return 0.0

    tf1 = token_f1_score(qt, it)
    jac = jaccard_score(qt, it)

    score = 0.75 * tf1 + 0.15 * jac

    if qn == inorm:
        score += float(exact_bonus)
    if same_source:
        score += float(same_source_bonus)

    return max(0.0, min(1.0, float(score)))


def top_k_lexical(
    query_norm: str,
    query_tokens: Sequence[str] | None,
    candidates: Iterable[Tuple[str, Sequence[str], T]],
    *,
    k: int = 5,
    min_score: float = 0.0,
    same_source_ids: Sequence[int] | None = None,
) -> List[Tuple[float, T]]:
    """
    Rank lexical candidates by `lexical_score`.

    Args:
      query_norm:
        Normalized query text
      query_tokens:
        Tokenized normalized query
      candidates:
        Iterable of (item_norm, item_tokens, payload)
      k:
        Number of top results to return
      min_score:
        Minimum lexical score required to keep a candidate
      same_source_ids:
        Optional indices of candidates that should receive same_source=True.
        This is a convenience hook; most callers will likely score candidates
        one at a time and set same_source directly instead.

    Returns:
      List of (score, payload), sorted by descending score.
    """
    if k <= 0:
        return []

    same_source_idx = set(int(x) for x in (same_source_ids or []))
    scored: List[Tuple[float, T]] = []

    for idx, (item_norm, item_tokens, payload) in enumerate(candidates):
        score = lexical_score(
            query_norm=query_norm,
            query_tokens=query_tokens,
            item_norm=item_norm,
            item_tokens=item_tokens,
            same_source=(idx in same_source_idx),
        )
        if score >= min_score:
            scored.append((score, payload))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
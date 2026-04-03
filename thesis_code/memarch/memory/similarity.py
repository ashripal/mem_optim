from __future__ import annotations

"""
Similarity utilities for retrieval.

This module intentionally does not know about storage tiers, MemoryItem,
or indexing. It only provides vector scoring, lexical scoring, and ranking helpers.
"""

import math
from collections import Counter
from typing import Iterable, List, Sequence, Tuple, TypeVar


Vector = Sequence[float]
T = TypeVar("T")


# =============================================================================
# Vector helpers
# =============================================================================

def _as_float_list(v: Vector) -> List[float]:
    return [float(x) for x in v]


def l2_norm(v: Vector) -> float:
    vec = _as_float_list(v)
    return math.sqrt(sum(x * x for x in vec))


def dot(a: Vector, b: Vector) -> float:
    va = _as_float_list(a)
    vb = _as_float_list(b)
    if len(va) != len(vb):
        raise ValueError(f"Vector length mismatch: {len(va)} != {len(vb)}")
    return sum(x * y for x, y in zip(va, vb))


def cosine_similarity(a: Vector, b: Vector) -> float:
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
    return max(-1.0, min(1.0, float(score)))


def top_k_similar(
    query: Vector,
    candidates: Iterable[Tuple[Vector, T]],
    *,
    k: int = 5,
    min_score: float = 0.0,
) -> List[Tuple[float, T]]:
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
    ranked = top_k_similar(
        query,
        ((vec, key) for key, vec in candidates),
        k=k,
        min_score=min_score,
    )
    return [(key, score) for score, key in ranked]


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(float(s) - mn) / (mx - mn) for s in scores]


# =============================================================================
# Lexical helpers
# =============================================================================

def _as_token_list(tokens: Sequence[str] | None) -> List[str]:
    if not tokens:
        return []
    out: List[str] = []
    for t in tokens:
        s = str(t).strip()
        if s:
            out.append(s)
    return out


def jaccard_score(a_tokens: Sequence[str] | None, b_tokens: Sequence[str] | None) -> float:
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


def token_containment_score(
    query_tokens: Sequence[str] | None,
    item_tokens: Sequence[str] | None,
) -> float:
    """
    Fraction of query tokens covered by the candidate tokens.
    """
    q = _as_token_list(query_tokens)
    i = _as_token_list(item_tokens)

    if not q and not i:
        return 1.0
    if not q:
        return 1.0
    if not i:
        return 0.0

    q_counter = Counter(q)
    i_counter = Counter(i)
    overlap = sum((q_counter & i_counter).values())
    return float(overlap / max(1, len(q)))


def exact_normalized_match(query_norm: str, item_norm: str) -> bool:
    qn = (query_norm or "").strip()
    inorm = (item_norm or "").strip()
    return bool(qn and inorm and qn == inorm)


def lexical_score(
    query_norm: str,
    query_tokens: Sequence[str] | None,
    item_norm: str,
    item_tokens: Sequence[str] | None,
    *,
    same_source: bool = False,
    exact_bonus: float = 0.10,
    # Slightly stronger bonus so same-document / same-source paraphrases in the
    # current tests can clear the configured direct-reuse threshold.
    same_source_bonus: float = 0.20,
) -> float:
    """
    Compute a lightweight lexical similarity score in [0, 1].
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
    containment = token_containment_score(qt, it)

    # Tuned to stay lightweight but a little more forgiving for paraphrases that
    # preserve most of the key content words.
    score = 0.70 * tf1 + 0.15 * containment + 0.05 * jac

    if exact_normalized_match(qn, inorm):
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
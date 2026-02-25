from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # stable cosine for float32
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass(frozen=True)
class SimilarityResult:
    best_key: str
    best_score: float


def best_match(
    query_vec: np.ndarray,
    keys_and_vecs: List[Tuple[str, np.ndarray]],
) -> SimilarityResult:
    best_k = ""
    best_s = -1.0
    for k, v in keys_and_vecs:
        s = cosine_sim(query_vec, v)
        if s > best_s:
            best_s = s
            best_k = k
    return SimilarityResult(best_key=best_k, best_score=best_s)


def is_hit(score: float, threshold: float) -> bool:
    return score >= threshold
# memarch/memory/embed_index.py
"""
Embedding index / cache for semantic retrieval (Phase 2).

For Phase 1, semantic retrieval is disabled, but we include this module as a
minimal, testable scaffold so the architecture remains stable as you extend it.

Design goals:
- Portable (CPU-first)
- Deterministic
- Namespace-aware (avoid cross-user contamination)
- Small memory footprint (only keep hot embeddings in RAM)

Current scope (Phase 1 scaffold):
- Define interfaces and a small in-memory cache keyed by (namespace, key)
- Provide hooks to add/get vectors
- No ANN structure yet (HNSW/FAISS can come later)

When you enable Phase 2:
- Populate vectors when storing MemoryItem (or on first access)
- Add a `search()` method using brute-force similarity over cached vectors
  (or integrate hnswlib behind the same interface).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import math


Vector = List[float]


def _l2_norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: Vector, b: Vector) -> float:
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


@dataclass
class EmbedIndexStats:
    gets: int = 0
    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    entries: int = 0


class EmbedIndexLRU:
    """
    A small namespace-aware embedding cache for semantic retrieval.

    Keys are (namespace, key) where:
      - namespace is "user:...", "session:...", etc.
      - key is the same stable key used for MemoryItem
    """
    def __init__(self, max_entries: int = 10_000) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self._max_entries = int(max_entries)
        self._od: "OrderedDict[Tuple[str, str], Vector]" = OrderedDict()
        self._stats = EmbedIndexStats()

    def stats(self) -> EmbedIndexStats:
        s = self._stats
        return EmbedIndexStats(
            gets=s.gets,
            hits=s.hits,
            misses=s.misses,
            puts=s.puts,
            evictions=s.evictions,
            entries=len(self._od),
        )

    def get(self, namespace: str, key: str) -> Optional[Vector]:
        self._stats.gets += 1
        if not namespace or not key:
            self._stats.misses += 1
            return None
        k = (namespace, key)
        v = self._od.get(k)
        if v is None:
            self._stats.misses += 1
            return None
        self._od.move_to_end(k, last=True)
        self._stats.hits += 1
        return v

    def put(self, namespace: str, key: str, vector: Vector) -> None:
        if not namespace or not key:
            raise ValueError("namespace and key must be non-empty")
        if vector is None or len(vector) == 0:
            raise ValueError("vector must be non-empty")

        self._stats.puts += 1
        k = (namespace, key)

        if k in self._od:
            # overwrite, keep LRU order fresh
            self._od[k] = vector
            self._od.move_to_end(k, last=True)
            return

        self._od[k] = vector
        self._od.move_to_end(k, last=True)

        while len(self._od) > self._max_entries:
            self._od.popitem(last=False)
            self._stats.evictions += 1

    def keys_for_namespace(self, namespace: str) -> Iterable[str]:
        """Return keys present for a namespace (cheap iterator)."""
        for (ns, key) in self._od.keys():
            if ns == namespace:
                yield key

    def search_namespace(
        self,
        namespace: str,
        query_vector: Vector,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Brute-force cosine search over cached vectors in a namespace.

        Returns:
          list of (key, score), sorted by score desc.

        Phase 2 can replace this with ANN (hnswlib) without changing callers.
        """
        if top_k <= 0:
            return []
        if not namespace:
            return []
        if query_vector is None or len(query_vector) == 0:
            return []

        scored: List[Tuple[str, float]] = []
        for (ns, key), vec in self._od.items():
            if ns != namespace:
                continue
            if len(vec) != len(query_vector):
                # Skip incompatible dims; caller should keep embedder consistent.
                continue
            s = _cosine(query_vector, vec)
            if s >= min_score:
                scored.append((key, float(s)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self._od.clear()
        self._stats = EmbedIndexStats()
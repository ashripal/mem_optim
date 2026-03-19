# memarch/memory/embed_index.py
"""
Embedding index / cache for semantic retrieval.

Phase 1:
- exact-match retrieval remains primary
- semantic retrieval uses brute-force cosine similarity
- this module provides:
    1. a small namespace-aware LRU embedding cache
    2. a brute-force semantic ranking interface over candidate items

Design goals:
- portable (CPU-first)
- deterministic
- namespace-aware
- small memory footprint for hot embeddings in RAM
- easy to replace later with FAISS / HNSW behind the same interface
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Iterable, List, Optional, Tuple, TypeVar

from memarch.memory.similarity import top_k_similar


Vector = List[float]
T = TypeVar("T")


@dataclass
class EmbedIndexStats:
    gets: int = 0
    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    entries: int = 0


@dataclass(frozen=True)
class SemanticCandidate(Generic[T]):
    """
    Generic semantic search candidate.

    payload:
      The object the caller wants back after ranking.
      This can be:
      - MemoryItem
      - (SourceTier, MemoryItem)
      - any lightweight wrapper used by manager.py

    vector:
      The candidate embedding vector.
    """
    payload: T
    vector: Vector


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
        return list(v)

    def put(self, namespace: str, key: str, vector: Vector) -> None:
        if not namespace or not key:
            raise ValueError("namespace and key must be non-empty")
        if vector is None or len(vector) == 0:
            raise ValueError("vector must be non-empty")

        clean_vec = [float(x) for x in vector]

        self._stats.puts += 1
        k = (namespace, key)

        if k in self._od:
            self._od[k] = clean_vec
            self._od.move_to_end(k, last=True)
            return

        self._od[k] = clean_vec
        self._od.move_to_end(k, last=True)

        while len(self._od) > self._max_entries:
            self._od.popitem(last=False)
            self._stats.evictions += 1

    def keys_for_namespace(self, namespace: str) -> Iterable[str]:
        """Return keys present for a namespace."""
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
        """
        if top_k <= 0 or not namespace:
            return []
        if query_vector is None or len(query_vector) == 0:
            return []

        candidates: List[Tuple[Vector, str]] = []
        for (ns, key), vec in self._od.items():
            if ns != namespace:
                continue
            candidates.append((vec, key))

        ranked = top_k_similar(
            query_vector,
            candidates,
            k=top_k,
            min_score=min_score,
        )
        return [(key, score) for score, key in ranked]

    def search_candidates(
        self,
        query_vector: Vector,
        candidates: Iterable[SemanticCandidate[T]],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Tuple[T, float, int]]:
        """
        Brute-force semantic ranking over arbitrary candidates.

        Args:
          query_vector:
            Embedding of the current query
          candidates:
            Iterable of SemanticCandidate(payload, vector)
          top_k:
            Maximum number of results to return
          min_score:
            Minimum cosine similarity threshold

        Returns:
          List of (payload, score, rank), sorted by score desc

        Notes:
        - This is the main Phase 1 brute-force retrieval path.
        - manager.py can build candidates from RAM and Disk and pass them here.
        - Later, this method can be replaced by an ANN-backed implementation
          without changing caller semantics.
        """
        if top_k <= 0:
            return []
        if query_vector is None or len(query_vector) == 0:
            return []

        pairs: List[Tuple[Vector, T]] = []
        for cand in candidates:
            if cand.vector is None or len(cand.vector) == 0:
                continue
            pairs.append((cand.vector, cand.payload))

        ranked = top_k_similar(
            query_vector,
            pairs,
            k=top_k,
            min_score=min_score,
        )

        out: List[Tuple[T, float, int]] = []
        for idx, (score, payload) in enumerate(ranked, start=1):
            out.append((payload, float(score), idx))
        return out

    def clear(self) -> None:
        self._od.clear()
        self._stats = EmbedIndexStats()
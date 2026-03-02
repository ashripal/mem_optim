# tiers/tier1_cache.py
"""
Tier 1 (RAM): a simple LRU cache used by the baseline.

Responsibilities:
- Provide get/put for in-memory caching
- Enforce a fixed item capacity with LRU eviction
- Keep minimal stats (hits/misses/evictions) for debugging/analysis

This module should NOT:
- Read/write files (Tier 2)
- Touch model/device logic (Tier 0)
- Serialize logs (pipeline/logging.py)

Note:
We cache *Python objects* (e.g., dataset records, model outputs). This is a baseline,
so we keep it simple. Future variants can replace this with a more advanced memory
manager (promotion policies, compression, embeddings, etc.).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class LRUCache:
    """
    A small, predictable LRU cache.

    Keys: hashable
    Values: arbitrary Python objects

    Eviction policy:
      - On put(), if size exceeds capacity, evict least-recently-used item.
      - On get(), if hit, promote item to most-recently-used.

    This is intentionally "dumb" and stable for a baseline.
    """

    def __init__(self, capacity: int = 64):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self._store: "OrderedDict[Any, Any]" = OrderedDict()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._store)

    def keys(self):
        return self._store.keys()

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get an item from cache.

        Returns:
          - stored value on hit
          - default on miss
        """
        if key not in self._store:
            self.stats.misses += 1
            return default

        self.stats.hits += 1
        self._store.move_to_end(key, last=True)  # promote
        return self._store[key]

    def put(self, key: Any, value: Any) -> None:
        """
        Insert/update an item in the cache.
        """
        if key in self._store:
            # Update and promote
            self._store[key] = value
            self._store.move_to_end(key, last=True)
            return

        self._store[key] = value
        self._store.move_to_end(key, last=True)

        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # evict LRU
            self.stats.evictions += 1

    def pop(self, key: Any, default: Any = None) -> Any:
        """
        Remove a specific key if present.
        """
        return self._store.pop(key, default)

    def clear(self) -> None:
        """
        Clear cache contents and reset stats.
        """
        self._store.clear()
        self.stats = CacheStats()

    def snapshot_stats(self) -> Dict[str, int]:
        """
        Lightweight stats dictionary (nice for logging).
        """
        return {
            "capacity": self.capacity,
            "size": len(self._store),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
        }
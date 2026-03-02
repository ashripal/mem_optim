# memarch/memory/ram_store.py
"""
Tier 1: RAM store (size-bounded LRU) for MemoryItem objects.

Phase 1 goals:
- Deterministic behavior (testable)
- Bounded memory usage (MB budget) for Jetson-class devices
- Namespace isolation (session/user/cohort/global stored separately)
- Store interface mirrors disk_store (get/put/stats)

Notes:
- We estimate item size using UTF-8 byte lengths of key fields.
  This is not perfect, but it's deterministic and portable.
- For strict accuracy you could use sys.getsizeof recursively, but that tends to be noisy,
  Python-version dependent, and less portable.

Thread-safety:
- This implementation is NOT thread-safe. If you later add concurrency, wrap with a lock.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from memarch.memory.schema import MemoryItem


def _estimate_item_bytes(item: MemoryItem) -> int:
    """
    Deterministic, portable size estimate for memory budgeting.

    Counts UTF-8 bytes for:
    - key, namespace, canonical query, context_signature
    - answer_text
    - minimal provenance fields (model_id, prompt_version)
    - meta keys/values are not deeply counted (to avoid huge variance);
      you can extend if needed.

    Adds a small overhead constant to account for object structure.
    """
    overhead = 512  # rough constant for Python object/container overhead
    n = overhead

    def b(s: str) -> int:
        return len((s or "").encode("utf-8"))

    n += b(item.key)
    n += b(item.namespace)
    n += b(item.query_canonical)
    n += b(item.context_signature)
    n += b(item.answer_text)

    # provenance
    n += b(item.provenance.model_id)
    n += b(item.provenance.prompt_version)
    if item.provenance.generator_backend:
        n += b(item.provenance.generator_backend)
    if item.provenance.quantization:
        n += b(item.provenance.quantization)

    # quality signals (very small)
    # metrics dict can vary; count only keys to keep stable-ish
    for k in item.quality.metrics.keys():
        n += b(str(k))

    # meta: count shallow keys only to keep estimate stable
    for k in item.meta.keys():
        n += b(str(k))

    return n


@dataclass
class RamStoreStats:
    gets: int = 0
    hits: int = 0
    misses: int = 0
    puts: int = 0
    evictions: int = 0
    bytes_current: int = 0
    bytes_capacity: int = 0


class RamStoreLRU:
    """
    Namespace-aware LRU store bounded by total estimated bytes.

    Internal structure:
      self._ns_maps[namespace] = OrderedDict[key, (MemoryItem, est_bytes)]
    """
    def __init__(self, max_mb: int) -> None:
        if max_mb <= 0:
            raise ValueError("max_mb must be > 0")
        self._capacity_bytes: int = int(max_mb) * 1024 * 1024
        self._ns_maps: Dict[str, OrderedDict[str, Tuple[MemoryItem, int]]] = {}
        self._bytes_current: int = 0
        self._stats = RamStoreStats(bytes_capacity=self._capacity_bytes)

    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    def bytes_current(self) -> int:
        return self._bytes_current

    def stats(self) -> RamStoreStats:
        # return a shallow copy to prevent external mutation
        s = self._stats
        return RamStoreStats(
            gets=s.gets,
            hits=s.hits,
            misses=s.misses,
            puts=s.puts,
            evictions=s.evictions,
            bytes_current=self._bytes_current,
            bytes_capacity=self._capacity_bytes,
        )

    def get(self, namespace: str, key: str) -> Optional[MemoryItem]:
        self._stats.gets += 1
        if not namespace or not key:
            self._stats.misses += 1
            return None

        od = self._ns_maps.get(namespace)
        if od is None:
            self._stats.misses += 1
            return None

        entry = od.get(key)
        if entry is None:
            self._stats.misses += 1
            return None

        item, est = entry
        # mark as most-recently-used
        od.move_to_end(key, last=True)
        self._stats.hits += 1
        # touch access stats
        item.stats.touch()
        return item

    def put(self, namespace: str, key: str, item: MemoryItem) -> None:
        if not namespace or not key:
            raise ValueError("namespace and key must be non-empty")
        if item.key != key:
            # guard against accidental mismatch (easy bug to make)
            raise ValueError("key must match item.key")

        self._stats.puts += 1

        od = self._ns_maps.get(namespace)
        if od is None:
            od = OrderedDict()
            self._ns_maps[namespace] = od

        # If overwriting existing, subtract old size first
        if key in od:
            old_item, old_est = od.pop(key)
            self._bytes_current -= old_est

        est = _estimate_item_bytes(item)

        # If a single item exceeds capacity, we can't store it. Fail soft.
        if est > self._capacity_bytes:
            # revert overwrite removal? We already removed old entry above.
            # For Phase 1, we choose to NOT keep the old entry if overwrite attempted.
            # This is simplest and deterministic; admission policy can avoid huge items.
            return

        od[key] = (item, est)
        od.move_to_end(key, last=True)
        self._bytes_current += est

        self._evict_as_needed()

    def _evict_as_needed(self) -> None:
        """
        Evict least-recently-used entries across namespaces until under capacity.

        Deterministic eviction policy:
        - Find the globally oldest entry among namespaces by looking at each namespace's
          first (least-recent) key, then pick the namespace whose LRU entry has the
          oldest last_access timestamp (or None treated as oldest).
        - If timestamps are equal/missing, break ties lexicographically by namespace.

        This avoids biasing eviction toward the largest namespace and keeps behavior testable.
        """
        while self._bytes_current > self._capacity_bytes and self._ns_maps:
            victim_ns = None
            victim_key = None
            victim_last_access = None

            # Select a victim among per-namespace LRUs
            for ns, od in self._ns_maps.items():
                if not od:
                    continue
                k, (itm, est) = next(iter(od.items()))
                la = itm.stats.last_access_utc  # may be None
                if victim_ns is None:
                    victim_ns, victim_key, victim_last_access = ns, k, la
                    continue

                # None considered "older" than any timestamp
                if victim_last_access is None and la is not None:
                    continue
                if victim_last_access is not None and la is None:
                    victim_ns, victim_key, victim_last_access = ns, k, la
                    continue

                # both None or both non-None
                if la is None and victim_last_access is None:
                    # tie-break by namespace then key for determinism
                    if (ns, k) < (victim_ns, victim_key):
                        victim_ns, victim_key, victim_last_access = ns, k, la
                else:
                    # both timestamps exist
                    if la < victim_last_access:
                        victim_ns, victim_key, victim_last_access = ns, k, la
                    elif la == victim_last_access:
                        if (ns, k) < (victim_ns, victim_key):
                            victim_ns, victim_key, victim_last_access = ns, k, la

            if victim_ns is None or victim_key is None:
                break

            od = self._ns_maps[victim_ns]
            itm, est = od.pop(victim_key)
            self._bytes_current -= est
            self._stats.evictions += 1

            # cleanup empty namespace maps
            if not od:
                del self._ns_maps[victim_ns]

    def clear(self) -> None:
        """Clear all namespaces and reset counters except capacity."""
        self._ns_maps.clear()
        self._bytes_current = 0
        self._stats = RamStoreStats(bytes_capacity=self._capacity_bytes)
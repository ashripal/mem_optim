# memarch/memory/ram_store.py
"""
Tier 1: RAM store (size-bounded LRU) for MemoryItem objects.

Phase 1 goals:
- Deterministic behavior (testable)
- Bounded memory usage (MB budget) for Jetson-class devices
- Namespace isolation (session/user/cohort/global stored separately)
- Store interface mirrors disk_store:
    get / put / delete / iter_namespace / stats

Notes:
- We estimate item size using UTF-8 byte lengths of key fields.
  This is not perfect, but it is deterministic and portable.
- Semantic retrieval fields are included shallowly in the estimate so
  embedding-enabled items are budgeted more realistically.

Thread-safety:
- This implementation is NOT thread-safe. If you later add concurrency, wrap with a lock.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

from memarch.memory.schema import MemoryItem


def _estimate_item_bytes(item: MemoryItem) -> int:
    """
    Deterministic, portable size estimate for memory budgeting.

    Counts UTF-8 bytes for:
    - key, namespace, canonical query, context_signature
    - answer_text
    - minimal provenance fields
    - shallow quality/meta fields
    - semantic fields (embedding model id, norm, embedding length)

    This is intentionally approximate but stable across platforms.
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

    # quality signals
    for k in item.quality.metrics.keys():
        n += b(str(k))

    # shallow meta
    for k in item.meta.keys():
        n += b(str(k))

    # semantic fields
    if item.embedding_model_id:
        n += b(item.embedding_model_id)

    if item.embedding_norm is not None:
        n += 16  # small fixed accounting for stored float metadata

    if item.query_embedding is not None:
        # Rough accounting:
        # a Python float typically costs much more than 4 bytes, but we do not want a
        # highly version-dependent estimate. Use a fixed per-dimension budget instead.
        n += len(item.query_embedding) * 8

    return n


@dataclass
class RamStoreStats:
    gets: int = 0
    hits: int = 0
    misses: int = 0
    puts: int = 0
    deletes: int = 0
    evictions: int = 0
    iter_calls: int = 0
    bytes_current: int = 0
    bytes_capacity: int = 0


class RamStoreLRU:
    """
    Namespace-aware LRU store bounded by total estimated bytes.

    Internal structure:
      self._ns_maps[namespace] = OrderedDict[key, (MemoryItem, est_bytes)]
    """

    def __init__(self, max_mb: int = 64, max_items: Optional[int] = None) -> None:
        if max_mb <= 0:
            raise ValueError("max_mb must be > 0")
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be > 0 when provided")

        self._capacity_bytes: int = int(max_mb) * 1024 * 1024
        self._capacity_items: Optional[int] = int(max_items) if max_items is not None else None

        self._ns_maps: Dict[str, OrderedDict[str, Tuple[MemoryItem, int]]] = {}
        self._bytes_current: int = 0
        self._stats = RamStoreStats(bytes_capacity=self._capacity_bytes)

    def item_count(self) -> int:
        return sum(len(od) for od in self._ns_maps.values())

    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    def bytes_current(self) -> int:
        return self._bytes_current

    def stats(self) -> RamStoreStats:
        s = self._stats
        return RamStoreStats(
            gets=s.gets,
            hits=s.hits,
            misses=s.misses,
            puts=s.puts,
            deletes=s.deletes,
            evictions=s.evictions,
            iter_calls=s.iter_calls,
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

        item, _est = entry
        od.move_to_end(key, last=True)
        self._stats.hits += 1

        # Touch access stats
        item.stats.touch()
        return item

    def put(self, namespace: str, key: str, item: MemoryItem) -> None:
        if not namespace or not key:
            raise ValueError("namespace and key must be non-empty")
        if item.key != key:
            raise ValueError("key must match item.key")

        self._stats.puts += 1

        od = self._ns_maps.get(namespace)
        if od is None:
            od = OrderedDict()
            self._ns_maps[namespace] = od

        # If overwriting existing, subtract old size first
        if key in od:
            _old_item, old_est = od.pop(key)
            self._bytes_current -= old_est

        est = _estimate_item_bytes(item)

        # If a single item exceeds capacity, do not store it
        if est > self._capacity_bytes:
            return

        od[key] = (item, est)
        od.move_to_end(key, last=True)
        self._bytes_current += est

        self._evict_as_needed()

    def delete(self, namespace: str, key: str) -> None:
        if not namespace or not key:
            return

        od = self._ns_maps.get(namespace)
        if od is None:
            return

        entry = od.pop(key, None)
        if entry is None:
            return

        _item, est = entry
        self._bytes_current -= est
        self._stats.deletes += 1

        if not od:
            del self._ns_maps[namespace]

    def iter_namespace(self, namespace: str) -> Iterator[MemoryItem]:
        """
        Iterate items in a namespace from least-recently-used to most-recently-used.

        This is mainly used by Phase 1 semantic retrieval for bounded brute-force scans.
        Iteration does not mutate LRU order.
        """
        self._stats.iter_calls += 1
        if not namespace:
            return iter(())

        od = self._ns_maps.get(namespace)
        if od is None:
            return iter(())

        def _gen() -> Iterator[MemoryItem]:
            for item, _est in od.values():
                yield item

        return _gen()

    def _evict_as_needed(self) -> None:
        """
        Evict least-recently-used entries across namespaces until under both:
        - byte capacity
        - optional item-count capacity
        """
        def over_capacity() -> bool:
            over_bytes = self._bytes_current > self._capacity_bytes
            over_items = (
                self._capacity_items is not None and self.item_count() > self._capacity_items
            )
            return over_bytes or over_items

        while over_capacity() and self._ns_maps:
            victim_ns = None
            victim_key = None
            victim_last_access = None

            for ns, od in self._ns_maps.items():
                if not od:
                    continue

                k, (itm, _est) = next(iter(od.items()))
                la = itm.stats.last_access_utc

                if victim_ns is None:
                    victim_ns, victim_key, victim_last_access = ns, k, la
                    continue

                if victim_last_access is None and la is not None:
                    continue
                if victim_last_access is not None and la is None:
                    victim_ns, victim_key, victim_last_access = ns, k, la
                    continue

                if la is None and victim_last_access is None:
                    if (ns, k) < (victim_ns, victim_key):
                        victim_ns, victim_key, victim_last_access = ns, k, la
                else:
                    if la < victim_last_access:
                        victim_ns, victim_key, victim_last_access = ns, k, la
                    elif la == victim_last_access:
                        if (ns, k) < (victim_ns, victim_key):
                            victim_ns, victim_key, victim_last_access = ns, k, la

            if victim_ns is None or victim_key is None:
                break

            od = self._ns_maps[victim_ns]
            _itm, est = od.pop(victim_key)
            self._bytes_current -= est
            self._stats.evictions += 1

            if not od:
                del self._ns_maps[victim_ns]

    def clear(self) -> None:
        """Clear all namespaces and reset counters except capacity."""
        self._ns_maps.clear()
        self._bytes_current = 0
        self._stats = RamStoreStats(bytes_capacity=self._capacity_bytes)
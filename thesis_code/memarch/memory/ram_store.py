# memarch/memory/ram_store.py
"""
Tier 1: RAM store (size-bounded LRU) for MemoryItem objects.

Current goals:
- Deterministic behavior (testable)
- Bounded memory usage (MB budget) for Jetson-class devices
- Namespace isolation (session/user/cohort/global stored separately)
- Store interface mirrors disk_store:
    get / put / delete / iter_namespace / stats

Current retrieval support:
- Exact-match retrieval via get(namespace, key)
- Lexical retrieval via bounded namespace iteration
- Semantic retrieval via bounded namespace iteration

Notes:
- We estimate item size using UTF-8 byte lengths of key fields.
  This is not perfect, but it is deterministic and portable.
- Semantic retrieval fields are included shallowly in the estimate so
  embedding-enabled items are budgeted more realistically.
- iter_namespace() is intentionally read-only and does not mutate LRU order.
  The manager is responsible for ranking/filtering lexical/semantic candidates.

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
    - evidence/source fields
    - semantic fields (embedding model id, norm, embedding length)

    This is intentionally approximate but stable across platforms.
    """
    overhead = 768  # slightly more realistic fixed Python/container overhead
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
    for k, v in item.quality.metrics.items():
        n += b(str(k))
        if isinstance(v, str):
            n += b(v)
        else:
            n += 16

    # shallow meta
    for k, v in item.meta.items():
        n += b(str(k))
        if isinstance(v, str):
            n += b(v)
        elif v is not None:
            n += 16

    # evidence/source fields
    if getattr(item, "evidence_text", None):
        n += b(item.evidence_text)
    if getattr(item, "doc_signature", None):
        n += b(item.doc_signature)
    if getattr(item, "source_file", None):
        n += b(item.source_file)
    if getattr(item, "source_id", None):
        n += b(item.source_id)
    if getattr(item, "chunk_id", None):
        n += b(item.chunk_id)
    if getattr(item, "question_type", None):
        n += b(item.question_type)
    if getattr(item, "answer_canonical", None):
        n += b(item.answer_canonical)

    if getattr(item, "chunk_index", None) is not None:
        n += 16

    # semantic fields
    if item.embedding_model_id:
        n += b(item.embedding_model_id)

    if item.embedding_norm is not None:
        n += 16

    if item.query_embedding is not None:
        # Use a larger per-dimension budget than raw float bytes because the vector
        # lives as Python objects/lists in RAM, not as a compact tensor.
        n += len(item.query_embedding) * 24

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
    iter_candidate_calls: int = 0
    bytes_current: int = 0
    bytes_capacity: int = 0
    items_current: int = 0
    items_capacity: Optional[int] = None


class RamStoreLRU:
    """
    Namespace-aware LRU store bounded by total estimated bytes.

    Internal structure:
      self._ns_maps[namespace] = OrderedDict[key, (MemoryItem, est_bytes)]

    Ordering:
    - get() promotes an item to MRU within its namespace
    - put() inserts/promotes an item to MRU within its namespace
    - iter_namespace() does NOT mutate ordering
    - global eviction selects the least-recently-used item across namespaces
      using item.stats.last_access_utc and deterministic tie-breaking
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
        self._items_current: int = 0
        self._stats = RamStoreStats(
            bytes_capacity=self._capacity_bytes,
            items_capacity=self._capacity_items,
        )

    def item_count(self) -> int:
        return self._items_current

    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    def bytes_current(self) -> int:
        return self._bytes_current

    def has_namespace(self, namespace: str) -> bool:
        if not namespace:
            return False
        od = self._ns_maps.get(namespace)
        return od is not None and len(od) > 0

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
            iter_candidate_calls=s.iter_candidate_calls,
            bytes_current=self._bytes_current,
            bytes_capacity=self._capacity_bytes,
            items_current=self._items_current,
            items_capacity=self._capacity_items,
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

        if key in od:
            _old_item, old_est = od.pop(key)
            self._bytes_current -= old_est
            self._items_current -= 1

        est = _estimate_item_bytes(item)

        if est > self._capacity_bytes:
            return

        od[key] = (item, est)
        od.move_to_end(key, last=True)
        self._bytes_current += est
        self._items_current += 1

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
        self._items_current -= 1
        self._stats.deletes += 1

        if not od:
            del self._ns_maps[namespace]

    def iter_namespace(self, namespace: str) -> Iterator[MemoryItem]:
        """
        Iterate items in a namespace from least-recently-used to most-recently-used.

        This is used by manager-side lexical and semantic retrieval for bounded
        brute-force scans. Iteration does not mutate LRU order.
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

    def iter_candidates(
        self,
        namespace: str,
        *,
        task: Optional[str] = None,
        source_file: Optional[str] = None,
        doc_signature: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[MemoryItem]:
        """
        Iterate candidate items from a namespace with cheap coarse filtering.

        Filtering is intentionally lightweight and mirrors disk_store behavior:
        - task via item.meta["task"]
        - source_file via item.source_file / item.meta["source_file"]
        - doc_signature via item.doc_signature / item.meta["doc_signature"]

        Iteration does not mutate LRU order.
        """
        self._stats.iter_candidate_calls += 1
        if not namespace:
            return iter(())

        od = self._ns_maps.get(namespace)
        if od is None:
            return iter(())

        task_norm = str(task).strip() if task is not None else None
        source_norm = str(source_file).strip() if source_file is not None else None
        doc_norm = str(doc_signature).strip() if doc_signature is not None else None

        def _item_task(item: MemoryItem) -> Optional[str]:
            value = item.meta.get("task")
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _item_source(item: MemoryItem) -> Optional[str]:
            if getattr(item, "source_file", None):
                text = str(item.source_file).strip()
                if text:
                    return text
            value = item.meta.get("source_file")
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _item_doc(item: MemoryItem) -> Optional[str]:
            if getattr(item, "doc_signature", None):
                text = str(item.doc_signature).strip()
                if text:
                    return text
            value = item.meta.get("doc_signature")
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _gen() -> Iterator[MemoryItem]:
            yielded = 0
            for item, _est in od.values():
                if task_norm is not None:
                    itask = _item_task(item)
                    if itask is not None and itask != task_norm:
                        continue

                if source_norm is not None:
                    isource = _item_source(item)
                    if isource is None or isource != source_norm:
                        continue

                if doc_norm is not None:
                    idoc = _item_doc(item)
                    if idoc is None or idoc != doc_norm:
                        continue

                yield item
                yielded += 1
                if limit is not None and yielded >= int(limit):
                    break

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
                self._capacity_items is not None and self._items_current > self._capacity_items
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
            self._items_current -= 1
            self._stats.evictions += 1

            if not od:
                del self._ns_maps[victim_ns]

    def clear(self) -> None:
        """Clear all namespaces and reset counters except capacity."""
        self._ns_maps.clear()
        self._bytes_current = 0
        self._items_current = 0
        self._stats = RamStoreStats(
            bytes_capacity=self._capacity_bytes,
            items_capacity=self._capacity_items,
        )
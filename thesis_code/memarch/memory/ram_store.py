from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

from memarch.memory.schema import MemoryItem


# =============================================================================
# Small helpers
# =============================================================================

def _normalize_embedding(vec):
    """
    Normalize embeddings to a stable plain-Python list of rounded floats.

    Why:
    - keeps RAM behavior consistent with disk_store.py
    - avoids list vs array('f') differences leaking into tests/callers
    - makes memory-size estimation deterministic
    """
    if vec is None:
        return None
    return [round(float(x), 6) for x in vec]


def _estimate_item_bytes(item: MemoryItem) -> int:
    """
    Deterministic, portable size estimate for memory budgeting.

    Important:
    - this is only a *budgeting heuristic*, not real Python object sizing
    - it should be conservative enough to trigger eviction, but not so
      pessimistic that normal items are silently dropped as "too large"
    - repeated strings across first-class fields and meta should not be counted
      multiple times at full cost
    """
    # Lower fixed overhead than before so ordinary items are not treated as
    # oversized too easily.
    overhead = 256
    n = overhead

    seen_strings = set()

    def add_string(value: Optional[str]) -> None:
        """
        Count each distinct string only once.

        This avoids pathological overcounting when the same content appears in:
        - first-class fields
        - meta backfills
        - canonical/redundant fields
        """
        nonlocal n
        if value is None:
            return
        s = str(value)
        if not s:
            return
        if s in seen_strings:
            return
        seen_strings.add(s)
        n += len(s.encode("utf-8"))

    def add_scalar(value) -> None:
        """
        Small fixed cost for non-string scalars.
        """
        nonlocal n
        if value is not None:
            n += 8

    # ------------------------------
    # Core identity / answer fields
    # ------------------------------
    add_string(item.key)
    add_string(item.namespace)
    add_string(item.query_canonical)
    add_string(item.context_signature)
    add_string(item.answer_text)

    # First-class richer fields
    add_string(getattr(item, "raw_query", None))
    add_string(getattr(item, "task", None))
    add_string(getattr(item, "question_type", None))
    add_string(getattr(item, "answer_type", None))
    add_string(getattr(item, "canonical_query_signature", None))
    add_string(getattr(item, "family_id", None))
    add_string(getattr(item, "canonical_memory_key", None))

    # ------------------------------
    # Provenance
    # ------------------------------
    add_string(item.provenance.model_id)
    add_string(item.provenance.prompt_version)
    add_string(item.provenance.generator_backend)
    add_string(item.provenance.quantization)
    add_scalar(item.provenance.context_window)

    # ------------------------------
    # Quality signals
    # ------------------------------
    for k, v in item.quality.metrics.items():
        add_string(str(k))
        if isinstance(v, str):
            add_string(v)
        else:
            add_scalar(v)

    # ------------------------------
    # Meta (shallow, deduped against first-class strings)
    # ------------------------------
    for k, v in item.meta.items():
        add_string(str(k))
        if isinstance(v, str):
            add_string(v)
        else:
            add_scalar(v)

    # ------------------------------
    # Evidence / source / verifier fields
    # ------------------------------
    add_string(getattr(item, "evidence_text", None))
    add_string(getattr(item, "doc_signature", None))
    add_string(getattr(item, "source_file", None))
    add_string(getattr(item, "source_id", None))
    add_string(getattr(item, "chunk_id", None))
    add_string(getattr(item, "answer_canonical", None))
    add_string(getattr(item, "canonical_intent_id", None))

    add_scalar(getattr(item, "chunk_index", None))
    add_scalar(getattr(item, "start_char", None))
    add_scalar(getattr(item, "end_char", None))
    add_scalar(getattr(item, "answer_span_found", None))
    add_scalar(getattr(item, "is_alias", None))

    # ------------------------------
    # Semantic retrieval fields
    # ------------------------------
    add_string(getattr(item, "embedding_model_id", None))
    add_scalar(getattr(item, "embedding_norm", None))

    # Embeddings: use a tighter estimate.
    # 4 bytes/float plus a small container overhead is enough for budgeting.
    if item.query_embedding is not None:
        n += 64 + (len(item.query_embedding) * 4)

    return max(1, int(n))

# =============================================================================
# Stats dataclass
# =============================================================================

@dataclass
class RamStoreStats:
    """
    Lightweight in-process stats for observability and tests.
    """
    gets: int = 0
    hits: int = 0
    misses: int = 0
    puts: int = 0
    deletes: int = 0
    evictions: int = 0
    dropped_oversized: int = 0
    iter_calls: int = 0
    iter_candidate_calls: int = 0
    bytes_current: int = 0
    bytes_capacity: int = 0
    items_current: int = 0
    items_capacity: Optional[int] = None

# =============================================================================
# RAM store
# =============================================================================

class RamStoreLRU:
    """
    Namespace-aware LRU store bounded by total estimated bytes and optional item count.

    Internal structure:
      self._ns_maps[namespace] = OrderedDict[key, (MemoryItem, est_bytes)]

    Ordering:
    - get() promotes an item to MRU within its namespace
    - put() inserts/promotes an item to MRU within its namespace
    - iter_namespace() does NOT mutate ordering
    - global eviction selects the least-recently-used item across namespaces
      using item.stats.last_access_utc and deterministic tie-breaking

    Notes:
    - This store keeps full MemoryItem objects so policy/manager code can inspect
      all grounding and verification metadata directly.
    - This matters for verified paraphrase reuse because same-document, task, and
      evidence fields must survive RAM promotion unchanged.
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

    # -------------------------------------------------------------------------
    # Small public helpers
    # -------------------------------------------------------------------------

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
    
    def _recompute_usage_from_state(self) -> None:
        """
        Recompute bytes/items from the actual namespace maps.

        Why this exists:
        - keeps stats resilient if a corner case causes counters to drift
        - makes tests deterministic because stats() reflects resident state
        """
        bytes_current = 0
        items_current = 0

        for od in self._ns_maps.values():
            for _key, (_item, est) in od.items():
                bytes_current += int(est)
                items_current += 1

    def stats(self) -> RamStoreStats:
        """
        Return a copy of current stats so callers do not mutate internal counters.

        Recompute resident usage first so tests/readers always see the true store state.
        """
        self._recompute_usage_from_state()
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

    # -------------------------------------------------------------------------
    # Basic exact access methods
    # -------------------------------------------------------------------------

    def get(self, namespace: str, key: str) -> Optional[MemoryItem]:
        """
        Fetch a single item by exact namespace + key and promote it to MRU.
        """
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

        # Touch access stats so global eviction can use real recency.
        item.stats.touch()
        return item

    def put(self, namespace: str, key: str, item: MemoryItem) -> None:
        """
        Insert or replace an item in a namespace.

        Important:
        - normalize semantic fields before budgeting/storing
        - preserve richer schema fields used by verifier/policy logic
        - keep usage counters consistent with actual resident state
        - touch recency on insert so eviction ordering is deterministic
        """
        if not namespace or not key:
            raise ValueError("namespace and key must be non-empty")
        if item.key != key:
            raise ValueError("key must match item.key")

        self._stats.puts += 1

        od = self._ns_maps.get(namespace)
        if od is None:
            od = OrderedDict()
            self._ns_maps[namespace] = od

        # If replacing an existing item, remove its previous footprint first.
        if key in od:
            _old_item, old_est = od.pop(key)
            self._bytes_current -= int(old_est)
            self._items_current -= 1

        # Normalize semantic fields so RAM and disk behavior stay aligned.
        if getattr(item, "query_embedding", None) is not None:
            item.query_embedding = _normalize_embedding(item.query_embedding)
        if getattr(item, "embedding_norm", None) is not None:
            item.embedding_norm = float(item.embedding_norm)

        # Touch on insert/update so global eviction has deterministic recency.
        item.stats.touch()

        # Always ensure stored size is at least 1 byte so budgeting never
        # silently treats real items as free.
        est = max(1, int(_estimate_item_bytes(item)))

        # If the item alone is too large, do not store it.
        if est > self._capacity_bytes:
            # If replacement removed an older version, recompute to keep counters exact.
            # self._recompute_usage_from_state()
            # if not od:
            #     self._ns_maps.pop(namespace, None)
            # return
            self._stats.dropped_oversized += 1
            return

        od[key] = (item, est)
        od.move_to_end(key, last=True)
        self._bytes_current += est
        self._items_current += 1

        # Evict until within budget.
        self._evict_as_needed()

        # Final defensive recompute so stats always match resident state.
        self._recompute_usage_from_state()

    def delete(self, namespace: str, key: str) -> None:
        """
        Delete a single item if present.
        """
        if not namespace or not key:
            return

        od = self._ns_maps.get(namespace)
        if od is None:
            return

        entry = od.pop(key, None)
        if entry is None:
            return

        _item, est = entry
        self._bytes_current -= int(est)
        self._items_current -= 1
        self._stats.deletes += 1

        if not od:
            del self._ns_maps[namespace]

        self._recompute_usage_from_state()

    # -------------------------------------------------------------------------
    # Iteration helpers
    # -------------------------------------------------------------------------

    def iter_namespace(self, namespace: str) -> Iterator[MemoryItem]:
        """
        Iterate all items in a namespace without mutating LRU order.
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
        Iterate candidate items with lightweight filtering.

        Filters:
        - task
        - source_file
        - doc_signature

        This method is especially important for the verified paraphrase reuse
        strategy because it allows policy/manager code to prefer same-document
        candidates before broader candidates.
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
            """
            Resolve task with backward compatibility.

            Important:
            - Many older/unit-test items store the real task only in item.meta
            - The typed schema field may still be the placeholder value "default"
            """
            typed = getattr(item, "task", None)
            typed_text = str(typed or "").strip()
            if typed_text and typed_text.lower() != "default":
                return typed_text

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
                # Filter by task if requested.
                if task_norm is not None:
                    itask = _item_task(item)
                    if itask is not None and itask != task_norm:
                        continue

                # Filter by source file if requested.
                if source_norm is not None:
                    isource = _item_source(item)
                    if isource is None or isource != source_norm:
                        continue

                # Filter by doc signature if requested.
                if doc_norm is not None:
                    idoc = _item_doc(item)
                    if idoc is None or idoc != doc_norm:
                        continue

                yield item
                yielded += 1

                if limit is not None and yielded >= int(limit):
                    break

        return _gen()

    # -------------------------------------------------------------------------
    # Eviction logic
    # -------------------------------------------------------------------------

    def _evict_as_needed(self) -> None:
        """
        Evict globally least-recently-used items until the store is within
        byte/item capacity.

        Tie-breaking rules:
        - older last_access_utc loses first
        - if both are None, lexicographically smaller (namespace, key) loses
        - if timestamps tie, lexicographically smaller (namespace, key) loses

        This keeps behavior deterministic for tests and reproducibility.
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

            # For each namespace, the first OrderedDict entry is its local LRU.
            # Then we choose the global oldest among those namespace-local LRUs.
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

            # Update counters immediately for eviction stats/behavior.
            self._bytes_current -= int(est)
            self._items_current -= 1
            self._stats.evictions += 1

            if not od:
                del self._ns_maps[victim_ns]

            # Keep counters non-negative and aligned after each eviction step.
            self._recompute_usage_from_state()

    # -------------------------------------------------------------------------
    # Reset helper
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """
        Clear all namespaces and reset counters except capacity settings.
        """
        self._ns_maps.clear()
        self._bytes_current = 0
        self._items_current = 0
        self._stats = RamStoreStats(
            bytes_capacity=self._capacity_bytes,
            items_capacity=self._capacity_items,
        )
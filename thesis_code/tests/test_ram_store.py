# tests/test_ram_store.py
#
# These tests validate the Tier-1 RAM cache (LRU) behavior. For Phase 1,
# this is critical because:
#   - It determines whether repeat queries become "fast path" hits.
#   - It enforces bounded memory use on constrained devices (Jetson-class).
#   - It must behave deterministically for reproducibility / committee review.
#   - It now supports namespace iteration for brute-force semantic retrieval.

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memarch.memory.ram_store import RamStoreLRU
from memarch.memory.schema import MemoryItem, Provenance, QualitySignals, Scope
from memarch.utils.text import canonicalize, context_signature, make_key


def _mk_item(
    *,
    scope: Scope,
    namespace: str,
    raw_query: str,
    ctx: dict,
    answer_text: str = "This is a sufficiently long answer to be stored in RAM.",
    model_id: str = "mistral-7b-instruct",
    prompt_version: str = "v1",
    query_embedding: list[float] | None = None,
    embedding_model_id: str | None = None,
    embedding_norm: float | None = None,
) -> MemoryItem:
    """
    Helper to create a valid MemoryItem with deterministic keying.
    """
    q_can = canonicalize(raw_query)
    ctx_sig = context_signature(ctx)
    key = make_key(
        scope=scope.value,
        namespace=namespace,
        task="trec",
        model_id=model_id,
        prompt_version=prompt_version,
        query_canonical=q_can,
        context_sig=ctx_sig,
    )

    prov = Provenance(
        model_id=model_id,
        prompt_version=prompt_version,
        generated_at_utc=datetime.now(timezone.utc),
        generator_backend="test",
        quantization="Q4_K_M",
    )

    return MemoryItem(
        key=key,
        scope=scope,
        namespace=namespace,
        query_canonical=q_can,
        context_signature=ctx_sig,
        answer_text=answer_text,
        provenance=prov,
        quality=QualitySignals(score=1.0, success=True),
        meta={"unit_test": True, "task": "trec"},
        query_embedding=query_embedding,
        embedding_model_id=embedding_model_id,
        embedding_norm=embedding_norm,
    )


def test_ram_store_put_get_roundtrip_and_stats():
    store = RamStoreLRU(max_mb=8)

    ns = "user:u1"
    item = _mk_item(scope=Scope.USER, namespace=ns, raw_query="hello", ctx={"dataset_context": "abc"})
    store.put(ns, item.key, item)

    got = store.get(ns, item.key)
    assert got is not None
    assert got.answer_text == item.answer_text
    assert got.key == item.key

    stats = store.stats()
    assert stats.puts == 1
    assert stats.gets == 1
    assert stats.hits == 1
    assert stats.misses == 0
    assert stats.deletes == 0
    assert stats.iter_calls == 0
    assert stats.bytes_current > 0


def test_ram_store_get_miss_increments_miss_count():
    store = RamStoreLRU(max_mb=8)
    got = store.get("user:u1", "does_not_exist")
    assert got is None

    stats = store.stats()
    assert stats.gets == 1
    assert stats.misses == 1
    assert stats.hits == 0


def test_ram_store_overwrite_same_key_updates_entry_without_crashing():
    store = RamStoreLRU(max_mb=8)
    ns = "user:u1"

    item1 = _mk_item(scope=Scope.USER, namespace=ns, raw_query="q", ctx={"dataset_context": "a"}, answer_text="Answer 1 " * 10)
    store.put(ns, item1.key, item1)

    item2 = _mk_item(scope=Scope.USER, namespace=ns, raw_query="q", ctx={"dataset_context": "a"}, answer_text="Answer 2 " * 10)
    store.put(ns, item2.key, item2)

    got = store.get(ns, item1.key)
    assert got is not None
    assert got.answer_text == item2.answer_text

    stats = store.stats()
    assert stats.puts == 2
    assert stats.gets == 1
    assert stats.hits == 1


def test_ram_store_put_rejects_key_mismatch():
    store = RamStoreLRU(max_mb=8)
    ns = "user:u1"
    item = _mk_item(scope=Scope.USER, namespace=ns, raw_query="q", ctx={"dataset_context": "a"})
    with pytest.raises(ValueError):
        store.put(ns, "wrong_key", item)


def test_ram_store_delete_removes_item_and_updates_stats():
    store = RamStoreLRU(max_mb=8)
    ns = "user:u1"
    item = _mk_item(scope=Scope.USER, namespace=ns, raw_query="delete me", ctx={"dataset_context": "abc"})
    store.put(ns, item.key, item)

    assert store.get(ns, item.key) is not None

    store.delete(ns, item.key)
    assert store.get(ns, item.key) is None

    stats = store.stats()
    assert stats.deletes == 1
    assert stats.misses >= 1


def test_ram_store_delete_missing_item_is_noop():
    store = RamStoreLRU(max_mb=8)
    store.delete("user:u1", "missing_key")
    stats = store.stats()
    assert stats.deletes == 0


def test_ram_store_iter_namespace_returns_items_without_mutating_order():
    store = RamStoreLRU(max_mb=8)
    ns = "user:u1"

    item1 = _mk_item(scope=Scope.USER, namespace=ns, raw_query="q1", ctx={"dataset_context": "abc"})
    item2 = _mk_item(scope=Scope.USER, namespace=ns, raw_query="q2", ctx={"dataset_context": "abc"})
    item3 = _mk_item(scope=Scope.USER, namespace=ns, raw_query="q3", ctx={"dataset_context": "abc"})

    store.put(ns, item1.key, item1)
    store.put(ns, item2.key, item2)
    store.put(ns, item3.key, item3)

    first_pass = list(store.iter_namespace(ns))
    second_pass = list(store.iter_namespace(ns))

    assert [x.key for x in first_pass] == [item1.key, item2.key, item3.key]
    assert [x.key for x in second_pass] == [item1.key, item2.key, item3.key]

    stats = store.stats()
    assert stats.iter_calls == 2


def test_ram_store_iter_namespace_empty_namespace_returns_empty_iterator():
    store = RamStoreLRU(max_mb=8)
    items = list(store.iter_namespace("user:u1"))
    assert items == []

    stats = store.stats()
    assert stats.iter_calls == 1


def test_ram_store_can_store_items_with_embeddings():
    store = RamStoreLRU(max_mb=8)
    ns = "user:u1"

    item = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="semantic query",
        ctx={"dataset_context": "abc"},
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        embedding_norm=1.0,
    )

    store.put(ns, item.key, item)
    got = store.get(ns, item.key)

    assert got is not None
    assert got.query_embedding == [0.1, 0.2, 0.3, 0.4]
    assert got.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert got.embedding_norm == 1.0
    assert store.bytes_current() > 0


def test_ram_store_lru_eviction_happens_under_small_budget():
    store = RamStoreLRU(max_mb=1)  # 1 MB
    ns = "user:u1"

    items = []
    for i in range(10):
        it = _mk_item(
            scope=Scope.USER,
            namespace=ns,
            raw_query=f"q{i}",
            ctx={"dataset_context": "abc"},
            answer_text=("X" * 200_000) + f"_{i}",
        )
        items.append(it)
        store.put(ns, it.key, it)

    stats = store.stats()
    assert stats.evictions > 0
    assert store.bytes_current() <= store.capacity_bytes()

    remaining = 0
    for it in items:
        if store.get(ns, it.key) is not None:
            remaining += 1

    assert remaining < len(items)


def test_ram_store_eviction_is_namespace_aware_and_deterministic():
    """
    The implementation does global eviction across namespaces by comparing each
    namespace's LRU head using last_access_utc and lexicographic tie-breaking.
    """
    store = RamStoreLRU(max_mb=1)

    ns_a = "cohort:c1"
    ns_b = "user:u1"

    item_a = _mk_item(
        scope=Scope.COHORT,
        namespace=ns_a,
        raw_query="qa",
        ctx={"dataset_context": "abc"},
        answer_text="A" * 400_000,
    )
    item_b = _mk_item(
        scope=Scope.USER,
        namespace=ns_b,
        raw_query="qb",
        ctx={"dataset_context": "abc"},
        answer_text="B" * 400_000,
    )

    store.put(ns_a, item_a.key, item_a)
    store.put(ns_b, item_b.key, item_b)

    ns_c = "global:trec"
    item_c = _mk_item(
        scope=Scope.GLOBAL,
        namespace=ns_c,
        raw_query="qc",
        ctx={"dataset_context": "abc"},
        answer_text="C" * 400_000,
    )
    store.put(ns_c, item_c.key, item_c)

    assert store.stats().evictions > 0

    got_a = store.get(ns_a, item_a.key)
    got_b = store.get(ns_b, item_b.key)
    got_c = store.get(ns_c, item_c.key)

    missing = [name for name, got in [("a", got_a), ("b", got_b), ("c", got_c)] if got is None]
    assert len(missing) >= 1
    if len(missing) == 1:
        assert missing[0] == "a"


def test_ram_store_clear_resets_state_and_stats():
    store = RamStoreLRU(max_mb=8)
    ns = "user:u1"
    item = _mk_item(scope=Scope.USER, namespace=ns, raw_query="q", ctx={"dataset_context": "a"})
    store.put(ns, item.key, item)
    assert store.bytes_current() > 0

    store.clear()

    assert store.bytes_current() == 0
    stats = store.stats()
    assert stats.puts == 0
    assert stats.gets == 0
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.deletes == 0
    assert stats.iter_calls == 0
    assert stats.evictions == 0
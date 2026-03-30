from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memarch.memory.disk_store import DiskStoreSQLite
from memarch.memory.schema import MemoryItem, Provenance, QualitySignals, Scope
from memarch.utils.text import canonicalize, context_signature, make_key


def _mk_item(
    *,
    scope: Scope,
    namespace: str,
    raw_query: str,
    ctx: dict,
    answer_text: str = "This is a sufficiently long answer to be stored on disk.",
    model_id: str = "mistral-7b-instruct",
    prompt_version: str = "v1",
    ttl_seconds: int | None = None,
    expires_at_utc: datetime | None = None,
    query_embedding: list[float] | None = None,
    embedding_model_id: str | None = None,
    embedding_norm: float | None = None,
    evidence_text: str | None = None,
    doc_signature: str | None = None,
    source_file: str | None = None,
    source_id: str | None = None,
    chunk_index: int | None = None,
    chunk_id: str | None = None,
    question_type: str | None = None,
    answer_canonical: str | None = None,
) -> MemoryItem:
    """
    Helper to create a valid MemoryItem with deterministic keying.
    We allow TTL/expires fields to test freshness semantics and serialization.
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
        context_window=4096,
    )

    resolved_doc_signature = doc_signature if doc_signature is not None else ctx.get("doc_signature")
    resolved_source_id = source_id if source_id is not None else ctx.get("source_id")

    return MemoryItem(
        key=key,
        scope=scope,
        namespace=namespace,
        query_canonical=q_can,
        context_signature=ctx_sig,
        answer_text=answer_text,
        provenance=prov,
        quality=QualitySignals(score=1.0, success=True, metrics={"em": 1.0}),
        ttl_seconds=ttl_seconds,
        expires_at_utc=expires_at_utc,
        meta={
            "unit_test": True,
            "task": "trec",
            "doc_signature": resolved_doc_signature,
            "source_file": source_file,
            "source_id": resolved_source_id,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "question_type": question_type,
            "evidence_text": evidence_text,
            "answer_canonical": answer_canonical,
        },
        evidence_text=evidence_text,
        doc_signature=resolved_doc_signature,
        source_file=source_file,
        source_id=resolved_source_id,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        question_type=question_type,
        answer_canonical=answer_canonical,
        query_embedding=query_embedding,
        embedding_model_id=embedding_model_id,
        embedding_norm=embedding_norm,
    )


def test_disk_store_put_get_roundtrip(tmp_path):
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns = "user:u1"
    item = _mk_item(scope=Scope.USER, namespace=ns, raw_query="hello", ctx={"dataset_context": "abc"})
    store.put(ns, item.key, item)

    got = store.get(ns, item.key)
    assert got is not None
    assert got.key == item.key
    assert got.namespace == item.namespace
    assert got.scope == item.scope
    assert got.query_canonical == item.query_canonical
    assert got.context_signature == item.context_signature
    assert got.answer_text == item.answer_text

    # Provenance roundtrip
    assert got.provenance.model_id == item.provenance.model_id
    assert got.provenance.prompt_version == item.provenance.prompt_version
    assert got.provenance.quantization == item.provenance.quantization

    # Quality roundtrip
    assert got.quality.success is True
    assert got.quality.metrics.get("em") == 1.0

    stats = store.stats()
    assert stats["puts"] == 1
    assert stats["gets"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 0

    store.close()


def test_disk_store_persists_across_reopen(tmp_path):
    """
    Disk tier must persist across process restarts / runs.
    We simulate that by closing and reopening the SQLite store.
    """
    db_path = tmp_path / "mem.sqlite"

    ns = "user:u1"
    item = _mk_item(scope=Scope.USER, namespace=ns, raw_query="persist", ctx={"dataset_context": "abc"})

    store1 = DiskStoreSQLite(str(db_path))
    store1.put(ns, item.key, item)
    store1.close()

    store2 = DiskStoreSQLite(str(db_path))
    got = store2.get(ns, item.key)
    assert got is not None
    assert got.answer_text == item.answer_text
    store2.close()


def test_disk_store_delete_removes_item(tmp_path):
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns = "user:u1"
    item = _mk_item(scope=Scope.USER, namespace=ns, raw_query="delete", ctx={"dataset_context": "abc"})
    store.put(ns, item.key, item)

    assert store.get(ns, item.key) is not None

    store.delete(ns, item.key)
    assert store.get(ns, item.key) is None

    stats = store.stats()
    assert stats["deletes"] >= 1

    store.close()


def test_disk_store_iter_namespace_returns_only_that_namespace(tmp_path):
    """
    Ensures namespace isolation at the disk tier and that iter_namespace works.
    """
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns_user = "user:u1"
    ns_global = "global:trec"

    i1 = _mk_item(scope=Scope.USER, namespace=ns_user, raw_query="q1", ctx={"dataset_context": "abc"})
    i2 = _mk_item(scope=Scope.USER, namespace=ns_user, raw_query="q2", ctx={"dataset_context": "abc"})
    i3 = _mk_item(scope=Scope.GLOBAL, namespace=ns_global, raw_query="q3", ctx={"dataset_context": "abc"})

    store.put(ns_user, i1.key, i1)
    store.put(ns_user, i2.key, i2)
    store.put(ns_global, i3.key, i3)

    user_items = list(store.iter_namespace(ns_user))
    global_items = list(store.iter_namespace(ns_global))

    assert {it.key for it in user_items} == {i1.key, i2.key}
    assert {it.key for it in global_items} == {i3.key}

    stats = store.stats()
    assert stats["iter_calls"] >= 2

    store.close()


def test_disk_store_iter_namespace_empty_namespace_returns_empty_list(tmp_path):
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    items = list(store.iter_namespace("user:u1"))
    assert items == []

    stats = store.stats()
    assert stats["iter_calls"] == 1

    store.close()


def test_disk_store_serializes_ttl_and_expires_at(tmp_path):
    """
    Verifies that TTL and expires_at fields survive disk roundtrip and remain UTC-aware.
    """
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns = "user:u1"
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=60)

    item = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="ttl_test",
        ctx={"dataset_context": "abc"},
        ttl_seconds=60,
        expires_at_utc=expires,
    )
    store.put(ns, item.key, item)

    got = store.get(ns, item.key)
    assert got is not None
    assert got.ttl_seconds == 60
    assert got.expires_at_utc is not None
    assert got.expires_at_utc.tzinfo is not None
    assert abs((got.expires_at_utc - expires).total_seconds()) < 1.0

    store.close()


def test_disk_store_put_requires_key_matches_item_key(tmp_path):
    """
    Defensive check: prevents writing an item under the wrong key.
    """
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns = "user:u1"
    item = _mk_item(scope=Scope.USER, namespace=ns, raw_query="q", ctx={"dataset_context": "abc"})

    with pytest.raises(ValueError):
        store.put(ns, "wrong_key", item)

    store.close()


def test_disk_store_get_miss_increments_miss_count(tmp_path):
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    got = store.get("user:u1", "does_not_exist")
    assert got is None

    stats = store.stats()
    assert stats["gets"] == 1
    assert stats["misses"] == 1
    assert stats["hits"] == 0

    store.close()


def test_disk_store_roundtrip_preserves_semantic_fields(tmp_path):
    """
    Semantic retrieval depends on embeddings surviving disk persistence.
    """
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns = "user:u1"
    item = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="semantic question",
        ctx={"dataset_context": "abc", "doc_signature": "doc123"},
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        embedding_norm=0.5477,
    )
    store.put(ns, item.key, item)

    got = store.get(ns, item.key)
    assert got is not None
    assert got.query_embedding == [0.1, 0.2, 0.3, 0.4]
    assert got.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert got.embedding_norm == pytest.approx(0.5477, abs=1e-6)

    store.close()


def test_disk_store_semantic_fields_persist_across_reopen(tmp_path):
    db_path = tmp_path / "mem.sqlite"
    ns = "user:u1"

    item = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="persist semantic",
        ctx={"dataset_context": "abc", "doc_signature": "docXYZ"},
        query_embedding=[0.5, 0.6, 0.7],
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
        embedding_norm=1.0488088,
    )

    store1 = DiskStoreSQLite(str(tmp_path / "mem.sqlite"))
    store1.put(ns, item.key, item)
    store1.close()

    store2 = DiskStoreSQLite(str(tmp_path / "mem.sqlite"))
    got = store2.get(ns, item.key)

    assert got is not None
    assert got.query_embedding == [0.5, 0.6, 0.7]
    assert got.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert got.embedding_norm == pytest.approx(1.0488088, abs=1e-6)

    store2.close()


def test_disk_store_allows_items_without_semantic_fields(tmp_path):
    """
    Backward compatibility: items without embedding fields should still load cleanly.
    """
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns = "user:u1"
    item = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="legacy style",
        ctx={"dataset_context": "abc"},
        query_embedding=None,
        embedding_model_id=None,
        embedding_norm=None,
    )
    store.put(ns, item.key, item)

    got = store.get(ns, item.key)
    assert got is not None
    assert got.query_embedding is None
    assert got.embedding_model_id is None
    assert got.embedding_norm is None

    store.close()


def test_disk_store_roundtrip_preserves_evidence_guided_fields(tmp_path):
    """
    Evidence-guided retrieval depends on these fields surviving disk persistence.
    """
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns = "user:u1"
    item = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="who founded the company",
        ctx={
            "dataset_context": "The company was founded in 1998 by Alice Doe.",
            "doc_signature": "doc-evidence-1",
            "source_id": "company-founding",
        },
        evidence_text="The company was founded in 1998 by Alice Doe.",
        doc_signature="doc-evidence-1",
        source_file="trec_train.jsonl",
        source_id="company-founding",
        chunk_index=7,
        chunk_id="chunk-7",
        question_type="qa",
        answer_canonical="Alice Doe",
    )
    store.put(ns, item.key, item)

    got = store.get(ns, item.key)
    assert got is not None
    assert got.evidence_text == "The company was founded in 1998 by Alice Doe."
    assert got.doc_signature == "doc-evidence-1"
    assert got.source_file == "trec_train.jsonl"
    assert got.source_id == "company-founding"
    assert got.chunk_index == 7
    assert got.chunk_id == "chunk-7"
    assert got.question_type == "qa"
    assert got.answer_canonical == "Alice Doe"

    # Backward-compatibility mirror in meta should also survive.
    assert got.meta.get("doc_signature") == "doc-evidence-1"
    assert got.meta.get("source_file") == "trec_train.jsonl"
    assert got.meta.get("source_id") == "company-founding"
    assert got.meta.get("chunk_index") == 7
    assert got.meta.get("chunk_id") == "chunk-7"
    assert got.meta.get("question_type") == "qa"
    assert got.meta.get("evidence_text") == "The company was founded in 1998 by Alice Doe."
    assert got.meta.get("answer_canonical") == "Alice Doe"

    store.close()


def test_disk_store_evidence_guided_fields_persist_across_reopen(tmp_path):
    db_path = tmp_path / "mem.sqlite"
    ns = "user:u1"

    item = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="where is the capital located",
        ctx={
            "dataset_context": "The capital city is located on the northern coast.",
            "doc_signature": "doc-evidence-2",
            "source_id": "geo-capital",
        },
        evidence_text="The capital city is located on the northern coast.",
        doc_signature="doc-evidence-2",
        source_file="geo_eval.jsonl",
        source_id="geo-capital",
        chunk_index=3,
        chunk_id="geo-3",
        question_type="qa",
        answer_canonical="northern coast",
    )

    store1 = DiskStoreSQLite(str(db_path))
    store1.put(ns, item.key, item)
    store1.close()

    store2 = DiskStoreSQLite(str(db_path))
    got = store2.get(ns, item.key)

    assert got is not None
    assert got.evidence_text == "The capital city is located on the northern coast."
    assert got.doc_signature == "doc-evidence-2"
    assert got.source_file == "geo_eval.jsonl"
    assert got.source_id == "geo-capital"
    assert got.chunk_index == 3
    assert got.chunk_id == "geo-3"
    assert got.question_type == "qa"
    assert got.answer_canonical == "northern coast"

    store2.close()


def test_disk_store_iter_namespace_preserves_evidence_guided_fields(tmp_path):
    db_path = tmp_path / "mem.sqlite"
    store = DiskStoreSQLite(str(db_path))

    ns = "user:u1"
    i1 = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="q1",
        ctx={"dataset_context": "ctx1", "doc_signature": "doc-a", "source_id": "src-a"},
        evidence_text="evidence one",
        doc_signature="doc-a",
        source_file="a.jsonl",
        source_id="src-a",
        chunk_index=1,
        chunk_id="a-1",
        question_type="qa",
        answer_canonical="ans1",
    )
    i2 = _mk_item(
        scope=Scope.USER,
        namespace=ns,
        raw_query="q2",
        ctx={"dataset_context": "ctx2", "doc_signature": "doc-b", "source_id": "src-b"},
        evidence_text="evidence two",
        doc_signature="doc-b",
        source_file="b.jsonl",
        source_id="src-b",
        chunk_index=2,
        chunk_id="b-2",
        question_type="classification",
        answer_canonical="DESC",
    )

    store.put(ns, i1.key, i1)
    store.put(ns, i2.key, i2)

    items = list(store.iter_namespace(ns))
    by_key = {it.key: it for it in items}

    assert by_key[i1.key].evidence_text == "evidence one"
    assert by_key[i1.key].doc_signature == "doc-a"
    assert by_key[i1.key].source_file == "a.jsonl"
    assert by_key[i1.key].source_id == "src-a"
    assert by_key[i1.key].chunk_index == 1
    assert by_key[i1.key].chunk_id == "a-1"
    assert by_key[i1.key].question_type == "qa"
    assert by_key[i1.key].answer_canonical == "ans1"

    assert by_key[i2.key].evidence_text == "evidence two"
    assert by_key[i2.key].doc_signature == "doc-b"
    assert by_key[i2.key].source_file == "b.jsonl"
    assert by_key[i2.key].source_id == "src-b"
    assert by_key[i2.key].chunk_index == 2
    assert by_key[i2.key].chunk_id == "b-2"
    assert by_key[i2.key].question_type == "classification"
    assert by_key[i2.key].answer_canonical == "DESC"

    store.close()
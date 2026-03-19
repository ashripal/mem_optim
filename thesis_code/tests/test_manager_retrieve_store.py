# tests/test_manager_retrieve_store.py
#
# These tests validate the end-to-end behavior of MemoryManager across:
#   - tiered retrieval (RAM then DISK)
#   - disk-hit promotion back to RAM
#   - generator invocation on miss
#   - deterministic scoping/namespace ordering (SESSION -> USER -> COHORT -> GLOBAL)
#   - TTL/expiration gating (expired items must not be reused)
#   - Phase 1 semantic retrieval as context assistance after exact-match miss
#
# This is one of the most important gates because it proves the architecture
# actually behaves like a multi-tier memory system rather than a collection of modules.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pytest

from memarch.memory.disk_store import DiskStoreSQLite
from memarch.memory.manager import MemoryManager, MemoryManagerConfig
from memarch.memory.policy import RetrievalPolicy
from memarch.memory.ram_store import RamStoreLRU
from memarch.memory.schema import (
    MatchType,
    MemoryHit,
    MemoryItem,
    MemoryQuery,
    Provenance,
    QualitySignals,
    Scope,
)
from memarch.models.embedder import EmbedderConfig, HFEmbedder
from memarch.utils.text import canonicalize, context_signature, make_key


# -------------------------
# Test doubles
# -------------------------

@dataclass
class FakeGenerator:
    """
    Tiny deterministic generator used for unit tests.

    - Records how many times it was called
    - Records the retrieved hit passed into generation
    - Returns a stable answer based on input query
    """
    call_count: int = 0
    last_query: Optional[str] = None
    last_retrieved: Optional[MemoryHit] = None

    def generate(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> Tuple[str, Provenance, QualitySignals]:
        self.call_count += 1
        self.last_query = mq.raw_query
        self.last_retrieved = retrieved

        suffix = ""
        if retrieved is not None:
            suffix = f" | retrieved={retrieved.match_type.value}:{retrieved.score:.4f}"

        ans = (
            f"GEN:{mq.raw_query} | "
            f"ctx_len={len((mq.context or {}).get('dataset_context', '') or '')}"
            f"{suffix}"
        )
        prov = Provenance(
            model_id=mq.model_id,
            prompt_version=mq.prompt_version,
            generated_at_utc=datetime.now(timezone.utc),
            generator_backend="fake",
            quantization="Q4_K_M",
            context_window=4096,
        )
        qual = QualitySignals(score=1.0, success=True, metrics={"unit": 1.0})
        return ans, prov, qual


class TinyTestEmbedder:
    """
    Deterministic, dependency-free embedder for semantic retrieval tests.
    """
    def __init__(self, model_id: str = "tiny-test-embedder") -> None:
        self.cfg = type("Cfg", (), {"model_id": model_id})()

    def embed(self, text: str):
        text = (text or "").lower().strip()

        # Small handcrafted semantic mapping for deterministic tests.
        if "reset the device" in text:
            return [1.0, 0.0, 0.0]
        if "restart the device" in text:
            return [0.99, 0.01, 0.0]
        if "torque setting" in text:
            return [0.0, 1.0, 0.0]
        if "sensor a" in text:
            return [0.0, 0.0, 1.0]

        return [0.2, 0.2, 0.2]

    @staticmethod
    def embedding_norm(vec):
        return sum(float(x) * float(x) for x in vec) ** 0.5


# -------------------------
# Helpers
# -------------------------

def _mk_item_for_scope(
    *,
    scope: Scope,
    namespace: str,
    mq: MemoryQuery,
    answer_text: str,
    expires_at_utc: Optional[datetime] = None,
    query_embedding: Optional[list[float]] = None,
    embedding_model_id: Optional[str] = None,
    embedding_norm: Optional[float] = None,
    raw_query_override: Optional[str] = None,
) -> MemoryItem:
    """
    Create a MemoryItem consistent with manager keying.

    raw_query_override is useful for semantic tests where the stored query differs
    from the current query and therefore must not exact-match.
    """
    raw_query = raw_query_override if raw_query_override is not None else mq.raw_query
    q_can = canonicalize(raw_query)
    ctx_sig = context_signature(mq.context)

    key = make_key(
        scope=scope.value,
        namespace=namespace,
        task=mq.task,
        model_id=mq.model_id,
        prompt_version=mq.prompt_version,
        query_canonical=q_can,
        context_sig=ctx_sig,
    )

    prov = Provenance(
        model_id=mq.model_id,
        prompt_version=mq.prompt_version,
        generated_at_utc=datetime.now(timezone.utc),
        generator_backend="test",
        quantization="Q4_K_M",
        context_window=4096,
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
        expires_at_utc=expires_at_utc,
        meta={
            "unit_test": True,
            "task": mq.task,
            "doc_signature": mq.context.get("doc_signature"),
        },
        query_embedding=query_embedding,
        embedding_model_id=embedding_model_id,
        embedding_norm=embedding_norm,
    )


def _build_manager(
    tmp_path,
    *,
    retrieval_policy: Optional[RetrievalPolicy] = None,
    embedder=None,
) -> Tuple[MemoryManager, RamStoreLRU, DiskStoreSQLite]:
    """
    Build a fresh manager + stores for each test.
    """
    ram = RamStoreLRU(max_mb=8)
    disk = DiskStoreSQLite(str(tmp_path / "mem.sqlite"))

    mm_cfg = MemoryManagerConfig(
        retrieval_policy=retrieval_policy or RetrievalPolicy(
            scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL]
        ),
        promote_disk_hits_to_ram=True,
        return_memory_directly=True,
        embedder=embedder,
    )
    mgr = MemoryManager(ram=ram, disk=disk, cfg=mm_cfg)
    return mgr, ram, disk


# -------------------------
# Exact-match tests
# -------------------------

def test_manager_miss_calls_generator_and_stores_then_hits_from_ram(tmp_path):
    """
    Expected behavior:
      - First call: miss -> generator called -> stored -> returned generated answer
      - Second call: hit -> generator NOT called again -> returns memory answer
    """
    mgr, ram, disk = _build_manager(tmp_path)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="What is the label for class 3?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={"dataset_context": "This is a LongBench passage."},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    ans1, meta1 = mgr.answer(mq, gen)
    assert gen.call_count == 1
    assert meta1.get("generated") is True
    assert meta1.get("used_memory") is False
    assert ans1.startswith("GEN:")

    ans2, meta2 = mgr.answer(mq, gen)
    assert gen.call_count == 1
    assert meta2.get("used_memory") is True
    assert ans2 == ans1

    ram_stats = ram.stats()
    disk_stats = disk.stats()
    assert ram_stats.puts >= 1
    assert disk_stats["puts"] >= 1


def test_manager_disk_hit_promotes_to_ram(tmp_path):
    """
    Expected behavior:
      - Populate memory (generator miss path)
      - Clear RAM to simulate cold start
      - Next retrieval should hit DISK and then promote into RAM
      - A subsequent retrieval should hit RAM
    """
    mgr, ram, disk = _build_manager(tmp_path)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="Explain the error code E42.",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={"dataset_context": "Device manual excerpt..."},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    ans1, _meta1 = mgr.answer(mq, gen)
    assert gen.call_count == 1

    ram.clear()

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.source_tier.value == "disk"
    assert hit.item.answer_text == ans1

    hit2 = mgr.retrieve(mq)
    assert hit2 is not None
    assert hit2.source_tier.value == "ram"
    assert hit2.item.answer_text == ans1


def test_manager_respects_scope_order_session_over_user(tmp_path):
    """
    Validates SESSION -> USER ordering.
    """
    mgr, ram, disk = _build_manager(tmp_path)

    mq = MemoryQuery(
        raw_query="How do I reset the device?",
        user_id="u1",
        session_id="s1",
        cohort_id=None,
        task="trec",
        context={"dataset_context": "Reset instructions from LongBench."},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    ns_session = "session:s1"
    ns_user = "user:u1"

    session_item = _mk_item_for_scope(
        scope=Scope.SESSION, namespace=ns_session, mq=mq, answer_text="SESSION_ANSWER"
    )
    user_item = _mk_item_for_scope(
        scope=Scope.USER, namespace=ns_user, mq=mq, answer_text="USER_ANSWER"
    )

    disk.put(ns_session, session_item.key, session_item)
    disk.put(ns_user, user_item.key, user_item)

    ram.clear()

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.item.answer_text == "SESSION_ANSWER"
    assert hit.item.scope == Scope.SESSION


def test_manager_rejects_expired_items(tmp_path):
    """
    If an item is expired, MemoryManager.retrieve should return None even if it exists.
    """
    mgr, ram, disk = _build_manager(tmp_path)

    mq = MemoryQuery(
        raw_query="What is the recommended torque setting?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={"dataset_context": "Spec sheet context."},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    ns_user = "user:u1"
    expired_time = datetime.now(timezone.utc) - timedelta(seconds=10)

    expired_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        answer_text="EXPIRED_ANSWER",
        expires_at_utc=expired_time,
    )

    disk.put(ns_user, expired_item.key, expired_item)
    ram.put(ns_user, expired_item.key, expired_item)

    hit = mgr.retrieve(mq)
    assert hit is None


def test_manager_miss_after_expired_item_calls_generator(tmp_path):
    """
    If only expired memory exists, manager.answer should generate a fresh response.
    """
    mgr, ram, disk = _build_manager(tmp_path)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How to calibrate sensor A?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={"dataset_context": "Calibration procedure context."},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    ns_user = "user:u1"
    expired_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    expired_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        answer_text="OLD_EXPIRED",
        expires_at_utc=expired_time,
    )
    disk.put(ns_user, expired_item.key, expired_item)
    ram.put(ns_user, expired_item.key, expired_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 1
    assert meta.get("generated") is True
    assert ans.startswith("GEN:")

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.item.answer_text == ans


# -------------------------
# Semantic retrieval tests
# -------------------------

def test_manager_semantic_hit_is_used_as_generation_context_not_direct_bypass(tmp_path):
    """
    Phase 1 default:
      - exact miss
      - semantic hit found
      - generator is still called
      - retrieved semantic hit is passed into generator
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=1.01,  # disable bypass in Phase 1
    )
    embedder = TinyTestEmbedder()
    mgr, ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-reset-001",
        },
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        allow_semantic=True,
    )

    ns_user = "user:u1"
    stored_query = "How do I reset the device?"
    stored_vec = embedder.embed(stored_query)

    sem_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override=stored_query,
        answer_text="Press and hold the reset button for 10 seconds.",
        query_embedding=stored_vec,
        embedding_model_id=embedder.cfg.model_id,
        embedding_norm=embedder.embedding_norm(stored_vec),
    )
    disk.put(ns_user, sem_item.key, sem_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 1
    assert gen.last_retrieved is not None
    assert gen.last_retrieved.match_type == MatchType.SEMANTIC
    assert gen.last_retrieved.item.answer_text == "Press and hold the reset button for 10 seconds."
    assert meta["generated"] is True
    assert meta["used_memory"] is False
    assert meta["semantic_used"] is True
    assert meta["semantic_bypassed"] is False
    assert "retrieved=semantic" in ans


def test_manager_semantic_bypass_returns_directly_when_policy_allows(tmp_path):
    """
    If semantic score passes bypass threshold, manager may return directly.
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=0.95,
    )
    embedder = TinyTestEmbedder()
    mgr, _ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-reset-001",
        },
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        allow_semantic=True,
    )

    ns_user = "user:u1"
    stored_query = "How do I reset the device?"
    stored_vec = embedder.embed(stored_query)

    sem_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override=stored_query,
        answer_text="DIRECT_SEMANTIC_ANSWER",
        query_embedding=stored_vec,
        embedding_model_id=embedder.cfg.model_id,
        embedding_norm=embedder.embedding_norm(stored_vec),
    )
    disk.put(ns_user, sem_item.key, sem_item)

    ans, meta = mgr.answer(mq, gen)

    assert ans == "DIRECT_SEMANTIC_ANSWER"
    assert gen.call_count == 0
    assert meta["used_memory"] is True
    assert meta["generated"] is False
    assert meta["semantic_used"] is True
    assert meta["semantic_bypassed"] is True
    assert meta["match_type"] == "semantic"


def test_manager_semantic_candidate_rejected_on_document_mismatch(tmp_path):
    """
    Semantic retrieval should reject candidates from a different document when
    doc signatures are available and differ.
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=1.01,
    )
    embedder = TinyTestEmbedder()
    mgr, _ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-A",
        },
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        allow_semantic=True,
    )

    ns_user = "user:u1"
    stored_query = "How do I reset the device?"
    stored_vec = embedder.embed(stored_query)

    sem_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override=stored_query,
        answer_text="WRONG_DOCUMENT_ANSWER",
        query_embedding=stored_vec,
        embedding_model_id=embedder.cfg.model_id,
        embedding_norm=embedder.embedding_norm(stored_vec),
    )
    sem_item.meta["doc_signature"] = "doc-B"
    disk.put(ns_user, sem_item.key, sem_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 1
    assert gen.last_retrieved is None
    assert meta["semantic_used"] is False
    assert meta["generated"] is True
    assert ans.startswith("GEN:")


def test_manager_store_adds_embedding_fields_when_embedder_present(tmp_path):
    """
    Generated items should store query embeddings when an embedder is configured.
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=1.01,
    )
    embedder = TinyTestEmbedder()
    mgr, ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="What is the recommended torque setting?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={
            "dataset_context": "Spec sheet context.",
            "doc_signature": "doc-torque-001",
        },
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        allow_semantic=True,
    )

    ans, meta = mgr.answer(mq, gen)
    assert ans.startswith("GEN:")
    assert meta["generated"] is True

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.match_type == MatchType.EXACT
    assert hit.item.query_embedding is not None
    assert hit.item.embedding_model_id == embedder.cfg.model_id
    assert hit.item.embedding_norm is not None
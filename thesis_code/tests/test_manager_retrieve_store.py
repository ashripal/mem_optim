from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

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

        # Make TREC outputs valid short labels so admission can store them.
        if (mq.task or "").strip().lower() == "trec":
            ans = "DESC" if "label" in mq.raw_query.lower() else "HUM"
        else:
            if retrieved is not None:
                ans = (
                    "Generated answer using retrieved supporting memory and current document context."
                )
            else:
                ans = (
                    "Generated answer using the current document context only."
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
        if "reboot the device" in text:
            return [0.98, 0.02, 0.0]
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

def _construct_with_supported_kwargs(cls: Any, **kwargs: Any) -> Any:
    """
    Construct an object using only kwargs supported by its signature.

    This keeps tests forward-compatible while the memarch internals evolve.
    """
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**kwargs)

    supported = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return cls(**filtered)


def _match_type_value(x: Any) -> Any:
    if x is None:
        return None
    return getattr(x, "value", x)


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
    evidence_text: Optional[str] = None,
    doc_signature: Optional[str] = None,
    source_file: Optional[str] = None,
    source_id: Optional[str] = None,
    chunk_index: Optional[int] = None,
    chunk_id: Optional[str] = None,
    question_type: Optional[str] = None,
    answer_canonical: Optional[str] = None,
) -> MemoryItem:
    """
    Create a MemoryItem consistent with manager keying.

    raw_query_override is useful for semantic/lexical tests where the stored query differs
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

    resolved_doc_signature = (
        doc_signature
        if doc_signature is not None
        else mq.doc_signature
        if getattr(mq, "doc_signature", None) is not None
        else mq.context.get("doc_signature")
    )

    resolved_source_id = (
        source_id
        if source_id is not None
        else mq.source_id
        if getattr(mq, "source_id", None) is not None
        else mq.context.get("source_id")
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


def _build_manager(
    tmp_path,
    *,
    retrieval_policy: Optional[RetrievalPolicy] = None,
    embedder=None,
    lexical_enabled: bool = False,
    lexical_context_threshold: float = 0.55,
    lexical_direct_threshold: float = 0.90,
    lexical_top_k: int = 3,
    prefer_same_source: bool = True,
    safe_direct_reuse_tasks: Optional[list[str]] = None,
) -> Tuple[MemoryManager, RamStoreLRU, DiskStoreSQLite]:
    """
    Build a fresh manager + stores for each test.
    """
    ram = RamStoreLRU(max_mb=8)
    disk = DiskStoreSQLite(str(tmp_path / "mem.sqlite"))

    policy = retrieval_policy or _construct_with_supported_kwargs(
        RetrievalPolicy,
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        lexical_enabled=lexical_enabled,
        lexical_threshold_context=lexical_context_threshold,
        lexical_threshold_bypass=lexical_direct_threshold,
        lexical_top_k=lexical_top_k,
        prefer_same_source=prefer_same_source,
        safe_direct_reuse_tasks=safe_direct_reuse_tasks or ["trec"],
    )

    mm_cfg = _construct_with_supported_kwargs(
        MemoryManagerConfig,
        retrieval_policy=policy,
        promote_disk_hits_to_ram=True,
        return_memory_directly=True,
        embedder=embedder,
        lexical_enabled=lexical_enabled,
        lexical_context_threshold=lexical_context_threshold,
        lexical_direct_threshold=lexical_direct_threshold,
        lexical_top_k=lexical_top_k,
        prefer_same_source=prefer_same_source,
        safe_direct_reuse_tasks=safe_direct_reuse_tasks or ["trec"],
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
        question_type="classification",
    )

    ans1, meta1 = mgr.answer(mq, gen)
    assert gen.call_count == 1
    assert meta1.get("generated") is True
    assert meta1.get("used_memory") is False
    assert ans1 == "DESC"

    ans2, meta2 = mgr.answer(mq, gen)
    assert gen.call_count == 1
    assert meta2.get("used_memory") is True
    assert ans2 == ans1

    ram_stats = ram.stats()
    disk_stats = disk.stats()
    assert ram_stats.puts >= 1
    assert disk_stats["puts"] >= 1


def test_manager_exact_normalized_hit_collapses_whitespace(tmp_path):
    """
    Exact retrieval should still succeed when canonicalization collapses whitespace
    between the stored query and the incoming query.
    """
    mgr, _ram, disk = _build_manager(tmp_path)

    mq = MemoryQuery(
        raw_query="Who founded the company?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={"dataset_context": "The company was founded by Alice Doe."},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    ns_session = "session:s1"
    item = _mk_item_for_scope(
        scope=Scope.SESSION,
        namespace=ns_session,
        mq=mq,
        raw_query_override="Who   founded   the company?",
        answer_text="Alice Doe",
    )
    disk.put(ns_session, item.key, item)

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert _match_type_value(hit.match_type) == "exact"
    assert hit.item.answer_text == "Alice Doe"


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
        raw_query="Who documented error code E42?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
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
        task="qa_task",
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
        task="qa_task",
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
        task="qa_task",
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
    assert ans == "Generated answer using the current document context only."

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.item.answer_text == ans


# -------------------------
# Lexical retrieval tests
# -------------------------

def test_manager_lexical_same_source_retrieval_prefers_same_document_candidate(tmp_path):
    """
    When lexical retrieval is enabled and multiple approximate candidates exist,
    a same-source/same-document candidate should be preferred over a broader one.
    """
    mgr, _ram, disk = _build_manager(
        tmp_path,
        lexical_enabled=True,
        lexical_context_threshold=0.50,
        lexical_direct_threshold=0.95,
        prefer_same_source=True,
    )

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-A",
            "source_file": "manual_a.jsonl",
            "source_id": "manual-a",
        },
        doc_signature="doc-A",
        source_file="manual_a.jsonl",
        source_id="manual-a",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        question_type="qa",
        evidence_text="Restart instructions mention the front-panel button.",
    )

    ns_user = "user:u1"

    broader_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override="How do I reboot the device?",
        answer_text="BROADER_DOC_ANSWER",
        evidence_text="Broader document reboot instructions.",
        doc_signature="doc-B",
        source_file="manual_b.jsonl",
        source_id="manual-b",
        question_type="qa",
    )
    same_doc_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override="How do I reset the device?",
        answer_text="SAME_DOC_ANSWER",
        evidence_text="Same document restart instructions.",
        doc_signature="doc-A",
        source_file="manual_a.jsonl",
        source_id="manual-a",
        question_type="qa",
    )

    disk.put(ns_user, broader_item.key, broader_item)
    disk.put(ns_user, same_doc_item.key, same_doc_item)

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert _match_type_value(hit.match_type) == "lexical"
    assert hit.item.answer_text == "SAME_DOC_ANSWER"


def test_manager_lexical_context_only_route_calls_generator_with_retrieved_hit(tmp_path):
    """
    For non-safe open-ended QA, a lexical hit should be used as retrieved context
    rather than directly bypassing generation.
    """
    mgr, _ram, disk = _build_manager(
        tmp_path,
        lexical_enabled=True,
        lexical_context_threshold=0.50,
        lexical_direct_threshold=0.90,
        prefer_same_source=True,
        safe_direct_reuse_tasks=["trec"],
    )
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-reset-001",
            "source_file": "manual.jsonl",
            "source_id": "manual-reset",
        },
        doc_signature="doc-reset-001",
        source_file="manual.jsonl",
        source_id="manual-reset",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        question_type="qa",
        evidence_text="Restart instructions mention the front-panel button.",
    )

    ns_user = "user:u1"
    lex_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override="How do I reboot the device?",
        answer_text="Hold the reset button for ten seconds.",
        evidence_text="Hold the reset button for ten seconds.",
        doc_signature="doc-reset-001",
        source_file="manual.jsonl",
        source_id="manual-reset",
        question_type="qa",
        answer_canonical="Hold the reset button for ten seconds.",
    )
    disk.put(ns_user, lex_item.key, lex_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 1
    assert gen.last_retrieved is not None
    assert _match_type_value(gen.last_retrieved.match_type) == "lexical"
    assert meta.get("generated") is True
    assert meta.get("used_memory") is True
    assert meta.get("lexical_used") is True
    assert meta.get("lexical_bypassed") is False
    assert "retrieved supporting memory" in ans


def test_manager_lexical_direct_route_bypasses_generator_on_safe_task(tmp_path):
    """
    For safe short-label tasks like TREC, a strong lexical hit should be allowed
    to bypass generation directly.
    """
    mgr, _ram, disk = _build_manager(
        tmp_path,
        lexical_enabled=True,
        lexical_context_threshold=0.50,
        lexical_direct_threshold=0.75,
        prefer_same_source=True,
        safe_direct_reuse_tasks=["trec"],
    )
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="What label should class 3 get?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={
            "dataset_context": "Short TREC context.",
            "doc_signature": "doc-trec-1",
            "source_file": "trec.jsonl",
            "source_id": "trec-main",
        },
        doc_signature="doc-trec-1",
        source_file="trec.jsonl",
        source_id="trec-main",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        question_type="classification",
        answer_canonical="DESC",
    )

    ns_user = "user:u1"
    lex_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override="What is the label for class 3?",
        answer_text="DESC",
        evidence_text="Class 3 maps to DESC.",
        doc_signature="doc-trec-1",
        source_file="trec.jsonl",
        source_id="trec-main",
        question_type="classification",
        answer_canonical="DESC",
    )
    disk.put(ns_user, lex_item.key, lex_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 0
    assert ans == "DESC"
    assert meta.get("used_memory") is True
    assert meta.get("generated") is False
    assert meta.get("lexical_used") is True
    assert meta.get("lexical_bypassed") is True


def test_manager_lexical_candidate_below_threshold_is_rejected(tmp_path):
    """
    An unrelated lexical candidate should be rejected and the manager should fall
    back to normal generation.
    """
    mgr, _ram, disk = _build_manager(
        tmp_path,
        lexical_enabled=True,
        lexical_context_threshold=0.70,
        lexical_direct_threshold=0.90,
        prefer_same_source=True,
        safe_direct_reuse_tasks=["trec"],
    )
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-reset-001",
            "source_file": "manual.jsonl",
            "source_id": "manual-reset",
        },
        doc_signature="doc-reset-001",
        source_file="manual.jsonl",
        source_id="manual-reset",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        question_type="qa",
    )

    ns_user = "user:u1"
    unrelated_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override="What is the population of France?",
        answer_text="67 million",
        evidence_text="France population is 67 million.",
        doc_signature="doc-france-1",
        source_file="facts.jsonl",
        source_id="facts-france",
        question_type="qa",
    )
    disk.put(ns_user, unrelated_item.key, unrelated_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 1
    assert gen.last_retrieved is None
    assert meta.get("generated") is True
    assert meta.get("used_memory") is False
    assert meta.get("lexical_used") in (False, None)
    assert ans == "Generated answer using the current document context only."


# -------------------------
# Semantic retrieval tests
# -------------------------

def test_manager_semantic_hit_is_used_as_generation_context_not_direct_bypass(tmp_path):
    """
    Evidence-guided default:
      - exact miss
      - semantic hit found
      - generator is still called
      - retrieved semantic hit is passed into generator
      - semantic bypass is disabled
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=1.01,
        allow_semantic_bypass=False,
    )
    embedder = TinyTestEmbedder()
    mgr, ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-reset-001",
            "source_id": "manual-reset",
        },
        doc_signature="doc-reset-001",
        source_id="manual-reset",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        allow_semantic=True,
        question_type="qa",
        evidence_text="Restart instructions mention the front-panel button.",
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
        evidence_text="Press and hold the reset button for 10 seconds.",
        doc_signature="doc-reset-001",
        source_file="manual.jsonl",
        source_id="manual-reset",
        chunk_index=4,
        chunk_id="reset-4",
        question_type="qa",
        answer_canonical="Press and hold the reset button for 10 seconds.",
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
    assert meta["used_memory"] is True
    assert meta["semantic_used"] is True
    assert meta["semantic_bypassed"] is False
    assert gen.last_retrieved is not None
    assert gen.last_retrieved.match_type == MatchType.SEMANTIC


def test_manager_semantic_hit_never_direct_bypasses_even_with_low_bypass_threshold(tmp_path):
    """
    New spec:
      semantic retrieval is context-only, so direct bypass must not happen
      even if the bypass threshold would otherwise allow it.
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=0.95,
        allow_semantic_bypass=False,
    )
    embedder = TinyTestEmbedder()
    mgr, _ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-reset-001",
            "source_id": "manual-reset",
        },
        doc_signature="doc-reset-001",
        source_id="manual-reset",
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
        evidence_text="Hold reset for ten seconds.",
        doc_signature="doc-reset-001",
        source_id="manual-reset",
        query_embedding=stored_vec,
        embedding_model_id=embedder.cfg.model_id,
        embedding_norm=embedder.embedding_norm(stored_vec),
    )
    disk.put(ns_user, sem_item.key, sem_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 1
    assert gen.last_retrieved is not None
    assert meta["used_memory"] is True
    assert meta["generated"] is True
    assert meta["semantic_used"] is True
    assert meta["semantic_bypassed"] is False
    assert ans != "DIRECT_SEMANTIC_ANSWER"


def test_manager_semantic_prefers_same_document_candidate(tmp_path):
    """
    If both same-document and broader semantic candidates are available,
    the same-document candidate should be chosen first.
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=1.01,
        allow_semantic_bypass=False,
        prefer_same_document_for_semantic=True,
    )
    embedder = TinyTestEmbedder()
    mgr, _ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-A",
            "source_id": "manual-a",
        },
        doc_signature="doc-A",
        source_id="manual-a",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        allow_semantic=True,
        question_type="qa",
    )

    ns_user = "user:u1"
    stored_query = "How do I reset the device?"
    stored_vec = embedder.embed(stored_query)

    broader_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override=stored_query,
        answer_text="BROADER_DOC_ANSWER",
        evidence_text="Broader document reset instructions.",
        doc_signature="doc-B",
        source_id="manual-b",
        query_embedding=stored_vec,
        embedding_model_id=embedder.cfg.model_id,
        embedding_norm=embedder.embedding_norm(stored_vec),
    )
    same_doc_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override="How do I reboot the device?",
        answer_text="SAME_DOC_ANSWER",
        evidence_text="Same document restart instructions.",
        doc_signature="doc-A",
        source_id="manual-a",
        query_embedding=embedder.embed("How do I reboot the device?"),
        embedding_model_id=embedder.cfg.model_id,
        embedding_norm=embedder.embedding_norm(embedder.embed("How do I reboot the device?")),
    )

    disk.put(ns_user, broader_item.key, broader_item)
    disk.put(ns_user, same_doc_item.key, same_doc_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 1
    assert gen.last_retrieved is not None
    assert gen.last_retrieved.match_type == MatchType.SEMANTIC
    assert gen.last_retrieved.item.answer_text == "SAME_DOC_ANSWER"
    assert meta["semantic_used"] is True
    assert meta["semantic_bypassed"] is False


def test_manager_semantic_broader_candidate_used_when_no_same_document_available(tmp_path):
    """
    Broader semantic candidates should still be usable for context assistance
    when no same-document candidate survives.
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=1.01,
        allow_semantic_bypass=False,
        prefer_same_document_for_semantic=True,
    )
    embedder = TinyTestEmbedder()
    mgr, _ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "Device troubleshooting manual.",
            "doc_signature": "doc-A",
            "source_id": "manual-a",
        },
        doc_signature="doc-A",
        source_id="manual-a",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        allow_semantic=True,
    )

    ns_user = "user:u1"
    stored_query = "How do I reset the device?"
    stored_vec = embedder.embed(stored_query)

    broader_item = _mk_item_for_scope(
        scope=Scope.USER,
        namespace=ns_user,
        mq=mq,
        raw_query_override=stored_query,
        answer_text="BROADER_DOC_ANSWER",
        evidence_text="Broader document reset instructions.",
        doc_signature="doc-B",
        source_id="manual-b",
        query_embedding=stored_vec,
        embedding_model_id=embedder.cfg.model_id,
        embedding_norm=embedder.embedding_norm(stored_vec),
    )
    disk.put(ns_user, broader_item.key, broader_item)

    ans, meta = mgr.answer(mq, gen)

    assert gen.call_count == 1
    assert gen.last_retrieved is not None
    assert gen.last_retrieved.item.answer_text == "BROADER_DOC_ANSWER"
    assert meta["semantic_used"] is True
    assert meta["semantic_bypassed"] is False


def test_manager_store_adds_embedding_fields_when_embedder_present(tmp_path):
    """
    Generated items should store query embeddings when an embedder is configured.
    """
    pol = RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL],
        semantic_enabled=True,
        semantic_threshold_context=0.85,
        semantic_threshold_bypass=1.01,
        allow_semantic_bypass=False,
    )
    embedder = TinyTestEmbedder()
    mgr, ram, disk = _build_manager(tmp_path, retrieval_policy=pol, embedder=embedder)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="What is the recommended torque setting?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "Spec sheet context.",
            "doc_signature": "doc-torque-001",
            "source_id": "spec-torque",
        },
        doc_signature="doc-torque-001",
        source_id="spec-torque",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        allow_semantic=True,
        evidence_text="Torque specification appears in section 2.",
    )

    ans, meta = mgr.answer(mq, gen)
    assert ans == "Generated answer using the current document context only."
    assert meta["generated"] is True

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.match_type == MatchType.EXACT
    assert hit.item.query_embedding is not None
    assert hit.item.embedding_model_id == embedder.cfg.model_id
    assert hit.item.embedding_norm is not None


def test_manager_store_persists_evidence_guided_fields(tmp_path):
    """
    Generated items should store evidence/source fields explicitly so later
    retrieval can use them for same-document preference and reduced prompting.
    """
    mgr, ram, disk = _build_manager(tmp_path)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="Who founded the company?",
        user_id="u1",
        session_id="s1",
        task="qa_task",
        context={
            "dataset_context": "The company was founded in 1998 by Alice Doe.",
            "doc_signature": "doc-company-1",
            "source_file": "company.jsonl",
            "source_id": "company-main",
            "chunk_index": 5,
            "chunk_id": "company-5",
            "question_type": "qa",
            "evidence_text": "The company was founded in 1998 by Alice Doe.",
            "answer_canonical": "Alice Doe",
        },
        doc_signature="doc-company-1",
        source_file="company.jsonl",
        source_id="company-main",
        chunk_index=5,
        chunk_id="company-5",
        question_type="qa",
        evidence_text="The company was founded in 1998 by Alice Doe.",
        answer_canonical="Alice Doe",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    ans, meta = mgr.answer(mq, gen)
    assert meta["generated"] is True

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.match_type == MatchType.EXACT

    item = hit.item
    assert item.doc_signature == "doc-company-1"
    assert item.source_file == "company.jsonl"
    assert item.source_id == "company-main"
    assert item.chunk_index == 5
    assert item.chunk_id == "company-5"
    assert item.question_type == "qa"
    assert item.evidence_text == "The company was founded in 1998 by Alice Doe."
    assert item.answer_canonical == "Alice Doe"

    assert item.meta.get("doc_signature") == "doc-company-1"
    assert item.meta.get("source_file") == "company.jsonl"
    assert item.meta.get("source_id") == "company-main"
    assert item.meta.get("chunk_index") == 5
    assert item.meta.get("chunk_id") == "company-5"
    assert item.meta.get("question_type") == "qa"
    assert item.meta.get("evidence_text") == "The company was founded in 1998 by Alice Doe."
    assert item.meta.get("answer_canonical") == "Alice Doe"


def test_manager_store_allows_short_valid_trec_label(tmp_path):
    """
    Admission should allow valid short classification outputs like TREC labels.
    """
    mgr, ram, disk = _build_manager(tmp_path)
    gen = FakeGenerator()

    mq = MemoryQuery(
        raw_query="What is the label for class 7?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={"dataset_context": "Short TREC context."},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        question_type="classification",
    )

    ans, meta = mgr.answer(mq, gen)
    assert ans == "DESC"
    assert meta["generated"] is True

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.match_type == MatchType.EXACT
    assert hit.item.answer_text == "DESC"
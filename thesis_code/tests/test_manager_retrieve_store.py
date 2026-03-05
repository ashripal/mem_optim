# tests/test_manager_retrieve_store.py
#
# These tests validate the *end-to-end* behavior of MemoryManager across:
#   - tiered retrieval (RAM then DISK)
#   - disk-hit promotion back to RAM
#   - generator invocation on miss
#   - deterministic scoping/namespace ordering (SESSION -> USER -> COHORT -> GLOBAL)
#   - TTL/expiration gating (expired items must not be reused)
#
# This is one of the most important Phase 1 gates because it proves the architecture
# actually behaves like a multi-tier cache rather than a collection of independent modules.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pytest

from memarch.memory.disk_store import DiskStoreSQLite
from memarch.memory.manager import MemoryManager, MemoryManagerConfig
from memarch.memory.ram_store import RamStoreLRU
from memarch.memory.schema import (
    MemoryItem,
    MemoryQuery,
    Provenance,
    QualitySignals,
    Scope,
    MemoryHit,
)
from memarch.utils.text import canonicalize, context_signature, make_key


# -------------------------
# Test doubles
# -------------------------

@dataclass
class FakeGenerator:
    """
    A tiny deterministic generator used for unit tests.

    - Records how many times it was called
    - Returns a stable answer based on input query
    - Returns fixed provenance/quality for storage decisions
    """
    call_count: int = 0
    last_query: Optional[str] = None

    def generate(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> Tuple[str, Provenance, QualitySignals]:
        self.call_count += 1
        self.last_query = mq.raw_query

        ans = f"GEN:{mq.raw_query} | ctx_len={len((mq.context or {}).get('dataset_context', '') or '')}"
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
) -> MemoryItem:
    """
    Create a MemoryItem consistent with the manager's keying scheme.
    This lets us inject controlled items directly into stores for scope-order testing.
    """
    q_can = canonicalize(mq.raw_query)
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
        meta={"unit_test": True},
    )


def _build_manager(tmp_path) -> Tuple[MemoryManager, RamStoreLRU, DiskStoreSQLite]:
    """
    Build a fresh manager + stores for each test.
    """
    ram = RamStoreLRU(max_mb=8)
    disk = DiskStoreSQLite(str(tmp_path / "mem.sqlite"))

    mm_cfg = MemoryManagerConfig(
        promote_disk_hits_to_ram=True,
        return_memory_directly=True,
    )
    mgr = MemoryManager(ram=ram, disk=disk, cfg=mm_cfg)
    return mgr, ram, disk


# -------------------------
# Tests
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

    # 1) First call should invoke generator
    ans1, meta1 = mgr.answer(mq, gen)
    assert gen.call_count == 1
    assert meta1.get("generated") is True
    assert meta1.get("used_memory") is False
    assert ans1.startswith("GEN:")

    # 2) Second call should be a RAM hit (since we write-through RAM+DISK)
    ans2, meta2 = mgr.answer(mq, gen)
    assert gen.call_count == 1  # no additional calls
    assert meta2.get("used_memory") is True
    assert ans2 == ans1

    # Sanity: RAM/DISK stats should reflect activity
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

    # Populate (miss -> generate -> store)
    ans1, meta1 = mgr.answer(mq, gen)
    assert gen.call_count == 1

    # Clear RAM to force DISK path
    ram.clear()

    # Retrieve: should hit DISK (and promote)
    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.source_tier.value == "disk"
    assert hit.item.answer_text == ans1

    # After promotion, RAM should now have the item; retrieve again should hit RAM
    hit2 = mgr.retrieve(mq)
    assert hit2 is not None
    assert hit2.source_tier.value == "ram"
    assert hit2.item.answer_text == ans1


def test_manager_respects_scope_order_session_over_user(tmp_path):
    """
    Validates SESSION -> USER ordering.

    We inject two items:
      - session-scoped item with answer "SESSION_ANSWER"
      - user-scoped item with answer "USER_ANSWER"
    Retrieval should return the session-scoped answer first.
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

    # Build deterministic namespaces exactly like namespace.py does
    ns_session = "session:s1"
    ns_user = "user:u1"

    # Create and insert items directly into DISK (source of truth)
    session_item = _mk_item_for_scope(
        scope=Scope.SESSION, namespace=ns_session, mq=mq, answer_text="SESSION_ANSWER"
    )
    user_item = _mk_item_for_scope(
        scope=Scope.USER, namespace=ns_user, mq=mq, answer_text="USER_ANSWER"
    )

    disk.put(ns_session, session_item.key, session_item)
    disk.put(ns_user, user_item.key, user_item)

    # Ensure RAM is empty so we test pure retrieval ordering
    ram.clear()

    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.item.answer_text == "SESSION_ANSWER"
    assert hit.item.scope == Scope.SESSION


def test_manager_rejects_expired_items(tmp_path):
    """
    Validates TTL/expiration gating at retrieval time.

    If an item is expired, MemoryManager.retrieve should return None even if it exists in stores.
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

    # Insert expired item into both tiers
    disk.put(ns_user, expired_item.key, expired_item)
    ram.put(ns_user, expired_item.key, expired_item)

    hit = mgr.retrieve(mq)
    assert hit is None  # must not reuse expired memory


def test_manager_miss_after_expired_item_calls_generator(tmp_path):
    """
    If only expired memory exists, manager.answer should generate a fresh response
    and overwrite stored memory.
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

    # Should generate because expired item must be rejected
    assert gen.call_count == 1
    assert meta.get("generated") is True
    assert ans.startswith("GEN:")

    # After generation, a new retrieval should hit memory with fresh answer
    hit = mgr.retrieve(mq)
    assert hit is not None
    assert hit.item.answer_text == ans
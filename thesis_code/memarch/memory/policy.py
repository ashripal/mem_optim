# memarch/memory/policy.py
"""
Deterministic retrieval + budget policies for memarch.

Phase 1 focus:
- Exact-match routing remains primary
- Semantic retrieval is optional and context-assistive by default
- Personalization-first scope ordering
- Safety gating: freshness (TTL) + context match + version scoping

This module contains decision rules, not storage or model calls.
Keep it pure and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from memarch.memory.schema import MemoryItem, MemoryQuery, Scope


@dataclass(frozen=True)
class BudgetPolicy:
    """
    Guardrails to keep memory operations bounded on constrained devices.

    Phase 1:
      - exact-match lookups are cheap
      - semantic retrieval may require bounded candidate scans
      - disk can still be slow, so reads remain capped
    """
    max_ram_reads: int = 64
    max_disk_reads: int = 16
    allow_semantic: bool = False

    def __post_init__(self) -> None:
        if self.max_ram_reads < 0 or self.max_disk_reads < 0:
            raise ValueError("max_ram_reads/max_disk_reads must be >= 0")


@dataclass(frozen=True)
class RetrievalPolicy:
    """
    Rules for deciding if a retrieved MemoryItem is safe to use.

    These checks should be cheap and deterministic.
    """
    scope_order: List[Scope]

    # Exact-match gating
    require_context_match: bool = True
    require_prompt_version_match: bool = True
    require_model_id_match: bool = True
    enforce_ttl: bool = True

    # Semantic retrieval policy
    semantic_enabled: bool = False

    # Phase 1 default:
    # - allow semantic retrieval to assist generation if score >= threshold_context
    # - bypass is typically disabled initially by setting threshold_bypass > 1.0
    semantic_threshold_context: float = 0.85
    semantic_threshold_bypass: float = 1.01

    # Limit number of scored candidates after filtering
    max_semantic_candidates: int = 5

    # Safety gating for semantic reuse
    require_same_task_for_semantic: bool = True
    require_same_model_for_semantic: bool = True
    require_same_prompt_version_for_semantic: bool = True
    require_same_context_for_bypass: bool = True

    def __post_init__(self) -> None:
        if not self.scope_order:
            raise ValueError("scope_order must be non-empty")
        if not (0.0 <= float(self.semantic_threshold_context) <= 1.0):
            raise ValueError("semantic_threshold_context must be in [0,1]")
        if not (0.0 <= float(self.semantic_threshold_bypass)):
            raise ValueError("semantic_threshold_bypass must be >= 0")
        if self.max_semantic_candidates < 1:
            raise ValueError("max_semantic_candidates must be >= 1")
        if self.semantic_threshold_bypass < self.semantic_threshold_context:
            raise ValueError(
                "semantic_threshold_bypass must be >= semantic_threshold_context"
            )


def default_retrieval_policy() -> RetrievalPolicy:
    return RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL]
    )


def budget_from_query(mq: MemoryQuery) -> BudgetPolicy:
    """
    Construct BudgetPolicy from MemoryQuery.

    This keeps budgets configurable from config/runner without coupling policy to config.py.
    """
    return BudgetPolicy(
        max_ram_reads=mq.max_ram_reads,
        max_disk_reads=mq.max_disk_reads,
        allow_semantic=mq.allow_semantic,
    )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def is_fresh(item: MemoryItem, now_utc: Optional[datetime] = None) -> bool:
    if item.expires_at_utc is None:
        return True
    now = now_utc or _now_utc()
    return now < item.expires_at_utc


def context_matches(mq: MemoryQuery, item: MemoryItem, *, query_context_signature: Optional[str] = None) -> bool:
    """
    Exact/context safety check.

    In practice, exact retrieval usually already includes context_signature in the key.
    This function remains as defense-in-depth and for future semantic workflows.
    """
    if query_context_signature is None:
        return True
    return item.context_signature == query_context_signature


def version_matches(
    mq: MemoryQuery,
    item: MemoryItem,
    *,
    require_model: bool,
    require_prompt: bool,
) -> bool:
    if require_model and item.provenance.model_id != mq.model_id:
        return False
    if require_prompt and item.provenance.prompt_version != mq.prompt_version:
        return False
    return True


def task_matches(mq: MemoryQuery, item: MemoryItem) -> bool:
    """
    Task/domain gating for semantic reuse.

    We first look for an explicit task in item.meta for forward compatibility.
    If absent, we fall back to True so older stored items do not break exact retrieval.
    """
    item_task = item.meta.get("task")
    if item_task is None:
        return True
    return item_task == mq.task


def same_document(
    mq: MemoryQuery,
    item: MemoryItem,
) -> bool:
    """
    Stronger semantic safety check when document signatures are available.
    Returns True if:
    - both doc signatures exist and match, or
    - document signature is unavailable on one or both sides
    """
    mq_doc = mq.context.get("doc_signature")
    item_doc = item.meta.get("doc_signature")

    if mq_doc is None or item_doc is None:
        return True
    return mq_doc == item_doc


def semantic_candidate_allowed(
    mq: MemoryQuery,
    item: MemoryItem,
    *,
    policy: RetrievalPolicy,
    now_utc: Optional[datetime] = None,
    query_context_signature: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether a MemoryItem is eligible to participate in semantic retrieval.

    This is candidate filtering before similarity thresholds are applied.
    """
    dbg: Dict[str, Any] = {"reason": "accepted"}

    if policy.enforce_ttl and not is_fresh(item, now_utc=now_utc):
        dbg["reason"] = "expired"
        return False, dbg

    if policy.require_same_task_for_semantic and not task_matches(mq, item):
        dbg["reason"] = "task_mismatch"
        dbg["item_task"] = item.meta.get("task")
        dbg["query_task"] = mq.task
        return False, dbg

    if policy.require_same_model_for_semantic and item.provenance.model_id != mq.model_id:
        dbg["reason"] = "model_mismatch"
        dbg["item_model_id"] = item.provenance.model_id
        dbg["query_model_id"] = mq.model_id
        return False, dbg

    if (
        policy.require_same_prompt_version_for_semantic
        and item.provenance.prompt_version != mq.prompt_version
    ):
        dbg["reason"] = "prompt_version_mismatch"
        dbg["item_prompt_version"] = item.provenance.prompt_version
        dbg["query_prompt_version"] = mq.prompt_version
        return False, dbg

    if item.query_embedding is None:
        dbg["reason"] = "missing_embedding"
        return False, dbg

    if not same_document(mq, item):
        dbg["reason"] = "document_mismatch"
        dbg["item_doc_signature"] = item.meta.get("doc_signature")
        dbg["query_doc_signature"] = mq.context.get("doc_signature")
        return False, dbg

    dbg["query_context_signature"] = query_context_signature
    return True, dbg


def semantic_decision(
    *,
    score: float,
    item: MemoryItem,
    policy: RetrievalPolicy,
    query_context_signature: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Decide what to do with a semantic candidate after similarity is computed.

    Returns:
      ("ignore" | "context" | "bypass", debug_info)

    Phase 1 recommendation:
    - context assistance can be enabled
    - bypass is usually disabled by setting threshold_bypass > 1.0
    """
    dbg: Dict[str, Any] = {
        "score": score,
        "semantic_threshold_context": policy.semantic_threshold_context,
        "semantic_threshold_bypass": policy.semantic_threshold_bypass,
    }

    if score < policy.semantic_threshold_context:
        dbg["reason"] = "below_context_threshold"
        return "ignore", dbg

    same_context = True
    if policy.require_same_context_for_bypass and query_context_signature is not None:
        same_context = item.context_signature == query_context_signature
        dbg["same_context_for_bypass"] = same_context

    if score >= policy.semantic_threshold_bypass and same_context:
        dbg["reason"] = "semantic_bypass"
        return "bypass", dbg

    dbg["reason"] = "semantic_context"
    return "context", dbg


def accept_item(
    mq: MemoryQuery,
    item: MemoryItem,
    *,
    policy: RetrievalPolicy,
    now_utc: Optional[datetime] = None,
    query_context_signature: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether to accept a retrieved item for exact-match reuse.

    Returns:
      (accepted, debug_info)
    """
    dbg: Dict[str, Any] = {"reason": "accepted"}

    if policy.enforce_ttl:
        if not is_fresh(item, now_utc=now_utc):
            dbg["reason"] = "expired"
            return False, dbg

    if not version_matches(
        mq,
        item,
        require_model=policy.require_model_id_match,
        require_prompt=policy.require_prompt_version_match,
    ):
        dbg["reason"] = "version_mismatch"
        dbg["item_model_id"] = item.provenance.model_id
        dbg["item_prompt_version"] = item.provenance.prompt_version
        return False, dbg

    if policy.require_context_match:
        if not context_matches(mq, item, query_context_signature=query_context_signature):
            dbg["reason"] = "context_mismatch"
            dbg["item_context_signature"] = item.context_signature
            dbg["query_context_signature"] = query_context_signature
            return False, dbg

    return True, dbg


def score_exact_hit() -> float:
    """Exact-match hits get a perfect confidence score."""
    return 1.0


def make_hit_debug(
    *,
    scope: Scope,
    namespace: str,
    source: str,
    accepted_reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "scope": scope.value,
        "namespace": namespace,
        "source": source,
        "reason": accepted_reason,
    }
    if extra:
        d.update(extra)
    return d
# memarch/memory/policy.py
"""
Deterministic retrieval + budget policies for memarch.

Phase 1 focus:
- Exact-match routing only (semantic disabled by default)
- Personalization-first scope ordering
- Safety gating: freshness (TTL) + context match + version scoping

This module contains *decision rules*, not storage or model calls.
Keep it pure and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from memarch.memory.schema import MemoryItem, MemoryQuery, MemoryHit, MatchType, Scope


@dataclass(frozen=True)
class BudgetPolicy:
    """
    Guardrails to keep memory operations bounded on constrained devices.

    Phase 1:
      - exact-match lookups are cheap, but disk can still be slow
      - these budgets prevent pathological loops or huge scans
    """
    max_ram_reads: int = 64
    max_disk_reads: int = 16
    allow_semantic: bool = False  # keep off for Phase 1

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
    require_context_match: bool = True
    require_prompt_version_match: bool = True
    require_model_id_match: bool = True
    enforce_ttl: bool = True

    def __post_init__(self) -> None:
        if not self.scope_order:
            raise ValueError("scope_order must be non-empty")


def default_retrieval_policy() -> RetrievalPolicy:
    return RetrievalPolicy(scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL])


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


def context_matches(mq: MemoryQuery, item: MemoryItem) -> bool:
    """
    Phase 1 context matching:
    - MemoryItem.context_signature must match mq.context_signature used for keying

    NOTE:
    In Phase 1, we usually enforce context matching through the key itself
    (since key includes context signature). This function exists for defense-in-depth
    and for cases where you may relax keying later.
    """
    # We do not compute signatures here to keep policy pure.
    # manager.py should pass/compare signatures if needed.
    return True


def version_matches(mq: MemoryQuery, item: MemoryItem, *, require_model: bool, require_prompt: bool) -> bool:
    if require_model and item.provenance.model_id != mq.model_id:
        return False
    if require_prompt and item.provenance.prompt_version != mq.prompt_version:
        return False
    return True


def accept_item(
    mq: MemoryQuery,
    item: MemoryItem,
    *,
    policy: RetrievalPolicy,
    now_utc: Optional[datetime] = None,
    query_context_signature: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether to accept a retrieved item for reuse.

    Returns:
      (accepted, debug_info)
    """
    dbg: Dict[str, Any] = {"reason": "accepted"}

    # TTL / freshness
    if policy.enforce_ttl:
        if not is_fresh(item, now_utc=now_utc):
            dbg["reason"] = "expired"
            return False, dbg

    # Version gating
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

    # Context gating (defense-in-depth)
    if policy.require_context_match and query_context_signature is not None:
        if item.context_signature != query_context_signature:
            dbg["reason"] = "context_mismatch"
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
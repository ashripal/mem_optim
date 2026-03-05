# memarch/memory/admission.py
"""
Admission policy: decides what gets stored after generation.

Phase 1 goals:
- deterministic (no LLM decisions)
- lightweight + unit-testable
- scope-aware TTL + storage enablement

This module is called by MemoryManager.store().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from memarch.memory.schema import MemoryQuery, QualitySignals, Scope


@dataclass(frozen=True)
class AdmissionPolicy:
    """
    Controls whether we store generated answers and where.

    Storage enablement:
      - allow_session/user/cohort/global decide which scopes can be written.

    Content gating:
      - min_answer_chars avoids storing trivial outputs
      - max_answer_chars avoids huge responses filling disk

    Quality gating (optional):
      - if require_success_if_provided=True, then QualitySignals.success must be True
        when it is provided (default behavior).
    """
    allow_session: bool = True
    allow_user: bool = True
    allow_cohort: bool = False
    allow_global: bool = False

    min_answer_chars: int = 20
    max_answer_chars: int = 50_000

    require_success_if_provided: bool = True

    # TTL per scope (seconds). None means "no expiry" (not recommended for session).
    ttl_session_seconds: int = 60 * 60 * 2        # 2 hours
    ttl_user_seconds: int = 60 * 60 * 24 * 14     # 14 days
    ttl_cohort_seconds: int = 60 * 60 * 24 * 30   # 30 days
    ttl_global_seconds: int = 60 * 60 * 24 * 90   # 90 days


def default_admission_policy() -> AdmissionPolicy:
    # Phase 1: write personalization memory (session/user), no cohort/global by default.
    return AdmissionPolicy()


def decide_store_scopes(mq: MemoryQuery, policy: AdmissionPolicy) -> List[Scope]:
    """
    Decide which scopes are *eligible* to store into for this query.

    Note: this does not apply content/quality gating; that's done in should_store().
    """
    scopes: List[Scope] = []

    if policy.allow_session and mq.session_id:
        scopes.append(Scope.SESSION)

    if policy.allow_user and mq.user_id:
        scopes.append(Scope.USER)

    if policy.allow_cohort and mq.cohort_id:
        scopes.append(Scope.COHORT)

    if policy.allow_global:
        scopes.append(Scope.GLOBAL)

    return scopes


def choose_ttl_seconds(scope: Scope, policy: AdmissionPolicy) -> int:
    if scope == Scope.SESSION:
        return int(policy.ttl_session_seconds)
    if scope == Scope.USER:
        return int(policy.ttl_user_seconds)
    if scope == Scope.COHORT:
        return int(policy.ttl_cohort_seconds)
    if scope == Scope.GLOBAL:
        return int(policy.ttl_global_seconds)
    # Defensive fallback
    return int(policy.ttl_user_seconds)


def should_store(
    mq: MemoryQuery,
    answer_text: str,
    quality: QualitySignals,
    *,
    scope: Scope,
    policy: AdmissionPolicy,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Content/quality gating: returns (ok_to_store, debug_dict).
    """
    ans = answer_text or ""
    n = len(ans)

    if n < policy.min_answer_chars:
        return False, {"reason": "too_short", "len": n, "min": policy.min_answer_chars}

    if n > policy.max_answer_chars:
        return False, {"reason": "too_large", "len": n, "max": policy.max_answer_chars}

    if policy.require_success_if_provided and (quality is not None):
        # If success is explicitly False, reject.
        if hasattr(quality, "success") and (quality.success is False):
            return False, {"reason": "quality_failed"}

    # Scope enablement checks (defensive; scopes are usually pre-filtered)
    if scope == Scope.SESSION and not (policy.allow_session and mq.session_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.USER and not (policy.allow_user and mq.user_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.COHORT and not (policy.allow_cohort and mq.cohort_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.GLOBAL and not policy.allow_global:
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}

    return True, {"reason": "accepted"}


# Convenience wrapper used by some callers
def should_store_default(
    mq: MemoryQuery,
    answer_text: str,
    quality: QualitySignals,
    *,
    scope: Scope,
) -> Tuple[bool, Dict[str, Any]]:
    return should_store(mq, answer_text, quality, scope=scope, policy=default_admission_policy())
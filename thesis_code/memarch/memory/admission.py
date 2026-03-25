# memarch/memory/admission.py
"""
Admission policy: decides what gets stored after generation.

Phase 1 goals:
- deterministic (no LLM decisions)
- lightweight + unit-testable
- scope-aware TTL + storage enablement

Evidence-guided extension:
- reject clearly low-value outputs before they enter memory
- keep storage quality high for future exact/semantic reuse
- use only cheap string heuristics (latency-friendly)

This module is called by MemoryManager.store().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from memarch.memory.schema import MemoryQuery, QualitySignals, Scope


_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# Simple prompt-scaffolding markers that often indicate low-quality stored outputs.
_SCAFFOLD_MARKERS = (
    "question:",
    "answer:",
    "context:",
    "instruction:",
    "response:",
    "user:",
    "assistant:",
)

# TREC coarse labels. Storing malformed free-form responses hurts reuse quality.
_TREC_LABELS = {"ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"}


@dataclass(frozen=True)
class AdmissionPolicy:
    """
    Controls whether we store generated answers and where.

    Storage enablement:
      - allow_session/user/cohort/global decide which scopes can be written.

    Content gating:
      - min_answer_chars avoids storing trivial outputs
      - max_answer_chars avoids huge responses filling disk
      - reject_question_echo avoids storing answers that mostly restate the query
      - reject_prompt_scaffolding avoids storing prompt-template leakage
      - validate_task_format enables cheap task-specific format checks
      - allow_short_task_labels permits valid short classification outputs

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

    # Cheap storage-quality gates
    reject_question_echo: bool = True
    reject_prompt_scaffolding: bool = True
    validate_task_format: bool = True
    allow_short_task_labels: bool = True

    # Heuristics
    max_question_echo_token_overlap: float = 0.80
    max_scaffold_marker_count: int = 2

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


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip()).lower()


def _token_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(_normalize_text(text)))


def _task_name(mq: MemoryQuery) -> str:
    return (mq.task or "").strip().lower()


def _question_type(mq: MemoryQuery) -> str:
    if getattr(mq, "question_type", None):
        return str(mq.question_type).strip().lower()
    return str((mq.context or {}).get("question_type") or "").strip().lower()


def _looks_like_short_task_label(mq: MemoryQuery, answer_text: str) -> bool:
    """
    Allow valid short outputs for label-style tasks.

    Current supported case:
    - TREC coarse class labels
    """
    task = _task_name(mq)
    qtype = _question_type(mq)
    ans = (answer_text or "").strip()

    if not ans:
        return False

    if task == "trec" or qtype == "classification":
        return ans.upper() in _TREC_LABELS

    return False


def _looks_like_question_echo(answer_text: str, raw_query: str, *, overlap_threshold: float) -> Tuple[bool, Dict[str, Any]]:
    """
    Cheap heuristic to reject answers that are mostly restatements of the question.

    Strategy:
    - exact normalized equality -> reject
    - answer contains the full normalized query -> reject
    - otherwise compute token overlap against query tokens
    """
    ans_norm = _normalize_text(answer_text)
    qry_norm = _normalize_text(raw_query)

    if not ans_norm or not qry_norm:
        return False, {}

    if ans_norm == qry_norm:
        return True, {"mode": "exact_match"}

    if qry_norm and qry_norm in ans_norm:
        return True, {"mode": "query_contained_in_answer"}

    q_tokens = _token_set(qry_norm)
    a_tokens = _token_set(ans_norm)

    if not q_tokens or not a_tokens:
        return False, {}

    overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))
    if overlap >= overlap_threshold:
        return True, {"mode": "token_overlap", "overlap": round(overlap, 4)}

    return False, {"mode": "token_overlap", "overlap": round(overlap, 4)}


def _count_scaffold_markers(answer_text: str) -> Dict[str, int]:
    ans_norm = _normalize_text(answer_text)
    counts: Dict[str, int] = {}
    for marker in _SCAFFOLD_MARKERS:
        c = ans_norm.count(marker)
        if c > 0:
            counts[marker] = c
    return counts


def _fails_scaffolding_check(answer_text: str, *, max_marker_count: int) -> Tuple[bool, Dict[str, Any]]:
    counts = _count_scaffold_markers(answer_text)
    total = sum(counts.values())
    if total > max_marker_count:
        return True, {"marker_counts": counts, "total_markers": total}
    return False, {"marker_counts": counts, "total_markers": total}


def _validate_task_specific_format(mq: MemoryQuery, answer_text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Cheap task-specific validation.

    Current strict rule:
    - TREC should be a single coarse label, not a sentence/explanation

    Returns:
      (is_valid, debug_info)
    """
    task = _task_name(mq)
    qtype = _question_type(mq)
    ans = (answer_text or "").strip()

    if task == "trec" or qtype == "classification":
        label = ans.upper().strip()
        is_valid = label in _TREC_LABELS
        return is_valid, {
            "task": task or "classification",
            "question_type": qtype or None,
            "expected_labels": sorted(_TREC_LABELS),
            "observed": ans,
        }

    return True, {
        "task": task or "default",
        "question_type": qtype or None,
        "validation": "not_applicable",
    }


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

    short_label_ok = bool(policy.allow_short_task_labels and _looks_like_short_task_label(mq, ans))

    if n < policy.min_answer_chars and not short_label_ok:
        return False, {"reason": "too_short", "len": n, "min": policy.min_answer_chars}

    if n > policy.max_answer_chars:
        return False, {"reason": "too_large", "len": n, "max": policy.max_answer_chars}

    if policy.require_success_if_provided and (quality is not None):
        # If success is explicitly False, reject.
        if hasattr(quality, "success") and (quality.success is False):
            return False, {"reason": "quality_failed"}

    if policy.reject_question_echo:
        is_echo, echo_debug = _looks_like_question_echo(
            ans,
            mq.raw_query,
            overlap_threshold=policy.max_question_echo_token_overlap,
        )
        if is_echo:
            return False, {"reason": "question_echo", **echo_debug}

    if policy.reject_prompt_scaffolding:
        has_too_much_scaffold, scaffold_debug = _fails_scaffolding_check(
            ans,
            max_marker_count=policy.max_scaffold_marker_count,
        )
        if has_too_much_scaffold:
            return False, {"reason": "prompt_scaffolding", **scaffold_debug}

    if policy.validate_task_format:
        is_valid_format, format_debug = _validate_task_specific_format(mq, ans)
        if not is_valid_format:
            return False, {"reason": "invalid_task_format", **format_debug}

    # Scope enablement checks (defensive; scopes are usually pre-filtered)
    if scope == Scope.SESSION and not (policy.allow_session and mq.session_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.USER and not (policy.allow_user and mq.user_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.COHORT and not (policy.allow_cohort and mq.cohort_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.GLOBAL and not policy.allow_global:
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}

    accepted_debug: Dict[str, Any] = {"reason": "accepted"}
    if short_label_ok:
        accepted_debug["accepted_via"] = "short_task_label"
    return True, accepted_debug


# Convenience wrapper used by some callers
def should_store_default(
    mq: MemoryQuery,
    answer_text: str,
    quality: QualitySignals,
    *,
    scope: Scope,
) -> Tuple[bool, Dict[str, Any]]:
    return should_store(mq, answer_text, quality, scope=scope, policy=default_admission_policy())
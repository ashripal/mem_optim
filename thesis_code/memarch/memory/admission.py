from __future__ import annotations

"""
memarch/memory/admission.py

Admission policy for deciding whether a newly generated answer should be stored.

Role in the overall verified paraphrase reuse strategy:
- This module decides whether an answer is worth storing at all.
- It does NOT decide whether a stored answer is safe for semantic direct reuse.
- Semantic direct reuse safety belongs in:
    memarch/memory/policy.py

Why this separation matters:
- admission = "should we keep this memory item?"
- retrieval/policy = "if retrieved later, is it safe to reuse directly?"

Current goals:
- reject obviously bad outputs
- reject prompt scaffolding and question echos
- allow narrow short-label outputs for classification tasks like TREC
- keep logic deterministic and cheap
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from memarch.memory.schema import MemoryQuery, QualitySignals, Scope


# =============================================================================
# Small constants / regexes
# =============================================================================

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

# Common prompt / chat scaffolding markers that should not dominate stored answers.
_SCAFFOLD_MARKERS = (
    "question:",
    "answer:",
    "context:",
    "instruction:",
    "response:",
    "user:",
    "assistant:",
)

# Known TREC coarse labels.
_TREC_LABELS = {"ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"}

# Placeholder outputs we do not want to persist into memory.
_PLACEHOLDER_ANSWERS = {
    "(no answer generated.)",
    "no answer generated",
    "n/a",
    "na",
    "none",
    "unknown",
}

# Bad ultra-short QA answers that are almost never useful memories.
_BAD_QA_ANSWERS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "by",
    "with",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "being",
    "it",
    "its",
    "their",
    "his",
    "her",
    "they",
    "them",
    "he",
    "she",
    "this",
    "that",
    "these",
    "those",
}

# Answers that end with a dangling one-letter token are often truncated spans.
_DANGLING_INITIAL_RE = re.compile(r".+\b[A-Za-z]\b$")

# Looks like an incomplete title/entity prefix.
_TRAILING_TITLE_PREFIX_RE = re.compile(
    r".+\b(mr|mrs|ms|dr|prof|sir|st|mt)\.?$",
    re.IGNORECASE,
)

# Generic one-token QA answers that are usually too weak to store.
_BAD_SINGLE_TOKEN_QA_ANSWERS = {
    "released",
    "written",
    "founded",
    "created",
    "built",
    "called",
    "known",
    "named",
    "part",
    "album",
    "song",
    "tour",
    "group",
    "band",
    "school",
    "college",
    "church",
    "team",
    "city",
    "country",
    "company",
    "president",
    "king",
    "queen",
    "father",
    "mother",
    "son",
    "daughter",
}

# Useful years / numbers patterns for short valid QA answers.
_YEAR_RE = re.compile(r"^\d{4}$")
_NUMERIC_SHORT_RE = re.compile(r"^\d+([.,]\d+)?(%| percent| million| billion| thousand)?$", re.IGNORECASE)
_MONTH_DAY_YEAR_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}$",
    re.IGNORECASE,
)


# =============================================================================
# Policy dataclass
# =============================================================================

@dataclass(frozen=True)
class AdmissionPolicy:
    """
    Rules for whether a freshly generated answer should be stored.

    Notes:
    - These are store-time quality gates only.
    - This module intentionally does not know about semantic bypass thresholds.
    """
    allow_session: bool = True
    allow_user: bool = True
    allow_cohort: bool = False
    allow_global: bool = False

    # Generic size bounds
    min_answer_chars: int = 3
    max_answer_chars: int = 50_000

    # If quality.success is provided and False, reject storage.
    require_success_if_provided: bool = True

    # Cheap content sanity checks
    reject_question_echo: bool = True
    reject_prompt_scaffolding: bool = True
    reject_placeholder_answers: bool = True
    validate_task_format: bool = True

    # Some tasks legitimately produce very short labels.
    allow_short_task_labels: bool = True

    # Echo/scaffolding thresholds
    max_question_echo_token_overlap: float = 0.80
    max_scaffold_marker_count: int = 2

    # Skip the more expensive echo check for very long answers.
    max_echo_check_chars: int = 2000

    # QA-specific sanity checks
    require_answer_span_for_qa_if_provided: bool = True
    reject_obviously_bad_qa_answers: bool = True
    reject_dangling_initial_qa_answers: bool = True

    # New stricter QA quality gates
    reject_single_token_generic_qa_answers: bool = True
    reject_short_alpha_qa_answers_without_signal: bool = True
    min_qa_alpha_token_count: int = 2
    min_qa_answer_chars: int = 4

    # TTLs by scope
    ttl_session_seconds: int = 60 * 60 * 2
    ttl_user_seconds: int = 60 * 60 * 24 * 14
    ttl_cohort_seconds: int = 60 * 60 * 24 * 30
    ttl_global_seconds: int = 60 * 60 * 24 * 90


def default_admission_policy() -> AdmissionPolicy:
    """
    Default admission policy used by the manager unless overridden.
    """
    return AdmissionPolicy()


# =============================================================================
# Scope / TTL helpers
# =============================================================================

def decide_store_scopes(mq: MemoryQuery, policy: AdmissionPolicy) -> List[Scope]:
    """
    Decide which scopes are eligible for storage for this query.
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
    """
    Choose TTL duration by storage scope.
    """
    if scope == Scope.SESSION:
        return int(policy.ttl_session_seconds)
    if scope == Scope.USER:
        return int(policy.ttl_user_seconds)
    if scope == Scope.COHORT:
        return int(policy.ttl_cohort_seconds)
    if scope == Scope.GLOBAL:
        return int(policy.ttl_global_seconds)
    return int(policy.ttl_user_seconds)


# =============================================================================
# Text helpers
# =============================================================================

def _normalize_text(text: str) -> str:
    """
    Lowercase + collapse whitespace.
    """
    return _WS_RE.sub(" ", (text or "").strip()).lower()


def _token_set(text: str) -> set[str]:
    """
    Token set used for cheap overlap checks.
    """
    return set(_WORD_RE.findall(_normalize_text(text)))


def _token_list(text: str) -> List[str]:
    return list(_WORD_RE.findall(_normalize_text(text)))


def _task_name(mq: MemoryQuery) -> str:
    return (mq.task or "").strip().lower()


def _question_type(mq: MemoryQuery) -> str:
    if getattr(mq, "question_type", None):
        return str(mq.question_type).strip().lower()
    return str((mq.context or {}).get("question_type") or "").strip().lower()


def _is_qa_like(mq: MemoryQuery) -> bool:
    task = _task_name(mq)
    qtype = _question_type(mq)
    return task in {"squad", "qa", "extractive_qa"} or qtype == "qa"


def _is_short_numeric_or_date_answer(ans_norm: str) -> bool:
    if not ans_norm:
        return False
    return bool(
        _YEAR_RE.fullmatch(ans_norm)
        or _NUMERIC_SHORT_RE.fullmatch(ans_norm)
        or _MONTH_DAY_YEAR_RE.fullmatch(ans_norm)
    )


def _has_uppercase_signal(ans: str) -> bool:
    # Helps keep proper nouns/entities like "Beyoncé" or "Georges Bizet".
    return any(ch.isupper() for ch in (ans or ""))


# =============================================================================
# Task-format helpers
# =============================================================================

def _looks_like_short_task_label(mq: MemoryQuery, answer_text: str) -> bool:
    """
    Return True when a very short answer is acceptable because the task expects
    a short class label.

    Currently supported:
    - TREC / classification labels
    """
    task = _task_name(mq)
    qtype = _question_type(mq)
    ans = (answer_text or "").strip()

    if not ans:
        return False

    if task == "trec" or qtype == "classification":
        return ans.upper() in _TREC_LABELS

    return False


def _validate_task_specific_format(mq: MemoryQuery, answer_text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate task-specific answer formatting where useful.

    Current behavior:
    - TREC / classification: must be one of the known coarse labels
    - other tasks: no strict validation here

    Important:
    We do NOT enforce extractive QA answer support here, because that belongs
    to retrieval-time verification, not store-time admission.
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


# =============================================================================
# Content sanity checks
# =============================================================================

def _looks_like_placeholder_answer(answer_text: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect obvious placeholder / failed outputs that should not be stored.
    """
    ans_norm = _normalize_text(answer_text)
    if ans_norm in _PLACEHOLDER_ANSWERS:
        return True, {"normalized_answer": ans_norm}
    return False, {"normalized_answer": ans_norm}


def _looks_like_question_echo(
    answer_text: str,
    raw_query: str,
    *,
    overlap_threshold: float,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect whether the answer is mostly just an echo of the question.

    Checks:
    - exact normalized match
    - query substring contained in answer
    - high token overlap ratio
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
    """
    Count prompt/chat scaffolding markers in an answer.
    """
    ans_norm = _normalize_text(answer_text)
    counts: Dict[str, int] = {}
    for marker in _SCAFFOLD_MARKERS:
        c = ans_norm.count(marker)
        if c > 0:
            counts[marker] = c
    return counts


def _fails_scaffolding_check(answer_text: str, *, max_marker_count: int) -> Tuple[bool, Dict[str, Any]]:
    """
    Reject if the answer contains too much prompt/chat scaffolding.
    """
    counts = _count_scaffold_markers(answer_text)
    total = sum(counts.values())
    if total > max_marker_count:
        return True, {"marker_counts": counts, "total_markers": total}
    return False, {"marker_counts": counts, "total_markers": total}


def _fails_qa_quality_checks(
    mq: MemoryQuery,
    answer_text: str,
    quality: QualitySignals,
    *,
    policy: AdmissionPolicy,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Cheap QA-specific store-time checks to avoid caching clearly bad spans.
    """
    if not _is_qa_like(mq):
        return False, {"validation": "not_applicable"}

    ans = (answer_text or "").strip()
    ans_norm = _normalize_text(ans)
    ans_tokens = _token_list(ans_norm)
    alpha_tokens = [t for t in ans_tokens if any(ch.isalpha() for ch in t)]
    token_count = len(ans_tokens)
    alpha_token_count = len(alpha_tokens)

    if policy.require_answer_span_for_qa_if_provided and quality is not None:
        if hasattr(quality, "answer_span_found") and (quality.answer_span_found is False):
            return True, {"reason": "qa_answer_span_not_found"}

    if policy.reject_obviously_bad_qa_answers:
        if ans_norm in _BAD_QA_ANSWERS:
            return True, {"reason": "qa_bad_stopword_answer", "normalized_answer": ans_norm}

    if policy.reject_dangling_initial_qa_answers:
        if len(ans_tokens) >= 2 and _DANGLING_INITIAL_RE.fullmatch(ans.strip()):
            return True, {"reason": "qa_dangling_initial", "normalized_answer": ans_norm}
        if _TRAILING_TITLE_PREFIX_RE.fullmatch(ans.strip()):
            return True, {"reason": "qa_trailing_title_prefix", "normalized_answer": ans_norm}

    if policy.reject_single_token_generic_qa_answers and token_count == 1:
        if ans_norm in _BAD_SINGLE_TOKEN_QA_ANSWERS:
            return True, {"reason": "qa_bad_single_token_generic", "normalized_answer": ans_norm}

    if policy.reject_short_alpha_qa_answers_without_signal:
        # Keep compact numeric/date answers like "1842", "118 million", "September 5, 2006".
        if not _is_short_numeric_or_date_answer(ans_norm):
            # Reject very short alpha answers with too little entity signal.
            if len(ans) < policy.min_qa_answer_chars and alpha_token_count > 0:
                return True, {
                    "reason": "qa_too_short_alpha_answer",
                    "normalized_answer": ans_norm,
                    "answer_len": len(ans),
                }

            # Reject single-token alpha answers unless they look like a strong named entity.
            if token_count == 1 and alpha_token_count == 1:
                if not _has_uppercase_signal(ans):
                    return True, {
                        "reason": "qa_single_token_no_entity_signal",
                        "normalized_answer": ans_norm,
                    }

            # Reject multi-token QA answers with too little lexical substance.
            if alpha_token_count > 0 and alpha_token_count < policy.min_qa_alpha_token_count:
                # Allow strong short entities like "Beyoncé" via uppercase signal.
                if not _has_uppercase_signal(ans):
                    return True, {
                        "reason": "qa_insufficient_alpha_tokens",
                        "normalized_answer": ans_norm,
                        "alpha_token_count": alpha_token_count,
                    }

    return False, {
        "reason": "qa_checks_passed",
        "normalized_answer": ans_norm,
        "token_count": token_count,
        "alpha_token_count": alpha_token_count,
    }


# =============================================================================
# Main admission decision
# =============================================================================

def should_store(
    mq: MemoryQuery,
    answer_text: str,
    quality: QualitySignals,
    *,
    scope: Scope,
    policy: AdmissionPolicy,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether a generated answer should be stored in the requested scope.

    Decision order:
    1. basic size checks
    2. quality-success check
    3. placeholder-answer rejection
    4. question-echo rejection
    5. prompt-scaffolding rejection
    6. task-format validation
    7. QA-specific sanity checks
    8. scope eligibility check
    """
    ans = answer_text or ""
    n = len(ans)

    short_label_ok = bool(policy.allow_short_task_labels and _looks_like_short_task_label(mq, ans))

    # -------------------------------------------------------------------------
    # Basic length checks
    # -------------------------------------------------------------------------
    if n < policy.min_answer_chars and not short_label_ok:
        return False, {"reason": "too_short", "len": n, "min": policy.min_answer_chars}

    if n > policy.max_answer_chars:
        return False, {"reason": "too_large", "len": n, "max": policy.max_answer_chars}

    # -------------------------------------------------------------------------
    # Optional success signal check
    # -------------------------------------------------------------------------
    if policy.require_success_if_provided and (quality is not None):
        if hasattr(quality, "success") and (quality.success is False):
            return False, {"reason": "quality_failed"}

    # -------------------------------------------------------------------------
    # Reject placeholder / failed outputs
    # -------------------------------------------------------------------------
    if policy.reject_placeholder_answers:
        is_placeholder, placeholder_debug = _looks_like_placeholder_answer(ans)
        if is_placeholder:
            return False, {"reason": "placeholder_answer", **placeholder_debug}

    # -------------------------------------------------------------------------
    # Question-echo check
    # Skip for very long answers to avoid unnecessary overhead.
    # -------------------------------------------------------------------------
    if policy.reject_question_echo and n <= policy.max_echo_check_chars:
        is_echo, echo_debug = _looks_like_question_echo(
            ans,
            mq.raw_query,
            overlap_threshold=policy.max_question_echo_token_overlap,
        )
        if is_echo:
            return False, {"reason": "question_echo", **echo_debug}

    # -------------------------------------------------------------------------
    # Prompt scaffolding check
    # -------------------------------------------------------------------------
    if policy.reject_prompt_scaffolding:
        has_too_much_scaffold, scaffold_debug = _fails_scaffolding_check(
            ans,
            max_marker_count=policy.max_scaffold_marker_count,
        )
        if has_too_much_scaffold:
            return False, {"reason": "prompt_scaffolding", **scaffold_debug}

    # -------------------------------------------------------------------------
    # Task-specific format validation
    # -------------------------------------------------------------------------
    if policy.validate_task_format:
        is_valid_format, format_debug = _validate_task_specific_format(mq, ans)
        if not is_valid_format:
            return False, {"reason": "invalid_task_format", **format_debug}

    # -------------------------------------------------------------------------
    # QA-specific sanity checks
    # -------------------------------------------------------------------------
    qa_bad, qa_debug = _fails_qa_quality_checks(
        mq,
        ans,
        quality,
        policy=policy,
    )
    if qa_bad:
        return False, qa_debug

    # -------------------------------------------------------------------------
    # Scope eligibility
    # -------------------------------------------------------------------------
    if scope == Scope.SESSION and not (policy.allow_session and mq.session_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.USER and not (policy.allow_user and mq.user_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.COHORT and not (policy.allow_cohort and mq.cohort_id):
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}
    if scope == Scope.GLOBAL and not policy.allow_global:
        return False, {"reason": "scope_disabled_or_missing_id", "scope": scope.value}

    # -------------------------------------------------------------------------
    # Accepted
    # -------------------------------------------------------------------------
    accepted_debug: Dict[str, Any] = {
        "reason": "accepted",
        "task": _task_name(mq) or "default",
        "question_type": _question_type(mq) or None,
        "answer_len": n,
    }
    if short_label_ok:
        accepted_debug["accepted_via"] = "short_task_label"
    if _is_qa_like(mq):
        accepted_debug["qa_validated"] = True

    return True, accepted_debug


def should_store_default(
    mq: MemoryQuery,
    answer_text: str,
    quality: QualitySignals,
    *,
    scope: Scope,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience wrapper using the default admission policy.
    """
    return should_store(
        mq,
        answer_text,
        quality,
        scope=scope,
        policy=default_admission_policy(),
    )
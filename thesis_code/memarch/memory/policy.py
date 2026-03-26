# memarch/memory/policy.py
"""
Deterministic retrieval + budget policies for memarch.

Current behavior:
- Exact-match routing remains primary
- Lexical retrieval is optional and occurs after exact-match miss
- Semantic retrieval is optional and context-assistive by default
- Personalization-first scope ordering
- Safety gating:
    - freshness (TTL)
    - context match for exact hits
    - version scoping
    - task/version gating for lexical + semantic reuse

Evidence-guided extension:
- Prefer same-document lexical and semantic candidates before broader matches
- Keep semantic retrieval context-only (no semantic direct bypass)
- Allow lexical direct reuse only for explicitly safe tasks and strict thresholds
- Surface cheap document/source signals for manager-side ranking/selection

This module contains decision rules, not storage or model calls.
Keep it pure and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from memarch.memory.schema import MemoryItem, MemoryQuery, Scope


@dataclass(frozen=True)
class BudgetPolicy:
    """
    Guardrails to keep memory operations bounded on constrained devices.

    Current behavior:
      - exact-match lookups are cheap
      - lexical retrieval may require bounded candidate scans
      - semantic retrieval may require bounded candidate scans + scoring
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

    # -------------------------
    # Lexical retrieval policy
    # -------------------------
    lexical_enabled: bool = False
    lexical_threshold_context: float = 0.55
    lexical_threshold_bypass: float = 0.90
    lexical_top_k: int = 3

    # Safety gating for lexical reuse
    require_same_task_for_lexical: bool = True
    require_same_model_for_lexical: bool = True
    require_same_prompt_version_for_lexical: bool = True

    # Preference signals (not hard eligibility gates)
    prefer_same_source: bool = True
    safe_direct_reuse_tasks: List[str] = field(default_factory=lambda: ["trec"])

    # -------------------------
    # Semantic retrieval policy
    # -------------------------
    semantic_enabled: bool = False

    # Evidence-guided spec:
    # - semantic retrieval may assist generation if score >= threshold_context
    # - semantic direct bypass is disabled
    semantic_threshold_context: float = 0.85
    semantic_threshold_bypass: float = 1.01

    # Limit number of scored candidates after filtering
    max_semantic_candidates: int = 5

    # Safety gating for semantic reuse
    require_same_task_for_semantic: bool = True
    require_same_model_for_semantic: bool = True
    require_same_prompt_version_for_semantic: bool = True

    # Exact same-context requirement remains relevant only for exact hits.
    # Semantic bypass is disabled in this spec, so this stays for compatibility.
    require_same_context_for_bypass: bool = True

    # Evidence-guided preference:
    # do not require same document, but prefer it when available.
    prefer_same_document_for_semantic: bool = True

    # Semantic retrieval should remain assistive/context-only.
    allow_semantic_bypass: bool = False

    def __post_init__(self) -> None:
        if not self.scope_order:
            raise ValueError("scope_order must be non-empty")

        if not (0.0 <= float(self.lexical_threshold_context) <= 1.0):
            raise ValueError("lexical_threshold_context must be in [0,1]")
        if not (0.0 <= float(self.lexical_threshold_bypass) <= 1.0):
            raise ValueError("lexical_threshold_bypass must be in [0,1]")
        if self.lexical_top_k < 1:
            raise ValueError("lexical_top_k must be >= 1")
        if self.lexical_threshold_bypass < self.lexical_threshold_context:
            raise ValueError(
                "lexical_threshold_bypass must be >= lexical_threshold_context"
            )

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

        for task in self.safe_direct_reuse_tasks:
            if not str(task).strip():
                raise ValueError("safe_direct_reuse_tasks must not contain empty task names")


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


def context_matches(
    mq: MemoryQuery,
    item: MemoryItem,
    *,
    query_context_signature: Optional[str] = None,
) -> bool:
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
    Task/domain gating for approximate reuse.

    We first look for an explicit task in item.meta for forward compatibility.
    If absent, we fall back to True so older stored items do not break exact retrieval.
    """
    item_task = item.meta.get("task")
    if item_task is None:
        return True
    return item_task == mq.task


def _query_doc_signature(mq: MemoryQuery) -> Optional[str]:
    if getattr(mq, "doc_signature", None) is not None:
        return mq.doc_signature
    value = mq.context.get("doc_signature")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _item_doc_signature(item: MemoryItem) -> Optional[str]:
    if getattr(item, "doc_signature", None) is not None:
        return item.doc_signature
    value = item.meta.get("doc_signature")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _query_source_file(mq: MemoryQuery) -> Optional[str]:
    if getattr(mq, "source_file", None) is not None:
        value = mq.source_file
        if value is not None:
            text = str(value).strip()
            return text or None
    value = mq.context.get("source_file")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _item_source_file(item: MemoryItem) -> Optional[str]:
    if getattr(item, "source_file", None) is not None:
        value = item.source_file
        if value is not None:
            text = str(value).strip()
            return text or None
    value = item.meta.get("source_file")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def same_document(
    mq: MemoryQuery,
    item: MemoryItem,
) -> bool:
    """
    Return True only when both sides have document signatures and they match.

    This is intentionally stricter than older behavior because we now use it
    as a preference/ranking signal rather than an eligibility gate.
    """
    mq_doc = _query_doc_signature(mq)
    item_doc = _item_doc_signature(item)

    if mq_doc is None or item_doc is None:
        return False
    return mq_doc == item_doc


def same_source(
    mq: MemoryQuery,
    item: MemoryItem,
) -> bool:
    """
    Return True when both sides have source identifiers/files and they match.

    This is weaker than same_document and is intended as a cheap lexical preference signal.
    """
    mq_src = _query_source_file(mq)
    item_src = _item_source_file(item)

    if mq_src is None or item_src is None:
        return False
    return mq_src == item_src


def document_relation(
    mq: MemoryQuery,
    item: MemoryItem,
) -> str:
    """
    Classify semantic/lexical candidate document relation.

    Returns:
      - "same_document" when both signatures exist and match
      - "same_source" when document signatures do not match but source files do
      - "different_document" when both document signatures exist and differ
      - "unknown_document" when one or both signatures are unavailable and source does not match
    """
    mq_doc = _query_doc_signature(mq)
    item_doc = _item_doc_signature(item)

    if mq_doc is not None and item_doc is not None:
        if mq_doc == item_doc:
            return "same_document"
        if same_source(mq, item):
            return "same_source"
        return "different_document"

    if same_source(mq, item):
        return "same_source"

    return "unknown_document"


def lexical_candidate_allowed(
    mq: MemoryQuery,
    item: MemoryItem,
    *,
    policy: RetrievalPolicy,
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether a MemoryItem is eligible to participate in lexical retrieval.

    This is candidate filtering before lexical similarity thresholds are applied.

    Lexical behavior:
    - context signature is NOT required to match
    - same-document / same-source are surfaced as preference signals
    - lexical direct reuse remains controlled later by score + task gating
    """
    dbg: Dict[str, Any] = {"reason": "accepted"}

    if policy.enforce_ttl and not is_fresh(item, now_utc=now_utc):
        dbg["reason"] = "expired"
        return False, dbg

    if policy.require_same_task_for_lexical and not task_matches(mq, item):
        dbg["reason"] = "task_mismatch"
        dbg["item_task"] = item.meta.get("task")
        dbg["query_task"] = mq.task
        return False, dbg

    if policy.require_same_model_for_lexical and item.provenance.model_id != mq.model_id:
        dbg["reason"] = "model_mismatch"
        dbg["item_model_id"] = item.provenance.model_id
        dbg["query_model_id"] = mq.model_id
        return False, dbg

    if (
        policy.require_same_prompt_version_for_lexical
        and item.provenance.prompt_version != mq.prompt_version
    ):
        dbg["reason"] = "prompt_version_mismatch"
        dbg["item_prompt_version"] = item.provenance.prompt_version
        dbg["query_prompt_version"] = mq.prompt_version
        return False, dbg

    dbg["document_relation"] = document_relation(mq, item)
    dbg["same_document"] = same_document(mq, item)
    dbg["same_source"] = same_source(mq, item)
    dbg["item_doc_signature"] = _item_doc_signature(item)
    dbg["query_doc_signature"] = _query_doc_signature(mq)
    dbg["item_source_file"] = _item_source_file(item)
    dbg["query_source_file"] = _query_source_file(mq)
    return True, dbg


def lexical_decision(
    *,
    score: float,
    item: MemoryItem,
    mq: MemoryQuery,
    policy: RetrievalPolicy,
) -> Tuple[str, Dict[str, Any]]:
    """
    Decide what to do with a lexical candidate after similarity is computed.

    Returns:
      ("ignore" | "context" | "direct", debug_info)

    Lexical behavior:
    - context if score >= lexical_threshold_context
    - direct reuse only if:
        * score >= lexical_threshold_bypass
        * task is in safe_direct_reuse_tasks
    """
    dbg: Dict[str, Any] = {
        "score": score,
        "lexical_threshold_context": policy.lexical_threshold_context,
        "lexical_threshold_bypass": policy.lexical_threshold_bypass,
        "safe_direct_reuse_tasks": list(policy.safe_direct_reuse_tasks),
    }

    if score < policy.lexical_threshold_context:
        dbg["reason"] = "below_context_threshold"
        return "ignore", dbg

    task_l = str(mq.task or "").strip().lower()
    safe_tasks = {str(t).strip().lower() for t in policy.safe_direct_reuse_tasks}

    if score >= policy.lexical_threshold_bypass and task_l in safe_tasks:
        dbg["reason"] = "lexical_direct"
        return "direct", dbg

    dbg["reason"] = "lexical_context"
    return "context", dbg


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

    Evidence-guided behavior:
    - document mismatch does NOT reject a candidate
    - document relation is surfaced in debug so manager can prefer same-document
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

    dbg["query_context_signature"] = query_context_signature
    dbg["document_relation"] = document_relation(mq, item)
    dbg["same_document"] = same_document(mq, item)
    dbg["same_source"] = same_source(mq, item)
    dbg["item_doc_signature"] = _item_doc_signature(item)
    dbg["query_doc_signature"] = _query_doc_signature(mq)
    dbg["item_source_file"] = _item_source_file(item)
    dbg["query_source_file"] = _query_source_file(mq)
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
      ("ignore" | "context", debug_info)

    Evidence-guided spec:
    - semantic retrieval is assistive/context-only
    - semantic direct bypass is disabled
    """
    dbg: Dict[str, Any] = {
        "score": score,
        "semantic_threshold_context": policy.semantic_threshold_context,
        "semantic_threshold_bypass": policy.semantic_threshold_bypass,
        "semantic_bypass_allowed": bool(policy.allow_semantic_bypass),
    }

    if score < policy.semantic_threshold_context:
        dbg["reason"] = "below_context_threshold"
        return "ignore", dbg

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
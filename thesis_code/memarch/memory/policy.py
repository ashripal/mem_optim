from __future__ import annotations

"""
memarch/memory/policy.py

Deterministic retrieval + budget policies for memarch.

Design goals for the current architecture:
- Let embeddings (e.g. MiniLM) do the heavy lifting for semantic retrieval.
- Keep policy.py focused on *safety and precision*, not expensive semantics.
- Prefer same-document reuse for QA-style tasks.
- Allow semantic context only when it is likely helpful and low-risk.
- Allow semantic direct reuse only when it is strongly supported.

Important philosophy:
- This file is NOT trying to solve semantic equivalence from scratch.
- It is a calibration / gating layer on top of retrieval scores.
- Cheap canonical/query checks are used as guardrails, especially for Jetson.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from memarch.memory.schema import (
    MemoryItem,
    MemoryQuery,
    Scope,
)


# =============================================================================
# Policy dataclasses
# =============================================================================

@dataclass(frozen=True)
class BudgetPolicy:
    """
    Guardrails to keep memory operations bounded on constrained devices.

    On Jetson-class hardware, these limits matter because:
    - we want exact / lexical / metadata filtering to be cheap
    - we want semantic comparison to run on small candidate pools
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

    Notes:
    - embeddings / vector similarity should decide *which* candidates are close
    - this policy decides whether a candidate is safe for:
        - direct reuse
        - context-only use
        - reject
    """

    scope_order: List[Scope]

    # ------------------------------------------------------------------
    # Exact-match gating
    # ------------------------------------------------------------------
    require_context_match: bool = True
    require_prompt_version_match: bool = True
    require_model_id_match: bool = True
    enforce_ttl: bool = True

    # ------------------------------------------------------------------
    # Lexical retrieval policy
    # ------------------------------------------------------------------
    lexical_enabled: bool = False
    lexical_threshold_context: float = 0.55
    lexical_threshold_bypass: float = 0.90
    lexical_top_k: int = 3

    require_same_task_for_lexical: bool = True
    require_same_model_for_lexical: bool = True
    require_same_prompt_version_for_lexical: bool = True

    prefer_same_source: bool = True
    safe_direct_reuse_tasks: List[str] = field(default_factory=lambda: ["trec"])

    # ------------------------------------------------------------------
    # Semantic retrieval policy
    # ------------------------------------------------------------------
    semantic_enabled: bool = False
    semantic_threshold_context: float = 0.65
    semantic_threshold_direct: float = 0.80

    # Keep > 1.0 allowed so configs/tests can use 1.01 as "disable bypass"
    semantic_threshold_bypass: float = 0.95

    max_semantic_candidates: int = 5

    require_same_task_for_semantic: bool = True
    require_same_model_for_semantic: bool = True
    require_same_prompt_version_for_semantic: bool = True

    # If exact prompt/context/model drift matters for your benchmark, keep these strict.
    require_same_context_for_bypass: bool = True
    prefer_same_document_for_semantic: bool = True
    allow_semantic_bypass: bool = False

    # ------------------------------------------------------------------
    # Verified semantic direct reuse gates
    # ------------------------------------------------------------------
    require_same_document_for_semantic_bypass: bool = True
    semantic_bypass_min_margin: float = 0.02
    require_evidence_support_for_semantic_bypass: bool = True
    semantic_direct_reuse_tasks: List[str] = field(
        default_factory=lambda: ["squad", "extractive_qa", "qa", "trec"]
    )
    semantic_bypass_max_answer_words: int = 12

    # ------------------------------------------------------------------
    # Tightened QA semantic-context controls
    # ------------------------------------------------------------------
    # Default True because QA family-clustered tests are where semantic bleed
    # is most dangerous.
    require_same_document_for_semantic_context_qa: bool = True

    # If same-document is False, a broader semantic hit must be extremely strong
    # before it is allowed to contribute context to QA generation.
    broader_semantic_context_threshold_qa: float = 0.90

    # ------------------------------------------------------------------
    # Canonical compatibility controls
    # ------------------------------------------------------------------
    # These are intentionally cheap and deterministic. They do NOT replace
    # embeddings; they simply prevent obvious bad reuse.
    require_query_compatibility_for_semantic_context_qa: bool = True
    require_answer_type_match_for_semantic_bypass_qa: bool = True
    require_answer_type_match_for_semantic_context_qa: bool = False

    # ------------------------------------------------------------------
    # Hard-negative suppression
    # ------------------------------------------------------------------
    # These are used to suppress "same-domain but different question" collisions.
    semantic_min_content_overlap_ratio: float = 0.50
    semantic_min_content_jaccard: float = 0.30

    def __post_init__(self) -> None:
        if not self.scope_order:
            raise ValueError("scope_order must be non-empty")

        if not (0.0 <= float(self.lexical_threshold_context) <= 1.0):
            raise ValueError("lexical_threshold_context must be in [0,1]")
        if float(self.lexical_threshold_bypass) < 0.0:
            raise ValueError("lexical_threshold_bypass must be >= 0")
        if self.lexical_top_k < 1:
            raise ValueError("lexical_top_k must be >= 1")
        if self.lexical_threshold_bypass < self.lexical_threshold_context:
            raise ValueError(
                "lexical_threshold_bypass must be >= lexical_threshold_context"
            )

        if not (0.0 <= float(self.semantic_threshold_context) <= 1.0):
            raise ValueError("semantic_threshold_context must be in [0,1]")
        if float(self.semantic_threshold_bypass) < 0.0:
            raise ValueError("semantic_threshold_bypass must be >= 0")
        if self.max_semantic_candidates < 1:
            raise ValueError("max_semantic_candidates must be >= 1")
        if self.semantic_threshold_bypass < self.semantic_threshold_context:
            raise ValueError(
                "semantic_threshold_bypass must be >= semantic_threshold_context"
            )

        if float(self.semantic_bypass_min_margin) < 0.0:
            raise ValueError("semantic_bypass_min_margin must be >= 0")

        if int(self.semantic_bypass_max_answer_words) < 1:
            raise ValueError("semantic_bypass_max_answer_words must be >= 1")

        if not (0.0 <= float(self.broader_semantic_context_threshold_qa) <= 1.0):
            raise ValueError("broader_semantic_context_threshold_qa must be in [0,1]")

        if not (0.0 <= float(self.semantic_min_content_overlap_ratio) <= 1.0):
            raise ValueError("semantic_min_content_overlap_ratio must be in [0,1]")

        if not (0.0 <= float(self.semantic_min_content_jaccard) <= 1.0):
            raise ValueError("semantic_min_content_jaccard must be in [0,1]")

        for task in self.safe_direct_reuse_tasks:
            if not str(task).strip():
                raise ValueError("safe_direct_reuse_tasks must not contain empty task names")

        for task in self.semantic_direct_reuse_tasks:
            if not str(task).strip():
                raise ValueError("semantic_direct_reuse_tasks must not contain empty task names")


def default_retrieval_policy() -> RetrievalPolicy:
    return RetrievalPolicy(
        scope_order=[Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL]
    )


def budget_from_query(mq: MemoryQuery) -> BudgetPolicy:
    """
    Construct BudgetPolicy from MemoryQuery.
    """
    return BudgetPolicy(
        max_ram_reads=mq.max_ram_reads,
        max_disk_reads=mq.max_disk_reads,
        allow_semantic=mq.allow_semantic,
    )


# =============================================================================
# Small helpers
# =============================================================================

# Cheap tokenization helpers used only for *guardrails*.
# MiniLM similarity should still be the main semantic signal upstream.
_WORD_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)

# Remove mostly non-informative tokens when building cheap compatibility features.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "by", "with", "and", "or",
    "at", "from", "into", "during", "after", "before", "over", "under", "between",
    "through", "about", "what", "who", "when", "where", "which", "why", "how",
    "is", "was", "were", "are", "be", "been", "being", "did", "do", "does",
    "has", "have", "had", "that", "this", "these", "those", "many", "much",
    "name", "kind", "type", "sort", "title", "individuals", "people", "person",
}

# This is intentionally small. It is not meant to be a semantic ontology.
# It only catches a few high-frequency paraphrase forms cheaply.
_VERB_CANONICAL_MAP = {
    "lives": "live",
    "living": "live",
    "lived": "live",
    "reside": "live",
    "resides": "live",
    "resided": "live",
    "resident": "live",
    "residents": "live",
    "founded": "found",
    "founder": "found",
    "foundedby": "found",
    "created": "create",
    "creator": "create",
    "established": "establish",
    "establishes": "establish",
    "establishing": "establish",
    "wrote": "write",
    "written": "write",
    "writes": "write",
    "born": "birth",
    "birth": "birth",
    "located": "locate",
    "location": "locate",
    "locatedin": "locate",
}

# These are not actually stopwords; they are coarse entity/domain anchors that
# often cause false semantic matches. We keep them visible for debugging.
_ENTITY_HINT_STOPWORDS = {
    "notre", "dame", "university", "college", "school", "team", "group",
    "company", "city", "country", "state",
}

PERSON_QUERY_TOKENS = {"who", "whom", "whose", "person", "people", "individual", "individuals"}
COUNT_QUERY_PATTERNS = ("how many", "number of", "count of")
DATE_QUERY_TOKENS = {"when", "year", "date", "month", "day"}
LOCATION_QUERY_TOKENS = {"where", "location", "located"}
BOOLEAN_QUERY_PREFIXES = ("is ", "are ", "was ", "were ", "did ", "does ", "do ", "has ", "have ", "had ", "can ", "could ")
NAME_QUERY_TOKENS = {"name", "called", "known", "title"}

YEAR_RE = re.compile(r"^\d{4}$")
NUMERIC_RE = re.compile(r"^\d+([.,]\d+)?(%| percent| million| billion| thousand)?$", re.IGNORECASE)
MONTH_DAY_YEAR_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}$",
    re.IGNORECASE,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


def _tokenize(s: Optional[str]) -> List[str]:
    text = _norm_text(s)
    if not text:
        return []
    return text.split(" ")


def _word_count(s: Optional[str]) -> int:
    return len(_tokenize(s))


def _task_norm(s: Optional[str]) -> str:
    return str(s or "").strip().lower()


def _safe_task_set(values: Sequence[str]) -> set[str]:
    return {_task_norm(v) for v in values if str(v).strip()}


def _is_qa_task_name(task_name: str) -> bool:
    return task_name in {"squad", "qa", "extractive_qa"}


def _safe_meta(obj: Any) -> Dict[str, Any]:
    meta = getattr(obj, "meta", None)
    return meta if isinstance(meta, dict) else {}


def _first_nonempty(*values: Any) -> Optional[str]:
    for v in values:
        if v is None:
            continue
        text = str(v).strip()
        if text:
            return text
    return None


def _normalize_token(token: str) -> str:
    tok = str(token or "").strip().lower()
    if not tok:
        return ""
    tok = _VERB_CANONICAL_MAP.get(tok, tok)
    if tok.endswith("'s"):
        tok = tok[:-2]
    return tok


def _wordish_tokens(text: str) -> List[str]:
    return [_normalize_token(t) for t in _WORD_RE.findall(_norm_text(text)) if _normalize_token(t)]


def _content_tokens(text: str) -> List[str]:
    """
    Return lightly normalized content tokens.

    This is deliberately cheap:
    - lowercase
    - tiny verb canonicalization
    - remove stopwords
    """
    toks: List[str] = []
    for tok in _wordish_tokens(text):
        if tok in _STOPWORDS:
            continue
        if len(tok) <= 1:
            continue
        toks.append(tok)
    return toks


def _content_token_set(text: str) -> set[str]:
    return set(_content_tokens(text))


def _is_short_numeric_or_date_answer(ans_norm: str) -> bool:
    if not ans_norm:
        return False
    return bool(
        YEAR_RE.fullmatch(ans_norm)
        or NUMERIC_RE.fullmatch(ans_norm)
        or MONTH_DAY_YEAR_RE.fullmatch(ans_norm)
    )


def _question_text_from_mq(mq_obj: MemoryQuery) -> str:
    return _first_nonempty(
        getattr(mq_obj, "raw_query", None),
        getattr(mq_obj, "query_text", None),
        getattr(mq_obj, "question", None),
        getattr(mq_obj, "query", None),
        getattr(mq_obj, "text", None),
        (getattr(mq_obj, "context", {}) or {}).get("query_text"),
        (getattr(mq_obj, "context", {}) or {}).get("question"),
    ) or ""


def _question_text_from_item(item_obj: MemoryItem) -> str:
    meta = _safe_meta(item_obj)
    return _first_nonempty(
        getattr(item_obj, "raw_query", None),
        getattr(item_obj, "query_text", None),
        getattr(item_obj, "question", None),
        meta.get("raw_query"),
        meta.get("query_text"),
        meta.get("question"),
    ) or ""


def _answer_text_from_item(item_obj: MemoryItem) -> str:
    meta = _safe_meta(item_obj)
    return _first_nonempty(
        getattr(item_obj, "answer_canonical", None),
        getattr(item_obj, "answer_text", None),
        meta.get("answer_canonical"),
        meta.get("answer_text"),
        meta.get("stored_answer_text"),
    ) or ""


def _question_type_from_mq(mq_obj: MemoryQuery) -> str:
    return _first_nonempty(
        getattr(mq_obj, "question_type", None),
        (getattr(mq_obj, "context", {}) or {}).get("question_type"),
    ) or ""


def _question_type_from_item(item_obj: MemoryItem) -> str:
    meta = _safe_meta(item_obj)
    return _first_nonempty(
        getattr(item_obj, "question_type", None),
        meta.get("question_type"),
    ) or ""


def _query_answer_canonical(mq_obj: MemoryQuery) -> Optional[str]:
    return _first_nonempty(
        getattr(mq_obj, "answer_canonical", None),
        (getattr(mq_obj, "context", {}) or {}).get("answer_canonical"),
    )


def _canonical_query_signature_explicit(obj: Any) -> Optional[str]:
    meta = _safe_meta(obj)
    return _first_nonempty(
        getattr(obj, "canonical_query_signature", None),
        getattr(obj, "query_signature", None),
        meta.get("canonical_query_signature"),
        meta.get("query_signature"),
    )


def _answer_type_explicit(obj: Any) -> Optional[str]:
    meta = _safe_meta(obj)
    value = _first_nonempty(
        getattr(obj, "answer_type", None),
        meta.get("answer_type"),
    )
    return value.upper() if value else None


def _derive_answer_type_from_query_text(query_text: str) -> str:
    """
    Derive a coarse answer type from the *query*.

    This is intentionally coarse because:
    - it is only used as a safety signal
    - it should be cheap enough to run on every retrieval
    """
    q = _norm_text(query_text)
    if not q:
        return "UNKNOWN"

    if q.startswith(COUNT_QUERY_PATTERNS):
        return "COUNT"

    first = q.split()[0]

    if first in PERSON_QUERY_TOKENS or any(tok in PERSON_QUERY_TOKENS for tok in _tokenize(q)):
        return "PERSON"

    if first in DATE_QUERY_TOKENS or any(tok in DATE_QUERY_TOKENS for tok in _tokenize(q)):
        return "DATE"

    if first in LOCATION_QUERY_TOKENS or any(tok in LOCATION_QUERY_TOKENS for tok in _tokenize(q)):
        return "LOCATION"

    if any(q.startswith(prefix) for prefix in BOOLEAN_QUERY_PREFIXES):
        return "BOOLEAN"

    if any(tok in NAME_QUERY_TOKENS for tok in _tokenize(q)):
        return "NAME"

    return "UNKNOWN"


def _derive_answer_type_from_answer_text(answer_text: str) -> str:
    """
    Fallback answer-type derivation from the stored answer text.

    This should only be used when query-derived type is unavailable.
    """
    ans = str(answer_text or "").strip()
    ans_norm = _norm_text(ans)
    if not ans_norm:
        return "UNKNOWN"

    if _is_short_numeric_or_date_answer(ans_norm):
        if YEAR_RE.fullmatch(ans_norm) or MONTH_DAY_YEAR_RE.fullmatch(ans_norm):
            return "DATE"
        return "COUNT"

    toks = ans.split()
    if len(toks) >= 1 and all(t[:1].isupper() for t in toks if t and t[0].isalnum()):
        return "NAME"

    return "UNKNOWN"


def canonical_answer_type_for_query(obj: Any) -> str:
    explicit = _answer_type_explicit(obj)
    if explicit:
        return explicit
    return _derive_answer_type_from_query_text(
        _question_text_from_mq(obj) if isinstance(obj, MemoryQuery) else _question_text_from_item(obj)
    )


def canonical_answer_type_for_item(item: MemoryItem) -> str:
    explicit = _answer_type_explicit(item)
    if explicit:
        return explicit

    from_q = _derive_answer_type_from_query_text(_question_text_from_item(item))
    if from_q != "UNKNOWN":
        return from_q

    return _derive_answer_type_from_answer_text(_answer_text_from_item(item))


def _predicate_signature(tokens: List[str]) -> List[str]:
    """
    Extract a tiny predicate signature.

    This is deliberately *small* because the main semantic engine should be the
    embedder. These predicates only help suppress obvious false positives.
    """
    preds: List[str] = []
    for tok in tokens:
        if tok in {"live", "found", "create", "establish", "write", "birth", "locate"}:
            preds.append(tok)
    return preds


def _entity_signature(tokens: List[str]) -> List[str]:
    """
    Extract a compact entity-ish signature.

    This is used for lightweight debugging / compatibility checks, not as the
    main semantic retrieval mechanism.
    """
    ents: List[str] = []
    for tok in tokens:
        if tok in _ENTITY_HINT_STOPWORDS:
            ents.append(tok)
            continue
        if len(tok) >= 3:
            ents.append(tok)
    return ents[:8]


def derive_canonical_query_signature_from_text(query_text: str, answer_type: Optional[str] = None) -> str:
    """
    Produce a small canonical query signature.

    Example:
      PERSON::live::fatima|house|notre|dame

    This is NOT meant to replace MiniLM. It is only a cheap, deterministic
    family signature for policy-time compatibility checks.
    """
    q = _norm_text(query_text)
    inferred_answer_type = (answer_type or _derive_answer_type_from_query_text(q)).upper()
    toks = _content_tokens(q)
    preds = _predicate_signature(toks)
    ents = _entity_signature(toks)

    pred_sig = "|".join(sorted(set(preds))) if preds else "none"
    ent_sig = "|".join(sorted(set(ents[:8]))) if ents else "none"
    return f"{inferred_answer_type}::{pred_sig}::{ent_sig}"


def canonical_query_signature_for_mq(mq: MemoryQuery) -> str:
    explicit = _canonical_query_signature_explicit(mq)
    if explicit:
        return explicit
    return derive_canonical_query_signature_from_text(
        _question_text_from_mq(mq),
        answer_type=canonical_answer_type_for_query(mq),
    )


def canonical_query_signature_for_item(item: MemoryItem) -> str:
    explicit = _canonical_query_signature_explicit(item)
    if explicit:
        return explicit
    return derive_canonical_query_signature_from_text(
        _question_text_from_item(item),
        answer_type=canonical_answer_type_for_item(item),
    )


def answer_type_compatible(mq: MemoryQuery, item: MemoryItem) -> Tuple[bool, Dict[str, Any]]:
    """
    Coarse answer-type compatibility check.

    This is important because semantically nearby questions can still ask for
    different answer *types*.
    """
    query_type = canonical_answer_type_for_query(mq)
    item_type = canonical_answer_type_for_item(item)

    dbg = {
        "query_answer_type": query_type,
        "item_answer_type": item_type,
    }

    # Unknown should not automatically block a candidate. Treat it as "no signal".
    if query_type == "UNKNOWN" or item_type == "UNKNOWN":
        dbg["reason"] = "unknown_type_allowed"
        return True, dbg

    ok = query_type == item_type
    dbg["reason"] = "match" if ok else "mismatch"
    return ok, dbg


def canonical_query_compatible(mq: MemoryQuery, item: MemoryItem) -> Tuple[bool, Dict[str, Any]]:
    """
    Cheap compatibility check between current and stored questions.

    Important:
    - this is not the primary semantic model
    - it is a low-cost precision gate to prevent obvious bad reuse
    """
    current_query_text = _question_text_from_mq(mq)
    stored_query_text = _question_text_from_item(item)

    cq_sig = canonical_query_signature_for_mq(mq)
    sq_sig = canonical_query_signature_for_item(item)

    current_tokens = _content_token_set(current_query_text)
    stored_tokens = _content_token_set(stored_query_text)

    overlap = current_tokens.intersection(stored_tokens)
    union = current_tokens.union(stored_tokens)

    overlap_ratio = (
        len(overlap) / max(1, min(len(current_tokens), len(stored_tokens)))
        if current_tokens and stored_tokens
        else 0.0
    )
    jaccard = len(overlap) / max(1, len(union)) if union else 0.0

    current_answer_type = canonical_answer_type_for_query(mq)
    stored_answer_type = canonical_answer_type_for_item(item)

    pred_current = cq_sig.split("::")[1] if "::" in cq_sig else "none"
    pred_stored = sq_sig.split("::")[1] if "::" in sq_sig else "none"
    ent_current = cq_sig.split("::")[2] if "::" in cq_sig else "none"
    ent_stored = sq_sig.split("::")[2] if "::" in sq_sig else "none"

    predicate_overlap = bool(pred_current != "none" and pred_current == pred_stored)
    entity_overlap = bool(
        ent_current != "none"
        and ent_stored != "none"
        and len(set(ent_current.split("|")).intersection(set(ent_stored.split("|")))) > 0
    )

    dbg: Dict[str, Any] = {
        "current_query_signature": cq_sig,
        "stored_query_signature": sq_sig,
        "query_answer_type": current_answer_type,
        "item_answer_type": stored_answer_type,
        "current_content_tokens": sorted(current_tokens),
        "stored_content_tokens": sorted(stored_tokens),
        "token_overlap": sorted(overlap),
        "token_overlap_ratio": overlap_ratio,
        "token_jaccard": jaccard,
        "predicate_overlap": predicate_overlap,
        "entity_overlap": entity_overlap,
    }

    if (
        current_answer_type != "UNKNOWN"
        and stored_answer_type != "UNKNOWN"
        and current_answer_type != stored_answer_type
    ):
        dbg["reason"] = "answer_type_mismatch"
        return False, dbg

    if cq_sig == sq_sig:
        dbg["reason"] = "canonical_signature_match"
        return True, dbg

    # Good enough when both predicate and entity anchors line up.
    if predicate_overlap and entity_overlap and overlap_ratio >= 0.50:
        dbg["reason"] = "predicate_entity_overlap"
        return True, dbg

    # Strong fallback on token overlap only.
    if overlap_ratio >= 0.70 and jaccard >= 0.45:
        dbg["reason"] = "strong_token_overlap"
        return True, dbg

    dbg["reason"] = "low_query_overlap"
    return False, dbg


# =============================================================================
# Freshness / exact-match helpers
# =============================================================================

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
    """
    item_task = getattr(item, "task", None)
    item_task_text = str(item_task or "").strip().lower()

    legacy_item_task = _safe_meta(item).get("task")
    legacy_item_task_text = str(legacy_item_task or "").strip().lower()

    if item_task_text and item_task_text != "default":
        return item_task_text == _task_norm(mq.task)

    if legacy_item_task_text:
        return legacy_item_task_text == _task_norm(mq.task)

    return True


# =============================================================================
# Document/source helpers
# =============================================================================

def _query_doc_signature(mq: MemoryQuery) -> Optional[str]:
    if getattr(mq, "doc_signature", None) is not None:
        value = mq.doc_signature
        if value is not None:
            text = str(value).strip()
            return text or None

    value = (mq.context or {}).get("doc_signature")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _item_doc_signature(item: MemoryItem) -> Optional[str]:
    if getattr(item, "doc_signature", None) is not None:
        value = item.doc_signature
        if value is not None:
            text = str(value).strip()
            return text or None

    value = _safe_meta(item).get("doc_signature")
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
    value = (mq.context or {}).get("source_file")
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
    value = _safe_meta(item).get("source_file")
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


# =============================================================================
# Evidence helpers
# =============================================================================

def _query_evidence_text(mq: MemoryQuery) -> Optional[str]:
    if getattr(mq, "evidence_text", None):
        text = str(mq.evidence_text).strip()
        if text:
            return text

    value = (mq.context or {}).get("evidence_text")
    if value is not None:
        text = str(value).strip()
        if text:
            return text

    value = (mq.context or {}).get("dataset_context")
    if value is not None:
        text = str(value).strip()
        if text:
            return text

    return None


def answer_supported_by_evidence(
    mq: MemoryQuery,
    item: MemoryItem,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Cheap deterministic evidence-support check.

    Why cheap?
    - This runs in the retrieval policy loop.
    - On Jetson, we want to avoid expensive verifier calls unless really needed.
    """
    dbg: Dict[str, Any] = {}

    evidence_text = _query_evidence_text(mq)
    answer_text = getattr(item, "answer_canonical", None) or item.answer_text

    evidence_norm = _norm_text(evidence_text)
    answer_norm = _norm_text(answer_text)

    dbg["answer_text"] = answer_text
    dbg["answer_norm"] = answer_norm
    dbg["evidence_present"] = bool(evidence_norm)

    if not answer_norm:
        dbg["reason"] = "missing_answer_text"
        return False, dbg

    if not evidence_norm:
        dbg["reason"] = "missing_query_evidence"
        return False, dbg

    if answer_norm in evidence_norm:
        dbg["reason"] = "answer_substring_found_in_evidence"
        return True, dbg

    answer_tokens = set(_tokenize(answer_norm))
    evidence_tokens = set(_tokenize(evidence_norm))

    if not answer_tokens:
        dbg["reason"] = "empty_answer_tokens"
        return False, dbg

    overlap = len(answer_tokens.intersection(evidence_tokens))
    overlap_ratio = overlap / max(1, len(answer_tokens))

    dbg["answer_token_count"] = len(answer_tokens)
    dbg["evidence_token_count"] = len(evidence_tokens)
    dbg["answer_token_overlap"] = overlap
    dbg["answer_token_overlap_ratio"] = overlap_ratio

    if overlap_ratio >= 0.8:
        dbg["reason"] = "strong_answer_token_overlap"
        return True, dbg

    if bool(getattr(item, "answer_span_found", False)) and same_document(mq, item):
        dbg["reason"] = "stored_answer_span_found_same_document"
        return True, dbg

    dbg["reason"] = "answer_not_supported_in_current_evidence"
    return False, dbg


# =============================================================================
# Lexical policy
# =============================================================================

def lexical_candidate_allowed(
    mq: MemoryQuery,
    item: MemoryItem,
    *,
    policy: RetrievalPolicy,
    now_utc: Optional[datetime] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether a MemoryItem is eligible to participate in lexical retrieval.
    """
    dbg: Dict[str, Any] = {"reason": "accepted"}

    if policy.enforce_ttl and not is_fresh(item, now_utc=now_utc):
        dbg["reason"] = "expired"
        return False, dbg

    if policy.require_same_task_for_lexical and not task_matches(mq, item):
        dbg["reason"] = "task_mismatch"
        dbg["item_task"] = getattr(item, "task", None) or _safe_meta(item).get("task")
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

    task_l = _task_norm(mq.task)
    safe_tasks = _safe_task_set(policy.safe_direct_reuse_tasks)

    if score >= policy.lexical_threshold_bypass and task_l in safe_tasks:
        dbg["reason"] = "lexical_direct"
        return "direct", dbg

    dbg["reason"] = "lexical_context"
    return "context", dbg


# =============================================================================
# Semantic policy
# =============================================================================

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

    Important:
    - this is only the *pre-filter*
    - final direct/context/reject happens in semantic_decision()
    """
    dbg: Dict[str, Any] = {"reason": "accepted"}

    if policy.enforce_ttl and not is_fresh(item, now_utc=now_utc):
        dbg["reason"] = "expired"
        return False, dbg

    if policy.require_same_task_for_semantic and not task_matches(mq, item):
        dbg["reason"] = "task_mismatch"
        dbg["item_task"] = getattr(item, "task", None) or _safe_meta(item).get("task")
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

    # Semantic path requires a stored embedding.
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
    dbg["canonical_query_signature"] = canonical_query_signature_for_item(item)
    dbg["answer_type"] = canonical_answer_type_for_item(item)
    return True, dbg


def semantic_decision(
    *,
    mq: MemoryQuery,
    score: float | None,
    item: MemoryItem,
    policy: RetrievalPolicy,
    query_context_signature: str | None = None,
    next_best_score: float | None = None,
    competing_item: Optional[MemoryItem] = None,
) -> tuple[str, dict]:
    """
    Decide how to use a semantic candidate.

    Returns:
      ("ignore" | "context" | "direct", debug_dict)

    Design:
    - semantic context is allowed only when the hit looks compatible enough
    - QA semantic context is stricter and defaults to same-document only
    - semantic direct reuse is strict: it additionally requires query
      compatibility, same-document/evidence gates, answer-length bounds, and
      non-ambiguity
    """

    item_doc_sig = _item_doc_signature(item)
    query_doc_sig = _query_doc_signature(mq)
    same_document_flag = bool(
        query_doc_sig is not None
        and item_doc_sig is not None
        and query_doc_sig == item_doc_sig
    )

    same_source_flag = same_source(mq, item)
    relation = document_relation(mq, item)

    task_name = _task_norm(getattr(mq, "task", None))
    is_qa_task = _is_qa_task_name(task_name) or _question_type_from_mq(mq).strip().lower() == "qa"

    allowed_tasks = [
        _task_norm(x)
        for x in getattr(policy, "semantic_direct_reuse_tasks", []) or []
        if str(x).strip()
    ]
    task_compatible = (not allowed_tasks) or (task_name in allowed_tasks)

    query_compatible, query_compat_dbg = canonical_query_compatible(mq, item)
    answer_type_ok, answer_type_dbg = answer_type_compatible(mq, item)
    evidence_supported, evidence_dbg = answer_supported_by_evidence(mq, item)

    threshold_context = float(getattr(policy, "semantic_threshold_context", 0.85))
    threshold_bypass = float(getattr(policy, "semantic_threshold_bypass", 0.95))
    allow_bypass = bool(getattr(policy, "allow_semantic_bypass", False))
    require_same_document = bool(
        getattr(policy, "require_same_document_for_semantic_bypass", True)
    )
    require_evidence_support = bool(
        getattr(policy, "require_evidence_support_for_semantic_bypass", True)
    )
    min_margin = float(getattr(policy, "semantic_bypass_min_margin", 0.0))
    max_answer_words = int(getattr(policy, "semantic_bypass_max_answer_words", 12))

    # These thresholds are used for hard-negative suppression.
    min_overlap_ratio = float(getattr(policy, "semantic_min_content_overlap_ratio", 0.50))
    min_jaccard = float(getattr(policy, "semantic_min_content_jaccard", 0.30))

    competing_same_answer = False
    competing_same_document = False
    ambiguous = False
    score_margin_vs_next = None

    if score is not None and next_best_score is not None:
        score_margin_vs_next = float(score) - float(next_best_score)

        if competing_item is not None:
            competing_doc_sig = _item_doc_signature(competing_item)
            competing_answer = _answer_text_from_item(competing_item)
            item_answer_for_compare = _answer_text_from_item(item)

            competing_same_document = bool(
                item_doc_sig is not None
                and competing_doc_sig is not None
                and item_doc_sig == competing_doc_sig
            )
            competing_same_answer = bool(
                _norm_text(item_answer_for_compare)
                and _norm_text(item_answer_for_compare) == _norm_text(competing_answer)
            )

        # Same-document + same-answer competitors are usually family duplicates,
        # not true ambiguity.
        if competing_same_document and competing_same_answer:
            ambiguous = False
        else:
            ambiguous = score_margin_vs_next < min_margin

    item_answer_text = _answer_text_from_item(item)
    answer_word_count = _word_count(item_answer_text)
    answer_len_ok = answer_word_count <= max_answer_words

    # Read overlap stats from the cheap compatibility debug payload.
    token_overlap_ratio = float(query_compat_dbg.get("token_overlap_ratio", 0.0) or 0.0)
    token_jaccard = float(query_compat_dbg.get("token_jaccard", 0.0) or 0.0)

    hard_negative_suspected = (
        not query_compatible
        and token_overlap_ratio < min_overlap_ratio
        and token_jaccard < min_jaccard
    )

    # -----------------------------
    # Context-use decision
    # -----------------------------
    use_context = bool(score is not None and float(score) >= threshold_context)
    context_reason = "below_context_threshold"

    if use_context:
        context_reason = "semantic_context_threshold_pass"

        if use_context and is_qa_task:
            # IMPORTANT:
            # Same-document semantic hits should remain usable as generation
            # context even when the cheap compatibility layer is imperfect.
            #
            # The current manager tests expect:
            # - semantic hit is passed as retrieved context
            # - not direct bypass
            #
            # So only apply the stricter guards to broader/non-same-document hits.
            if not same_document_flag:
                # Hard negative suppression should only block broader hits.
                if hard_negative_suspected:
                    use_context = False
                    context_reason = "qa_context_hard_negative_suspected"

                # For broader QA context, optionally require stronger score.
                if use_context and bool(getattr(policy, "require_same_document_for_semantic_context_qa", False)):
                    broader_threshold = float(
                        getattr(policy, "broader_semantic_context_threshold_qa", 0.90)
                    )
                    if score is None or float(score) < broader_threshold:
                        use_context = False
                        context_reason = "qa_context_requires_same_document_or_higher_threshold"

                # For broader hits, optional cheap compatibility check.
                if (
                    use_context
                    and bool(
                        getattr(
                            policy,
                            "require_query_compatibility_for_semantic_context_qa",
                            False,
                        )
                    )
                    and not query_compatible
                ):
                    use_context = False
                    context_reason = "qa_context_query_incompatible"

                if (
                    use_context
                    and bool(
                        getattr(
                            policy,
                            "require_answer_type_match_for_semantic_context_qa",
                            False,
                        )
                    )
                    and not answer_type_ok
                ):
                    use_context = False
                    context_reason = "qa_context_answer_type_mismatch"

            else:
                # Same-document QA semantic hits are allowed as context by default.
                # This preserves recall for paraphrase-family retrieval.
                context_reason = "qa_same_document_semantic_context"

    # -----------------------------
    # Direct reuse decision
    # -----------------------------
    bypass = bool(
        allow_bypass
        and score is not None
        and float(score) >= threshold_bypass
        and task_compatible
        and query_compatible
        and not hard_negative_suspected
        and (same_document_flag or not require_same_document)
        and (evidence_supported or not require_evidence_support)
        and answer_len_ok
        and not ambiguous
        and (
            (not is_qa_task)
            or (not bool(getattr(policy, "require_answer_type_match_for_semantic_bypass_qa", True)))
            or answer_type_ok
        )
    )

    dbg = {
        "score": float(score) if score is not None else None,
        "threshold_context": threshold_context,
        "threshold_bypass": threshold_bypass,
        "same_document": same_document_flag,
        "same_source": same_source_flag,
        "document_relation": relation,
        "query_context_signature": query_context_signature,
        "item_doc_signature": item_doc_sig,
        "query_doc_signature": query_doc_sig,
        "task_compatible": task_compatible,
        "allowed_tasks": allowed_tasks,
        "is_qa_task": is_qa_task,
        "evidence_supported": evidence_supported,
        "evidence_debug": evidence_dbg,
        "ambiguous": ambiguous,
        "score_margin_vs_next": score_margin_vs_next,
        "next_best_score": next_best_score,
        "competing_item_present": competing_item is not None,
        "competing_same_document": competing_same_document,
        "competing_same_answer": competing_same_answer,
        "answer_text": item_answer_text,
        "answer_word_count": answer_word_count,
        "semantic_bypass_max_answer_words": max_answer_words,
        "query_text_current": _question_text_from_mq(mq),
        "query_text_stored": _question_text_from_item(item),
        "query_compatible": query_compatible,
        "query_compatibility_debug": query_compat_dbg,
        "answer_type_match": answer_type_ok,
        "answer_type_debug": answer_type_dbg,
        "canonical_query_signature_current": canonical_query_signature_for_mq(mq),
        "canonical_query_signature_item": canonical_query_signature_for_item(item),
        "allow_bypass": allow_bypass,
        "require_same_document_for_bypass": require_same_document,
        "require_evidence_support_for_bypass": require_evidence_support,
        "require_same_document_for_semantic_context_qa": bool(
            getattr(policy, "require_same_document_for_semantic_context_qa", True)
        ),
        "broader_semantic_context_threshold_qa": float(
            getattr(policy, "broader_semantic_context_threshold_qa", 0.90)
        ),
        "semantic_min_content_overlap_ratio": min_overlap_ratio,
        "semantic_min_content_jaccard": min_jaccard,
        "hard_negative_suspected": hard_negative_suspected,
        "context_reason": context_reason,
    }

    if bypass:
        dbg["reason"] = "direct"
        return "direct", dbg

    if use_context:
        dbg["reason"] = "context"
        return "context", dbg

    dbg["reason"] = context_reason
    return "ignore", dbg


# =============================================================================
# Exact accept policy
# =============================================================================

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
    """
    Standardized hit-debug payload used by manager.py logging and tests.
    """
    d: Dict[str, Any] = {
        "scope": scope.value,
        "namespace": namespace,
        "source": source,
        "reason": accepted_reason,
    }
    if extra:
        d.update(extra)
    return d
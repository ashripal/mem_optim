from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


# =============================================================================
# Enums
# =============================================================================

class Scope(str, Enum):
    """
    Logical namespace scope for a memory item.
    """
    SESSION = "session"
    USER = "user"
    COHORT = "cohort"
    GLOBAL = "global"


class SourceTier(str, Enum):
    """
    Physical storage tier where the hit came from.
    """
    RAM = "ram"
    DISK = "disk"


class MatchType(str, Enum):
    """
    High-level retrieval match category.
    """
    EXACT = "exact"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"


class VerificationOutcome(str, Enum):
    """
    Verification result for semantic paraphrase reuse.

    Intended usage:
    - DIRECT_REUSE: safe to return the stored answer directly
    - CONTEXT_ONLY: useful hit, but not safe enough to bypass the LLM
    - REJECT: do not use this candidate for reuse
    """
    DIRECT_REUSE = "semantic_direct_reuse"
    CONTEXT_ONLY = "semantic_context_only"
    REJECT = "semantic_reject"


# =============================================================================
# Helper dataclasses
# =============================================================================

@dataclass(frozen=True)
class Provenance:
    """
    Metadata about how an answer was originally generated.
    """
    model_id: str
    prompt_version: str
    generated_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    generator_backend: Optional[str] = None
    quantization: Optional[str] = None
    context_window: Optional[int] = None

    def __post_init__(self) -> None:
        if not str(self.model_id).strip():
            raise ValueError("Provenance.model_id must be non-empty")
        if not str(self.prompt_version).strip():
            raise ValueError("Provenance.prompt_version must be non-empty")
        if self.generated_at_utc.tzinfo is None:
            raise ValueError("Provenance.generated_at_utc must be timezone-aware (UTC)")


@dataclass
class QualitySignals:
    """
    Optional quality metadata attached to a stored memory item.
    """
    score: Optional[float] = None
    success: Optional[bool] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None and not (0.0 <= float(self.score) <= 1.0):
            raise ValueError(f"QualitySignals.score must be in [0,1], got {self.score}")

        cleaned: Dict[str, float] = {}
        for k, v in (self.metrics or {}).items():
            key = str(k).strip()
            if not key:
                raise ValueError("QualitySignals.metrics keys must be non-empty")
            cleaned[key] = float(v)
        self.metrics = cleaned


@dataclass
class AccessStats:
    """
    Lightweight access bookkeeping for RAM/disk promotion and analysis.
    """
    access_count: int = 0
    last_access_utc: Optional[datetime] = None

    def __post_init__(self) -> None:
        if int(self.access_count) < 0:
            raise ValueError("AccessStats.access_count must be >= 0")
        if self.last_access_utc is not None and self.last_access_utc.tzinfo is None:
            raise ValueError("AccessStats.last_access_utc must be timezone-aware (UTC)")

    def touch(self, when: Optional[datetime] = None) -> None:
        self.access_count += 1
        self.last_access_utc = when or datetime.now(timezone.utc)


# =============================================================================
# Main dataclasses
# =============================================================================

@dataclass(frozen=True)
class MemoryQuery:
    """
    Query-time request passed into the memory system.

    Notes:
    - raw_query is the user-visible question as asked now.
    - task/question_type/doc_signature/evidence_text are the key fields needed
      for verified paraphrase reuse.
    - context remains a flexible dict for workload-specific additions.
    """
    raw_query: str

    # Identity / namespace routing
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    cohort_id: Optional[str] = None

    # Task / model routing
    task: str = "default"
    context: Dict[str, Any] = field(default_factory=dict)
    prompt_version: str = "v1"
    model_id: str = "mistral-7b-instruct"

    # Retrieval budget / behavior knobs
    allow_semantic: bool = False
    max_disk_reads: int = 16
    max_ram_reads: int = 64

    # Document / evidence fields used by same-document and verifier logic
    doc_signature: Optional[str] = None
    source_file: Optional[str] = None
    source_id: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None
    question_type: Optional[str] = None

    # Local evidence used for current-query verification
    evidence_text: Optional[str] = None

    # Optional normalized answer hint for future use
    answer_canonical: Optional[str] = None

    # Optional normalized/canonicalized query form
    query_canonical: Optional[str] = None

    # Optional richer canonical reuse fields.
    # These let manager.py / policy.py use first-class fields instead of only meta.
    canonical_query_signature: Optional[str] = None
    answer_type: Optional[str] = None
    
    def __post_init__(self) -> None:
        if not self.raw_query or not self.raw_query.strip():
            raise ValueError("MemoryQuery.raw_query must be non-empty")

        normalized_task = str(self.task or "").strip() or "default"
        object.__setattr__(self, "task", normalized_task)

        if not str(self.prompt_version).strip():
            raise ValueError("MemoryQuery.prompt_version must be non-empty")

        if not str(self.model_id).strip():
            raise ValueError("MemoryQuery.model_id must be non-empty")

        if self.max_disk_reads < 0 or self.max_ram_reads < 0:
            raise ValueError("max_disk_reads/max_ram_reads must be >= 0")

        if not isinstance(self.context, dict):
            raise TypeError("MemoryQuery.context must be a dict")

        canonical = str(self.query_canonical or "").strip() or self.raw_query.strip()
        object.__setattr__(self, "query_canonical", canonical)

        if self.question_type is not None:
            object.__setattr__(self, "question_type", str(self.question_type).strip() or None)

        if self.answer_canonical is not None:
            object.__setattr__(self, "answer_canonical", str(self.answer_canonical).strip() or None)

        if self.canonical_query_signature is not None:
            object.__setattr__(
                self,
                "canonical_query_signature",
                str(self.canonical_query_signature).strip() or None,
            )

        if self.answer_type is not None:
            object.__setattr__(self, "answer_type", str(self.answer_type).strip().upper() or None)


@dataclass
class MemoryItem:
    """
    Stored memory record.

    Key additions for the newer semantic reuse strategy:
    - canonical_query_signature: canonicalized query-family representation
    - answer_type: coarse answer class used for safer compatibility checks
    - optional family_id / canonical_memory_key / is_alias hooks for future
      alias-based storage without breaking current code
    """
    key: str
    scope: Scope
    namespace: str

    # Canonical retrieval keys
    query_canonical: str
    context_signature: str
    answer_text: str

    # Original query text that created this memory
    raw_query: Optional[str] = None

    # Task metadata
    task: str = "default"
    question_type: Optional[str] = None
    answer_type: Optional[str] = None
    canonical_query_signature: Optional[str] = None

    # Optional family / alias metadata for future storage dedupe
    family_id: Optional[str] = None
    canonical_memory_key: Optional[str] = None
    is_alias: bool = False

    # Provenance / quality
    provenance: Provenance = field(
        default_factory=lambda: Provenance(
            model_id="unknown-model",
            prompt_version="v1",
        )
    )
    quality: QualitySignals = field(default_factory=QualitySignals)

    # Lifetime metadata
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: Optional[int] = None
    expires_at_utc: Optional[datetime] = None

    # Access bookkeeping and free-form metadata
    stats: AccessStats = field(default_factory=AccessStats)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Evidence / document grounding
    evidence_text: Optional[str] = None
    doc_signature: Optional[str] = None
    source_file: Optional[str] = None
    source_id: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Optional canonical answer / verifier hints
    answer_canonical: Optional[str] = None
    answer_span_found: Optional[bool] = None

    # Optional richer intent metadata for future canonical-question matching
    canonical_intent_id: Optional[str] = None

    # Jetson-friendly semantic retrieval fields
    query_embedding: Optional[array] = None
    embedding_model_id: Optional[str] = None
    embedding_norm: Optional[float] = None

    MAX_EMBED_DIM = 1024
    MAX_META_KEYS = 64

    def __post_init__(self) -> None:
        if not str(self.key).strip():
            raise ValueError("MemoryItem.key must be non-empty")

        if not isinstance(self.scope, Scope):
            raise TypeError("MemoryItem.scope must be a Scope")

        if not str(self.namespace).strip():
            raise ValueError("MemoryItem.namespace must be non-empty")

        if not str(self.query_canonical).strip():
            raise ValueError("MemoryItem.query_canonical must be non-empty")

        if not str(self.context_signature).strip():
            raise ValueError("MemoryItem.context_signature must be non-empty")

        if not str(self.answer_text).strip():
            raise ValueError("MemoryItem.answer_text must be non-empty")

        self.task = str(self.task or "").strip() or "default"

        if self.raw_query is None or not str(self.raw_query).strip():
            self.raw_query = self.query_canonical
        else:
            self.raw_query = str(self.raw_query).strip()

        if self.answer_canonical is None or not str(self.answer_canonical).strip():
            self.answer_canonical = self.answer_text
        else:
            self.answer_canonical = str(self.answer_canonical).strip()

        if self.question_type is not None:
            self.question_type = str(self.question_type).strip() or None

        if self.answer_type is not None:
            self.answer_type = str(self.answer_type).strip().upper() or None

        if self.canonical_query_signature is not None:
            self.canonical_query_signature = str(self.canonical_query_signature).strip() or None

        if self.family_id is not None:
            self.family_id = str(self.family_id).strip() or None

        if self.canonical_memory_key is not None:
            self.canonical_memory_key = str(self.canonical_memory_key).strip() or None

        if self.created_at_utc.tzinfo is None:
            raise ValueError("MemoryItem.created_at_utc must be timezone-aware (UTC)")

        if self.ttl_seconds is not None and int(self.ttl_seconds) < 0:
            raise ValueError("MemoryItem.ttl_seconds must be >= 0")

        if self.expires_at_utc is None and self.ttl_seconds is not None:
            self.expires_at_utc = self.created_at_utc + timedelta(seconds=int(self.ttl_seconds))

        if self.expires_at_utc is not None and self.expires_at_utc.tzinfo is None:
            raise ValueError("MemoryItem.expires_at_utc must be timezone-aware (UTC)")

        if not isinstance(self.meta, dict):
            raise TypeError("MemoryItem.meta must be a dict")

        if len(self.meta) > self.MAX_META_KEYS:
            raise ValueError("MemoryItem.meta exceeds max allowed keys")

        if self.chunk_index is not None and int(self.chunk_index) < 0:
            raise ValueError("MemoryItem.chunk_index must be >= 0")

        if self.start_char is not None and int(self.start_char) < 0:
            raise ValueError("MemoryItem.start_char must be >= 0")

        if self.end_char is not None and int(self.end_char) < 0:
            raise ValueError("MemoryItem.end_char must be >= 0")

        if self.start_char is not None and self.end_char is not None:
            if int(self.end_char) < int(self.start_char):
                raise ValueError("MemoryItem.end_char must be >= start_char")

        if self.query_embedding is not None:
            if isinstance(self.query_embedding, list):
                if len(self.query_embedding) > self.MAX_EMBED_DIM:
                    raise ValueError("Embedding too large for configured max dimension")
                self.query_embedding = array("f", [float(x) for x in self.query_embedding])

            elif isinstance(self.query_embedding, array):
                if self.query_embedding.typecode != "f":
                    raise TypeError("query_embedding array must be array('f')")
                if len(self.query_embedding) > self.MAX_EMBED_DIM:
                    raise ValueError("Embedding too large for configured max dimension")

            else:
                raise TypeError("query_embedding must be a list[float] or array('f')")

        if self.embedding_norm is not None and float(self.embedding_norm) < 0.0:
            raise ValueError("MemoryItem.embedding_norm must be >= 0")

        # Backfill commonly-used meta fields to keep older code paths working.
        self.meta.setdefault("task", self.task)
        self.meta.setdefault("question_type", self.question_type)
        self.meta.setdefault("answer_canonical", self.answer_canonical)
        self.meta.setdefault("answer_type", self.answer_type)
        self.meta.setdefault("canonical_query_signature", self.canonical_query_signature)
        self.meta.setdefault("doc_signature", self.doc_signature)
        self.meta.setdefault("source_file", self.source_file)
        self.meta.setdefault("source_id", self.source_id)
        self.meta.setdefault("chunk_index", self.chunk_index)
        self.meta.setdefault("chunk_id", self.chunk_id)
        self.meta.setdefault("family_id", self.family_id)
        self.meta.setdefault("canonical_memory_key", self.canonical_memory_key)
        self.meta.setdefault("is_alias", bool(self.is_alias))

    def is_expired(self, now_utc: Optional[datetime] = None) -> bool:
        """
        Return True if the item has expired relative to now_utc.
        """
        if self.expires_at_utc is None:
            return False
        now = now_utc or datetime.now(timezone.utc)
        return now >= self.expires_at_utc

    def has_same_document(self, mq: MemoryQuery) -> bool:
        """
        Cheap same-document check used by the verifier path.
        """
        query_doc = mq.doc_signature or (mq.context.get("doc_signature") if isinstance(mq.context, dict) else None)
        if not self.doc_signature or not query_doc:
            return False
        return str(self.doc_signature) == str(query_doc)

    def semantic_family_key(self) -> tuple[str, str, str]:
        """
        Stable logical-family key used by manager-side semantic dedupe.
        """
        return (
            str(self.doc_signature or ""),
            str(self.answer_canonical or self.answer_text or ""),
            str(self.canonical_query_signature or self.query_canonical or ""),
        )


@dataclass(frozen=True)
class MemoryHit:
    """
    Retrieved candidate returned by RAM or disk lookup.

    This carries both retrieval information and optional verification
    information so later policy/manager code can decide among:
      - exact reuse
      - semantic direct reuse
      - semantic context only
      - reject
    """
    item: MemoryItem
    source_tier: SourceTier
    match_type: MatchType = MatchType.EXACT

    # Primary retrieval score for this hit
    score: float = 1.0

    # Rank within lexical/semantic retrieval results, if applicable
    semantic_rank: Optional[int] = None

    # Older field kept for backward compatibility with existing code/tests
    bypass_allowed: bool = False

    # Verifier-oriented fields for paraphrase reuse
    verification_outcome: Optional[VerificationOutcome] = None
    same_document: Optional[bool] = None
    task_compatible: Optional[bool] = None
    evidence_supported: Optional[bool] = None
    ambiguous: Optional[bool] = None
    score_margin_vs_next: Optional[float] = None

    # Flexible debug payload for logs / tests / analysis
    debug: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_tier, SourceTier):
            raise TypeError("MemoryHit.source_tier must be a SourceTier")

        if not isinstance(self.match_type, MatchType):
            raise TypeError("MemoryHit.match_type must be a MatchType")

        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError(f"MemoryHit.score must be in [0,1], got {self.score}")

        if self.semantic_rank is not None and int(self.semantic_rank) < 0:
            raise ValueError("MemoryHit.semantic_rank must be >= 0")

        if self.score_margin_vs_next is not None and float(self.score_margin_vs_next) < 0.0:
            raise ValueError("MemoryHit.score_margin_vs_next must be >= 0")

        if self.verification_outcome is not None and not isinstance(
            self.verification_outcome, VerificationOutcome
        ):
            raise TypeError("MemoryHit.verification_outcome must be a VerificationOutcome")

    @property
    def semantic_bypassed(self) -> bool:
        """
        Convenience property for later logging/evaluation code.
        """
        return self.verification_outcome == VerificationOutcome.DIRECT_REUSE
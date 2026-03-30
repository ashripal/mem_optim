from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Mapping, List
from datetime import datetime, timedelta, timezone


# -------------------------
# Enums
# -------------------------

class Scope(str, Enum):
    """Where a memory item is allowed to be used from."""
    SESSION = "session"
    USER = "user"
    COHORT = "cohort"
    GLOBAL = "global"


class SourceTier(str, Enum):
    """Which storage tier produced the hit."""
    RAM = "ram"
    DISK = "disk"


class MatchType(str, Enum):
    """How we matched."""
    EXACT = "exact"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"


# -------------------------
# Helper dataclasses
# -------------------------

@dataclass(frozen=True)
class Provenance:
    """Tracks where an answer came from and under what configuration."""
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
    Quality metadata for gating admission/reuse.
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
    """Tracks usage for promotion/eviction policies."""
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


# -------------------------
# Main dataclasses
# -------------------------

@dataclass(frozen=True)
class MemoryQuery:
    """
    Inputs to MemoryManager for a single user request.
    """
    raw_query: str

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    cohort_id: Optional[str] = None

    task: str = "default"
    context: Dict[str, Any] = field(default_factory=dict)

    prompt_version: str = "v1"
    model_id: str = "mistral-7b-instruct"

    allow_semantic: bool = False
    max_disk_reads: int = 16
    max_ram_reads: int = 64

    doc_signature: Optional[str] = None
    source_file: Optional[str] = None
    source_id: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None
    question_type: Optional[str] = None

    evidence_text: Optional[str] = None
    answer_canonical: Optional[str] = None

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

        if self.chunk_index is not None and self.chunk_index < 0:
            raise ValueError("MemoryQuery.chunk_index must be >= 0 when provided")

        if self.doc_signature is not None and not str(self.doc_signature).strip():
            raise ValueError("MemoryQuery.doc_signature must be non-empty when provided")
        if self.source_file is not None and not str(self.source_file).strip():
            raise ValueError("MemoryQuery.source_file must be non-empty when provided")
        if self.source_id is not None and not str(self.source_id).strip():
            raise ValueError("MemoryQuery.source_id must be non-empty when provided")
        if self.chunk_id is not None and not str(self.chunk_id).strip():
            raise ValueError("MemoryQuery.chunk_id must be non-empty when provided")
        if self.question_type is not None and not str(self.question_type).strip():
            raise ValueError("MemoryQuery.question_type must be non-empty when provided")
        if self.evidence_text is not None and not str(self.evidence_text).strip():
            raise ValueError("MemoryQuery.evidence_text must be non-empty when provided")
        if self.answer_canonical is not None and not str(self.answer_canonical).strip():
            raise ValueError("MemoryQuery.answer_canonical must be non-empty when provided")


@dataclass
class MemoryItem:
    """
    A stored memory record.
    """
    key: str
    scope: Scope
    namespace: str

    query_canonical: str
    context_signature: str

    answer_text: str

    provenance: Provenance
    quality: QualitySignals = field(default_factory=QualitySignals)

    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: Optional[int] = None
    expires_at_utc: Optional[datetime] = None

    stats: AccessStats = field(default_factory=AccessStats)
    meta: Dict[str, Any] = field(default_factory=dict)

    evidence_text: Optional[str] = None
    doc_signature: Optional[str] = None
    source_file: Optional[str] = None
    source_id: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_id: Optional[str] = None
    question_type: Optional[str] = None
    answer_canonical: Optional[str] = None

    query_embedding: Optional[List[float]] = None
    embedding_model_id: Optional[str] = None
    embedding_norm: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("MemoryItem.key must be non-empty")
        if not isinstance(self.scope, Scope):
            raise TypeError("MemoryItem.scope must be a Scope")
        if not self.namespace:
            raise ValueError("MemoryItem.namespace must be non-empty")
        if not self.query_canonical:
            raise ValueError("MemoryItem.query_canonical must be non-empty")
        if not self.context_signature:
            raise ValueError("MemoryItem.context_signature must be non-empty")
        if not self.answer_text:
            raise ValueError("MemoryItem.answer_text must be non-empty")

        if self.created_at_utc.tzinfo is None:
            raise ValueError("MemoryItem.created_at_utc must be timezone-aware (UTC)")

        if self.ttl_seconds is not None and self.ttl_seconds < 0:
            raise ValueError("MemoryItem.ttl_seconds must be >= 0 or None")

        if self.expires_at_utc is None and self.ttl_seconds is not None:
            self.expires_at_utc = self.created_at_utc + timedelta(seconds=self.ttl_seconds)

        if self.expires_at_utc is not None and self.expires_at_utc.tzinfo is None:
            raise ValueError("MemoryItem.expires_at_utc must be timezone-aware (UTC)")

        if self.scope == Scope.SESSION and not self.namespace.startswith("session:"):
            raise ValueError("SESSION scope requires namespace starting with 'session:'")
        if self.scope == Scope.USER and not self.namespace.startswith("user:"):
            raise ValueError("USER scope requires namespace starting with 'user:'")
        if self.scope == Scope.COHORT and not self.namespace.startswith("cohort:"):
            raise ValueError("COHORT scope requires namespace starting with 'cohort:'")
        if self.scope == Scope.GLOBAL and not self.namespace.startswith("global:"):
            raise ValueError("GLOBAL scope requires namespace starting with 'global:'")

        if self.doc_signature is not None and not str(self.doc_signature).strip():
            raise ValueError("MemoryItem.doc_signature must be non-empty when provided")
        if self.source_file is not None and not str(self.source_file).strip():
            raise ValueError("MemoryItem.source_file must be non-empty when provided")
        if self.source_id is not None and not str(self.source_id).strip():
            raise ValueError("MemoryItem.source_id must be non-empty when provided")
        if self.chunk_index is not None and self.chunk_index < 0:
            raise ValueError("MemoryItem.chunk_index must be >= 0 when provided")
        if self.chunk_id is not None and not str(self.chunk_id).strip():
            raise ValueError("MemoryItem.chunk_id must be non-empty when provided")
        if self.question_type is not None and not str(self.question_type).strip():
            raise ValueError("MemoryItem.question_type must be non-empty when provided")
        if self.answer_canonical is not None and not str(self.answer_canonical).strip():
            raise ValueError("MemoryItem.answer_canonical must be non-empty when provided")
        if self.evidence_text is not None and not str(self.evidence_text).strip():
            raise ValueError("MemoryItem.evidence_text must be non-empty when provided")

        if not isinstance(self.meta, dict):
            raise TypeError("MemoryItem.meta must be a dict")

        if self.query_embedding is not None:
            if not isinstance(self.query_embedding, list):
                raise TypeError("MemoryItem.query_embedding must be a list[float] or None")
            if len(self.query_embedding) == 0:
                raise ValueError("MemoryItem.query_embedding cannot be an empty list")
            cleaned_vec: List[float] = []
            for i, value in enumerate(self.query_embedding):
                try:
                    cleaned_vec.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"MemoryItem.query_embedding[{i}] must be numeric, got {value!r}"
                    ) from exc
            self.query_embedding = cleaned_vec

        if self.embedding_model_id is not None and not str(self.embedding_model_id).strip():
            raise ValueError("MemoryItem.embedding_model_id must be non-empty when provided")

        if self.embedding_norm is not None and float(self.embedding_norm) < 0.0:
            raise ValueError("MemoryItem.embedding_norm must be >= 0 or None")

    def is_expired(self, now_utc: Optional[datetime] = None) -> bool:
        if self.expires_at_utc is None:
            return False
        now = now_utc or datetime.now(timezone.utc)
        return now >= self.expires_at_utc


@dataclass(frozen=True)
class MemoryHit:
    """
    Returned by MemoryManager.retrieve() when it finds an item.
    """
    item: MemoryItem
    source_tier: SourceTier
    match_type: MatchType = MatchType.EXACT

    score: float = 1.0

    semantic_rank: Optional[int] = None
    bypass_allowed: bool = False

    debug: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_tier, SourceTier):
            raise TypeError("MemoryHit.source_tier must be a SourceTier")
        if not isinstance(self.match_type, MatchType):
            raise TypeError("MemoryHit.match_type must be a MatchType")
        if not (0.0 <= float(self.score) <= 1.0):
            raise ValueError(f"MemoryHit.score must be in [0,1], got {self.score}")
        if self.semantic_rank is not None and self.semantic_rank < 1:
            raise ValueError("MemoryHit.semantic_rank must be >= 1 when provided")
        if not isinstance(self.debug, Mapping):
            raise TypeError("MemoryHit.debug must be a mapping")
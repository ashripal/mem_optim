from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Mapping, List
from datetime import datetime, timedelta, timezone
from array import array


# -------------------------
# Enums
# -------------------------

class Scope(str, Enum):
    SESSION = "session"
    USER = "user"
    COHORT = "cohort"
    GLOBAL = "global"


class SourceTier(str, Enum):
    RAM = "ram"
    DISK = "disk"


class MatchType(str, Enum):
    EXACT = "exact"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"


# -------------------------
# Helper dataclasses
# -------------------------

@dataclass(frozen=True)
class Provenance:
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


@dataclass
class MemoryItem:
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

    # 🔥 Jetson-optimized embedding storage
    query_embedding: Optional[array] = None
    embedding_model_id: Optional[str] = None
    embedding_norm: Optional[float] = None

    MAX_EMBED_DIM = 1024
    MAX_META_KEYS = 32

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

        # TTL handling
        if self.expires_at_utc is None and self.ttl_seconds is not None:
            self.expires_at_utc = self.created_at_utc + timedelta(seconds=self.ttl_seconds)

        # 🔒 Meta bounding (important for Jetson)
        if not isinstance(self.meta, dict):
            raise TypeError("MemoryItem.meta must be a dict")

        if len(self.meta) > self.MAX_META_KEYS:
            raise ValueError("MemoryItem.meta exceeds max allowed keys")

        # 🔥 Embedding optimization
        if self.query_embedding is not None:
            if isinstance(self.query_embedding, list):
                if len(self.query_embedding) > self.MAX_EMBED_DIM:
                    raise ValueError("Embedding too large for Jetson")

                self.query_embedding = array("f", self.query_embedding)

            elif not isinstance(self.query_embedding, array):
                raise TypeError("query_embedding must be list or array('f')")

    def is_expired(self, now_utc: Optional[datetime] = None) -> bool:
        if self.expires_at_utc is None:
            return False
        now = now_utc or datetime.now(timezone.utc)
        return now >= self.expires_at_utc


@dataclass(frozen=True)
class MemoryHit:
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
# memarch/memory/schema.py
"""
Core, shared data structures for memarch.

Phase 1 focus:
- Deterministic routing (no LLM-based policy)
- Exact-match retrieval remains primary
- Semantic retrieval is optional and assistive/context-only
- Personalization via scopes (SESSION / USER / COHORT / GLOBAL)
"""

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

    # Optional extra provenance fields (safe defaults)
    generator_backend: Optional[str] = None  # e.g., "llama_cpp", "mlx", "remote_http"
    quantization: Optional[str] = None       # e.g., "Q4_K_M"
    context_window: Optional[int] = None     # e.g., 4096


@dataclass
class QualitySignals:
    """
    Quality metadata for gating admission/reuse.

    Phase 1: keep simple. You can attach offline eval scores later.
    """
    # A generic normalized score in [0, 1] if you have one; else None.
    score: Optional[float] = None

    # If you have an explicit success indicator (user feedback, oracle label).
    success: Optional[bool] = None

    # Optional task-specific metrics (e.g., exact_match, rougeL, etc.)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None and not (0.0 <= float(self.score) <= 1.0):
            raise ValueError(f"QualitySignals.score must be in [0,1], got {self.score}")


@dataclass
class AccessStats:
    """Tracks usage for promotion/eviction policies."""
    access_count: int = 0
    last_access_utc: Optional[datetime] = None

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

    Keep this as the single object passed across memory/generator boundaries,
    so pipeline code stays simple and consistent across devices.
    """
    raw_query: str

    # Personalization identifiers
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Optional grouping for cohort-level memory
    cohort_id: Optional[str] = None

    # Task/domain scoping (recommended to avoid cross-task contamination)
    task: str = "default"

    # Context is structured and may include device/tool state, dataset metadata,
    # recent turns, etc. Must remain JSON-serializable if logged or stored.
    context: Dict[str, Any] = field(default_factory=dict)

    # Versioning to prevent stale reuse when templates change
    prompt_version: str = "v1"

    # Model identity used for version scoping (e.g., "mistral-7b-instruct")
    model_id: str = "mistral-7b-instruct"

    # Retrieval knobs
    allow_semantic: bool = False
    max_disk_reads: int = 16
    max_ram_reads: int = 64

    def __post_init__(self) -> None:
        if not self.raw_query or not self.raw_query.strip():
            raise ValueError("MemoryQuery.raw_query must be non-empty")
        if self.max_disk_reads < 0 or self.max_ram_reads < 0:
            raise ValueError("max_disk_reads/max_ram_reads must be >= 0")


@dataclass
class MemoryItem:
    """
    A stored memory record.

    Phase 1:
    - exact-match retrieval remains primary
    - semantic fields are optional and may be absent for older records
    """
    # Stable key (derived from canonical query + scope + namespace +
    # context signature + versioning)
    key: str

    # Scope & namespace determine who can see this memory item
    scope: Scope
    namespace: str  # e.g., "session:<id>", "user:<id>", "cohort:<id>", "global:<task>"

    # Canonical forms for debugging/auditing and collision detection
    query_canonical: str
    context_signature: str

    # The actual stored content
    answer_text: str

    # Provenance & quality for safe reuse
    provenance: Provenance
    quality: QualitySignals = field(default_factory=QualitySignals)

    # Lifespan control
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: Optional[int] = None  # None => no TTL
    expires_at_utc: Optional[datetime] = None

    # Usage stats (for promotion/eviction decisions)
    stats: AccessStats = field(default_factory=AccessStats)

    # Free-form metadata (must remain JSON-serializable if logged/stored)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -------------------------
    # Semantic retrieval fields
    # -------------------------
    # Stored as a plain list for JSON / SQLite compatibility.
    query_embedding: Optional[List[float]] = None

    # Embedding model used to generate the vector.
    embedding_model_id: Optional[str] = None

    # Useful for validation/debugging; especially if normalization behavior changes.
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

        # TTL/expires consistency
        if self.ttl_seconds is not None and self.ttl_seconds < 0:
            raise ValueError("MemoryItem.ttl_seconds must be >= 0 or None")

        if self.expires_at_utc is None and self.ttl_seconds is not None:
            self.expires_at_utc = self.created_at_utc + timedelta(seconds=self.ttl_seconds)

        if self.expires_at_utc is not None and self.expires_at_utc.tzinfo is None:
            raise ValueError("MemoryItem.expires_at_utc must be timezone-aware (UTC)")

        # Scope-specific sanity checks
        if self.scope == Scope.SESSION and not self.namespace.startswith("session:"):
            raise ValueError("SESSION scope requires namespace starting with 'session:'")
        if self.scope == Scope.USER and not self.namespace.startswith("user:"):
            raise ValueError("USER scope requires namespace starting with 'user:'")
        if self.scope == Scope.COHORT and not self.namespace.startswith("cohort:"):
            raise ValueError("COHORT scope requires namespace starting with 'cohort:'")
        if self.scope == Scope.GLOBAL and not self.namespace.startswith("global:"):
            raise ValueError("GLOBAL scope requires namespace starting with 'global:'")

        # Semantic field sanity
        if self.query_embedding is not None:
            if not isinstance(self.query_embedding, list):
                raise TypeError("MemoryItem.query_embedding must be a list[float] or None")
            if len(self.query_embedding) == 0:
                raise ValueError("MemoryItem.query_embedding cannot be an empty list")
            for i, value in enumerate(self.query_embedding):
                try:
                    float(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"MemoryItem.query_embedding[{i}] must be numeric, got {value!r}"
                    ) from exc

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

    Phase 1 notes:
    - EXACT hits can still return directly
    - SEMANTIC hits are primarily intended for context assistance
    - bypass_allowed is present for forward compatibility, but can remain False
    """
    item: MemoryItem
    source_tier: SourceTier
    match_type: MatchType = MatchType.EXACT

    # For EXACT this is typically 1.0. For SEMANTIC it is cosine similarity.
    score: float = 1.0

    # Semantic retrieval metadata
    semantic_rank: Optional[int] = None
    bypass_allowed: bool = False

    # Optional debug info (e.g., applied thresholds, filter reasons)
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
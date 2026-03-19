# memarch/memory/manager.py
"""
MemoryManager: deterministic multi-tier personalization memory.

Phase 1 behavior:
- Exact-match retrieval remains primary
- Semantic retrieval is optional and occurs only after exact-match miss
- Retrieval order:
    1. Exact RAM
    2. Exact DISK
    3. Semantic RAM / DISK candidate search
    4. Generator fallback
- Tiers:
    Tier 1: RAM (RamStoreLRU)
    Tier 2: DISK (DiskStoreSQLite)
- On DISK exact hit, promote to RAM
- On semantic hit in Phase 1, use as context for generation by default
- On miss, call generator, then store according to admission policy

Design principles:
- Single entry point for memory decisions
- Strict scoping to prevent cross-user contamination
- Deterministic, bounded operations
- Keep exact-match logic stable while adding semantic retrieval additively
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from memarch.memory.schema import (
    MatchType,
    MemoryHit,
    MemoryItem,
    MemoryQuery,
    Provenance,
    QualitySignals,
    Scope,
    SourceTier,
)
from memarch.memory.namespace import resolve_namespaces
from memarch.memory.policy import (
    RetrievalPolicy,
    accept_item,
    budget_from_query,
    default_retrieval_policy,
    make_hit_debug,
    score_exact_hit,
    semantic_candidate_allowed,
    semantic_decision,
)
from memarch.memory.admission import (
    AdmissionPolicy,
    choose_ttl_seconds,
    decide_store_scopes,
    default_admission_policy,
    should_store,
)
from memarch.memory.embed_index import EmbedIndexLRU, SemanticCandidate
from memarch.models.embedder import Embedder
from memarch.utils.text import canonicalize, context_signature, make_key


# -------------------------
# Store + Generator protocols
# -------------------------

class MemoryStore(Protocol):
    def get(self, namespace: str, key: str) -> Optional[MemoryItem]: ...
    def put(self, namespace: str, key: str, item: MemoryItem) -> None: ...
    def delete(self, namespace: str, key: str) -> None: ...
    def stats(self) -> Any: ...


class IterableMemoryStore(MemoryStore, Protocol):
    def iter_namespace(self, namespace: str) -> Iterable[MemoryItem]: ...


class Generator(Protocol):
    """
    Generator interface.

    Phase 1:
    - exact hit may return directly
    - semantic hit is typically passed into generator as retrieved context
    """
    def generate(
        self,
        mq: MemoryQuery,
        retrieved: Optional[MemoryHit] = None,
    ) -> Tuple[str, Provenance, QualitySignals]:
        ...


# -------------------------
# Manager config
# -------------------------

@dataclass(frozen=True)
class MemoryManagerConfig:
    retrieval_policy: RetrievalPolicy = default_retrieval_policy()
    admission_policy: AdmissionPolicy = default_admission_policy()

    # Promote exact disk hits to RAM for faster repeat access
    promote_disk_hits_to_ram: bool = True

    # If True, return exact hits directly.
    # Semantic hits may still be routed through generator depending on policy.
    return_memory_directly: bool = True

    # Semantic components
    embedder: Optional[Embedder] = None
    embed_index: Optional[EmbedIndexLRU] = None


# -------------------------
# MemoryManager
# -------------------------

class MemoryManager:
    def __init__(
        self,
        *,
        ram: MemoryStore,
        disk: MemoryStore,
        cfg: Optional[MemoryManagerConfig] = None,
    ) -> None:
        self._ram = ram
        self._disk = disk
        self._cfg = cfg or MemoryManagerConfig()
        self._embedder = self._cfg.embedder
        self._embed_index = self._cfg.embed_index or EmbedIndexLRU(max_entries=10_000)

    # -------------------------
    # Exact retrieval
    # -------------------------

    def _retrieve_exact(
        self,
        mq: MemoryQuery,
        *,
        now: datetime,
        ctx_sig: str,
    ) -> Optional[MemoryHit]:
        """
        Attempt exact-match retrieval across scopes and tiers.

        Exact-match behavior is intentionally kept unchanged.
        """
        pol = self._cfg.retrieval_policy
        budget = budget_from_query(mq)

        q_can = canonicalize(mq.raw_query)
        ram_reads = 0
        disk_reads = 0

        for rn in resolve_namespaces(mq, scope_order=pol.scope_order, include_missing=False):
            scope = rn.scope
            ns = rn.namespace

            key = make_key(
                scope=scope.value,
                namespace=ns,
                task=mq.task,
                model_id=mq.model_id,
                prompt_version=mq.prompt_version,
                query_canonical=q_can,
                context_sig=ctx_sig,
            )

            # 1) RAM exact
            if ram_reads < budget.max_ram_reads:
                ram_reads += 1
                item = self._ram.get(ns, key)
                if item is not None:
                    ok, dbg = accept_item(
                        mq,
                        item,
                        policy=pol,
                        now_utc=now,
                        query_context_signature=ctx_sig,
                    )
                    if ok:
                        return MemoryHit(
                            item=item,
                            source_tier=SourceTier.RAM,
                            match_type=MatchType.EXACT,
                            score=score_exact_hit(),
                            debug=make_hit_debug(
                                scope=scope,
                                namespace=ns,
                                source="ram",
                                accepted_reason=dbg.get("reason", "accepted"),
                                extra={"ram_reads": ram_reads, "disk_reads": disk_reads},
                            ),
                        )

            # 2) DISK exact
            if disk_reads < budget.max_disk_reads:
                disk_reads += 1
                item = self._disk.get(ns, key)
                if item is not None:
                    ok, dbg = accept_item(
                        mq,
                        item,
                        policy=pol,
                        now_utc=now,
                        query_context_signature=ctx_sig,
                    )
                    if ok:
                        hit = MemoryHit(
                            item=item,
                            source_tier=SourceTier.DISK,
                            match_type=MatchType.EXACT,
                            score=score_exact_hit(),
                            debug=make_hit_debug(
                                scope=scope,
                                namespace=ns,
                                source="disk",
                                accepted_reason=dbg.get("reason", "accepted"),
                                extra={"ram_reads": ram_reads, "disk_reads": disk_reads},
                            ),
                        )
                        if self._cfg.promote_disk_hits_to_ram:
                            try:
                                self._ram.put(ns, key, item)
                            except Exception:
                                pass
                        return hit

        return None

    # -------------------------
    # Semantic retrieval
    # -------------------------

    def _iter_store_namespace(self, store: MemoryStore, namespace: str) -> Iterable[MemoryItem]:
        if hasattr(store, "iter_namespace"):
            return getattr(store, "iter_namespace")(namespace)
        return ()

    def _build_semantic_candidates(
        self,
        mq: MemoryQuery,
        *,
        now: datetime,
        ctx_sig: str,
    ) -> List[SemanticCandidate[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]]]:
        """
        Collect semantic candidates from RAM and DISK after exact retrieval fails.

        Payload shape:
          (source_tier, scope, namespace, item, filter_debug)
        """
        pol = self._cfg.retrieval_policy
        budget = budget_from_query(mq)

        candidates: List[SemanticCandidate[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]]] = []
        ram_reads = 0
        disk_reads = 0

        for rn in resolve_namespaces(mq, scope_order=pol.scope_order, include_missing=False):
            scope = rn.scope
            ns = rn.namespace

            # RAM semantic scan
            if ram_reads < budget.max_ram_reads:
                ram_reads += 1
                for item in self._iter_store_namespace(self._ram, ns):
                    ok, dbg = semantic_candidate_allowed(
                        mq,
                        item,
                        policy=pol,
                        now_utc=now,
                        query_context_signature=ctx_sig,
                    )
                    if not ok:
                        continue
                    if item.query_embedding is None:
                        continue

                    candidates.append(
                        SemanticCandidate(
                            payload=(SourceTier.RAM, scope, ns, item, dbg),
                            vector=item.query_embedding,
                        )
                    )

            # DISK semantic scan
            if disk_reads < budget.max_disk_reads:
                disk_reads += 1
                for item in self._iter_store_namespace(self._disk, ns):
                    ok, dbg = semantic_candidate_allowed(
                        mq,
                        item,
                        policy=pol,
                        now_utc=now,
                        query_context_signature=ctx_sig,
                    )
                    if not ok:
                        continue
                    if item.query_embedding is None:
                        continue

                    candidates.append(
                        SemanticCandidate(
                            payload=(SourceTier.DISK, scope, ns, item, dbg),
                            vector=item.query_embedding,
                        )
                    )

        return candidates

    def _retrieve_semantic(
        self,
        mq: MemoryQuery,
        *,
        now: datetime,
        ctx_sig: str,
    ) -> Optional[MemoryHit]:
        """
        Attempt semantic retrieval after exact retrieval fails.

        Phase 1 behavior:
        - only enabled if query + policy allow it
        - usually used for generator context assistance
        - direct bypass only if policy explicitly allows it
        """
        pol = self._cfg.retrieval_policy
        budget = budget_from_query(mq)

        if not pol.semantic_enabled:
            return None
        if not budget.allow_semantic:
            return None
        if self._embedder is None:
            return None

        query_vec = self._embedder.embed(mq.raw_query)
        if not query_vec:
            return None

        candidates = self._build_semantic_candidates(
            mq,
            now=now,
            ctx_sig=ctx_sig,
        )
        if not candidates:
            return None

        ranked = self._embed_index.search_candidates(
            query_vector=query_vec,
            candidates=candidates,
            top_k=pol.max_semantic_candidates,
            min_score=pol.semantic_threshold_context,
        )
        if not ranked:
            return None

        payload, score, rank = ranked[0]
        source_tier, scope, ns, item, filter_dbg = payload

        decision, decision_dbg = semantic_decision(
            score=score,
            item=item,
            policy=pol,
            query_context_signature=ctx_sig,
        )
        if decision == "ignore":
            return None

        bypass_allowed = decision == "bypass"

        hit = MemoryHit(
            item=item,
            source_tier=source_tier,
            match_type=MatchType.SEMANTIC,
            score=float(score),
            semantic_rank=rank,
            bypass_allowed=bypass_allowed,
            debug=make_hit_debug(
                scope=scope,
                namespace=ns,
                source="semantic_ram" if source_tier == SourceTier.RAM else "semantic_disk",
                accepted_reason=decision_dbg.get("reason", "semantic_context"),
                extra={
                    "semantic_candidate_rank": rank,
                    "semantic_score": float(score),
                    "semantic_bypassed": bypass_allowed,
                    "filter_debug": filter_dbg,
                    **decision_dbg,
                },
            ),
        )

        # Promote semantic disk hit to RAM as a best-effort cache warmup.
        if source_tier == SourceTier.DISK and self._cfg.promote_disk_hits_to_ram:
            try:
                self._ram.put(ns, item.key, item)
            except Exception:
                pass

        return hit

    # -------------------------
    # Public retrieval API
    # -------------------------

    def retrieve(self, mq: MemoryQuery) -> Optional[MemoryHit]:
        """
        Retrieval cascade:
          1. exact RAM
          2. exact DISK
          3. semantic RAM/DISK
        """
        now = datetime.now(timezone.utc)
        ctx_sig = context_signature(mq.context)

        exact_hit = self._retrieve_exact(mq, now=now, ctx_sig=ctx_sig)
        if exact_hit is not None:
            return exact_hit

        semantic_hit = self._retrieve_semantic(mq, now=now, ctx_sig=ctx_sig)
        if semantic_hit is not None:
            return semantic_hit

        return None

    # -------------------------
    # Store path
    # -------------------------

    def _make_embedding_fields(self, mq: MemoryQuery) -> Tuple[Optional[List[float]], Optional[str], Optional[float]]:
        """
        Compute embedding-related fields for a stored MemoryItem.

        Best-effort only. Storage should not fail just because embedding generation fails.
        """
        if self._embedder is None:
            return None, None, None

        try:
            vec = self._embedder.embed(mq.raw_query)
            if not vec:
                return None, None, None

            model_id = None
            if hasattr(self._embedder, "cfg"):
                model_id = getattr(self._embedder.cfg, "model_id", None)

            norm = None
            if hasattr(self._embedder, "embedding_norm"):
                norm = self._embedder.embedding_norm(vec)

            return list(vec), model_id, norm
        except Exception:
            return None, None, None

    def store(
        self,
        mq: MemoryQuery,
        *,
        answer_text: str,
        provenance: Provenance,
        quality: Optional[QualitySignals] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a generated answer into memory according to admission policy.

        Returns:
          debug dict describing what was stored (or why not).
        """
        ap = self._cfg.admission_policy
        meta = dict(meta or {})
        q_can = canonicalize(mq.raw_query)
        ctx_sig = context_signature(mq.context)
        now = datetime.now(timezone.utc)

        # Semantic metadata for safer filtering later
        meta.setdefault("task", mq.task)
        if "doc_signature" in mq.context:
            meta.setdefault("doc_signature", mq.context.get("doc_signature"))

        query_embedding, embedding_model_id, embedding_norm = self._make_embedding_fields(mq)

        stored: Dict[str, Any] = {"stored": [], "skipped": []}
        quality = quality or QualitySignals()

        for scope in decide_store_scopes(mq, ap):
            try:
                ns = resolve_namespaces(
                    mq,
                    scope_order=[scope],
                    include_missing=False,
                )[0].namespace
            except Exception:
                stored["skipped"].append({"scope": scope.value, "reason": "missing_namespace"})
                continue

            ok, dbg = should_store(mq, answer_text, quality, scope=scope, policy=ap)
            if not ok:
                stored["skipped"].append({"scope": scope.value, **dbg})
                continue

            ttl_s = choose_ttl_seconds(scope, ap)
            key = make_key(
                scope=scope.value,
                namespace=ns,
                task=mq.task,
                model_id=mq.model_id,
                prompt_version=mq.prompt_version,
                query_canonical=q_can,
                context_sig=ctx_sig,
            )

            item = MemoryItem(
                key=key,
                scope=scope,
                namespace=ns,
                query_canonical=q_can,
                context_signature=ctx_sig,
                answer_text=answer_text,
                provenance=provenance,
                quality=quality,
                created_at_utc=now,
                ttl_seconds=ttl_s,
                meta=meta,
                query_embedding=query_embedding,
                embedding_model_id=embedding_model_id,
                embedding_norm=embedding_norm,
            )

            # Disk is the persistent source of truth
            try:
                self._disk.put(ns, key, item)
                disk_ok = True
            except Exception as e:
                disk_ok = False
                stored["skipped"].append(
                    {
                        "scope": scope.value,
                        "reason": "disk_write_failed",
                        "error": str(e),
                    }
                )

            # RAM is best-effort
            try:
                self._ram.put(ns, key, item)
                ram_ok = True
            except Exception:
                ram_ok = False

            # Best-effort embed cache warmup
            if query_embedding is not None:
                try:
                    self._embed_index.put(ns, key, query_embedding)
                except Exception:
                    pass

            stored["stored"].append(
                {
                    "scope": scope.value,
                    "namespace": ns,
                    "key": key,
                    "ttl_seconds": ttl_s,
                    "ram": ram_ok,
                    "disk": disk_ok,
                    "has_embedding": query_embedding is not None,
                    "embedding_model_id": embedding_model_id,
                }
            )

        return stored

    # -------------------------
    # Main entrypoint
    # -------------------------

    def answer(self, mq: MemoryQuery, generator: Generator) -> Tuple[str, Dict[str, Any]]:
        """
        Primary entrypoint used by pipeline:
        - retrieve from memory
        - exact hit may return directly
        - semantic hit may either:
            * return directly if policy allows bypass
            * be passed into generator as context
        - miss -> generate and store
        """
        hit = self.retrieve(mq)

        if (
            hit is not None
            and hit.match_type == MatchType.EXACT
            and self._cfg.return_memory_directly
        ):
            return hit.item.answer_text, {
                "used_memory": True,
                "generated": False,
                "source_tier": hit.source_tier.value,
                "match_type": hit.match_type.value,
                "score": hit.score,
                "semantic_used": False,
                "semantic_bypassed": False,
                "semantic_candidate_rank": None,
                "hit": {**dict(hit.debug)},
                "stored": False,
                "stored_scopes": [],
                "memory_lookup_ms": 0.0,
                "generation_ms_est": 0.0,
            }

        if (
            hit is not None
            and hit.match_type == MatchType.SEMANTIC
            and self._cfg.return_memory_directly
            and hit.bypass_allowed
        ):
            return hit.item.answer_text, {
                "used_memory": True,
                "generated": False,
                "source_tier": hit.source_tier.value,
                "match_type": hit.match_type.value,
                "score": hit.score,
                "semantic_used": True,
                "semantic_bypassed": True,
                "semantic_candidate_rank": hit.semantic_rank,
                "hit": {**dict(hit.debug)},
                "stored": False,
                "stored_scopes": [],
                "memory_lookup_ms": 0.0,
                "generation_ms_est": 0.0,
            }

        # Either:
        # - no hit
        # - semantic context hit (Phase 1 default)
        # - future mode where regeneration despite hit is desired
        retrieved_for_generation = hit if hit is not None and hit.match_type == MatchType.SEMANTIC else None

        answer_text, provenance, quality = generator.generate(
            mq,
            retrieved=retrieved_for_generation,
        )

        store_dbg = self.store(
            mq,
            answer_text=answer_text,
            provenance=provenance,
            quality=quality,
            meta={
                "used_memory_context": retrieved_for_generation is not None,
                "memory_context_source": retrieved_for_generation.source_tier.value if retrieved_for_generation else None,
                "memory_context_match_type": retrieved_for_generation.match_type.value if retrieved_for_generation else None,
                "semantic_score": retrieved_for_generation.score if retrieved_for_generation else None,
            },
        )

        return answer_text, {
            "used_memory": False,
            "generated": True,
            "source_tier": "compute",
            "match_type": None,
            "score": None,
            "semantic_used": retrieved_for_generation is not None,
            "semantic_bypassed": False,
            "semantic_candidate_rank": (
                retrieved_for_generation.semantic_rank if retrieved_for_generation else None
            ),
            "hit_before_generate": {
                "present": hit is not None,
                "source_tier": hit.source_tier.value if hit else None,
                "match_type": hit.match_type.value if hit else None,
                "score": hit.score if hit else None,
            },
            "stored": len(store_dbg.get("stored", [])) > 0,
            "stored_scopes": [x["scope"] for x in store_dbg.get("stored", [])],
            "store": store_dbg,
        }

    def stats(self) -> Dict[str, Any]:
        """Return combined stats from RAM, DISK, and embedding cache."""
        return {
            "ram": getattr(self._ram, "stats")() if callable(getattr(self._ram, "stats", None)) else None,
            "disk": getattr(self._disk, "stats")() if callable(getattr(self._disk, "stats", None)) else None,
            "embed_index": self._embed_index.stats() if self._embed_index is not None else None,
        }
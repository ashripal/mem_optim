# memarch/memory/manager.py
"""
MemoryManager: deterministic multi-tier personalization memory.

Phase 1 behavior:
- Exact-match only (semantic disabled by default; hooks exist via policy flags)
- Retrieval order: SESSION -> USER -> COHORT -> GLOBAL (unless overridden)
- Tiers:
    Tier 1: RAM (RamStoreLRU)
    Tier 2: DISK (DiskStoreSQLite)
- On DISK hit, promote to RAM (same namespace/key)
- On miss, call generator (provided by models/generator.py), then store according to admission policy

Design principles:
- Single entry point for memory decisions (committee-friendly, easy to test)
- Strict scoping to prevent cross-user contamination
- Deterministic, bounded operations (budgeted RAM/DISK reads)

This module should remain small and readable. Complex heuristics belong in policy/admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, Tuple

from memarch.memory.schema import (
    MemoryHit,
    MemoryItem,
    MemoryQuery,
    Provenance,
    QualitySignals,
    MatchType,
    Scope,
    SourceTier,
)
from memarch.memory.namespace import resolve_namespaces
from memarch.memory.policy import (
    BudgetPolicy,
    RetrievalPolicy,
    accept_item,
    budget_from_query,
    default_retrieval_policy,
    make_hit_debug,
    score_exact_hit,
)
from memarch.memory.admission import (
    AdmissionPolicy,
    choose_ttl_seconds,
    decide_store_scopes,
    default_admission_policy,
    should_store,
)
from memarch.utils.text import canonicalize, context_signature, make_key


# -------------------------
# Store + Generator protocols
# -------------------------

class MemoryStore(Protocol):
    def get(self, namespace: str, key: str) -> Optional[MemoryItem]: ...
    def put(self, namespace: str, key: str, item: MemoryItem) -> None: ...
    def delete(self, namespace: str, key: str) -> None: ...
    def stats(self) -> Any: ...


class Generator(Protocol):
    """
    Generator interface.

    Phase 1: generator is invoked only on memory miss.
    In later phases you might pass MemoryHit into generator to blend memory + dataset context.
    """
    def generate(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> Tuple[str, Provenance, QualitySignals]:
        ...


# -------------------------
# Manager config
# -------------------------

@dataclass(frozen=True)
class MemoryManagerConfig:
    retrieval_policy: RetrievalPolicy = default_retrieval_policy()
    admission_policy: AdmissionPolicy = default_admission_policy()

    # Promote disk hits to RAM for faster repeat access
    promote_disk_hits_to_ram: bool = True

    # If True, return memory hit answer directly.
    # If False, you can still call generator with retrieved context (not typical in Phase 1).
    return_memory_directly: bool = True


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

    def retrieve(self, mq: MemoryQuery) -> Optional[MemoryHit]:
        """
        Attempt exact-match retrieval across scopes and tiers.

        Returns:
          MemoryHit if accepted by policy, else None.
        """
        pol = self._cfg.retrieval_policy
        budget = budget_from_query(mq)
        now = datetime.now(timezone.utc)

        q_can = canonicalize(mq.raw_query)
        ctx_sig = context_signature(mq.context)

        ram_reads = 0
        disk_reads = 0

        # Search namespaces in personalization-first order
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

            # 1) RAM
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

            # 2) DISK
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
                        # Promote to RAM
                        if self._cfg.promote_disk_hits_to_ram:
                            try:
                                self._ram.put(ns, key, item)
                            except Exception:
                                # Best-effort promotion; do not fail retrieval.
                                pass
                        return hit

        return None

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
        meta = meta or {}
        q_can = canonicalize(mq.raw_query)
        ctx_sig = context_signature(mq.context)
        now = datetime.now(timezone.utc)

        stored: Dict[str, Any] = {"stored": [], "skipped": []}
        quality = quality or QualitySignals()

        for scope in decide_store_scopes(mq, ap):
            # Resolve namespace for this scope (skip if missing IDs)
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
                # expires_at_utc is computed in MemoryItem.__post_init__ if ttl provided
                meta=meta,
            )

            # Write-through: RAM + DISK (best effort RAM; disk is source of truth for persistence)
            try:
                self._disk.put(ns, key, item)
                disk_ok = True
            except Exception as e:
                disk_ok = False
                stored["skipped"].append({"scope": scope.value, "reason": "disk_write_failed", "error": str(e)})

            # Even if disk fails, we can still keep it in RAM for immediate benefit
            try:
                self._ram.put(ns, key, item)
                ram_ok = True
            except Exception:
                ram_ok = False

            stored["stored"].append(
                {
                    "scope": scope.value,
                    "namespace": ns,
                    "key": key,
                    "ttl_seconds": ttl_s,
                    "ram": ram_ok,
                    "disk": disk_ok,
                }
            )

        return stored

    def answer(self, mq: MemoryQuery, generator: Generator) -> Tuple[str, Dict[str, Any]]:
        """
        Primary entrypoint used by pipeline:
          - retrieve from memory
          - if hit and configured to return directly -> return answer
          - else generate and store

        Returns:
          (answer_text, metadata)
        """
        hit = self.retrieve(mq)
        if hit is not None and self._cfg.return_memory_directly:
            return hit.item.answer_text, {
                "used_memory": True,
                "hit": {
                    "source_tier": hit.source_tier.value,
                    "match_type": hit.match_type.value,
                    "score": hit.score,
                    **dict(hit.debug),
                },
            }

        # Miss -> generate
        answer_text, provenance, quality = generator.generate(mq, retrieved=hit)

        store_dbg = self.store(
            mq,
            answer_text=answer_text,
            provenance=provenance,
            quality=quality,
            meta={
                "used_memory_context": hit is not None,
                "memory_context_source": hit.source_tier.value if hit else None,
            },
        )

        return answer_text, {
            "used_memory": False,
            "generated": True,
            "hit_before_generate": {
                "present": hit is not None,
                "source_tier": hit.source_tier.value if hit else None,
                "match_type": hit.match_type.value if hit else None,
            },
            "store": store_dbg,
        }

    def stats(self) -> Dict[str, Any]:
        """Return combined stats from RAM and DISK stores."""
        return {
            "ram": getattr(self._ram, "stats")() if callable(getattr(self._ram, "stats", None)) else None,
            "disk": getattr(self._disk, "stats")() if callable(getattr(self._disk, "stats", None)) else None,
        }
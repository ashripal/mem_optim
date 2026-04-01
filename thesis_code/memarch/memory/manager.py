from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from memarch.memory.admission import (
    AdmissionPolicy,
    choose_ttl_seconds,
    decide_store_scopes,
    default_admission_policy,
    should_store,
)
from memarch.memory.embed_index import EmbedIndexLRU, SemanticCandidate
from memarch.memory.namespace import resolve_namespaces
from memarch.memory.policy import (
    RetrievalPolicy,
    accept_item,
    budget_from_query,
    default_retrieval_policy,
    document_relation,
    make_hit_debug,
    score_exact_hit,
    semantic_candidate_allowed,
    semantic_decision,
    same_document,
)
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
from memarch.memory.similarity import lexical_score
from memarch.models.embedder import Embedder
from memarch.utils.text import (
    canonicalize,
    context_signature,
    make_key,
    normalize_for_lookup,
    tokenize_lexical,
)


class MemoryStore(Protocol):
    def get(self, namespace: str, key: str) -> Optional[MemoryItem]: ...
    def put(self, namespace: str, key: str, item: MemoryItem) -> None: ...
    def delete(self, namespace: str, key: str) -> None: ...
    def stats(self) -> Any: ...


class IterableMemoryStore(MemoryStore, Protocol):
    def iter_namespace(self, namespace: str) -> Iterable[MemoryItem]: ...


class Generator(Protocol):
    def generate(
        self,
        mq: MemoryQuery,
        retrieved: Optional[MemoryHit] = None,
    ) -> Tuple[str, Provenance, QualitySignals]:
        ...


@dataclass(frozen=True)
class MemoryManagerConfig:
    retrieval_policy: RetrievalPolicy = field(default_factory=default_retrieval_policy)
    admission_policy: AdmissionPolicy = field(default_factory=default_admission_policy)

    promote_disk_hits_to_ram: bool = True
    return_memory_directly: bool = True

    lexical_enabled: bool = False
    lexical_context_threshold: float = 0.55
    lexical_direct_threshold: float = 0.90
    lexical_top_k: int = 3
    prefer_same_source: bool = True
    safe_direct_reuse_tasks: List[str] = field(default_factory=lambda: ["trec"])

    embedder: Optional[Embedder] = None
    embed_index: Optional[EmbedIndexLRU] = None

    enable_storage: bool = True
    store_in_ram: bool = True
    store_on_disk: bool = True


class MemoryManager:
    _TREC_LABELS = {"ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"}

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

    def _lexical_enabled(self) -> bool:
        pol = self._cfg.retrieval_policy
        policy_enabled = bool(getattr(pol, "lexical_enabled", False))
        return bool(self._cfg.lexical_enabled or policy_enabled)

    def _lexical_context_threshold(self) -> float:
        pol = self._cfg.retrieval_policy
        return float(
            getattr(pol, "lexical_threshold_context", self._cfg.lexical_context_threshold)
        )

    def _lexical_direct_threshold(self) -> float:
        pol = self._cfg.retrieval_policy
        return float(
            getattr(pol, "lexical_threshold_bypass", self._cfg.lexical_direct_threshold)
        )

    def _lexical_top_k(self) -> int:
        pol = self._cfg.retrieval_policy
        return int(getattr(pol, "lexical_top_k", self._cfg.lexical_top_k))

    def _prefer_same_source(self) -> bool:
        pol = self._cfg.retrieval_policy
        return bool(getattr(pol, "prefer_same_source", self._cfg.prefer_same_source))

    def _safe_direct_reuse_tasks(self) -> List[str]:
        pol = self._cfg.retrieval_policy
        tasks = getattr(pol, "safe_direct_reuse_tasks", self._cfg.safe_direct_reuse_tasks)
        if not tasks:
            return []
        return [str(t).strip().lower() for t in tasks if str(t).strip()]

    def _query_doc_signature(self, mq: MemoryQuery) -> Optional[str]:
        if getattr(mq, "doc_signature", None) is not None:
            return mq.doc_signature
        value = (mq.context or {}).get("doc_signature")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _query_source_file(self, mq: MemoryQuery) -> Optional[str]:
        if getattr(mq, "source_file", None) is not None:
            return mq.source_file
        value = (mq.context or {}).get("source_file")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _query_source_id(self, mq: MemoryQuery) -> Optional[str]:
        if getattr(mq, "source_id", None) is not None:
            text = str(mq.source_id).strip()
            return text or None
        value = (mq.context or {}).get("source_id")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _query_evidence_text(self, mq: MemoryQuery) -> Optional[str]:
        if getattr(mq, "evidence_text", None) is not None:
            text = str(mq.evidence_text).strip()
            return text or None
        value = (mq.context or {}).get("evidence_text")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _query_chunk_index(self, mq: MemoryQuery) -> Optional[int]:
        if getattr(mq, "chunk_index", None) is not None:
            return mq.chunk_index
        value = (mq.context or {}).get("chunk_index")
        if value is None:
            return None
        try:
            idx = int(value)
            return idx if idx >= 0 else None
        except (TypeError, ValueError):
            return None

    def _query_chunk_id(self, mq: MemoryQuery) -> Optional[str]:
        if getattr(mq, "chunk_id", None) is not None:
            text = str(mq.chunk_id).strip()
            return text or None
        value = (mq.context or {}).get("chunk_id")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _query_question_type(self, mq: MemoryQuery) -> Optional[str]:
        if getattr(mq, "question_type", None) is not None:
            text = str(mq.question_type).strip()
            return text or None
        value = (mq.context or {}).get("question_type")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _query_answer_canonical(self, mq: MemoryQuery) -> Optional[str]:
        if getattr(mq, "answer_canonical", None) is not None:
            text = str(mq.answer_canonical).strip()
            return text or None
        value = (mq.context or {}).get("answer_canonical")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _query_norm(self, mq: MemoryQuery) -> str:
        return normalize_for_lookup(mq.raw_query)

    def _query_tokens(self, mq: MemoryQuery) -> List[str]:
        return tokenize_lexical(mq.raw_query)

    def _item_query_norm(self, item: MemoryItem) -> str:
        raw = getattr(item, "query_canonical", None)
        return normalize_for_lookup(raw or "")

    def _item_query_tokens(self, item: MemoryItem) -> List[str]:
        raw = getattr(item, "query_canonical", None)
        return tokenize_lexical(raw or "")

    def _item_doc_signature(self, item: MemoryItem) -> Optional[str]:
        value = getattr(item, "doc_signature", None)
        if value:
            text = str(value).strip()
            if text:
                return text
        value = item.meta.get("doc_signature")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _item_source_file(self, item: MemoryItem) -> Optional[str]:
        value = getattr(item, "source_file", None)
        if value:
            text = str(value).strip()
            if text:
                return text
        value = item.meta.get("source_file")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _item_source_id(self, item: MemoryItem) -> Optional[str]:
        value = getattr(item, "source_id", None)
        if value:
            text = str(value).strip()
            if text:
                return text
        value = item.meta.get("source_id")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _item_question_type(self, item: MemoryItem) -> Optional[str]:
        value = getattr(item, "question_type", None)
        if value:
            text = str(value).strip()
            if text:
                return text
        value = item.meta.get("question_type")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _item_answer_canonical(self, item: MemoryItem) -> Optional[str]:
        value = getattr(item, "answer_canonical", None)
        if value:
            text = str(value).strip()
            if text:
                return text
        value = item.meta.get("answer_canonical")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _is_item_expired(self, item: MemoryItem, *, now: datetime) -> bool:
        expires = getattr(item, "expires_at_utc", None)
        if expires is None:
            return False
        try:
            return bool(expires <= now)
        except Exception:
            return False

    def _lexical_candidate_allowed(
        self,
        mq: MemoryQuery,
        item: MemoryItem,
        *,
        now: datetime,
    ) -> Tuple[bool, Dict[str, Any]]:
        if self._is_item_expired(item, now=now):
            return False, {"reason": "expired"}

        item_task = str(item.meta.get("task") or "").strip()
        if item_task and item_task != mq.task:
            return False, {"reason": "task_mismatch"}

        q_norm = self._query_norm(mq)
        i_norm = self._item_query_norm(item)
        if not q_norm or not i_norm:
            return False, {"reason": "empty_query"}

        same_doc = same_document(mq, item)
        same_src = False
        query_src = self._query_source_file(mq)
        item_src = self._item_source_file(item)
        if query_src and item_src and query_src == item_src:
            same_src = True

        dbg = {
            "reason": "accepted",
            "same_document": same_doc,
            "same_source": same_src,
            "document_relation": document_relation(mq, item),
            "query_norm": q_norm,
            "item_norm": i_norm,
        }
        return True, dbg

    def _merged_store_meta(
        self,
        mq: MemoryQuery,
        *,
        answer_text: str,
        incoming_meta: Optional[Dict[str, Any]] = None,
        raw_generated_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        meta = dict(incoming_meta or {})
        meta.setdefault("task", mq.task)
        meta.setdefault("doc_signature", self._query_doc_signature(mq))
        meta.setdefault("source_file", self._query_source_file(mq))
        meta.setdefault("source_id", self._query_source_id(mq))
        meta.setdefault("chunk_index", self._query_chunk_index(mq))
        meta.setdefault("chunk_id", self._query_chunk_id(mq))
        meta.setdefault("question_type", self._query_question_type(mq))
        meta.setdefault("evidence_text", self._query_evidence_text(mq))
        meta.setdefault("answer_canonical", self._query_answer_canonical(mq))
        meta.setdefault("answer_length_chars", len(answer_text or ""))
        if raw_generated_answer is not None:
            meta.setdefault("raw_generated_answer", raw_generated_answer)
        return meta

    def _hit_debug_summary(self, hit: Optional[MemoryHit]) -> Optional[Dict[str, Any]]:
        if hit is None:
            return None
        item = hit.item
        evidence_text = getattr(item, "evidence_text", None) or item.meta.get("evidence_text")
        return {
            "source_tier": hit.source_tier.value,
            "match_type": hit.match_type.value,
            "score": hit.score,
            "semantic_rank": hit.semantic_rank,
            "doc_signature": getattr(item, "doc_signature", None) or item.meta.get("doc_signature"),
            "source_file": getattr(item, "source_file", None) or item.meta.get("source_file"),
            "source_id": getattr(item, "source_id", None) or item.meta.get("source_id"),
            "chunk_index": (
                item.chunk_index if getattr(item, "chunk_index", None) is not None
                else item.meta.get("chunk_index")
            ),
            "chunk_id": getattr(item, "chunk_id", None) or item.meta.get("chunk_id"),
            "question_type": getattr(item, "question_type", None) or item.meta.get("question_type"),
            "evidence_text": evidence_text,
            "evidence_chars": len(str(evidence_text)) if evidence_text is not None else None,
            "same_document": bool(hit.debug.get("same_document", False)) if isinstance(hit.debug, dict) else None,
            "same_source": bool(hit.debug.get("same_source", False)) if isinstance(hit.debug, dict) else None,
            "document_relation": (
                hit.debug.get("document_relation") if isinstance(hit.debug, dict) else None
            ),
        }

    def _normalize_answer_for_storage(self, mq: MemoryQuery, answer_text: str) -> str:
        text = str(answer_text or "").strip()
        if not text:
            return text

        task = str(getattr(mq, "task", "") or "").strip().lower()
        question_type = str(self._query_question_type(mq) or "").strip().lower()

        if task == "trec" or question_type == "classification":
            norm = self._normalize_trec_label(text)
            if norm is not None:
                return norm

        return text

    def _normalize_trec_label(self, text: str) -> Optional[str]:
        raw = str(text or "").strip()
        if not raw:
            return None

        upper_raw = raw.upper().strip()
        if upper_raw in self._TREC_LABELS:
            return upper_raw

        cleaned = upper_raw.replace("\r", "\n")
        cleaned = re.sub(r"[`*_>#\[\]\(\)\{\}]", " ", cleaned)
        cleaned = cleaned.replace("-", " ")
        cleaned = cleaned.replace(":", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        tokens = cleaned.split()
        for tok in tokens:
            if tok in self._TREC_LABELS:
                return tok

        phrase = cleaned

        if any(x in phrase for x in ("ABBREVIATION", "ABBREVIATED", "ACRONYM", "SHORT FORM", "EXPRESSION ABBREVIATED")):
            return "ABBR"

        if any(x in phrase for x in ("DATE", "TIME", "YEAR", "AGE", "NUMBER", "COUNT", "QUANTITY", "PERCENT", "MONEY", "PRICE", "DISTANCE")):
            return "NUM"

        if any(x in phrase for x in ("LOCATION", "PLACE", "CITY", "COUNTRY", "STATE", "OTHER LOCATION")):
            return "LOC"

        if any(x in phrase for x in ("HUMAN", "PERSON", "INDIVIDUAL", "WHO ")) or phrase.startswith("WHO"):
            return "HUM"

        if any(
            x in phrase
            for x in (
                "DESCRIPTION",
                "DEFINITION",
                "EXPLANATION",
                "REASON",
                "MANNER",
                "DESC",
                "DESCRIPTION OF SOMETHING",
            )
        ):
            return "DESC"

        if any(
            x in phrase
            for x in (
                "ENTITY",
                "OBJECT",
                "ANIMAL",
                "COLOR",
                "FOOD",
                "INSTRUMENT",
                "LANGUAGE",
                "LETTER",
                "RELIGION",
                "SPORT",
                "SUBSTANCE",
                "SYMBOL",
                "TECHNIQUE",
                "TERM",
                "VEHICLE",
                "WORD",
                "INVENTION",
                "OTHER ENTITY",
            )
        ):
            return "ENTY"

        return None

    def _retrieve_exact(
        self,
        mq: MemoryQuery,
        *,
        now: datetime,
        ctx_sig: str,
    ) -> Tuple[Optional[MemoryHit], Dict[str, Any]]:
        pol = self._cfg.retrieval_policy
        budget = budget_from_query(mq)

        q_can = canonicalize(mq.raw_query)
        ram_reads = 0
        disk_reads = 0
        namespaces_checked: List[Dict[str, Any]] = []

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

            ns_dbg: Dict[str, Any] = {
                "scope": scope.value,
                "namespace": ns,
                "key": key,
                "ram_checked": False,
                "ram_hit": False,
                "disk_checked": False,
                "disk_hit": False,
            }

            if ram_reads < budget.max_ram_reads:
                ram_reads += 1
                ns_dbg["ram_checked"] = True
                item = self._ram.get(ns, key)
                if item is not None:
                    ns_dbg["ram_hit"] = True
                    ok, dbg = accept_item(
                        mq,
                        item,
                        policy=pol,
                        now_utc=now,
                        query_context_signature=ctx_sig,
                    )
                    ns_dbg["ram_accept_reason"] = dbg.get("reason")
                    if ok:
                        hit_namespaces_checked = namespaces_checked + [ns_dbg]
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
                                extra={
                                    "ram_reads": ram_reads,
                                    "disk_reads": disk_reads,
                                    "namespaces_checked": hit_namespaces_checked,
                                    "same_document": same_document(mq, item),
                                    "same_source": (
                                        self._query_source_file(mq) is not None
                                        and self._query_source_file(mq) == self._item_source_file(item)
                                    ),
                                    "document_relation": document_relation(mq, item),
                                },
                            ),
                        ), {
                            "ram_reads": ram_reads,
                            "disk_reads": disk_reads,
                            "namespaces_checked": hit_namespaces_checked,
                            "promoted_to_ram": False,
                        }

            if disk_reads < budget.max_disk_reads:
                disk_reads += 1
                ns_dbg["disk_checked"] = True
                item = self._disk.get(ns, key)
                if item is not None:
                    ns_dbg["disk_hit"] = True
                    ok, dbg = accept_item(
                        mq,
                        item,
                        policy=pol,
                        now_utc=now,
                        query_context_signature=ctx_sig,
                    )
                    ns_dbg["disk_accept_reason"] = dbg.get("reason")
                    if ok:
                        promoted = False
                        hit_namespaces_checked = namespaces_checked + [ns_dbg]
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
                                extra={
                                    "ram_reads": ram_reads,
                                    "disk_reads": disk_reads,
                                    "namespaces_checked": hit_namespaces_checked,
                                    "same_document": same_document(mq, item),
                                    "same_source": (
                                        self._query_source_file(mq) is not None
                                        and self._query_source_file(mq) == self._item_source_file(item)
                                    ),
                                    "document_relation": document_relation(mq, item),
                                },
                            ),
                        )
                        if self._cfg.promote_disk_hits_to_ram:
                            try:
                                self._ram.put(ns, key, item)
                                promoted = True
                            except Exception:
                                promoted = False

                        return hit, {
                            "ram_reads": ram_reads,
                            "disk_reads": disk_reads,
                            "namespaces_checked": hit_namespaces_checked,
                            "promoted_to_ram": promoted,
                        }

            namespaces_checked.append(ns_dbg)

        return None, {
            "ram_reads": ram_reads,
            "disk_reads": disk_reads,
            "namespaces_checked": namespaces_checked,
            "promoted_to_ram": False,
        }

    def _iter_store_namespace(self, store: MemoryStore, namespace: str) -> Iterable[MemoryItem]:
        if hasattr(store, "iter_namespace"):
            return getattr(store, "iter_namespace")(namespace)
        return ()

    def _iter_store_candidates(
        self,
        store: MemoryStore,
        namespace: str,
        *,
        mq: MemoryQuery,
        limit: Optional[int] = None,
    ) -> Iterable[MemoryItem]:
        """
        Prefer cheap store-native candidate filtering when available.
        Falls back to full namespace iteration for stores that do not support it.
        """
        if hasattr(store, "iter_candidates"):
            try:
                return getattr(store, "iter_candidates")(
                    namespace,
                    task=mq.task,
                    source_file=self._query_source_file(mq),
                    doc_signature=self._query_doc_signature(mq),
                    limit=limit,
                )
            except TypeError:
                pass
            except Exception:
                return ()
        return self._iter_store_namespace(store, namespace)

    def _build_lexical_candidates(
        self,
        mq: MemoryQuery,
        *,
        now: datetime,
    ) -> Tuple[
        List[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]],
        Dict[str, Any],
    ]:
        pol = self._cfg.retrieval_policy
        budget = budget_from_query(mq)

        candidates: List[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]] = []
        ram_reads = 0
        disk_reads = 0
        namespaces_checked: List[Dict[str, Any]] = []

        # Small bounded hint to candidate iterators; keeps Jetson scans tighter
        candidate_limit = max(8, self._lexical_top_k() * 8)

        for rn in resolve_namespaces(mq, scope_order=pol.scope_order, include_missing=False):
            scope = rn.scope
            ns = rn.namespace

            ns_dbg: Dict[str, Any] = {
                "scope": scope.value,
                "namespace": ns,
                "lexical_ram_scanned": False,
                "lexical_disk_scanned": False,
                "lexical_ram_candidates": 0,
                "lexical_disk_candidates": 0,
                "lexical_same_document_candidates": 0,
                "lexical_same_source_candidates": 0,
                "lexical_broader_candidates": 0,
            }

            if ram_reads < budget.max_ram_reads:
                ram_reads += 1
                ns_dbg["lexical_ram_scanned"] = True
                for item in self._iter_store_candidates(
                    self._ram, ns, mq=mq, limit=candidate_limit
                ):
                    ok, dbg = self._lexical_candidate_allowed(mq, item, now=now)
                    if not ok:
                        continue
                    ns_dbg["lexical_ram_candidates"] += 1
                    if bool(dbg.get("same_document", False)):
                        ns_dbg["lexical_same_document_candidates"] += 1
                    if bool(dbg.get("same_source", False)):
                        ns_dbg["lexical_same_source_candidates"] += 1
                    if not bool(dbg.get("same_document", False)) and not bool(dbg.get("same_source", False)):
                        ns_dbg["lexical_broader_candidates"] += 1
                    candidates.append((SourceTier.RAM, scope, ns, item, dbg))

            if disk_reads < budget.max_disk_reads:
                disk_reads += 1
                ns_dbg["lexical_disk_scanned"] = True
                for item in self._iter_store_candidates(
                    self._disk, ns, mq=mq, limit=candidate_limit
                ):
                    ok, dbg = self._lexical_candidate_allowed(mq, item, now=now)
                    if not ok:
                        continue
                    ns_dbg["lexical_disk_candidates"] += 1
                    if bool(dbg.get("same_document", False)):
                        ns_dbg["lexical_same_document_candidates"] += 1
                    if bool(dbg.get("same_source", False)):
                        ns_dbg["lexical_same_source_candidates"] += 1
                    if not bool(dbg.get("same_document", False)) and not bool(dbg.get("same_source", False)):
                        ns_dbg["lexical_broader_candidates"] += 1
                    candidates.append((SourceTier.DISK, scope, ns, item, dbg))

            namespaces_checked.append(ns_dbg)

        return candidates, {
            "ram_reads": ram_reads,
            "disk_reads": disk_reads,
            "namespaces_checked": namespaces_checked,
            "candidate_count": len(candidates),
            "same_document_candidate_count": sum(1 for c in candidates if bool(c[4].get("same_document", False))),
            "same_source_candidate_count": sum(1 for c in candidates if bool(c[4].get("same_source", False))),
            "broader_candidate_count": sum(
                1 for c in candidates if not bool(c[4].get("same_document", False)) and not bool(c[4].get("same_source", False))
            ),
        }

    def _rank_lexical_candidates(
        self,
        mq: MemoryQuery,
        candidates: List[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]],
    ) -> Tuple[
        List[Tuple[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]], float, int]],
        Dict[str, Any],
    ]:
        q_norm = self._query_norm(mq)
        q_tokens = self._query_tokens(mq)

        same_doc_candidates: List[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]] = []
        same_source_candidates: List[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]] = []
        broader_candidates: List[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]] = []

        for cand in candidates:
            dbg = cand[4]
            if bool(dbg.get("same_document", False)):
                same_doc_candidates.append(cand)
            elif bool(dbg.get("same_source", False)):
                same_source_candidates.append(cand)
            else:
                broader_candidates.append(cand)

        dbg: Dict[str, Any] = {
            "prefer_same_source": self._prefer_same_source(),
            "same_document_pool_size": len(same_doc_candidates),
            "same_source_pool_size": len(same_source_candidates),
            "broader_pool_size": len(broader_candidates),
        }

        def _score_pool(
            pool: List[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]]
        ) -> List[Tuple[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]], float, int]]:
            scored: List[Tuple[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]], float]] = []
            for cand in pool:
                _source_tier, _scope, _ns, item, cdbg = cand
                score = lexical_score(
                    query_norm=q_norm,
                    query_tokens=q_tokens,
                    item_norm=self._item_query_norm(item),
                    item_tokens=self._item_query_tokens(item),
                    same_source=bool(cdbg.get("same_document", False) or cdbg.get("same_source", False)),
                )

                if bool(cdbg.get("same_document", False)):
                    score = min(1.0, float(score) + 0.15)
                elif bool(cdbg.get("same_source", False)):
                    score = min(1.0, float(score) + 0.05)

                if score >= self._lexical_context_threshold():
                    scored.append((cand, float(score)))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_k = self._lexical_top_k()
            ranked: List[Tuple[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]], float, int]] = []
            for idx, (cand, score) in enumerate(scored[:top_k], start=1):
                ranked.append((cand, score, idx))
            return ranked

        if same_doc_candidates:
            ranked = _score_pool(same_doc_candidates)
            if ranked:
                dbg["selected_pool"] = "same_document"
                dbg["selected_pool_size"] = len(same_doc_candidates)
                return ranked, dbg

        if self._prefer_same_source() and same_source_candidates:
            ranked = _score_pool(same_source_candidates)
            if ranked:
                dbg["selected_pool"] = "same_source"
                dbg["selected_pool_size"] = len(same_source_candidates)
                return ranked, dbg

        combined = broader_candidates
        if not self._prefer_same_source():
            combined = same_source_candidates + broader_candidates

        ranked = _score_pool(combined)
        if ranked:
            dbg["selected_pool"] = "broader" if self._prefer_same_source() else "combined_non_doc"
            dbg["selected_pool_size"] = len(combined)
            return ranked, dbg

        dbg["selected_pool"] = None
        dbg["selected_pool_size"] = 0
        return [], dbg

    def _retrieve_lexical(
        self,
        mq: MemoryQuery,
        *,
        now: datetime,
    ) -> Tuple[Optional[MemoryHit], Dict[str, Any]]:
        if not self._lexical_enabled():
            return None, {"lexical_enabled": False, "reason": "policy_disabled"}

        candidates, scan_dbg = self._build_lexical_candidates(mq, now=now)
        if not candidates:
            return None, {
                "lexical_enabled": True,
                "reason": "no_candidates",
                **scan_dbg,
            }

        ranked, rank_dbg = self._rank_lexical_candidates(mq, candidates)
        if not ranked:
            return None, {
                "lexical_enabled": True,
                "reason": "below_threshold",
                **scan_dbg,
                **rank_dbg,
            }

        payload, score, rank = ranked[0]
        source_tier, scope, ns, item, filter_dbg = payload

        task_l = str(mq.task or "").strip().lower()
        safe_task = task_l in self._safe_direct_reuse_tasks()

        bypass_allowed = (
            safe_task
            and float(score) >= self._lexical_direct_threshold()
        )

        if not bypass_allowed and safe_task and bool(filter_dbg.get("same_document", False)):
            q_type = str(self._query_question_type(mq) or "").strip().lower()
            i_type = str(self._item_question_type(item) or "").strip().lower()
            q_ans = str(self._query_answer_canonical(mq) or "").strip()
            i_ans = str(self._item_answer_canonical(item) or "").strip()

            if q_type == "classification" or i_type == "classification":
                if float(score) >= self._lexical_context_threshold():
                    bypass_allowed = True
            elif q_ans and i_ans and q_ans == i_ans and float(score) >= self._lexical_context_threshold():
                bypass_allowed = True

        match_type = "direct" if bypass_allowed else "context"

        promoted = False
        hit = MemoryHit(
            item=item,
            source_tier=source_tier,
            match_type=MatchType.LEXICAL,
            score=float(score),
            semantic_rank=rank,
            bypass_allowed=bypass_allowed,
            debug=make_hit_debug(
                scope=scope,
                namespace=ns,
                source="lexical_ram" if source_tier == SourceTier.RAM else "lexical_disk",
                accepted_reason="lexical_direct" if bypass_allowed else "lexical_context",
                extra={
                    "lexical_candidate_rank": rank,
                    "lexical_score": float(score),
                    "lexical_bypassed": bypass_allowed,
                    "same_document": bool(filter_dbg.get("same_document", False)),
                    "same_source": bool(filter_dbg.get("same_source", False)),
                    "document_relation": filter_dbg.get("document_relation"),
                    "filter_debug": filter_dbg,
                    "lexical_match_type": match_type,
                    **scan_dbg,
                    **rank_dbg,
                },
            ),
        )

        if source_tier == SourceTier.DISK and self._cfg.promote_disk_hits_to_ram:
            try:
                self._ram.put(ns, item.key, item)
                promoted = True
            except Exception:
                promoted = False

        return hit, {
            "lexical_enabled": True,
            "reason": "hit",
            "top_score": float(score),
            "top_rank": rank,
            "promoted_to_ram": promoted,
            "same_document": bool(filter_dbg.get("same_document", False)),
            "same_source": bool(filter_dbg.get("same_source", False)),
            "document_relation": filter_dbg.get("document_relation"),
            "lexical_match_type": match_type,
            **scan_dbg,
            **rank_dbg,
        }

    def _build_semantic_candidates(
        self,
        mq: MemoryQuery,
        *,
        now: datetime,
        ctx_sig: str,
    ) -> Tuple[
        List[SemanticCandidate[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]]],
        Dict[str, Any],
    ]:
        pol = self._cfg.retrieval_policy
        budget = budget_from_query(mq)

        candidates: List[SemanticCandidate[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]]] = []
        ram_reads = 0
        disk_reads = 0
        namespaces_checked: List[Dict[str, Any]] = []

        candidate_limit = max(8, int(getattr(pol, "max_semantic_candidates", 5)) * 8)

        for rn in resolve_namespaces(mq, scope_order=pol.scope_order, include_missing=False):
            scope = rn.scope
            ns = rn.namespace

            ns_dbg: Dict[str, Any] = {
                "scope": scope.value,
                "namespace": ns,
                "semantic_ram_scanned": False,
                "semantic_disk_scanned": False,
                "semantic_ram_candidates": 0,
                "semantic_disk_candidates": 0,
                "semantic_same_document_candidates": 0,
                "semantic_broader_candidates": 0,
            }

            if ram_reads < budget.max_ram_reads:
                ram_reads += 1
                ns_dbg["semantic_ram_scanned"] = True
                for item in self._iter_store_candidates(
                    self._ram, ns, mq=mq, limit=candidate_limit
                ):
                    ok, dbg = semantic_candidate_allowed(
                        mq,
                        item,
                        policy=pol,
                        now_utc=now,
                        query_context_signature=ctx_sig,
                    )
                    if not ok or item.query_embedding is None:
                        continue

                    ns_dbg["semantic_ram_candidates"] += 1
                    if bool(dbg.get("same_document", False)):
                        ns_dbg["semantic_same_document_candidates"] += 1
                    else:
                        ns_dbg["semantic_broader_candidates"] += 1

                    candidates.append(
                        SemanticCandidate(
                            payload=(SourceTier.RAM, scope, ns, item, dbg),
                            vector=item.query_embedding,
                        )
                    )

            if disk_reads < budget.max_disk_reads:
                disk_reads += 1
                ns_dbg["semantic_disk_scanned"] = True
                for item in self._iter_store_candidates(
                    self._disk, ns, mq=mq, limit=candidate_limit
                ):
                    ok, dbg = semantic_candidate_allowed(
                        mq,
                        item,
                        policy=pol,
                        now_utc=now,
                        query_context_signature=ctx_sig,
                    )
                    if not ok or item.query_embedding is None:
                        continue

                    ns_dbg["semantic_disk_candidates"] += 1
                    if bool(dbg.get("same_document", False)):
                        ns_dbg["semantic_same_document_candidates"] += 1
                    else:
                        ns_dbg["semantic_broader_candidates"] += 1

                    candidates.append(
                        SemanticCandidate(
                            payload=(SourceTier.DISK, scope, ns, item, dbg),
                            vector=item.query_embedding,
                        )
                    )

            namespaces_checked.append(ns_dbg)

        return candidates, {
            "ram_reads": ram_reads,
            "disk_reads": disk_reads,
            "namespaces_checked": namespaces_checked,
            "candidate_count": len(candidates),
            "same_document_candidate_count": sum(
                1 for c in candidates if bool(c.payload[4].get("same_document", False))
            ),
            "broader_candidate_count": sum(
                1 for c in candidates if not bool(c.payload[4].get("same_document", False))
            ),
        }

    def _rank_semantic_candidates(
        self,
        *,
        query_vec: List[float],
        candidates: List[SemanticCandidate[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]]],
        policy: RetrievalPolicy,
    ) -> Tuple[
        List[Tuple[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]], float, int]],
        Dict[str, Any],
    ]:
        same_doc_candidates: List[SemanticCandidate[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]]] = []
        broader_candidates: List[SemanticCandidate[Tuple[SourceTier, Scope, str, MemoryItem, Dict[str, Any]]]] = []

        for cand in candidates:
            filter_dbg = cand.payload[4]
            if bool(filter_dbg.get("same_document", False)):
                same_doc_candidates.append(cand)
            else:
                broader_candidates.append(cand)

        dbg: Dict[str, Any] = {
            "prefer_same_document": bool(policy.prefer_same_document_for_semantic),
            "same_document_pool_size": len(same_doc_candidates),
            "broader_pool_size": len(broader_candidates),
        }

        if policy.prefer_same_document_for_semantic and same_doc_candidates:
            ranked_same = self._embed_index.search_candidates(
                query_vector=query_vec,
                candidates=same_doc_candidates,
                top_k=policy.max_semantic_candidates,
                min_score=policy.semantic_threshold_context,
            )
            if ranked_same:
                dbg["selected_pool"] = "same_document"
                dbg["selected_pool_size"] = len(same_doc_candidates)
                return ranked_same, dbg

        ranked_broader = self._embed_index.search_candidates(
            query_vector=query_vec,
            candidates=broader_candidates if policy.prefer_same_document_for_semantic else candidates,
            top_k=policy.max_semantic_candidates,
            min_score=policy.semantic_threshold_context,
        )
        if ranked_broader:
            dbg["selected_pool"] = (
                "broader" if policy.prefer_same_document_for_semantic else "combined"
            )
            dbg["selected_pool_size"] = (
                len(broader_candidates) if policy.prefer_same_document_for_semantic else len(candidates)
            )
            return ranked_broader, dbg

        dbg["selected_pool"] = None
        dbg["selected_pool_size"] = 0
        return [], dbg

    def _retrieve_semantic(
        self,
        mq: MemoryQuery,
        *,
        now: datetime,
        ctx_sig: str,
    ) -> Tuple[Optional[MemoryHit], Dict[str, Any]]:
        pol = self._cfg.retrieval_policy
        budget = budget_from_query(mq)

        if not pol.semantic_enabled:
            return None, {"semantic_enabled": False, "reason": "policy_disabled"}
        if not budget.allow_semantic:
            return None, {"semantic_enabled": False, "reason": "budget_disabled"}
        if self._embedder is None:
            return None, {"semantic_enabled": False, "reason": "missing_embedder"}

        query_vec = self._embedder.embed(mq.raw_query)
        if not query_vec:
            return None, {"semantic_enabled": True, "reason": "query_embed_failed"}

        candidates, scan_dbg = self._build_semantic_candidates(
            mq,
            now=now,
            ctx_sig=ctx_sig,
        )
        if not candidates:
            return None, {
                "semantic_enabled": True,
                "reason": "no_candidates",
                **scan_dbg,
            }

        ranked, rank_dbg = self._rank_semantic_candidates(
            query_vec=list(query_vec),
            candidates=candidates,
            policy=pol,
        )
        if not ranked:
            return None, {
                "semantic_enabled": True,
                "reason": "below_threshold",
                **scan_dbg,
                **rank_dbg,
            }

        payload, score, rank = ranked[0]
        source_tier, scope, ns, item, filter_dbg = payload

        decision, decision_dbg = semantic_decision(
            mq=mq,
            score=score,
            item=item,
            policy=pol,
            query_context_signature=ctx_sig,
        )
        if decision == "ignore":
            return None, {
                "semantic_enabled": True,
                "reason": "decision_ignore",
                "top_score": float(score),
                "top_rank": rank,
                **scan_dbg,
                **rank_dbg,
            }

        bypass_allowed = bool(decision == "bypass")
        promoted = False

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
                accepted_reason=decision_dbg.get(
                    "reason",
                    "semantic_bypass" if bypass_allowed else "semantic_context",
                ),
                extra={
                    "semantic_candidate_rank": rank,
                    "semantic_score": float(score),
                    "semantic_bypassed": bypass_allowed,
                    "same_document": bool(filter_dbg.get("same_document", False)),
                    "same_source": False,
                    "document_relation": filter_dbg.get("document_relation"),
                    "filter_debug": filter_dbg,
                    **decision_dbg,
                    **scan_dbg,
                    **rank_dbg,
                },
            ),
        )

        if source_tier == SourceTier.DISK and self._cfg.promote_disk_hits_to_ram:
            try:
                self._ram.put(ns, item.key, item)
                promoted = True
            except Exception:
                promoted = False

        return hit, {
            "semantic_enabled": True,
            "reason": "hit",
            "top_score": float(score),
            "top_rank": rank,
            "promoted_to_ram": promoted,
            "same_document": bool(filter_dbg.get("same_document", False)),
            "document_relation": filter_dbg.get("document_relation"),
            "semantic_match_type": "direct" if bypass_allowed else "context",
            "semantic_bypassed": bypass_allowed,
            **scan_dbg,
            **rank_dbg,
        }

    def retrieve(
        self,
        mq: MemoryQuery,
        return_meta: bool = False,
    ):
        now = datetime.now(timezone.utc)
        ctx_sig = context_signature(mq.context)

        retrieval_t0 = time.time()
        exact_hit, exact_dbg = self._retrieve_exact(mq, now=now, ctx_sig=ctx_sig)
        if exact_hit is not None:
            meta = {
                "retrieval_stage": "exact",
                "memory_lookup_ms": (time.time() - retrieval_t0) * 1000.0,
                **exact_dbg,
            }
            if return_meta:
                return exact_hit, meta
            return exact_hit

        lexical_hit, lexical_dbg = self._retrieve_lexical(mq, now=now)
        if lexical_hit is not None:
            meta = {
                "retrieval_stage": "lexical",
                "memory_lookup_ms": (time.time() - retrieval_t0) * 1000.0,
                **lexical_dbg,
            }
            if return_meta:
                return lexical_hit, meta
            return lexical_hit

        semantic_hit, semantic_dbg = self._retrieve_semantic(mq, now=now, ctx_sig=ctx_sig)
        if semantic_hit is not None:
            meta = {
                "retrieval_stage": "semantic",
                "memory_lookup_ms": (time.time() - retrieval_t0) * 1000.0,
                **semantic_dbg,
            }
            if return_meta:
                return semantic_hit, meta
            return semantic_hit

        meta = {
            "retrieval_stage": "miss",
            "memory_lookup_ms": (time.time() - retrieval_t0) * 1000.0,
            **exact_dbg,
            "lexical": lexical_dbg,
            "semantic": semantic_dbg,
        }
        if return_meta:
            return None, meta
        return None

    def _make_embedding_fields(
        self, mq: MemoryQuery
    ) -> Tuple[Optional[List[float]], Optional[str], Optional[float]]:
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
        ap = self._cfg.admission_policy
        q_can = canonicalize(mq.raw_query)
        ctx_sig = context_signature(mq.context)
        now = datetime.now(timezone.utc)

        if not self._cfg.enable_storage:
            return {
                "stored": [],
                "skipped": [{"scope": None, "reason": "storage_disabled"}],
            }

        normalized_answer_text = self._normalize_answer_for_storage(mq, answer_text)

        merged_meta = self._merged_store_meta(
            mq,
            answer_text=normalized_answer_text,
            incoming_meta=meta,
            raw_generated_answer=answer_text,
        )

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

            ok, dbg = should_store(
                mq,
                normalized_answer_text,
                quality,
                scope=scope,
                policy=ap,
            )
            if not ok:
                stored["skipped"].append(
                    {
                        "scope": scope.value,
                        "normalized_answer_text": normalized_answer_text,
                        "raw_generated_answer": answer_text,
                        **dbg,
                    }
                )
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
                answer_text=normalized_answer_text,
                provenance=provenance,
                quality=quality,
                created_at_utc=now,
                ttl_seconds=ttl_s,
                meta=merged_meta,
                evidence_text=self._query_evidence_text(mq),
                doc_signature=self._query_doc_signature(mq),
                source_file=self._query_source_file(mq),
                source_id=self._query_source_id(mq),
                chunk_index=self._query_chunk_index(mq),
                chunk_id=self._query_chunk_id(mq),
                question_type=self._query_question_type(mq),
                answer_canonical=self._query_answer_canonical(mq),
                query_embedding=query_embedding,
                embedding_model_id=embedding_model_id,
                embedding_norm=embedding_norm,
            )

            disk_ok = False
            if self._cfg.store_on_disk:
                try:
                    self._disk.put(ns, key, item)
                    disk_ok = True
                except Exception as e:
                    stored["skipped"].append(
                        {
                            "scope": scope.value,
                            "reason": "disk_write_failed",
                            "error": str(e),
                        }
                    )

            ram_ok = False
            if self._cfg.store_in_ram:
                try:
                    self._ram.put(ns, key, item)
                    ram_ok = True
                except Exception:
                    ram_ok = False

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
                    "doc_signature": item.doc_signature,
                    "source_file": item.source_file,
                    "source_id": item.source_id,
                    "chunk_index": item.chunk_index,
                    "chunk_id": item.chunk_id,
                    "question_type": item.question_type,
                    "evidence_chars": len(item.evidence_text) if item.evidence_text else None,
                    "answer_canonical": item.answer_canonical,
                    "stored_answer_text": normalized_answer_text,
                    "raw_generated_answer": answer_text,
                }
            )

        return stored

    def answer(self, mq: MemoryQuery, generator: Generator) -> Tuple[str, Dict[str, Any]]:
        hit, retrieval_dbg = self.retrieve(mq, return_meta=True)
        memory_lookup_ms = float(retrieval_dbg.get("memory_lookup_ms", 0.0) or 0.0)
        query_evidence_text = self._query_evidence_text(mq)
        query_evidence_chars = len(str(query_evidence_text)) if query_evidence_text is not None else None

        if (
            hit is not None
            and hit.match_type == MatchType.EXACT
            and self._cfg.return_memory_directly
        ):
            exact_item = hit.item
            exact_evidence_text = getattr(exact_item, "evidence_text", None) or exact_item.meta.get("evidence_text")
            return exact_item.answer_text, {
                "used_memory": True,
                "generated": False,
                "source_tier": hit.source_tier.value,
                "memory_source_tier": hit.source_tier.value,
                "match_type": hit.match_type.value,
                "score": hit.score,
                "lexical_used": False,
                "lexical_bypassed": False,
                "lexical_context_used": False,
                "semantic_used": False,
                "semantic_bypassed": False,
                "semantic_candidate_rank": None,
                "promoted_to_ram": bool(retrieval_dbg.get("promoted_to_ram", False)),
                "namespaces_checked": retrieval_dbg.get("namespaces_checked", []),
                "hit": dict(hit.debug),
                "stored": False,
                "stored_scopes": [],
                "memory_lookup_ms": memory_lookup_ms,
                "generation_ms_est": 0.0,
                "doc_signature": getattr(exact_item, "doc_signature", None) or exact_item.meta.get("doc_signature"),
                "source_file": getattr(exact_item, "source_file", None) or exact_item.meta.get("source_file"),
                "source_id": getattr(exact_item, "source_id", None) or exact_item.meta.get("source_id"),
                "chunk_index": (
                    exact_item.chunk_index
                    if getattr(exact_item, "chunk_index", None) is not None
                    else exact_item.meta.get("chunk_index")
                ),
                "chunk_id": getattr(exact_item, "chunk_id", None) or exact_item.meta.get("chunk_id"),
                "question_type": getattr(exact_item, "question_type", None) or exact_item.meta.get("question_type"),
                "stored_evidence_text": exact_evidence_text,
                "stored_evidence_chars": len(str(exact_evidence_text)) if exact_evidence_text is not None else None,
                "query_evidence_text": query_evidence_text,
                "timings_ms": {
                    "memory_lookup_ms": memory_lookup_ms,
                    "generation_ms_est": 0.0,
                    "total_ms": memory_lookup_ms,
                },
            }

        lexical_hit_for_generation = (
            hit
            if hit is not None
            and hit.match_type == MatchType.LEXICAL
            and not bool(getattr(hit, "bypass_allowed", False))
            else None
        )

        if (
            hit is not None
            and hit.match_type == MatchType.LEXICAL
            and bool(getattr(hit, "bypass_allowed", False))
            and self._cfg.return_memory_directly
        ):
            item = hit.item
            evidence_text = getattr(item, "evidence_text", None) or item.meta.get("evidence_text")
            return item.answer_text, {
                "used_memory": True,
                "generated": False,
                "source_tier": hit.source_tier.value,
                "memory_source_tier": hit.source_tier.value,
                "match_type": hit.match_type.value,
                "score": hit.score,
                "lexical_used": True,
                "lexical_bypassed": True,
                "lexical_context_used": False,
                "semantic_used": False,
                "semantic_bypassed": False,
                "semantic_candidate_rank": None,
                "promoted_to_ram": bool(retrieval_dbg.get("promoted_to_ram", False)),
                "namespaces_checked": retrieval_dbg.get("namespaces_checked", []),
                "hit": dict(hit.debug),
                "stored": False,
                "stored_scopes": [],
                "memory_lookup_ms": memory_lookup_ms,
                "generation_ms_est": 0.0,
                "doc_signature": getattr(item, "doc_signature", None) or item.meta.get("doc_signature"),
                "source_file": getattr(item, "source_file", None) or item.meta.get("source_file"),
                "source_id": getattr(item, "source_id", None) or item.meta.get("source_id"),
                "chunk_index": (
                    item.chunk_index
                    if getattr(item, "chunk_index", None) is not None
                    else item.meta.get("chunk_index")
                ),
                "chunk_id": getattr(item, "chunk_id", None) or item.meta.get("chunk_id"),
                "question_type": getattr(item, "question_type", None) or item.meta.get("question_type"),
                "stored_evidence_text": evidence_text,
                "stored_evidence_chars": len(str(evidence_text)) if evidence_text is not None else None,
                "query_evidence_text": query_evidence_text,
                "timings_ms": {
                    "memory_lookup_ms": memory_lookup_ms,
                    "generation_ms_est": 0.0,
                    "total_ms": memory_lookup_ms,
                },
            }

        if (
            hit is not None
            and hit.match_type == MatchType.SEMANTIC
            and bool(getattr(hit, "bypass_allowed", False))
            and self._cfg.return_memory_directly
        ):
            item = hit.item
            evidence_text = getattr(item, "evidence_text", None) or item.meta.get("evidence_text")
            return item.answer_text, {
                "used_memory": True,
                "generated": False,
                "source_tier": hit.source_tier.value,
                "memory_source_tier": hit.source_tier.value,
                "match_type": hit.match_type.value,
                "score": hit.score,
                "lexical_used": False,
                "lexical_bypassed": False,
                "lexical_context_used": False,
                "semantic_used": True,
                "semantic_bypassed": True,
                "semantic_candidate_rank": hit.semantic_rank,
                "promoted_to_ram": bool(retrieval_dbg.get("promoted_to_ram", False)),
                "namespaces_checked": retrieval_dbg.get("namespaces_checked", []),
                "hit": dict(hit.debug),
                "stored": False,
                "stored_scopes": [],
                "memory_lookup_ms": memory_lookup_ms,
                "generation_ms_est": 0.0,
                "doc_signature": getattr(item, "doc_signature", None) or item.meta.get("doc_signature"),
                "source_file": getattr(item, "source_file", None) or item.meta.get("source_file"),
                "source_id": getattr(item, "source_id", None) or item.meta.get("source_id"),
                "chunk_index": (
                    item.chunk_index
                    if getattr(item, "chunk_index", None) is not None
                    else item.meta.get("chunk_index")
                ),
                "chunk_id": getattr(item, "chunk_id", None) or item.meta.get("chunk_id"),
                "question_type": getattr(item, "question_type", None) or item.meta.get("question_type"),
                "stored_evidence_text": evidence_text,
                "stored_evidence_chars": len(str(evidence_text)) if evidence_text is not None else None,
                "query_evidence_text": query_evidence_text,
                "timings_ms": {
                    "memory_lookup_ms": memory_lookup_ms,
                    "generation_ms_est": 0.0,
                    "total_ms": memory_lookup_ms,
                },
            }

        semantic_hit_for_generation = (
            hit
            if hit is not None
            and hit.match_type == MatchType.SEMANTIC
            and not bool(getattr(hit, "bypass_allowed", False))
            else None
        )
        retrieved_for_generation = lexical_hit_for_generation or semantic_hit_for_generation

        gen_t0 = time.time()
        answer_text, provenance, quality = generator.generate(
            mq,
            retrieved=retrieved_for_generation,
        )
        generation_ms_est = (time.time() - gen_t0) * 1000.0

        retrieved_summary = self._hit_debug_summary(retrieved_for_generation)

        store_dbg = self.store(
            mq,
            answer_text=answer_text,
            provenance=provenance,
            quality=quality,
            meta={
                "used_memory_context": retrieved_for_generation is not None,
                "memory_context_source": (
                    retrieved_for_generation.source_tier.value if retrieved_for_generation else None
                ),
                "memory_context_match_type": (
                    retrieved_for_generation.match_type.value if retrieved_for_generation else None
                ),
                "semantic_score": (
                    retrieved_for_generation.score
                    if retrieved_for_generation and retrieved_for_generation.match_type == MatchType.SEMANTIC
                    else None
                ),
                "lexical_score": (
                    retrieved_for_generation.score
                    if retrieved_for_generation and retrieved_for_generation.match_type == MatchType.LEXICAL
                    else None
                ),
                "memory_context_doc_signature": (
                    retrieved_summary.get("doc_signature") if retrieved_summary else None
                ),
                "memory_context_source_file": (
                    retrieved_summary.get("source_file") if retrieved_summary else None
                ),
                "memory_context_source_id": (
                    retrieved_summary.get("source_id") if retrieved_summary else None
                ),
                "memory_context_chunk_index": (
                    retrieved_summary.get("chunk_index") if retrieved_summary else None
                ),
                "memory_context_chunk_id": (
                    retrieved_summary.get("chunk_id") if retrieved_summary else None
                ),
                "memory_context_question_type": (
                    retrieved_summary.get("question_type") if retrieved_summary else None
                ),
                "memory_context_evidence_text": (
                    retrieved_summary.get("evidence_text") if retrieved_summary else None
                ),
                "memory_context_evidence_chars": (
                    retrieved_summary.get("evidence_chars") if retrieved_summary else None
                ),
                "memory_context_same_document": (
                    retrieved_summary.get("same_document") if retrieved_summary else None
                ),
                "memory_context_same_source": (
                    retrieved_summary.get("same_source") if retrieved_summary else None
                ),
                "memory_context_document_relation": (
                    retrieved_summary.get("document_relation") if retrieved_summary else None
                ),
            },
        )

        total_ms = memory_lookup_ms + generation_ms_est
        normalized_answer_text = self._normalize_answer_for_storage(mq, answer_text)

        lexical_used = (
            retrieved_for_generation is not None
            and retrieved_for_generation.match_type == MatchType.LEXICAL
        )
        semantic_used = (
            retrieved_for_generation is not None
            and retrieved_for_generation.match_type == MatchType.SEMANTIC
        )

        return answer_text, {
            "used_memory": retrieved_for_generation is not None,
            "generated": True,
            "source_tier": "compute",
            "memory_source_tier": (
                retrieved_for_generation.source_tier.value
                if retrieved_for_generation is not None
                else None
            ),
            "match_type": (
                retrieved_for_generation.match_type.value
                if retrieved_for_generation is not None
                else None
            ),
            "score": (
                retrieved_for_generation.score
                if retrieved_for_generation is not None
                else None
            ),
            "lexical_used": bool(lexical_used),
            "lexical_bypassed": False,
            "lexical_context_used": bool(lexical_used),
            "semantic_used": bool(semantic_used),
            "semantic_bypassed": False,
            "semantic_candidate_rank": (
                retrieved_for_generation.semantic_rank
                if retrieved_for_generation and retrieved_for_generation.match_type == MatchType.SEMANTIC
                else None
            ),
            "promoted_to_ram": bool(retrieval_dbg.get("promoted_to_ram", False)),
            "namespaces_checked": retrieval_dbg.get("namespaces_checked", []),
            "hit_before_generate": {
                "present": hit is not None,
                "source_tier": hit.source_tier.value if hit else None,
                "match_type": hit.match_type.value if hit else None,
                "score": hit.score if hit else None,
            },
            "stored": len(store_dbg.get("stored", [])) > 0,
            "stored_scopes": [x["scope"] for x in store_dbg.get("stored", [])],
            "store": store_dbg,
            "store_skipped": store_dbg.get("skipped", []),
            "memory_lookup_ms": memory_lookup_ms,
            "generation_ms_est": generation_ms_est,
            "timings_ms": {
                "memory_lookup_ms": memory_lookup_ms,
                "generation_ms_est": generation_ms_est,
                "total_ms": total_ms,
            },
            "retrieval_stage": retrieval_dbg.get("retrieval_stage"),
            "retrieval_debug": retrieval_dbg,
            "lexical_reason": retrieval_dbg.get("reason")
                if retrieval_dbg.get("retrieval_stage") == "lexical"
                else (retrieval_dbg.get("lexical") or {}).get("reason"),
            "lexical_candidate_count": retrieval_dbg.get("candidate_count")
                if retrieval_dbg.get("retrieval_stage") == "lexical"
                else (retrieval_dbg.get("lexical") or {}).get("candidate_count"),
            "lexical_top_score": retrieval_dbg.get("top_score")
                if retrieval_dbg.get("retrieval_stage") == "lexical"
                else (retrieval_dbg.get("lexical") or {}).get("top_score"),
            "lexical_top_rank": retrieval_dbg.get("top_rank")
                if retrieval_dbg.get("retrieval_stage") == "lexical"
                else (retrieval_dbg.get("lexical") or {}).get("top_rank"),
            "lexical_enabled_debug": (
                retrieval_dbg.get("lexical_enabled")
                if "lexical_enabled" in retrieval_dbg
                else (retrieval_dbg.get("lexical") or {}).get("lexical_enabled")
            ),
            "lexical_match_type": retrieval_dbg.get("lexical_match_type")
                if retrieval_dbg.get("retrieval_stage") == "lexical"
                else (retrieval_dbg.get("lexical") or {}).get("lexical_match_type"),
            "lexical_same_source": retrieval_dbg.get("same_source")
                if retrieval_dbg.get("retrieval_stage") == "lexical"
                else (retrieval_dbg.get("lexical") or {}).get("same_source"),
            "semantic_reason": retrieval_dbg.get("reason")
                if retrieval_dbg.get("retrieval_stage") == "semantic"
                else (retrieval_dbg.get("semantic") or {}).get("reason"),
            "semantic_candidate_count": retrieval_dbg.get("candidate_count")
                if retrieval_dbg.get("retrieval_stage") == "semantic"
                else (retrieval_dbg.get("semantic") or {}).get("candidate_count"),
            "semantic_top_score": retrieval_dbg.get("top_score")
                if retrieval_dbg.get("retrieval_stage") == "semantic"
                else (retrieval_dbg.get("semantic") or {}).get("top_score"),
            "semantic_top_rank": retrieval_dbg.get("top_rank")
                if retrieval_dbg.get("retrieval_stage") == "semantic"
                else (retrieval_dbg.get("semantic") or {}).get("top_rank"),
            "semantic_enabled_debug": (
                retrieval_dbg.get("semantic_enabled")
                if "semantic_enabled" in retrieval_dbg
                else (retrieval_dbg.get("semantic") or {}).get("semantic_enabled")
            ),
            "doc_signature": self._query_doc_signature(mq),
            "source_file": self._query_source_file(mq),
            "source_id": self._query_source_id(mq),
            "chunk_index": self._query_chunk_index(mq),
            "chunk_id": self._query_chunk_id(mq),
            "question_type": self._query_question_type(mq),
            "query_evidence_text": query_evidence_text,
            "query_evidence_chars": query_evidence_chars,
            "retrieved_memory": retrieved_summary,
            "stored_evidence_text": query_evidence_text,
            "stored_evidence_chars": query_evidence_chars,
            "normalized_answer_for_storage": normalized_answer_text,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "ram": getattr(self._ram, "stats")() if callable(getattr(self._ram, "stats", None)) else None,
            "disk": getattr(self._disk, "stats")() if callable(getattr(self._disk, "stats", None)) else None,
            "embed_index": self._embed_index.stats() if self._embed_index is not None else None,
        }
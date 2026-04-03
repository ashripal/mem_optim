from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from memarch.memory.schema import (
    AccessStats,
    MemoryItem,
    Provenance,
    QualitySignals,
    Scope,
)


# =============================================================================
# Disk schema version
# =============================================================================
# Bump to 5 because we are now persisting the additional fields needed for
# verified paraphrase reuse:
# - raw_query
# - task
# - start_char / end_char
# - answer_span_found
# - canonical_intent_id
SCHEMA_VERSION = 5


# =============================================================================
# Datetime helpers
# =============================================================================

def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    """
    Convert a timezone-aware datetime to a stable UTC ISO string.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    return dt.astimezone(timezone.utc).isoformat()


def _dt_from_str(s: Optional[str]) -> Optional[datetime]:
    """
    Parse a stored ISO datetime string back into a UTC-aware datetime.
    """
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# =============================================================================
# Embedding helpers
# =============================================================================

def _normalize_embedding(vec: Any) -> Optional[list[float]]:
    """
    Normalize embeddings to a stable plain-Python list of rounded floats.

    Why:
    - keeps JSON payload portable
    - avoids tests failing because of array('f') vs list differences
    - keeps on-disk representation deterministic
    """
    if vec is None:
        return None
    return [round(float(x), 6) for x in vec]


# =============================================================================
# Serialization / deserialization
# =============================================================================

def _serialize_item(item: MemoryItem) -> str:
    """
    Serialize a MemoryItem into a compact JSON payload for SQLite storage.

    Important:
    - We persist the richer schema fields required by the verified paraphrase
      reuse strategy.
    - We keep the payload deterministic for easier testing and debugging.
    """
    query_embedding = _normalize_embedding(getattr(item, "query_embedding", None))

    d: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "key": item.key,
        "scope": item.scope.value,
        "namespace": item.namespace,
        "query_canonical": item.query_canonical,
        "context_signature": item.context_signature,
        "answer_text": item.answer_text,
        # Newer verified-reuse fields
        "raw_query": item.raw_query,
        "task": item.task,
        "question_type": item.question_type,
        "evidence_text": item.evidence_text,
        "doc_signature": item.doc_signature,
        "source_file": item.source_file,
        "source_id": item.source_id,
        "chunk_index": item.chunk_index,
        "chunk_id": item.chunk_id,
        "start_char": item.start_char,
        "end_char": item.end_char,
        "answer_canonical": item.answer_canonical,
        "answer_span_found": item.answer_span_found,
        "canonical_intent_id": item.canonical_intent_id,
        # Provenance block
        "provenance": {
            "model_id": item.provenance.model_id,
            "prompt_version": item.provenance.prompt_version,
            "generated_at_utc": _dt_to_str(item.provenance.generated_at_utc),
            "generator_backend": item.provenance.generator_backend,
            "quantization": item.provenance.quantization,
            "context_window": item.provenance.context_window,
        },
        # Quality block
        "quality": {
            "score": item.quality.score,
            "success": item.quality.success,
            "metrics": item.quality.metrics,
        },
        # Lifetime metadata
        "created_at_utc": _dt_to_str(item.created_at_utc),
        "ttl_seconds": item.ttl_seconds,
        "expires_at_utc": _dt_to_str(item.expires_at_utc),
        # Access stats
        "stats": {
            "access_count": item.stats.access_count,
            "last_access_utc": _dt_to_str(item.stats.last_access_utc),
        },
        # Free-form metadata
        "meta": item.meta,
        # Semantic retrieval fields
        "query_embedding": query_embedding,
        "embedding_model_id": item.embedding_model_id,
        "embedding_norm": (
            round(float(item.embedding_norm), 6)
            if item.embedding_norm is not None
            else None
        ),
    }

    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deserialize_item(payload: str) -> MemoryItem:
    """
    Deserialize a JSON payload from SQLite back into a MemoryItem.

    Backward compatibility:
    - Supports older schema versions so existing benchmark DB files still load.
    - Missing newer fields are filled with safe defaults.
    """
    d = json.loads(payload)
    sv = int(d.get("schema_version", 1))

    if sv not in (1, 2, 3, 4, 5):
        raise ValueError(
            f"Unsupported schema_version: {sv} (expected one of 1, 2, 3, 4, 5)"
        )

    # ------------------------------
    # Rebuild provenance
    # ------------------------------
    prov_d = d["provenance"]
    provenance = Provenance(
        model_id=prov_d["model_id"],
        prompt_version=prov_d["prompt_version"],
        generated_at_utc=(
            _dt_from_str(prov_d.get("generated_at_utc"))
            or datetime.now(timezone.utc)
        ),
        generator_backend=prov_d.get("generator_backend"),
        quantization=prov_d.get("quantization"),
        context_window=prov_d.get("context_window"),
    )

    # ------------------------------
    # Rebuild quality block
    # ------------------------------
    qd = d.get("quality", {})
    quality = QualitySignals(
        score=qd.get("score"),
        success=qd.get("success"),
        metrics=dict(qd.get("metrics") or {}),
    )

    # ------------------------------
    # Rebuild access stats
    # ------------------------------
    sd = d.get("stats", {})
    stats = AccessStats(
        access_count=int(sd.get("access_count", 0)),
        last_access_utc=_dt_from_str(sd.get("last_access_utc")),
    )

    # ------------------------------
    # Normalize optional numeric fields
    # ------------------------------
    query_embedding = _normalize_embedding(d.get("query_embedding"))

    embedding_model_id = d.get("embedding_model_id")

    embedding_norm = d.get("embedding_norm")
    if embedding_norm is not None:
        embedding_norm = float(embedding_norm)

    chunk_index = d.get("chunk_index")
    if chunk_index is not None:
        chunk_index = int(chunk_index)

    start_char = d.get("start_char")
    if start_char is not None:
        start_char = int(start_char)

    end_char = d.get("end_char")
    if end_char is not None:
        end_char = int(end_char)

    # ------------------------------
    # Older payload compatibility
    # ------------------------------
    # Older schema versions may not have:
    # - raw_query
    # - task
    # - start_char / end_char
    # - answer_span_found
    # - canonical_intent_id
    raw_query = d.get("raw_query")
    task = d.get("task")
    if task is None:
        # Fallback to older meta conventions if available.
        task = (d.get("meta") or {}).get("task", "default")

    # ------------------------------
    # Rebuild MemoryItem
    # ------------------------------
    item = MemoryItem(
        key=d["key"],
        scope=Scope(d["scope"]),
        namespace=d["namespace"],
        query_canonical=d["query_canonical"],
        context_signature=d["context_signature"],
        answer_text=d["answer_text"],
        raw_query=raw_query,
        task=task,
        question_type=d.get("question_type"),
        provenance=provenance,
        quality=quality,
        created_at_utc=(
            _dt_from_str(d.get("created_at_utc"))
            or datetime.now(timezone.utc)
        ),
        ttl_seconds=d.get("ttl_seconds"),
        expires_at_utc=_dt_from_str(d.get("expires_at_utc")),
        stats=stats,
        meta=dict(d.get("meta") or {}),
        evidence_text=d.get("evidence_text"),
        doc_signature=d.get("doc_signature"),
        source_file=d.get("source_file"),
        source_id=d.get("source_id"),
        chunk_index=chunk_index,
        chunk_id=d.get("chunk_id"),
        start_char=start_char,
        end_char=end_char,
        answer_canonical=d.get("answer_canonical"),
        answer_span_found=d.get("answer_span_found"),
        canonical_intent_id=d.get("canonical_intent_id"),
        query_embedding=query_embedding,
        embedding_model_id=embedding_model_id,
        embedding_norm=embedding_norm,
    )

    # MemoryItem may internally coerce embeddings to array('f').
    # For portability and stable public behavior, convert back to a plain list.
    item.query_embedding = _normalize_embedding(item.query_embedding)

    if item.embedding_norm is not None:
        item.embedding_norm = float(item.embedding_norm)

    return item


# =============================================================================
# Stats container
# =============================================================================

class DiskStoreStats:
    """
    Lightweight in-process stats for observability and tests.
    """

    def __init__(self) -> None:
        self.gets = 0
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.deletes = 0
        self.iter_calls = 0
        self.iter_candidate_calls = 0


# =============================================================================
# SQLite-backed disk store
# =============================================================================

class DiskStoreSQLite:
    """
    Simple SQLite-backed persistence layer for MemoryItem records.

    Design goals:
    - deterministic behavior
    - easy debugging
    - backward compatibility with older payload versions
    - small surface area for RAM/disk manager integration
    """

    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("path must be non-empty")

        self._path = str(Path(path))
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._path, timeout=30.0, check_same_thread=False)

        # WAL helps concurrent-ish read/write usage and is generally safer for
        # long-running benchmark workflows.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")

        self._create_tables()
        self._stats = DiskStoreStats()

    def _create_tables(self) -> None:
        """
        Create the simple key-value table used to store serialized MemoryItem
        payloads. We keep this intentionally minimal because most filtering is
        still done in Python after deserialization.
        """
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
              namespace TEXT NOT NULL,
              key TEXT NOT NULL,
              value_json TEXT NOT NULL,
              updated_at_utc TEXT NOT NULL,
              PRIMARY KEY (namespace, key)
            );
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_kv_namespace
            ON kv(namespace);
            """
        )
        self._conn.commit()

    def stats(self) -> Dict[str, int]:
        """
        Return lightweight store statistics for tests and analysis.
        """
        s = self._stats
        return {
            "gets": s.gets,
            "hits": s.hits,
            "misses": s.misses,
            "puts": s.puts,
            "deletes": s.deletes,
            "iter_calls": s.iter_calls,
            "iter_candidate_calls": s.iter_candidate_calls,
        }

    def get(self, namespace: str, key: str) -> Optional[MemoryItem]:
        """
        Fetch a single item by exact namespace + key.
        """
        self._stats.gets += 1

        if not namespace or not key:
            self._stats.misses += 1
            return None

        cur = self._conn.execute(
            "SELECT value_json FROM kv WHERE namespace=? AND key=?;",
            (namespace, key),
        )
        row = cur.fetchone()
        if row is None:
            self._stats.misses += 1
            return None

        self._stats.hits += 1
        item = _deserialize_item(row[0])

        # Touch the in-memory object for callers that inspect access stats
        # after retrieval. Note that this does not automatically write back.
        item.stats.touch()
        return item

    def put(self, namespace: str, key: str, item: MemoryItem) -> None:
        """
        Insert or replace an item in the namespace.
        """
        if not namespace or not key:
            raise ValueError("namespace and key must be non-empty")
        if item.key != key:
            raise ValueError("key must match item.key")

        self._stats.puts += 1
        payload = _serialize_item(item)
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO kv(namespace, key, value_json, updated_at_utc)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
              value_json=excluded.value_json,
              updated_at_utc=excluded.updated_at_utc;
            """,
            (namespace, key, payload, now),
        )
        self._conn.commit()

    def delete(self, namespace: str, key: str) -> None:
        """
        Delete an item if it exists.
        """
        if not namespace or not key:
            return

        self._stats.deletes += 1
        self._conn.execute(
            "DELETE FROM kv WHERE namespace=? AND key=?;",
            (namespace, key),
        )
        self._conn.commit()

    def iter_namespace(self, namespace: str) -> Iterator[MemoryItem]:
        """
        Iterate all items in a namespace in stable updated-time order.
        """
        self._stats.iter_calls += 1

        if not namespace:
            return iter(())

        cur = self._conn.execute(
            "SELECT value_json FROM kv WHERE namespace=? ORDER BY updated_at_utc ASC;",
            (namespace,),
        )

        def _gen() -> Iterator[MemoryItem]:
            for (value_json,) in cur:
                yield _deserialize_item(value_json)

        return _gen()

    def iter_candidates(
        self,
        namespace: str,
        *,
        task: Optional[str] = None,
        source_file: Optional[str] = None,
        doc_signature: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[MemoryItem]:
        """
        Iterate candidate items from a namespace with lightweight Python-side
        filtering.

        Why Python-side filtering for now:
        - keeps the SQLite schema simple
        - supports backward-compatible payload evolution
        - is sufficient for the current benchmark scale

        Filters:
        - task
        - source_file
        - doc_signature
        """
        self._stats.iter_candidate_calls += 1

        if not namespace:
            return iter(())

        task_norm = str(task).strip() if task is not None else None
        source_norm = str(source_file).strip() if source_file is not None else None
        doc_norm = str(doc_signature).strip() if doc_signature is not None else None

        cur = self._conn.execute(
            "SELECT value_json FROM kv WHERE namespace=? ORDER BY updated_at_utc ASC;",
            (namespace,),
        )

        def _item_task(item: MemoryItem) -> Optional[str]:
            """
            Resolve task with backward compatibility.

            Important:
            - Many older/unit-test items store the real task only in item.meta
            - The typed schema field may still be the placeholder value "default"
            """
            typed = getattr(item, "task", None)
            typed_text = str(typed or "").strip()
            if typed_text and typed_text.lower() != "default":
                return typed_text

            value = item.meta.get("task")
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _item_source(item: MemoryItem) -> Optional[str]:
            if getattr(item, "source_file", None):
                text = str(item.source_file).strip()
                if text:
                    return text

            value = item.meta.get("source_file")
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _item_doc(item: MemoryItem) -> Optional[str]:
            if getattr(item, "doc_signature", None):
                text = str(item.doc_signature).strip()
                if text:
                    return text

            value = item.meta.get("doc_signature")
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _gen() -> Iterator[MemoryItem]:
            yielded = 0

            for (value_json,) in cur:
                item = _deserialize_item(value_json)

                # Filter by task if requested.
                if task_norm is not None:
                    itask = _item_task(item)
                    # If the item has a task and it does not match, reject it.
                    if itask is not None and itask != task_norm:
                        continue

                # Filter by source file if requested.
                if source_norm is not None:
                    isource = _item_source(item)
                    if isource is None or isource != source_norm:
                        continue

                # Filter by doc signature if requested.
                if doc_norm is not None:
                    idoc = _item_doc(item)
                    if idoc is None or idoc != doc_norm:
                        continue

                yield item
                yielded += 1

                if limit is not None and yielded >= int(limit):
                    break

        return _gen()

    def close(self) -> None:
        """
        Flush and close the SQLite connection.
        """
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    def __enter__(self) -> "DiskStoreSQLite":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


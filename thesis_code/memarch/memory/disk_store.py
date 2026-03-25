# memarch/memory/disk_store.py
"""
Tier 2: Disk store (portable, persistent) for MemoryItem objects.

Phase 1 goals:
- Cross-platform (macOS + Jetson Linux + other constrained devices)
- No external services/daemons
- Deterministic semantics matching RamStoreLRU interface:
    get(namespace, key) -> Optional[MemoryItem]
    put(namespace, key, item) -> None
    delete(namespace, key) -> None
    iter_namespace(namespace) -> Iterator[MemoryItem]
    stats() -> Dict[str, int]
    close() -> None

Implementation choice (Phase 1):
- SQLite via Python stdlib sqlite3 (no extra dependency).
- Store full MemoryItem as JSON (TEXT), with a small schema version.

Notes:
- This is not designed for multi-process concurrent writers. Single-process usage is assumed.
- SQLite WAL mode is enabled for better write performance.
- Semantic retrieval fields are stored as part of the serialized MemoryItem payload.
- Evidence-guided fields are also stored explicitly so disk-restored items behave
  the same as RAM-resident items.
"""

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


SCHEMA_VERSION = 3


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    return dt.astimezone(timezone.utc).isoformat()


def _dt_from_str(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _serialize_item(item: MemoryItem) -> str:
    """
    Convert MemoryItem to a JSON string.

    We avoid naive asdict() for enums/datetimes by doing controlled serialization.
    """
    d: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "key": item.key,
        "scope": item.scope.value,
        "namespace": item.namespace,
        "query_canonical": item.query_canonical,
        "context_signature": item.context_signature,
        "answer_text": item.answer_text,
        "provenance": {
            "model_id": item.provenance.model_id,
            "prompt_version": item.provenance.prompt_version,
            "generated_at_utc": _dt_to_str(item.provenance.generated_at_utc),
            "generator_backend": item.provenance.generator_backend,
            "quantization": item.provenance.quantization,
            "context_window": item.provenance.context_window,
        },
        "quality": {
            "score": item.quality.score,
            "success": item.quality.success,
            "metrics": item.quality.metrics,
        },
        "created_at_utc": _dt_to_str(item.created_at_utc),
        "ttl_seconds": item.ttl_seconds,
        "expires_at_utc": _dt_to_str(item.expires_at_utc),
        "stats": {
            "access_count": item.stats.access_count,
            "last_access_utc": _dt_to_str(item.stats.last_access_utc),
        },
        "meta": item.meta,

        # Evidence-guided fields
        "evidence_text": item.evidence_text,
        "doc_signature": item.doc_signature,
        "source_file": item.source_file,
        "chunk_index": item.chunk_index,
        "chunk_id": item.chunk_id,
        "question_type": item.question_type,
        "answer_canonical": item.answer_canonical,

        # Semantic retrieval fields
        "query_embedding": item.query_embedding,
        "embedding_model_id": item.embedding_model_id,
        "embedding_norm": item.embedding_norm,
    }
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deserialize_item(payload: str) -> MemoryItem:
    d = json.loads(payload)
    sv = int(d.get("schema_version", 1))

    if sv not in (1, 2, 3):
        raise ValueError(
            f"Unsupported schema_version: {sv} (expected one of 1, 2, 3)"
        )

    prov_d = d["provenance"]
    provenance = Provenance(
        model_id=prov_d["model_id"],
        prompt_version=prov_d["prompt_version"],
        generated_at_utc=_dt_from_str(prov_d.get("generated_at_utc")) or datetime.now(timezone.utc),
        generator_backend=prov_d.get("generator_backend"),
        quantization=prov_d.get("quantization"),
        context_window=prov_d.get("context_window"),
    )

    qd = d.get("quality", {})
    quality = QualitySignals(
        score=qd.get("score"),
        success=qd.get("success"),
        metrics=dict(qd.get("metrics") or {}),
    )

    sd = d.get("stats", {})
    stats = AccessStats(
        access_count=int(sd.get("access_count", 0)),
        last_access_utc=_dt_from_str(sd.get("last_access_utc")),
    )

    query_embedding = d.get("query_embedding")
    if query_embedding is not None:
        query_embedding = [float(x) for x in query_embedding]

    embedding_model_id = d.get("embedding_model_id")
    embedding_norm = d.get("embedding_norm")
    if embedding_norm is not None:
        embedding_norm = float(embedding_norm)

    chunk_index = d.get("chunk_index")
    if chunk_index is not None:
        chunk_index = int(chunk_index)

    item = MemoryItem(
        key=d["key"],
        scope=Scope(d["scope"]),
        namespace=d["namespace"],
        query_canonical=d["query_canonical"],
        context_signature=d["context_signature"],
        answer_text=d["answer_text"],
        provenance=provenance,
        quality=quality,
        created_at_utc=_dt_from_str(d.get("created_at_utc")) or datetime.now(timezone.utc),
        ttl_seconds=d.get("ttl_seconds"),
        expires_at_utc=_dt_from_str(d.get("expires_at_utc")),
        stats=stats,
        meta=dict(d.get("meta") or {}),

        # Evidence-guided fields
        evidence_text=d.get("evidence_text"),
        doc_signature=d.get("doc_signature"),
        source_file=d.get("source_file"),
        chunk_index=chunk_index,
        chunk_id=d.get("chunk_id"),
        question_type=d.get("question_type"),
        answer_canonical=d.get("answer_canonical"),

        # Semantic retrieval fields
        query_embedding=query_embedding,
        embedding_model_id=embedding_model_id,
        embedding_norm=embedding_norm,
    )
    return item


class DiskStoreStats:
    def __init__(self) -> None:
        self.gets = 0
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.deletes = 0
        self.iter_calls = 0


class DiskStoreSQLite:
    """
    SQLite-backed persistent key-value store for MemoryItem, partitioned by namespace.

    Table:
      kv(namespace TEXT, key TEXT, value_json TEXT, updated_at_utc TEXT, PRIMARY KEY(namespace,key))
    """

    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("path must be non-empty")

        self._path = str(Path(path))
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._path, timeout=30.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")

        self._create_tables()
        self._stats = DiskStoreStats()

    def _create_tables(self) -> None:
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
        s = self._stats
        return {
            "gets": s.gets,
            "hits": s.hits,
            "misses": s.misses,
            "puts": s.puts,
            "deletes": s.deletes,
            "iter_calls": s.iter_calls,
        }

    def get(self, namespace: str, key: str) -> Optional[MemoryItem]:
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

        # Touch access stats on read, mirroring RAM-store behavior in spirit.
        item.stats.touch()
        return item

    def put(self, namespace: str, key: str, item: MemoryItem) -> None:
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
        Iterate all items in a namespace.

        Phase 1 semantic retrieval may use this for bounded brute-force scanning.
        """
        self._stats.iter_calls += 1
        if not namespace:
            return iter(())

        cur = self._conn.execute(
            "SELECT value_json FROM kv WHERE namespace=? ORDER BY updated_at_utc ASC;",
            (namespace,),
        )
        for (value_json,) in cur:
            yield _deserialize_item(value_json)

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    def __enter__(self) -> "DiskStoreSQLite":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
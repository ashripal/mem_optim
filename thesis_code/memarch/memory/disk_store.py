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

SCHEMA_VERSION = 4


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

        "evidence_text": item.evidence_text,
        "doc_signature": item.doc_signature,
        "source_file": item.source_file,
        "source_id": item.source_id,
        "chunk_index": item.chunk_index,
        "chunk_id": item.chunk_id,
        "question_type": item.question_type,
        "answer_canonical": item.answer_canonical,

        "query_embedding": item.query_embedding,
        "embedding_model_id": item.embedding_model_id,
        "embedding_norm": item.embedding_norm,
    }
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def _deserialize_item(payload: str) -> MemoryItem:
    d = json.loads(payload)

    prov_d = d["provenance"]
    provenance = Provenance(
        model_id=prov_d["model_id"],
        prompt_version=prov_d["prompt_version"],
        generated_at_utc=_dt_from_str(prov_d.get("generated_at_utc")) or datetime.now(timezone.utc),
        generator_backend=prov_d.get("generator_backend"),
        quantization=prov_d.get("quantization"),
        context_window=prov_d.get("context_window"),
    )

    quality = QualitySignals(**(d.get("quality") or {}))
    stats = AccessStats(**(d.get("stats") or {}))

    return MemoryItem(
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
        evidence_text=d.get("evidence_text"),
        doc_signature=d.get("doc_signature"),
        source_file=d.get("source_file"),
        source_id=d.get("source_id"),
        chunk_index=d.get("chunk_index"),
        chunk_id=d.get("chunk_id"),
        question_type=d.get("question_type"),
        answer_canonical=d.get("answer_canonical"),
        query_embedding=d.get("query_embedding"),
        embedding_model_id=d.get("embedding_model_id"),
        embedding_norm=d.get("embedding_norm"),
    )


class DiskStoreSQLite:
    def __init__(self, path: str, *, autocommit_every: int = 32) -> None:
        self._path = str(Path(path))
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._path, timeout=30.0, check_same_thread=False)

        # 🔥 Jetson-optimized pragmas
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA cache_size=-20000;")  # ~20MB
        self._conn.execute("PRAGMA mmap_size=268435456;")  # 256MB

        self._create_tables()

        self._autocommit_every = max(1, int(autocommit_every))
        self._pending_writes = 0

    def _maybe_commit(self):
        self._pending_writes += 1
        if self._pending_writes >= self._autocommit_every:
            self._conn.commit()
            self._pending_writes = 0

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS kv (
              namespace TEXT NOT NULL,
              key TEXT NOT NULL,
              value_json TEXT NOT NULL,
              updated_at_utc TEXT NOT NULL,
              PRIMARY KEY (namespace, key)
            );
        """)
        self._conn.commit()

    def get(self, namespace: str, key: str) -> Optional[MemoryItem]:
        cur = self._conn.execute(
            "SELECT value_json FROM kv WHERE namespace=? AND key=?;",
            (namespace, key),
        )
        row = cur.fetchone()
        if not row:
            return None

        item = _deserialize_item(row[0])
        item.stats.touch()
        return item

    def put(self, namespace: str, key: str, item: MemoryItem) -> None:
        payload = _serialize_item(item)
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute("""
            INSERT INTO kv(namespace, key, value_json, updated_at_utc)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
              value_json=excluded.value_json,
              updated_at_utc=excluded.updated_at_utc;
        """, (namespace, key, payload, now))

        self._maybe_commit()

    def delete(self, namespace: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM kv WHERE namespace=? AND key=?;",
            (namespace, key),
        )
        self._maybe_commit()

    def iter_candidates(
        self,
        namespace: str,
        *,
        task: Optional[str] = None,
        source_file: Optional[str] = None,
        doc_signature: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[MemoryItem]:

        cur = self._conn.execute(
            "SELECT value_json FROM kv WHERE namespace=?;",
            (namespace,),
        )

        count = 0
        for (value_json,) in cur:
            # 🔥 lightweight pre-filter
            d = json.loads(value_json)

            if task and d.get("meta", {}).get("task") != task:
                continue
            if source_file and d.get("source_file") != source_file:
                continue
            if doc_signature and d.get("doc_signature") != doc_signature:
                continue

            yield _deserialize_item(value_json)

            count += 1
            if limit and count >= limit:
                break

    def close(self):
        self._conn.commit()
        self._conn.close()
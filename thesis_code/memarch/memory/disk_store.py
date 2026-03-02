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
    stats() -> DiskStoreStats
    close() -> None

Implementation choice (Phase 1):
- SQLite via Python stdlib sqlite3 (no extra dependency).
- Store full MemoryItem as JSON (TEXT), with a small schema version.

Notes:
- This is not designed for multi-process concurrent writers. Single-process usage is assumed.
- SQLite WAL mode is enabled for better write performance.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from memarch.memory.schema import (
    MemoryItem,
    Provenance,
    QualitySignals,
    AccessStats,
    Scope,
)


SCHEMA_VERSION = 1


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        # enforce UTC-awareness to avoid portability bugs
        raise ValueError("datetime must be timezone-aware (UTC)")
    return dt.astimezone(timezone.utc).isoformat()


def _dt_from_str(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # treat as UTC if missing tz (should not happen if we wrote it)
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _serialize_item(item: MemoryItem) -> str:
    """
    Convert MemoryItem to a JSON string.

    We avoid naive asdict() for enums/datetimes by doing a controlled serialization.
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
    }
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deserialize_item(payload: str) -> MemoryItem:
    d = json.loads(payload)
    # Backward/forward compatibility hooks
    sv = int(d.get("schema_version", 0))
    if sv != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {sv} (expected {SCHEMA_VERSION})")

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

        # check_same_thread=False allows usage across threads if you later add a lock;
        # for Phase 1, single-thread/single-process is assumed.
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
        # touch access stats on read (like RAM store)
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
        self._conn.execute("DELETE FROM kv WHERE namespace=? AND key=?;", (namespace, key))
        self._conn.commit()

    def iter_namespace(self, namespace: str) -> Iterator[MemoryItem]:
        """
        Iterate all items in a namespace.

        Use sparingly (e.g., future index building). Not needed for Phase 1 exact-match.
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
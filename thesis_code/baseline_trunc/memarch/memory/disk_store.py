from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class DiskRecord:
    key: str
    question: str
    answer: str
    # embedding stored as bytes (packed float32) or None
    embedding: Optional[bytes]
    meta_json: str  # JSON string


class DiskStore:
    """
    SQLite-backed store for QA + optional embeddings.
    Acts as Tier-2 persistence and the source-of-truth for cache warmup.
    """
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_store (
                key TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                embedding BLOB,
                meta_json TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON qa_store(updated_at);")
        self._conn.commit()

    def get(self, key: str) -> Optional[DiskRecord]:
        cur = self._conn.execute(
            "SELECT key, question, answer, embedding, meta_json FROM qa_store WHERE key=?;",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return DiskRecord(
            key=row[0],
            question=row[1],
            answer=row[2],
            embedding=row[3],
            meta_json=row[4],
        )

    def upsert(
        self,
        key: str,
        question: str,
        answer: str,
        embedding: Optional[bytes],
        meta: Dict[str, Any],
        updated_at: int,
    ) -> None:
        meta_json = json.dumps(meta, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO qa_store(key, question, answer, embedding, meta_json, updated_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                question=excluded.question,
                answer=excluded.answer,
                embedding=excluded.embedding,
                meta_json=excluded.meta_json,
                updated_at=excluded.updated_at;
            """,
            (key, question, answer, embedding, meta_json, updated_at),
        )
        self._conn.commit()

    def get_embedding_blob(self, key: str) -> Optional[bytes]:
        cur = self._conn.execute("SELECT embedding FROM qa_store WHERE key=?;", (key,))
        row = cur.fetchone()
        return None if not row else row[0]

    def set_embedding_blob(self, key: str, embedding: bytes, updated_at: int) -> None:
        self._conn.execute(
            "UPDATE qa_store SET embedding=?, updated_at=? WHERE key=?;",
            (embedding, updated_at, key),
        )
        self._conn.commit()
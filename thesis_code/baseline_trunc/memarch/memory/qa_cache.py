from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class QACache:
    """
    Small RAM cache for exact (or canonicalized) question hits.
    Stores only: key -> (answer, meta_id)
    """
    max_size: int = 2048

    def __post_init__(self) -> None:
        self._store: Dict[str, Tuple[str, str]] = {}
        self._order: list[str] = []

    def get(self, key: str) -> Optional[Tuple[str, str]]:
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None

    def put(self, key: str, answer: str, meta_id: str = "") -> None:
        if key in self._store:
            self._order.remove(key)
        self._store[key] = (answer, meta_id)
        self._order.append(key)

        if len(self._store) > self.max_size:
            evict = self._order.pop(0)
            self._store.pop(evict, None)

    def __len__(self) -> int:
        return len(self._store)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class EmbedCache:
    """
    Small RAM cache for embeddings:
      key -> np.ndarray float32 shape (d,)
    """
    max_size: int = 4096

    def __post_init__(self) -> None:
        self._store: Dict[str, np.ndarray] = {}
        self._order: list[str] = []

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None

    def put(self, key: str, vec: np.ndarray) -> None:
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)

        if key in self._store:
            self._order.remove(key)
        self._store[key] = vec
        self._order.append(key)

        if len(self._store) > self.max_size:
            evict = self._order.pop(0)
            self._store.pop(evict, None)

    def __len__(self) -> int:
        return len(self._store)
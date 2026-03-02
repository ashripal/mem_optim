# tests/test_tier1_cache.py
"""
Unit tests for Tier 1 (RAM) LRUCache.

These tests are fast, deterministic, and do not touch disk/model/network.

Run:
  pytest -q
"""

from __future__ import annotations

import pytest

from baseline.tiers.tier1_cache import LRUCache


def test_get_miss_increments_misses_and_returns_default():
    c = LRUCache(capacity=2)

    assert c.get("missing") is None
    assert c.get("missing", default=123) == 123

    stats = c.snapshot_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 2
    assert stats["evictions"] == 0
    assert stats["size"] == 0


def test_put_and_get_hit_increments_hits():
    c = LRUCache(capacity=2)
    c.put("a", 1)

    assert c.get("a") == 1

    stats = c.snapshot_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    assert stats["evictions"] == 0
    assert stats["size"] == 1


def test_lru_eviction_removes_least_recently_used():
    c = LRUCache(capacity=2)

    # Fill
    c.put("a", "A")
    c.put("b", "B")
    assert c.get("a") == "A"  # 'a' becomes MRU, 'b' is now LRU

    # Add one more => should evict 'b'
    c.put("c", "C")

    assert c.get("b") is None
    assert c.get("a") == "A"
    assert c.get("c") == "C"

    stats = c.snapshot_stats()
    assert stats["evictions"] == 1
    # hits: get("a"), get("a"), get("c") = 3 hits total
    # misses: get("b") = 1 miss
    assert stats["hits"] == 3
    assert stats["misses"] == 1
    assert stats["size"] == 2


def test_put_existing_key_updates_value_and_promotes_without_eviction():
    c = LRUCache(capacity=2)

    c.put("a", 1)
    c.put("b", 2)

    # Update "a" (should not evict)
    c.put("a", 10)
    assert c.get("a") == 10

    # Now insert "c" -> should evict LRU which should be "b" (since "a" was promoted)
    c.put("c", 3)

    assert c.get("b") is None
    assert c.get("a") == 10
    assert c.get("c") == 3

    stats = c.snapshot_stats()
    assert stats["evictions"] == 1


def test_clear_resets_cache_and_stats():
    c = LRUCache(capacity=2)
    c.put("a", 1)
    _ = c.get("a")
    _ = c.get("missing")  # miss

    assert len(c) == 1
    stats_before = c.snapshot_stats()
    assert stats_before["hits"] == 1
    assert stats_before["misses"] == 1

    c.clear()

    assert len(c) == 0
    stats_after = c.snapshot_stats()
    assert stats_after["hits"] == 0
    assert stats_after["misses"] == 0
    assert stats_after["evictions"] == 0
    assert stats_after["size"] == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        _ = LRUCache(capacity=0)
    with pytest.raises(ValueError):
        _ = LRUCache(capacity=-5)
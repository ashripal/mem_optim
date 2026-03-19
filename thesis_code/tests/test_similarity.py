# tests/test_similarity.py
#
# Unit tests for memarch/memory/similarity.py
#
# These tests validate the low-level vector similarity helpers used by
# Phase 1 semantic retrieval:
#   - L2 norm
#   - dot product
#   - cosine similarity
#   - top-k semantic ranking
#   - score normalization
#
# Why these matter:
# - semantic retrieval correctness depends directly on these utilities
# - brute-force ranking must be deterministic and safe
# - low-level math should fail loudly where appropriate, while retrieval helpers
#   should remain robust to bad candidates

from __future__ import annotations

import pytest

from memarch.memory.similarity import (
    cosine_similarity,
    dot,
    l2_norm,
    normalize_scores,
    top_k_cosine,
    top_k_similar,
)


def test_l2_norm_of_basic_vector():
    assert l2_norm([3.0, 4.0]) == pytest.approx(5.0)


def test_l2_norm_of_zero_vector():
    assert l2_norm([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_dot_product_basic():
    assert dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)


def test_dot_product_raises_on_dimension_mismatch():
    with pytest.raises(ValueError):
        dot([1.0, 2.0], [1.0])


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_returns_zero_for_zero_norm_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == pytest.approx(0.0)


def test_cosine_similarity_raises_on_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_top_k_similar_returns_ranked_payloads():
    query = [1.0, 0.0]

    candidates = [
        ([1.0, 0.0], "exact"),
        ([0.8, 0.2], "close"),
        ([0.0, 1.0], "orthogonal"),
    ]

    ranked = top_k_similar(query, candidates, k=2)

    assert len(ranked) == 2
    assert ranked[0][1] == "exact"
    assert ranked[0][0] == pytest.approx(1.0)
    assert ranked[1][1] == "close"
    assert ranked[1][0] < ranked[0][0]


def test_top_k_similar_respects_min_score():
    query = [1.0, 0.0]

    candidates = [
        ([1.0, 0.0], "exact"),
        ([0.7, 0.7], "medium"),
        ([0.0, 1.0], "low"),
    ]

    ranked = top_k_similar(query, candidates, k=5, min_score=0.8)

    payloads = [payload for score, payload in ranked]
    assert "exact" in payloads
    assert "medium" not in payloads
    assert "low" not in payloads


def test_top_k_similar_skips_dimension_mismatched_candidates():
    query = [1.0, 0.0]

    candidates = [
        ([1.0, 0.0], "good"),
        ([1.0, 2.0, 3.0], "bad_dim"),
        ([0.5, 0.5], "also_good"),
    ]

    ranked = top_k_similar(query, candidates, k=5)

    payloads = [payload for score, payload in ranked]
    assert "good" in payloads
    assert "also_good" in payloads
    assert "bad_dim" not in payloads


def test_top_k_similar_returns_empty_for_nonpositive_k():
    query = [1.0, 0.0]
    candidates = [([1.0, 0.0], "x")]

    assert top_k_similar(query, candidates, k=0) == []
    assert top_k_similar(query, candidates, k=-1) == []


def test_top_k_similar_returns_empty_for_empty_query():
    candidates = [([1.0, 0.0], "x")]
    assert top_k_similar([], candidates, k=3) == []


def test_top_k_cosine_returns_key_score_pairs():
    query = [1.0, 0.0]
    candidates = [
        ("k1", [1.0, 0.0]),
        ("k2", [0.0, 1.0]),
    ]

    ranked = top_k_cosine(query, candidates, k=2)

    assert ranked[0][0] == "k1"
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[1][0] == "k2"
    assert ranked[1][1] == pytest.approx(0.0)


def test_normalize_scores_basic_case():
    scores = [2.0, 4.0, 6.0]
    normalized = normalize_scores(scores)

    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(1.0)


def test_normalize_scores_all_equal_returns_ones():
    scores = [3.0, 3.0, 3.0]
    normalized = normalize_scores(scores)

    assert normalized == [1.0, 1.0, 1.0]


def test_normalize_scores_empty_returns_empty():
    assert normalize_scores([]) == []
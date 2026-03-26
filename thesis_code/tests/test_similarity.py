# tests/test_similarity.py
#
# Unit tests for memarch/memory/similarity.py
#
# These tests validate:
#   - low-level vector similarity helpers used by semantic retrieval
#   - lexical retrieval helpers added for approximate matching
#
# Coverage includes:
#   - L2 norm
#   - dot product
#   - cosine similarity
#   - top-k semantic ranking
#   - score normalization
#   - Jaccard similarity
#   - token-F1 similarity
#   - lexical similarity scoring
#   - lexical top-k ranking
#
# Why these matter:
# - semantic retrieval correctness depends directly on these utilities
# - lexical retrieval quality depends directly on these overlap/scoring helpers
# - brute-force ranking must be deterministic and safe
# - low-level math should fail loudly where appropriate, while retrieval helpers
#   should remain robust to bad candidates

from __future__ import annotations

import pytest

from memarch.memory.similarity import (
    cosine_similarity,
    dot,
    jaccard_score,
    l2_norm,
    lexical_score,
    normalize_scores,
    token_f1_score,
    top_k_cosine,
    top_k_lexical,
    top_k_similar,
)


# -----------------------------------------------------------------------------
# Vector similarity tests
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Lexical similarity tests
# -----------------------------------------------------------------------------

def test_jaccard_score_identical_token_sets_is_one():
    score = jaccard_score(["who", "is", "the", "president"], ["who", "is", "the", "president"])
    assert score == pytest.approx(1.0)


def test_jaccard_score_both_empty_is_one():
    assert jaccard_score([], []) == pytest.approx(1.0)


def test_jaccard_score_one_empty_is_zero():
    assert jaccard_score(["president"], []) == pytest.approx(0.0)
    assert jaccard_score([], ["president"]) == pytest.approx(0.0)


def test_token_f1_score_identical_tokens_is_one():
    score = token_f1_score(["who", "is", "the", "president"], ["who", "is", "the", "president"])
    assert score == pytest.approx(1.0)


def test_token_f1_score_both_empty_is_one():
    assert token_f1_score([], []) == pytest.approx(1.0)


def test_token_f1_score_one_empty_is_zero():
    assert token_f1_score(["president"], []) == pytest.approx(0.0)
    assert token_f1_score([], ["president"]) == pytest.approx(0.0)


def test_lexical_score_identical_query_is_one_or_near_one():
    query_norm = "who is the president"
    query_tokens = ["who", "is", "the", "president"]

    score = lexical_score(
        query_norm=query_norm,
        query_tokens=query_tokens,
        item_norm="who is the president",
        item_tokens=["who", "is", "the", "president"],
        same_source=False,
    )

    assert score == pytest.approx(1.0)


def test_lexical_score_paraphrase_is_higher_than_unrelated():
    query_norm = "who is the president of the united states"
    query_tokens = ["who", "is", "the", "president", "of", "the", "united", "states"]

    paraphrase_score = lexical_score(
        query_norm=query_norm,
        query_tokens=query_tokens,
        item_norm="who is the us president",
        item_tokens=["who", "is", "the", "us", "president"],
        same_source=False,
    )

    unrelated_score = lexical_score(
        query_norm=query_norm,
        query_tokens=query_tokens,
        item_norm="what is the population of france",
        item_tokens=["what", "is", "the", "population", "of", "france"],
        same_source=False,
    )

    assert paraphrase_score > unrelated_score


def test_lexical_score_same_source_bonus_increases_score():
    query_norm = "what is the capital of france"
    query_tokens = ["what", "is", "the", "capital", "of", "france"]
    item_norm = "what is the capital city of france"
    item_tokens = ["what", "is", "the", "capital", "city", "of", "france"]

    score_without_bonus = lexical_score(
        query_norm=query_norm,
        query_tokens=query_tokens,
        item_norm=item_norm,
        item_tokens=item_tokens,
        same_source=False,
    )

    score_with_bonus = lexical_score(
        query_norm=query_norm,
        query_tokens=query_tokens,
        item_norm=item_norm,
        item_tokens=item_tokens,
        same_source=True,
    )

    assert score_with_bonus > score_without_bonus


def test_lexical_score_empty_text_edge_cases():
    assert lexical_score(
        query_norm="",
        query_tokens=[],
        item_norm="",
        item_tokens=[],
        same_source=False,
    ) == pytest.approx(1.0)

    assert lexical_score(
        query_norm="who is the president",
        query_tokens=["who", "is", "the", "president"],
        item_norm="",
        item_tokens=[],
        same_source=False,
    ) == pytest.approx(0.0)

    assert lexical_score(
        query_norm="",
        query_tokens=[],
        item_norm="who is the president",
        item_tokens=["who", "is", "the", "president"],
        same_source=False,
    ) == pytest.approx(0.0)


def test_top_k_lexical_returns_ranked_payloads():
    query_norm = "what is the capital of france"
    query_tokens = ["what", "is", "the", "capital", "of", "france"]

    candidates = [
        ("what is the capital of france", ["what", "is", "the", "capital", "of", "france"], "exact"),
        ("what is the capital city of france", ["what", "is", "the", "capital", "city", "of", "france"], "close"),
        ("who wrote hamlet", ["who", "wrote", "hamlet"], "unrelated"),
    ]

    ranked = top_k_lexical(query_norm, query_tokens, candidates, k=2)

    assert len(ranked) == 2
    assert ranked[0][1] == "exact"
    assert ranked[0][0] == pytest.approx(1.0)
    assert ranked[1][1] == "close"
    assert ranked[1][0] < ranked[0][0]


def test_top_k_lexical_respects_min_score():
    query_norm = "what is the capital of france"
    query_tokens = ["what", "is", "the", "capital", "of", "france"]

    candidates = [
        ("what is the capital of france", ["what", "is", "the", "capital", "of", "france"], "exact"),
        ("what is the population of france", ["what", "is", "the", "population", "of", "france"], "medium"),
        ("who wrote hamlet", ["who", "wrote", "hamlet"], "low"),
    ]

    ranked = top_k_lexical(query_norm, query_tokens, candidates, k=5, min_score=0.8)

    payloads = [payload for score, payload in ranked]
    assert "exact" in payloads
    assert "low" not in payloads


def test_top_k_lexical_returns_empty_for_nonpositive_k():
    query_norm = "what is the capital of france"
    query_tokens = ["what", "is", "the", "capital", "of", "france"]
    candidates = [
        ("what is the capital of france", ["what", "is", "the", "capital", "of", "france"], "x")
    ]

    assert top_k_lexical(query_norm, query_tokens, candidates, k=0) == []
    assert top_k_lexical(query_norm, query_tokens, candidates, k=-1) == []
# tests/test_embedder.py
#
# Unit tests for memarch/models/embedder.py
#
# These are unit tests, not real-model integration tests.
# They validate:
#   - device-independent wrapper behavior
#   - single-text and batch embedding APIs
#   - deterministic output shape/content with fake model/tokenizer
#   - normalization behavior
#   - embedding_dim / info helpers
#   - embedding_norm helper
#
# Why use fakes?
# - fast
# - deterministic
# - offline
# - isolates wrapper bugs from HF download / environment issues
#
# Real model loading tests, if needed, should live separately.

from __future__ import annotations

import torch
import pytest

from memarch.models.embedder import EmbedderConfig, HFEmbedder


# --------------------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------------------

class FakeTokenizer:
    """
    Minimal tokenizer stub for HFEmbedder.

    It returns attention masks so mean pooling can be exercised deterministically.
    """
    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ):
        batch_size = len(texts)
        seq_len = 3

        # Same fake tokenization for every input; enough for wrapper testing.
        input_ids = torch.tensor([[10, 11, 12]] * batch_size, dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]] * batch_size, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class FakeModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class FakeModel:
    """
    Minimal encoder stub.

    Produces deterministic hidden states of shape [batch, seq_len, hidden_dim].
    Mean pooling over the sequence dimension yields a predictable result.
    """
    def __init__(self):
        self.config = type("Config", (), {"hidden_size": 4})()

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        hidden_dim = 4

        # Every token embedding is identical, so mean pooling is easy to predict.
        token_vec = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        last_hidden_state = token_vec.repeat(batch_size, seq_len, 1)

        return FakeModelOutput(last_hidden_state=last_hidden_state)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _patch_embedder_deps(monkeypatch) -> None:
    monkeypatch.setattr(
        "memarch.models.embedder.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "memarch.models.embedder.AutoModel.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_embed_returns_single_vector(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu", normalize=False))
    vec = emb.embed("What is systems engineering?")

    assert isinstance(vec, list)
    assert len(vec) == 4
    assert vec == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_embed_batch_returns_multiple_vectors(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu", normalize=False, batch_size=2))
    vecs = emb.embed_batch(["q1", "q2", "q3"])

    assert isinstance(vecs, list)
    assert len(vecs) == 3
    assert all(isinstance(v, list) for v in vecs)
    assert all(len(v) == 4 for v in vecs)
    assert vecs[0] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert vecs[1] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert vecs[2] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_encode_and_embed_are_equivalent(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu", normalize=False))
    a = emb.embed("hello")
    b = emb.encode("hello")

    # assert a == pytest.approx(b)
    assert a == b


def test_encode_batch_and_embed_batch_are_equivalent(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu", normalize=False))
    a = emb.embed_batch(["a", "b"])
    b = emb.encode_batch(["a", "b"])

    # assert a == pytest.approx(b)
    assert a == b


def test_embed_batch_empty_input_returns_empty_list(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu"))
    assert emb.embed_batch([]) == []


def test_embed_batch_none_raises(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu"))
    with pytest.raises(ValueError):
        emb.embed_batch(None)


def test_embed_sanitizes_none_and_whitespace_inputs(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu", normalize=False))
    vec_none = emb.embed(None)
    vec_ws = emb.embed("   ")

    assert len(vec_none) == 4
    assert len(vec_ws) == 4
    assert vec_none == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert vec_ws == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_normalize_true_produces_unit_norm_vectors(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu", normalize=True))
    vec = emb.embed("normalize me")

    norm = emb.embedding_norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_normalize_false_preserves_raw_pooled_vector(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu", normalize=False))
    vec = emb.embed("raw vector")

    assert vec == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert emb.embedding_norm(vec) == pytest.approx((1**2 + 2**2 + 3**2 + 4**2) ** 0.5)


def test_embedding_dim_uses_model_config(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu"))
    assert emb.embedding_dim() == 4


def test_info_returns_expected_metadata(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    cfg = EmbedderConfig(
        model_id="fake-embedder-model",
        device="cpu",
        max_length=128,
        batch_size=7,
        normalize=True,
        local_files_only=True,
    )
    emb = HFEmbedder(cfg)

    info = emb.info()
    assert info["model_id"] == "fake-embedder-model"
    assert info["device"] == "cpu"
    assert info["max_length"] == 128
    assert info["batch_size"] == 7
    assert info["normalize"] is True
    assert info["embedding_dim"] == 4


def test_embedding_norm_helper(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu"))
    norm = emb.embedding_norm([3.0, 4.0])

    assert norm == pytest.approx(5.0)


def test_embedding_norm_raises_on_none(monkeypatch):
    _patch_embedder_deps(monkeypatch)

    emb = HFEmbedder(EmbedderConfig(device="cpu"))
    with pytest.raises(ValueError):
        emb.embedding_norm(None)
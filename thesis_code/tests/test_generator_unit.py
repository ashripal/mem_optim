# tests/test_generator_unit.py
#
# Unit tests for memarch/models/generator.py
#
# IMPORTANT:
# These are unit tests, not real-model integration tests.
# They use fake tokenizer/model objects to validate the wrapper logic only:
#   - prompt construction
#   - document context injection
#   - optional retrieved-memory injection
#   - provenance / quality packaging
#   - prompt recording via last_prompt
#
# Why use fakes here?
# - fast
# - deterministic
# - offline
# - isolates bugs in our code from bugs in model loading / environment setup
#
# Real-model tests should live in a separate file, e.g.:
#   tests/test_generator_integration.py

from __future__ import annotations

import torch

from memarch.memory.schema import (
    MatchType,
    MemoryHit,
    MemoryItem,
    MemoryQuery,
    Provenance,
    QualitySignals,
    Scope,
    SourceTier,
)
from memarch.models.generator import GeneratorConfig, HFGenerator
from memarch.utils.text import canonicalize, context_signature, make_key


# --------------------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------------------

class FakeTokenizer:
    """
    Minimal tokenizer stub that mimics the methods/attributes HFGenerator expects.
    """
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 99

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=2048, padding=False):
        return {
            "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "generated answer from fake model"


class FakeModel:
    """
    Minimal causal LM stub that mimics .generate().
    """
    def eval(self):
        return self

    def to(self, device):
        return self

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        generated = torch.tensor([[20, 21, 22]], dtype=torch.long)
        return torch.cat([input_ids, generated], dim=1)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _make_memory_hit(
    answer_text: str = "previous useful answer",
    *,
    match_type: MatchType = MatchType.EXACT,
    score: float = 1.0,
    semantic_rank: int | None = None,
) -> MemoryHit:
    """
    Construct a valid MemoryHit so we can test prompt injection
    of previously retrieved memory.
    """
    mq = MemoryQuery(
        raw_query="What is systems engineering?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "NASA handbook context"},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    q_can = canonicalize(mq.raw_query)
    ctx_sig = context_signature(mq.context)
    key = make_key(
        scope="user",
        namespace="user:u1",
        task=mq.task,
        model_id=mq.model_id,
        prompt_version=mq.prompt_version,
        query_canonical=q_can,
        context_sig=ctx_sig,
    )

    item = MemoryItem(
        key=key,
        scope=Scope.USER,
        namespace="user:u1",
        query_canonical=q_can,
        context_signature=ctx_sig,
        answer_text=answer_text,
        provenance=Provenance(
            model_id=mq.model_id,
            prompt_version=mq.prompt_version,
        ),
        quality=QualitySignals(success=True),
    )

    return MemoryHit(
        item=item,
        source_tier=SourceTier.RAM,
        match_type=match_type,
        score=score,
        semantic_rank=semantic_rank,
    )


def _patch_generator_deps(monkeypatch) -> None:
    """
    Monkeypatch HF model/tokenizer loading so we test only wrapper logic.
    """
    monkeypatch.setattr(
        "memarch.models.generator.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "memarch.models.generator.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_build_prompt_includes_document_context(monkeypatch):
    """
    Verifies the most important prompt invariant:
    dataset/document context from MemoryQuery must appear in the final prompt.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(GeneratorConfig(device="cpu"))

    mq = MemoryQuery(
        raw_query="What is systems engineering?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={
            "dataset_context": "SYSTEMS_ENGINEERING_MARKER",
            "doc_signature": "abc123",
        },
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    prompt = gen.build_prompt(mq)

    assert "DOCUMENT CONTEXT:" in prompt
    assert "SYSTEMS_ENGINEERING_MARKER" in prompt
    assert "CURRENT QUESTION:" in prompt
    assert "What is systems engineering?" in prompt
    assert "DOCUMENT SIGNATURE: abc123" in prompt
    assert "ANSWER:" in prompt


def test_build_prompt_can_include_retrieved_memory(monkeypatch):
    """
    Verifies optional injection of previously retrieved memory context.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(
        GeneratorConfig(
            device="cpu",
            include_retrieved_memory_context=True,
        )
    )

    mq = MemoryQuery(
        raw_query="What is systems engineering?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "NASA context"},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_memory_hit("A prior cached explanation")

    prompt = gen.build_prompt(mq, retrieved=hit)

    assert "RETRIEVED ANSWER FOR A SIMILAR EARLIER QUESTION:" in prompt
    assert "A prior cached explanation" in prompt
    assert "RETRIEVAL METADATA:" in prompt
    assert "match_type=exact" in prompt
    assert "source_tier=ram" in prompt
    assert "score=1.0000" in prompt
    assert "DOCUMENT CONTEXT:" in prompt
    assert "NASA context" in prompt


def test_build_prompt_for_semantic_hit_includes_safety_instruction(monkeypatch):
    """
    Semantic retrieved memory should be explicitly framed as advisory.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(
        GeneratorConfig(
            device="cpu",
            include_retrieved_memory_context=True,
        )
    )

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "Device manual context"},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_memory_hit(
        "Press the reset button.",
        match_type=MatchType.SEMANTIC,
        score=0.97,
        semantic_rank=1,
    )

    prompt = gen.build_prompt(mq, retrieved=hit)

    assert "RETRIEVED ANSWER FOR A SIMILAR EARLIER QUESTION:" in prompt
    assert "Press the reset button." in prompt
    assert "match_type=semantic" in prompt
    assert "score=0.9700" in prompt
    assert "semantic_rank=1" in prompt
    assert "Use the retrieved answer only if it is consistent" in prompt


def test_generate_returns_answer_provenance_and_quality(monkeypatch):
    """
    Verifies the generator's return contract:
      (answer_text, provenance, quality)
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(GeneratorConfig(device="cpu"))

    mq = MemoryQuery(
        raw_query="Define systems engineering.",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "NASA handbook excerpt"},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    answer, provenance, quality = gen.generate(mq)

    assert answer == "generated answer from fake model"
    assert provenance.generator_backend == "transformers"
    assert provenance.model_id == "mistral-7b-instruct"
    assert provenance.prompt_version == "v1"
    assert provenance.generated_at_utc.tzinfo is not None
    assert quality.success is True


def test_generate_with_semantic_retrieval_adds_quality_metric(monkeypatch):
    """
    Semantic retrieval score should be surfaced in quality metrics for logging/eval.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(GeneratorConfig(device="cpu"))

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "Device manual excerpt"},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_memory_hit(
        "Press the reset button.",
        match_type=MatchType.SEMANTIC,
        score=0.96,
        semantic_rank=1,
    )

    _answer, _provenance, quality = gen.generate(mq, retrieved=hit)

    assert quality.metrics["semantic_retrieval_score"] == 0.96


def test_generate_records_last_prompt(monkeypatch):
    """
    Verifies that the final prompt is saved to last_prompt.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(GeneratorConfig(device="cpu"))

    mq = MemoryQuery(
        raw_query="What is systems engineering?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "NASA handbook excerpt"},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    gen.generate(mq)

    assert gen.last_prompt is not None
    assert "NASA handbook excerpt" in gen.last_prompt
    assert "What is systems engineering?" in gen.last_prompt
    assert "CURRENT QUESTION:" in gen.last_prompt


def test_generator_sets_pad_token_when_missing(monkeypatch):
    """
    Verifies a small but important robustness behavior:
    if the tokenizer has no pad_token but does have eos_token,
    HFGenerator should set pad_token = eos_token.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(GeneratorConfig(device="cpu"))
    assert gen.tokenizer.pad_token == gen.tokenizer.eos_token


def test_info_returns_expected_metadata(monkeypatch):
    """
    Verifies the info() helper used for logging/debugging/demo display.
    """
    _patch_generator_deps(monkeypatch)

    cfg = GeneratorConfig(
        model_id="fake-local-model",
        device="cpu",
        max_input_length=1024,
        max_new_tokens=64,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
    )
    gen = HFGenerator(cfg)

    info = gen.info()
    assert info["model_id"] == "fake-local-model"
    assert info["device"] == "cpu"
    assert info["max_input_length"] == 1024
    assert info["max_new_tokens"] == 64
    assert info["temperature"] == 0.1
    assert info["top_p"] == 0.9
    assert info["do_sample"] is False
    assert info["include_retrieved_memory_context"] is True
    assert info["include_dataset_context"] is True
    assert info["include_doc_signature"] is True
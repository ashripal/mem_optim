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
#   - evidence-guided reduced-context prompting
#   - provenance / quality packaging
#   - prompt recording via last_prompt
#   - generation metadata emission
#   - decoding config plumbing
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

import pytest
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
    def __init__(self):
        self.last_generate_kwargs = None

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
        num_beams=1,
        early_stopping=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        self.last_generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }
        generated = torch.tensor([[20, 21, 22]], dtype=torch.long)
        return torch.cat([input_ids, generated], dim=1)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _make_memory_hit(
    answer_text: str = "previous useful answer",
    *,
    evidence_text: str | None = None,
    match_type: MatchType = MatchType.EXACT,
    score: float = 1.0,
    semantic_rank: int | None = None,
    doc_signature: str | None = "abc123",
    source_file: str | None = "source.jsonl",
    chunk_index: int | None = 2,
    chunk_id: str | None = "chunk-2",
    question_type: str | None = "qa",
    same_document: bool = True,
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
        context={"dataset_context": "NASA handbook context", "doc_signature": "abc123"},
        doc_signature="abc123",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        question_type="qa",
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
        evidence_text=evidence_text,
        doc_signature=doc_signature,
        source_file=source_file,
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        question_type=question_type,
        meta={
            "doc_signature": doc_signature,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "question_type": question_type,
            "evidence_text": evidence_text,
        },
    )

    return MemoryHit(
        item=item,
        source_tier=SourceTier.RAM,
        match_type=match_type,
        score=score,
        semantic_rank=semantic_rank,
        debug={
            "same_document": same_document,
            "document_relation": "same_document" if same_document else "different_document",
        },
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
        doc_signature="abc123",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    prompt = gen.build_prompt(mq)

    assert "CONTEXT:" in prompt
    assert "SYSTEMS_ENGINEERING_MARKER" in prompt
    assert "CURRENT QUESTION:" in prompt
    assert "What is systems engineering?" in prompt
    assert "DOCUMENT SIGNATURE: abc123" in prompt
    assert "FINAL ANSWER:" in prompt


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
    hit = _make_memory_hit(
        "A prior cached explanation",
        evidence_text="Systems engineering coordinates the design of complex systems.",
        match_type=MatchType.EXACT,
    )

    prompt = gen.build_prompt(mq, retrieved=hit)

    assert "RETRIEVED MEMORY SUPPORT:" in prompt
    assert "RETRIEVED EVIDENCE SNIPPET:" in prompt
    assert "Systems engineering coordinates the design of complex systems." in prompt
    assert "PRIOR RELATED ANSWER:" in prompt
    assert "A prior cached explanation" in prompt
    assert "match_type=exact" in prompt
    assert "source_tier=ram" in prompt
    assert "score=1.0000" in prompt
    assert "same_document=true" in prompt
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
        context={"dataset_context": "Device manual context", "doc_signature": "doc-1"},
        doc_signature="doc-1",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_memory_hit(
        "Press the reset button.",
        evidence_text="To restart the device, press and hold reset for ten seconds.",
        match_type=MatchType.SEMANTIC,
        score=0.97,
        semantic_rank=1,
        doc_signature="doc-1",
        same_document=True,
    )

    prompt = gen.build_prompt(mq, retrieved=hit)

    assert "RETRIEVED MEMORY SUPPORT:" in prompt
    assert "To restart the device, press and hold reset for ten seconds." in prompt
    assert "Press the reset button." in prompt
    assert "match_type=semantic" in prompt
    assert "score=0.9700" in prompt
    assert "semantic_rank=1" in prompt
    assert "Use the retrieved material only if it is consistent" in prompt


def test_build_prompt_uses_reduced_context_on_semantic_hit(monkeypatch):
    """
    On semantic support, the generator should prefer compact evidence context
    instead of dumping the full long context.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(
        GeneratorConfig(
            device="cpu",
            include_retrieved_memory_context=True,
            prefer_retrieved_evidence_context=True,
            reduce_context_on_semantic_hit=True,
            max_evidence_chars=120,
            max_local_context_chars=80,
            max_full_context_chars=500,
        )
    )

    long_context = "LONG_CONTEXT_BLOCK " * 80
    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={
            "dataset_context": long_context,
            "doc_signature": "doc-1",
            "evidence_text": "Current local context says use the front-panel reset button.",
        },
        doc_signature="doc-1",
        evidence_text="Current local context says use the front-panel reset button.",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_memory_hit(
        "Press the reset button.",
        evidence_text="Retrieved evidence says hold reset for ten seconds.",
        match_type=MatchType.SEMANTIC,
        score=0.96,
        semantic_rank=1,
        doc_signature="doc-1",
        same_document=True,
    )

    prompt = gen.build_prompt(mq, retrieved=hit)

    assert "CONTEXT:" in prompt
    assert "RETRIEVED EVIDENCE:" in prompt
    assert "Retrieved evidence says hold reset for ten seconds." in prompt
    assert "CURRENT LOCAL CONTEXT:" in prompt
    assert "Current local context says use the front-panel reset button." in prompt
    assert "LONG_CONTEXT_BLOCK LONG_CONTEXT_BLOCK LONG_CONTEXT_BLOCK" not in prompt

    assert gen.last_generation_meta is not None
    assert gen.last_generation_meta["reduced_context_used"] is True
    assert gen.last_generation_meta["retrieved_doc_signature_match"] is True
    assert gen.last_generation_meta["retrieved_evidence_chars"] is not None
    assert gen.last_generation_meta["full_context_chars"] > gen.last_generation_meta["final_context_chars"]


def test_build_prompt_uses_full_context_without_semantic_support(monkeypatch):
    """
    Without semantic support, the generator should use dataset context directly.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(
        GeneratorConfig(
            device="cpu",
            max_full_context_chars=500,
        )
    )

    mq = MemoryQuery(
        raw_query="What is systems engineering?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "FULL_CONTEXT_MARKER", "doc_signature": "abc123"},
        doc_signature="abc123",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    prompt = gen.build_prompt(mq, retrieved=None)

    assert "CONTEXT:" in prompt
    assert "FULL_CONTEXT_MARKER" in prompt
    assert gen.last_generation_meta is not None
    assert gen.last_generation_meta["reduced_context_used"] is False
    assert gen.last_generation_meta["full_context_chars"] == len("FULL_CONTEXT_MARKER")


def test_build_prompt_for_trec_includes_label_only_instruction(monkeypatch):
    """
    Classification-like tasks should enforce terse label-only outputs.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(GeneratorConfig(device="cpu"))

    mq = MemoryQuery(
        raw_query="What is the label for this question?",
        user_id="u1",
        session_id="s1",
        task="trec",
        context={"dataset_context": "TREC context"},
        question_type="classification",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    prompt = gen.build_prompt(mq)

    assert "You are a classifier for TREC coarse question types." in prompt
    assert "Valid labels: ABBR, DESC, ENTY, HUM, LOC, NUM." in prompt
    assert "Return exactly one label and nothing else." in prompt
    assert "Question: What is the label for this question?" in prompt
    assert "Label:" in prompt
    assert "OUTPUT RULES:" not in prompt


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
        context={"dataset_context": "Device manual excerpt", "doc_signature": "doc-1"},
        doc_signature="doc-1",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_memory_hit(
        "Press the reset button.",
        evidence_text="Hold reset for ten seconds.",
        match_type=MatchType.SEMANTIC,
        score=0.96,
        semantic_rank=1,
        doc_signature="doc-1",
        same_document=True,
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


def test_generate_records_reduced_context_metadata(monkeypatch):
    """
    Reduced-context prompting should be reflected in generation metadata.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(
        GeneratorConfig(
            device="cpu",
            include_retrieved_memory_context=True,
            prefer_retrieved_evidence_context=True,
            reduce_context_on_semantic_hit=True,
        )
    )

    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={
            "dataset_context": "LONG_CONTEXT " * 60,
            "doc_signature": "doc-1",
            "evidence_text": "Use the front-panel reset switch.",
        },
        doc_signature="doc-1",
        evidence_text="Use the front-panel reset switch.",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_memory_hit(
        "Press reset.",
        evidence_text="Hold reset for ten seconds.",
        match_type=MatchType.SEMANTIC,
        score=0.95,
        semantic_rank=1,
        doc_signature="doc-1",
        same_document=True,
    )

    _answer, _provenance, quality = gen.generate(mq, retrieved=hit)

    assert gen.last_generation_meta is not None
    assert gen.last_generation_meta["reduced_context_used"] is True
    assert gen.last_generation_meta["retrieved_doc_signature_match"] is True
    assert gen.last_generation_meta["retrieved_evidence_chars"] is not None
    assert gen.last_generation_meta["full_context_chars"] > gen.last_generation_meta["final_context_chars"]

    assert quality.metrics["reduced_context_used"] == 1.0
    assert quality.metrics["retrieved_same_document"] == 1.0
    assert quality.metrics["full_context_chars"] > quality.metrics["final_context_chars"]


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
        decoding_mode="greedy",
        num_beams=1,
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
    assert info["decoding_mode"] == "greedy"
    assert info["num_beams"] == 1
    assert info["temperature"] == 0.1
    assert info["top_p"] == 0.9
    assert info["do_sample"] is False
    assert info["include_retrieved_memory_context"] is True
    assert info["include_dataset_context"] is True
    assert info["include_doc_signature"] is True
    assert info["prefer_retrieved_evidence_context"] is True
    assert info["reduce_context_on_semantic_hit"] is True


def test_generate_records_decoding_metadata(monkeypatch):
    """
    Generation metadata should include decoding settings for reproducibility.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(
        GeneratorConfig(
            device="cpu",
            decoding_mode="beam",
            num_beams=4,
        )
    )

    mq = MemoryQuery(
        raw_query="Define systems engineering.",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "NASA handbook excerpt"},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    _answer, _provenance, _quality = gen.generate(mq)

    assert gen.last_generation_meta is not None
    assert gen.last_generation_meta["generation_backend"] == "hf_generate"
    assert gen.last_generation_meta["decoding_mode"] == "beam"
    assert gen.last_generation_meta["num_beams"] == 4
    assert gen.last_generation_meta["do_sample"] is False


def test_beam_mode_passes_num_beams_to_generate(monkeypatch):
    """
    Beam decoding should forward num_beams and early_stopping to HF generate.
    """
    _patch_generator_deps(monkeypatch)

    gen = HFGenerator(
        GeneratorConfig(
            device="cpu",
            decoding_mode="beam",
            num_beams=5,
        )
    )

    mq = MemoryQuery(
        raw_query="Define systems engineering.",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={"dataset_context": "NASA handbook excerpt"},
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    gen.generate(mq)

    assert getattr(gen.model, "last_generate_kwargs", None) is not None
    assert gen.model.last_generate_kwargs["num_beams"] == 5
    assert gen.model.last_generate_kwargs["do_sample"] is False
    assert gen.model.last_generate_kwargs["early_stopping"] is True


def test_invalid_beam_config_raises(monkeypatch):
    """
    Beam mode must require num_beams > 1.
    """
    _patch_generator_deps(monkeypatch)

    with pytest.raises(ValueError, match="num_beams > 1"):
        HFGenerator(
            GeneratorConfig(
                device="cpu",
                decoding_mode="beam",
                num_beams=1,
            )
        )


def test_invalid_sampling_config_raises(monkeypatch):
    """
    do_sample=True is only valid when decoding_mode='sample'.
    """
    _patch_generator_deps(monkeypatch)

    with pytest.raises(ValueError, match="do_sample=True"):
        HFGenerator(
            GeneratorConfig(
                device="cpu",
                decoding_mode="greedy",
                do_sample=True,
            )
        )
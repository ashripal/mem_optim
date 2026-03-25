# tests/test_generator_integration.py
#
# Real-model integration tests for memarch/models/generator.py
#
# Unlike the unit tests, these tests use an actual local Transformers model.
# That means they validate:
#   - real tokenizer loading
#   - real model loading
#   - real prompt tokenization
#   - real generation
#   - real provenance / quality packaging
#   - evidence-guided prompt construction
#   - reduced-context metadata emission
#   - decode-policy metadata plumbing
#
# IMPORTANT:
# These tests are intentionally opt-in and require a LOCAL model directory or a
# previously cached local model snapshot. They are designed to work offline.
#
# Usage:
#   MEMARCH_TEST_MODEL_PATH=/absolute/path/to/local/model pytest -q tests/test_generator_integration.py
#
# Recommended local test model choices:
#   - a very small causal LM for fast local tests
#   - not necessarily the same model you use for the full demo
#
# Why use a tiny model here?
#   - keeps tests reasonably fast
#   - avoids requiring large GPU memory
#   - still validates the real HFGenerator path end-to-end
#
# If MEMARCH_TEST_MODEL_PATH is not set, these tests are skipped.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from memarch.memory.schema import MatchType, MemoryHit, MemoryItem, MemoryQuery, Provenance, QualitySignals, Scope, SourceTier
from memarch.models.generator import GeneratorConfig, HFGenerator
from memarch.utils.text import canonicalize, context_signature, make_key


def _get_local_model_path() -> str:
    """
    Resolve the local model path for integration tests.

    We require the model to exist locally because the system is expected to run offline.
    """
    model_path = os.environ.get("MEMARCH_TEST_MODEL_PATH", "").strip()
    if not model_path:
        pytest.skip("MEMARCH_TEST_MODEL_PATH is not set; skipping real-model integration tests.")
    if not Path(model_path).exists():
        pytest.skip(f"Local model path does not exist: {model_path}")
    return model_path


def _make_semantic_hit(mq: MemoryQuery) -> MemoryHit:
    q_can = canonicalize("How do I reset the device?")
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
        answer_text="Press and hold the reset button for ten seconds.",
        provenance=Provenance(
            model_id=mq.model_id,
            prompt_version=mq.prompt_version,
        ),
        quality=QualitySignals(success=True),
        evidence_text="Hold the reset button for ten seconds to restart the device.",
        doc_signature=mq.doc_signature or mq.context.get("doc_signature"),
        source_file="manual.jsonl",
        chunk_index=4,
        chunk_id="reset-4",
        question_type="qa",
        meta={
            "doc_signature": mq.doc_signature or mq.context.get("doc_signature"),
            "source_file": "manual.jsonl",
            "chunk_index": 4,
            "chunk_id": "reset-4",
            "question_type": "qa",
            "evidence_text": "Hold the reset button for ten seconds to restart the device.",
        },
    )

    return MemoryHit(
        item=item,
        source_tier=SourceTier.RAM,
        match_type=MatchType.SEMANTIC,
        score=0.97,
        semantic_rank=1,
        debug={
            "same_document": True,
            "document_relation": "same_document",
        },
    )


@pytest.fixture(scope="module")
def local_generator() -> HFGenerator:
    """
    Create one real HFGenerator for the entire module to avoid repeated model loads.

    Configuration notes:
    - local_files_only=True ensures no network access is attempted
    - CPU is the safest default for portability in tests
    - do_sample=False keeps outputs more deterministic
    - keep max_new_tokens small so tests stay lightweight
    """
    model_path = _get_local_model_path()

    cfg = GeneratorConfig(
        model_id=model_path,
        device="cpu",
        local_files_only=True,
        max_input_length=1024,
        max_new_tokens=48,
        decoding_mode="greedy",
        num_beams=1,
        temperature=0.2,
        top_p=0.95,
        do_sample=False,
        include_dataset_context=True,
        include_retrieved_memory_context=True,
        prefer_retrieved_evidence_context=True,
        reduce_context_on_semantic_hit=True,
    )
    return HFGenerator(cfg)


def test_real_generator_build_prompt_includes_dataset_context(local_generator: HFGenerator):
    """
    Sanity check: the real generator should still build the explicit prompt structure
    for the no-retrieval path.
    """
    mq = MemoryQuery(
        raw_query="What is systems engineering?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={
            "dataset_context": "SYSTEMS_ENGINEERING_MARKER from the NASA handbook.",
            "doc_signature": "docsig_123",
        },
        doc_signature="docsig_123",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    prompt = local_generator.build_prompt(mq)

    assert "CONTEXT:" in prompt
    assert "SYSTEMS_ENGINEERING_MARKER" in prompt
    assert "CURRENT QUESTION:" in prompt
    assert "What is systems engineering?" in prompt
    assert "DOCUMENT SIGNATURE: docsig_123" in prompt
    assert "FINAL ANSWER:" in prompt


def test_real_generator_build_prompt_uses_reduced_context_on_semantic_hit(local_generator: HFGenerator):
    """
    With semantic support, the prompt should use compact retrieved/local evidence
    rather than the entire long context.
    """
    long_context = "LONG_CONTEXT_SEGMENT " * 100
    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={
            "dataset_context": long_context,
            "doc_signature": "docsig_reset",
            "evidence_text": "Local troubleshooting context says use the front-panel reset button.",
        },
        doc_signature="docsig_reset",
        evidence_text="Local troubleshooting context says use the front-panel reset button.",
        question_type="qa",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_semantic_hit(mq)

    prompt = local_generator.build_prompt(mq, retrieved=hit)

    assert "CONTEXT:" in prompt
    assert "RETRIEVED EVIDENCE:" in prompt
    assert "CURRENT LOCAL CONTEXT:" in prompt
    assert "Hold the reset button for ten seconds to restart the device." in prompt
    assert "Local troubleshooting context says use the front-panel reset button." in prompt
    assert "RETRIEVED MEMORY SUPPORT:" in prompt
    assert "same_document=true" in prompt
    assert "LONG_CONTEXT_SEGMENT LONG_CONTEXT_SEGMENT LONG_CONTEXT_SEGMENT" not in prompt


def test_real_generator_generate_returns_nonempty_text(local_generator: HFGenerator):
    """
    Core real integration test:
    - load real local model
    - tokenize real prompt
    - run real generation
    - ensure non-empty answer is returned
    """
    mq = MemoryQuery(
        raw_query="In one sentence, what is systems engineering?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={
            "dataset_context": (
                "Systems engineering is an interdisciplinary approach that governs "
                "the total technical and managerial effort required to transform a "
                "set of stakeholder needs into a systems solution."
            ),
            "doc_signature": "docsig_nasa",
        },
        doc_signature="docsig_nasa",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    answer, provenance, quality = local_generator.generate(mq)

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

    assert provenance.generator_backend == "transformers"
    assert provenance.prompt_version == "v1"
    assert provenance.model_id == "mistral-7b-instruct"
    assert provenance.context_window == local_generator.cfg.max_input_length

    assert quality.success is True


def test_real_generator_generate_with_semantic_hit_records_reduced_context_metadata(local_generator: HFGenerator):
    """
    Reduced-context prompting should be visible in generation metadata and quality metrics.
    """
    mq = MemoryQuery(
        raw_query="How do I restart the device?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={
            "dataset_context": "LONG_CONTEXT_SEGMENT " * 100,
            "doc_signature": "docsig_reset",
            "evidence_text": "Local troubleshooting context says use the front-panel reset button.",
        },
        doc_signature="docsig_reset",
        evidence_text="Local troubleshooting context says use the front-panel reset button.",
        question_type="qa",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )
    hit = _make_semantic_hit(mq)

    _answer, _provenance, quality = local_generator.generate(mq, retrieved=hit)

    assert local_generator.last_generation_meta is not None
    assert local_generator.last_generation_meta["used_retrieved_context"] is True
    assert local_generator.last_generation_meta["retrieved_match_type"] == "semantic"
    assert local_generator.last_generation_meta["retrieved_doc_signature_match"] is True
    assert local_generator.last_generation_meta["reduced_context_used"] is True
    assert local_generator.last_generation_meta["retrieved_evidence_chars"] is not None
    assert local_generator.last_generation_meta["full_context_chars"] > local_generator.last_generation_meta["final_context_chars"]

    assert quality.metrics["semantic_retrieval_score"] == pytest.approx(0.97, abs=1e-6)
    assert quality.metrics["reduced_context_used"] == 1.0
    assert quality.metrics["retrieved_same_document"] == 1.0
    assert quality.metrics["full_context_chars"] > quality.metrics["final_context_chars"]


def test_real_generator_records_last_prompt(local_generator: HFGenerator):
    """
    The demo depends on last_prompt being available as evidence that dataset context
    was fed into the model path.
    """
    marker = "UNIQUE_CONTEXT_MARKER_ABC123"

    mq = MemoryQuery(
        raw_query="What phrase appears in the context?",
        user_id="u1",
        session_id="s1",
        task="pdf_qa",
        context={
            "dataset_context": f"The required phrase is {marker}.",
            "doc_signature": "docsig_prompt",
        },
        doc_signature="docsig_prompt",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    _answer, _prov, _quality = local_generator.generate(mq)

    assert local_generator.last_prompt is not None
    assert marker in local_generator.last_prompt
    assert "CONTEXT:" in local_generator.last_prompt
    assert "CURRENT QUESTION:" in local_generator.last_prompt


def test_real_generator_info_reports_local_model_metadata(local_generator: HFGenerator):
    """
    Lightweight metadata check for logging/debugging.
    """
    info = local_generator.info()

    assert info["device"] == "cpu"
    assert info["max_input_length"] == local_generator.cfg.max_input_length
    assert info["max_new_tokens"] == local_generator.cfg.max_new_tokens
    assert info["decoding_mode"] == "greedy"
    assert info["num_beams"] == 1
    assert info["do_sample"] is False
    assert info["model_id"] == local_generator.cfg.model_id
    assert info["prefer_retrieved_evidence_context"] is True
    assert info["reduce_context_on_semantic_hit"] is True
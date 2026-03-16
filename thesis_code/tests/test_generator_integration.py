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

from memarch.memory.schema import MemoryQuery
from memarch.models.generator import GeneratorConfig, HFGenerator


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
        temperature=0.2,
        top_p=0.95,
        do_sample=False,
        include_dataset_context=True,
        include_retrieved_memory_context=True,
    )
    return HFGenerator(cfg)


def test_real_generator_build_prompt_includes_dataset_context(local_generator: HFGenerator):
    """
    Sanity check: the real generator should still build the same explicit prompt structure
    as the unit-tested wrapper path.
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
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    prompt = local_generator.build_prompt(mq)

    assert "DATASET CONTEXT:" in prompt
    assert "SYSTEMS_ENGINEERING_MARKER" in prompt
    assert "QUESTION:" in prompt
    assert "What is systems engineering?" in prompt
    assert "DOCUMENT SIGNATURE: docsig_123" in prompt


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

    # QualitySignals are intentionally simple in Phase 1, but success should reflect
    # that generation produced some output.
    assert quality.success is True


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
        model_id="mistral-7b-instruct",
        prompt_version="v1",
    )

    _answer, _prov, _quality = local_generator.generate(mq)

    assert local_generator.last_prompt is not None
    assert marker in local_generator.last_prompt
    assert "DATASET CONTEXT:" in local_generator.last_prompt
    assert "QUESTION:" in local_generator.last_prompt


def test_real_generator_info_reports_local_model_metadata(local_generator: HFGenerator):
    """
    Lightweight metadata check for logging/debugging.
    """
    info = local_generator.info()

    assert info["device"] == "cpu"
    assert info["max_input_length"] == local_generator.cfg.max_input_length
    assert info["max_new_tokens"] == local_generator.cfg.max_new_tokens
    assert info["do_sample"] is False
    assert info["model_id"] == local_generator.cfg.model_id
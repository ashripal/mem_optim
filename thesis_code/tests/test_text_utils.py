# tests/test_text_utils.py

import pytest

from memarch.utils.text import canonicalize, context_signature, make_key


def test_canonicalize_strips_and_collapses_whitespace():
    s = " \n  hello\t\tworld \r\n  "
    assert canonicalize(s) == "hello world"


def test_canonicalize_preserves_case_and_punctuation():
    s = "Hello, WORLD!"
    assert canonicalize(s) == "Hello, WORLD!"


def test_canonicalize_none_returns_empty_string():
    assert canonicalize(None) == ""  # type: ignore[arg-type]


def test_context_signature_deterministic_for_same_dict_different_key_order():
    ctx1 = {"a": 1, "b": {"x": 2, "y": 3}, "c": ["z", 4]}
    ctx2 = {"c": ["z", 4], "b": {"y": 3, "x": 2}, "a": 1}
    assert context_signature(ctx1) == context_signature(ctx2)


def test_context_signature_changes_when_context_changes():
    ctx1 = {"a": 1, "b": 2}
    ctx2 = {"a": 1, "b": 3}
    assert context_signature(ctx1) != context_signature(ctx2)


def test_context_signature_requires_json_serializable_context():
    ctx = {"bad": set([1, 2, 3])}
    with pytest.raises(TypeError):
        context_signature(ctx)


def test_make_key_is_deterministic():
    q = canonicalize("  test   query ")
    ctx_sig = context_signature({"dataset_context": "abc"})
    k1 = make_key(
        scope="user",
        namespace="user:u1",
        task="trec",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        query_canonical=q,
        context_sig=ctx_sig,
    )
    k2 = make_key(
        scope="user",
        namespace="user:u1",
        task="trec",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        query_canonical=q,
        context_sig=ctx_sig,
    )
    assert k1 == k2


@pytest.mark.parametrize(
    "field, value",
    [
        ("scope", "session"),
        ("namespace", "user:u2"),
        ("task", "other_task"),
        ("model_id", "other_model"),
        ("prompt_version", "v2"),
        ("query_canonical", "different query"),
        ("context_sig", "deadbeef"),
    ],
)
def test_make_key_changes_when_inputs_change(field, value):
    base = dict(
        scope="user",
        namespace="user:u1",
        task="trec",
        model_id="mistral-7b-instruct",
        prompt_version="v1",
        query_canonical=canonicalize("test query"),
        context_sig=context_signature({"dataset_context": "abc"}),
    )
    k_base = make_key(**base)

    modified = dict(base)
    modified[field] = value
    k_mod = make_key(**modified)

    assert k_base != k_mod


def test_make_key_validates_required_fields():
    q = canonicalize("test query")
    ctx_sig = context_signature({"dataset_context": "abc"})

    with pytest.raises(ValueError):
        make_key(
            scope="",
            namespace="user:u1",
            task="trec",
            model_id="mistral-7b-instruct",
            prompt_version="v1",
            query_canonical=q,
            context_sig=ctx_sig,
        )

    with pytest.raises(ValueError):
        make_key(
            scope="user",
            namespace="",
            task="trec",
            model_id="mistral-7b-instruct",
            prompt_version="v1",
            query_canonical=q,
            context_sig=ctx_sig,
        )

    with pytest.raises(ValueError):
        make_key(
            scope="user",
            namespace="user:u1",
            task="trec",
            model_id="mistral-7b-instruct",
            prompt_version="v1",
            query_canonical="",
            context_sig=ctx_sig,
        )

    with pytest.raises(ValueError):
        make_key(
            scope="user",
            namespace="user:u1",
            task="trec",
            model_id="mistral-7b-instruct",
            prompt_version="v1",
            query_canonical=q,
            context_sig="",
        )
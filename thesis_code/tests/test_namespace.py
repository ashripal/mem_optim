# tests/test_namespace.py

import pytest

from memarch.memory.namespace import (
    namespace_for,
    resolve_namespaces,
    split_namespace,
    default_scope_order,
)
from memarch.memory.schema import MemoryQuery, Scope


def test_default_scope_order_is_personalization_first():
    assert default_scope_order() == [Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL]


def test_namespace_for_session_requires_session_id():
    mq = MemoryQuery(raw_query="q", user_id="u1", session_id=None, task="trec")
    with pytest.raises(ValueError):
        namespace_for(Scope.SESSION, mq)


def test_namespace_for_user_requires_user_id():
    mq = MemoryQuery(raw_query="q", user_id=None, session_id="s1", task="trec")
    with pytest.raises(ValueError):
        namespace_for(Scope.USER, mq)


def test_namespace_for_cohort_requires_cohort_id():
    mq = MemoryQuery(raw_query="q", user_id="u1", session_id="s1", cohort_id=None, task="trec")
    with pytest.raises(ValueError):
        namespace_for(Scope.COHORT, mq)


def test_namespace_for_global_uses_task_default():
    mq = MemoryQuery(raw_query="q", task="trec")
    assert namespace_for(Scope.GLOBAL, mq) == "global:trec"

    mq2 = MemoryQuery(raw_query="q", task="")
    assert namespace_for(Scope.GLOBAL, mq2) == "global:default"


def test_resolve_namespaces_skips_missing_ids_by_default():
    mq = MemoryQuery(raw_query="q", user_id="u1", session_id=None, cohort_id=None, task="trec")
    resolved = resolve_namespaces(mq)  # include_missing=False default
    # should include USER then GLOBAL (SESSION/COHORT skipped)
    assert [r.scope for r in resolved] == [Scope.USER, Scope.GLOBAL]
    assert [r.namespace for r in resolved] == ["user:u1", "global:trec"]


def test_resolve_namespaces_include_missing_raises():
    mq = MemoryQuery(raw_query="q", user_id=None, session_id=None, cohort_id=None, task="trec")
    # SESSION is first in default order and should raise if include_missing=True
    with pytest.raises(ValueError):
        resolve_namespaces(mq, include_missing=True)


def test_resolve_namespaces_custom_scope_order():
    mq = MemoryQuery(raw_query="q", user_id="u1", session_id="s1", task="trec")
    resolved = resolve_namespaces(mq, scope_order=[Scope.GLOBAL, Scope.USER])
    assert [r.scope for r in resolved] == [Scope.GLOBAL, Scope.USER]
    assert [r.namespace for r in resolved] == ["global:trec", "user:u1"]


def test_split_namespace_valid():
    assert split_namespace("user:alice") == ("user", "alice")
    assert split_namespace("global:trec") == ("global", "trec")
    assert split_namespace("session:s1") == ("session", "s1")
    assert split_namespace("cohort:c1") == ("cohort", "c1")


def test_split_namespace_invalid():
    with pytest.raises(ValueError):
        split_namespace("")
    with pytest.raises(ValueError):
        split_namespace("user")          # missing colon
    with pytest.raises(ValueError):
        split_namespace(":alice")        # missing prefix
    with pytest.raises(ValueError):
        split_namespace("user:")         # missing value
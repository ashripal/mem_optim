# memarch/memory/namespace.py
"""
Namespace resolution and scope validation.

Why this exists:
- Prevent cross-user contamination
- Make personalization routing explicit and testable
- Keep namespace string formats consistent across RAM/Disk stores

Phase 1 behavior:
- Deterministic scope order: SESSION -> USER -> COHORT -> GLOBAL
- Namespaces are simple string prefixes:
    session:<session_id>
    user:<user_id>
    cohort:<cohort_id>
    global:<task>

This file should NOT do any retrieval; it only determines *where* to look.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from memarch.memory.schema import Scope, MemoryQuery


@dataclass(frozen=True)
class ResolvedNamespace:
    """A concrete namespace search target."""
    scope: Scope
    namespace: str


def namespace_for(scope: Scope, mq: MemoryQuery) -> str:
    """
    Build the namespace string for a given scope and MemoryQuery.

    Raises ValueError if required identifiers are missing.
    """
    if scope == Scope.SESSION:
        if not mq.session_id:
            raise ValueError("SESSION scope requires MemoryQuery.session_id")
        return f"session:{mq.session_id}"

    if scope == Scope.USER:
        if not mq.user_id:
            raise ValueError("USER scope requires MemoryQuery.user_id")
        return f"user:{mq.user_id}"

    if scope == Scope.COHORT:
        if not mq.cohort_id:
            raise ValueError("COHORT scope requires MemoryQuery.cohort_id")
        return f"cohort:{mq.cohort_id}"

    if scope == Scope.GLOBAL:
        # Global scope is still constrained by task/domain to avoid cross-task reuse.
        task = mq.task or "default"
        return f"global:{task}"

    # Defensive: should never happen because Scope is an Enum
    raise ValueError(f"Unhandled scope: {scope}")


def validate_scope_requirements(scope: Scope, mq: MemoryQuery) -> None:
    """
    Validate that MemoryQuery has the identifiers required for a given scope.

    Useful for early failures and more readable tests.
    """
    if scope == Scope.SESSION and not mq.session_id:
        raise ValueError("SESSION scope requires session_id")
    if scope == Scope.USER and not mq.user_id:
        raise ValueError("USER scope requires user_id")
    if scope == Scope.COHORT and not mq.cohort_id:
        raise ValueError("COHORT scope requires cohort_id")
    # GLOBAL requires nothing beyond task, which defaults to "default"


def default_scope_order() -> List[Scope]:
    """
    The personalization-first ordering for retrieval.

    Phase 1: deterministic and fixed.
    Later you can make this configurable in policy.py.
    """
    return [Scope.SESSION, Scope.USER, Scope.COHORT, Scope.GLOBAL]


def resolve_namespaces(
    mq: MemoryQuery,
    *,
    scope_order: Optional[List[Scope]] = None,
    include_missing: bool = False,
) -> List[ResolvedNamespace]:
    """
    Resolve all namespaces to search for a given query.

    Args:
      mq: MemoryQuery describing the request.
      scope_order: optional override ordering; defaults to personalization-first.
      include_missing: if False (default), skip scopes where required IDs are missing.
                       if True, raise on missing IDs.

    Returns:
      List of ResolvedNamespace entries, in search order.
    """
    order = scope_order or default_scope_order()
    resolved: List[ResolvedNamespace] = []

    for scope in order:
        try:
            ns = namespace_for(scope, mq)
        except ValueError:
            if include_missing:
                raise
            continue
        resolved.append(ResolvedNamespace(scope=scope, namespace=ns))

    return resolved


def split_namespace(namespace: str) -> Tuple[str, str]:
    """
    Split a namespace string into (prefix, value).

    Examples:
      "user:alice"   -> ("user", "alice")
      "global:trec"  -> ("global", "trec")

    Useful for debugging and for validating store records.
    """
    if not namespace or ":" not in namespace:
        raise ValueError(f"Invalid namespace: {namespace!r}")
    prefix, value = namespace.split(":", 1)
    if not prefix or not value:
        raise ValueError(f"Invalid namespace: {namespace!r}")
    return prefix, value
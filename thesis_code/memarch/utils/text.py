# memarch/utils/text.py
"""
Text normalization + stable keying utilities for memarch.

Design goals:
- Deterministic across platforms (Mac/Jetson/other Linux)
- Stable across runs
- Explicitly scoped (scope/namespace/task/model/prompt_version) to avoid unsafe reuse
- Lightweight (Phase 1 exact-match; Phase 2 semantic can reuse canonical forms)

IMPORTANT:
- Keep canonicalization conservative. Over-normalizing can cause incorrect cache hits.
- Always store the canonical text alongside the hash (done in MemoryItem) to aid debugging
  and detect rare collisions.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional


_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def canonicalize(text: str) -> str:
    """
    Canonicalize user text for exact-match keying.

    Phase 1 approach (conservative):
    - strip leading/trailing whitespace
    - collapse internal whitespace runs to a single space

    We do NOT:
    - lowercase by default (can change meaning in some domains)
    - remove punctuation
    - remove numbers

    If you later want more aggressive normalization, do it behind a config flag.
    """
    if text is None:
        return ""
    s = text.strip()
    s = _WS_RE.sub(" ", s)
    return s


def _stable_json_dumps(obj: Any) -> str:
    """
    Convert an object to a stable JSON string.

    Requirements:
    - deterministic ordering
    - no whitespace noise

    Notes:
    - Default json can't serialize sets/bytes; callers should ensure context
      is JSON-serializable (recommended across the pipeline).
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def context_signature(context: Mapping[str, Any]) -> str:
    """
    Produce a compact, deterministic fingerprint of structured context.

    Context should include only information that should influence reuse, e.g.:
    - device_type ("orin_nano", "mac_silicon")
    - tool state (error code, firmware version)
    - domain ("field_support", "customer_support")
    - (optional) a short summary of recent turns, not the full transcript

    Returns:
      hex sha256 digest of stable JSON encoding.
    """
    if context is None:
        context = {}
    try:
        payload = _stable_json_dumps(context)
    except TypeError as e:
        raise TypeError(
            "context_signature requires JSON-serializable context. "
            "Convert non-serializable types (e.g., set -> list) before calling."
        ) from e
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_key(
    *,
    scope: str,
    namespace: str,
    task: str,
    model_id: str,
    prompt_version: str,
    query_canonical: str,
    context_sig: str,
    algo_version: str = "k1",
) -> str:
    """
    Create a stable key for exact-match memory.

    The key is explicitly scoped to:
    - scope + namespace (personalization isolation)
    - task (avoid cross-task contamination)
    - model_id + prompt_version (avoid stale reuse after changes)
    - canonical query + context signature (what the user asked + relevant context)

    algo_version allows changing key structure later without breaking old stores.
    """
    if not scope or not namespace:
        raise ValueError("scope and namespace must be non-empty")
    if not task:
        task = "default"
    if not model_id:
        model_id = "unknown"
    if not prompt_version:
        prompt_version = "v1"
    if not query_canonical:
        raise ValueError("query_canonical must be non-empty")
    if not context_sig:
        raise ValueError("context_sig must be non-empty")

    # A single stable string; avoid JSON here for maximum portability/clarity.
    payload = (
        f"{algo_version}|"
        f"scope={scope}|ns={namespace}|task={task}|"
        f"model={model_id}|prompt={prompt_version}|"
        f"q={query_canonical}|ctx={context_sig}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_namespace(scope: str, *, session_id: Optional[str] = None,
                   user_id: Optional[str] = None, cohort_id: Optional[str] = None,
                   task: str = "default") -> str:
    """
    Convenience helper to build namespace strings consistently.

    You may also implement this in memory/namespace.py; keeping a small helper here
    can be useful for tests and debugging. If you prefer a single source of truth,
    delete this and keep namespace construction only in memory/namespace.py.
    """
    scope_l = (scope or "").lower().strip()
    if scope_l == "session":
        if not session_id:
            raise ValueError("session_id required for SESSION namespace")
        return f"session:{session_id}"
    if scope_l == "user":
        if not user_id:
            raise ValueError("user_id required for USER namespace")
        return f"user:{user_id}"
    if scope_l == "cohort":
        if not cohort_id:
            raise ValueError("cohort_id required for COHORT namespace")
        return f"cohort:{cohort_id}"
    if scope_l == "global":
        # global can be scoped to a task/domain
        return f"global:{task or 'default'}"
    raise ValueError(f"Unknown scope: {scope}")


def safe_preview(text: str, max_len: int = 120) -> str:
    """
    Utility for logging/debugging: return a compact preview of a text.

    Avoid leaking full content into logs if you later store sensitive data.
    """
    if text is None:
        return ""
    s = canonicalize(text)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"
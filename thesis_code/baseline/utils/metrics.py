# utils/metrics.py
"""
Metrics utilities for the LongBench baseline.

Baseline philosophy:
- Keep evaluation lightweight, fast, and dependency-minimal.
- Provide a small set of generic, reproducible metrics that work across tasks.
- Avoid task-specific scoring unless you explicitly add a mapping later.

This module provides:
- Token-level / character-level overlap metrics
- Simple normalization helpers
- A single `compute_basic_metrics()` entrypoint for per-example scoring

Note:
LongBench tasks can have different "correctness" definitions. For a baseline that
focuses on performance + truncation behavior, these generic text metrics are useful
without introducing heavy task-specific evaluation logic.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """
    Lowercase + strip + collapse whitespace.
    (Intentionally minimal; avoids aggressive normalization that could hide errors.)
    """
    if s is None:
        return ""
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    return s


def simple_tokenize(s: str) -> List[str]:
    """
    A lightweight tokenizer:
    - normalize whitespace
    - split on spaces
    """
    s = normalize_text(s)
    if not s:
        return []
    return s.split(" ")


def char_f1(pred: str, ref: str) -> float:
    """
    Character-level F1 based on multiset overlap.
    Useful when tokenization is unclear or outputs are short.

    Returns a value in [0, 1].
    """
    pred = normalize_text(pred)
    ref = normalize_text(ref)

    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    # Multiset counts
    from collections import Counter

    pc = Counter(pred)
    rc = Counter(ref)
    overlap = sum((pc & rc).values())

    precision = overlap / max(1, len(pred))
    recall = overlap / max(1, len(ref))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def token_f1(pred: str, ref: str) -> float:
    """
    Token-level F1 based on multiset overlap.

    Returns a value in [0, 1].
    """
    pt = simple_tokenize(pred)
    rt = simple_tokenize(ref)

    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0

    from collections import Counter

    pc = Counter(pt)
    rc = Counter(rt)
    overlap = sum((pc & rc).values())

    precision = overlap / max(1, len(pt))
    recall = overlap / max(1, len(rt))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    """
    Exact string match after minimal normalization. Returns 1.0 or 0.0.
    """
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


def contains_answer(pred: str, ref: str) -> float:
    """
    Returns 1.0 if normalized ref appears as a substring of normalized pred else 0.0.
    Useful for extractive-style tasks.
    """
    p = normalize_text(pred)
    r = normalize_text(ref)
    if not r:
        return 0.0
    return 1.0 if r in p else 0.0


def _extract_reference(example: Dict[str, Any]) -> str:
    """
    Extract a 'reference' answer from a LongBench-style example.

    LongBench variants sometimes use:
      - "answer"
      - "answers" (list)
      - "output"
      - "target"

    We keep it simple:
      - if list, use first element
      - else string-cast
    """
    for k in ("answer", "answers", "output", "target", "reference"):
        if k in example and example[k] is not None:
            v = example[k]
            if isinstance(v, list) and v:
                return str(v[0])
            return str(v)
    return ""


def compute_basic_metrics(
    prediction: str,
    example: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute generic baseline metrics comparing `prediction` to the example reference.

    Returns:
      {
        "ref_text": <reference>,
        "exact_match": ...,
        "contains_answer": ...,
        "token_f1": ...,
        "char_f1": ...,
      }
    """
    ref = _extract_reference(example)

    return {
        "ref_text": ref,
        "exact_match": exact_match(prediction, ref),
        "contains_answer": contains_answer(prediction, ref),
        "token_f1": token_f1(prediction, ref),
        "char_f1": char_f1(prediction, ref),
    }
# memarch/utils/metrics.py
"""
Metrics utilities for memarch.

Goals:
- Lightweight, dependency-free (pure Python)
- Useful for both per-example evaluation and aggregate run summaries
- Deterministic behavior
- Friendly to short-answer QA workloads such as SQuAD-style evaluation

We avoid pandas here to keep the core package small and portable.
(You can use pandas in scripts/analysis if you want.)
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------
# Basic numeric helpers
# ---------------------------------------------------------------------


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: List[float], p: float) -> float:
    """
    Compute percentile p in [0,100] using linear interpolation.

    Deterministic, no numpy dependency.
    """
    if not xs:
        return 0.0
    if p <= 0:
        return float(min(xs))
    if p >= 100:
        return float(max(xs))

    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return float(ys[f])
    d0 = ys[f] * (c - k)
    d1 = ys[c] * (k - f)
    return float(d0 + d1)


# ---------------------------------------------------------------------
# Text normalization + QA helpers
# ---------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", _safe_text(text)).strip()


def _lower_no_punc(text: str) -> str:
    text = _normalize_whitespace(text).lower()
    return "".join(ch for ch in text if ch not in string.punctuation)


def _normalize_answer(text: str) -> str:
    """
    SQuAD-style light normalization:
    - lowercase
    - remove punctuation
    - remove articles
    - collapse whitespace
    """
    text = _safe_text(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _tokenize_normalized(text: str) -> List[str]:
    norm = _normalize_answer(text)
    return norm.split() if norm else []


def _char_bag(text: str) -> List[str]:
    norm = _normalize_answer(text)
    return list(norm.replace(" ", "")) if norm else []


def _multiset_overlap_count(xs: List[str], ys: List[str]) -> int:
    if not xs or not ys:
        return 0
    counts: Dict[str, int] = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    overlap = 0
    for y in ys:
        c = counts.get(y, 0)
        if c > 0:
            overlap += 1
            counts[y] = c - 1
    return overlap


def _f1_from_lists(pred_items: List[str], ref_items: List[str]) -> float:
    if not pred_items and not ref_items:
        return 1.0
    if not pred_items or not ref_items:
        return 0.0

    overlap = _multiset_overlap_count(pred_items, ref_items)
    if overlap <= 0:
        return 0.0

    precision = overlap / len(pred_items)
    recall = overlap / len(ref_items)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _token_f1(pred: str, ref: str) -> float:
    return _f1_from_lists(_tokenize_normalized(pred), _tokenize_normalized(ref))


def _char_f1(pred: str, ref: str) -> float:
    return _f1_from_lists(_char_bag(pred), _char_bag(ref))


def _contains_answer(pred: str, ref: str) -> float:
    """
    Returns 1.0 if normalized reference appears inside normalized prediction,
    else 0.0.

    This is useful for short-answer QA when the model returns a slightly
    longer phrase that still contains the correct answer span.
    """
    pred_norm = _normalize_answer(pred)
    ref_norm = _normalize_answer(ref)
    if not ref_norm:
        return 0.0
    return 1.0 if ref_norm in pred_norm else 0.0


def _extract_reference_text(example: Dict[str, Any]) -> str:
    """
    Extract a single reference answer string from a dataset/example record.

    Preference order:
    - ref_text
    - answer
    - target
    - answers[0]
    """
    for key in ("ref_text", "answer", "target"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    answers = example.get("answers")
    if isinstance(answers, list) and answers:
        first = answers[0]
        if first is None:
            return ""
        return str(first).strip()

    return ""


def compute_basic_metrics(prediction: str, example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute lightweight per-example QA metrics.

    Returns:
      {
        "ref_text": str,
        "exact_match": float,
        "contains_answer": float,
        "token_f1": float,
        "char_f1": float,
      }

    Exact match is computed on normalized text.
    """
    pred = _safe_text(prediction)
    ref = _extract_reference_text(example)

    pred_norm = _normalize_answer(pred)
    ref_norm = _normalize_answer(ref)

    exact_match = 1.0 if pred_norm == ref_norm and ref_norm != "" else 0.0
    contains = _contains_answer(pred, ref)
    token_f1 = _token_f1(pred, ref)
    char_f1 = _char_f1(pred, ref)

    return {
        "ref_text": ref,
        "exact_match": float(exact_match),
        "contains_answer": float(contains),
        "token_f1": float(token_f1),
        "char_f1": float(char_f1),
    }


# ---------------------------------------------------------------------
# Aggregate run metrics
# ---------------------------------------------------------------------


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @staticmethod
    def from_samples(samples_ms: List[float]) -> "LatencyStats":
        return LatencyStats(
            count=len(samples_ms),
            mean_ms=mean(samples_ms),
            p50_ms=percentile(samples_ms, 50),
            p95_ms=percentile(samples_ms, 95),
            p99_ms=percentile(samples_ms, 99),
        )


@dataclass
class RunMetrics:
    num_examples: int
    num_used_memory: int
    hit_rate: float

    total: LatencyStats
    memory_lookup: LatencyStats
    generation_est: LatencyStats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_examples": self.num_examples,
            "num_used_memory": self.num_used_memory,
            "hit_rate": self.hit_rate,
            "total": self.total.__dict__,
            "memory_lookup": self.memory_lookup.__dict__,
            "generation_est": self.generation_est.__dict__,
        }


def compute_run_metrics(example_events: Iterable[Dict[str, Any]]) -> RunMetrics:
    """
    Compute aggregate metrics from per-example log events.

    Supported event shapes:
      1) Older shape:
         {
           "type": "example",
           "meta": {... "used_memory": bool ...},
           "timings_ms": {"total_ms": ..., "memory_lookup_ms": ..., "generation_ms_est": ...}
         }

      2) Current memarch example_result shape:
         {
           "type": "example_result",
           "used_memory": bool,
           "timings_ms": {...}
         }
    """
    total_ms: List[float] = []
    mem_ms: List[float] = []
    gen_ms: List[float] = []
    used_mem = 0
    n = 0

    for evt in example_events:
        evt_type = evt.get("type")
        if evt_type not in {"example", "example_result"}:
            continue

        n += 1

        if evt_type == "example":
            meta = evt.get("meta") or {}
            if bool(meta.get("used_memory")):
                used_mem += 1
        else:
            if bool(evt.get("used_memory")):
                used_mem += 1

        t = evt.get("timings_ms") or {}
        total_ms.append(float(t.get("total_ms", 0.0) or 0.0))
        mem_ms.append(float(t.get("memory_lookup_ms", 0.0) or 0.0))
        gen_ms.append(float(t.get("generation_ms_est", 0.0) or 0.0))

    hit_rate = (used_mem / n) if n else 0.0

    return RunMetrics(
        num_examples=n,
        num_used_memory=used_mem,
        hit_rate=float(hit_rate),
        total=LatencyStats.from_samples(total_ms),
        memory_lookup=LatencyStats.from_samples(mem_ms),
        generation_est=LatencyStats.from_samples(gen_ms),
    )
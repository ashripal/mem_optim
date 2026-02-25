# utils/text.py
"""
text.py

Text and prompt utilities for the baseline memory architecture.

Goals:
- Normalize questions so paraphrase detection & cache keys are stable.
- Provide a "soft key" normalization for near-duplicate matching.
- Build prompts consistently (plain or chat-template if tokenizer supports it).
- Provide safe token-counting helper.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# --- Normalization helpers ---

_PUNCT_FIX_RE = re.compile(r"\s+([?.!,;:])")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    """
    Normalize a question for display and for stable caching.
    - trims whitespace
    - collapses spaces
    - fixes " ?" spacing
    - ensures it ends with '?' or '.'
    """
    q = (q or "").strip()
    q = _MULTI_SPACE_RE.sub(" ", q)
    q = _PUNCT_FIX_RE.sub(r"\1", q)
    if q and q[-1] not in ("?", "."):
        q += "?"
    return q


def soft_key(s: str) -> str:
    """
    Aggressive normalization for near-duplicate detection:
    - lowercase
    - remove punctuation (keep alnum + space)
    - collapse spaces
    """
    s = (s or "").lower()
    s = _PUNCT_FIX_RE.sub(r"\1", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def build_plain_prompt(context: str, question: str) -> str:
    """
    Baseline prompt used by both:
    - "LLM-only" baseline (no memory retrieval)
    - memory-augmented pipeline (after selecting chunks / similar QAs)
    """
    context = context or ""
    question = normalize_question(question)
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n"
        "Answer (be concise):"
    )


def apply_chat_template_if_available(tokenizer: Any, user_text: str) -> str:
    """
    If tokenizer has apply_chat_template (chat/instruct models), format accordingly.
    Otherwise return raw user_text.

    This keeps your runner consistent across models.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return user_text


def count_tokens(tokenizer: Any, text: str, max_length: Optional[int] = None) -> int:
    """
    Count tokens without creating attention masks. If max_length provided, counts with truncation.
    """
    kwargs: Dict[str, Any] = dict(add_special_tokens=True, return_attention_mask=False)
    if max_length is not None:
        kwargs.update(dict(truncation=True, max_length=max_length))
    else:
        kwargs.update(dict(truncation=False))
    ids = tokenizer(text, **kwargs)["input_ids"]
    return len(ids)


def truncate_to_tokens(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, bool, int]:
    """
    Truncate text to max_tokens using tokenizer, returning:
      (truncated_text, was_truncated, used_tokens)

    This is helpful when you want a consistent input cap for local runs.
    """
    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors=None,
    )
    used = len(enc["input_ids"])
    was_truncated = used >= max_tokens
    # decode back to text for reproducible prompt content
    truncated_text = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
    return truncated_text, was_truncated, used
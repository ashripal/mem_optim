# demo/pdf/chunking.py
"""
demo/pdf/chunking.py

PDF text chunking + document signature utilities for the Streamlit demo.

Goals (Phase 1):
- Deterministic chunking (character based)
- Stable doc_signature used for cache invalidation
- Lightweight enough for laptop + Jetson Orin Nano

This module is *not* a RAG embedding index. It simply prepares context for the LLM path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------------------
# Core utilities
# --------------------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for stable hashing/chunking:
    - collapse all whitespace
    - strip leading/trailing
    """
    return " ".join((text or "").split()).strip()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


@dataclass(frozen=True)
class Chunk:
    """
    Chunk metadata (Phase 1: character offsets).
    page_hint is optional (future enhancement if extractor provides mapping).
    """
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    page_hint: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_hint": self.page_hint,
        }


def chunk_text_char(
    text: str,
    *,
    chunk_size_chars: int,
    overlap_chars: int,
    prefix: str = "c",
) -> List[Chunk]:
    """
    Deterministic character-based chunking.

    Parameters:
      text: normalized or raw text (we normalize internally)
      chunk_size_chars: size of each chunk in characters
      overlap_chars: overlap between adjacent chunks
      prefix: chunk id prefix

    Returns:
      list[Chunk]
    """
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_size_chars:
        # Overlap must be smaller than chunk size, else step becomes 0
        raise ValueError("overlap_chars must be < chunk_size_chars")

    text_n = normalize_text(text)
    n = len(text_n)
    if n == 0:
        return []

    step = max(1, chunk_size_chars - overlap_chars)

    chunks: List[Chunk] = []
    i = 0
    idx = 1
    while i < n:
        start = i
        end = min(n, i + chunk_size_chars)
        ch_text = text_n[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{prefix}{idx:05d}",
                text=ch_text,
                start_char=start,
                end_char=end,
                page_hint=None,
            )
        )
        idx += 1
        i += step

    return chunks


def build_doc_signature_from_chunks(chunks: List[Chunk]) -> str:
    """
    Stable document signature used for cache invalidation.

    We hash the *normalized chunk texts* concatenated in order.
    This changes if:
      - PDF text changes
      - chunking parameters change (size/overlap affects boundary text)
    which is what we want for this demo.
    """
    joined = "\n".join([normalize_text(c.text) for c in chunks])
    return sha256_text(joined)


def build_doc_signature_from_text(text: str) -> str:
    """
    Signature of full normalized text (alternate strategy).
    This is less sensitive to chunk boundary changes.
    """
    return sha256_text(normalize_text(text))


def build_chunks_and_signature(
    text: str,
    *,
    mode: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    One-stop helper for the demo.

    mode:
      - "full": return a single chunk with all text
      - "chunked": return multiple chunks using char-based chunking

    Returns:
      (chunks_as_dicts, doc_signature, stats)
    """
    mode = (mode or "").lower().strip()
    if mode not in {"full", "chunked"}:
        raise ValueError("mode must be one of {'full','chunked'}")

    text_n = normalize_text(text)

    if mode == "full":
        # Single chunk; useful for very small PDFs.
        chunks = [
            Chunk(
                chunk_id="c00001",
                text=text_n,
                start_char=0,
                end_char=len(text_n),
                page_hint=None,
            )
        ]
        doc_sig = build_doc_signature_from_text(text_n)
        return [c.to_dict() for c in chunks], doc_sig, {"num_chunks": 1, "num_chars": len(text_n), "mode": "full"}

    chunks = chunk_text_char(
        text_n,
        chunk_size_chars=int(chunk_size_chars),
        overlap_chars=int(overlap_chars),
    )
    doc_sig = build_doc_signature_from_chunks(chunks)

    return [c.to_dict() for c in chunks], doc_sig, {"num_chunks": len(chunks), "num_chars": len(text_n), "mode": "chunked"}
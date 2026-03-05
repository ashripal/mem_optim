# demo/pdf/extract.py
"""
demo/pdf/extract.py

PDF -> text extraction utilities for the Streamlit demo.

Design goals:
- Keep Phase 1 extraction lightweight and portable across:
    - MacBook (Apple Silicon/Intel)
    - Jetson Orin Nano (Linux/ARM)
- Provide clear stats + errors for demo robustness
- Avoid heavy dependencies unless needed

Default extractor:
- PyPDF2 (pure Python, commonly available)

Future upgrades (not implemented here):
- pdfplumber (better layout but heavier)
- pymupdf / fitz (fast + good, but native dependency)
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _collapse_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


@dataclass(frozen=True)
class ExtractResult:
    """
    Standardized extraction result.
    """
    ok: bool
    text: str
    stats: Dict[str, Any]
    error: Optional[str] = None


def extract_text_pypdf2(pdf_bytes: bytes) -> ExtractResult:
    """
    Extract text using PyPDF2.

    Returns normalized (whitespace-collapsed) text for stable hashing & chunking.
    """
    t0 = time.perf_counter()

    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return ExtractResult(
            ok=False,
            text="",
            stats={"extract_ms": (time.perf_counter() - t0) * 1000.0, "backend": "pypdf2"},
            error="PyPDF2 not installed. Install with: pip install PyPDF2",
        )

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")

        raw = "\n".join(pages_text)
        text = _collapse_ws(raw)

        return ExtractResult(
            ok=True,
            text=text,
            stats={
                "extract_ms": (time.perf_counter() - t0) * 1000.0,
                "backend": "pypdf2",
                "num_pages": len(reader.pages),
                "num_chars": len(text),
            },
            error=None,
        )
    except Exception as e:
        return ExtractResult(
            ok=False,
            text="",
            stats={"extract_ms": (time.perf_counter() - t0) * 1000.0, "backend": "pypdf2"},
            error=f"PDF extraction failed: {e}",
        )


def extract_pdf_text(pdf_bytes: bytes, *, backend: str = "pypdf2") -> Tuple[str, Dict[str, Any]]:
    """
    Convenience wrapper used by demo/app.py.

    Returns:
      (text, stats_dict)

    stats_dict always includes:
      - ok: bool
      - backend: str
      - extract_ms: float
      - optional error fields
    """
    backend = (backend or "").lower().strip()

    if backend == "pypdf2":
        res = extract_text_pypdf2(pdf_bytes)
        stats = dict(res.stats)
        stats["ok"] = res.ok
        if res.error:
            stats["error"] = res.error
        return res.text, stats

    # Unknown backend
    t0 = time.perf_counter()
    return (
        "",
        {
            "ok": False,
            "backend": backend,
            "extract_ms": (time.perf_counter() - t0) * 1000.0,
            "error": f"Unknown backend '{backend}'. Supported: pypdf2",
        },
    )
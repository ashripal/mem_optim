# demo/render/cards.py
"""
demo/render/cards.py

UI helpers for rendering "evidence cards" in Streamlit.

Why this exists:
- Keep demo/app.py readable and focused on orchestration
- Centralize a consistent, committee-friendly visual language:
    - Tier used (RAM/DISK/COMPUTE)
    - LLM bypass badge
    - Latency breakdown
    - Context proof + prompt preview
    - Namespace trace

This module should contain *no* business logic. It only renders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


# --------------------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------------------

def _fmt_ms(x: Any) -> str:
    try:
        return f"{float(x):.3f} ms"
    except Exception:
        return "—"


def _tier_label(tier: str) -> str:
    t = (tier or "").lower().strip()
    if t == "ram":
        return "RAM HIT"
    if t == "disk":
        return "DISK HIT"
    if t == "compute":
        return "COMPUTE"
    return (tier or "UNKNOWN").upper()


def _bypass_label(llm_bypassed: bool) -> str:
    return "✅ LLM BYPASSED" if llm_bypassed else "🔄 LLM CALLED"


def _bool_badge(v: bool) -> str:
    return "✅" if v else "❌"


# --------------------------------------------------------------------------------------
# Public rendering functions
# --------------------------------------------------------------------------------------

def render_turn_card(turn: Dict[str, Any]) -> None:
    """
    Render one turn's evidence.

    Expected fields (best-effort):
      - turn_id: int
      - source_tier: "ram" | "disk" | "compute"
      - llm_bypassed: bool
      - used_memory: bool
      - timings_ms: dict
      - question: str
      - answer: str
      - selected_chunk_ids: list[str]
      - doc_signature: str
      - context_preview: str
      - context_len: int
      - namespaces_checked: list[dict]
      - prompt_preview: str | None
      - stored: bool | None
      - stored_scopes: list[str]
    """
    turn_id = turn.get("turn_id", "?")
    tier = _tier_label(str(turn.get("source_tier", "unknown")))
    bypass = _bypass_label(bool(turn.get("llm_bypassed", False)))

    st.markdown(f"### Turn {turn_id}: **{tier}** · {bypass}")

    timings = turn.get("timings_ms") or {}
    st.write(
        {
            "selection_ms": _fmt_ms(timings.get("selection_ms")),
            "memory_lookup_ms": _fmt_ms(timings.get("memory_lookup_ms")),
            "generation_ms_est": _fmt_ms(timings.get("generation_ms_est")),
            "total_ms": _fmt_ms(timings.get("total_ms")),
        }
    )

    # --- Core Q/A ---
    st.markdown("**Question**")
    st.write(turn.get("question", ""))

    st.markdown("**Answer**")
    st.write(turn.get("answer", ""))

    # --- Context proof ---
    with st.expander("Context used (PDF chunks)"):
        st.write(
            {
                "doc_signature": turn.get("doc_signature"),
                "selected_chunk_ids": turn.get("selected_chunk_ids", []),
                "context_len": turn.get("context_len", 0),
            }
        )
        st.text(turn.get("context_preview", "") or "")

    # --- Namespace trace ---
    with st.expander("Namespaces checked (trace)"):
        ns = turn.get("namespaces_checked", []) or []
        if not ns:
            st.info("No namespace trace recorded for this turn (optional).")
        else:
            st.write(ns)

    # --- Prompt evidence (only on compute path) ---
    prompt_preview = turn.get("prompt_preview")
    if prompt_preview:
        with st.expander("Prompt preview (proof of context injection)"):
            st.text(prompt_preview)

    # --- Storage decision ---
    with st.expander("Storage decision"):
        stored = turn.get("stored")
        st.write(
            {
                "stored": stored,
                "stored_scopes": turn.get("stored_scopes", []),
                "used_memory": turn.get("used_memory"),
            }
        )


def render_summary_badges(summary: Dict[str, Any]) -> None:
    """
    Render a small set of badges that are useful in the sidebar or a top summary row.

    Suggested summary keys:
      - turns
      - memory_hits
      - misses_compute
      - llm_calls
      - hit_rate
    """
    turns = summary.get("turns", 0)
    hits = summary.get("memory_hits", 0)
    misses = summary.get("misses_compute", 0)
    llm_calls = summary.get("llm_calls", 0)
    hit_rate = summary.get("hit_rate", None)

    cols = st.columns(5)
    cols[0].metric("Turns", int(turns))
    cols[1].metric("Memory hits", int(hits))
    cols[2].metric("Compute misses", int(misses))
    cols[3].metric("LLM calls", int(llm_calls))
    if hit_rate is not None:
        try:
            cols[4].metric("Hit rate", f"{float(hit_rate)*100:.1f}%")
        except Exception:
            cols[4].metric("Hit rate", "—")
    else:
        cols[4].metric("Hit rate", "—")


def render_doc_summary(doc: Optional[Dict[str, Any]]) -> None:
    """
    Render a compact document summary (useful in sidebar or top panel).
    """
    if not doc:
        st.info("No PDF loaded yet.")
        return

    ingest = doc.get("ingest_stats") or {}
    st.write(
        {
            "pdf_name": doc.get("pdf_name"),
            "doc_signature": doc.get("doc_signature"),
            "num_chunks": ingest.get("num_chunks"),
            "num_chars": ingest.get("num_chars"),
            "num_pages": ingest.get("num_pages"),
        }
    )


def render_memory_stats(ram_stats: Optional[Dict[str, Any]], disk_stats: Optional[Dict[str, Any]]) -> None:
    """
    Render memory store stats in a compact format.
    """
    st.markdown("#### Memory stats")

    if ram_stats:
        st.write({"ram": ram_stats})
    else:
        st.write({"ram": "—"})

    if disk_stats:
        st.write({"disk": disk_stats})
    else:
        st.write({"disk": "—"})


def render_namespace_trace_table(namespaces_checked: List[Dict[str, Any]]) -> None:
    """
    Optional: a more structured trace table.
    Only call when you have trace events that share keys.

    Expected per-event keys (suggested):
      - scope, namespace, hit, tier
    """
    if not namespaces_checked:
        st.info("No namespace trace to display.")
        return
    st.table(namespaces_checked)
# demo/render/plots.py
"""
demo/render/plots.py

Plotting helpers for the Streamlit demo.

Constraints / style:
- Keep plots lightweight and readable (committee-friendly)
- Use Streamlit native charts when possible
- Avoid seaborn (project convention)
- Provide both quick summary stats and trend lines:
    - total latency per turn
    - hit/miss timeline
    - tier timeline (optional)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def compute_summary(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute high-level summary metrics from turn history.
    """
    n = len(turns)
    hits = sum(1 for t in turns if bool(t.get("used_memory")))
    misses = n - hits

    total_ms = [_safe_float((t.get("timings_ms") or {}).get("total_ms"), 0.0) for t in turns]
    avg_total = sum(total_ms) / max(1, n)

    # Rough p95 without numpy to keep dependencies minimal
    p95 = 0.0
    if total_ms:
        xs = sorted(total_ms)
        # 95th percentile index (nearest-rank)
        k = max(0, min(len(xs) - 1, int(round(0.95 * (len(xs) - 1)))))
        p95 = xs[k]

    return {
        "turns": n,
        "memory_hits": hits,
        "misses_compute": misses,
        "hit_rate": hits / max(1, n),
        "avg_total_ms": avg_total,
        "p95_total_ms": p95,
    }


def build_series(turns: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Convert turns into numeric series for plotting.

    Returns:
      dict with keys:
        - total_ms
        - memory_hit (0/1)
        - tier_ram (0/1)
        - tier_disk (0/1)
        - tier_compute (0/1)
        - selection_ms
        - memory_lookup_ms
        - generation_ms_est
    """
    total_ms: List[float] = []
    memory_hit: List[float] = []
    tier_ram: List[float] = []
    tier_disk: List[float] = []
    tier_compute: List[float] = []

    selection_ms: List[float] = []
    memory_lookup_ms: List[float] = []
    generation_ms_est: List[float] = []

    for t in turns:
        tm = t.get("timings_ms") or {}
        total_ms.append(_safe_float(tm.get("total_ms"), 0.0))
        selection_ms.append(_safe_float(tm.get("selection_ms"), 0.0))
        memory_lookup_ms.append(_safe_float(tm.get("memory_lookup_ms"), 0.0))
        generation_ms_est.append(_safe_float(tm.get("generation_ms_est"), 0.0))

        hit = 1.0 if bool(t.get("used_memory")) else 0.0
        memory_hit.append(hit)

        tier = str(t.get("source_tier", "")).lower().strip()
        tier_ram.append(1.0 if tier == "ram" else 0.0)
        tier_disk.append(1.0 if tier == "disk" else 0.0)
        tier_compute.append(1.0 if tier == "compute" else 0.0)

    return {
        "total_ms": total_ms,
        "memory_hit": memory_hit,
        "tier_ram": tier_ram,
        "tier_disk": tier_disk,
        "tier_compute": tier_compute,
        "selection_ms": selection_ms,
        "memory_lookup_ms": memory_lookup_ms,
        "generation_ms_est": generation_ms_est,
    }


def render_performance(turns: List[Dict[str, Any]]) -> None:
    """
    Render all performance plots + summary stats.
    """
    if not turns:
        st.info("No turns yet. Ask a question to generate performance plots.")
        return

    summary = compute_summary(turns)
    st.write(
        {
            "turns": summary["turns"],
            "memory_hits": summary["memory_hits"],
            "misses_compute": summary["misses_compute"],
            "hit_rate": round(float(summary["hit_rate"]), 3),
            "avg_total_ms": round(float(summary["avg_total_ms"]), 3),
            "p95_total_ms": round(float(summary["p95_total_ms"]), 3),
        }
    )

    series = build_series(turns)

    st.markdown("#### Latency over turns")
    st.line_chart(
        {
            "total_ms": series["total_ms"],
            "memory_lookup_ms": series["memory_lookup_ms"],
            "selection_ms": series["selection_ms"],
            "generation_ms_est": series["generation_ms_est"],
        }
    )

    st.markdown("#### Memory hits over turns")
    st.line_chart({"memory_hit": series["memory_hit"]})

    st.markdown("#### Tier timeline (one-hot)")
    st.line_chart(
        {
            "ram": series["tier_ram"],
            "disk": series["tier_disk"],
            "compute": series["tier_compute"],
        }
    )


def render_latency_table(turns: List[Dict[str, Any]], *, last_n: int = 20) -> None:
    """
    Render a compact table for the last N turns with tier + timings.
    Great for quick debugging / live demo narration.
    """
    if not turns:
        st.info("No turns yet.")
        return

    slice_turns = turns[-last_n:]
    rows: List[Dict[str, Any]] = []
    for t in slice_turns:
        tm = t.get("timings_ms") or {}
        rows.append(
            {
                "turn": t.get("turn_id"),
                "tier": t.get("source_tier"),
                "hit": bool(t.get("used_memory")),
                "bypassed": bool(t.get("llm_bypassed")),
                "selection_ms": round(_safe_float(tm.get("selection_ms"), 0.0), 3),
                "lookup_ms": round(_safe_float(tm.get("memory_lookup_ms"), 0.0), 3),
                "gen_ms_est": round(_safe_float(tm.get("generation_ms_est"), 0.0), 3),
                "total_ms": round(_safe_float(tm.get("total_ms"), 0.0), 3),
            }
        )

    st.table(rows)


def get_summary_and_series(turns: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    If app.py prefers to compute once and render elsewhere.
    """
    return compute_summary(turns), build_series(turns)
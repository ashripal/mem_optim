# demo/app.py
"""
Streamlit demo: PDF Knowledge Agent with Multi-Tier Memory (RAM/Disk) + Latency Evidence

This app is an *initial* thesis demo that visually proves:
  1) PDF context is injected into the prompt path (context utilization)
  2) Multi-tier memory works (RAM hit / Disk hit / Compute miss)
  3) LLM bypass happens on cache hits (Phase 1: deterministic exact-match reuse)
  4) Disk persistence + promotion (Disk -> RAM) via "Simulate restart"
  5) Namespace isolation (session/user/cohort/global ordering)

Assumptions:
- This demo folder lives at thesis_code/demo/app.py
- memarch/ package is importable (repo root on PYTHONPATH or pytest.ini pythonpath=.)
- Phase 1: we use lexical chunk selection (NOT semantic embeddings/RAG)

Run:
  streamlit run demo/app.py
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

# --- memarch imports (your implemented backend) ---
from memarch.memory.disk_store import DiskStoreSQLite
from memarch.memory.manager import MemoryManager, MemoryManagerConfig
from memarch.memory.ram_store import RamStoreLRU
from memarch.memory.schema import MemoryHit, MemoryQuery, Provenance, QualitySignals


# ======================================================================================
# Small utilities (kept in this file for v1 demo simplicity)
# ======================================================================================

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(s: str) -> str:
    # Simple normalization: collapse whitespace (deterministic)
    return " ".join((s or "").split())


def tokenize(s: str) -> List[str]:
    # Deterministic lightweight tokenization (Phase 1)
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """
    Deterministic character-based chunking.
    Returns a list of dict chunks with chunk_id, text, start_char, end_char.
    """
    text = text or ""
    n = len(text)
    chunks: List[Dict[str, Any]] = []
    if n == 0:
        return chunks

    step = max(1, chunk_size - overlap)
    i = 0
    idx = 1
    while i < n:
        start = i
        end = min(n, i + chunk_size)
        chunk = text[start:end]
        chunks.append(
            {
                "chunk_id": f"c{idx:05d}",
                "text": chunk,
                "start_char": start,
                "end_char": end,
                "page_hint": None,  # placeholder for future extraction w/ page mapping
            }
        )
        idx += 1
        i += step
    return chunks


def select_topk_chunks_lexical(question: str, chunks: List[Dict[str, Any]], top_k: int) -> Tuple[List[Dict[str, Any]], float]:
    """
    Phase 1 chunk selection: lexical token overlap.
    - No embeddings
    - No vector DB
    - Deterministic and lightweight for edge devices

    Returns: (selected_chunks, selection_ms)
    """
    t0 = time.perf_counter()

    q_tokens = set(tokenize(question))
    if not q_tokens or not chunks:
        return [], (time.perf_counter() - t0) * 1000.0

    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for ch in chunks:
        c_tokens = set(tokenize(ch.get("text", "")))
        score = len(q_tokens.intersection(c_tokens))
        # tie-break deterministically by chunk_id
        scored.append((score, ch.get("chunk_id", ""), ch))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [x[2] for x in scored[: max(0, top_k)] if x[0] > 0]

    return selected, (time.perf_counter() - t0) * 1000.0


def extract_pdf_text(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    PDF -> text extraction.

    For demo stability, we try PyPDF2 first (common lightweight dependency).
    If PyPDF2 is not installed, we fall back to a friendly message.

    Returns:
      (full_text, stats_dict)
    """
    t0 = time.perf_counter()

    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return (
            "",
            {
                "ok": False,
                "error": "PyPDF2 not installed. Install with: pip install PyPDF2",
                "extract_ms": (time.perf_counter() - t0) * 1000.0,
            },
        )

    try:
        import io

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        text = "\n".join(pages)
        text = normalize_text(text)
        return (
            text,
            {
                "ok": True,
                "num_pages": len(reader.pages),
                "extract_ms": (time.perf_counter() - t0) * 1000.0,
            },
        )
    except Exception as e:
        return (
            "",
            {
                "ok": False,
                "error": f"PDF extraction failed: {e}",
                "extract_ms": (time.perf_counter() - t0) * 1000.0,
            },
        )


def ensure_artifacts_dir() -> str:
    """
    Keep demo artifacts in thesis_code/artifacts/demo (relative to CWD).
    """
    base = os.path.join(os.getcwd(), "artifacts", "demo")
    os.makedirs(base, exist_ok=True)
    return base


# ======================================================================================
# Generator: Fake prompt-building generator (Phase 1 demo)
# ======================================================================================

@dataclass
class PromptBuildingFakeGenerator:
    """
    Generator used for the demo so we can:
      - show prompt preview (context injection)
      - run without downloading a 7B model
      - measure bypass clearly

    If/when you add a real LLM backend, this class can remain as a "demo mode".
    """
    call_count: int = 0
    last_prompt: Optional[str] = None

    def generate(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> Tuple[str, Provenance, QualitySignals]:
        self.call_count += 1

        dataset_ctx = (mq.context or {}).get("dataset_context", "") or ""
        doc_sig = (mq.context or {}).get("doc_signature", "unknown")

        # Prompt preview = proof that PDF context is used in the LLM path.
        prompt = (
            f"DOC_SIGNATURE: {doc_sig}\n\n"
            f"DATASET_CONTEXT:\n{dataset_ctx}\n\n"
            f"QUESTION:\n{mq.raw_query}\n"
        )
        self.last_prompt = prompt

        # Deterministic "answer" that references context length as a sanity check.
        answer = f"[FAKE_ANSWER] ctx_len={len(dataset_ctx)} | Q: {mq.raw_query}"

        prov = Provenance(
            model_id=mq.model_id,
            prompt_version=mq.prompt_version,
            generated_at_utc=datetime.now(timezone.utc),
            generator_backend="fake_prompt_builder",
            quantization="Q4_K_M",
            context_window=4096,
        )
        qual = QualitySignals(score=1.0, success=True, metrics={"demo": 1.0})
        return answer, prov, qual


# ======================================================================================
# Session-state schema initialization (keep consistent with our earlier spec)
# ======================================================================================

def init_state() -> None:
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = {
            "ram_max_mb": 256,
            "promote_disk_hits_to_ram": True,
            "return_memory_directly": True,
            "enable_global_writes": False,
            "model_mode": "fake",  # "fake" | "llama_cpp" (future)
            "show_prompt_preview": True,
            "selector_top_k": 2,
        }

    if "identity" not in st.session_state:
        st.session_state["identity"] = {
            "user_id": "user_a",
            "session_id": "session_a",
            "cohort_id": "",
            "task": "pdf_qa",
            "model_id": "mistral-7b-instruct",
            "prompt_version": "v1",
        }

    if "doc" not in st.session_state:
        st.session_state["doc"] = None  # will be dict after PDF upload

    if "mem" not in st.session_state:
        # We keep manager + stores here (runtime-only objects)
        artifacts = ensure_artifacts_dir()
        disk_path = os.path.join(artifacts, "demo_memory.sqlite")
        st.session_state["mem"] = {
            "disk_path": disk_path,
            "ram": None,
            "disk": None,
            "manager": None,
            "run_id": sha256_text(utc_now_iso())[:10],
            "generator": PromptBuildingFakeGenerator(),
        }

    if "turns" not in st.session_state:
        st.session_state["turns"] = []  # list of per-turn evidence dicts


def build_or_rebuild_manager(clear_ram: bool = False) -> None:
    """
    Create (or recreate) memory objects based on cfg.
    If clear_ram=True, we "simulate restart" by only clearing RAM.
    """
    cfg = st.session_state["cfg"]
    mem = st.session_state["mem"]

    # Disk store is persistent across RAM clears.
    if mem["disk"] is None:
        mem["disk"] = DiskStoreSQLite(mem["disk_path"])

    if mem["ram"] is None:
        mem["ram"] = RamStoreLRU(max_mb=int(cfg["ram_max_mb"]))

    if clear_ram:
        mem["ram"].clear()

    # Rebuild manager every time config changes (cheap) to apply flags.
    mm_cfg = MemoryManagerConfig(
        promote_disk_hits_to_ram=bool(cfg["promote_disk_hits_to_ram"]),
        return_memory_directly=bool(cfg["return_memory_directly"]),
    )
    mem["manager"] = MemoryManager(ram=mem["ram"], disk=mem["disk"], cfg=mm_cfg)


# ======================================================================================
# UI rendering helpers
# ======================================================================================

def render_turn_card(turn: Dict[str, Any]) -> None:
    """
    Render one turn "evidence card" with a high-signal summary and expandable details.
    """
    tier = turn["source_tier"].upper()
    badge = "✅ LLM BYPASSED" if turn["llm_bypassed"] else "🔄 LLM CALLED"
    st.markdown(f"### Turn {turn['turn_id']}: **{tier}**  ·  {badge}")

    # Latency summary (always visible)
    t = turn["timings_ms"]
    st.write(
        {
            "memory_lookup_ms": round(t.get("memory_lookup_ms", 0.0), 3),
            "selection_ms": round(t.get("selection_ms", 0.0), 3),
            "generation_ms_est": round(t.get("generation_ms_est", 0.0), 3),
            "total_ms": round(t.get("total_ms", 0.0), 3),
        }
    )

    st.markdown("**Question**")
    st.write(turn["question"])

    st.markdown("**Answer**")
    st.write(turn["answer"])

    with st.expander("Debug: raw meta"):
        st.write(turn.get("meta_raw", {}))

    with st.expander("Context used (PDF chunks)"):
        st.write(
            {
                "doc_signature": turn.get("doc_signature"),
                "selected_chunk_ids": turn.get("selected_chunk_ids", []),
                "context_len": turn.get("context_len", 0),
            }
        )
        st.text(turn.get("context_preview", ""))

    with st.expander("Namespaces checked"):
        st.write(turn.get("namespaces_checked", []))

    if turn.get("prompt_preview"):
        with st.expander("Prompt preview (proof of context injection)"):
            st.text(turn["prompt_preview"])

    with st.expander("Storage decision"):
        st.write(
            {
                "stored": turn.get("stored"),
                "stored_scopes": turn.get("stored_scopes", []),
            }
        )


def render_performance_panel(turns: List[Dict[str, Any]]) -> None:
    """
    Show simple performance charts from turn history.
    We keep it lightweight (no seaborn).
    """
    if not turns:
        st.info("No turns yet. Ask a question to generate performance plots.")
        return

    total_ms = [float(t["timings_ms"].get("total_ms", 0.0)) for t in turns]
    hits = [1 if t.get("used_memory") else 0 for t in turns]
    idxs = list(range(1, len(turns) + 1))

    st.markdown("### Performance")
    st.line_chart({"total_ms": total_ms})
    st.line_chart({"memory_hit": hits})

    hit_rate = sum(hits) / max(1, len(hits))
    st.write(
        {
            "turns": len(turns),
            "hit_rate": round(hit_rate, 3),
            "avg_total_ms": round(sum(total_ms) / max(1, len(total_ms)), 3),
        }
    )


# ======================================================================================
# Main interaction: ask a question
# ======================================================================================

def handle_ask(question: str) -> None:
    question = (question or "").strip()
    if not question:
        st.warning("Enter a question.")
        return

    doc = st.session_state["doc"]
    if not doc:
        st.warning("Upload and ingest a PDF first.")
        return

    cfg = st.session_state["cfg"]
    ident = st.session_state["identity"]
    mem = st.session_state["mem"]

    manager: MemoryManager = mem["manager"]
    generator: PromptBuildingFakeGenerator = mem["generator"]

    # 1) Select chunks (Phase 1 lexical selection)
    selected_chunks, selection_ms = select_topk_chunks_lexical(
        question=question,
        chunks=doc["chunks"],
        top_k=int(cfg["selector_top_k"]),
    )
    selected_chunk_ids = [c["chunk_id"] for c in selected_chunks]
    selected_context_text = "\n\n".join([c["text"] for c in selected_chunks])

    # 2) Build MemoryQuery (this is where we inject PDF context)
    mq = MemoryQuery(
        raw_query=question,
        user_id=(ident["user_id"] or None),
        session_id=(ident["session_id"] or None),
        cohort_id=(ident["cohort_id"] or None) if ident.get("cohort_id") else None,
        # task=ident["task"],
        task=f"{ident['task']}|doc={doc['doc_signature'][:12]}",
        model_id=ident["model_id"],
        prompt_version=ident["prompt_version"],
        context={
            "dataset_context": selected_context_text,
            "doc_signature": doc["doc_signature"],
            "selected_chunk_ids": selected_chunk_ids,
        },
    )

    # 3) Call manager.answer (records hit/miss, may bypass generator)
    t0 = time.perf_counter()
    answer, meta = manager.answer(mq, generator)
    total_ms = (time.perf_counter() - t0) * 1000.0

    # Some timing fields might already be included by your backend meta; we merge safely.
    timings = dict(meta.get("timings_ms") or {})
    timings.setdefault("selection_ms", selection_ms)
    timings.setdefault("total_ms", total_ms)
    timings.setdefault("memory_lookup_ms", float(meta.get("memory_lookup_ms", timings.get("memory_lookup_ms", 0.0))))
    timings.setdefault("generation_ms_est", float(meta.get("generation_ms_est", timings.get("generation_ms_est", 0.0))))

    # 4) Build turn record (fully renderable without recomputation)
    turns: List[Dict[str, Any]] = st.session_state["turns"]
    turn_id = len(turns) + 1

    # Namespaces checked: your manager may already provide this; otherwise keep minimal.
    # namespaces_checked = meta.get("namespaces_checked") or []
    expected_scopes = []
    if ident.get("user_id"):
        expected_scopes.append(f"user:{ident['user_id']}")
    if ident.get("session_id"):
        expected_scopes.append(f"session:{ident['session_id']}")
    if ident.get("cohort_id"):
        expected_scopes.append(f"cohort:{ident['cohort_id']}")
    expected_scopes.append("global")

    namespaces_checked = meta.get("namespaces_checked") or expected_scopes

    # Prompt preview: only available on compute path (miss). On hit, leave None.
    prompt_preview = generator.last_prompt if (not meta.get("used_memory") and cfg["show_prompt_preview"]) else None

    turn = {
        "turn_id": turn_id,
        "timestamp_utc": utc_now_iso(),
        "question": question,
        "answer": answer,
        "used_memory": bool(meta.get("used_memory")),
        "source_tier": str(meta.get("source_tier", "compute" if meta.get("generated") else "unknown")),
        "llm_bypassed": bool(meta.get("used_memory")) and bool(cfg["return_memory_directly"]),
        "doc_signature": doc["doc_signature"],
        "selected_chunk_ids": selected_chunk_ids,
        "context_preview": (selected_context_text[:1200] + "…") if len(selected_context_text) > 1200 else selected_context_text,
        "context_len": len(selected_context_text),
        "prompt_preview": prompt_preview,
        "namespaces_checked": namespaces_checked,
        "timings_ms": timings,
        "stored": meta.get("stored", None),
        "stored_scopes": meta.get("stored_scopes", []),
        "meta_raw": meta
    }
    turns.append(turn)

def hard_reset_memory() -> None:
    mem = st.session_state["mem"]

    # 1) Clear RAM if present
    if mem.get("ram") is not None:
        mem["ram"].clear()

    # 2) Close disk handle if your DiskStoreSQLite has it (safe best-effort)
    if mem.get("disk") is not None:
        try:
            mem["disk"].close()  # implement in DiskStoreSQLite if missing
        except Exception:
            pass

    # 3) Drop references so Streamlit will recreate objects
    mem["disk"] = None
    mem["manager"] = None

    # 4) Delete the sqlite file
    try:
        if os.path.exists(mem["disk_path"]):
            os.remove(mem["disk_path"])
    except Exception as e:
        st.warning(f"Could not delete disk db: {e}")

    # 5) Rebuild fresh
    build_or_rebuild_manager(clear_ram=False)

# In sidebar:
if st.button("Clear RAM"):
    st.session_state["mem"]["ram"].clear()
    st.success("RAM cleared.")

if st.button("Hard reset (clear RAM + delete Disk DB)"):
    hard_reset_memory()
    st.success("RAM cleared and disk DB deleted; fresh manager created.")

# ======================================================================================
# Streamlit app
# ======================================================================================

def main() -> None:
    st.set_page_config(page_title="MemArch PDF Demo", layout="wide")

    init_state()
    build_or_rebuild_manager(clear_ram=False)

    st.title("MemArch Demo: PDF Knowledge Agent with Multi-Tier Memory (RAM/Disk)")

    # ----------------------------------------------------------------------------------
    # Sidebar: configuration
    # ----------------------------------------------------------------------------------
    with st.sidebar:
        st.header("Demo Controls")

        cfg = st.session_state["cfg"]
        ident = st.session_state["identity"]
        mem = st.session_state["mem"]

        st.subheader("Identity (Namespaces)")
        ident["user_id"] = st.text_input("user_id", value=ident.get("user_id", "user_a"))
        ident["session_id"] = st.text_input("session_id", value=ident.get("session_id", "session_a"))
        ident["cohort_id"] = st.text_input("cohort_id (optional)", value=ident.get("cohort_id", ""))

        st.subheader("Memory")
        cfg["ram_max_mb"] = st.slider("RAM budget (MB)", min_value=64, max_value=1024, value=int(cfg["ram_max_mb"]), step=64)
        cfg["promote_disk_hits_to_ram"] = st.checkbox("Promote disk hits → RAM", value=bool(cfg["promote_disk_hits_to_ram"]))
        cfg["return_memory_directly"] = st.checkbox("Return memory directly (LLM bypass on hit)", value=bool(cfg["return_memory_directly"]))

        st.subheader("Context selection (Phase 1 lexical)")
        cfg["selector_top_k"] = st.slider("Top-k chunks per question", min_value=1, max_value=5, value=int(cfg["selector_top_k"]), step=1)

        st.subheader("Prompt evidence")
        cfg["show_prompt_preview"] = st.checkbox("Show prompt preview (compute path)", value=bool(cfg["show_prompt_preview"]))

        st.subheader("Actions")
        if st.button("Simulate restart (clear RAM only)"):
            build_or_rebuild_manager(clear_ram=True)
            st.success("RAM cleared. Disk preserved.")

        if st.button("Reset turns (clear history)"):
            st.session_state["turns"] = []
            st.success("Turn history cleared.")

        # Apply config changes immediately
        build_or_rebuild_manager(clear_ram=False)

        st.divider()
        st.caption("Disk store")
        st.code(mem["disk_path"])

    # ----------------------------------------------------------------------------------
    # Main layout: PDF ingest + tabs
    # ----------------------------------------------------------------------------------
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.header("1) Upload PDF & Ingest")

        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        chunk_mode = st.radio("Chunking mode", options=["chunked", "full"], index=0, horizontal=True)
        chunk_size = st.slider("Chunk size (chars)", 400, 3000, 1200, 100)
        overlap = st.slider("Overlap (chars)", 0, 500, 120, 10)

        if st.button("Ingest PDF") and uploaded is not None:
            pdf_bytes = uploaded.getvalue()
            pdf_sha = sha256_bytes(pdf_bytes)

            # Extract
            text, stats = extract_pdf_text(pdf_bytes)
            if not stats.get("ok"):
                st.error(stats.get("error", "Unknown PDF extraction error"))
            else:
                # Chunk
                t0 = time.perf_counter()
                if chunk_mode == "full":
                    # single chunk (useful only for small PDFs)
                    chunks = [{"chunk_id": "c00001", "text": text, "start_char": 0, "end_char": len(text), "page_hint": None}]
                else:
                    chunks = chunk_text(text, int(chunk_size), int(overlap))
                chunk_ms = (time.perf_counter() - t0) * 1000.0

                # Signature: based on normalized chunk texts (stable + invalidates on doc change)
                joined = "\n".join([normalize_text(c["text"]) for c in chunks])
                doc_sig = sha256_text(joined)

                st.session_state["doc"] = {
                    "pdf_name": uploaded.name,
                    "pdf_bytes_sha256": pdf_sha,
                    "doc_signature": doc_sig,
                    "text_preview": (text[:1200] + "…") if len(text) > 1200 else text,
                    "chunks": chunks,
                    "chunking": {"mode": chunk_mode, "chunk_size_chars": int(chunk_size), "overlap_chars": int(overlap)},
                    "ingest_stats": {
                        "num_chars": len(text),
                        "num_chunks": len(chunks),
                        "extract_ms": stats.get("extract_ms", 0.0),
                        "chunk_ms": chunk_ms,
                        "num_pages": stats.get("num_pages", None),
                    },
                }

                # For a clean demo, clearing turn history when doc changes is usually best.
                st.session_state["turns"] = []

                st.success("PDF ingested successfully.")

        doc = st.session_state["doc"]
        if doc:
            st.subheader("Document summary")
            st.write(
                {
                    "pdf_name": doc["pdf_name"],
                    "doc_signature": doc["doc_signature"],
                    "num_chunks": doc["ingest_stats"]["num_chunks"],
                    "num_chars": doc["ingest_stats"]["num_chars"],
                }
            )
            with st.expander("Text preview"):
                st.text(doc["text_preview"])
            with st.expander("Chunk preview (first 3)"):
                for ch in doc["chunks"][:3]:
                    st.markdown(f"**{ch['chunk_id']}**  ({ch['start_char']}–{ch['end_char']})")
                    st.text((ch["text"][:600] + "…") if len(ch["text"]) > 600 else ch["text"])

    with col_right:
        st.header("2) Ask Questions")

        question = st.text_input("Enter a question about the PDF", value="", placeholder="e.g., What does the document say about … ?")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Ask"):
                handle_ask(question)
        with c2:
            if st.button("Ask again (repeat)"):
                handle_ask(question)

        st.divider()

        turns: List[Dict[str, Any]] = st.session_state["turns"]
        if turns:
            # Show newest first for a "chat" feel
            for turn in reversed(turns):
                st.divider()
                render_turn_card(turn)
        else:
            st.info("No turns yet. Ingest a PDF, then ask a question.")

    st.divider()

    # ----------------------------------------------------------------------------------
    # Bottom section: performance plots + global stats
    # ----------------------------------------------------------------------------------
    turns = st.session_state["turns"]
    mem = st.session_state["mem"]
    gen: PromptBuildingFakeGenerator = mem["generator"]

    st.header("3) Performance & Evidence Summary")

    # High-signal counters (committee-friendly)
    hits = sum(1 for t in turns if t.get("used_memory"))
    misses = len(turns) - hits
    st.write(
        {
            "turns": len(turns),
            "memory_hits": hits,
            "misses_compute": misses,
            "llm_calls (demo generator)": gen.call_count,
        }
    )

    # Memory store stats
    if mem.get("ram") is not None:
        ram_stats = mem["ram"].stats()
        st.write({"ram_bytes_current": ram_stats.bytes_current, "ram_capacity_bytes": ram_stats.bytes_capacity, "ram_evictions": ram_stats.evictions})

    if mem.get("disk") is not None:
        st.write({"disk_stats": mem["disk"].stats()})

    render_performance_panel(turns)


if __name__ == "__main__":
    main()
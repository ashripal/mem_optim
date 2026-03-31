from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memarch.memory.schema import MatchType, MemoryHit, MemoryQuery, Provenance, QualitySignals


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _select_device(device: str = "auto") -> str:
    device = (device or "auto").lower().strip()

    if device == "auto":
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if device == "cuda":
        if not _cuda_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available.")
        return "cuda"

    if device == "mps":
        if not _mps_available():
            raise RuntimeError("Requested device='mps' but MPS is not available.")
        return "mps"

    if device == "cpu":
        return "cpu"

    raise ValueError(f"Unsupported device: {device}")


@dataclass(frozen=True)
class GeneratorConfig:
    model_id: str = "distilgpt2"
    device: str = "auto"

    max_input_length: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = False

    decoding_mode: str = "greedy"
    num_beams: int = 1

    local_files_only: bool = False
    torch_dtype: str = "auto"
    use_fast_tokenizer: bool = False

    cpu_fallback_on_failure: bool = True
    generation_backend: str = "auto"

    include_retrieved_memory_context: bool = True
    include_dataset_context: bool = True
    include_doc_signature: bool = False

    prefer_retrieved_evidence_context: bool = True
    reduce_context_on_semantic_hit: bool = True
    max_evidence_chars: int = 400
    max_local_context_chars: int = 260
    max_full_context_chars: int = 1200

    prefer_local_context_for_qa: bool = True
    qa_max_output_words: int = 6

    trec_use_few_shot: bool = False
    skip_special_tokens: bool = True


class HFGenerator:
    _TREC_LABELS = {"ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"}

    _TREC_ALIAS_MAP = {
        "ABBREVIATION": "ABBR",
        "ABBR": "ABBR",
        "DESCRIPTION": "DESC",
        "DEFINITION": "DESC",
        "DESC": "DESC",
        "ENTITY": "ENTY",
        "ENTY": "ENTY",
        "HUMAN": "HUM",
        "PERSON": "HUM",
        "INDIVIDUAL": "HUM",
        "HUM": "HUM",
        "LOCATION": "LOC",
        "PLACE": "LOC",
        "LOC": "LOC",
        "NUMBER": "NUM",
        "NUMERIC": "NUM",
        "NUM": "NUM",
    }

    _ANSWER_PREFIX_RE = re.compile(
        r"^\s*(?:OUTPUT|FINAL ANSWER|ANSWER|RESPONSE|LABEL)\s*:\s*",
        flags=re.IGNORECASE,
    )

    _LEADING_ANSWER_PHRASE_RE = re.compile(
        r"^\s*(?:"
        r"the answer is|answer is|it is|it's|the answer|"
        r"from|the threat is|the film was|film was|"
        r"he was|she was|they were|there were|there was|"
        r"located in|based in|set in|called"
        r")\s+",
        flags=re.IGNORECASE,
    )

    _CHAT_TURN_RE = re.compile(
        r"\b(?:Human|Assistant|User|System)\s*:",
        flags=re.IGNORECASE,
    )

    _BOILERPLATE_PATTERNS = [
        re.compile(r"\bto answer this question\b.*", flags=re.IGNORECASE | re.DOTALL),
        re.compile(r"\byou need to\b.*", flags=re.IGNORECASE | re.DOTALL),
        re.compile(r"\bbased on the (?:context|passage|document)\b.*", flags=re.IGNORECASE | re.DOTALL),
        re.compile(r"\byou are an ai assistant\b.*", flags=re.IGNORECASE | re.DOTALL),
    ]

    _WS_RE = re.compile(r"\s+")
    _PUNCT_EDGE_RE = re.compile(r"^[\-\–\—\,\.\:\;\"\']+\s*|\s*[\-\–\—\,\.\:\;\"\']+$")
    _WORDISH_RE = re.compile(r"\S+")

    def __init__(self, cfg: Optional[GeneratorConfig] = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self._validate_config()

        self.device = _select_device(self.cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            use_fast=self.cfg.use_fast_tokenizer,
            local_files_only=self.cfg.local_files_only,
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_dtype = self._resolve_torch_dtype(self.cfg.torch_dtype, self.device)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            local_files_only=self.cfg.local_files_only,
            torch_dtype=self.model_dtype,
        )
        self.model.eval()
        self.model.to(device=self.device, dtype=self.model_dtype)

        self.last_prompt: Optional[str] = None
        self.last_generation_meta: Optional[Dict[str, object]] = None

    def _validate_config(self) -> None:
        mode = (self.cfg.decoding_mode or "greedy").lower().strip()

        if mode not in {"greedy", "beam", "sample"}:
            raise ValueError("decoding_mode must be one of: greedy, beam, sample")

        if mode == "beam" and int(self.cfg.num_beams) <= 1:
            raise ValueError("Beam mode requires num_beams > 1")

        if mode != "sample" and bool(self.cfg.do_sample):
            raise ValueError("do_sample=True is only valid when decoding_mode='sample'")

        if mode == "sample" and int(self.cfg.num_beams) != 1:
            raise ValueError("Sample mode requires num_beams == 1")

        if int(self.cfg.max_input_length) <= 0:
            raise ValueError("max_input_length must be > 0")

        if int(self.cfg.max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")

        if int(self.cfg.qa_max_output_words) <= 0:
            raise ValueError("qa_max_output_words must be > 0")

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str, device: str):
        dtype_name = (dtype_name or "auto").lower().strip()

        if dtype_name == "auto":
            if device == "cuda":
                return torch.float16
            if device == "mps":
                return torch.float16
            return torch.float32

        if dtype_name == "float16":
            return torch.float32 if device == "cpu" else torch.float16
        if dtype_name == "bfloat16":
            return torch.float32 if device == "cpu" else torch.bfloat16
        if dtype_name == "float32":
            return torch.float32

        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")

    @staticmethod
    def _safe_text(value: Optional[str]) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _collapse_ws(cls, text: str) -> str:
        return cls._WS_RE.sub(" ", cls._safe_text(text)).strip()

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        text = str(text or "").lower()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_trec_output(self, text: str) -> str:
        raw = self._safe_text(text)
        if not raw:
            return raw

        upper = raw.upper()

        for pat in [
            r"\b(ABBR|DESC|ENTY|HUM|LOC|NUM)\b",
            r"^(ABBR|DESC|ENTY|HUM|LOC|NUM)[\.\:\-\s_]",
        ]:
            m = re.search(pat, upper)
            if m:
                return m.group(1)

        for pat in [
            r"\bTYPE\s*:\s*([A-Z]+)\b",
            r"\bOUTPUT\s*:\s*([A-Z]+)\b",
            r"^([A-Z]+)\b",
        ]:
            m = re.search(pat, upper)
            if m:
                token = m.group(1)
                mapped = self._TREC_ALIAS_MAP.get(token)
                if mapped in self._TREC_LABELS:
                    return mapped

        for token in re.findall(r"[A-Z]+", upper):
            mapped = self._TREC_ALIAS_MAP.get(token)
            if mapped in self._TREC_LABELS:
                return mapped

        return raw

    def _strip_chat_turns(self, text: str) -> str:
        raw = self._safe_text(text)
        if not raw:
            return raw
        m = self._CHAT_TURN_RE.search(raw)
        if m:
            raw = raw[:m.start()].rstrip()
        return raw

    def _limit_words(self, text: str, max_words: int) -> str:
        words = self._collapse_ws(text).split()
        if len(words) <= max_words:
            return self._collapse_ws(text)
        return " ".join(words[:max_words]).strip()

    def _trim_edge_punct(self, text: str) -> str:
        return self._PUNCT_EDGE_RE.sub("", self._safe_text(text)).strip()

    def _candidate_spans_from_context(self, context: str, max_words: int) -> List[str]:
        ctx = self._safe_text(context)
        if not ctx:
            return []

        matches = list(self._WORDISH_RE.finditer(ctx))
        spans: List[str] = []

        for i in range(len(matches)):
            for j in range(i, min(len(matches), i + max_words)):
                start = matches[i].start()
                end = matches[j].end()
                span = ctx[start:end].strip()
                span = self._trim_edge_punct(span)
                if span:
                    spans.append(span)

        # preserve order but dedupe
        seen = set()
        out: List[str] = []
        for s in spans:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _snap_answer_to_context(self, answer: str, context: str) -> str:
        """
        Try to convert a verbose answer into the shortest answer-like span
        present in the local context. This is especially useful for outputs like:
        - "The threat is climate change" -> "climate change"
        - "From Germany" -> "Germany"
        - "... Marc Nelson ..." -> "Marc Nelson"
        """
        raw = self._collapse_ws(answer)
        ctx = self._safe_text(context)
        if not raw or not ctx:
            return raw

        raw_norm = self._normalize_for_match(raw)
        if not raw_norm:
            return raw

        candidates = self._candidate_spans_from_context(ctx, int(self.cfg.qa_max_output_words))
        best: Optional[Tuple[int, int, str]] = None

        for cand in candidates:
            cand_norm = self._normalize_for_match(cand)
            if not cand_norm:
                continue

            if cand_norm == raw_norm:
                score = (10_000 + len(cand_norm), -len(cand.split()), cand)
            elif cand_norm in raw_norm:
                score = (1_000 + len(cand_norm), -len(cand.split()), cand)
            elif raw_norm in cand_norm and len(cand.split()) <= int(self.cfg.qa_max_output_words):
                score = (100 + len(raw_norm), -len(cand.split()), cand)
            else:
                continue

            if best is None or score > best:
                best = score

        if best is not None:
            return best[2].strip()

        return raw

    def _normalize_short_qa_output(self, text: str, context_text: str = "") -> str:
        raw = self._safe_text(text)
        if not raw:
            return raw

        raw = self._ANSWER_PREFIX_RE.sub("", raw).strip()
        raw = self._strip_chat_turns(raw)

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if lines:
            raw = lines[0]

        for pat in self._BOILERPLATE_PATTERNS:
            raw = pat.sub("", raw).strip()

        raw = self._strip_chat_turns(raw)
        raw = re.sub(r"\s{2,}", " ", raw).strip()
        raw = raw.strip(" \t\r\n\"'`")

        # Remove leading answer phrases like "The answer is ..." / "From ..."
        raw = self._LEADING_ANSWER_PHRASE_RE.sub("", raw).strip()

        # If the model gives "Name. More text", keep the first answer-like chunk.
        parts = re.split(r"(?<=[\.\!\?])\s+|;\s+|,\s+(?=[A-Z])", raw, maxsplit=1)
        if parts:
            raw = parts[0].strip()

        raw = self._strip_chat_turns(raw)

        # Remove leading punctuation fragments often produced by continuation-style models
        raw = re.sub(r"^[\-\–\—\,\.\:\;\"\']+\s*", "", raw).strip()

        # If a short answer is followed by glued chat text like "Arne Frager.Human"
        raw = re.sub(r"([A-Za-z0-9\)])\.(?=(?:Human|Assistant|User|System)\s*:)", r"\1", raw)

        raw = self._trim_edge_punct(raw)
        raw = self._collapse_ws(raw)

        # Snap verbose answer back to a short span from context when possible.
        raw = self._snap_answer_to_context(raw, context_text)

        # Keep answers short for extractive QA.
        raw = self._limit_words(raw, int(self.cfg.qa_max_output_words))
        raw = self._trim_edge_punct(raw)

        if len(raw.split()) <= 12:
            raw = raw.rstrip(" .,:;")

        return raw or self._safe_text(text)

    @staticmethod
    def _truncate_chars(text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

    def _model_max_positions(self) -> int:
        model_config = getattr(self.model, "config", None)
        if model_config is None:
            return int(self.cfg.max_input_length)

        for value in [
            getattr(model_config, "max_position_embeddings", None),
            getattr(model_config, "n_positions", None),
            getattr(model_config, "max_sequence_length", None),
            getattr(model_config, "seq_length", None),
        ]:
            try:
                if value is not None and int(value) > 0:
                    return int(value)
            except Exception:
                pass
        return int(self.cfg.max_input_length)

    def _effective_prompt_token_budget(self) -> int:
        model_limit = self._model_max_positions()
        requested_input = int(self.cfg.max_input_length)
        requested_new = max(1, int(self.cfg.max_new_tokens))
        safe_budget = min(requested_input, max(1, model_limit - requested_new))
        return max(1, safe_budget)

    def _allowed_new_tokens(self, input_tokens: int) -> int:
        model_limit = self._model_max_positions()
        remaining = max(0, int(model_limit) - int(input_tokens))
        return max(0, min(int(self.cfg.max_new_tokens), remaining))

    def _select_generation_backend(self) -> str:
        if self.cfg.generation_backend != "auto":
            return self.cfg.generation_backend

        mode = (self.cfg.decoding_mode or "greedy").lower().strip()
        if mode in {"beam", "sample"}:
            return "hf_generate"
        if self.device in {"cpu", "mps"}:
            return "manual_greedy"
        return "hf_generate"

    def _query_dataset_context(self, mq: MemoryQuery) -> str:
        return self._safe_text((mq.context or {}).get("dataset_context", ""))

    def _query_doc_signature(self, mq: MemoryQuery) -> str:
        if getattr(mq, "doc_signature", None):
            return self._safe_text(mq.doc_signature)
        return self._safe_text((mq.context or {}).get("doc_signature", ""))

    def _query_question_type(self, mq: MemoryQuery) -> str:
        if getattr(mq, "question_type", None):
            return self._safe_text(mq.question_type)
        return self._safe_text((mq.context or {}).get("question_type", ""))

    def _query_evidence_text(self, mq: MemoryQuery) -> str:
        if getattr(mq, "evidence_text", None):
            return self._safe_text(mq.evidence_text)
        return self._safe_text((mq.context or {}).get("evidence_text", ""))

    def _retrieved_evidence_text(self, retrieved: MemoryHit) -> str:
        item = retrieved.item
        direct = getattr(item, "evidence_text", None)
        if direct:
            return self._safe_text(direct)
        return self._safe_text(item.meta.get("evidence_text", ""))

    def _retrieved_doc_signature(self, retrieved: MemoryHit) -> str:
        item = retrieved.item
        direct = getattr(item, "doc_signature", None)
        if direct:
            return self._safe_text(direct)
        return self._safe_text(item.meta.get("doc_signature", ""))

    def _retrieved_same_document(self, mq: MemoryQuery, retrieved: Optional[MemoryHit]) -> bool:
        if retrieved is None:
            return False

        dbg = getattr(retrieved, "debug", {}) or {}
        if "same_document" in dbg:
            try:
                return bool(dbg.get("same_document"))
            except Exception:
                pass

        query_doc = self._query_doc_signature(mq)
        item_doc = self._retrieved_doc_signature(retrieved)
        return bool(query_doc and item_doc and query_doc == item_doc)

    def _retrieved_section(self, mq: MemoryQuery, retrieved: MemoryHit) -> str:
        evidence_text = self._truncate_chars(
            self._retrieved_evidence_text(retrieved),
            int(self.cfg.max_evidence_chars),
        )
        answer_text = self._safe_text(retrieved.item.answer_text)
        answer_text = self._truncate_chars(answer_text, 180)

        same_doc = self._retrieved_same_document(mq, retrieved)

        meta_parts = [
            f"match_type={retrieved.match_type.value}",
            f"source_tier={retrieved.source_tier.value}",
            f"score={retrieved.score:.4f}",
            f"same_document={str(bool(same_doc)).lower()}",
        ]
        if retrieved.semantic_rank is not None:
            meta_parts.append(f"semantic_rank={retrieved.semantic_rank}")

        lines = [
            "RETRIEVED MEMORY SUPPORT:",
            ", ".join(meta_parts),
        ]

        if evidence_text:
            lines.append("")
            lines.append("RETRIEVED EVIDENCE:")
            lines.append(evidence_text)

        if answer_text:
            lines.append("")
            lines.append("PRIOR ANSWER:")
            lines.append(answer_text)

        return "\n".join(lines)

    def _task_instruction(self, mq: MemoryQuery) -> str:
        task = self._safe_text(getattr(mq, "task", "")).lower()
        question_type = self._query_question_type(mq).lower()

        if task == "trec" or question_type == "classification":
            return (
                "Return exactly one label from: ABBR, DESC, ENTY, HUM, LOC, NUM.\n"
                "Do not output anything else."
            )

        if question_type in {"qa", "boolean_qa", "unknown"}:
            return (
                "You must answer the question using ONLY a short span from the context.\n"
                "STRICT RULES:\n"
                "- Output ONLY the exact answer span.\n"
                "- Do NOT include explanations.\n"
                "- Do NOT include prefixes like 'Answer:'.\n"
                "- Do NOT include assistant text or chat markers.\n"
                f"- Output at most {int(self.cfg.qa_max_output_words)} words.\n"
                "- If unsure, output the shortest possible answer."
            )

        return "Answer briefly and directly."

    def _build_trec_prompt(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> str:
        question = self._safe_text(mq.raw_query)

        parts = [
            "You are a classifier for TREC coarse question types.",
            "Valid labels: ABBR, DESC, ENTY, HUM, LOC, NUM.",
            "Return exactly one label and nothing else.",
        ]

        retrieved_block = ""
        retrieved_evidence = None
        same_doc = None

        if retrieved is not None and self.cfg.include_retrieved_memory_context:
            retrieved_block = self._retrieved_section(mq, retrieved)
            retrieved_evidence = self._truncate_chars(
                self._retrieved_evidence_text(retrieved),
                int(self.cfg.max_evidence_chars),
            )
            same_doc = self._retrieved_same_document(mq, retrieved)

            retrieved_answer = self._safe_text(retrieved.item.answer_text)
            if retrieved_answer:
                parts.append(f"Related prior label or answer: {self._truncate_chars(retrieved_answer, 80)}")
            if retrieved_block:
                parts.append(retrieved_block)

        if self.cfg.trec_use_few_shot:
            parts.extend(
                [
                    "Examples:",
                    "Question: What does CIA stand for?",
                    "Label: ABBR",
                    "Question: Why do airplanes leave contrails?",
                    "Label: DESC",
                    "Question: What city is the Eiffel Tower in?",
                    "Label: LOC",
                ]
            )

        parts.append(f"Question: {question}")
        parts.append("Label:")

        prompt = "\n".join(parts)
        self.last_prompt = prompt

        self.last_generation_meta = dict(self.last_generation_meta or {})
        self.last_generation_meta["reduced_context_used"] = bool(retrieved is not None and retrieved_block)
        self.last_generation_meta["full_context_chars"] = len(self._query_dataset_context(mq))
        self.last_generation_meta["final_context_chars"] = len(retrieved_block) if retrieved_block else 0
        self.last_generation_meta["retrieved_evidence_chars"] = len(retrieved_evidence) if retrieved_evidence else None
        self.last_generation_meta["retrieved_doc_signature_match"] = same_doc

        return prompt

    def _effective_char_budget(self, requested_chars: int) -> int:
        requested_chars = max(0, int(requested_chars))
        token_budget = max(1, int(self.cfg.max_input_length))
        context_token_budget = max(16, int(token_budget * 0.5))
        approx_char_budget = context_token_budget * 4
        return min(requested_chars, approx_char_budget)

    def _select_context_block(
        self,
        mq: MemoryQuery,
        retrieved: Optional[MemoryHit],
    ) -> Tuple[str, Dict[str, object]]:
        dataset_ctx = self._query_dataset_context(mq)
        query_evidence = self._truncate_chars(
            self._query_evidence_text(mq),
            self._effective_char_budget(int(self.cfg.max_local_context_chars)),
        )
        full_context_chars = len(dataset_ctx) if dataset_ctx else 0

        question_type = self._query_question_type(mq).lower()

        if self.cfg.prefer_local_context_for_qa and question_type in {"qa", "boolean_qa", "unknown"}:
            if query_evidence:
                return query_evidence, {
                    "reduced_context_used": True,
                    "full_context_chars": full_context_chars,
                    "final_context_chars": len(query_evidence),
                    "retrieved_evidence_chars": None,
                    "retrieved_doc_signature_match": None,
                }

        if (
            retrieved is not None
            and retrieved.match_type == MatchType.SEMANTIC
            and self.cfg.include_retrieved_memory_context
            and self.cfg.prefer_retrieved_evidence_context
            and self.cfg.reduce_context_on_semantic_hit
        ):
            retrieved_evidence = self._truncate_chars(
                self._retrieved_evidence_text(retrieved),
                self._effective_char_budget(int(self.cfg.max_evidence_chars)),
            )
            same_document = self._retrieved_same_document(mq, retrieved)

            parts = []
            if retrieved_evidence:
                parts.append("RETRIEVED EVIDENCE:\n" + retrieved_evidence)
            if query_evidence:
                parts.append("LOCAL CONTEXT:\n" + query_evidence)

            context_block = "\n\n".join(parts)

            return context_block, {
                "reduced_context_used": bool(context_block),
                "full_context_chars": full_context_chars,
                "final_context_chars": len(context_block),
                "retrieved_evidence_chars": len(retrieved_evidence) if retrieved_evidence else None,
                "retrieved_doc_signature_match": same_document,
            }

        trimmed_full_ctx = self._truncate_chars(dataset_ctx, int(self.cfg.max_full_context_chars))
        return trimmed_full_ctx, {
            "reduced_context_used": False,
            "full_context_chars": full_context_chars,
            "final_context_chars": len(trimmed_full_ctx),
            "retrieved_evidence_chars": None,
            "retrieved_doc_signature_match": None,
        }

    def build_prompt(self, mq: MemoryQuery, retrieved: Optional[MemoryHit] = None) -> str:
        task = self._safe_text(getattr(mq, "task", "")).lower()
        question_type = self._query_question_type(mq).lower()

        if task == "trec" or question_type == "classification":
            return self._build_trec_prompt(mq, retrieved=retrieved)

        task_instruction = self._task_instruction(mq)
        context_block, prompt_stats = self._select_context_block(mq, retrieved=retrieved)

        parts = [
            "Answer the question using the context.",
            task_instruction,
        ]

        if context_block:
            parts.append(f"Context:\n{context_block}")

        if self.cfg.include_retrieved_memory_context and retrieved is not None:
            retrieved_block = self._retrieved_section(mq, retrieved)
            if retrieved_block:
                parts.append(retrieved_block)

        if self.cfg.include_doc_signature:
            doc_sig = self._query_doc_signature(mq)
            if doc_sig:
                parts.append(f"Document signature: {doc_sig}")

        parts.append(f"Question: {self._safe_text(mq.raw_query)}")
        parts.append("Answer:")

        prompt = "\n\n".join(parts)
        self.last_prompt = prompt

        self.last_generation_meta = dict(self.last_generation_meta or {})
        self.last_generation_meta["reduced_context_used"] = prompt_stats["reduced_context_used"]
        self.last_generation_meta["full_context_chars"] = prompt_stats["full_context_chars"]
        self.last_generation_meta["final_context_chars"] = prompt_stats["final_context_chars"]
        self.last_generation_meta["retrieved_evidence_chars"] = prompt_stats["retrieved_evidence_chars"]
        self.last_generation_meta["retrieved_doc_signature_match"] = prompt_stats["retrieved_doc_signature_match"]

        return prompt

    def _manual_greedy_decode(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        allowed_new_tokens: int,
    ) -> torch.Tensor:
        generated_ids = input_ids
        generated_mask = attention_mask
        eos_token_id = self.tokenizer.eos_token_id

        if allowed_new_tokens <= 0:
            return generated_ids

        with torch.inference_mode():
            for _ in range(int(allowed_new_tokens)):
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=generated_mask,
                    use_cache=False,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                if generated_mask is not None:
                    next_mask = torch.ones(
                        (generated_mask.shape[0], 1),
                        dtype=generated_mask.dtype,
                        device=generated_mask.device,
                    )
                    generated_mask = torch.cat([generated_mask, next_mask], dim=-1)

                if eos_token_id is not None and bool((next_token_id == eos_token_id).all()):
                    break

        return generated_ids

    def _move_model(self, device: str, dtype=None) -> None:
        if dtype is None:
            dtype = self.model_dtype
        self.model.to(device=device, dtype=dtype)
        self.device = device

    def _record_meta(
        self,
        *,
        prompt: str,
        input_tokens: int,
        output_tokens: int,
        truncated: bool,
        tokenize_time_s: float,
        gen_time_s: float,
        decode_time_s: float,
        backend_used: str,
        retrieved: Optional[MemoryHit],
    ) -> None:
        prior_meta = dict(self.last_generation_meta or {})

        meta: Dict[str, object] = {
            "device": self.device,
            "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
            "generation_backend": backend_used,
            "decoding_mode": self.cfg.decoding_mode,
            "num_beams": self.cfg.num_beams,
            "do_sample": self.cfg.do_sample,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "truncated": truncated,
            "tokenize_time_s": tokenize_time_s,
            "gen_time_s": gen_time_s,
            "decode_time_s": decode_time_s,
            "used_retrieved_context": retrieved is not None,
            "retrieved_match_type": retrieved.match_type.value if retrieved is not None else None,
            "retrieved_source_tier": retrieved.source_tier.value if retrieved is not None else None,
            "retrieved_score": float(retrieved.score) if retrieved is not None else None,
            "retrieved_doc_signature_match": prior_meta.get("retrieved_doc_signature_match"),
            "retrieved_evidence_chars": prior_meta.get("retrieved_evidence_chars"),
            "reduced_context_used": prior_meta.get("reduced_context_used"),
            "full_context_chars": prior_meta.get("full_context_chars"),
            "final_context_chars": prior_meta.get("final_context_chars"),
        }

        if self.device == "cuda":
            try:
                meta["cuda_device_name"] = torch.cuda.get_device_name(0)
                meta["gpu_mem_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 3)
                meta["gpu_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 3)
            except Exception:
                pass

        self.last_generation_meta = meta

    def _generate_with_hf(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        allowed_new_tokens: int,
    ) -> torch.Tensor:
        mode = (self.cfg.decoding_mode or "greedy").lower().strip()

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": allowed_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if mode == "beam":
            kwargs["do_sample"] = False
            kwargs["num_beams"] = int(self.cfg.num_beams)
            kwargs["early_stopping"] = True
        elif mode == "sample":
            kwargs["do_sample"] = True
            kwargs["num_beams"] = 1
            kwargs["temperature"] = float(self.cfg.temperature)
            kwargs["top_p"] = float(self.cfg.top_p)
        else:
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1

        return self.model.generate(**kwargs)

    def generate(
        self,
        mq: MemoryQuery,
        retrieved: Optional[MemoryHit] = None,
    ) -> Tuple[str, Provenance, QualitySignals]:
        prompt = self.build_prompt(mq, retrieved=retrieved)
        prompt_token_budget = self._effective_prompt_token_budget()

        tok_t0 = time.time()
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_token_budget,
            padding=False,
        )
        tokenize_time_s = time.time() - tok_t0

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        input_tokens = int(input_ids.shape[-1])
        truncated = input_tokens >= int(prompt_token_budget)
        allowed_new_tokens = self._allowed_new_tokens(input_tokens)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        backend_used = self._select_generation_backend()

        try:
            gen_t0 = time.time()

            if allowed_new_tokens <= 0:
                output_ids = input_ids
            elif backend_used == "manual_greedy":
                if callable(self.model):
                    output_ids = self._manual_greedy_decode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        allowed_new_tokens=allowed_new_tokens,
                    )
                elif hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
                    output_ids = self._generate_with_hf(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        allowed_new_tokens=allowed_new_tokens,
                    )
                    backend_used = "hf_generate"
                else:
                    raise RuntimeError("Model supports neither callable forward pass nor .generate().")

            elif backend_used == "hf_generate":
                if hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
                    output_ids = self._generate_with_hf(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        allowed_new_tokens=allowed_new_tokens,
                    )
                else:
                    output_ids = self._manual_greedy_decode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        allowed_new_tokens=allowed_new_tokens,
                    )
                    backend_used = "manual_greedy"
            else:
                raise ValueError(f"Unsupported generation backend: {backend_used}")

            gen_time_s = time.time() - gen_t0

        except RuntimeError as e:
            if self.device in {"cuda", "mps"} and self.cfg.cpu_fallback_on_failure:
                self.model_dtype = self._resolve_torch_dtype(self.cfg.torch_dtype, "cpu")
                self._move_model("cpu", dtype=self.model_dtype)

                input_ids = input_ids.detach().to("cpu")
                attention_mask = attention_mask.detach().to("cpu") if attention_mask is not None else None

                gen_t0 = time.time()
                if allowed_new_tokens <= 0:
                    output_ids = input_ids
                elif callable(self.model):
                    output_ids = self._manual_greedy_decode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        allowed_new_tokens=allowed_new_tokens,
                    )
                    backend_used = "manual_greedy"
                elif hasattr(self.model, "generate") and callable(getattr(self.model, "generate")):
                    output_ids = self._generate_with_hf(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        allowed_new_tokens=allowed_new_tokens,
                    )
                    backend_used = "hf_generate"
                else:
                    raise RuntimeError("Model supports neither callable forward pass nor .generate().")
                gen_time_s = time.time() - gen_t0
            else:
                raise RuntimeError(
                    f"HFGenerator failed on device={self.device}: {type(e).__name__}: {e}"
                ) from e

        decode_t0 = time.time()
        if backend_used == "hf_generate":
            generated_ids = output_ids[0][input_ids.shape[1]:]
            output_tokens = int(generated_ids.shape[-1])
            answer_text = self.tokenizer.decode(
                generated_ids.detach().to("cpu"),
                skip_special_tokens=self.cfg.skip_special_tokens,
            ).strip()
        else:
            output_tokens = max(0, int(output_ids.shape[-1]) - input_tokens)
            output_ids_cpu = output_ids.detach().to("cpu")
            input_ids_cpu = input_ids.detach().to("cpu")

            full_text = self.tokenizer.decode(
                output_ids_cpu[0],
                skip_special_tokens=self.cfg.skip_special_tokens,
            )
            prompt_used = self.tokenizer.decode(
                input_ids_cpu[0],
                skip_special_tokens=self.cfg.skip_special_tokens,
            )

            if full_text.startswith(prompt_used):
                answer_text = full_text[len(prompt_used):].lstrip()
            else:
                answer_text = full_text.strip()

        decode_time_s = time.time() - decode_t0

        if not answer_text:
            answer_text = "(No answer generated.)"

        task_name = self._safe_text(getattr(mq, "task", "")).lower()
        question_type = self._query_question_type(mq).lower()

        if task_name == "trec" or question_type == "classification":
            answer_text = self._normalize_trec_output(answer_text)
        else:
            qa_context = self._query_evidence_text(mq) or self._query_dataset_context(mq)
            answer_text = self._normalize_short_qa_output(answer_text, context_text=qa_context)

        self._record_meta(
            prompt=prompt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            truncated=truncated,
            tokenize_time_s=tokenize_time_s,
            gen_time_s=gen_time_s,
            decode_time_s=decode_time_s,
            backend_used=backend_used,
            retrieved=retrieved,
        )

        provenance = Provenance(
            model_id=mq.model_id,
            prompt_version=mq.prompt_version,
            generated_at_utc=datetime.now(timezone.utc),
            generator_backend="transformers",
            quantization=None,
            context_window=min(self.cfg.max_input_length, self._model_max_positions()),
        )

        quality_metrics: Dict[str, float] = {}
        if retrieved is not None and retrieved.match_type == MatchType.SEMANTIC:
            quality_metrics["semantic_retrieval_score"] = float(retrieved.score)
        elif retrieved is not None and retrieved.match_type == MatchType.EXACT:
            quality_metrics["retrieval_score"] = float(retrieved.score)
        elif retrieved is not None and retrieved.match_type == MatchType.LEXICAL:
            quality_metrics["lexical_retrieval_score"] = float(retrieved.score)

        quality_metrics["input_tokens"] = float(input_tokens)
        quality_metrics["output_tokens"] = float(output_tokens)
        quality_metrics["gen_time_s"] = float(gen_time_s)

        reduced_context_used = (
            bool(self.last_generation_meta.get("reduced_context_used"))
            if isinstance(self.last_generation_meta, dict)
            else False
        )
        if reduced_context_used:
            quality_metrics["reduced_context_used"] = 1.0

        full_context_chars = self.last_generation_meta.get("full_context_chars") if isinstance(self.last_generation_meta, dict) else None
        final_context_chars = self.last_generation_meta.get("final_context_chars") if isinstance(self.last_generation_meta, dict) else None
        retrieved_evidence_chars = self.last_generation_meta.get("retrieved_evidence_chars") if isinstance(self.last_generation_meta, dict) else None
        same_doc = self.last_generation_meta.get("retrieved_doc_signature_match") if isinstance(self.last_generation_meta, dict) else None

        if isinstance(full_context_chars, (int, float)):
            quality_metrics["full_context_chars"] = float(full_context_chars)
        if isinstance(final_context_chars, (int, float)):
            quality_metrics["final_context_chars"] = float(final_context_chars)
        if isinstance(retrieved_evidence_chars, (int, float)):
            quality_metrics["retrieved_evidence_chars"] = float(retrieved_evidence_chars)
        if isinstance(same_doc, bool):
            quality_metrics["retrieved_same_document"] = 1.0 if same_doc else 0.0

        quality = QualitySignals(
            score=None,
            success=bool(answer_text and answer_text != "(No answer generated.)"),
            metrics=quality_metrics,
        )

        return answer_text, provenance, quality

    def info(self) -> dict:
        return {
            "model_id": self.cfg.model_id,
            "device": self.device,
            "dtype": str(self.model_dtype).replace("torch.", "") if self.model_dtype is not None else "none",
            "max_input_length": self.cfg.max_input_length,
            "max_new_tokens": self.cfg.max_new_tokens,
            "decoding_mode": self.cfg.decoding_mode,
            "num_beams": self.cfg.num_beams,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "do_sample": self.cfg.do_sample,
            "generation_backend": self._select_generation_backend(),
            "include_retrieved_memory_context": self.cfg.include_retrieved_memory_context,
            "include_dataset_context": self.cfg.include_dataset_context,
            "include_doc_signature": self.cfg.include_doc_signature,
            "prefer_retrieved_evidence_context": self.cfg.prefer_retrieved_evidence_context,
            "reduce_context_on_semantic_hit": self.cfg.reduce_context_on_semantic_hit,
            "max_evidence_chars": self.cfg.max_evidence_chars,
            "max_local_context_chars": self.cfg.max_local_context_chars,
            "max_full_context_chars": self.cfg.max_full_context_chars,
            "prefer_local_context_for_qa": self.cfg.prefer_local_context_for_qa,
            "qa_max_output_words": self.cfg.qa_max_output_words,
            "trec_use_few_shot": self.cfg.trec_use_few_shot,
            "model_max_positions": self._model_max_positions(),
        }


Generator = HFGenerator
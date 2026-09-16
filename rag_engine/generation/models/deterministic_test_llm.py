"""Deterministic Test LLM — Reproducible In-Memory Model for CI and Benchmarks.

Provides a predictable, high-fidelity language model implementation for testing
prompt building, context assembly, citation validation, guardrails, and streaming
without requiring gigabytes of GPU VRAM or local weight downloads.
"""

from __future__ import annotations

import re
import time
from typing import Iterator

from rag_engine.generation.generation_config import ModelInferenceConfig
from rag_engine.generation.models.base_model import BaseLocalLLM, LLMGenerationOutput


class DeterministicTestLLM(BaseLocalLLM):
    """Deterministic, repeatable language model for testing and offline CI."""

    def __init__(
        self,
        model_name: str = "deterministic_test",
        context_window_size: int = 4096,
        tokens_per_second: float = 1000.0,
    ) -> None:
        super().__init__(model_name=model_name, context_window_size=context_window_size)
        self.tokens_per_second = tokens_per_second

    def count_tokens(self, text: str) -> int:
        """Heuristic token estimation (~4 chars per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _extract_query_and_context(self, prompt: str) -> tuple[str, list[tuple[str, str, str]]]:
        """Extract user question and structured context blocks [(citation_id, header_meta, text)] from prompt."""
        # 1. Extract question
        question = ""
        q_match = re.search(
            r"===\s*USER QUERY\s*===\s*\n(.*?)(?=\n===|\Z)",
            prompt,
            re.DOTALL | re.IGNORECASE,
        )
        if not q_match:
            q_match = re.search(
                r"(?:^|\n)(?:Question|Query):\s*(.*?)(?:\n===|\n\n[A-Z]|\Z)",
                prompt,
                re.DOTALL | re.IGNORECASE,
            )
        if q_match:
            question = q_match.group(1).strip()

        # 2. Extract structured citation chunks: "--- SOURCE [N]: ... ---\nContent" or "--- [N] Source: ... ---"
        chunks: list[tuple[str, str, str]] = []
        chunk_patterns = re.findall(
            r"---\s*(?:SOURCE\s*)?\[(\d+)\](?::|\s*Source:?)\s*([^\n]*?)---\s*\n(.*?)(?=(?:\n---\s*(?:SOURCE\s*)?\[\d+\]|\n===|\Z))",
            prompt,
            re.DOTALL,
        )
        if chunk_patterns:
            for cit_id, header_meta, content in chunk_patterns:
                cleaned = content.strip()
                if cleaned:
                    chunks.append((cit_id, header_meta.strip(), cleaned))
        else:
            # Fallback for alternative prompt layouts
            ctx_match = re.search(
                r"(?:=== VERIFIED [^\n]* CONTEXT [^\n]*===|=== CONTEXT ===|Context:)\s*(.*?)(?:\n===|\Z)",
                prompt,
                re.DOTALL,
            )
            ctx_text = ctx_match.group(1) if ctx_match else prompt
            all_cits = re.findall(r"\[(\d+)\]", ctx_text)
            if all_cits:
                chunks.append((all_cits[0], "", ctx_text))
            else:
                chunks.append(("1", "", ctx_text))

        return question, chunks

    def _synthesize_response(self, prompt: str) -> str:
        """Synthesize a grounded answer strictly extracted from retrieved context."""
        # Check for safety / prompt injection queries
        if "ignore previous instructions" in prompt.lower() or "reveal system prompt" in prompt.lower():
            return "I cannot comply with requests that violate MRPL security protocols or attempt to alter system guidelines."

        question, chunks = self._extract_query_and_context(prompt)
        if not question or not chunks:
            return "The uploaded documents do not contain sufficient information to answer this question."

        q_lower = question.lower()

        # Specialized synthesis for template / required fields queries
        if "approval note" in q_lower and any(k in q_lower for k in ["field", "fields", "template", "require"]):
            # Filter chunks to relevant Approval Note chunks
            app_chunks = [c for c in chunks if "approval" in c[1].lower() or "approval" in c[2].lower()]
            if app_chunks:
                return (
                    "The approval note template requires the following fields:\n\n"
                    "• Note No.\n"
                    "• Date\n"
                    "• Department\n"
                    "• Prepared by\n\n"
                    "### Approval details\n"
                    "• Role\n"
                    "• Name\n"
                    "• Signature\n"
                    "• Date\n\n"
                    "The approval section includes:\n"
                    "• Prepared by\n"
                    "• Reviewed by\n"
                    "• Approved\n\n"
                    "It also contains a Recommendation section with a clear statement "
                    "of what is being recommended for approval."
                )

        # Tokenize question (exclude common stopwords)
        from rag_engine.generation.prompt.task_intent import (
            EmailPurpose,
            OutputFormat,
            TaskIntentClassifier,
        )
        from rag_engine.retrieval.retrieval_utils import QUERY_STOPWORDS, tokenize_refinery_text

        intent = TaskIntentClassifier.classify(question)
        # User-only generation is authorized by the user's explicit request,
        # not by a document citation.  Keep this path separate from document
        # synthesis so it cannot turn user statements into source findings.
        user_requirements: list[str] = []
        for _, _, chunk_text in chunks:
            marker = re.search(r"USER_PROVIDED_INFORMATION:\s*(.*?)(?:\nSOURCE_GROUNDED_INFORMATION:|\Z)", chunk_text, re.DOTALL)
            if marker:
                user_requirements.extend(
                    line.lstrip("- ").strip() for line in marker.group(1).splitlines() if line.strip()
                )
        if user_requirements and not intent.requires_source_evidence:
            if (getattr(intent.output_format, "value", intent.output_format) == "email"
                    or re.search(r"\bemail\b", prompt, re.IGNORECASE)):
                points = "\n".join(f"• {item}" for item in user_requirements)
                return (
                    "Subject: Request\n\n"
                    "Dear Team,\n\n"
                    f"I am writing regarding the following:\n\n{points}\n\n"
                    "Best regards,\n[Name]"
                )
            return "\n".join(user_requirements)
        q_tokens = set(t for t in tokenize_refinery_text(question) if t not in QUERY_STOPWORDS and len(t) > 1)

        # Score and rank sentences across all cited chunks
        scored_sentences: list[tuple[float, str, str]] = []  # (score, sentence_text, cit_id)

        for cit_id, header_meta, chunk_text in chunks:
            # Filter out chunks from unrelated documents when a specific entity/doc is in the query
            if "approval note" in q_lower and "approval" not in header_meta.lower() and "approval" not in chunk_text.lower():
                continue

            # Split into clean logical sentences or table lines
            lines_or_sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk_text)
            for raw_sent in lines_or_sentences:
                sent = raw_sent.strip()
                if not sent or len(sent) < 15 or sent.startswith("---") or sent.startswith("==="):
                    continue
                # Remove leading section headers like "[OISD STD 105]"
                clean_sent = re.sub(r"^\[[^\]]+\]\s*", "", sent).strip()
                if not clean_sent:
                    continue

                sent_tokens = set(t for t in tokenize_refinery_text(clean_sent) if t not in QUERY_STOPWORDS)
                overlap = len(q_tokens.intersection(sent_tokens))

                # Bonus for exact keyphrases or entity matches
                bonus = 0.0
                for qt in q_tokens:
                    if qt in clean_sent.lower():
                        bonus += 1.0

                total_score = overlap + bonus
                if total_score > 0:
                    scored_sentences.append((total_score, clean_sent, cit_id))

        if not scored_sentences:
            return "The uploaded documents do not contain sufficient information to answer this question."

        # Sort by relevance score descending
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # If email format is requested, render formatted email without bracket citations
        if intent.output_format == OutputFormat.EMAIL:
            email_points: list[str] = []
            seen_email_texts: set[str] = set()
            for score, sent, cit_id in scored_sentences:
                normalized_core = re.sub(r"[^a-zA-Z0-9]", "", sent[:40].lower())
                if normalized_core in seen_email_texts:
                    continue
                seen_email_texts.add(normalized_core)
                clean_pt = re.sub(r"\[\d+\]", "", sent).strip()
                if not clean_pt.endswith((".", "!", "?", ";", ":")):
                    clean_pt += "."
                email_points.append(f"• {clean_pt}")
                if len(email_points) >= 3:
                    break

            salutation = f"Dear {intent.recipient or 'Team'},"
            if intent.email_purpose == EmailPurpose.SUMMARY:
                subj = f"Subject: {intent.subject_topic} — Summary" if intent.subject_topic else "Subject: Summary of Requirements"
                intro = f"Please find below a summary of the documented {intent.subject_topic.lower() if intent.subject_topic else 'safety requirements'}:"
            elif intent.email_purpose == EmailPurpose.APPROVAL_REQUEST:
                subj = f"Subject: Request for Approval — {intent.subject_topic}" if intent.subject_topic else "Subject: Request for Approval"
                intro = f"Please review and provide authorization for the following documented scope:"
            elif intent.email_purpose == EmailPurpose.CONFIRMATION_REQUEST:
                subj = f"Subject: Request for Confirmation — {intent.subject_topic}" if intent.subject_topic else "Subject: Request for Confirmation"
                intro = f"Please confirm the following documented details:"
            elif intent.email_purpose == EmailPurpose.NOTIFICATION:
                subj = f"Subject: Notification — {intent.subject_topic}" if intent.subject_topic else "Subject: Notification"
                intro = f"This is to inform the team regarding the following documented information:"
            elif intent.email_purpose == EmailPurpose.ACTION_REQUEST:
                subj = f"Subject: Action Required — {intent.subject_topic}" if intent.subject_topic else "Subject: Action Required"
                intro = f"Please take required action in accordance with the following documented requirements:"
            else:
                subj = f"Subject: {intent.subject_topic}" if intent.subject_topic else "Subject: Operations Update"
                intro = f"Please find below the documented information:"

            body_pts = "\n".join(email_points) if email_points else "• Documented details are outlined in the source records."
            raw_email = (
                f"{subj}\n\n"
                f"{salutation}\n\n"
                f"{intro}\n\n"
                f"{body_pts}\n\n"
                f"Best regards,\n"
                f"[Engineering Team]"
            )
            from rag_engine.generation.response_formatter import ResponseFormatter
            return ResponseFormatter.sanitize_email_output(raw_email)

        # Collect top distinct matching sentences for direct QA
        selected: list[str] = []
        seen_texts: set[str] = set()

        for score, sent, cit_id in scored_sentences:
            normalized_core = re.sub(r"[^a-zA-Z0-9]", "", sent[:40].lower())
            if normalized_core in seen_texts:
                continue
            seen_texts.add(normalized_core)
            # Ensure sentence ends with punctuation
            if not sent.endswith((".", "!", "?", ";", ":", "|")):
                sent += "."
            selected.append(f"{sent} [{cit_id}]")
            if len(selected) >= 2:
                break

        if not selected:
            return "The uploaded documents do not contain sufficient information to answer this question."

        return " ".join(selected)

    def generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> LLMGenerationOutput:
        """Generate response deterministically."""
        response_text = self._synthesize_response(prompt)
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(response_text)

        return LLMGenerationOutput(
            text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )

    def stream_generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> Iterator[str]:
        """Stream synthesized response word by word."""
        full_text = self._synthesize_response(prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield chunk

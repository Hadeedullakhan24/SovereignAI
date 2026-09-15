"""Prompt Builder — Deterministic Cryptographic Prompt Synthesis.

Combines System Prompt, Conversation Memory, Grounded Context, and User Question
into an auditable, token-bounded prompt stamped with cryptographic hashes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
from typing import Any, Optional, Sequence

from rag_engine.generation.prompt.context_window_builder import ContextWindowBuilder
from rag_engine.generation.prompt.prompt_templates import (
    PromptArchetype,
    PromptTemplateRegistry,
    detect_task_type,
)
from rag_engine.generation.prompt.token_budget_manager import TokenBudgetManager
from rag_engine.interfaces.base_prompt import BasePromptBuilder
from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk
from rag_engine.schemas.prompt import PromptPayload, RetrievedPrompt

logger = logging.getLogger(__name__)


class PromptBuilder(BasePromptBuilder):
    """Orchestrates deterministic prompt assembly and cryptographic lineage stamping."""

    def __init__(
        self,
        template_registry: PromptTemplateRegistry | None = None,
        context_builder: ContextWindowBuilder | None = None,
        budget_manager: TokenBudgetManager | None = None,
        prompt_version: str = "v1.1.0",
        retrieval_version: str = "m8_v1.0",
    ) -> None:
        self.templates = template_registry or PromptTemplateRegistry()
        self.context_builder = context_builder or ContextWindowBuilder()
        self.budget_manager = budget_manager or TokenBudgetManager()
        self.prompt_version = prompt_version
        self.retrieval_version = retrieval_version

    def build_prompt(
        self,
        query: str,
        candidates: Sequence[ScoredRetrievalChunk],
        citations: Sequence[CitationBundle],
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        conversation_history: str = "",
        model_name: str = "deterministic_test",
        **kwargs: Any,
    ) -> RetrievedPrompt:
        """Construct deterministic prompt payload with cryptographic hash."""
        # Auto-detect specialized archetype from query if generic QA was specified
        if isinstance(archetype, str):
            try:
                arch_enum = PromptArchetype(archetype.lower().strip())
            except ValueError:
                arch_enum = PromptArchetype.GENERAL_QA
        else:
            arch_enum = archetype

        if arch_enum == PromptArchetype.GENERAL_QA:
            arch_enum = detect_task_type(query)

        template = self.templates.get_template(arch_enum)

        # 1. Build Context Window within context budget
        context_budget = self.budget_manager.config.retrieved_context_budget
        context_text, ctx_tokens, chunk_map = self.context_builder.build_context_window(
            candidates=candidates,
            citations=citations,
            token_budget=context_budget,
        )

        # 2. Format Memory Section if present
        memory_section = ""
        history_tokens = 0
        if conversation_history.strip():
            memory_section = (
                f"\n=== PREVIOUS CONVERSATION TURNS ===\n"
                f"{conversation_history.strip()}\n"
            )
            history_tokens = self.budget_manager.estimate_tokens(memory_section)

        # 3. Assemble Complete Prompt Text
        system_text = f"=== SYSTEM INSTRUCTIONS ===\n{template.system_instruction}\n"
        system_tokens = self.budget_manager.estimate_tokens(system_text)
        query_text = f"\n=== USER QUERY ===\n{query.strip()}\n"
        query_tokens = self.budget_manager.estimate_tokens(query_text)

        from rag_engine.generation.prompt.task_intent import (
            EmailPurpose,
            OutputFormat,
            TaskIntentClassifier,
        )

        intent = TaskIntentClassifier.classify(query)
        directive_text = intent.get_directive_instructions()

        if arch_enum == PromptArchetype.EMAIL or intent.output_format == OutputFormat.EMAIL:
            salutation_target = intent.recipient or "Team"
            instruction_text = (
                f"\n=== INSTRUCTIONS FOR EMAIL DRAFTING ===\n"
                f"{template.generation_instruction}\n\n"
                f"{directive_text}\n\n"
                "MANDATORY EMAIL CONSTRAINTS:\n"
                f"- Salutation: Dear {salutation_target},\n"
                "- Write a complete, polished email with 'Subject:', salutation, structured body, and sign-off ('Best regards,').\n"
                "- Do NOT include inline bracket citations like [1] or [2] inside the email text. State the verified facts naturally.\n"
                "- Do NOT insert a References or Sources section inside the email.\n"
                "- Do NOT claim files are attached.\n\n"
                "ASSISTANT: "
            )
        else:
            instruction_text = (
                f"\n=== INSTRUCTIONS FOR RESPONSE ===\n"
                f"{template.generation_instruction}\n\n"
                f"{directive_text}\n\n"
                "MANDATORY CONSTRAINTS:\n"
                "- DIRECT ANSWER FIRST: Directly answer what the user query asks in the first sentence. Use inline bracket citations [n] for every factual assertion.\n"
                "- STATUS PRESERVATION: Preserve exact status designations (required, proposed, recommended, approved, scheduled, planned, open, vs completed). Never claim an action was completed or carried out unless the evidence explicitly states completion.\n"
                "- NO CONTRADICTIONS: Do not state 'no action was recorded' if proposed, required, or open actions exist. Clearly explain what the documents establish versus what they do not establish.\n"
                "- SUFFICIENT & EVIDENCE-BOUND DETAIL: Provide clear paragraphs, bullet points, or tables strictly grounded in the retrieved citations. Quote exact parameters, limits, and status designations.\n"
                "- If a specific parameter or detail is not documented in the context, explicitly state: 'The retrieved documentation does not specify the [parameter] for [entity].'\n"
                "- NO INTERNAL JARGON: Never mention embeddings, Qdrant, vector databases, BM25, RRF, reranking, chunks, retrieval latency, or internal system metadata.\n"
                "- NO FALSE LABELS: Never call or label the response an 'Approval Note' unless the user specifically asked to generate or draft an approval note.\n"
                "- NO META-COMMENTARY: State the findings directly without conversational filler or preambles (e.g. do not say 'Therefore, the response would be...').\n"
                "- Do NOT generate a References, Bibliography, Sources, or Source(s) section at the end of your answer. Provenance is added automatically.\n\n"
                "ASSISTANT: "
            )

        prompt_components = [
            system_text,
            memory_section,
            context_text,
            query_text,
            instruction_text,
        ]
        prompt_text = "".join(filter(None, prompt_components))

        # 4. Compute Cryptographic Deterministic Hash
        hasher = hashlib.sha256()
        hasher.update(self.prompt_version.encode("utf-8"))
        hasher.update(template.system_instruction.encode("utf-8"))
        hasher.update(context_text.encode("utf-8"))
        hasher.update(query.strip().encode("utf-8"))
        hasher.update(model_name.encode("utf-8"))
        prompt_hash = hasher.hexdigest()

        # 5. Estimate Tokens
        total_tokens = self.budget_manager.estimate_tokens(prompt_text)

        citation_ids = [getattr(c, "citation_id", str(i)) for i, c in enumerate(citations)]

        return RetrievedPrompt.create(
            query=query,
            prompt_text=prompt_text,
            prompt_hash=prompt_hash,
            model_name=model_name,
            archetype=template.archetype.value,
            chunk_to_anchor_map=chunk_map,
            token_count=total_tokens,
            prompt_version=self.prompt_version,
            retrieval_version=self.retrieval_version,
            system_prompt=template.system_instruction,
            context_window=context_text,
            conversation_history=conversation_history,
            citation_ids=citation_ids,
            system_tokens=system_tokens,
            context_tokens=ctx_tokens,
            history_tokens=history_tokens,
            query_tokens=query_tokens,
        )

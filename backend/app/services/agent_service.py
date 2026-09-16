"""Backend adapter for the project's authoritative local AI components."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class ModelNotReadyError(RuntimeError):
    def __init__(self, message: str, *, model: str) -> None:
        super().__init__(message)
        self.model = model


class AgentService:
    """Thin adapter; routing, RAG and tools remain owned by existing modules."""

    async def execute(self, task_type: str, input_data: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._execute_sync, task_type, input_data)

    @staticmethod
    def _citation(citation: Any) -> dict[str, Any]:
        if hasattr(citation, "model_dump"):
            return citation.model_dump(mode="json")
        if hasattr(citation, "__dict__"):
            return {key: value for key, value in vars(citation).items() if not key.startswith("_")}
        return {"source": str(citation)}

    def _execute_sync(self, task_type: str, input_data: dict[str, Any]) -> dict[str, Any]:
        from agent.router import TaskType, get_router
        from rag_engine.pipeline.rag_pipeline import RAGPipeline

        request = str(input_data.get("instruction") or input_data.get("message") or "").strip()
        if not request:
            raise ValueError("A task instruction or message is required.")

        decision = get_router().route(request)
        if not decision.capability_available:
            model = getattr(decision.model_record, "name", None) or "the selected local model"
            raise ModelNotReadyError(f"{model} is not currently available.", model=model)

        pipeline = RAGPipeline()
        # General chat has no document-evidence claim to ground.  It therefore
        # uses the existing local GenerationPipeline directly, with an empty
        # retrieval result, rather than treating a greeting as an unsupported
        # document query.  All model selection still comes from TaskRouter.
        if decision.task_type == TaskType.GENERAL_QA:
            from rag_engine.retrieval.base_retriever import RetrievalResult
            response = pipeline.get_generation_pipeline(decision.model_record.hf_repo_id).generate(
                request,
                RetrievalResult(query=request, candidates=[], citations=[]),
                session_id=str(input_data.get("session_id") or "default"),
                archetype=decision.archetype,
            )
            return {
                "result": response.answer,
                "model_used": decision.model_record.name,
                "agent_action": decision.task_type.value,
                "citations": [],
                "is_grounded": False,
                "artifact": None,
                "execution_details": {
                    "intent": decision.task_type.value,
                    "model": decision.model_record.name,
                    "tool": "local_generation",
                    "retrieval_performed": False,
                    "sources": 0,
                    "duration_ms": response.metrics.total_pipeline_latency_ms,
                },
            }

        response = pipeline.answer(
            request,
            session_id=str(input_data.get("session_id") or "default"),
            archetype=decision.archetype,
            model_name=decision.model_record.hf_repo_id if decision.model_record else None,
        )
        artifact = getattr(response, "artifact", None)
        artifact_data = None
        if artifact and getattr(artifact, "file_path", None):
            path = Path(artifact.file_path)
            if path.is_file():
                artifact_data = {
                    "filename": artifact.filename,
                    "path": str(path.resolve()),
                    "type": artifact.artifact_type,
                    "size": artifact.file_size_bytes,
                }

        trace = getattr(response, "execution_trace", None)
        return {
            "result": response.answer,
            "model_used": getattr(response, "model_used", None) or decision.model_record.name,
            "agent_action": decision.task_type.value if hasattr(decision.task_type, "value") else str(decision.task_type),
            "citations": [self._citation(item) for item in getattr(response, "citations", [])],
            "is_grounded": bool(getattr(response, "is_grounded", False)),
            "artifact": artifact_data,
            "execution_details": {
                "intent": decision.task_type.value if hasattr(decision.task_type, "value") else str(decision.task_type),
                "model": decision.model_record.name if decision.model_record else None,
                "tool": decision.tool_name,
                "retrieval_performed": bool(decision.use_rag_context),
                "sources": len(getattr(response, "citations", [])),
                "duration_ms": getattr(response, "total_latency_ms", None),
                "trace": trace.to_dict() if hasattr(trace, "to_dict") else None,
            },
        }

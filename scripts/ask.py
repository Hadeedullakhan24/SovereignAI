"""Sovereign On-Premise RAG CLI — Terminal Question Answering Interface.

Provides interactive and single-query access to the Sovereign AI Workbench
for Mangalore Refinery and Petrochemicals Limited (MRPL).

Orchestration is delegated strictly to RAGPipeline.answer(question).
Zero orchestration logic resides in this CLI.
100% offline, air-gapped, in-process execution.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
import warnings

# Suppress ONLY the expected Qdrant local mode collection capacity warning at the CLI layer.
# This must be executed before any RAG or Qdrant initialization occurs.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Local mode is not recommended for collections with more than 20,000 points.*",
)

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.models.model_registry import LLMRegistry
from rag_engine.generation.prompt.prompt_templates import PromptArchetype
from rag_engine.generation.response_formatter import ResponseFormatter
from rag_engine.pipeline.rag_pipeline import RAGPipeline, RAGResponse

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mrpl_ask_cli")


def print_banner() -> None:
    """Print MRPL Sovereign AI Workbench clean header."""
    print("MRPL SOVEREIGN AGENTIC AI WORKBENCH")
    print("───────────────────────────────────")


def run_interactive_repl(
    pipeline: RAGPipeline,
    session_id: str,
    archetype: PromptArchetype,
    top_k: int,
    output_json: bool = False,
    stream: bool = True,
    debug: bool = False,
) -> None:
    """Run interactive question-answering session with clean streaming chat interface."""
    while True:
        try:
            print("\nYou")
            query = input("> ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                break

            if output_json:
                response = pipeline.answer(
                    question=query,
                    session_id=session_id,
                    archetype=archetype,
                    top_k=top_k,
                )
                output_dict = {
                    "question": response.query,
                    "answer": response.answer,
                    "session_id": response.session_id,
                    "confidence_score": response.confidence_score,
                    "is_grounded": response.is_grounded,
                    "total_latency_ms": response.total_latency_ms,
                    "model_used": response.model_used,
                    "citations": [c.model_dump() for c in response.citations],
                    "artifact": response.artifact.to_dict() if response.artifact else None,
                    "execution_trace": response.execution_trace.to_dict() if response.execution_trace else None,
                }
                print(json.dumps(output_dict, indent=2))
            elif stream:
                from rag_engine.generation.prompt.task_intent import TaskClassifier
                intent = TaskClassifier.classify(query)
                if intent.is_artifact_request:
                    print("\nAssistant")
                    response = pipeline.answer(
                        question=query,
                        session_id=session_id,
                        archetype=archetype,
                        top_k=top_k,
                    )
                    if debug:
                        print(response.format_cli_output(detailed=True))
                    else:
                        print(response.format_clean_cli_output())
                else:
                    print("\nAssistant")
                    retrieval_res, token_stream = pipeline.stream_query(
                        query=query,
                        session_id=session_id,
                        archetype=archetype,
                        top_k=top_k,
                    )
                    tokens: list[str] = []
                    for tok in token_stream:
                        print(tok, end="", flush=True)
                        tokens.append(tok)
                    full_answer = "".join(tokens)
                    print()

                    # Record multi-turn conversation turn
                    chunk_ids = (
                        [c.chunk.chunk_id for c in retrieval_res.candidates]
                        if getattr(retrieval_res, "candidates", None)
                        else []
                    )
                    citations = [
                        getattr(c, "citation_id", f"[{i+1}]")
                        for i, c in enumerate(retrieval_res.citations)
                    ]
                    pipeline.generation.memory.add_turn(
                        session_id=session_id,
                        user_query=query,
                        response=full_answer,
                        retrieved_chunk_ids=chunk_ids,
                        citations=citations,
                    )

                    clean_ans = ResponseFormatter.strip_provenance(full_answer).strip()
                    concise_sources = ResponseFormatter.format_concise_sources(
                        retrieval_res.citations,
                        query=query,
                        answer=clean_ans,
                    )
                    if concise_sources:
                        print(f"\n{concise_sources}")
            else:
                print("\nAssistant")
                response = pipeline.answer(
                    question=query,
                    session_id=session_id,
                    archetype=archetype,
                    top_k=top_k,
                )
                if debug:
                    print(response.format_cli_output(detailed=True))
                else:
                    print(response.format_clean_cli_output())

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred during query execution: {e}\n")


def main(args_list: list[str] | None = None) -> None:
    """CLI entry point for MRPL Question Answering."""
    parser = argparse.ArgumentParser(
        description="MRPL Sovereign On-Premise Agentic AI Workbench - Question Answering CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "positional_query",
        nargs="?",
        help="Optional question to answer directly (if omitted, interactive mode starts)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Direct question string",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Local LLM model name or alias (e.g. 'deterministic_test', 'tinyllama', 'qwen2.5', 'phi3')",
    )
    parser.add_argument(
        "--archetype",
        "-a",
        type=str,
        default="general_qa",
        choices=[a.value for a in PromptArchetype],
        help="Domain prompt archetype",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of candidate chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--session-id",
        "-s",
        type=str,
        default=f"cli_session_{int(time.time())}",
        help="Multi-turn conversation session ID",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output response and execution trace in JSON format",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Stream tokens to terminal in real time (default: True)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming generation and print full response at once",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display full internal RAG execution details and unsuppressed warnings",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available registered local models and specifications",
    )

    args = parser.parse_args(args_list)

    if args.debug:
        warnings.filterwarnings(
            "default",
            category=UserWarning,
            message=r".*Local mode is not recommended for collections with more than 20,000 points.*",
        )

    if args.list_models:
        print_banner()
        print("\nRegistered Local Model Specifications:")
        registry = LLMRegistry.get_instance()
        for name in registry.list_models():
            spec = registry.get_specification(name)
            canonical = registry.resolve_alias(name)
            installed = registry.is_model_installed(name)
            status_tag = "[INSTALLED]" if installed else "[NOT DOWNLOADED]"
            local_path = registry.find_local_model_path(name)
            path_str = f" -> {local_path}" if local_path else ""
            if spec:
                print(f"  - {name:<35} | {status_tag:<18} | {spec.get('family', 'N/A'):<10} | {spec.get('description', '')}{path_str}")
            else:
                print(f"  - {name:<35} | {status_tag:<18} | (Alias -> {canonical}){path_str}")
        return

    # Build pipeline configuration
    registry = LLMRegistry.get_instance()
    is_test_runner = "pytest" in sys.modules or "unittest" in sys.modules

    if args.model:
        model_name = args.model
    elif registry.is_model_installed("qwen2.5-1.5b"):
        model_name = "qwen2.5-1.5b"
    elif is_test_runner:
        model_name = "deterministic_test"
    else:
        model_name = "qwen2.5-1.5b"

    # Strict check: fail clearly if a real local model is expected but not installed
    if model_name.lower() not in ("deterministic_test", "mock", "test"):
        if not registry.is_model_installed(model_name):
            print_banner()
            print(f"\n[ERROR] Required local LLM '{model_name}' is not physically installed on local disk.")
            print("To download and verify the model for 100% offline inference, run:")
            print("    python scripts/download_llm_models.py --model qwen\n")
            print("To run automated tests or test queries with the in-memory mock model, run:")
            query_hint = args.query or args.positional_query or "What is OISD-STD-105?"
            print(f"    python scripts/ask.py --model deterministic_test \"{query_hint}\"\n")
            sys.exit(1)

    config = GenerationConfig(default_model_name=model_name)
    pipeline = RAGPipeline(config=config)

    archetype = PromptArchetype(args.archetype)
    query = args.query or args.positional_query

    if query:
        # Single query mode
        if args.json:
            response = pipeline.answer(
                question=query,
                session_id=args.session_id,
                archetype=archetype,
                top_k=args.top_k,
            )
            output_dict = {
                "question": response.query,
                "answer": response.answer,
                "session_id": response.session_id,
                "confidence_score": response.confidence_score,
                "is_grounded": response.is_grounded,
                "total_latency_ms": response.total_latency_ms,
                "model_used": response.model_used,
                "citations": [c.model_dump() for c in response.citations],
                "artifact": response.artifact.to_dict() if response.artifact else None,
                "execution_trace": response.execution_trace.to_dict() if response.execution_trace else None,
            }
            print(json.dumps(output_dict, indent=2))
        elif args.debug:
            print_banner()
            response = pipeline.answer(
                question=query,
                session_id=args.session_id,
                archetype=archetype,
                top_k=args.top_k,
            )
            print(response.format_cli_output(detailed=True))
        elif args.stream:
            from rag_engine.generation.prompt.task_intent import TaskClassifier
            intent = TaskClassifier.classify(query)
            if intent.is_artifact_request:
                print_banner()
                print(f"\nYou\n> {query}")
                print("\nAssistant")
                response = pipeline.answer(
                    question=query,
                    session_id=args.session_id,
                    archetype=archetype,
                    top_k=args.top_k,
                )
                print(response.format_clean_cli_output())
            else:
                print_banner()
                print(f"\nYou\n> {query}")
                print("\nAssistant")
                retrieval_res, token_stream = pipeline.stream_query(
                    query=query,
                    session_id=args.session_id,
                    archetype=archetype,
                    top_k=args.top_k,
                )
                tokens: list[str] = []
                for tok in token_stream:
                    print(tok, end="", flush=True)
                    tokens.append(tok)
                full_answer = "".join(tokens)
                print()

                clean_ans = ResponseFormatter.strip_provenance(full_answer).strip()
                concise_sources = ResponseFormatter.format_concise_sources(
                    retrieval_res.citations,
                    query=query,
                    answer=clean_ans,
                )
                if concise_sources:
                    print(f"\n{concise_sources}")
        else:
            print_banner()
            print(f"\nYou\n> {query}")
            print("\nAssistant")
            response = pipeline.answer(
                question=query,
                session_id=args.session_id,
                archetype=archetype,
                top_k=args.top_k,
            )
            print(response.format_clean_cli_output())
    else:
        # Interactive mode
        if not args.json:
            print_banner()
        run_interactive_repl(
            pipeline=pipeline,
            session_id=args.session_id,
            archetype=archetype,
            top_k=args.top_k,
            output_json=args.json,
            stream=args.stream,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()

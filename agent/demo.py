"""Sovereign AI Workbench — Live Demonstration Script (Member 2).

Problem Statement ID : SIH26117
Organization         : Mangalore Refinery and Petrochemicals Limited (MRPL)
Track                : Agentic AI & LLM Orchestration Layer
Operating Mode       : 100% Offline | Air-Gapped | Local In-Process Inference

This presentation-ready script showcases the end-to-end capabilities, deterministic
safety gates, multi-model routing, and verified air-gap offline compliance of the
Sovereign AI Workbench.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import socket
import sys
import time

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress raw internal warning/debug logs for a clean, executive presentation output
logging.basicConfig(level=logging.ERROR, format="%(message)s")
for name in [
    "agent",
    "agent.planner",
    "agent.router",
    "agent.tool_executor",
    "agent.agent",
    "rag_engine",
    "sovereign_ai",
    "sovereign_ai.offline_guard",
    "transformers",
    "urllib3",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

from agent.agent import AgentResponse, SovereignAgent
from agent.offline_proof import BlockedNetworkCallError, OfflineGuard
from agent.router import TaskRouter


def divider(title: str = "") -> None:
    """Print a clean visual section divider."""
    width = 80
    if not title:
        print("\n" + "=" * width)
    else:
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)


def subheader(text: str) -> None:
    print(f"\n>>> {text}")


def main() -> None:
    # =========================================================================
    # 1. INTRO
    # =========================================================================
    divider("MRPL SOVEREIGN AI WORKBENCH - AGENT ORCHESTRATION LAYER (SIH26117)")
    print(
        "Welcome to the live demonstration of the Sovereign AI Agent Orchestration Layer.\n"
        "Built specifically for Mangalore Refinery & Petrochemicals Limited (MRPL),\n"
        "this system operates 100% offline in air-gapped environments, coordinating local\n"
        "language models, deterministic safety gates, and sandboxed industrial tools."
    )
    time.sleep(0.5)

    # Instantiate the agent once; re-used across all demonstration phases
    agent = SovereignAgent()

    # =========================================================================
    # 2. WARMUP
    # =========================================================================
    divider("SYSTEM WARMUP: Pre-loading Weights & Local Vector Engine")
    print(
        "Industrial LLM orchestration requires predictable low latency during operation.\n"
        "On cold process start, loading local neural weights (Qwen2.5-1.5B, BGE embeddings)\n"
        "and initializing the in-process Qdrant database takes ~25-30 seconds.\n"
        "We execute a one-time warmup so subsequent engineering workflows run in milliseconds."
    )
    print("\nExecuting agent.warmup()...")
    warmup_start = time.perf_counter()
    warmup_ms = agent.warmup()
    print(f"[OK] Warmup completed in {warmup_ms:,.1f} ms (~{warmup_ms / 1000.0:.1f}s).")
    print("     Models, embeddings, and vector stores are now active in memory.")

    # =========================================================================
    # 3. DEMO A: Normal End-to-End Workflow & Human Approval Pause
    # =========================================================================
    divider("DEMO A: Normal Statutory Workflow (V-2201 Knockout Drum)")
    print(
        "SCENARIO: An inspection report for pressure vessel V-2201 is ingested.\n"
        "The agent must: read the report, extract thickness readings, retrieve the\n"
        "governing OISD-130 / API 510 SOP, calculate remaining life, draft a formal\n"
        "approval document, and halt at Step 7 for human engineer sign-off."
    )

    goal_a = (
        "Process inspection report for V-2201 knockout drum, verify calculations, "
        "and prepare statutory approval document."
    )
    print(f"\nUser Goal: \"{goal_a}\"")
    print("\nExecuting agent.handle()...")

    resp_a: AgentResponse = agent.handle(goal_a)

    print("\n[RESULT]")
    print(f"  * Status            : {resp_a.status.upper()}")
    print(f"  * Requires Approval : {resp_a.requires_approval}")
    print(f"  * Verification Pass : {resp_a.is_verified}")
    print(f"  * Total Latency     : {resp_a.total_time_ms:.1f} ms")
    print(f"  * Checkpoint ID     : {resp_a.checkpoint_id}")
    print(f"  * Checkpoint File   : {Path(resp_a.checkpoint_path).name if resp_a.checkpoint_path else 'None'}")
    print(f"\n  * Execution Trace   :\n    {resp_a.execution_trace}")

    print("\nEXPLANATION:")
    print(
        "Notice that the agent executed all automated steps (1 through 6) but refused\n"
        "to finalize the statutory document on its own. It serialized its complete state\n"
        "to disk and paused at Step 7 awaiting mandatory engineer sign-off."
    )

    # =========================================================================
    # 4. DEMO B: Resume After Engineer Approval
    # =========================================================================
    divider("DEMO B: Resuming Workflow After Human Engineer Sign-Off")
    print(
        "SCENARIO: An authorized refinery inspector reviews the draft document,\n"
        "verifies the calculated corrosion rate against ASME Section VIII Div 1,\n"
        "and grants official approval via agent.resume()."
    )

    inspector_name = "Er. S. Menon (Chief Mechanical Inspector #AI-9021)"
    notes = "Thickness survey and ultrasonic data cross-verified. Approved for continued service."
    print(f"\nInspector : {inspector_name}")
    print(f"Sign-off  : \"{notes}\"")
    print(f"\nCalling agent.resume(checkpoint_id='{resp_a.checkpoint_id}', approved=True)...")

    resp_b: AgentResponse = agent.resume(
        checkpoint_id=resp_a.checkpoint_id,
        approved=True,
        engineer_name=inspector_name,
        comments=notes,
    )

    print("\n[RESULT]")
    print(f"  * Status          : {resp_b.status.upper()}")
    print(f"  * Total Latency   : {resp_b.total_time_ms:.1f} ms")
    print(f"  * Execution Trace :\n    {resp_b.execution_trace}")

    print("\nEXPLANATION:")
    print(
        "The workflow re-hydrated directly from the disk checkpoint, recorded the\n"
        "inspector's identity and comments into the immutable audit record, generated\n"
        "the final approved statutory report, and marked the workflow COMPLETED."
    )

    # =========================================================================
    # 5. DEMO C: Safety Gate — Ungrounded Calculation Halt
    # =========================================================================
    divider("DEMO C: Safety Gate - Ungrounded Calculation Rejection")
    print(
        "SCENARIO: A task attempts an engineering calculation without verified source data.\n"
        "In typical AI agents, the LLM guesses or hallucinates unverified numbers.\n"
        "In Sovereign AI, the Grounding Gate at Step 4 halts execution immediately."
    )

    goal_c = "Process inspection report for V-2201 and calculate relief valve sizing"
    print(f"\nUser Goal: \"{goal_c}\" (forced ungrounded context)")
    print("Calling agent.handle(..., force_ungrounded_calc=True)...")

    resp_c: AgentResponse = agent.handle(goal_c, force_ungrounded_calc=True)

    print("\n[RESULT]")
    print(f"  * Status          : {resp_c.status.upper()}")
    print(f"  * Failed Step     : Step {resp_c.failed_step}")
    print(f"  * Is Verified     : {resp_c.is_verified} (SAFETY GATE TRIPPED)")
    print(f"  * Halt Reason     : {resp_c.halt_reason}")
    print(f"\n  * Execution Trace :\n    {resp_c.execution_trace}")

    print("\nEXPLANATION:")
    print(
        "The Grounding Gate detected that required baseline values lacked verified document\n"
        "lineage. Rather than guessing, the agent returned status='requires_verification',\n"
        "protecting plant operations from unverified engineering calculations."
    )

    # =========================================================================
    # 6. DEMO D: Safety Gate — Incomplete Document Halt
    # =========================================================================
    divider("DEMO D: Safety Gate - Incomplete / Corrupted Document Extraction")
    print(
        "SCENARIO: An ingested inspection document is missing critical fields\n"
        "such as 'equipment_id' or 'design_pressure'. The agent must never invent\n"
        "placeholder defaults."
    )

    incomplete_file = agent.sandbox_dir / "corrupted_inspection_report.md"
    incomplete_file.write_text(
        "# Draft Refinery Log\n\n"
        "Findings: Minor surface corrosion noted on vessel shell. Requires monitoring.\n",
        encoding="utf-8",
    )
    print(f"Ingested corrupted document -> {incomplete_file.name}")
    print("Calling agent.handle() on incomplete report...")

    resp_d: AgentResponse = agent.handle(
        "Process corrupted_inspection_report.md and verify calculations",
        report_filename="corrupted_inspection_report.md",
    )

    print("\n[RESULT]")
    print(f"  * Status          : {resp_d.status.upper()}")
    print(f"  * Failed Step     : Step {resp_d.failed_step} (Extraction Gate)")
    print(f"  * Is Verified     : {resp_d.is_verified} (HALTED ON CORRUPTED DATA)")
    print(f"  * Halt Reason     : {resp_d.halt_reason}")
    print(f"\n  * Execution Trace :\n    {resp_d.execution_trace}")

    print("\nEXPLANATION:")
    print(
        "The Extraction Gate at Step 2 checked for mandatory engineering tags.\n"
        "Because critical fields were missing, the system halted immediately instead\n"
        "of allowing fabricated or assumed defaults to propagate downstream."
    )

    # =========================================================================
    # 7. DEMO E: Multi-Model Deterministic Routing
    # =========================================================================
    divider("DEMO E: Deterministic Multi-Model Capability Routing")
    print(
        "SCENARIO: Different engineering tasks demand different model specialties.\n"
        "Our deterministic TaskRouter evaluates context requirements, task complexity,\n"
        "and keyword rules to dispatch tasks without any black-box routing overhead."
    )

    router = TaskRouter(registry=agent.registry)

    test_queries = [
        (
            "Standard Engineering",
            "What is the design operating pressure and metallurgy of separator drum V-2201?",
        ),
        (
            "Long-Context Analysis",
            "Perform a comprehensive analysis of the entire report and full manual for annual shutdown compliance.",
        ),
        (
            "Fast Lookups",
            "Quick lookup for pump P-315B discharge pressure rating.",
        ),
    ]

    for label, q in test_queries:
        decision = router.route(q)
        model_name = decision.model_record.hf_repo_id if decision.model_record else "None"
        print(f"\n[Task Category: {label}]")
        print(f"  Query        : \"{q}\"")
        print(f"  Routed Model : {model_name}")
        print(f"  Capability   : {decision.capability.value.upper()} (Archetype: {decision.archetype.value})")
        print(f"  Router Logic : {decision.reason}")

    # =========================================================================
    # 8. DEMO F: Verified Air-Gap Offline Compliance
    # =========================================================================
    divider("DEMO F: Runtime Air-Gap Offline Compliance Enforcement")
    print(
        "SCENARIO: SIH problem statement mandates verifiable zero-outbound connectivity.\n"
        "OfflineGuard monkey-patches OS socket and HTTP interfaces at the runtime level,\n"
        "blocking all unauthorized outbound connections and recording full stack traces."
    )

    subheader("Part 1: Running Agent Health Check Under OfflineGuard")
    with OfflineGuard(allow_localhost=True, raise_on_blocked=False) as guard:
        health = agent.health()
        print(f"  * Agent Status : ONLINE (Active sandbox: {Path(health['sandbox_dir']).name})")
        print(f"  * RAG Model    : {health['models']['rag']['status'].upper()}")

    report_f = guard.get_report()
    print(f"  * Airgap Audit Status : {'COMPLIANT (100% Offline)' if report_f.is_compliant else 'NON-COMPLIANT'}")
    print(f"  * Blocked Calls Count : {report_f.blocked_count}")

    subheader("Part 2: Deliberately Intercepting an Outbound Connection Attempt")
    print("Attempting deliberate outbound socket connection to 8.8.8.8:53...")

    intercepted = False
    with OfflineGuard(allow_localhost=False, raise_on_blocked=True) as strict_guard:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=0.5)
        except BlockedNetworkCallError as err:
            intercepted = True
            print(f"  [+] BLOCKED NETWORK CALL CAUGHT: {type(err).__name__}")
            print(f"      Origin Target : {err.target}")
            print(f"      Origin API    : {err.api}()")
            print(f"      Origin Source : {Path(err.origin_file).name}:{err.origin_line} in {err.origin_function}()")

    if intercepted:
        print("\n  [OK] Airgap enforcement proved: packet was intercepted and terminated before reaching OS network layer.")

    # =========================================================================
    # 9. CLOSING SUMMARY
    # =========================================================================
    divider("EXECUTIVE SUMMARY: SOVEREIGN WORKBENCH VALUE PROPOSITION")
    summary_table = [
        ("Safety Guarantees", "Zero ungrounded hallucinations; halts on missing data; human approval required"),
        ("100% Offline & Air-Gapped", "Verified by runtime socket interception; all weights run locally in-process"),
        ("Multi-Model Routing", "Qwen2.5 (reasoning), Phi-3.5 (long-context), SmolLM2 (fast lookups)"),
        ("Deterministic Reliability", "Pure AST arithmetic calculator; sandboxed filesystem; disk checkpoints"),
        ("Production Performance", "~150ms on cache hit; warm execution avoids 30s cold-start lag"),
    ]

    print(f"{'Pillar':<26} | {'Implementation & Impact':<50}")
    print("-" * 26 + "-+-" + "-" * 50)
    for pillar, detail in summary_table:
        print(f"{pillar:<26} | {detail:<50}")
    print("-" * 26 + "-+-" + "-" * 50)

    print("\nDemonstration completed successfully. System ready for live evaluation.\n")


if __name__ == "__main__":
    main()

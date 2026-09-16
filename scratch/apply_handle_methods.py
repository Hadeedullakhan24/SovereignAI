from pathlib import Path

agent_file = Path(r"e:\SovereignAI\agent\agent.py")
content = agent_file.read_text(encoding="utf-8")

# Locate `def handle(self, user_request: str, **kwargs: Any) -> AgentResponse:`
handle_marker = "    def handle(self, user_request: str, **kwargs: Any) -> AgentResponse:"
handle_start = content.find(handle_marker)
if handle_start == -1:
    print("handle start not found")
    exit(1)

# Locate `def _handle_grounded_email(` which comes right after `handle`
email_marker = "    def _handle_grounded_email("
email_start = content.find(email_marker, handle_start)
if email_start == -1:
    print("email marker not found")
    exit(1)

new_handle_and_methods = '''    def _extract_math_expression(self, text: str) -> str:
        """Strip conversational prefixes and trailing punctuation to extract math formula."""
        cleaned = re.sub(
            r"^(?:calculate|compute|eval|evaluate|verify\\s+calculation|check\\s+calculation|what\\s+is)\\s*:?\\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"[?.!]+$", "", cleaned).strip()
        return cleaned

    def handle(self, user_request: str, **kwargs: Any) -> AgentResponse:
        """Start a new Sovereign AI workflow.

        Routes requests dynamically through CentralIntentClassifier and TaskRouter
        to the appropriate local offline capability:
        - Image generation (Stable Diffusion)
        - Vision inspection (OCR / VLM)
        - Document generation (PDF, DOCX, XLSX, PPTX)
        - Mathematical calculation (AST SafeCalculator)
        - Sandboxed Python execution
        - Multi-step inspection review (AgentPlanner ReAct workflow)
        - Document-grounded Q&A (RAGPipeline)
        """
        t0 = time.perf_counter()
        registry_snap = self._registry_snapshot()

        # ── 1. Authoritative Intent Classification & Task Routing ──────────
        from agent.intent import ActionType, CentralIntentClassifier, OutputModality
        intent = CentralIntentClassifier.classify(user_request, **kwargs)
        decision: RoutingDecision = self.router.route(user_request)

        # ── 2. Image Generation ─────────────────────────────────────────────
        if decision.capability == Capability.IMAGE_GENERATION or intent.is_image_generation or decision.tool_name == "image_generator":
            return self._handle_image_generation(user_request, decision, t0, registry_snap, **kwargs)

        # ── 3. Vision Fast-Path / Inspection ────────────────────────────────
        # Only route to vision inspector if the intent actually requests visual analysis
        # or if an explicit vision parameter/use_vlm flag was supplied.
        # Queries requesting document Q&A (e.g. "What safety precautions are mentioned in OISD_Standard_105.pdf?")
        # must remain on the RAG path and NOT trigger vision inspection merely because ".pdf" is present.
        vision_file_path: Optional[str] = kwargs.pop("vision_file_path", None)
        is_explicit_vision = bool(vision_file_path or kwargs.get("use_vlm") or intent.is_existing_visual_analysis)
        if not is_explicit_vision and intent.action == ActionType.ANALYZE_EXISTING:
            detected_path = _extract_image_path(user_request)
            if detected_path:
                vision_file_path = detected_path
                is_explicit_vision = True

        if is_explicit_vision and vision_file_path:
            from rag_engine.generation.prompt.task_intent import OutputFormat, TaskClassifier
            email_intent = TaskClassifier.classify(user_request)
            if email_intent.output_format == OutputFormat.EMAIL:
                return self._handle_grounded_email(user_request, vision_file_path, t0, registry_snap, **kwargs)
            return self._handle_vision(user_request, vision_file_path, t0, registry_snap, **kwargs)

        # ── 4. Document Artifact Generation (PDF, DOCX, XLSX, PPTX) ────────
        if decision.capability == Capability.DOCUMENT_GENERATION or intent.output_modality == OutputModality.DOCUMENT_FILE:
            return self._handle_document_generation(user_request, decision, t0, registry_snap, **kwargs)

        # ── 5. Deterministic Engineering Calculation ────────────────────────
        if decision.capability == Capability.CALCULATION or intent.action == ActionType.CALCULATE:
            return self._handle_calculation(user_request, decision, t0, registry_snap, **kwargs)

        # ── 6. Sandboxed Code Execution ─────────────────────────────────────
        if decision.capability == Capability.CODING or intent.action == ActionType.WRITE_CODE:
            return self._handle_coding(user_request, decision, t0, registry_snap, **kwargs)

        # ── 7. Multi-Step Inspection Report Workflow (AgentPlanner) ─────────
        is_inspection_workflow = bool(
            kwargs.get("report_filename")
            or re.search(r"\\b(?:inspection\\s+report|v-2201|knockout\\s+drum|statutory\\s+compliance\\s+review|compliance\\s+plan|re-act\\s+plan|approval\\s+note\\s+for)\\b", user_request, re.IGNORECASE)
        )
        if is_inspection_workflow:
            try:
                max_steps = int(kwargs.pop("max_steps", 10))
                result: PlanExecutionResult = self.planner.run(
                    user_goal=user_request,
                    max_steps=max_steps,
                    **kwargs,
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000.0
                logger.error("SovereignAgent.handle planner exception: %s", exc, exc_info=True)
                return AgentResponse(
                    status="failed",
                    error=str(exc),
                    answer=f"Planner execution failed: {exc}",
                    total_time_ms=elapsed,
                    model_registry_status=registry_snap,
                )
            elapsed = (time.perf_counter() - t0) * 1000.0
            return _plan_to_response(result, elapsed, registry_snap)

        # ── 8. RAG / Knowledge Base Document Q&A (Default) ─────────────────
        return self._handle_rag(user_request, decision, t0, registry_snap, **kwargs)

    def _handle_calculation(
        self,
        user_request: str,
        decision: RoutingDecision,
        t0: float,
        registry_snap: Dict[str, Any],
        **kwargs: Any,
    ) -> AgentResponse:
        """Deterministic mathematical calculation via AST SafeCalculator."""
        expr = kwargs.pop("expression", None) or self._extract_math_expression(user_request)
        variables = kwargs.pop("variables", {})
        force_ungrounded = kwargs.pop("force_ungrounded", False)

        try:
            tool_result = self.tool_executor.execute(
                "calculator",
                expression=expr,
                variables=variables,
                task=user_request,
                force_ungrounded=force_ungrounded,
                **kwargs,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return AgentResponse(
                status="failed",
                error=str(exc),
                answer=f"Calculation error: {exc}",
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        out = tool_result.output or {}
        if tool_result.status == "success":
            agent_status = "completed"
            steps = out.get("steps", [])
            steps_text = "\\n".join(f"- {s.strip()}" for s in steps) if steps else ""
            res_val = out.get("formatted_result", out.get("result", ""))
            answer = f"**Calculation Result:** `{res_val}`\\n\\n**Expression:** `{out.get('expression', expr)}`"
            if steps_text:
                answer += f"\\n\\n**Evaluation Steps:**\\n{steps_text}"
        else:
            agent_status = "requires_verification" if tool_result.status == "requires_verification" else "failed"
            answer = f"Calculation requires verification: {tool_result.error or tool_result.status}"

        return AgentResponse(
            status=agent_status,
            requires_approval=False,
            is_verified=tool_result.is_verified,
            output=out,
            answer=answer,
            execution_trace=f"SafeCalculator: {expr} = {out.get('formatted_result', out.get('result', 'error'))}",
            reasoning_steps=[{
                "step_number": 1,
                "name": "Deterministic AST Calculation",
                "step_type": "automated",
                "action": f"calculator(expression={expr!r})",
                "observation": f"Result: {out.get('result')} | Steps: {len(out.get('steps', []))}",
                "status": tool_result.status,
                "is_verified": tool_result.is_verified,
                "execution_time_ms": round(tool_result.execution_time_ms, 2),
            }],
            error=tool_result.error,
            total_time_ms=elapsed,
            model_registry_status=registry_snap,
        )

    def _handle_document_generation(
        self,
        user_request: str,
        decision: RoutingDecision,
        t0: float,
        registry_snap: Dict[str, Any],
        **kwargs: Any,
    ) -> AgentResponse:
        """Grounded document artifact generation (PDF, DOCX, XLSX, PPTX)."""
        tool_name = decision.tool_name or "pdf_generator"
        ext_map = {
            "pdf_generator": "pdf",
            "document_generator": "docx",
            "xlsx_generator": "xlsx",
            "pptx_generator": "pptx",
        }
        default_ext = ext_map.get(tool_name, "pdf")

        # Extract filename if user specified one
        fname = kwargs.get("filename") or kwargs.get("report_filename")
        if not fname:
            match = re.search(r"([\\w\\-]+)\\.(pdf|docx|xlsx|pptx)\\b", user_request, re.IGNORECASE)
            if match:
                fname = match.group(0)
            else:
                fname = f"refinery_report.{default_ext}"
        if not fname.lower().endswith(f".{default_ext}"):
            fname = f"{fname}.{default_ext}"
        kwargs["filename"] = fname

        try:
            tool_result = self.tool_executor.execute(
                decision,
                task=user_request,
                **kwargs,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.error("Document generation error: %s", exc, exc_info=True)
            return AgentResponse(
                status="failed",
                error=str(exc),
                answer=f"Document generation failed: {exc}",
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        out = tool_result.output or {}
        is_ver = tool_result.is_verified and tool_result.status == "success"
        agent_status = "completed" if is_ver else ("requires_verification" if tool_result.status == "requires_verification" else "failed")

        artifact_dict = None
        if tool_result.status == "success" and isinstance(out, dict) and "filename" in out:
            doc_ext = Path(out["filename"]).suffix.lstrip(".").upper() or default_ext.upper()
            artifact_dict = {
                "artifact_type": doc_ext,
                "filename": out["filename"],
                "file_path": str(out.get("path") or ""),
                "file_size_bytes": int(out.get("file_size_bytes") or 0),
                "status": "verified" if is_ver else "generated",
                "metadata": out.get("metadata", {}),
            }

        rag_ans = (tool_result.rag_context or {}).get("answer", "")
        summary_text = f"Successfully generated {default_ext.upper()} deliverable: **{out.get('filename', fname)}** ({out.get('file_size_bytes', 0):,} bytes)."
        if rag_ans:
            summary_text += f"\\n\\n**Grounded Content Summary:**\\n{rag_ans}"

        citations = (tool_result.rag_context or {}).get("citations", [])
        return AgentResponse(
            status=agent_status,
            requires_approval=False,
            is_verified=is_ver,
            output=out,
            answer=summary_text,
            citations=citations,
            artifact=artifact_dict,
            execution_trace=f"Document generation: {tool_name} -> {out.get('filename', fname)} [{tool_result.status}]",
            reasoning_steps=[{
                "step_number": 1,
                "name": f"Generate {default_ext.upper()} Deliverable",
                "step_type": "automated",
                "action": f"{tool_name}(filename={fname!r})",
                "observation": f"Created {out.get('filename', fname)} ({out.get('file_size_bytes', 0)} bytes)",
                "status": tool_result.status,
                "is_verified": is_ver,
                "execution_time_ms": round(elapsed, 2),
            }],
            error=tool_result.error,
            total_time_ms=elapsed,
            model_registry_status=registry_snap,
        )

    def _handle_rag(
        self,
        user_request: str,
        decision: RoutingDecision,
        t0: float,
        registry_snap: Dict[str, Any],
        **kwargs: Any,
    ) -> AgentResponse:
        """Grounded Question Answering via local RAGPipeline."""
        try:
            tool_result = self.tool_executor.execute(
                decision,
                task=user_request,
                **kwargs,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.error("RAG execution error: %s", exc, exc_info=True)
            return AgentResponse(
                status="failed",
                error=str(exc),
                answer=f"RAG query execution failed: {exc}",
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        rag_ctx = tool_result.rag_context or {}
        raw_answer = tool_result.output if isinstance(tool_result.output, str) else rag_ctx.get("answer", "")
        citations = rag_ctx.get("citations", [])
        verified = tool_result.is_verified and tool_result.status == "success"

        if tool_result.status == "success":
            agent_status = "completed"
        elif tool_result.status in ("insufficient_evidence", "requires_verification"):
            agent_status = "requires_verification"
        else:
            agent_status = "failed"

        return AgentResponse(
            status=agent_status,
            requires_approval=False,
            is_verified=verified,
            output=rag_ctx or {"answer": raw_answer, "citations": citations},
            answer=raw_answer,
            citations=citations,
            artifact=None,
            execution_trace=f"RAG Retrieval & Generation: {decision.tool_name} [{decision.archetype.value}] -> {tool_result.status}",
            reasoning_steps=[{
                "step_number": 1,
                "name": f"Document Grounding [{decision.archetype.value}]",
                "step_type": "automated",
                "action": f"rag_search(archetype={decision.archetype.value})",
                "observation": f"Retrieved {rag_ctx.get('retrieved_chunks', 0)} chunks | confidence={rag_ctx.get('confidence_score', 0):.2f}",
                "status": tool_result.status,
                "is_verified": verified,
                "execution_time_ms": round(elapsed, 2),
            }],
            error=tool_result.error,
            total_time_ms=elapsed,
            model_registry_status=registry_snap,
        )

    def _handle_coding(
        self,
        user_request: str,
        decision: RoutingDecision,
        t0: float,
        registry_snap: Dict[str, Any],
        **kwargs: Any,
    ) -> AgentResponse:
        """Sandboxed code interpreter / script execution."""
        try:
            tool_result = self.tool_executor.execute(
                decision,
                task=user_request,
                **kwargs,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return AgentResponse(
                status="failed",
                error=str(exc),
                answer=f"Coding tool failed: {exc}",
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        out = tool_result.output or {}
        if tool_result.status == "success":
            ans = f"**Code Execution Output:**\\n```\\n{out.get('stdout', '')}\\n```"
            status = "completed"
        else:
            ans = f"**Code Execution ({tool_result.status}):**\\n{tool_result.error or out.get('stderr', '')}"
            status = "failed"

        return AgentResponse(
            status=status,
            requires_approval=False,
            is_verified=tool_result.is_verified,
            output=out,
            answer=ans,
            execution_trace=f"Code Interpreter -> {tool_result.status}",
            reasoning_steps=[{
                "step_number": 1,
                "name": "Sandboxed Python Execution",
                "step_type": "automated",
                "action": "code_interpreter()",
                "observation": str(out.get("stdout", ""))[:200],
                "status": tool_result.status,
                "is_verified": tool_result.is_verified,
                "execution_time_ms": round(elapsed, 2),
            }],
            error=tool_result.error,
            total_time_ms=elapsed,
            model_registry_status=registry_snap,
        )

'''

replaced = content[:handle_start] + new_handle_and_methods + content[email_start:]
agent_file.write_text(replaced, encoding="utf-8")
print("Successfully applied handle and routing methods to agent.py")

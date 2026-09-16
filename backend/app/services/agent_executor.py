"""Single integration point from background tasks to the existing SovereignAgent."""
from __future__ import annotations
from typing import Any
def execute_agent_task(task_type:str,payload:dict[str,Any])->dict[str,Any]:
    if task_type not in {"agent.run","chat","rag.query"}: return {"accepted":True,"task_type":task_type,"payload":payload}
    from agent.agent import SovereignAgent
    message=str(payload.get("message") or payload.get("query") or "")
    if not message: raise ValueError("Agent task requires message or query")
    response=SovereignAgent().handle(message,report_filename=payload.get("report_filename"))
    return response.to_dict()

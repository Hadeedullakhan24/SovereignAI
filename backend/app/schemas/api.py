from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, EmailStr, Field
class RegisterIn(BaseModel): email: EmailStr; password: str = Field(min_length=12,max_length=256)
class LoginIn(RegisterIn): pass
class RefreshIn(BaseModel): refresh_token: str
class TokenOut(BaseModel): access_token: str; refresh_token: str; token_type: str="bearer"; expires_in: int
class ChatIn(BaseModel): message: str=Field(min_length=1,max_length=20000); conversation_id: str|None=None; agent: str|None=None; model: str|None=None; file_ids: list[str]=Field(default_factory=list,max_length=20); stream: bool=False; task_id: str|None=None
class ChatError(BaseModel):
    code: str
    message: str
class ChatOut(BaseModel):
    """Stable REST/SSE chat response. Output fields remain optional."""
    conversation_id: str
    message_id: str
    content: str
    status: str
    is_verified: bool
    insufficient_evidence: bool = False
    citations: list[Any] = Field(default_factory=list)
    artifact: dict[str, Any] | None = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    error: ChatError | None = None
class TaskIn(BaseModel): type: str=Field(min_length=1,max_length=80); payload: dict[str,Any]=Field(default_factory=dict)
class TaskOut(BaseModel): id: str; type: str; state: str; result: dict|None=None; error: str|None=None; attempts: int

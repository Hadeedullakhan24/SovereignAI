from typing import Any
from pydantic import BaseModel, EmailStr, Field
class RegisterRequest(BaseModel): email: EmailStr; password: str=Field(min_length=12,max_length=256); name: str=Field(min_length=1,max_length=120)
class LoginRequest(BaseModel): email: EmailStr; password: str=Field(min_length=1,max_length=256)
class ChatCreate(BaseModel): title: str=Field(default="New chat",min_length=1,max_length=255)
class MessageCreate(BaseModel): content: str=Field(min_length=1,max_length=20_000)
class TaskCreate(BaseModel): task_type: str=Field(min_length=1,max_length=100); input: dict[str,Any]=Field(default_factory=dict)

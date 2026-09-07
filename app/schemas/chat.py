from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RouteName(StrEnum):
    rag = "rag"
    tool = "tool"
    plan = "plan"
    chat = "chat"
    fallback = "fallback"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, description="Existing session id")
    user_id: str = Field(default="local-user", min_length=1, max_length=128)


class RouteDecision(BaseModel):
    route: RouteName
    confidence: float = Field(ge=0, le=1)
    reason: str


class SourceChunk(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    route: RouteDecision
    sources: list[SourceChunk] = Field(default_factory=list)
    tool_result: dict[str, Any] | None = None

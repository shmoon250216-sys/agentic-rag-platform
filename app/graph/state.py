from typing import Any, TypedDict

from app.schemas.chat import RouteDecision, SourceChunk


class AgentState(TypedDict, total=False):
    user_id: str
    session_id: str
    message: str
    memories: list[str]
    route: RouteDecision
    sources: list[SourceChunk]
    tool_result: dict[str, Any] | None
    answer: str
    error: str | None

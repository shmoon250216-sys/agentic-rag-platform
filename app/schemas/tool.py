from typing import Any

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolInvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    tool: str
    ok: bool
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None


class ToolListResponse(BaseModel):
    tools: list[ToolDefinition]

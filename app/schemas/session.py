from datetime import datetime

from pydantic import BaseModel, Field
from app.schemas.chat import SourceChunk


class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class MessageItem(BaseModel):
    message_id: int
    session_id: str
    role: str
    content: str
    created_at: datetime
    sources: list[SourceChunk] = Field(default_factory=list)
    citation_warnings: list[str] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class MessageListResponse(BaseModel):
    messages: list[MessageItem]

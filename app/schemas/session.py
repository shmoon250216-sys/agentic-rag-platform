from datetime import datetime

from pydantic import BaseModel


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


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class MessageListResponse(BaseModel):
    messages: list[MessageItem]

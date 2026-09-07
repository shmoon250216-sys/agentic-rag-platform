from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    profile = "profile"
    preference = "preference"
    goal = "goal"
    project_fact = "project_fact"
    decision = "decision"


class MemoryItem(BaseModel):
    memory_id: int
    user_id: str
    memory_type: MemoryType
    content: str
    importance: float = Field(ge=0, le=1)
    created_at: str
    updated_at: str


class MemoryCreate(BaseModel):
    memory_type: MemoryType = MemoryType.preference
    content: str = Field(min_length=4, max_length=1000)
    importance: float = Field(default=0.7, ge=0, le=1)


class MemoryListResponse(BaseModel):
    memories: list[MemoryItem]


class MemoryCreateResponse(BaseModel):
    memory: MemoryItem

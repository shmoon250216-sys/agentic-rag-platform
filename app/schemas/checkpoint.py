from datetime import datetime
from typing import Any

from pydantic import BaseModel


class GraphCheckpointSummary(BaseModel):
    checkpoint_id: int
    session_id: str
    route: str
    created_at: datetime


class GraphCheckpointDetail(GraphCheckpointSummary):
    state: dict[str, Any]


class GraphCheckpointListResponse(BaseModel):
    checkpoints: list[GraphCheckpointSummary]


class GraphCheckpointDetailResponse(BaseModel):
    checkpoint: GraphCheckpointDetail

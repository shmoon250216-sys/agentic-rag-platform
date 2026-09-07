from pydantic import BaseModel, Field

from app.schemas.chat import RouteName


class EvaluationCase(BaseModel):
    case_id: str
    category: str
    message: str
    expected_route: RouteName
    expected_source_title: str | None = None
    expected_tool: str | None = None


class EvaluationCaseResult(BaseModel):
    case_id: str
    category: str
    expected_route: RouteName
    actual_route: RouteName
    route_correct: bool
    rag_hit: bool | None = None
    tool_correct: bool | None = None
    latency_ms: float = Field(ge=0)


class EvaluationReport(BaseModel):
    total_cases: int
    route_accuracy: float
    rag_hit_rate: float | None
    tool_accuracy: float | None
    average_latency_ms: float
    results: list[EvaluationCaseResult]

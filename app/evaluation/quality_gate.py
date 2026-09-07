from pydantic import BaseModel, Field

from app.core.config import get_settings

from .models import EvaluationReport


class QualityGateThresholds(BaseModel):
    min_total_cases: int = Field(ge=1)
    min_route_accuracy: float = Field(ge=0, le=1)
    min_rag_hit_rate: float = Field(ge=0, le=1)
    min_tool_accuracy: float = Field(ge=0, le=1)
    max_average_latency_ms: float = Field(gt=0)


class QualityGateCheck(BaseModel):
    metric: str
    actual: float | int | None
    expected: float | int
    passed: bool


class QualityGateReport(BaseModel):
    passed: bool
    thresholds: QualityGateThresholds
    checks: list[QualityGateCheck]
    evaluation: EvaluationReport
    failed_reasons: list[str]


def default_quality_gate_thresholds() -> QualityGateThresholds:
    settings = get_settings()
    return QualityGateThresholds(
        min_total_cases=settings.quality_min_total_cases,
        min_route_accuracy=settings.quality_min_route_accuracy,
        min_rag_hit_rate=settings.quality_min_rag_hit_rate,
        min_tool_accuracy=settings.quality_min_tool_accuracy,
        max_average_latency_ms=settings.quality_max_average_latency_ms,
    )


def evaluate_quality_gate(
    report: EvaluationReport,
    thresholds: QualityGateThresholds | None = None,
) -> QualityGateReport:
    gate = thresholds or default_quality_gate_thresholds()
    checks = [
        QualityGateCheck(
            metric="total_cases",
            actual=report.total_cases,
            expected=gate.min_total_cases,
            passed=report.total_cases >= gate.min_total_cases,
        ),
        QualityGateCheck(
            metric="route_accuracy",
            actual=report.route_accuracy,
            expected=gate.min_route_accuracy,
            passed=report.route_accuracy >= gate.min_route_accuracy,
        ),
        QualityGateCheck(
            metric="rag_hit_rate",
            actual=report.rag_hit_rate,
            expected=gate.min_rag_hit_rate,
            passed=(report.rag_hit_rate or 0) >= gate.min_rag_hit_rate,
        ),
        QualityGateCheck(
            metric="tool_accuracy",
            actual=report.tool_accuracy,
            expected=gate.min_tool_accuracy,
            passed=(report.tool_accuracy or 0) >= gate.min_tool_accuracy,
        ),
        QualityGateCheck(
            metric="average_latency_ms",
            actual=report.average_latency_ms,
            expected=gate.max_average_latency_ms,
            passed=report.average_latency_ms <= gate.max_average_latency_ms,
        ),
    ]
    failed_reasons = [
        f"{check.metric}: actual={check.actual}, expected={check.expected}"
        for check in checks
        if not check.passed
    ]
    return QualityGateReport(
        passed=not failed_reasons,
        thresholds=gate,
        checks=checks,
        evaluation=report,
        failed_reasons=failed_reasons,
    )

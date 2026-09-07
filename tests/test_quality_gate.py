from app.evaluation.models import EvaluationCaseResult, EvaluationReport
from app.evaluation.quality_gate import QualityGateThresholds, evaluate_quality_gate
from app.schemas.chat import RouteName


def test_quality_gate_passes_when_metrics_meet_thresholds() -> None:
    report = EvaluationReport(
        total_cases=12,
        route_accuracy=1.0,
        rag_hit_rate=1.0,
        tool_accuracy=1.0,
        average_latency_ms=10,
        results=[],
    )
    thresholds = QualityGateThresholds(
        min_total_cases=12,
        min_route_accuracy=0.95,
        min_rag_hit_rate=0.9,
        min_tool_accuracy=0.95,
        max_average_latency_ms=500,
    )

    gate = evaluate_quality_gate(report, thresholds)

    assert gate.passed is True
    assert gate.failed_reasons == []


def test_quality_gate_fails_with_actionable_reasons() -> None:
    report = EvaluationReport(
        total_cases=2,
        route_accuracy=0.5,
        rag_hit_rate=0.5,
        tool_accuracy=1.0,
        average_latency_ms=900,
        results=[
            EvaluationCaseResult(
                case_id="bad",
                category="rag",
                expected_route=RouteName.rag,
                actual_route=RouteName.chat,
                route_correct=False,
                rag_hit=False,
                latency_ms=900,
            )
        ],
    )
    thresholds = QualityGateThresholds(
        min_total_cases=12,
        min_route_accuracy=0.95,
        min_rag_hit_rate=0.9,
        min_tool_accuracy=0.95,
        max_average_latency_ms=500,
    )

    gate = evaluate_quality_gate(report, thresholds)

    assert gate.passed is False
    assert any("route_accuracy" in reason for reason in gate.failed_reasons)

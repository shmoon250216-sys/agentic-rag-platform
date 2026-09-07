import pytest

from app.evaluation.dataset import DEFAULT_EVALUATION_CASES
from app.evaluation.runner import EvaluationRunner


@pytest.mark.asyncio
async def test_evaluation_runner_builds_report() -> None:
    report = await EvaluationRunner().run(DEFAULT_EVALUATION_CASES[:4])

    assert report.total_cases == 4
    assert report.route_accuracy >= 0.75
    assert report.average_latency_ms >= 0
    assert report.results[0].case_id == "rag_architecture"

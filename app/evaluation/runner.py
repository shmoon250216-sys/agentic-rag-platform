import time
from uuid import uuid4

from app.graph.workflow import AgentWorkflow
from app.llm.client import FakeLLMClient
from app.schemas.chat import ChatRequest

from .dataset import DEFAULT_EVALUATION_CASES
from .models import EvaluationCase, EvaluationCaseResult, EvaluationReport


class EvaluationRunner:
    def __init__(self, workflow: AgentWorkflow | None = None) -> None:
        self.workflow = workflow or AgentWorkflow(llm=FakeLLMClient())

    async def run(self, cases: list[EvaluationCase] | None = None) -> EvaluationReport:
        selected_cases = cases or DEFAULT_EVALUATION_CASES
        results = [await self._run_case(case) for case in selected_cases]
        return _build_report(results)

    async def _run_case(self, case: EvaluationCase) -> EvaluationCaseResult:
        started = time.perf_counter()
        response = await self.workflow.run(
            ChatRequest(message=case.message, user_id="evaluation-user"),
            session_id=f"eval-{case.case_id}-{uuid4()}",
        )
        latency_ms = (time.perf_counter() - started) * 1000

        rag_hit = None
        if case.expected_source_title is not None:
            rag_hit = any(source.title == case.expected_source_title for source in response.sources)

        tool_correct = None
        if case.expected_tool is not None:
            tool_correct = bool(
                response.tool_result
                and response.tool_result.get("ok") is True
                and response.tool_result.get("tool") == case.expected_tool
            )

        return EvaluationCaseResult(
            case_id=case.case_id,
            category=case.category,
            expected_route=case.expected_route,
            actual_route=response.route.route,
            route_correct=response.route.route == case.expected_route,
            rag_hit=rag_hit,
            tool_correct=tool_correct,
            latency_ms=round(latency_ms, 2),
        )


def _build_report(results: list[EvaluationCaseResult]) -> EvaluationReport:
    rag_results = [result for result in results if result.rag_hit is not None]
    tool_results = [result for result in results if result.tool_correct is not None]
    average_latency = sum(result.latency_ms for result in results) / max(len(results), 1)

    return EvaluationReport(
        total_cases=len(results),
        route_accuracy=_ratio(result.route_correct for result in results),
        rag_hit_rate=_ratio(result.rag_hit for result in rag_results) if rag_results else None,
        tool_accuracy=_ratio(result.tool_correct for result in tool_results) if tool_results else None,
        average_latency_ms=round(average_latency, 2),
        results=results,
    )


def _ratio(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0
    return round(sum(1 for value in materialized if value) / len(materialized), 4)

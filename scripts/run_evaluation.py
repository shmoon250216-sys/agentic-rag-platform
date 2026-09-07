import asyncio
import json
from pathlib import Path

from app.evaluation.quality_gate import evaluate_quality_gate
from app.evaluation.runner import EvaluationRunner


async def main() -> None:
    report = await EvaluationRunner().run()
    gate = evaluate_quality_gate(report)
    output_path = Path("docs/evaluation-results.json")
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    gate_output_path = Path("docs/quality-gate-results.json")
    gate_output_path.write_text(
        json.dumps(gate.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Evaluation complete")
    print(f"cases: {report.total_cases}")
    print(f"route_accuracy: {report.route_accuracy:.2%}")
    if report.rag_hit_rate is not None:
        print(f"rag_hit_rate: {report.rag_hit_rate:.2%}")
    if report.tool_accuracy is not None:
        print(f"tool_accuracy: {report.tool_accuracy:.2%}")
    print(f"average_latency_ms: {report.average_latency_ms}")
    print(f"saved: {output_path}")
    print(f"quality_gate: {'PASS' if gate.passed else 'FAIL'}")
    print(f"quality_gate_saved: {gate_output_path}")
    if not gate.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

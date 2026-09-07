import asyncio

from app.evaluation.quality_gate import evaluate_quality_gate
from app.evaluation.runner import EvaluationRunner


async def main() -> None:
    report = await EvaluationRunner().run()
    gate = evaluate_quality_gate(report)
    print(gate.model_dump_json(indent=2))
    if not gate.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())

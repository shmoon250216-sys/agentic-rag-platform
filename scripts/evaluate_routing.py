"""Real classifier-only evaluation: never runs tools or retrieval / answer generation."""
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import get_settings  # noqa: E402
from app.graph.semantic_router import SemanticRouter  # noqa: E402
from app.tools.registry import ToolRegistry  # noqa: E402


async def main():
    settings = get_settings()
    if settings.llm_provider != "openai-compatible" or not settings.llm_api_key:
        raise SystemExit("Configure real LLM before this evaluation")
    settings = settings.model_copy(update={"router_mode": "llm"})
    router = SemanticRouter(settings, ToolRegistry())
    dataset = json.loads(Path('examples/routing_benchmark_v1.json').read_text(encoding='utf-8'))
    report = {"dataset": dataset['version'], "model": settings.llm_model,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Synthetic classifier-only regression; no tools executed; no calibrated confidence",
              "timeout_seconds": settings.router_timeout_seconds, "cases": []}
    semaphore = asyncio.Semaphore(3)
    async def evaluate(case):
        async with semaphore:
            started = time.perf_counter()
            result = await router.decide(case)
            correct = (result.route.value == case['expected_route']
                       and result.status == case['expected_status'])
            if case.get('expected_tool'):
                correct &= result.tool_name == case['expected_tool']
            if case.get('expected_arguments'):
                correct &= result.tool_arguments == case['expected_arguments']
            return {**case, 'decision': result.model_dump(mode='json'), 'correct': bool(correct),
                    'latency_ms': round((time.perf_counter()-started)*1000,2)}
    for task in asyncio.as_completed([evaluate(c) for c in dataset['cases']]):
        row = await task
        report['cases'].append(row)
        print(row['id'], row['decision']['route'], row['decision']['status'], row['correct'],flush=True)
    rows = sorted(report['cases'], key=lambda r: r['id'])
    report['cases'] = rows
    report['summary'] = {'total':len(rows),'correct':sum(r['correct'] for r in rows),
                         'accuracy':sum(r['correct'] for r in rows)/len(rows),
                         'status_counts':dict(Counter(r['decision']['status'] for r in rows)),
                         'confusion':dict(Counter(r['expected_route']+'->'+r['decision']['route'] for r in rows))}
    Path('docs/routing-evaluation-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())

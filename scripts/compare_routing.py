"""Re-score the preserved real-model run against the previous rule router.

No model re-run, no cherry-picking; same stored questions, labels and context.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.graph.router import SupervisorRouter  # noqa: E402
from app.schemas.chat import RouteName  # noqa: E402
from app.tools.registry import ToolRegistry  # noqa: E402


def main():
    source = ROOT / 'docs/routing-evaluation-results.json'
    run = json.loads(source.read_text(encoding='utf-8'))
    dataset = json.loads((ROOT / 'examples/routing_benchmark_v1.json').read_text(encoding='utf-8'))
    expected = {c['id']: c for c in dataset['cases']}
    assert len(run['cases']) == len(expected) == len({r['id'] for r in run['cases']})
    rows = []
    rules, tools = SupervisorRouter(), ToolRegistry()
    for row in run['cases']:
        case = expected[row['id']]
        assert all(row.get(k) == v for k, v in case.items())
        decision = rules.decide(case['message'])
        if decision.route == RouteName.chat:
            contextual = rules.decide(case.get('query', case['message']))
            if contextual.route == RouteName.rag:
                decision = contextual
        baseline = decision.model_dump(mode='json')
        if decision.route == RouteName.tool:
            name, args = tools._select_tool(case['message'])
            baseline.update(tool_name=name, tool_arguments=args)
        def correct(result, strict):
            ok = result['route'] == case['expected_route']
            if strict:
                ok &= result['status'] == case['expected_status']
                if case.get('expected_tool'):
                    ok &= result['tool_name'] == case['expected_tool']
                if case.get('expected_arguments'):
                    ok &= result['tool_arguments'] == case['expected_arguments']
            return bool(ok)
        rows.append({'id':case['id'], 'message':case['message'],
                     'expected_route':case['expected_route'], 'baseline':baseline,
                     'semantic':row['decision'],
                     'baseline_route_correct':correct(baseline,False),
                     'semantic_route_correct':correct(row['decision'],False),
                     'baseline_strict_correct':correct(baseline,True),
                     'semantic_strict_correct':correct(row['decision'],True)})
    summary = {k: {'correct':sum(r[k] for r in rows), 'total':len(rows),
                   'accuracy':sum(r[k] for r in rows)/len(rows)}
               for k in ['baseline_route_correct','semantic_route_correct',
                         'baseline_strict_correct','semantic_strict_correct']}
    report = {'scope':'25 synthetic classifier cases; same inputs and labels, not production accuracy. '
                     'Historical real model run preserved; baseline is prior rule policy and tool selector. '
                     'No end-to-end chat or tool execution. Query absent in original cases remains absent.',
              'source_run_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'baseline_router_sha256':hashlib.sha256((ROOT/'app/graph/router.py').read_bytes()).hexdigest(),
              'summary':summary,'cases':rows}
    (ROOT/'docs/routing-comparison-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

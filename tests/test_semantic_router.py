import asyncio
import json

import pytest

from app.core.config import Settings
from app.graph.semantic_router import SemanticRouter
from app.graph.workflow import AgentWorkflow
from app.schemas.chat import ChatRequest, RouteName
from app.tools.registry import ToolRegistry


def settings(**kwargs):
    return Settings(_env_file=None, router_mode="llm", **kwargs)


def proposal(**kwargs):
    return json.dumps({"route": "rag", "reason": "需要制度证据", "needs_clarification": False,
                       **kwargs}, ensure_ascii=False)


async def test_semantic_input_and_no_keyword_override():
    seen = []
    async def model(system, message):
        seen.append(json.loads(message))
        return proposal()
    router = SemanticRouter(settings(), ToolRegistry(), model)
    result = await router.decide({"message": "帮我查一下住酒店的上限", "history": [],
                                  "memories": ["private"], "summary": "private"})
    assert result.route == RouteName.rag
    assert result.confidence is None and result.source == "llm"
    assert "memories" not in seen[0] and "summary" not in seen[0]


@pytest.mark.parametrize("raw", [
    "not json", proposal(route="unknown"), proposal(confidence=.99),
    proposal(needs_clarification="false"), proposal(needs_clarification=True),
    proposal(route="tool", tool_name="delete_all"),
    proposal(route="tool", tool_name="calculator", tool_arguments={"left": 1}),
    proposal(route="tool", tool_name="doc_stats", tool_arguments={"command": "rm"}),
    proposal(tool_name="doc_stats"),
])
async def test_invalid_output_never_executes(raw):
    async def model(*_):
        return raw
    result = await SemanticRouter(settings(), ToolRegistry(), model).decide({"message": "test"})
    assert result.route == RouteName.fallback and result.status == "invalid_output"
    assert result.tool_name is None


async def test_timeout_provider_failure_and_cancellation():
    async def slow(*_):
        await asyncio.sleep(1)
    r = SemanticRouter(settings(router_timeout_seconds=.01), ToolRegistry(), slow)
    assert (await r.decide({"message": "test"})).status == "timeout"
    async def broken(*_):
        raise RuntimeError("secret-provider-body")
    r.classifier = broken
    result = await r.decide({"message": "test"})
    assert result.status == "provider_error" and "secret" not in result.reason
    async def cancelled(*_):
        raise asyncio.CancelledError()
    r.classifier = cancelled
    with pytest.raises(asyncio.CancelledError):
        await r.decide({"message": "test"})


async def test_missing_provider_does_not_silently_use_rules():
    r = SemanticRouter(settings(llm_provider="openai-compatible", llm_api_key=""), ToolRegistry())
    assert (await r.decide({"message": "统计文档"})).status == "configuration_error"


@pytest.mark.parametrize("stream", [False, True])
async def test_workflow_uses_model_selected_tool_and_clarifies(stream):
    workflow = AgentWorkflow()
    async def model(*_):
        return proposal(route="tool", tool_name="calculator",
                        tool_arguments={"left": 12, "operator": "+", "right": 30})
    workflow.router = SemanticRouter(settings(), workflow.tools, model)
    async def forbidden(*_):
        raise AssertionError("Must not repeat legacy keyword selection")
    workflow.tools.run_best_effort = forbidden
    request = ChatRequest(message="十二加三十", user_id="test")
    if stream:
        events = [event async for event in workflow.stream(request, "semantic-stream")]
        result = events[-1][1]
    else:
        result = await workflow.run(request, "semantic-normal")
    assert result.tool_result["result"]["value"] == 42
    async def clarify(*_):
        return proposal(needs_clarification=True, clarification_question="你指的是哪份制度？")
    workflow.router.classifier = clarify
    if stream:
        result = [e async for e in workflow.stream(request, "clarify-stream")][-1][1]
    else:
        result = await workflow.run(request, "clarify-normal")
    assert result.answer == "你指的是哪份制度？" and result.tool_result is None

async def test_classifier_http_contract(monkeypatch):
    import httpx
    from app.llm.client import OpenAICompatibleLLMClient
    observed = []
    def handler(request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": proposal()}}]})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    client = OpenAICompatibleLLMClient(settings(llm_api_key="test-only", llm_model="test-model"))
    assert json.loads(await client.classify_route("policy", "input"))["route"] == "rag"
    assert observed[0]["response_format"] == {"type": "json_object"}
    assert observed[0]["temperature"] == 0
    assert observed[0]["messages"] == [{"role": "system", "content": "policy"},
                                        {"role": "user", "content": "input"}]

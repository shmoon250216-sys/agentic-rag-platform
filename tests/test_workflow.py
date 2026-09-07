import pytest

from app.graph.workflow import AgentWorkflow
from app.schemas.chat import ChatRequest, RouteName


@pytest.mark.asyncio
async def test_workflow_routes_rag_branch() -> None:
    response = await AgentWorkflow().run(
        ChatRequest(message="请介绍一下这个项目的架构"),
        session_id="session-1",
    )

    assert response.route.route == RouteName.rag
    assert response.sources
    assert "知识库片段" in response.answer


@pytest.mark.asyncio
async def test_workflow_routes_tool_branch() -> None:
    response = await AgentWorkflow().run(
        ChatRequest(message="帮我计算 12 + 30"),
        session_id="session-1",
    )

    assert response.route.route == RouteName.tool
    assert response.tool_result is not None
    assert response.tool_result["tool"] == "calculator"
    assert response.tool_result["ok"] is True
    assert response.tool_result["result"]["value"] == 42.0


@pytest.mark.asyncio
async def test_workflow_routes_chat_branch() -> None:
    response = await AgentWorkflow().run(
        ChatRequest(message="你好，今天适合学习什么"),
        session_id="session-1",
    )

    assert response.route.route == RouteName.chat
    assert "chat 分支" in response.answer


@pytest.mark.asyncio
async def test_workflow_routes_plan_branch() -> None:
    response = await AgentWorkflow().run(
        ChatRequest(message="请帮我制定一个把新报销制度接入知识库的方案"),
        session_id="session-1",
    )

    assert response.route.route == RouteName.plan
    assert "复杂任务规划分支" in response.answer

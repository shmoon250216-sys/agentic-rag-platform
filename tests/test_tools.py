from types import SimpleNamespace

import pytest

from app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_registry_lists_mcp_style_tool_definitions() -> None:
    registry = ToolRegistry()

    tools = registry.list_tools()

    assert {tool.name for tool in tools} == {"calculator", "route_planner", "doc_stats"}
    assert tools[0].input_schema["type"] == "object"


@pytest.mark.asyncio
async def test_registry_calculator_call_uses_standard_envelope() -> None:
    result = await ToolRegistry().call_tool(
        "calculator",
        {"left": 12, "operator": "+", "right": 30},
    )

    assert result.ok is True
    assert result.tool == "calculator"
    assert result.result["value"] == 42


@pytest.mark.asyncio
async def test_registry_returns_validation_error() -> None:
    result = await ToolRegistry().call_tool(
        "calculator",
        {"left": 12, "operator": "/", "right": 0},
    )

    assert result.ok is False
    assert "除数不能为 0" in result.error


@pytest.mark.asyncio
async def test_doc_stats_uses_configured_async_knowledge_base(monkeypatch) -> None:
    class AsyncKnowledgeBase:
        async def list_documents(self):
            return [
                SimpleNamespace(title="Redis 文档", chunk_count=3),
                SimpleNamespace(title="业务手册", chunk_count=2),
            ]

    monkeypatch.setattr(
        "app.tools.local_tools.get_knowledge_base",
        lambda: AsyncKnowledgeBase(),
    )

    result = await ToolRegistry().call_tool("doc_stats", {"include_titles": True})

    assert result.ok is True
    assert result.result["document_count"] == 2
    assert result.result["total_chunks"] == 5
    assert result.result["titles"] == ["Redis 文档", "业务手册"]

import re
from typing import Any

from app.schemas.tool import ToolCallResult, ToolDefinition
from app.tools.local_tools import CalculatorTool, DocStatsTool, RoutePlannerTool
from app.tools.protocol import MCPStyleTool


class ToolRegistry:
    """MCP-style local tool registry.

    The registry exposes tool definitions and a standard call envelope. A real
    MCP client/server can replace this class later without changing graph nodes.
    """

    def __init__(self) -> None:
        self._tools: dict[str, MCPStyleTool] = {
            tool.name: tool
            for tool in (
                CalculatorTool(),
                RoutePlannerTool(),
                DocStatsTool(),
            )
        }

    def list_tools(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolCallResult(
                tool=name,
                ok=False,
                arguments=arguments,
                error=f"未知工具：{name}",
            )
        return await tool.call(arguments)

    async def run_best_effort(self, message: str) -> dict:
        name, arguments = self._select_tool(message)
        result = await self.call_tool(name, arguments)
        return result.model_dump()

    def _select_tool(self, message: str) -> tuple[str, dict[str, Any]]:
        expression = _parse_expression(message)
        if expression is not None:
            return "calculator", expression

        if "统计" in message or "多少文档" in message or "多少个文档" in message:
            return "doc_stats", {"include_titles": True}

        if "路线" in message or "规划" in message:
            return "route_planner", _parse_route(message)

        return "doc_stats", {"include_titles": False}


def _parse_expression(message: str) -> dict[str, Any] | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)", message)
    if not match:
        return None
    return {
        "left": float(match.group(1)),
        "operator": match.group(2),
        "right": float(match.group(3)),
    }


def _parse_route(message: str) -> dict[str, Any]:
    match = re.search(r"从(?P<origin>.+?)到(?P<destination>.+?)(?:怎么走|路线|$)", message)
    if match:
        return {
            "origin": match.group("origin").strip(),
            "destination": match.group("destination").strip(),
            "preference": "时间优先",
        }
    return {
        "origin": "当前位置",
        "destination": "目标地点",
        "preference": "时间优先",
    }

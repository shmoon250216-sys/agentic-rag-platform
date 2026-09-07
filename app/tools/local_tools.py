import inspect
from typing import Literal

from pydantic import BaseModel, Field

from app.rag.factory import get_knowledge_base
from app.tools.protocol import MCPStyleTool


class CalculatorInput(BaseModel):
    left: float
    operator: Literal["+", "-", "*", "/"]
    right: float


class RoutePlannerInput(BaseModel):
    origin: str = Field(min_length=1, max_length=120)
    destination: str = Field(min_length=1, max_length=120)
    preference: str = Field(default="时间优先", max_length=80)


class DocStatsInput(BaseModel):
    include_titles: bool = True


class CalculatorTool(MCPStyleTool):
    name = "calculator"
    description = "执行基础四则运算。"
    input_model = CalculatorInput

    async def _execute(self, arguments: CalculatorInput) -> dict:
        if arguments.operator == "+":
            value = arguments.left + arguments.right
        elif arguments.operator == "-":
            value = arguments.left - arguments.right
        elif arguments.operator == "*":
            value = arguments.left * arguments.right
        else:
            if arguments.right == 0:
                raise ValueError("除数不能为 0")
            value = arguments.left / arguments.right

        return {
            "expression": f"{arguments.left} {arguments.operator} {arguments.right}",
            "value": value,
        }


class RoutePlannerTool(MCPStyleTool):
    name = "route_planner"
    description = "根据起点、终点和偏好生成路线规划占位结果。"
    input_model = RoutePlannerInput

    async def _execute(self, arguments: RoutePlannerInput) -> dict:
        return {
            "summary": (
                f"已为你生成从 {arguments.origin} 到 {arguments.destination} 的路线规划占位结果。"
            ),
            "preference": arguments.preference,
            "steps": [
                "确认起点和终点",
                "根据偏好选择路线",
                "后续可接入真实地图 API 返回距离、时间和费用",
            ],
        }


class DocStatsTool(MCPStyleTool):
    name = "doc_stats"
    description = "统计当前知识库文档数量和分片数量。"
    input_model = DocStatsInput

    async def _execute(self, arguments: DocStatsInput) -> dict:
        documents = get_knowledge_base().list_documents()
        if inspect.isawaitable(documents):
            documents = await documents
        payload = {
            "document_count": len(documents),
            "total_chunks": sum(document.chunk_count for document in documents),
        }
        if arguments.include_titles:
            payload["titles"] = [document.title for document in documents]
        return payload

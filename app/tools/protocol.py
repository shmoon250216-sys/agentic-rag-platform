from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from app.schemas.tool import ToolCallResult, ToolDefinition


class MCPStyleTool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
        )

    async def call(self, arguments: dict[str, Any]) -> ToolCallResult:
        try:
            parsed = self.input_model.model_validate(arguments)
            result = await self._execute(parsed)
            return ToolCallResult(
                tool=self.name,
                ok=True,
                arguments=parsed.model_dump(),
                result=result,
            )
        except ValidationError as exc:
            return ToolCallResult(
                tool=self.name,
                ok=False,
                arguments=arguments,
                error=f"参数校验失败：{exc.errors()}",
            )
        except Exception as exc:
            return ToolCallResult(
                tool=self.name,
                ok=False,
                arguments=arguments,
                error=str(exc),
            )

    @abstractmethod
    async def _execute(self, arguments: BaseModel) -> Any:
        raise NotImplementedError

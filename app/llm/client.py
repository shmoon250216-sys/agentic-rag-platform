import json
from collections.abc import AsyncIterator
from typing import Protocol

import httpx

from app.core.cache import TTLCache, make_cache_key
from app.core.config import Settings
from app.schemas.chat import SourceChunk

llm_response_cache = TTLCache()

ANSWER_FORMAT_INSTRUCTIONS = (
    "输出格式要求：使用清晰的 Markdown；优先用中文短段落回答；"
    "涉及步骤、原因、对比、结论时使用列表；代码、命令、配置示例必须使用 fenced code block；"
    "不要把所有内容挤在一个段落里；不要为了排版添加无意义标题。"
)


class LLMClient(Protocol):
    async def chat(self, message: str) -> str:
        raise NotImplementedError

    def stream_chat(self, message: str) -> AsyncIterator[str]:
        raise NotImplementedError

    async def answer_with_context(self, message: str, sources: list[SourceChunk]) -> str:
        raise NotImplementedError

    def stream_answer_with_context(
        self,
        message: str,
        sources: list[SourceChunk],
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    async def answer_with_tool_result(self, message: str, tool_result: dict) -> str:
        raise NotImplementedError

    def stream_answer_with_tool_result(
        self,
        message: str,
        tool_result: dict,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


class FakeLLMClient:
    """Deterministic fake LLM for local development before real model wiring."""

    async def chat(self, message: str) -> str:
        return f"收到：{message}。当前是本地开发版 chat 分支，后续会接入真实 LLM。"

    async def stream_chat(self, message: str) -> AsyncIterator[str]:
        async for chunk in _stream_text(await self.chat(message)):
            yield chunk

    async def answer_with_context(self, message: str, sources: list[SourceChunk]) -> str:
        if not sources:
            return "知识库暂时没有命中内容。后续接入 Redis 向量检索后会返回引用来源。"
        titles = "、".join(source.title for source in sources)
        return f"根据知识库片段（{titles}），这是对「{message}」的占位回答。"

    async def stream_answer_with_context(
        self,
        message: str,
        sources: list[SourceChunk],
    ) -> AsyncIterator[str]:
        async for chunk in _stream_text(await self.answer_with_context(message, sources)):
            yield chunk

    async def answer_with_tool_result(self, message: str, tool_result: dict) -> str:
        return f"工具分支已处理「{message}」。工具结果：{tool_result}"

    async def stream_answer_with_tool_result(
        self,
        message: str,
        tool_result: dict,
    ) -> AsyncIterator[str]:
        async for chunk in _stream_text(await self.answer_with_tool_result(message, tool_result)):
            yield chunk


class OpenAICompatibleLLMClient:
    """LLM adapter for OpenAI-compatible chat-completions APIs."""

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds

    async def chat(self, message: str) -> str:
        return await self._complete(
            system_prompt=(
                "你是一个中文 AI Agent 项目助手。回答要清晰、具体，"
                "如果信息不足，要说明当前系统能力边界。"
                f"{ANSWER_FORMAT_INSTRUCTIONS}"
            ),
            user_prompt=message,
        )

    def stream_chat(self, message: str) -> AsyncIterator[str]:
        return self._stream_complete(
            system_prompt=(
                "你是一个中文 AI Agent 项目助手。回答要清晰、具体，"
                "如果信息不足，要说明当前系统能力边界。"
                f"{ANSWER_FORMAT_INSTRUCTIONS}"
            ),
            user_prompt=message,
        )

    async def answer_with_context(self, message: str, sources: list[SourceChunk]) -> str:
        context = "\n\n".join(
            f"来源标题：{source.title}\n片段：{source.snippet}" for source in sources
        )
        return await self._complete(
            system_prompt=(
                "你是一个 RAG 问答助手。必须优先依据给定知识库片段回答，"
                "不要编造来源中没有的信息。回答末尾简要说明依据了哪些来源标题。"
                f"{ANSWER_FORMAT_INSTRUCTIONS}"
            ),
            user_prompt=f"用户问题：{message}\n\n知识库片段：\n{context or '无命中片段'}",
        )

    def stream_answer_with_context(
        self,
        message: str,
        sources: list[SourceChunk],
    ) -> AsyncIterator[str]:
        context = "\n\n".join(
            f"来源标题：{source.title}\n片段：{source.snippet}" for source in sources
        )
        return self._stream_complete(
            system_prompt=(
                "你是一个 RAG 问答助手。必须优先依据给定知识库片段回答，"
                "不要编造来源中没有的信息。回答末尾简要说明依据了哪些来源标题。"
                f"{ANSWER_FORMAT_INSTRUCTIONS}"
            ),
            user_prompt=f"用户问题：{message}\n\n知识库片段：\n{context or '无命中片段'}",
        )

    async def answer_with_tool_result(self, message: str, tool_result: dict) -> str:
        return await self._complete(
            system_prompt=(
                "你是一个工具调用结果解释助手。根据工具返回的结构化结果，"
                "用中文给出简洁结论；如果工具失败，要说明失败原因。"
                f"{ANSWER_FORMAT_INSTRUCTIONS}"
            ),
            user_prompt=f"用户问题：{message}\n\n工具结果：{tool_result}",
        )

    def stream_answer_with_tool_result(
        self,
        message: str,
        tool_result: dict,
    ) -> AsyncIterator[str]:
        return self._stream_complete(
            system_prompt=(
                "你是一个工具调用结果解释助手。根据工具返回的结构化结果，"
                "用中文给出简洁结论；如果工具失败，要说明失败原因。"
                f"{ANSWER_FORMAT_INSTRUCTIONS}"
            ),
            user_prompt=f"用户问题：{message}\n\n工具结果：{tool_result}",
        )

    async def _complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is required for OpenAI-compatible provider")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def _stream_complete(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is required for OpenAI-compatible provider")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = _parse_openai_stream_line(line)
                    if chunk:
                        yield chunk


class CachedLLMClient:
    """Cache deterministic LLM calls by prompt, context, and tool result."""

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.enabled = settings.cache_enabled
        llm_response_cache.max_entries = settings.cache_max_entries
        llm_response_cache.ttl_seconds = settings.cache_ttl_seconds

    async def chat(self, message: str) -> str:
        return await self._cached(
            operation="chat",
            payload={"message": message},
            call=lambda: self.client.chat(message),
        )

    async def stream_chat(self, message: str) -> AsyncIterator[str]:
        async for chunk in self._cached_stream(
            operation="chat",
            payload={"message": message},
            stream_call=lambda: self.client.stream_chat(message),
        ):
            yield chunk

    async def answer_with_context(self, message: str, sources: list[SourceChunk]) -> str:
        return await self._cached(
            operation="rag-answer",
            payload={
                "message": message,
                "sources": [source.model_dump(mode="json") for source in sources],
            },
            call=lambda: self.client.answer_with_context(message, sources),
        )

    async def stream_answer_with_context(
        self,
        message: str,
        sources: list[SourceChunk],
    ) -> AsyncIterator[str]:
        async for chunk in self._cached_stream(
            operation="rag-answer",
            payload={
                "message": message,
                "sources": [source.model_dump(mode="json") for source in sources],
            },
            stream_call=lambda: self.client.stream_answer_with_context(message, sources),
        ):
            yield chunk

    async def answer_with_tool_result(self, message: str, tool_result: dict) -> str:
        return await self._cached(
            operation="tool-answer",
            payload={
                "message": message,
                "tool_result": tool_result,
            },
            call=lambda: self.client.answer_with_tool_result(message, tool_result),
        )

    async def stream_answer_with_tool_result(
        self,
        message: str,
        tool_result: dict,
    ) -> AsyncIterator[str]:
        async for chunk in self._cached_stream(
            operation="tool-answer",
            payload={
                "message": message,
                "tool_result": tool_result,
            },
            stream_call=lambda: self.client.stream_answer_with_tool_result(message, tool_result),
        ):
            yield chunk

    async def _cached(self, operation: str, payload: dict, call) -> str:
        if not self.enabled:
            return await call()

        cache_key = make_cache_key(operation, payload)
        cached = llm_response_cache.get(cache_key)
        if cached is not None:
            return cached

        answer = await call()
        llm_response_cache.set(cache_key, answer)
        return answer

    async def _cached_stream(self, operation: str, payload: dict, stream_call) -> AsyncIterator[str]:
        if not self.enabled:
            async for chunk in stream_call():
                yield chunk
            return

        cache_key = make_cache_key(operation, payload)
        cached = llm_response_cache.get(cache_key)
        if cached is not None:
            async for chunk in _stream_text(cached):
                yield chunk
            return

        parts: list[str] = []
        async for chunk in stream_call():
            parts.append(chunk)
            yield chunk
        llm_response_cache.set(cache_key, "".join(parts))


def get_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai-compatible" and settings.llm_api_key:
        return CachedLLMClient(OpenAICompatibleLLMClient(settings), settings)
    return CachedLLMClient(FakeLLMClient(), settings)


async def _stream_text(text: str, chunk_size: int = 12) -> AsyncIterator[str]:
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def _parse_openai_stream_line(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    payload = line.removeprefix("data:").strip()
    if not payload or payload == "[DONE]":
        return None
    data = json.loads(payload)
    return data["choices"][0].get("delta", {}).get("content")

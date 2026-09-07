from app.core.config import Settings
from app.llm.client import (
    CachedLLMClient,
    FakeLLMClient,
    OpenAICompatibleLLMClient,
    get_llm_client,
    llm_response_cache,
)
from app.schemas.chat import SourceChunk


def test_llm_factory_uses_fake_by_default() -> None:
    client = get_llm_client(Settings(llm_provider="fake"))

    assert isinstance(client, CachedLLMClient)
    assert isinstance(client.client, FakeLLMClient)


def test_llm_factory_uses_openai_compatible_when_configured() -> None:
    client = get_llm_client(
        Settings(
            llm_provider="openai-compatible",
            llm_api_key="test-key",
            llm_base_url="https://example.com/v1",
            llm_model="test-model",
        )
    )

    assert isinstance(client, CachedLLMClient)
    assert isinstance(client.client, OpenAICompatibleLLMClient)
    assert client.client.base_url == "https://example.com/v1"
    assert client.client.model == "test-model"


async def test_fake_llm_answers_with_context() -> None:
    answer = await FakeLLMClient().answer_with_context(
        "项目架构是什么",
        [
            SourceChunk(
                doc_id="doc-1",
                title="项目架构说明",
                snippet="系统使用 LangGraph。",
                score=0.9,
            )
        ],
    )

    assert "项目架构说明" in answer


async def test_cached_llm_client_reuses_same_prompt() -> None:
    llm_response_cache.clear()
    client = get_llm_client(Settings(llm_provider="fake", cache_enabled=True))

    first = await client.chat("你好")
    second = await client.chat("你好")
    stats = llm_response_cache.stats()

    assert first == second
    assert stats["sets"] == 1
    assert stats["hits"] == 1


async def test_cached_llm_client_streams_and_caches_answer() -> None:
    llm_response_cache.clear()
    client = get_llm_client(Settings(llm_provider="fake", cache_enabled=True))

    first = "".join([chunk async for chunk in client.stream_chat("你好")])
    second = "".join([chunk async for chunk in client.stream_chat("你好")])
    stats = llm_response_cache.stats()

    assert first == second
    assert stats["sets"] == 1
    assert stats["hits"] == 1

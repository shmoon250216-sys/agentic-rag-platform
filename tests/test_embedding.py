import math

import httpx
import pytest

from app.core.config import Settings
from app.rag.embedding import (
    HashEmbeddingModel,
    OpenAICompatibleEmbeddingModel,
    cosine_similarity,
    get_embedding_model,
)


def test_hash_embedding_is_normalized() -> None:
    vector = HashEmbeddingModel(dimensions=32).embed("LangGraph Supervisor RAG")

    norm = math.sqrt(sum(value * value for value in vector))

    assert norm == 1.0


def test_cosine_similarity_prefers_related_text() -> None:
    model = HashEmbeddingModel(dimensions=64)
    query = model.embed("Redis RAG 向量检索")
    related = model.embed("Redis Stack 保存向量索引，支持 RAG 检索")
    unrelated = model.embed("今天午饭吃什么")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_embedding_factory_uses_hash_provider() -> None:
    model = get_embedding_model(Settings(embedding_provider="hash", embedding_dimensions=16))

    assert isinstance(model, HashEmbeddingModel)
    assert len(model.embed("hello")) == 16


def test_openai_compatible_embedding_model_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "embedding": [3.0, 4.0],
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    model = OpenAICompatibleEmbeddingModel(
        api_key="test-key",
        base_url="https://embedding.example/v1",
        model="embedding-model",
        dimensions=2,
        client_factory=lambda timeout: httpx.Client(transport=transport, timeout=timeout),
    )

    vector = model.embed("测试 embedding")

    assert vector == [0.6, 0.8]


def test_openai_compatible_embedding_rejects_unexpected_dimensions() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})
    )
    model = OpenAICompatibleEmbeddingModel(
        api_key="test-key",
        base_url="https://embedding.example/v1",
        model="embedding-model",
        dimensions=3,
        client_factory=lambda timeout: httpx.Client(transport=transport, timeout=timeout),
    )

    with pytest.raises(RuntimeError, match="unexpected vector dimension"):
        model.embed("test")

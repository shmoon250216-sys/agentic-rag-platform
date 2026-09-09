from types import SimpleNamespace

import pytest
from redis.exceptions import ConnectionError

from app.rag.embedding import HashEmbeddingModel
from app.rag.redis_store import (
    RedisBackendUnavailable,
    RedisIndexConfigurationError,
    RedisKnowledgeBase,
)


class UnavailableRedis:
    async def ping(self) -> None:
        raise ConnectionError("connection refused")


class SearchClient:
    async def search(self, query, *_args, **_kwargs):
        if "KNN" in query.query_string():
            return SimpleNamespace(
                docs=[
                    SimpleNamespace(
                        doc_id="doc-1".encode(),
                        chunk_id="doc-1:0".encode(),
                        title="内部制度".encode(),
                        content="报销申请需要主管审批".encode(),
                        vector_distance=b"0.1",
                    )
                ]
            )
        return SimpleNamespace(
            docs=[
                SimpleNamespace(
                    doc_id="doc-1".encode(),
                    chunk_id="doc-1:0".encode(),
                    title="内部制度".encode(),
                    content="报销申请需要主管审批".encode(),
                    score=b"2.4",
                )
            ]
        )


class SearchRedis:
    def ft(self, _index_name: str) -> SearchClient:
        return SearchClient()


class ExistingIndexClient:
    async def info(self) -> dict:
        return {
            b"attributes": [
                [
                    b"identifier",
                    b"embedding",
                    b"attribute",
                    b"embedding",
                    b"type",
                    b"VECTOR",
                    b"dim",
                    128,
                ]
            ]
        }


class ExistingIndexRedis:
    def ft(self, _index_name: str) -> ExistingIndexClient:
        return ExistingIndexClient()


def build_store(dimensions: int = 16) -> RedisKnowledgeBase:
    return RedisKnowledgeBase(
        redis_url="redis://localhost:6379/15",
        index_name="idx:test-rag",
        vector_dimensions=dimensions,
        embedding_model=HashEmbeddingModel(dimensions=dimensions),
    )


@pytest.mark.asyncio
async def test_redis_health_reports_disconnected_without_leaking_details() -> None:
    store = build_store()
    store.redis = UnavailableRedis()

    status = await store.health()

    assert status == {"backend": "redis", "connected": False, "index": "idx:test-rag"}
    with pytest.raises(RedisBackendUnavailable) as exc_info:
        await store.ensure_available()
    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "RAG_BACKEND_UNAVAILABLE"


@pytest.mark.asyncio
async def test_redis_search_decodes_utf8_fields() -> None:
    store = build_store()
    store.redis = SearchRedis()
    store._ready = True

    results = await store.search("报销审批", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "doc-1"
    assert results[0].title == "内部制度"
    assert results[0].snippet == "报销申请需要主管审批"


@pytest.mark.asyncio
async def test_redis_rejects_an_existing_index_with_wrong_dimensions() -> None:
    store = build_store(dimensions=256)
    store.redis = ExistingIndexRedis()

    with pytest.raises(RedisIndexConfigurationError) as exc_info:
        await store._ensure_index()

    assert exc_info.value.code == "RAG_INDEX_DIMENSION_MISMATCH"
    assert exc_info.value.status_code == 503

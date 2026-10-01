import pytest

from app.core.config import get_settings
from app.rag.factory import get_knowledge_base
from app.rag.redis_store import RedisKnowledgeBase
from app.rag.store import InMemoryKnowledgeBase
from app.rag.milvus_store import MilvusKnowledgeBase


@pytest.fixture(autouse=True)
def clear_rag_factory_caches():
    get_settings.cache_clear()
    get_knowledge_base.cache_clear()
    yield
    get_settings.cache_clear()
    get_knowledge_base.cache_clear()


def test_rag_factory_uses_memory_by_default(monkeypatch) -> None:
    monkeypatch.setenv("RAG_BACKEND", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "32")
    get_settings.cache_clear()
    get_knowledge_base.cache_clear()

    store = get_knowledge_base()

    assert isinstance(store, InMemoryKnowledgeBase)
    assert store.embedding_model.dimensions == 32


def test_rag_factory_can_select_redis(monkeypatch) -> None:
    monkeypatch.setenv("RAG_BACKEND", "redis")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "128")
    monkeypatch.setenv("RAG_VECTOR_DIMENSIONS", "128")

    store = get_knowledge_base()

    assert isinstance(store, RedisKnowledgeBase)
    assert store.vector_dimensions == 128


def test_rag_factory_rejects_vector_dimension_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("RAG_BACKEND", "redis")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "256")
    monkeypatch.setenv("RAG_VECTOR_DIMENSIONS", "128")

    with pytest.raises(ValueError, match="RAG_VECTOR_DIMENSIONS must match"):
        get_knowledge_base()


def test_rag_factory_selects_milvus_without_connecting(monkeypatch) -> None:
    monkeypatch.setenv("RAG_BACKEND", "milvus")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "32")
    monkeypatch.setenv("RAG_VECTOR_DIMENSIONS", "32")
    backend = get_knowledge_base()
    assert isinstance(backend, MilvusKnowledgeBase)
    assert backend._client is None


def test_unknown_backend_does_not_silently_use_memory(monkeypatch) -> None:
    monkeypatch.setenv("RAG_BACKEND", "milvuss")
    with pytest.raises(ValueError, match="RAG_BACKEND"):
        get_knowledge_base()

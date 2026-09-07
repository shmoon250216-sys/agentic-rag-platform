from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

from app.main import create_app
from app.rag.embedding import HashEmbeddingModel
from app.rag.redis_store import RedisKnowledgeBase


def test_system_info_exposes_runtime_configuration_without_secrets() -> None:
    client = TestClient(create_app())
    response = client.get(
        "/api/v1/system/info",
        headers={"Authorization": "Bearer dev-token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm"]["provider"]
    assert payload["rag"]["backend"]
    assert payload["memory"]["session_store"] == "sqlite"
    serialized = response.text.lower()
    assert "api_key" not in serialized
    assert "api-token" not in serialized
    assert "dev-token" not in serialized


def test_readiness_reports_memory_backend() -> None:
    response = TestClient(create_app()).get("/health/ready")

    assert response.status_code == 200
    assert response.json()["rag"] == {
        "backend": "memory",
        "connected": True,
        "persistent": False,
    }


def test_readiness_returns_503_when_redis_is_unavailable(monkeypatch) -> None:
    class UnavailableRedis:
        async def ping(self) -> None:
            raise ConnectionError("connection refused")

    store = RedisKnowledgeBase(
        redis_url="redis://localhost:6379/15",
        index_name="idx:test-rag",
        vector_dimensions=8,
        embedding_model=HashEmbeddingModel(dimensions=8),
    )
    store.redis = UnavailableRedis()
    monkeypatch.setattr("app.api.routes.get_knowledge_base", lambda: store)

    response = TestClient(create_app(), raise_server_exceptions=False).get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "RAG_BACKEND_UNAVAILABLE",
            "message": "Redis RAG 知识库暂时不可用，请检查 Redis Stack 服务",
        }
    }

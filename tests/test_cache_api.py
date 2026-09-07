from fastapi.testclient import TestClient

from app.llm.client import llm_response_cache
from app.main import create_app
from app.rag.retriever import retrieval_cache


def test_cache_api_reports_and_clears_cache() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}
    retrieval_cache.set("search", ["result"])
    llm_response_cache.set("answer", "cached")

    stats_response = client.get("/api/v1/cache/stats", headers=headers)
    clear_response = client.delete("/api/v1/cache", headers=headers)
    stats_after_clear = client.get("/api/v1/cache/stats", headers=headers)

    assert stats_response.status_code == 200
    assert stats_response.json()["retrieval"]["size"] == 1
    assert stats_response.json()["llm_response"]["size"] == 1
    assert clear_response.status_code == 200
    assert clear_response.json()["cleared"] is True
    assert stats_after_clear.json()["retrieval"]["size"] == 0
    assert stats_after_clear.json()["llm_response"]["size"] == 0

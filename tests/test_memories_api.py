from fastapi.testclient import TestClient

from app.main import create_app


def test_memories_api_lists_extracted_memories() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}

    chat_response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "message": "我希望以后用中文详细解释，我的目标是把报销制度接入知识库",
            "user_id": "memory-api-user",
        },
    )
    memories_response = client.get("/api/v1/users/memory-api-user/memories", headers=headers)

    assert chat_response.status_code == 200
    assert memories_response.status_code == 200
    contents = [memory["content"] for memory in memories_response.json()["memories"]]
    assert any("用户偏好" in content for content in contents)
    assert any("用户长期目标" in content for content in contents)


def test_memories_api_creates_and_deletes_manual_memory() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}

    create_response = client.post(
        "/api/v1/users/manual-memory-user/memories",
        headers=headers,
        json={
            "memory_type": "preference",
            "content": "用户偏好：回答时先给结论，再解释原因。",
            "importance": 0.75,
        },
    )
    memory = create_response.json()["memory"]
    list_response = client.get("/api/v1/users/manual-memory-user/memories", headers=headers)
    delete_response = client.delete(
        f"/api/v1/users/manual-memory-user/memories/{memory['memory_id']}",
        headers=headers,
    )
    list_after_delete = client.get("/api/v1/users/manual-memory-user/memories", headers=headers)

    assert create_response.status_code == 200
    assert memory["memory_type"] == "preference"
    assert list_response.status_code == 200
    assert any(
        item["memory_id"] == memory["memory_id"]
        for item in list_response.json()["memories"]
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert all(
        item["memory_id"] != memory["memory_id"]
        for item in list_after_delete.json()["memories"]
    )


def test_memories_api_rejects_sensitive_manual_memory() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}

    response = client.post(
        "/api/v1/users/manual-memory-user/memories",
        headers=headers,
        json={
            "memory_type": "preference",
            "content": "用户的 api key 是 sk-sensitive1234567890",
            "importance": 0.75,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MEMORY_REJECTED"

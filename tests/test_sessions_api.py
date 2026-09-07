from fastapi.testclient import TestClient

from app.main import create_app


def test_sessions_api_lists_chat_history() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}

    chat_response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={"message": "你好，今天适合学习什么", "user_id": "history-test-user"},
    )
    session_id = chat_response.json()["session_id"]

    sessions_response = client.get(
        "/api/v1/sessions?user_id=history-test-user",
        headers=headers,
    )
    messages_response = client.get(
        f"/api/v1/sessions/{session_id}/messages",
        headers=headers,
    )

    assert sessions_response.status_code == 200
    assert sessions_response.json()["sessions"][0]["session_id"] == session_id
    assert sessions_response.json()["sessions"][0]["title"] == "今天适合学习什么"
    assert messages_response.status_code == 200
    assert [message["role"] for message in messages_response.json()["messages"]] == [
        "user",
        "assistant",
    ]

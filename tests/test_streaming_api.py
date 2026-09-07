import json
import re

from fastapi.testclient import TestClient

from app.main import create_app


def test_chat_stream_emits_sse_events() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        headers=headers,
        json={"message": "请介绍一下这个项目的架构", "user_id": "stream-user"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: start" in body
    assert "event: route" in body
    assert "event: token" in body
    assert "event: done" in body
    assert '"route":"rag"' in body or '"route": "rag"' in body


def test_chat_stream_persists_messages_and_checkpoint() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}

    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        headers=headers,
        json={"message": "帮我计算 12 + 30", "user_id": "stream-persist-user"},
    ) as response:
        body = "".join(response.iter_text())

    done_payload = _sse_payload(body, "done")
    session_id = done_payload["session_id"]
    messages_response = client.get(
        f"/api/v1/sessions/{session_id}/messages",
        headers=headers,
    )
    checkpoints_response = client.get(
        f"/api/v1/sessions/{session_id}/checkpoints",
        headers=headers,
    )

    assert response.status_code == 200
    assert messages_response.status_code == 200
    assert [message["role"] for message in messages_response.json()["messages"]] == [
        "user",
        "assistant",
    ]
    assert checkpoints_response.status_code == 200
    assert checkpoints_response.json()["checkpoints"][0]["route"] == "tool"


def _sse_payload(body: str, event_name: str) -> dict:
    pattern = rf"event: {event_name}\r?\ndata: (.+)"
    match = re.search(pattern, body)
    assert match is not None
    return json.loads(match.group(1))

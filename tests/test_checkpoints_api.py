from fastapi.testclient import TestClient

from app.main import create_app


def test_checkpoints_api_returns_graph_snapshot() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}

    chat_response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={"message": "请介绍一下这个项目的架构", "user_id": "checkpoint-user"},
    )
    session_id = chat_response.json()["session_id"]

    checkpoints_response = client.get(
        f"/api/v1/sessions/{session_id}/checkpoints",
        headers=headers,
    )
    checkpoint_id = checkpoints_response.json()["checkpoints"][0]["checkpoint_id"]
    detail_response = client.get(
        f"/api/v1/checkpoints/{checkpoint_id}",
        headers=headers,
    )

    assert checkpoints_response.status_code == 200
    assert checkpoints_response.json()["checkpoints"][0]["route"] == "rag"
    assert detail_response.status_code == 200
    assert detail_response.json()["checkpoint"]["state"]["route"]["route"] == "rag"

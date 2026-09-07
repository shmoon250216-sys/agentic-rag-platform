from fastapi.testclient import TestClient

from app.main import create_app


def test_tools_api_lists_and_invokes_tools() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}

    list_response = client.get("/api/v1/tools", headers=headers)
    invoke_response = client.post(
        "/api/v1/tools/calculator/invoke",
        headers=headers,
        json={"arguments": {"left": 6, "operator": "*", "right": 7}},
    )

    assert list_response.status_code == 200
    assert "calculator" in {tool["name"] for tool in list_response.json()["tools"]}
    assert invoke_response.status_code == 200
    assert invoke_response.json()["result"]["value"] == 42

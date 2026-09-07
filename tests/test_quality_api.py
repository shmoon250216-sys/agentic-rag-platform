from fastapi.testclient import TestClient

from app.main import create_app


def test_quality_gate_api_returns_passed_report() -> None:
    client = TestClient(create_app())

    response = client.get(
        "/api/v1/quality/gate",
        headers={"Authorization": "Bearer dev-token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["passed"] is True
    assert payload["evaluation"]["total_cases"] >= 12

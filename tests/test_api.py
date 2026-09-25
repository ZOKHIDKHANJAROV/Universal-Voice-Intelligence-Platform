from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_scenarios() -> None:
    response = client.get("/api/v1/scenarios")
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_resolve_scenario() -> None:
    response = client.post(
        "/api/v1/scenarios/resolve",
        json={"text": "Internetim ishlamayapti"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["matched"] is True
    assert payload["scenario_id"] == "internet_problem"


def test_get_unknown_scenario() -> None:
    response = client.get("/api/v1/scenarios/unknown")
    assert response.status_code == 404

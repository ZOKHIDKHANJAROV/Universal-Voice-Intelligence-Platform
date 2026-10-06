from fastapi.testclient import TestClient

from app.core.config import get_settings
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
        json={"text": "Apparat suv chiqmayapti"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["matched"] is True
    assert payload["scenario_id"] == "vending_no_water_uz"


def test_resolve_scenario_prefers_requested_language() -> None:
    response = client.post(
        "/api/v1/scenarios/resolve",
        json={"text": "Аппарат не дает воду", "language": "ru"},
    )
    assert response.json()["scenario_id"] == "vending_no_water_ru"


def test_get_unknown_scenario() -> None:
    response = client.get("/api/v1/scenarios/unknown")
    assert response.status_code == 404


def test_stt_rejects_unsupported_language() -> None:
    response = client.post(
        "/api/v1/stt/transcribe?language=de",
        files={"audio": ("test.wav", b"not-a-real-audio-file", "audio/wav")},
    )
    assert response.status_code == 400


def test_api_key_is_enforced_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "api_key", "secret")
    assert client.get("/api/v1/scenarios").status_code == 401
    assert client.get("/api/v1/scenarios", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/scenarios", headers={"X-API-Key": "secret"}).status_code == 200
    # Health stays open for container probes.
    assert client.get("/health").status_code == 200

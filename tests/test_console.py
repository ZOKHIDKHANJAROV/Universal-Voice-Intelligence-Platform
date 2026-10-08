import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routes.scenarios as scenario_routes
import app.tts.prompt_cache as prompt_cache
import app.voice.analyze as analyze_module
from app.audio.resample import pcm16_to_wav
from app.core.config import get_settings
from app.main import app
from app.services.scenario_service import ScenarioService
from app.speech.models import TranscriptionResult

client = TestClient(app)
DATA = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios.json"


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "tts_prompt_cache_dir", tmp_path)
    prompt_cache.get_prompt_cache.cache_clear()
    yield tmp_path
    prompt_cache.get_prompt_cache.cache_clear()


def _wav(seconds: float = 1.0) -> bytes:
    tone = 0.3 * np.sin(2 * np.pi * 300 * np.arange(int(16000 * seconds)) / 16000)
    return pcm16_to_wav((tone * 32767).astype("<i2").tobytes(), 16000)


def test_console_page_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "UniVoice" in response.text


def test_analyze_runs_the_call_path(prompt_dir, monkeypatch) -> None:
    heard = []

    def fake_stt(audio, rate, language, allowed):
        heard.append((len(audio), rate, allowed))
        return TranscriptionResult(text="Suv chiqmayapti", language="uz",
                                   language_probability=0.8, duration_seconds=1.0)

    monkeypatch.setattr(analyze_module, "transcribe_pcm16", fake_stt)
    monkeypatch.setattr(prompt_cache, "synthesize_bytes", lambda text, lang: _wav(0.2))

    response = client.post(
        "/api/v1/console/analyze",
        files={"audio": ("call.wav", _wav(), "audio/wav")},
        data={"telephone": "true", "language": ""},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # One second of 16 kHz audio reaches STT as 8 kHz phone audio.
    assert heard == [(8000, 8000, ("uz", "ru"))]
    assert body["intent"]["scenario_id"] == "vending_no_water_uz"
    assert body["response_language"] == "uz"
    assert set(body["timings_ms"]) == {"stt", "intent", "audio"}

    audio = client.get(f"/api/v1/console/prompts/audio/{body['response_audio']}")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"


def test_analyze_reports_missing_tts(prompt_dir, monkeypatch) -> None:
    monkeypatch.setattr(analyze_module, "transcribe_pcm16", lambda *a: TranscriptionResult(
        text="", language="ru", language_probability=0.5, duration_seconds=1.0))

    def unavailable(text, lang):
        raise OSError("TTS service is unavailable")

    monkeypatch.setattr(prompt_cache, "synthesize_bytes", unavailable)
    body = client.post(
        "/api/v1/console/analyze", files={"audio": ("a.wav", _wav(), "audio/wav")}
    ).json()
    assert body["intent"]["source"] == "empty"
    assert body["response_audio"] is None
    assert "render-prompts" in body["warning"]


@pytest.mark.parametrize("name", ["../secret.wav", "uz-zzzz.wav", "scenarios.json"])
def test_prompt_audio_only_serves_prompt_files(prompt_dir, name) -> None:
    assert client.get(f"/api/v1/console/prompts/audio/{name}").status_code == 404


def test_prompts_and_system_status(prompt_dir) -> None:
    prompts = client.get("/api/v1/console/prompts").json()
    assert prompts and all(p["rendered"] is False for p in prompts)
    system = client.get("/api/v1/console/system").json()
    assert system["prompts_total"] == len(prompts)
    assert system["prompts_rendered"] == 0


@pytest.fixture
def scenario_copy(tmp_path, monkeypatch):
    path = tmp_path / "scenarios.json"
    shutil.copy(DATA, path)
    service = ScenarioService(path)
    monkeypatch.setattr(scenario_routes, "get_scenario_service", lambda: service)
    return path, service


def test_edit_scenario_persists_and_applies(scenario_copy) -> None:
    path, service = scenario_copy
    assert service.resolve("Suv umuman yo'q", "uz")[0] is None

    response = client.put(
        "/api/v1/scenarios/vending_no_water_uz",
        json={"keywords": ["chiqma*", "suv yo'q"], "message": "Yangi javob."},
    )
    assert response.status_code == 200, response.text
    assert service.resolve("Suv umuman yo'q", "uz")[0].id == "vending_no_water_uz"

    saved = {s["id"]: s for s in json.loads(path.read_text(encoding="utf-8"))}
    assert saved["vending_no_water_uz"]["keywords"] == ["chiqma*", "suv yo'q"]
    assert saved["vending_no_water_uz"]["steps"][0]["message"] == "Yangi javob."
    # Untouched fields survive the rewrite.
    assert saved["vending_no_water_uz"]["context_keywords"] == ["suv", "apparat*", "tugma*"]


def test_enabled_scenario_needs_a_problem_keyword(scenario_copy) -> None:
    response = client.put("/api/v1/scenarios/vending_leak_ru", json={"keywords": [" "]})
    assert response.status_code == 422


def test_disabled_scenarios_are_listed_for_editing(scenario_copy) -> None:
    client.put("/api/v1/scenarios/vending_taste_ru", json={"enabled": False})
    enabled = {s["id"] for s in client.get("/api/v1/scenarios").json()}
    everything = {s["id"] for s in client.get("/api/v1/scenarios?include_disabled=true").json()}
    assert "vending_taste_ru" not in enabled
    assert "vending_taste_ru" in everything

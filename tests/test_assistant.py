import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.routes.admin as admin_routes
import app.assistant.answer as answer_module
import app.assistant.llm as llm
import app.realtime.audio_socket as audio_socket
import app.tts.prompt_cache as prompt_cache
import app.voice.analyze as analyze_module
from app.assistant import runtime as assistant
from app.assistant.answer import LiveAnswer, clean_reply, generate_answer, in_expected_script
from app.assistant.knowledge import KnowledgeEntryInput, Passage, get_knowledge_base, tokenize
from app.audio.resample import pcm16_to_wav
from app.calls.log import get_call_log
from app.core.config import get_settings
from app.main import app
from app.realtime.audio_socket import AudioSocketServer, AudioSocketSession
from app.speech.models import TranscriptionResult

client = TestClient(app)


# ---- modes ----

def test_minimum_is_the_default_and_a_choice_survives_a_restart() -> None:
    assert assistant.current().mode == assistant.MINIMUM
    assistant.get_assistant_store().update(assistant.AssistantSettingsUpdate(mode="maximum", top_k=4))
    assistant.get_assistant_store.cache_clear()  # a restart
    settings = assistant.current()
    assert (settings.mode, settings.top_k) == ("maximum", 4)
    assert settings.llm_model == get_settings().llm_model


def test_only_changed_settings_are_saved(monkeypatch) -> None:
    assistant.get_assistant_store().update(assistant.AssistantSettingsUpdate(mode="maximum"))
    saved = get_settings().assistant_settings_path.read_text(encoding="utf-8")
    assert "maximum" in saved and "llm_model" not in saved
    # A new model in .env still reaches an operator who only switched the mode.
    monkeypatch.setattr(get_settings(), "llm_model", "new-model")
    assistant.get_assistant_store.cache_clear()
    assert assistant.current().llm_model == "new-model"


def test_admin_api_switches_mode_and_rejects_unknown_ones() -> None:
    assert client.get("/api/v1/admin/mode").json()["mode"] == "minimum"
    assert client.put("/api/v1/admin/mode", json={"mode": "maximum"}).json()["mode"] == "maximum"
    assert client.put("/api/v1/admin/mode", json={"mode": "turbo"}).status_code == 422
    assert client.get("/admin").status_code == 200


def test_admin_api_is_protected_by_the_api_key(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "api_key", "secret")
    assert client.get("/api/v1/admin/mode").status_code == 401
    assert client.get("/api/v1/admin/mode", headers={"X-API-Key": "secret"}).status_code == 200


def test_status_reports_an_unreachable_llm(monkeypatch) -> None:
    monkeypatch.setattr(admin_routes.llm, "status", lambda url, model: {
        "reachable": False, "models": [], "model_ready": False, "error": "refused"})
    body = client.get("/api/v1/admin/status").json()
    assert body["llm"]["reachable"] is False
    assert body["knowledge"]["entries"] > 0
    assert set(body["tts_live"]) == {"uz", "ru"}


# ---- knowledge base ----

def test_new_base_starts_from_answering_scenarios() -> None:
    entries = get_knowledge_base().entries()
    ids = {e.id for e in entries}
    assert "scenario-vending_no_water_uz" in ids and "scenario-vending_no_water_ru" in ids
    assert not any("operator_request" in i for i in ids)  # transfers are not knowledge
    water = next(e for e in entries if e.id == "scenario-vending_no_water_uz")
    assert "Uzur so'raymiz" in water.text


@pytest.mark.parametrize("query, language, expected", [
    ("apparatdan suv chiqmayapti", "uz", "scenario-vending_no_water_uz"),
    ("апараттан сув чиқмаяпти", "uz", "scenario-vending_no_water_uz"),  # Uzbek Cyrillic
    ("автомат не наливает воду", "ru", "scenario-vending_no_water_ru"),
])
def test_search_finds_the_right_entry(query, language, expected) -> None:
    passages = get_knowledge_base().search(query, language, 3)
    assert passages and passages[0].id == expected


def test_search_uses_word_stems() -> None:
    assert tokenize("Chiqmayapti, chiqmadi!") == ["chiqm", "chiqm"]


def test_manual_entries_can_be_added_edited_and_deleted() -> None:
    created = client.post("/api/v1/admin/knowledge", json={
        "title": "Ish vaqti", "language": "uz", "text": "Qo'llab-quvvatlash xizmati har kuni ishlaydi."}).json()
    assert get_knowledge_base().search("xizmat qachon ishlaydi", "uz", 1)[0].id == created["id"]
    updated = client.put(f"/api/v1/admin/knowledge/{created['id']}", json={
        "title": "Ish vaqti", "language": "uz", "text": "Yangi matn", "enabled": False}).json()
    assert updated["enabled"] is False
    assert client.delete(f"/api/v1/admin/knowledge/{created['id']}").status_code == 200
    assert client.delete(f"/api/v1/admin/knowledge/{created['id']}").status_code == 404


def test_refreshing_from_scenarios_keeps_an_entry_switched_off() -> None:
    base = get_knowledge_base()
    entry = next(e for e in base.entries() if e.id == "scenario-vending_leak_uz")
    base.update(entry.id, KnowledgeEntryInput(title=entry.title, language=entry.language,
                                              text=entry.text, enabled=False))
    assert client.post("/api/v1/admin/knowledge/import-scenarios").json()["imported"] > 0
    assert not next(e for e in base.entries() if e.id == entry.id).enabled


# ---- answer generation ----

def test_reply_is_cleaned_for_speech() -> None:
    reply = "**Uzur so'raymiz.** Idishni tekshiring.\n- Tugmani bosing.\nYana bir gap."
    assert clean_reply(reply, 2) == "Uzur so'raymiz. Idishni tekshiring."
    assert clean_reply("Bez tochki", 2) == "Bez tochki."


def test_answer_must_be_in_the_callers_script() -> None:
    assert in_expected_script("Suv chiqmasa, nolni bosing.", "uz")
    assert not in_expected_script("Нажмите ноль.", "uz")
    assert in_expected_script("Нажмите ноль.", "ru")


def test_answer_is_grounded_in_the_knowledge_base(monkeypatch) -> None:
    sent = {}

    def fake_chat(url, model, messages, timeout, max_tokens=200, gpu_layers=None):
        sent["messages"] = messages
        return "Uzur so'raymiz. Idish to'g'ri turganini tekshiring. Uchinchi gap."

    monkeypatch.setattr(answer_module, "chat", fake_chat)
    answer = generate_answer("suv chiqmayapti", "uz", assistant.current())
    assert answer.text == "Uzur so'raymiz. Idish to'g'ri turganini tekshiring."
    assert answer.passages[0].id == "scenario-vending_no_water_uz"
    system = sent["messages"][0]["content"]
    assert "Uzbek, Latin script" in system and "Uzur so'raymiz" in system
    assert sent["messages"][-1] == {"role": "user", "content": "suv chiqmayapti"}


def test_an_answer_in_the_wrong_language_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(answer_module, "chat", lambda *a, **k: "Нажмите ноль.")
    with pytest.raises(llm.LlmError):
        generate_answer("suv chiqmayapti", "uz", assistant.current())


def test_gpu_layers_reach_ollama_and_can_go_back_to_automatic(monkeypatch) -> None:
    payloads = []
    monkeypatch.setattr(llm, "_post", lambda url, payload, timeout: payloads.append(payload) or {
        "message": {"content": "Javob."}})
    llm.chat("http://x", "m", [], 5, gpu_layers=12)
    llm.chat("http://x", "m", [], 5)
    assert payloads[0]["options"]["num_gpu"] == 12 and "num_gpu" not in payloads[1]["options"]

    store = assistant.get_assistant_store()
    assert store.update(assistant.AssistantSettingsUpdate(llm_gpu_layers=12)).llm_gpu_layers == 12
    assert store.update(assistant.AssistantSettingsUpdate(llm_gpu_layers=-1)).llm_gpu_layers is None


def test_chat_retries_without_think_for_models_that_lack_it(monkeypatch) -> None:
    payloads = []

    def fake_post(url, payload, timeout):
        payloads.append(dict(payload))
        if "think" in payload:
            raise llm.LlmError('HTTP 400: {"error":"model does not support thinking"}')
        return {"message": {"content": "<think>x</think> Javob."}}

    monkeypatch.setattr(llm, "_post", fake_post)
    assert llm.chat("http://x", "gemma3:4b", [], 5) == "Javob."
    assert "think" in payloads[0] and "think" not in payloads[1]


def test_admin_test_box_answers_and_speaks(monkeypatch) -> None:
    monkeypatch.setattr(admin_routes, "generate_answer", lambda text, lang, settings: LiveAnswer(
        text="Uzur so'raymiz.", language="uz", model="m", retrieve_ms=1, llm_ms=900,
        passages=[Passage(id="a", title="Suv", language="uz", text="t", score=1.0)]))
    monkeypatch.setattr(admin_routes, "live_pcm8", lambda text, lang: b"\x00\x00" * 800)
    body = client.post("/api/v1/admin/test", json={"text": "suv?", "language": "uz"}).json()
    assert body["answer"] == "Uzur so'raymiz."
    assert body["audio"].startswith("data:audio/wav;base64,")
    assert body["passages"][0]["title"] == "Suv"


# ---- calls ----

class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, chunk: bytes) -> None:
        self.data.extend(chunk)

    async def drain(self) -> None:
        pass


def _live_call(monkeypatch, generate):
    server = AudioSocketServer()
    said = []

    async def fake_prompt(_session, key, every_language=False):
        said.append(key)

    server._play_prompt = fake_prompt
    monkeypatch.setattr(audio_socket, "generate_answer", generate)
    monkeypatch.setattr(audio_socket, "live_pcm8", lambda text, lang: b"\x00\x00" * 160)
    get_call_log().start_call("call-1", "phone")

    async def scenario():
        session = AudioSocketSession("t", asyncio.StreamReader(), _Writer(), asyncio.Lock())
        session.asterisk_uuid = "call-1"
        return await server._answer_live(session, "suv chiqmayapti", "uz", 0.0, 0.5), session

    handled, session = asyncio.run(scenario())
    return handled, said, session


def test_maximum_mode_answers_live_and_remembers_the_exchange(monkeypatch) -> None:
    answer = LiveAnswer(text="Uzur so'raymiz.", language="uz", passages=[], model="m",
                        retrieve_ms=2, llm_ms=1500)
    handled, said, session = _live_call(monkeypatch, lambda *a: answer)
    assert handled and said == ["thinking", "anything_else"]
    assert session.answered and session.history == [("suv chiqmayapti", "Uzur so'raymiz.")]
    logged = [e for e in get_call_log().events("call-1") if e["kind"] == "answer"][0]
    assert logged["data"]["source"] == "rag-llm" and logged["text"] == "Uzur so'raymiz."


def test_maximum_mode_falls_back_to_recordings_when_the_llm_fails(monkeypatch) -> None:
    def broken(*a):
        raise llm.LlmError("connection refused")

    handled, said, session = _live_call(monkeypatch, broken)
    assert not handled and said == ["thinking"]
    kinds = [e["kind"] for e in get_call_log().events("call-1")]
    assert "live_fallback" in kinds and "answer" not in kinds


# ---- console ----

def _wav(seconds: float = 1.0) -> bytes:
    tone = 0.3 * np.sin(2 * np.pi * 300 * np.arange(int(16000 * seconds)) / 16000)
    return pcm16_to_wav((tone * 32767).astype("<i2").tobytes(), 16000)


@pytest.fixture
def console_max(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "tts_prompt_cache_dir", tmp_path)
    prompt_cache.get_prompt_cache.cache_clear()
    monkeypatch.setattr(prompt_cache, "synthesize_bytes", lambda text, lang: _wav(0.2))
    monkeypatch.setattr(analyze_module, "transcribe_pcm16", lambda *a: TranscriptionResult(
        text="Suv chiqmayapti", language="uz", language_probability=0.9, duration_seconds=1.0))
    assistant.get_assistant_store().update(assistant.AssistantSettingsUpdate(mode="maximum"))
    yield
    prompt_cache.get_prompt_cache.cache_clear()


def _analyze():
    return client.post("/api/v1/console/analyze", files={"audio": ("call.wav", _wav(), "audio/wav")},
                       data={"telephone": "true", "language": ""}).json()


def test_console_answers_live_in_maximum_mode(console_max, monkeypatch) -> None:
    monkeypatch.setattr(analyze_module, "generate_answer", lambda *a: LiveAnswer(
        text="Uzur so'raymiz.", language="uz", model="m", retrieve_ms=1, llm_ms=10,
        passages=[Passage(id="a", title="Suv chiqmayapti", language="uz", text="t", score=2.0)]))
    monkeypatch.setattr(analyze_module, "live_pcm8", lambda text, lang: b"\x00\x00" * 800)
    body = _analyze()
    assert body["live"] and body["mode"] == "maximum"
    assert body["response_text"] == "Uzur so'raymiz."
    assert body["response_audio_data"].startswith("data:audio/wav")
    assert body["passages"] == ["Suv chiqmayapti"]


def test_console_falls_back_to_the_recording_when_the_llm_is_down(console_max, monkeypatch) -> None:
    def broken(*a):
        raise llm.LlmError("connection refused")

    monkeypatch.setattr(analyze_module, "generate_answer", broken)
    body = _analyze()
    assert not body["live"] and body["intent"]["scenario_id"] == "vending_no_water_uz"
    assert "ИИ-ответ недоступен" in body["warning"]

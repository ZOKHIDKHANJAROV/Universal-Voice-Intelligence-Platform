import asyncio
import struct
import time
import uuid

import pytest
from fastapi.testclient import TestClient

import app.realtime.audio_socket as audio_socket
from app.calls import hub
from app.calls.log import BOT, CALLER, ENDED, SYSTEM, TRANSFERRED, CallLog, get_call_log
from app.core.config import get_settings
from app.main import app
from app.realtime.audio_socket import AUDIO_TYPE, DTMF_TYPE, HANGUP_TYPE, UUID_TYPE, AudioSocketServer

client = TestClient(app)


def _log(tmp_path, record_audio=False) -> CallLog:
    return CallLog(tmp_path / "c.db", tmp_path / "audio", record_audio)


def test_call_is_a_chat_of_events(tmp_path) -> None:
    log = _log(tmp_path)
    log.start_call("c1", "phone")
    log.add_event("c1", SYSTEM, "call_started")
    log.add_event("c1", CALLER, "utterance", "Suv chiqmayapti", {"language": "uz"})
    log.add_event("c1", BOT, "answer", "Uzur so'raymiz.", {"scenario_id": "vending_no_water_uz", "total_ms": 1400})
    log.end_call("c1")

    summary = log.summary("c1")
    assert summary["status"] == ENDED and summary["language"] == "uz"
    assert summary["turns"] == 1 and summary["last_text"] == "Uzur so'raymiz."
    assert [e["kind"] for e in log.events("c1")] == ["call_started", "utterance", "answer"]


def test_returning_call_is_reopened_not_duplicated(tmp_path) -> None:
    log = _log(tmp_path)
    log.start_call("c1", "phone")
    log.end_call("c1", TRANSFERRED)
    log.start_call("c1", "phone")
    assert [c["id"] for c in log.list_calls()] == ["c1"]
    assert log.summary("c1")["status"] == "active"


def test_stats_count_understood_answers_and_latency(tmp_path) -> None:
    log = _log(tmp_path)
    log.start_call("c1", "phone")
    log.add_event("c1", BOT, "answer", "a", {"scenario_id": "x", "total_ms": 1000})
    log.add_event("c1", BOT, "answer", "b", {"scenario_id": None, "total_ms": 3000})
    log.add_event("c1", BOT, "answer", "c", {"scenario_id": "y", "total_ms": 2000})
    stats = log.stats(time.time() - 60)
    assert stats["answers"] == 3 and stats["understood"] == 2
    assert stats["median_latency_ms"] == 2000
    assert stats["active"] == 1


def test_audio_is_kept_only_when_enabled(tmp_path) -> None:
    pcm = b"\x01\x00" * 800
    off = _log(tmp_path / "off")
    off.start_call("c1", "phone")
    event = off.add_event("c1", CALLER, "utterance", "x", audio_pcm16=pcm)
    assert event["audio"] is None and off.audio_path("c1", event["id"]) is None

    on = _log(tmp_path / "on", record_audio=True)
    on.start_call("c1", "phone")
    event = on.add_event("c1", CALLER, "utterance", "x", audio_pcm16=pcm)
    assert on.audio_path("c1", event["id"]).read_bytes()[:4] == b"RIFF"


def test_prune_removes_old_calls_and_audio(tmp_path) -> None:
    log = _log(tmp_path, record_audio=True)
    log.start_call("old", "phone")
    log.add_event("old", CALLER, "utterance", "x", audio_pcm16=b"\x00\x00" * 10)
    log._db.execute("UPDATE calls SET started_at = 0 WHERE id = 'old'")
    log.start_call("new", "phone")
    assert log.prune(days=30) == 1
    assert [c["id"] for c in log.list_calls()] == ["new"]
    assert not (tmp_path / "audio" / "old").exists()


def test_hub_delivers_updates_to_subscribers() -> None:
    async def scenario():
        queue = hub.subscribe()
        try:
            hub.publish({"type": "ping"})
            return await asyncio.wait_for(queue.get(), 1)
        finally:
            hub.unsubscribe(queue)

    assert asyncio.run(scenario()) == {"type": "ping"}


def test_monitor_api_lists_and_shows_calls() -> None:
    log = get_call_log()
    log.start_call("c1", "console")
    log.add_event("c1", CALLER, "utterance", "salom", {"language": "uz"})
    assert [c["id"] for c in client.get("/api/v1/monitor/calls").json()] == ["c1"]
    body = client.get("/api/v1/monitor/calls/c1").json()
    assert body["call"]["channel"] == "console" and body["events"][0]["text"] == "salom"
    assert client.get("/api/v1/monitor/calls/missing").status_code == 404
    assert client.get("/api/v1/monitor/stats").json()["calls"] == 1
    assert client.get("/monitor").status_code == 200


def test_stream_and_audio_check_key_parameter(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "api_key", "secret")
    assert client.get("/api/v1/monitor/stream").status_code == 401
    assert client.get("/api/v1/monitor/calls/c1/audio/1").status_code == 401
    assert client.get("/api/v1/monitor/calls/c1/audio/1?key=secret").status_code == 404


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, chunk: bytes) -> None:
        self.data.extend(chunk)

    async def drain(self) -> None:
        pass

    def get_extra_info(self, name):
        return ("test", 0)

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


def _message(payload: bytes, kind: int = AUDIO_TYPE) -> bytes:
    return bytes((kind,)) + struct.pack(">H", len(payload)) + payload


@pytest.fixture
def _short_prompts(monkeypatch):
    monkeypatch.setattr(audio_socket, "get_prompt_cache", lambda: type(
        "Cache", (), {"get_or_render_pcm8": staticmethod(lambda text, lang: b"\x00" * 1600)})())


def test_phone_call_is_logged_from_greeting_to_handover(_short_prompts) -> None:
    call = uuid.uuid4()

    async def scenario():
        reader = asyncio.StreamReader()
        reader.feed_data(_message(call.bytes, UUID_TYPE))
        task = asyncio.create_task(AudioSocketServer()._handle_client(reader, _Writer()))
        await asyncio.sleep(0.2)  # the caller presses 0 during the greeting
        reader.feed_data(_message(b"0", DTMF_TYPE))
        await asyncio.sleep(0.6)
        reader.feed_data(_message(b"", HANGUP_TYPE))
        await task

    asyncio.run(scenario())
    log = get_call_log()
    events = [(e["role"], e["kind"]) for e in log.events(str(call))]
    assert events[:2] == [(SYSTEM, "call_started"), (BOT, "greeting")]
    assert (SYSTEM, "dtmf") in events and (BOT, "transfer") in events
    assert events[-2:] == [(SYSTEM, "transfer"), (SYSTEM, "handed_over")]
    assert log.summary(str(call))["status"] == TRANSFERRED


def test_transfer_counts_even_when_the_caller_comes_back(tmp_path) -> None:
    log = _log(tmp_path)
    log.start_call("c1", "phone")
    log.add_event("c1", SYSTEM, "transfer")
    log.end_call("c1", TRANSFERRED)
    log.start_call("c1", "phone")  # nobody answered; back to the bot
    log.end_call("c1")
    assert log.stats(time.time() - 60)["transferred"] == 1

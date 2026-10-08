import asyncio
import struct
import uuid

import pytest
from fastapi.testclient import TestClient

import app.realtime.audio_socket as audio_socket
from app.main import app
from app.realtime import call_actions
from app.realtime.audio_socket import (
    AUDIO_TYPE,
    DTMF_TYPE,
    FRAME_BYTES,
    HANGUP_TYPE,
    UUID_TYPE,
    AudioSocketServer,
    AudioSocketSession,
)
from app.speech.models import TranscriptionResult

client = TestClient(app)


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, chunk: bytes) -> None:
        self.data.extend(chunk)

    async def drain(self) -> None:
        pass

    def messages(self) -> list[tuple[int, int]]:
        out, offset = [], 0
        while offset < len(self.data):
            kind, length = self.data[offset], struct.unpack(">H", self.data[offset + 1 : offset + 3])[0]
            out.append((kind, length))
            offset += 3 + length
        return out


def _message(payload: bytes, kind: int = AUDIO_TYPE) -> bytes:
    return bytes((kind,)) + struct.pack(">H", len(payload)) + payload


@pytest.fixture(autouse=True)
def _quiet_prompts(monkeypatch):
    # 0.1 s of prompt audio instead of the real cache.
    monkeypatch.setattr(audio_socket, "get_prompt_cache", lambda: type(
        "Cache", (), {"get_or_render_pcm8": staticmethod(lambda text, lang: b"\x00" * 1600)})())


def test_next_action_is_operator_once_then_hangup() -> None:
    call = str(uuid.uuid4())
    call_actions.request(call, call_actions.OPERATOR)
    assert client.get(f"/internal/calls/{call}/next").text == "operator"
    assert client.get(f"/internal/calls/{call}/next").text == "hangup"


def test_next_action_defaults_to_hangup_and_rejects_junk() -> None:
    assert client.get(f"/internal/calls/{uuid.uuid4()}/next").text == "hangup"
    assert client.get("/internal/calls/not-a-uuid/next").status_code == 404


async def _call(messages: list[bytes], call: uuid.UUID | None = None) -> tuple[_Writer, str]:
    call = call or uuid.uuid4()
    reader = asyncio.StreamReader()
    reader.feed_data(_message(call.bytes, UUID_TYPE))
    for message in messages:
        reader.feed_data(message)
    writer = _Writer()
    session = AudioSocketSession("test", reader, writer, asyncio.Lock())
    server = AudioSocketServer()
    run = asyncio.create_task(server._run_session(session))
    await asyncio.sleep(0.6)  # prompt playback is paced in real time
    reader.feed_data(_message(b"", HANGUP_TYPE))
    await run
    if session.response_task:
        await asyncio.gather(session.response_task, return_exceptions=True)
    return writer, str(call)


def test_pressing_zero_hands_the_call_to_an_operator() -> None:
    writer, call = asyncio.run(_call([_message(b"0", DTMF_TYPE)]))
    # Prompt audio frames, then a hangup message that ends AudioSocket().
    assert writer.messages()[-1] == (HANGUP_TYPE, 0)
    assert call_actions.take(call) == "operator"


def test_other_digits_do_not_transfer() -> None:
    writer, call = asyncio.run(_call([_message(b"5", DTMF_TYPE)]))
    assert (HANGUP_TYPE, 0) not in writer.messages()
    assert call_actions.take(call) == "hangup"


def test_asking_for_an_operator_transfers(monkeypatch) -> None:
    monkeypatch.setattr(audio_socket, "transcribe_pcm16", lambda *a: TranscriptionResult(
        text="operator bilan gaplashmoqchiman", language="uz",
        language_probability=0.9, duration_seconds=1.0))
    server = AudioSocketServer()
    call = str(uuid.uuid4())
    writer = _Writer()

    async def utterance() -> AudioSocketSession:
        session = AudioSocketSession("test", asyncio.StreamReader(), writer, asyncio.Lock(),
                                     asterisk_uuid=call)
        await server._process_utterance(session, b"\x00" * FRAME_BYTES * 30)
        return session

    session = asyncio.run(utterance())
    assert session.transferring
    assert writer.messages()[-1] == (HANGUP_TYPE, 0)
    assert call_actions.take(call) == "operator"


def test_failed_transfer_returns_the_caller_to_the_bot() -> None:
    call = str(uuid.uuid4())
    call_actions.request(call, call_actions.OPERATOR, "ru")
    assert client.get(f"/internal/calls/{call}/next").text == "operator"
    # The dialplan reports that nobody answered and reconnects the caller.
    assert client.get(f"/internal/calls/{call}/operator-unavailable").text == "ok"
    assert call_actions.take_operator_unavailable(call) == (True, "ru")
    # Only once: a later reconnect greets normally again.
    assert call_actions.take_operator_unavailable(call) == (False, None)


def test_returning_caller_hears_operators_busy(monkeypatch) -> None:
    played = []

    class _Cache:
        @staticmethod
        def get_or_render_pcm8(text, language):
            played.append((text, language))
            return b"\x00" * 1600

    monkeypatch.setattr(audio_socket, "get_prompt_cache", lambda: _Cache())
    call = uuid.uuid4()
    call_actions.request(str(call), call_actions.OPERATOR, "ru")
    call_actions.take(str(call))
    call_actions.mark_operator_unavailable(str(call))

    asyncio.run(_call([], call))
    assert played == [("Сейчас все операторы заняты. Пожалуйста, опишите проблему, я постараюсь помочь.", "ru")]

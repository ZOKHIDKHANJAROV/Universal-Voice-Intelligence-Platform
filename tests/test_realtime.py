import asyncio
import struct

import pytest

import app.realtime.audio_socket as audio_socket
from app.realtime.audio_socket import (
    AUDIO_TYPE,
    FRAME_BYTES,
    HANGUP_TYPE,
    AudioSocketServer,
    AudioSocketSession,
)

SPEECH = b"\x01" * FRAME_BYTES
SILENCE = b"\x00" * FRAME_BYTES


class _FakeVad:
    """Treats any non-zero frame as speech, so tests control VAD exactly."""

    def __init__(self, _mode: int) -> None:
        pass

    def is_speech(self, frame: bytes, _rate: int) -> bool:
        return frame[0] != 0


def _message(payload: bytes, message_type: int = AUDIO_TYPE) -> bytes:
    return bytes((message_type,)) + struct.pack(">H", len(payload)) + payload


async def _run(frames: list[bytes], playing: bool, barge_in: bool = True):
    server = AudioSocketServer()
    server.barge_in = barge_in
    utterances: list[bytes] = []

    async def capture(_session, pcm: bytes) -> None:
        utterances.append(pcm)

    server._process_utterance = capture
    reader = asyncio.StreamReader()
    for frame in frames:
        reader.feed_data(_message(frame))
    reader.feed_data(_message(b"", HANGUP_TYPE))

    playback = asyncio.create_task(asyncio.sleep(10)) if playing else None
    session = AudioSocketSession("test", reader, None, asyncio.Lock(), playback)
    await server._run_session(session)
    await asyncio.sleep(0)
    cancelled = playback is not None and playback.cancelled()
    if playback and not playback.done():
        playback.cancel()
    return cancelled, utterances


@pytest.fixture(autouse=True)
def _fake_vad(monkeypatch):
    monkeypatch.setattr(audio_socket.webrtcvad, "Vad", _FakeVad)


def test_short_noise_does_not_interrupt_playback() -> None:
    frames = [SPEECH] * 3 + [SILENCE] * 5 + [SPEECH] * 3
    cancelled, utterances = asyncio.run(_run(frames, playing=True))
    assert not cancelled
    assert utterances == []


def test_sustained_speech_interrupts_and_is_captured() -> None:
    # 5 frames trigger barge-in, then the caller keeps talking for 0.5 s and
    # pauses long enough to end the utterance.
    frames = [SPEECH] * 30 + [SILENCE] * 40
    cancelled, utterances = asyncio.run(_run(frames, playing=True))
    assert cancelled
    assert len(utterances) == 1
    # Every speech frame, including the ones that triggered barge-in, is kept.
    assert utterances[0].count(SPEECH) >= 30


def test_without_barge_in_the_bot_finishes_and_speech_is_not_heard() -> None:
    frames = [SPEECH] * 30 + [SILENCE] * 40
    cancelled, utterances = asyncio.run(_run(frames, playing=True, barge_in=False))
    assert not cancelled
    assert utterances == []


def test_barge_in_is_off_by_default() -> None:
    assert AudioSocketServer().barge_in is False


def test_utterance_is_processed_without_playback() -> None:
    frames = [SILENCE] * 5 + [SPEECH] * 25 + [SILENCE] * 40
    _, utterances = asyncio.run(_run(frames, playing=False))
    assert len(utterances) == 1


def test_call_end_to_end_over_tcp(tmp_path, monkeypatch) -> None:
    """Greeting in Uzbek then Russian, then a Russian complaint answered from the prompt cache."""
    import uuid

    import app.tts.prompt_cache as prompt_cache
    from app.audio.resample import pcm16_to_wav
    from app.core.config import get_settings
    from app.speech.models import TranscriptionResult

    synthesized: list[tuple[str, str]] = []

    def fake_tts(text: str, language: str) -> bytes:
        synthesized.append((text, language))
        return pcm16_to_wav(b"\x10\x00" * 2400, 24000)  # 0.1 s

    heard: list[tuple[int, tuple]] = []

    def fake_stt(audio, sample_rate, language, allowed):
        heard.append((len(audio), allowed))
        return TranscriptionResult(
            text="Аппарат не дает воду", language="ru",
            language_probability=0.9, duration_seconds=1.0,
        )

    monkeypatch.setattr(get_settings(), "tts_prompt_cache_dir", tmp_path)
    monkeypatch.setattr(prompt_cache, "synthesize_bytes", fake_tts)
    monkeypatch.setattr(audio_socket, "transcribe_pcm16", fake_stt)
    prompt_cache.get_prompt_cache.cache_clear()

    async def scenario() -> None:
        server = AudioSocketServer()
        server.host, server.port = "127.0.0.1", 0
        await server.start()
        port = server.server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        async def read_audio(frames: int) -> bytes:
            received = b""
            for _ in range(frames):
                header = await asyncio.wait_for(reader.readexactly(3), 3)
                received += await reader.readexactly(struct.unpack(">H", header[1:])[0])
            return received

        writer.write(_message(uuid.uuid4().bytes, audio_socket.UUID_TYPE))
        assert len(await read_audio(10)) == 10 * FRAME_BYTES  # greeting, uz + ru
        await asyncio.sleep(0.05)

        for frame in [SPEECH] * 25 + [SILENCE] * 40:
            writer.write(_message(frame))
        await writer.drain()
        response = await read_audio(5)
        assert len(response) == 5 * FRAME_BYTES

        writer.write(_message(b"", HANGUP_TYPE))
        await writer.drain()
        writer.close()
        await server.stop()

    try:
        asyncio.run(scenario())
    finally:
        prompt_cache.get_prompt_cache.cache_clear()

    # 25 speech frames plus the 700 ms end-of-speech wait, and up to 9 frames
    # of silence from the 200 ms preroll (stood in while no audio arrived),
    # with language detection clamped to Uzbek/Russian.
    [(samples, allowed)] = heard
    assert (25 + 35) * 160 <= samples <= (25 + 35 + 9) * 160
    assert allowed == ("uz", "ru")
    assert [language for _, language in synthesized] == ["uz", "ru", "ru", "ru"]
    assert synthesized[0][0].startswith("Assalomu alaykum")
    assert synthesized[2][0].startswith("Приносим извинения")
    assert synthesized[3][0] == "Есть ещё вопросы?"


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, chunk: bytes) -> None:
        self.data.extend(chunk)

    async def drain(self) -> None:
        pass


async def _quiet_line(frames: list[bytes], answered: bool = False, started_ago: float = 0.0):
    """Feed frames one by one (letting the bot's tasks run) and record what it says."""
    import time

    server = AudioSocketServer()
    server.no_input_ms, server.followup_ms, server.max_call_seconds = 200, 100, 60
    said: list[str] = []

    async def fake_prompt(_session, key, every_language=False) -> None:
        said.append(key)

    server._play_prompt = fake_prompt
    reader = asyncio.StreamReader()
    writer = _Writer()
    session = AudioSocketSession("test", reader, writer, asyncio.Lock())
    session.answered = answered
    session.started_at = time.time() - started_ago

    async def feed() -> None:
        for frame in frames:
            reader.feed_data(_message(frame))
            await asyncio.sleep(0)
        reader.feed_data(_message(b"", HANGUP_TYPE))

    feeder = asyncio.create_task(feed())
    await server._run_session(session)
    await feeder
    return said, session, writer


HANGUP_FRAME = bytes((HANGUP_TYPE,)) + struct.pack(">H", 0)


def test_silent_caller_is_asked_again_then_the_bot_hangs_up() -> None:
    said, session, writer = asyncio.run(_quiet_line([SILENCE] * 30))
    assert said == ["no_input", "goodbye"]
    assert session.ending and bytes(writer.data).endswith(HANGUP_FRAME)


def test_speech_restarts_the_silence_timer() -> None:
    frames = [SILENCE] * 8 + [SPEECH] * 3 + [SILENCE] * 8
    said, _, writer = asyncio.run(_quiet_line(frames))
    assert said == [] and not writer.data


def test_silence_after_an_answer_ends_the_call_without_asking_again() -> None:
    said, _, writer = asyncio.run(_quiet_line([SILENCE] * 10, answered=True))
    assert said == ["goodbye"]
    assert bytes(writer.data).endswith(HANGUP_FRAME)


def test_long_call_ends_at_the_time_limit() -> None:
    said, _, writer = asyncio.run(_quiet_line([SPEECH] * 3, started_ago=61))
    assert said == ["time_limit"]
    assert bytes(writer.data).endswith(HANGUP_FRAME)


def test_call_end_defaults() -> None:
    server = AudioSocketServer()
    assert (server.no_input_ms, server.followup_ms, server.max_call_seconds) == (15000, 8000, 300)


def test_missing_audio_counts_as_silence() -> None:
    # Phones with silence suppression send nothing while nobody talks.
    async def read():
        return await AudioSocketServer._read_message(asyncio.StreamReader(), 0.2)

    kind, payload = asyncio.run(read())
    assert kind == AUDIO_TYPE
    assert payload == SILENCE * 10  # 200 ms

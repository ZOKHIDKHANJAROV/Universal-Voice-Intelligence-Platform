from __future__ import annotations

import asyncio
import logging
import struct
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np
import webrtcvad

from app.core.config import get_settings
from app.intent.service import get_intent_service
from app.services.scenario_service import ScenarioService
from app.speech.service import transcribe_pcm16
from app.tts.service import synthesize_bytes

LOGGER = logging.getLogger("univoice.realtime")

AUDIO_TYPE = 0x10
UUID_TYPE = 0x01
HANGUP_TYPE = 0x00
FRAME_BYTES = 320  # 20 ms @ 8 kHz, 16-bit mono
FRAME_MS = 20


@dataclass
class AudioSocketSession:
    call_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    output_lock: asyncio.Lock
    response_task: asyncio.Task | None = None


class AudioSocketServer:
    def __init__(self) -> None:
        settings = get_settings()
        self.host = settings.realtime_audio_host
        self.port = settings.realtime_audio_port
        self.vad_mode = settings.realtime_vad_mode
        self.silence_ms = settings.realtime_silence_ms
        self.max_utterance_ms = settings.realtime_max_utterance_ms
        self.min_utterance_ms = settings.realtime_min_utterance_ms
        self.server: asyncio.AbstractServer | None = None
        self._sessions: dict[str, AudioSocketSession] = {}

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=self.port,
        )
        LOGGER.info("AudioSocket server listening on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self.server is None:
            return
        self.server.close()
        await self.server.wait_closed()
        self.server = None
        LOGGER.info("AudioSocket server stopped")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        call_id = str(uuid.uuid4())
        session = AudioSocketSession(
            call_id=call_id,
            reader=reader,
            writer=writer,
            output_lock=asyncio.Lock(),
        )
        self._sessions[session.call_id] = session
        LOGGER.info("AudioSocket connected call_id=%s peer=%s", call_id, peer)

        try:
            await self._run_session(session)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            LOGGER.info("AudioSocket disconnected call_id=%s", call_id)
        except Exception:
            LOGGER.exception("AudioSocket session failed call_id=%s", call_id)
        finally:
            if session.response_task and not session.response_task.done():
                session.response_task.cancel()
            self._sessions.pop(session.call_id, None)
            writer.close()
            await writer.wait_closed()

    async def _run_session(self, session: AudioSocketSession) -> None:
        vad = webrtcvad.Vad(self.vad_mode)
        frame_buffer = bytearray()
        speech_buffer = bytearray()
        preroll: deque[bytes] = deque(maxlen=10)
        in_speech = False
        silence_frames = 0
        max_frames = max(1, self.max_utterance_ms // FRAME_MS)
        min_frames = max(1, self.min_utterance_ms // FRAME_MS)
        speech_frames = 0

        while True:
            message_type, payload = await self._read_message(session.reader)

            if message_type == HANGUP_TYPE:
                return
            if message_type == UUID_TYPE:
                if len(payload) == 16:
                    call_uuid = str(uuid.UUID(bytes=payload))
                    LOGGER.info(
                        "AudioSocket UUID call_id=%s asterisk_uuid=%s",
                        session.call_id,
                        call_uuid,
                    )
                continue
            if message_type != AUDIO_TYPE:
                continue

            frame_buffer.extend(payload)
            while len(frame_buffer) >= FRAME_BYTES:
                frame = bytes(frame_buffer[:FRAME_BYTES])
                del frame_buffer[:FRAME_BYTES]
                is_speech = vad.is_speech(frame, 8000)
                preroll.append(frame)

                if session.response_task and not session.response_task.done():
                    if is_speech:
                        session.response_task.cancel()
                        LOGGER.info(
                            "barge-in call_id=%s: stopping response playback",
                            session.call_id,
                        )
                    continue

                if is_speech and not in_speech:
                    in_speech = True
                    speech_buffer.clear()
                    speech_buffer.extend(b"".join(preroll))
                    speech_frames = len(preroll)
                    silence_frames = 0
                    continue

                if in_speech:
                    speech_buffer.extend(frame)
                    speech_frames += 1
                    if is_speech:
                        silence_frames = 0
                    else:
                        silence_frames += 1

                    if (
                        silence_frames * FRAME_MS >= self.silence_ms
                        or speech_frames >= max_frames
                    ):
                        utterance = bytes(speech_buffer)
                        in_speech = False
                        speech_buffer.clear()
                        preroll.clear()
                        silence_frames = 0
                        if speech_frames >= min_frames:
                            session.response_task = asyncio.create_task(
                                self._process_utterance(session, utterance)
                            )

    async def _process_utterance(
        self,
        session: AudioSocketSession,
        pcm16: bytes,
    ) -> None:
        try:
            audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
            transcription = await asyncio.to_thread(
                transcribe_pcm16,
                audio,
                8000,
                "uz",
            )
            text = transcription.text.strip()
            if not text:
                return

            LOGGER.info(
                "STT call_id=%s text=%r confidence=%.3f",
                session.call_id,
                text,
                transcription.language_probability,
            )

            intent = get_intent_service().resolve(text)
            scenario = (
                ScenarioService().get_scenario(intent.scenario_id)
                if intent.scenario_id
                else None
            )
            if scenario is None:
                response_text = (
                    "Kechirasiz, muammoingizni aniqlay olmadim. "
                    "Operator bilan bog'lanish uchun 0 ni bosing."
                )
            else:
                response_text = (
                    scenario.steps[0].message
                    if scenario.steps
                    else "Operator bilan bog'lanish uchun 0 ni bosing."
                )

            LOGGER.info(
                "Scenario call_id=%s scenario=%s confidence=%.3f",
                session.call_id,
                intent.scenario_id,
                intent.confidence,
            )

            wav_bytes = await asyncio.to_thread(
                synthesize_bytes,
                response_text,
                "uz",
            )
            pcm8 = await asyncio.to_thread(_wav24_to_pcm8, wav_bytes)
            await self._send_pcm(session, pcm8)

            LOGGER.info(
                "TTS call_id=%s bytes=%d response=%r",
                session.call_id,
                len(pcm8),
                response_text,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception(
                "Realtime utterance failed call_id=%s",
                session.call_id,
            )

    async def _send_pcm(
        self,
        session: AudioSocketSession,
        pcm8: bytes,
    ) -> None:
        async with session.output_lock:
            for offset in range(0, len(pcm8), FRAME_BYTES):
                frame = pcm8[offset : offset + FRAME_BYTES]
                if len(frame) < FRAME_BYTES:
                    frame += b"\x00" * (FRAME_BYTES - len(frame))
                session.writer.write(
                    bytes((AUDIO_TYPE,))
                    + struct.pack(">H", len(frame))
                    + frame
                )
                await session.writer.drain()
                await asyncio.sleep(FRAME_MS / 1000)

    @staticmethod
    async def _read_message(
        reader: asyncio.StreamReader,
    ) -> tuple[int, bytes]:
        header = await reader.readexactly(3)
        message_type = header[0]
        length = struct.unpack(">H", header[1:3])[0]
        payload = await reader.readexactly(length)
        return message_type, payload


def _wav24_to_pcm8(wav_bytes: bytes) -> bytes:
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels != 1 or sample_width != 2:
        raise ValueError(
            f"Unsupported Navoiy WAV: channels={channels}, sample_width={sample_width}"
        )
    if rate != 24000:
        raise ValueError(f"Expected Navoiy WAV at 24000 Hz, got {rate}")

    samples = np.frombuffer(frames, dtype="<i2")
    usable = len(samples) - (len(samples) % 3)
    if usable <= 0:
        return b""
    # Navoiy outputs 24 kHz; 3-sample averaging produces 8 kHz telephony PCM.
    downsampled = (
        samples[:usable]
        .reshape(-1, 3)
        .astype(np.int32)
        .mean(axis=1)
        .clip(-32768, 32767)
        .astype("<i2")
    )
    return downsampled.tobytes()

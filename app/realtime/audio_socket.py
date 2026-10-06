from __future__ import annotations

import asyncio
import logging
import struct
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import webrtcvad

from app.audio.resample import pcm16_to_float
from app.core.config import get_settings
from app.intent.service import get_intent_service
from app.speech.service import transcribe_pcm16
from app.tts.prompt_cache import get_prompt_cache
from app.tts.prompts import prompt_text
from app.voice.responses import build_response

LOGGER = logging.getLogger("univoice.realtime")

AUDIO_TYPE = 0x10
UUID_TYPE = 0x01
HANGUP_TYPE = 0x00
SAMPLE_RATE = 8000
FRAME_BYTES = 320  # 20 ms @ 8 kHz, 16-bit mono
FRAME_MS = 20
# Asterisk drops an AudioSocket that stays silent for too long, so a silent
# frame is sent whenever nothing else went out for this long.
KEEPALIVE_SECONDS = 1.0


@dataclass
class AudioSocketSession:
    call_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    output_lock: asyncio.Lock
    response_task: asyncio.Task | None = None
    last_output: float = field(default_factory=time.monotonic)


class AudioSocketServer:
    def __init__(self) -> None:
        settings = get_settings()
        self.host = settings.realtime_audio_host
        self.port = settings.realtime_audio_port
        self.vad_mode = settings.realtime_vad_mode
        self.silence_ms = settings.realtime_silence_ms
        self.max_utterance_ms = settings.realtime_max_utterance_ms
        self.min_utterance_ms = settings.realtime_min_utterance_ms
        self.barge_in_frames = max(1, settings.realtime_barge_in_frames)
        self.languages = settings.stt_realtime_languages
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
        keepalive_task = asyncio.create_task(self._keepalive(session))

        try:
            await self._run_session(session)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError) as e:
            LOGGER.info("AudioSocket disconnected call_id=%s reason=%r", call_id, e)
        except Exception:
            LOGGER.exception("AudioSocket session failed call_id=%s", call_id)
        finally:
            keepalive_task.cancel()
            if session.response_task and not session.response_task.done():
                session.response_task.cancel()
            self._sessions.pop(session.call_id, None)
            writer.close()
            await writer.wait_closed()

    async def _keepalive(self, session: AudioSocketSession) -> None:
        try:
            while True:
                await asyncio.sleep(KEEPALIVE_SECONDS)
                if time.monotonic() - session.last_output < KEEPALIVE_SECONDS:
                    continue
                async with session.output_lock:
                    # Playback may have held the lock while we waited.
                    if time.monotonic() - session.last_output < KEEPALIVE_SECONDS:
                        continue
                    session.writer.write(
                        bytes((AUDIO_TYPE,))
                        + struct.pack(">H", FRAME_BYTES)
                        + (b"\x00" * FRAME_BYTES)
                    )
                    await session.writer.drain()
                    session.last_output = time.monotonic()
                    LOGGER.debug("Sent keepalive call_id=%s", session.call_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            LOGGER.error("Keepalive failed call_id=%s: %s", session.call_id, e)

    async def _run_session(self, session: AudioSocketSession) -> None:
        vad = webrtcvad.Vad(self.vad_mode)
        frame_buffer = bytearray()
        speech_buffer = bytearray()
        preroll: deque[bytes] = deque(maxlen=10)
        in_speech = False
        silence_frames = 0
        barge_in_frames = 0
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
                    session.response_task = asyncio.create_task(self._play_greeting(session))
                continue
            if message_type != AUDIO_TYPE:
                continue

            frame_buffer.extend(payload)
            while len(frame_buffer) >= FRAME_BYTES:
                frame = bytes(frame_buffer[:FRAME_BYTES])
                del frame_buffer[:FRAME_BYTES]
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                preroll.append(frame)

                if session.response_task and not session.response_task.done():
                    # A single VAD hit is usually line noise or our own echo;
                    # require sustained speech before cutting the bot off.
                    barge_in_frames = barge_in_frames + 1 if is_speech else 0
                    if barge_in_frames < self.barge_in_frames:
                        continue
                    # Cancellation completes on a later loop iteration; drop the
                    # reference now so following frames are not treated as playback.
                    session.response_task.cancel()
                    session.response_task = None
                    barge_in_frames = 0
                    LOGGER.info(
                        "barge-in call_id=%s: stopping response playback",
                        session.call_id,
                    )
                    # Fall through: preroll holds the interrupting speech, so the
                    # new utterance is captured from its first frame.

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

    async def _play_greeting(self, session: AudioSocketSession) -> None:
        try:
            text, language = prompt_text("greeting", self.languages[0])
            pcm8 = await asyncio.to_thread(get_prompt_cache().get_or_render_pcm8, text, language)
            await self._send_pcm(session, pcm8)
            LOGGER.info("Greeting sent for call_id=%s", session.call_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            LOGGER.error("Greeting failed call_id=%s: %s", session.call_id, e)

    async def _process_utterance(
        self,
        session: AudioSocketSession,
        pcm16: bytes,
    ) -> None:
        try:
            started = time.perf_counter()
            transcription = await asyncio.to_thread(
                transcribe_pcm16,
                pcm16_to_float(pcm16),
                SAMPLE_RATE,
                None,
                self.languages,
            )
            stt_done = time.perf_counter()
            text = transcription.text.strip()
            if not text:
                return

            caller_language = (
                transcription.language
                if transcription.language in self.languages
                else self.languages[0]
            )
            LOGGER.info(
                "STT call_id=%s text=%r lang=%s confidence=%.3f",
                session.call_id,
                text,
                transcription.language,
                transcription.language_probability,
            )

            intent = get_intent_service().resolve(text, caller_language)
            response_text, response_language = build_response(intent.scenario_id, caller_language)
            intent_done = time.perf_counter()
            LOGGER.info(
                "Scenario call_id=%s scenario=%s confidence=%.3f source=%s",
                session.call_id,
                intent.scenario_id,
                intent.confidence,
                intent.source,
            )

            pcm8 = await asyncio.to_thread(
                get_prompt_cache().get_or_render_pcm8,
                response_text,
                response_language,
            )
            audio_ready = time.perf_counter()
            LOGGER.info(
                "Latency call_id=%s stt=%.0fms intent=%.0fms audio=%.0fms total=%.0fms "
                "(+%dms end-of-speech wait)",
                session.call_id,
                (stt_done - started) * 1000,
                (intent_done - stt_done) * 1000,
                (audio_ready - intent_done) * 1000,
                (audio_ready - started) * 1000,
                self.silence_ms,
            )

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
        loop = asyncio.get_running_loop()
        async with session.output_lock:
            started = loop.time()
            for index, offset in enumerate(range(0, len(pcm8), FRAME_BYTES)):
                frame = pcm8[offset : offset + FRAME_BYTES]
                if len(frame) < FRAME_BYTES:
                    frame += b"\x00" * (FRAME_BYTES - len(frame))
                session.writer.write(
                    bytes((AUDIO_TYPE,))
                    + struct.pack(">H", len(frame))
                    + frame
                )
                await session.writer.drain()
                session.last_output = time.monotonic()
                # Pace against a fixed schedule; sleeping a flat 20 ms per frame
                # accumulates scheduler overhead and drifts behind real time.
                delay = started + (index + 1) * FRAME_MS / 1000 - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)

    @staticmethod
    async def _read_message(
        reader: asyncio.StreamReader,
    ) -> tuple[int, bytes]:
        header = await reader.readexactly(3)
        message_type = header[0]
        length = struct.unpack(">H", header[1:3])[0]
        payload = await reader.readexactly(length)
        return message_type, payload

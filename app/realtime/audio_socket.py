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
from app.calls.log import BOT, CALLER, ENDED, SYSTEM, TRANSFERRED, get_call_log
from app.core.config import get_settings
from app.intent.service import get_intent_service
from app.realtime import call_actions
from app.services.scenario_service import get_scenario_service
from app.speech.service import transcribe_pcm16
from app.tts.prompt_cache import get_prompt_cache
from app.tts.prompts import get_prompts, prompt_text
from app.voice.responses import build_response

LOGGER = logging.getLogger("univoice.realtime")

AUDIO_TYPE = 0x10
UUID_TYPE = 0x01
HANGUP_TYPE = 0x00
DTMF_TYPE = 0x03  # payload: one ASCII digit (Asterisk 20+)
OPERATOR_DIGIT = "0"
TRANSFER_ACTION = "transfer"
SAMPLE_RATE = 8000
FRAME_BYTES = 320  # 20 ms @ 8 kHz, 16-bit mono
FRAME_MS = 20
# Asterisk drops an AudioSocket that stays silent for too long, so a silent
# frame is sent whenever nothing else went out for this long.
KEEPALIVE_SECONDS = 1.0
# No audio from Asterisk for this long counts as silence (see _read_message).
NO_AUDIO_SECONDS = 0.2


@dataclass
class AudioSocketSession:
    call_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    output_lock: asyncio.Lock
    response_task: asyncio.Task | None = None
    last_output: float = field(default_factory=time.monotonic)
    # Call UUID chosen by the dialplan; the key Asterisk uses to ask what next.
    asterisk_uuid: str | None = None
    # Language of the caller's last utterance, for prompts that follow it.
    language: str | None = None
    # Set once a handover to an operator has started: the bot stops listening.
    transferring: bool = False
    # Set once the bot has started saying goodbye: it stops listening.
    ending: bool = False
    # The caller got a scenario answer: silence now means the conversation is over.
    answered: bool = False
    # "Please describe your problem" was already repeated once.
    reprompted: bool = False
    # Wall clock start of the call, kept across a return from the operator.
    started_at: float = field(default_factory=time.time)


class AudioSocketServer:
    def __init__(self) -> None:
        settings = get_settings()
        self.host = settings.realtime_audio_host
        self.port = settings.realtime_audio_port
        self.vad_mode = settings.realtime_vad_mode
        self.silence_ms = settings.realtime_silence_ms
        self.max_utterance_ms = settings.realtime_max_utterance_ms
        self.min_utterance_ms = settings.realtime_min_utterance_ms
        self.barge_in = settings.realtime_barge_in
        self.barge_in_frames = max(1, settings.realtime_barge_in_frames)
        # 0 turns a limit off.
        self.no_input_ms = settings.realtime_no_input_seconds * 1000
        self.followup_ms = settings.realtime_followup_seconds * 1000
        self.max_call_seconds = settings.realtime_max_call_seconds
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
            if session.asterisk_uuid:
                kind = "handed_over" if session.transferring else "hangup"
                self._log(session, SYSTEM, kind)
                get_call_log().end_call(
                    session.asterisk_uuid, TRANSFERRED if session.transferring else ENDED
                )
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
        # Consecutive 20 ms frames with neither the bot nor the caller talking.
        idle_frames = 0

        while True:
            message_type, payload = await self._read_message(session.reader, NO_AUDIO_SECONDS)

            if message_type == HANGUP_TYPE:
                return
            if message_type == UUID_TYPE:
                if len(payload) == 16:
                    call_uuid = str(uuid.UUID(bytes=payload))
                    session.asterisk_uuid = call_uuid
                    returned, language = call_actions.take_operator_unavailable(call_uuid)
                    LOGGER.info(
                        "AudioSocket UUID call_id=%s asterisk_uuid=%s%s",
                        session.call_id,
                        call_uuid,
                        " (back from a failed transfer)" if returned else "",
                    )
                    log = get_call_log()
                    log.start_call(call_uuid, "phone")
                    summary = log.summary(call_uuid)
                    if summary:
                        session.started_at = summary["started_at"]
                    if returned:
                        # No second greeting: apologise and keep helping.
                        self._log(session, SYSTEM, "returned")
                        session.language = language
                        session.response_task = asyncio.create_task(
                            self._play_prompt(session, "operator_unavailable")
                        )
                    else:
                        self._log(session, SYSTEM, "call_started")
                        session.response_task = asyncio.create_task(
                            # Each language in its own voice, Uzbek first.
                            self._play_prompt(session, "greeting", every_language=True)
                        )
                continue
            if message_type == DTMF_TYPE:
                digit = payload[:1].decode("ascii", "ignore")
                LOGGER.info("DTMF call_id=%s digit=%s", session.call_id, digit)
                self._log(session, SYSTEM, "dtmf", digit)
                if digit == OPERATOR_DIGIT and not session.transferring and not session.ending:
                    if session.response_task and not session.response_task.done():
                        session.response_task.cancel()
                    session.response_task = asyncio.create_task(self._transfer_to_operator(session))
                continue
            if message_type != AUDIO_TYPE or session.transferring or session.ending:
                continue

            frame_buffer.extend(payload)
            while len(frame_buffer) >= FRAME_BYTES:
                frame = bytes(frame_buffer[:FRAME_BYTES])
                del frame_buffer[:FRAME_BYTES]
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                preroll.append(frame)

                if session.response_task and not session.response_task.done():
                    idle_frames = 0
                    if not self.barge_in:
                        # The bot finishes what it says; the caller is heard
                        # again once it stops (preroll keeps the last 200 ms).
                        continue
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
                    self._log(session, SYSTEM, "barge_in")
                    # Fall through: preroll holds the interrupting speech, so the
                    # new utterance is captured from its first frame.

                idle_frames = 0 if (is_speech or in_speech) else idle_frames + 1
                step = self._next_step(session, idle_frames * FRAME_MS)
                if step is not None:
                    idle_frames = 0
                    in_speech = False
                    speech_buffer.clear()
                    session.response_task = asyncio.create_task(step)
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

    def _next_step(self, session: AudioSocketSession, idle_ms: int):
        """What the bot does on its own when the line is quiet, or None."""
        if self.max_call_seconds and time.time() - session.started_at >= self.max_call_seconds:
            return self._end_call(session, "time_limit", "time_limit")
        if session.answered:
            if self.followup_ms and idle_ms >= self.followup_ms:
                return self._end_call(session, "goodbye", "done")
        elif self.no_input_ms and idle_ms >= self.no_input_ms:
            if not session.reprompted:
                session.reprompted = True
                # Before the caller has spoken their language is unknown: ask in each.
                return self._play_prompt(session, "no_input", every_language=session.language is None)
            return self._end_call(session, "goodbye", "silence")
        return None

    async def _end_call(self, session: AudioSocketSession, key: str, reason: str) -> None:
        """Say goodbye, then hang up."""
        session.ending = True
        await self._play_prompt(session, key, every_language=session.language is None)
        self._log(session, SYSTEM, "bot_hangup", reason)
        LOGGER.info("Bot ends call_id=%s reason=%s", session.call_id, reason)
        await self._send_hangup(session)

    async def _send_hangup(self, session: AudioSocketSession) -> None:
        async with session.output_lock:
            # A hangup message ends AudioSocket(); the dialplan continues
            # (operator on a transfer, otherwise it hangs up).
            session.writer.write(bytes((HANGUP_TYPE,)) + struct.pack(">H", 0))
            await session.writer.drain()

    def _log(self, session: AudioSocketSession, role: str, kind: str, text: str | None = None,
             data: dict | None = None, audio_pcm16: bytes | None = None) -> None:
        if session.asterisk_uuid:
            get_call_log().add_event(session.asterisk_uuid, role, kind, text, data, audio_pcm16)

    async def _play_prompt(
        self, session: AudioSocketSession, key: str, every_language: bool = False
    ) -> None:
        """Say a prompt in the caller's language, or (greeting) in each call language in turn."""
        if every_language:
            variants = get_prompts()[key]
            wanted = [language for language in self.languages if language in variants]
        else:
            wanted = [session.language or self.languages[0]]
        for preferred in wanted:
            try:
                text, language = prompt_text(key, preferred)
                self._log(session, BOT, key, text, {"language": language})
                pcm8 = await asyncio.to_thread(get_prompt_cache().get_or_render_pcm8, text, language)
                await self._send_pcm(session, pcm8)
                LOGGER.info("Prompt %s sent call_id=%s lang=%s", key, session.call_id, language)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOGGER.error("Prompt %s failed call_id=%s: %s", key, session.call_id, e)

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
            heard = {
                "detected": transcription.language,
                "probability": round(transcription.language_probability, 3),
                "stt_ms": round((stt_done - started) * 1000),
                "seconds": round(len(pcm16) / 2 / SAMPLE_RATE, 2),
            }
            if not text:
                self._log(session, CALLER, "unrecognized", "", heard, pcm16)
                return
            session.reprompted = False

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

            session.language = caller_language
            self._log(session, CALLER, "utterance", text, {**heard, "language": caller_language}, pcm16)
            intent = get_intent_service().resolve(text, caller_language)
            scenario = (
                get_scenario_service().get_scenario(intent.scenario_id) if intent.scenario_id else None
            )
            if scenario is not None and scenario.action == TRANSFER_ACTION:
                LOGGER.info("Caller asked for an operator call_id=%s", session.call_id)
                await self._transfer_to_operator(session)
                return
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

            self._log(session, BOT, "answer", response_text, {
                "language": response_language,
                "scenario_id": intent.scenario_id,
                "scenario_title": scenario.title if scenario else None,
                "confidence": intent.confidence,
                "source": intent.source,
                "stt_ms": round((stt_done - started) * 1000),
                "intent_ms": round((intent_done - stt_done) * 1000),
                "audio_ms": round((audio_ready - intent_done) * 1000),
                "total_ms": round((audio_ready - started) * 1000),
            })
            await self._send_pcm(session, pcm8)
            LOGGER.info(
                "TTS call_id=%s bytes=%d response=%r",
                session.call_id,
                len(pcm8),
                response_text,
            )
            # A real answer: offer more help, then silence ends the call. A
            # "not understood" keeps waiting for the caller to try again.
            session.answered = intent.scenario_id is not None
            if session.answered:
                await self._play_prompt(session, "anything_else")
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception(
                "Realtime utterance failed call_id=%s",
                session.call_id,
            )

    async def _transfer_to_operator(self, session: AudioSocketSession) -> None:
        """Tell the caller, then end the AudioSocket so the dialplan dials an operator."""
        session.transferring = True
        # Errors are logged inside; hand over anyway, an operator without the
        # announcement beats none.
        await self._play_prompt(session, "transfer")

        if session.asterisk_uuid is None:
            LOGGER.error("Cannot transfer call_id=%s: no call UUID from Asterisk", session.call_id)
            return
        call_actions.request(session.asterisk_uuid, call_actions.OPERATOR, session.language)
        self._log(session, SYSTEM, "transfer")
        LOGGER.info("Transferring call_id=%s to an operator", session.call_id)
        await self._send_hangup(session)

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
        idle_timeout: float | None = None,
    ) -> tuple[int, bytes]:
        try:
            # Safe to time out: readexactly consumes nothing until all 3 bytes arrived.
            header = await asyncio.wait_for(reader.readexactly(3), idle_timeout)
        except asyncio.TimeoutError:
            # Phones with silence suppression send no audio while nobody talks.
            # Stand in silence for the gap so utterances still end and the
            # silence timers still run.
            return AUDIO_TYPE, b"\x00" * (FRAME_BYTES * round(idle_timeout * 1000 / FRAME_MS))
        message_type = header[0]
        length = struct.unpack(">H", header[1:3])[0]
        payload = await reader.readexactly(length)
        return message_type, payload

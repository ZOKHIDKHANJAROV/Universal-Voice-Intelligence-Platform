"""Run an uploaded recording through the same path as a live call, for the web console."""

import logging
import time
from pathlib import Path

from pydantic import BaseModel

from app.audio.telephony import TELEPHONY_RATE, telephony
from app.core.config import get_settings
from app.intent.models import IntentResult
from app.intent.service import get_intent_service
from app.services.scenario_service import get_scenario_service
from app.speech.service import transcribe_pcm16
from app.tts.prompt_cache import get_prompt_cache
from app.voice.responses import build_response

LOGGER = logging.getLogger("univoice.console")


class CallAnalysis(BaseModel):
    transcription: str
    language: str
    language_probability: float
    intent: IntentResult
    response_text: str
    response_language: str
    # Name of the pre-rendered WAV, or None when it is not rendered and live
    # synthesis is unavailable.
    response_audio: str | None
    # "transfer" when a live call would now be handed to an operator.
    action: str | None = None
    warning: str | None = None
    timings_ms: dict[str, float]


def analyze_recording(path: Path, telephone: bool = True, language: str | None = None) -> CallAnalysis:
    from faster_whisper import decode_audio

    settings = get_settings()
    audio = decode_audio(str(path), sampling_rate=16000)
    rate = 16000
    if telephone:
        audio, rate = telephony(audio), TELEPHONY_RATE

    started = time.perf_counter()
    result = transcribe_pcm16(audio, rate, language, settings.stt_realtime_languages)
    stt_done = time.perf_counter()

    realtime = settings.stt_realtime_languages
    caller_language = result.language if result.language in realtime else realtime[0]
    intent = get_intent_service().resolve(result.text, caller_language) if result.text else IntentResult(
        intent=None, scenario_id=None, confidence=0.0, source="empty"
    )
    response_text, response_language = build_response(intent.scenario_id, caller_language)
    scenario = get_scenario_service().get_scenario(intent.scenario_id) if intent.scenario_id else None
    intent_done = time.perf_counter()

    cache = get_prompt_cache()
    audio_name: str | None = cache.path_for(response_text, response_language).name
    warning = None
    try:
        cache.get_or_render_path(response_text, response_language)
    except (OSError, ValueError) as exc:
        LOGGER.warning("Console response audio unavailable: %s", exc)
        audio_name = None
        warning = "Фраза не озвучена, а TTS-сервис недоступен. Выполните make render-prompts."
    audio_done = time.perf_counter()

    return CallAnalysis(
        transcription=result.text,
        language=result.language,
        language_probability=result.language_probability,
        intent=intent,
        response_text=response_text,
        response_language=response_language,
        response_audio=audio_name,
        action=scenario.action if scenario else None,
        warning=warning,
        timings_ms={
            "stt": round((stt_done - started) * 1000),
            "intent": round((intent_done - stt_done) * 1000),
            "audio": round((audio_done - intent_done) * 1000),
        },
    )

import wave
from pathlib import Path

from app.intent.service import get_intent_service
from app.speech.service import transcribe
from app.tts.prompt_cache import get_prompt_cache
from app.voice.models import VoicePipelineResult
from app.voice.responses import build_response


class VoicePipelineService:
    def __init__(self) -> None:
        self._intent = get_intent_service()

    def process(self, audio_path: Path, language: str = "uz") -> VoicePipelineResult:
        transcription = transcribe(audio_path, language=language)
        intent = self._intent.resolve(transcription.text, language)
        response_text, response_language = build_response(intent.scenario_id, language)

        # Same pre-rendered 8 kHz audio a phone caller hears.
        audio_path = get_prompt_cache().get_or_render_path(response_text, response_language)
        with wave.open(str(audio_path), "rb") as wav_file:
            duration_seconds = wav_file.getnframes() / wav_file.getframerate()

        return VoicePipelineResult(
            transcription=transcription.text,
            language=transcription.language,
            intent=intent,
            response_text=response_text,
            audio_path=str(audio_path),
            audio_format="wav",
            duration_seconds=duration_seconds,
        )

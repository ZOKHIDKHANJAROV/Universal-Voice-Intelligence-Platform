from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.speech.base import SpeechToText
from app.speech.models import TranscriptionResult
from app.speech.providers.whisper import FasterWhisperSpeechToText


@lru_cache(maxsize=1)
def get_stt() -> SpeechToText:
    settings = get_settings()
    return FasterWhisperSpeechToText(
        model_name=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
    )


def transcribe(audio_path: Path, language: str = "uz") -> TranscriptionResult:
    return get_stt().transcribe(audio_path, language=language)

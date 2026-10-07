import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.speech.base import SpeechToText
from app.speech.models import TranscriptionResult


@lru_cache(maxsize=1)
def get_stt() -> SpeechToText:
    from app.speech.providers.routed import LanguageRoutedSpeechToText
    from app.speech.providers.whisper import FasterWhisperSpeechToText

    settings = get_settings()

    def load(model_name: str, initial_prompt: str) -> FasterWhisperSpeechToText:
        return FasterWhisperSpeechToText(
            model_name=model_name,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            beam_size=settings.stt_beam_size,
            initial_prompt=initial_prompt,
        )

    general = load(settings.stt_model, settings.stt_initial_prompt)
    if not settings.stt_model_uz:
        return general
    return LanguageRoutedSpeechToText(
        general, {"uz": load(settings.stt_model_uz, settings.stt_initial_prompt_uz)}
    )


@lru_cache(maxsize=1)
def _stt_slots() -> threading.BoundedSemaphore:
    # Concurrent transcriptions each allocate their own activations; on a small
    # GPU that is the difference between queueing briefly and running out of VRAM.
    return threading.BoundedSemaphore(max(1, get_settings().stt_max_concurrency))


def transcribe(audio_path: Path, language: str | None = None) -> TranscriptionResult:
    with _stt_slots():
        return get_stt().transcribe(audio_path, language=language)


def transcribe_pcm16(
    audio: np.ndarray,
    sample_rate: int = 8000,
    language: str | None = None,
    allowed_languages: tuple[str, ...] = (),
) -> TranscriptionResult:
    with _stt_slots():
        return get_stt().transcribe_pcm16(
            audio,
            sample_rate=sample_rate,
            language=language,
            allowed_languages=allowed_languages,
        )

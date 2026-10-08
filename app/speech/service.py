import logging
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.speech.base import SpeechToText
from app.speech.models import TranscriptionResult

LOGGER = logging.getLogger("univoice.stt")


def _missing_local_model(name: str) -> bool:
    """True for a local model path that does not exist (vs. a Hugging Face repo id)."""
    path = Path(name)
    # path.root also catches "/models/..." on Windows, where it is not absolute.
    looks_local = bool(path.root) or name.startswith((".", "models/", "models\\"))
    return looks_local and not path.exists()


@lru_cache(maxsize=1)
def get_stt() -> SpeechToText:
    from app.speech.providers.routed import LanguageRoutedSpeechToText
    from app.speech.providers.whisper import FasterWhisperSpeechToText

    settings = get_settings()

    def load(model_name: str, initial_prompt: str, without_timestamps: bool = False) -> FasterWhisperSpeechToText:
        return FasterWhisperSpeechToText(
            model_name=model_name,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            beam_size=settings.stt_beam_size,
            initial_prompt=initial_prompt,
            without_timestamps=without_timestamps,
            temperature_fallback=settings.stt_temperature_fallback,
        )

    general = load(settings.stt_model, settings.stt_initial_prompt)
    if not settings.stt_model_uz:
        return general
    if _missing_local_model(settings.stt_model_uz):
        LOGGER.warning(
            "STT_MODEL_UZ=%s not found; Uzbek uses %s. Run `make uz-model`.",
            settings.stt_model_uz,
            settings.stt_model,
        )
        return general
    return LanguageRoutedSpeechToText(
        general,
        {"uz": load(settings.stt_model_uz, settings.stt_initial_prompt_uz, settings.stt_without_timestamps_uz)},
    )


@lru_cache(maxsize=1)
def _stt_slots() -> threading.BoundedSemaphore:
    # Concurrent transcriptions each allocate their own activations; on a small
    # GPU that is the difference between queueing briefly and running out of VRAM.
    return threading.BoundedSemaphore(max(1, get_settings().stt_max_concurrency))


def warm_up() -> float:
    """Load the models and run one tiny decode per language; return seconds taken."""
    started = time.perf_counter()
    silence = np.zeros(8000, dtype=np.float32)
    for language in get_settings().stt_realtime_languages:
        transcribe_pcm16(silence, 8000, language)
    return time.perf_counter() - started


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

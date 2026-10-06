from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.tts.base import TextToSpeech
from app.tts.providers.http import NavoiyHttpTextToSpeech
from app.tts.providers.navoiy import NavoiyTextToSpeech
from app.tts.providers.piper import PiperTextToSpeech


@lru_cache(maxsize=4)
def _build_provider(name: str) -> TextToSpeech:
    settings = get_settings()

    if name == "navoiy-http":
        return NavoiyHttpTextToSpeech(
            settings.tts_base_url,
            settings.tts_timeout_seconds,
        )

    if name == "navoiy":
        return NavoiyTextToSpeech(
            settings.tts_python_binary,
            settings.tts_navoiy_inference_script,
            settings.tts_navoiy_cosyvoice_dir,
            settings.tts_navoiy_base_model_dir,
            settings.tts_navoiy_checkpoint,
            settings.tts_navoiy_reference_audio,
            settings.tts_navoiy_emotion,
        )

    if name == "piper":
        return PiperTextToSpeech(
            settings.tts_binary,
            {"uz": settings.tts_model_path, "ru": settings.tts_model_path_ru},
        )

    raise ValueError(f"Unknown TTS provider: {name}")


def get_tts(language: str = "uz") -> TextToSpeech:
    settings = get_settings()
    name = settings.tts_provider
    if language == "ru" and settings.tts_provider_ru:
        name = settings.tts_provider_ru
    return _build_provider(name)


def synthesize(text: str, output_path: Path, language: str = "uz"):
    return get_tts(language).synthesize(text, output_path, language)


def synthesize_bytes(text: str, language: str = "uz") -> bytes:
    """Synthesize to in-memory WAV bytes with the provider for ``language``."""
    return get_tts(language).synthesize_bytes(text, language)

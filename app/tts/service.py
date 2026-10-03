from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.tts.base import TextToSpeech
from app.tts.providers.http import NavoiyHttpTextToSpeech
from app.tts.providers.navoiy import NavoiyTextToSpeech
from app.tts.providers.piper import PiperTextToSpeech


@lru_cache(maxsize=1)
def get_tts() -> TextToSpeech:
    settings = get_settings()

    if settings.tts_provider == "navoiy-http":
        return NavoiyHttpTextToSpeech(
            settings.tts_base_url,
            settings.tts_timeout_seconds,
        )

    if settings.tts_provider == "navoiy":
        return NavoiyTextToSpeech(
            settings.tts_python_binary,
            settings.tts_navoiy_inference_script,
            settings.tts_navoiy_cosyvoice_dir,
            settings.tts_navoiy_base_model_dir,
            settings.tts_navoiy_checkpoint,
            settings.tts_navoiy_reference_audio,
            settings.tts_navoiy_emotion,
        )

    return PiperTextToSpeech(settings.tts_binary, settings.tts_model_path)


def synthesize(text: str, output_path: Path, language: str = "uz"):
    return get_tts().synthesize(text, output_path, language)


def synthesize_bytes(text: str, language: str = "uz") -> bytes:
    settings = get_settings()
    if settings.tts_provider != "navoiy-http":
        raise ValueError("Realtime TTS requires TTS_PROVIDER=navoiy-http")
    provider = get_tts()
    if not isinstance(provider, NavoiyHttpTextToSpeech):
        raise ValueError("Realtime TTS requires the Navoiy HTTP provider")
    return provider.synthesize_bytes(text, language=language)

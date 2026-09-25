from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.tts.base import TextToSpeech
from app.tts.providers.piper import PiperTextToSpeech


@lru_cache(maxsize=1)
def get_tts() -> TextToSpeech:
    settings = get_settings()
    return PiperTextToSpeech(settings.tts_binary, settings.tts_model_path)


def synthesize(text: str, output_path: Path, language: str = "uz"):
    return get_tts().synthesize(text, output_path, language)

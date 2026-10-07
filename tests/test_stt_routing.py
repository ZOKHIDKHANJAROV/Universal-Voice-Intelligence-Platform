import numpy as np

from app.speech.models import TranscriptionResult
from app.speech.providers.routed import LanguageRoutedSpeechToText


class _Model:
    def __init__(self, name: str, detected: tuple[str, float] = ("ru", 0.9)) -> None:
        self.name = name
        self.detected = detected
        self.calls: list[tuple[int, int, str | None]] = []

    def detect_language(self, audio16, allowed):
        self.detect_input = len(audio16)
        return self.detected

    def transcribe_pcm16(self, audio, sample_rate, language=None, allowed_languages=()):
        self.calls.append((len(audio), sample_rate, language))
        return TranscriptionResult(
            text=f"{self.name}:{language}", language=language or "xx",
            language_probability=1.0, duration_seconds=1.0,
        )

    def transcribe(self, audio_path, language=None):
        self.calls.append((0, 0, language))
        return TranscriptionResult(
            text=f"{self.name}:{language}", language=language or "xx",
            language_probability=1.0, duration_seconds=1.0,
        )


def _route(detected):
    general, uzbek = _Model("general", detected), _Model("uz-model")
    return LanguageRoutedSpeechToText(general, {"uz": uzbek}), general, uzbek


def test_uzbek_audio_goes_to_the_uzbek_model() -> None:
    router, general, uzbek = _route(("uz", 0.77))
    result = router.transcribe_pcm16(np.zeros(8000, np.float32), 8000, allowed_languages=("uz", "ru"))
    assert result.text == "uz-model:uz"
    assert result.language_probability == 0.77
    assert general.detect_input == 16000  # detection sees 16 kHz audio
    assert uzbek.calls == [(16000, 16000, "uz")]
    assert general.calls == []


def test_russian_audio_stays_on_the_general_model() -> None:
    router, general, uzbek = _route(("ru", 0.95))
    result = router.transcribe_pcm16(np.zeros(8000, np.float32), 8000, allowed_languages=("uz", "ru"))
    assert result.text == "general:ru"
    assert uzbek.calls == []


def test_explicit_language_skips_detection() -> None:
    router, general, uzbek = _route(("ru", 0.95))
    assert router.transcribe("call.wav", language="uz").text == "uz-model:uz"
    assert router.transcribe("call.wav", language="ru").text == "general:ru"
    assert not hasattr(general, "detect_input")

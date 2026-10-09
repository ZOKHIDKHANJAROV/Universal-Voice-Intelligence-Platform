import numpy as np

from app.speech.models import TranscriptionResult
from app.speech.providers.routed import LanguageRoutedSpeechToText


class _Model:
    def __init__(self, name: str, detected: tuple[str, float] = ("ru", 0.9),
                 logprob: float = -0.1, text: str | None = None) -> None:
        self.name = name
        self.detected = detected
        self.logprob = logprob
        self.text = text
        self.calls: list[tuple[int, int, str | None]] = []

    def detect_language(self, audio16, allowed):
        self.detect_input = len(audio16)
        return self.detected

    def transcribe_pcm16(self, audio, sample_rate, language=None, allowed_languages=()):
        self.calls.append((len(audio), sample_rate, language))
        text = f"{self.name}:{language}" if self.text is None else self.text
        return TranscriptionResult(
            text=text, language=language or "xx",
            language_probability=1.0, duration_seconds=1.0,
            avg_logprob=self.logprob if text else None,
        )

    def transcribe(self, audio_path, language=None):
        self.calls.append((0, 0, language))
        return TranscriptionResult(
            text=f"{self.name}:{language}", language=language or "xx",
            language_probability=1.0, duration_seconds=1.0,
        )


def _route(detected, general_logprob=-0.1, uz_logprob=-0.1, general_text=None, uz_text=None):
    general = _Model("general", detected, general_logprob, general_text)
    uzbek = _Model("uz-model", logprob=uz_logprob, text=uz_text)
    return LanguageRoutedSpeechToText(general, {"uz": uzbek}), general, uzbek


def test_uzbek_audio_goes_to_the_uzbek_model() -> None:
    router, general, uzbek = _route(("uz", 0.93))
    result = router.transcribe_pcm16(np.zeros(8000, np.float32), 8000, allowed_languages=("uz", "ru"))
    assert result.text == "uz-model:uz"
    assert result.language_probability == 0.93
    assert general.detect_input == 16000  # detection sees 16 kHz audio
    assert uzbek.calls == [(16000, 16000, "uz")]
    assert general.calls == []


def test_russian_audio_stays_on_the_general_model() -> None:
    router, general, uzbek = _route(("ru", 0.95))
    result = router.transcribe_pcm16(np.zeros(8000, np.float32), 8000, allowed_languages=("uz", "ru"))
    assert result.text == "general:ru"
    assert uzbek.calls == []


def _phrase(router):
    return router.transcribe_pcm16(np.zeros(8000, np.float32), 8000, allowed_languages=("uz", "ru"))


def test_unsure_detection_lets_the_surer_model_decide() -> None:
    # Real call: "suv chiqmayapti" detected as Dutch/Russian at 0.13.
    router, general, uzbek = _route(("ru", 0.13), general_logprob=-0.44, uz_logprob=-0.07)
    result = _phrase(router)
    assert result.text == "uz-model:uz" and result.language == "uz"
    assert general.calls == []  # confident Uzbek: the large model is not asked
    assert result.language_probability > 0.9


def test_unsure_uzbek_is_compared_with_the_general_model() -> None:
    router, general, uzbek = _route(("ru", 0.13), general_logprob=-0.83, uz_logprob=-0.4)
    assert _phrase(router).text == "uz-model:uz"
    assert general.calls and uzbek.calls  # both listened


def test_unsure_russian_stays_russian() -> None:
    # Real call: "Раз, два, три" - the Uzbek model heard "ras do tirik" at -0.80.
    router, _, _ = _route(("ru", 0.30), general_logprob=-0.25, uz_logprob=-0.80)
    assert _phrase(router).text == "general:ru"


def test_noise_heard_by_no_model_is_not_text() -> None:
    # Real call: a cough; Whisper's "Продолжение следует" is filtered, the
    # Uzbek model guessed "puf" at -1.14.
    router, _, _ = _route(("ru", 0.06), general_text="", uz_logprob=-1.14, uz_text="puf")
    result = _phrase(router)
    assert result.text == "" and result.language == "ru"


def test_noise_the_general_model_does_not_hear_is_not_text() -> None:
    # Real call: background music; large-v3 heard nothing, the Uzbek model
    # wrote "yigʻlab yubordim" at -0.63.
    router, _, _ = _route(("ru", 0.07), general_text="", uz_logprob=-0.63, uz_text="yigʻlab yubordim")
    assert _phrase(router).text == ""


def test_uzbek_the_general_model_does_not_hear_still_counts() -> None:
    router, _, _ = _route(("ru", 0.1), general_text="", uz_logprob=-0.4, uz_text="suv chiqmayapti")
    assert _phrase(router).text == "suv chiqmayapti"


def test_russian_beats_a_transliterating_uzbek_model() -> None:
    # Real call after loudness normalization: "Списалась деньги с карты" -0.43
    # against the Uzbek model's "spessialniy zdeniy skarda" -0.63.
    router, _, _ = _route(("ru", 0.86), general_logprob=-0.43, uz_logprob=-0.63)
    assert _phrase(router).text == "general:ru"


def test_explicit_language_skips_detection() -> None:
    router, general, uzbek = _route(("ru", 0.95))
    assert router.transcribe("call.wav", language="uz").text == "uz-model:uz"
    assert router.transcribe("call.wav", language="ru").text == "general:ru"
    assert not hasattr(general, "detect_input")


def test_missing_local_uzbek_model_is_detected(tmp_path) -> None:
    from app.speech.service import _missing_local_model

    assert _missing_local_model("/models/no-such-model-ct2")
    assert _missing_local_model("models/no-such-model-ct2")
    assert not _missing_local_model(str(tmp_path))  # exists
    assert not _missing_local_model("navai-uz/whisper-medium-uzbek")  # HF repo id


def test_warm_up_decodes_once_per_call_language(monkeypatch) -> None:
    import app.speech.service as service

    calls = []

    class _Stt:
        def transcribe_pcm16(self, audio, sample_rate, language=None, allowed_languages=()):
            calls.append((len(audio), sample_rate, language))

    monkeypatch.setattr(service, "get_stt", lambda: _Stt())
    service.warm_up()
    assert calls == [(8000, 8000, "uz"), (8000, 8000, "ru")]

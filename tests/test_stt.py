from types import SimpleNamespace

import numpy as np

from app.speech.providers.whisper import FasterWhisperSpeechToText


class _FakeModel:
    def __init__(self, detected: str, probabilities: list[tuple[str, float]]) -> None:
        self.calls: list[dict] = []
        self._detected = detected
        self._probabilities = probabilities

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        language = kwargs["language"] or self._detected
        info = SimpleNamespace(
            language=language,
            language_probability=0.9,
            duration=len(audio) / 16000 if isinstance(audio, np.ndarray) else 1.0,
            all_language_probs=self._probabilities if kwargs["language"] is None else None,
        )
        return iter([SimpleNamespace(text=f" text in {language} ")]), info


def _stt(model: _FakeModel) -> FasterWhisperSpeechToText:
    stt = FasterWhisperSpeechToText.__new__(FasterWhisperSpeechToText)
    stt._model = model
    stt._beam_size = 1
    stt._initial_prompt = None
    stt._without_timestamps = False
    return stt


def test_telephony_audio_is_resampled_to_16k() -> None:
    model = _FakeModel("uz", [("uz", 0.9)])
    result = _stt(model).transcribe_pcm16(np.zeros(8000, dtype=np.float32), 8000)
    # Whisper assumes 16 kHz: one second of 8 kHz audio must arrive as 16000 samples.
    assert len(model.calls[0]["audio"]) == 16000
    assert result.duration_seconds == 1.0
    assert model.calls[0]["vad_filter"] is False


def test_unexpected_language_is_redone_with_best_allowed() -> None:
    model = _FakeModel("kk", [("kk", 0.5), ("ru", 0.1), ("uz", 0.3)])
    result = _stt(model).transcribe_pcm16(
        np.zeros(8000, dtype=np.float32), 8000, allowed_languages=("uz", "ru")
    )
    assert [call["language"] for call in model.calls] == [None, "uz"]
    assert result.language == "uz"
    # Detection score (uz 0.3 + Kazakh 0.5), not the 1.0 a forced language reports.
    assert result.language_probability == 0.8
    assert result.text == "text in uz"


def test_uzbek_detected_as_relative_turkic_language_maps_to_uzbek() -> None:
    # Detection large-v3 actually returned for two Uzbek recordings over 8 kHz.
    for probabilities in (
        [("az", 0.44), ("kk", 0.3), ("tr", 0.04), ("ru", 0.012), ("uz", 0.0)],
        [("kk", 0.95), ("ru", 0.03), ("be", 0.0), ("uz", 0.0)],
    ):
        model = _FakeModel(probabilities[0][0], probabilities)
        result = _stt(model).transcribe_pcm16(
            np.zeros(8000, dtype=np.float32), 8000, allowed_languages=("uz", "ru")
        )
        assert result.language == "uz"
        assert result.language_probability > 0.7


def test_ukrainian_detection_maps_to_russian() -> None:
    model = _FakeModel("uk", [("uk", 0.6), ("ru", 0.3), ("kk", 0.05)])
    result = _stt(model).transcribe_pcm16(
        np.zeros(8000, dtype=np.float32), 8000, allowed_languages=("uz", "ru")
    )
    assert result.language == "ru"


def test_allowed_language_is_decoded_once() -> None:
    model = _FakeModel("ru", [("ru", 0.8)])
    _stt(model).transcribe_pcm16(np.zeros(8000, dtype=np.float32), 8000, allowed_languages=("uz", "ru"))
    assert len(model.calls) == 1


def test_without_timestamps_is_passed_to_whisper() -> None:
    model = _FakeModel("uz", [("uz", 0.9)])
    stt = _stt(model)
    stt._without_timestamps = True
    stt.transcribe_pcm16(np.zeros(8000, dtype=np.float32), 8000, language="uz")
    assert model.calls[0]["without_timestamps"] is True

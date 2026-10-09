import numpy as np

from app.audio.level import normalize_loudness


def _dbfs(audio: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def test_quiet_phone_speech_is_brought_to_minus_20_dbfs() -> None:
    t = np.arange(8000) / 8000
    quiet = (0.014 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # about -40 dBFS
    assert round(_dbfs(normalize_loudness(quiet))) == -20


def test_gain_never_clips_and_is_capped() -> None:
    spiky = np.zeros(8000, np.float32)
    spiky[100] = 0.5
    assert np.max(np.abs(normalize_loudness(spiky))) <= 0.99
    hiss = np.full(8000, 1e-5, np.float32)
    assert np.allclose(normalize_loudness(hiss), hiss * 10 ** (30 / 20))


def test_silence_is_left_alone() -> None:
    silence = np.zeros(800, np.float32)
    assert normalize_loudness(silence) is silence


def test_realtime_transcription_gets_normalized_audio(monkeypatch) -> None:
    import app.speech.service as service

    seen = []

    class _Stt:
        def transcribe_pcm16(self, audio, sample_rate, language=None, allowed_languages=()):
            seen.append(_dbfs(audio))

    monkeypatch.setattr(service, "get_stt", lambda: _Stt())
    t = np.arange(8000) / 8000
    service.transcribe_pcm16((0.01 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 8000)
    assert round(seen[0]) == -20

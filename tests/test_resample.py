import numpy as np

from app.audio.resample import pcm16_to_wav, resample, wav_to_pcm16


def _tone(freq: float, rate: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    # Skip the edges where the filter sees zero padding.
    core = x[len(x) // 10 : -len(x) // 10]
    return float(np.sqrt(np.mean(core**2)))


def _peak_hz(x: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spectrum) * rate / len(x))


def test_upsample_8k_to_16k_keeps_frequency_and_level() -> None:
    out = resample(_tone(1000, 8000), 8000, 16000)
    assert len(out) == 16000
    assert abs(_peak_hz(out, 16000) - 1000) < 2
    assert abs(_rms(out) - 0.5 / np.sqrt(2)) < 0.01


def test_downsample_24k_to_8k_passes_speech_band() -> None:
    out = resample(_tone(1000, 24000), 24000, 8000)
    assert len(out) == 8000
    assert abs(_peak_hz(out, 8000) - 1000) < 2
    assert abs(_rms(out) - 0.5 / np.sqrt(2)) < 0.01


def test_downsample_24k_to_8k_rejects_aliasing() -> None:
    # 5 kHz is above the 4 kHz output Nyquist; a naive 3-sample average
    # leaks it back as a 3 kHz tone at about half amplitude.
    out = resample(_tone(5000, 24000), 24000, 8000)
    assert _rms(out) < 0.005


def test_wav_round_trip_to_telephony_rate() -> None:
    pcm24 = (_tone(440, 24000, 0.5) * 32767).astype("<i2").tobytes()
    pcm8 = wav_to_pcm16(pcm16_to_wav(pcm24, 24000), 8000)
    assert len(pcm8) == 2 * 4000

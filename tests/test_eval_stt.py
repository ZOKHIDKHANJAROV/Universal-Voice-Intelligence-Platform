import numpy as np

from scripts.eval_stt import error_rates, telephony


def test_telephony_keeps_speech_band_and_halves_rate() -> None:
    t = np.arange(16000) / 16000
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    phone = telephony(tone)
    assert len(phone) == 8000
    core = phone[800:-800]
    # mu-law adds quantization noise but keeps the level.
    assert abs(np.sqrt(np.mean(core**2)) - 0.5 / np.sqrt(2)) < 0.02


def test_error_rates_ignore_script_and_punctuation_for_uzbek() -> None:
    wer, cer = error_rates("Suv chiqmayapti!", "Сув чиқмаяпти", "uz")
    assert wer == 0.0 and cer == 0.0


def test_error_rates_count_wrong_words() -> None:
    wer, _ = error_rates("pulimni yeb qo'ydi", "pulimni yeb ketdi", "uz")
    assert abs(wer - 1 / 3) < 1e-9


def test_kazakh_cyrillic_output_is_compared_as_uzbek_latin() -> None:
    wer, _ = error_rates("o'z ichiga oladi", "өз ичига олади", "uz")
    assert wer == 0.0

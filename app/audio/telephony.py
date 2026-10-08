"""Simulate the phone channel callers are heard through."""

import numpy as np

from app.audio.resample import resample

TELEPHONY_RATE = 8000
_MU = 255.0


def telephony(audio16: np.ndarray, mu_law: bool = True) -> np.ndarray:
    """Downsample 16 kHz audio to 8 kHz and round-trip it through 8-bit mu-law like G.711."""
    audio = np.clip(resample(audio16, 16000, TELEPHONY_RATE), -1.0, 1.0)
    if not mu_law:
        return audio
    compressed = np.sign(audio) * np.log1p(_MU * np.abs(audio)) / np.log1p(_MU)
    quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
    return (np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(_MU)) / _MU).astype(np.float32)

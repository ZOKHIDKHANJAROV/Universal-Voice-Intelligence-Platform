"""Band-limited resampling and WAV helpers.

Pure numpy, so it needs no extra dependency and stays cheap on CPU.
"""

from __future__ import annotations

import io
import math
import wave
from functools import lru_cache

import numpy as np

# Kaiser-windowed sinc. 16 zero crossings per side gives >60 dB stopband
# rejection, which is plenty for 8 kHz telephony.
_ZERO_CROSSINGS = 16
_KAISER_BETA = 8.0
_ROLLOFF = 0.95
# Output samples per vectorised block; bounds peak memory to a few MB.
_BLOCK = 8192


@lru_cache(maxsize=16)
def _polyphase_table(up: int, down: int) -> tuple[np.ndarray, np.ndarray]:
    """Filter taps for each of the ``up`` fractional phases of an up/down ratio."""
    # Cutoff relative to the input Nyquist: when downsampling it must sit
    # below the output Nyquist, otherwise high frequencies fold back.
    cutoff = min(1.0, up / down) * _ROLLOFF
    half_width = int(math.ceil(_ZERO_CROSSINGS / cutoff))
    offsets = np.arange(-half_width + 1, half_width + 1)
    distance = (np.arange(up) / up)[:, None] - offsets[None, :]
    envelope = np.clip(1.0 - (distance / half_width) ** 2, 0.0, None)
    window = np.i0(_KAISER_BETA * np.sqrt(envelope)) / np.i0(_KAISER_BETA)
    taps = cutoff * np.sinc(cutoff * distance) * window
    return offsets, taps.astype(np.float32)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float audio between arbitrary rates with an anti-alias filter."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate:
        return x.copy()
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)

    divisor = math.gcd(src_rate, dst_rate)
    up, down = dst_rate // divisor, src_rate // divisor
    offsets, table = _polyphase_table(up, down)

    n_out = x.size * up // down
    out = np.empty(n_out, dtype=np.float32)
    for start in range(0, n_out, _BLOCK):
        scaled = np.arange(start, min(start + _BLOCK, n_out), dtype=np.int64) * down
        indices = (scaled // up)[:, None] + offsets[None, :]
        valid = (indices >= 0) & (indices < x.size)
        samples = np.where(valid, x[np.clip(indices, 0, x.size - 1)], 0.0)
        out[start : start + scaled.size] = np.einsum(
            "ij,ij->i", samples, table[scaled % up]
        )
    return out


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def wav_to_pcm16(wav_bytes: bytes, target_rate: int) -> bytes:
    """Decode a mono 16-bit WAV at any rate into raw PCM16 at ``target_rate``."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels != 1 or sample_width != 2:
        raise ValueError(
            f"Unsupported WAV: channels={channels}, sample_width={sample_width}"
        )
    if rate == target_rate:
        return frames
    return float_to_pcm16(resample(pcm16_to_float(frames), rate, target_rate))


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()

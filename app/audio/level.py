"""Loudness normalization before speech recognition."""

import numpy as np

# Softphones and phone lines often deliver speech at -40 dBFS. Whisper's
# language detector then loses confidence ("Верните мои деньги" scored Russian
# 0.82 at -41 dBFS, 0.94 at -20 dBFS) and Russian went to the Uzbek model.
TARGET_DBFS = -20.0
# Cap so near-silence is not blown up into loud noise.
MAX_GAIN_DB = 30.0
PEAK_LIMIT = 0.99


def normalize_loudness(audio: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    """Scale float audio to ``target_dbfs`` RMS without clipping or more than +30 dB."""
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms <= 0.0 or peak <= 0.0:
        return audio
    gain = min(
        10 ** (target_dbfs / 20) / rms,
        PEAK_LIMIT / peak,
        10 ** (MAX_GAIN_DB / 20),
    )
    return (audio * gain).astype(np.float32)

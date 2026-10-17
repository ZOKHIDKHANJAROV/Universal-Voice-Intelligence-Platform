"""Speak text that was not pre-rendered (maximum mode answers)."""

from functools import lru_cache

from app.audio.resample import wav_to_pcm16
from app.tts.pronunciation import spoken_text
from app.tts.service import synthesize_bytes

TELEPHONY_RATE = 8000


@lru_cache(maxsize=64)
def live_pcm8(text: str, language: str) -> bytes:
    """8 kHz PCM for ``text``; recent answers are kept in memory, never on disk."""
    return wav_to_pcm16(synthesize_bytes(spoken_text(text, language), language), TELEPHONY_RATE)

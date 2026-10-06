"""On-disk cache of phrases pre-rendered to 8 kHz telephony audio.

Every phrase the bot says is fixed text, so it is synthesized once (offline, by
scripts/render_prompts.py) instead of on every call. That keeps the GPU TTS
model out of the call path entirely: no VRAM held, no seconds of dead air.
"""

import hashlib
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

from app.audio.resample import pcm16_to_wav, wav_to_pcm16
from app.core.config import get_settings
from app.tts.service import synthesize_bytes

LOGGER = logging.getLogger("univoice.prompts")

TELEPHONY_RATE = 8000


class PromptAudioCache:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._memory: dict[Path, bytes] = {}
        self._lock = threading.Lock()

    def path_for(self, text: str, language: str) -> Path:
        # Keyed by content, so editing a phrase invalidates its audio automatically.
        digest = hashlib.sha1(f"{language}\n{text}".encode("utf-8")).hexdigest()[:16]
        return self._directory / f"{language}-{digest}.wav"

    def get_pcm8(self, text: str, language: str) -> bytes | None:
        path = self.path_for(text, language)
        with self._lock:
            cached = self._memory.get(path)
        if cached is not None:
            return cached
        if not path.exists():
            return None
        pcm8 = wav_to_pcm16(path.read_bytes(), TELEPHONY_RATE)
        with self._lock:
            self._memory[path] = pcm8
        return pcm8

    def render(self, text: str, language: str, force: bool = False) -> Path:
        path = self.path_for(text, language)
        if path.exists() and not force:
            return path
        pcm8 = wav_to_pcm16(synthesize_bytes(text, language), TELEPHONY_RATE)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(pcm16_to_wav(pcm8, TELEPHONY_RATE))
        os.replace(temporary, path)
        with self._lock:
            self._memory[path] = pcm8
        return path

    def get_or_render_pcm8(self, text: str, language: str) -> bytes:
        cached = self.get_pcm8(text, language)
        if cached is not None:
            return cached
        LOGGER.warning(
            "Prompt not pre-rendered, synthesizing live (run scripts/render_prompts.py): %r",
            text,
        )
        self.render(text, language)
        return self.get_pcm8(text, language) or b""

    def get_or_render_path(self, text: str, language: str) -> Path:
        path = self.path_for(text, language)
        if not path.exists():
            self.render(text, language)
        return path

    def preload(self, phrases: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Load rendered phrases into memory; return the ones still missing."""
        missing = []
        for text, language in phrases:
            try:
                if self.get_pcm8(text, language) is None:
                    missing.append((text, language))
            except (OSError, ValueError) as exc:
                LOGGER.error("Broken prompt audio for %r: %s", text, exc)
                missing.append((text, language))
        return missing


@lru_cache(maxsize=1)
def get_prompt_cache() -> PromptAudioCache:
    return PromptAudioCache(get_settings().tts_prompt_cache_dir)

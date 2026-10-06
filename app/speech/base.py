from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.speech.models import TranscriptionResult


class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        raise NotImplementedError

    @abstractmethod
    def transcribe_pcm16(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        allowed_languages: tuple[str, ...] = (),
    ) -> TranscriptionResult:
        """Transcribe float32 mono audio already segmented by the caller's VAD."""
        raise NotImplementedError

from abc import ABC, abstractmethod
from pathlib import Path

from app.speech.models import TranscriptionResult


class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        raise NotImplementedError

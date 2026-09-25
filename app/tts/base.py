from abc import ABC, abstractmethod
from pathlib import Path

from app.tts.models import SpeechSynthesisResult


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: Path, language: str = "uz") -> SpeechSynthesisResult:
        raise NotImplementedError

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from app.tts.models import SpeechSynthesisResult


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_path: Path, language: str = "uz") -> SpeechSynthesisResult:
        raise NotImplementedError

    def synthesize_bytes(self, text: str, language: str = "uz") -> bytes:
        """Return WAV bytes. Providers that can stream to memory override this."""
        with tempfile.TemporaryDirectory(prefix="univoice-tts-") as directory:
            output_path = Path(directory) / "speech.wav"
            self.synthesize(text, output_path, language)
            return output_path.read_bytes()

import subprocess
from pathlib import Path

from app.tts.base import TextToSpeech
from app.tts.models import SpeechSynthesisResult


class PiperTextToSpeech(TextToSpeech):
    def __init__(self, binary: str = "piper", model_path: str = "") -> None:
        self._binary = binary
        self._model_path = model_path

    def synthesize(self, text: str, output_path: Path, language: str = "uz") -> SpeechSynthesisResult:
        if language != "uz":
            raise ValueError("The configured Piper provider currently supports Uzbek only")
        if not self._model_path:
            raise ValueError("TTS_MODEL_PATH is required for the Piper provider")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self._binary, "--model", self._model_path, "--output_file", str(output_path)],
            input=text,
            text=True,
            check=True,
        )
        return SpeechSynthesisResult(
            audio_path=str(output_path), format="wav", language=language, duration_seconds=0.0
        )

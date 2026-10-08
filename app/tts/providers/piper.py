import os
import shutil
import subprocess
import wave
from pathlib import Path

from app.tts.base import TextToSpeech
from app.tts.models import SpeechSynthesisResult


class PiperTextToSpeech(TextToSpeech):
    """CPU-only Piper voices, one ONNX model per language."""

    def __init__(self, binary: str = "piper", model_paths: dict[str, str] | None = None) -> None:
        # Resolve now: Windows does not launch relative paths like
        # ".venv/Scripts/piper.exe" given to subprocess.
        self._binary = shutil.which(binary) or binary
        self._model_paths = {lang: path for lang, path in (model_paths or {}).items() if path}

    def synthesize(self, text: str, output_path: Path, language: str = "uz") -> SpeechSynthesisResult:
        model_path = self._model_paths.get(language)
        if not model_path:
            raise ValueError(f"No Piper model configured for language '{language}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self._binary, "--model", model_path, "--output_file", str(output_path)],
            input=text,
            # Cyrillic must reach Piper as UTF-8; Windows would default to cp1251.
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=True,
        )
        with wave.open(str(output_path), "rb") as wav_file:
            duration_seconds = wav_file.getnframes() / wav_file.getframerate()
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            format="wav",
            language=language,
            duration_seconds=duration_seconds,
        )

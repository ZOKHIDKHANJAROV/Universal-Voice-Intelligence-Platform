import os
import shutil
import subprocess
import wave
from pathlib import Path

from app.tts.base import TextToSpeech
from app.tts.models import SpeechSynthesisResult


class PiperTextToSpeech(TextToSpeech):
    """CPU-only Piper voices, one ONNX model per language."""

    def __init__(
        self,
        binary: str = "piper",
        model_paths: dict[str, str] | None = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ) -> None:
        # Resolve now: Windows does not launch relative paths like
        # ".venv/Scripts/piper.exe" given to subprocess.
        self._binary = shutil.which(binary) or binary
        self._model_paths = {lang: path for lang, path in (model_paths or {}).items() if path}
        # Piper's own defaults are left off the command line.
        self._voice_args: list[str] = []
        for flag, value, default in (
            ("--length_scale", length_scale, 1.0),
            ("--noise_scale", noise_scale, 0.667),
            ("--noise_w", noise_w, 0.8),
        ):
            if value != default:
                self._voice_args += [flag, str(value)]

    def synthesize(self, text: str, output_path: Path, language: str = "uz") -> SpeechSynthesisResult:
        model_path = self._model_paths.get(language)
        if not model_path:
            raise ValueError(f"No Piper model configured for language '{language}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self._binary, "--model", model_path, *self._voice_args, "--output_file", str(output_path)],
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

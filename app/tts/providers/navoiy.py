import subprocess
from pathlib import Path

from app.tts.base import TextToSpeech
from app.tts.models import SpeechSynthesisResult


class NavoiyTextToSpeech(TextToSpeech):
    def __init__(
        self,
        python_binary: str,
        inference_script: str,
        cosyvoice_dir: str,
        base_model_dir: str,
        checkpoint: str,
        reference_audio: str,
        emotion: str = "warm",
    ) -> None:
        self._python_binary = python_binary
        self._inference_script = inference_script
        self._cosyvoice_dir = cosyvoice_dir
        self._base_model_dir = base_model_dir
        self._checkpoint = checkpoint
        self._reference_audio = reference_audio
        self._emotion = emotion

    def synthesize(
        self, text: str, output_path: Path, language: str = "uz"
    ) -> SpeechSynthesisResult:
        if language != "uz":
            raise ValueError("Navoiy TTS provider currently supports Uzbek only")
        required = {
            "TTS_NAVOIY_INFERENCE_SCRIPT": self._inference_script,
            "TTS_NAVOIY_COSYVOICE_DIR": self._cosyvoice_dir,
            "TTS_NAVOIY_BASE_MODEL_DIR": self._base_model_dir,
            "TTS_NAVOIY_CHECKPOINT": self._checkpoint,
            "TTS_NAVOIY_REFERENCE_AUDIO": self._reference_audio,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing Navoiy TTS configuration: {', '.join(missing)}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                self._python_binary,
                self._inference_script,
                "--cosyvoice-dir",
                self._cosyvoice_dir,
                "--base-model-dir",
                self._base_model_dir,
                "--checkpoint",
                self._checkpoint,
                "--reference",
                self._reference_audio,
                "--text",
                text,
                "--emotion",
                self._emotion,
                "--output",
                str(output_path),
            ],
            check=True,
        )
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            format="wav",
            language=language,
            duration_seconds=0.0,
        )

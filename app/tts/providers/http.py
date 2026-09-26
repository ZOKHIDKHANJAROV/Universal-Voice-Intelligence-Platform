import urllib.error
import urllib.request
import wave
from pathlib import Path

from app.tts.base import TextToSpeech
from app.tts.models import SpeechSynthesisResult


class NavoiyHttpTextToSpeech(TextToSpeech):
    def __init__(self, base_url: str, timeout_seconds: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def synthesize(
        self, text: str, output_path: Path, language: str = "uz"
    ) -> SpeechSynthesisResult:
        if language != "uz":
            raise ValueError("Navoiy HTTP TTS currently supports Uzbek only")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = f'{{"text": {text!r}, "language": {language!r}}}'.replace("'", '"')
        request = urllib.request.Request(
            f"{self._base_url}/v1/synthesize",
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                output_path.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OSError(f"Navoiy TTS service returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OSError(f"Navoiy TTS service is unavailable: {exc.reason}") from exc

        try:
            with wave.open(str(output_path), "rb") as wav_file:
                duration_seconds = wav_file.getnframes() / wav_file.getframerate()
        except (wave.Error, OSError) as exc:
            raise OSError("Navoiy TTS service returned an invalid WAV file") from exc

        return SpeechSynthesisResult(
            audio_path=str(output_path),
            format="wav",
            language=language,
            duration_seconds=duration_seconds,
        )

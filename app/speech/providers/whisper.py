from pathlib import Path

from faster_whisper import WhisperModel

from app.speech.base import SpeechToText
from app.speech.models import TranscriptionResult


class FasterWhisperSpeechToText(SpeechToText):
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: Path, language: str = "uz") -> TranscriptionResult:
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_seconds=float(info.duration),
        )


    def transcribe_pcm16(
        self,
        audio,
        sample_rate: int = 8000,
        language: str = "uz",
    ) -> TranscriptionResult:
        if sample_rate != 8000:
            raise ValueError("Realtime STT currently expects 8000 Hz PCM")
        segments, info = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_seconds=float(info.duration),
        )

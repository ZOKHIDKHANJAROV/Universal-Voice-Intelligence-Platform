from pathlib import Path

import numpy as np

from app.audio.resample import resample
from app.speech.base import SpeechToText
from app.speech.models import TranscriptionResult

WHISPER_SAMPLE_RATE = 16000


class FasterWhisperSpeechToText(SpeechToText):
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        initial_prompt: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        self._beam_size = beam_size
        self._initial_prompt = initial_prompt or None

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        # Uploaded files may contain long silences, so keep Silero VAD here.
        return self._run(str(audio_path), language, vad_filter=True)

    def transcribe_pcm16(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        allowed_languages: tuple[str, ...] = (),
    ) -> TranscriptionResult:
        # faster-whisper treats every ndarray as 16 kHz; 8 kHz telephony audio
        # passed as-is would be heard at double speed.
        audio16 = resample(audio, sample_rate, WHISPER_SAMPLE_RATE)
        # The realtime bridge already trimmed the utterance with webrtcvad.
        return self._run(audio16, language, vad_filter=False, allowed_languages=allowed_languages)

    def _run(
        self,
        audio,
        language: str | None,
        vad_filter: bool,
        allowed_languages: tuple[str, ...] = (),
    ) -> TranscriptionResult:
        segments, info = self._transcribe(audio, language, vad_filter)

        # Language detection runs eagerly; decoding only starts when segments are
        # iterated. If Whisper picked a language the caller cannot speak (Uzbek is
        # often detected as Kazakh or Turkish), redo it with the best allowed one.
        if language is None and allowed_languages and info.language not in allowed_languages:
            probabilities = dict(info.all_language_probs or ())
            fallback = max(allowed_languages, key=lambda code: probabilities.get(code, 0.0))
            segments, info = self._transcribe(audio, fallback, vad_filter)

        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_seconds=float(info.duration),
        )

    def _transcribe(self, audio, language: str | None, vad_filter: bool):
        return self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=self._beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt,
        )

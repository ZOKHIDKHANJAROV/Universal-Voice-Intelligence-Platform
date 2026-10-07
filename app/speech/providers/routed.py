from pathlib import Path

import numpy as np

from app.audio.resample import resample
from app.speech.base import SpeechToText
from app.speech.models import TranscriptionResult
from app.speech.providers.whisper import WHISPER_SAMPLE_RATE, FasterWhisperSpeechToText


class LanguageRoutedSpeechToText(SpeechToText):
    """Detect the language with a general model, transcribe with a specialist.

    Uzbek fine-tunes transcribe Uzbek far better than stock Whisper but lose
    Russian entirely and cannot be trusted to detect the language, so the
    general model decides the language and handles everything that has no
    specialist.
    """

    def __init__(
        self,
        general: FasterWhisperSpeechToText,
        specialists: dict[str, SpeechToText],
    ) -> None:
        self._general = general
        self._specialists = specialists

    def _model_for(self, language: str | None) -> SpeechToText:
        return self._specialists.get(language, self._general) if language else self._general

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        return self._model_for(language).transcribe(audio_path, language=language)

    def transcribe_pcm16(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        allowed_languages: tuple[str, ...] = (),
    ) -> TranscriptionResult:
        audio16 = resample(audio, sample_rate, WHISPER_SAMPLE_RATE)
        probability = None
        if language is None:
            candidates = allowed_languages or tuple(self._specialists)
            language, probability = self._general.detect_language(audio16, candidates)

        result = self._model_for(language).transcribe_pcm16(
            audio16, WHISPER_SAMPLE_RATE, language=language
        )
        if probability is None:
            return result
        return result.model_copy(update={"language_probability": probability})

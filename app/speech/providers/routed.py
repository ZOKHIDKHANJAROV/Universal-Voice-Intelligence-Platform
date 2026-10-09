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

    Whisper's language detector barely knows Uzbek: on short phone phrases it
    guesses Indonesian, English or Dutch, and Russian then wins with a score
    of 0.03-0.3. Below ``sure_probability`` the models' confidence in their
    own text decides instead: the Uzbek fine-tune is near-certain on Uzbek
    (avg logprob ~ -0.05) and unsure on Russian (~ -0.6), stock Whisper the
    reverse. A specialist at or above ``accept_logprob`` wins outright;
    otherwise every candidate transcribes and the surest one wins. On 2.5 s
    phone-quality phrases this picks the right language for 96% of Uzbek
    clips (46% by detector alone) and 98% of Russian ones (100%).
    """

    def __init__(
        self,
        general: FasterWhisperSpeechToText,
        specialists: dict[str, SpeechToText],
        sure_probability: float = 0.9,
        accept_logprob: float = -0.25,
        min_logprob: float = -1.0,
        alone_min_logprob: float = -0.5,
    ) -> None:
        self._general = general
        self._specialists = specialists
        self._sure_probability = sure_probability
        self._accept_logprob = accept_logprob
        # Below this the text is a guess at noise ("puf", "siz" from a cough).
        self._min_logprob = min_logprob
        # When the general model heard no speech at all, a specialist's text
        # needs this much: the Uzbek fine-tune turns background noise into
        # "yigʻlab yubordim" at -0.51..-0.66, while 98% of short Uzbek phrases
        # in the evaluation clear -0.5.
        self._alone_min_logprob = alone_min_logprob

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
        if language is not None:
            return self._model_for(language).transcribe_pcm16(
                audio16, WHISPER_SAMPLE_RATE, language=language
            )

        candidates = allowed_languages or tuple(self._specialists)
        language, probability = self._general.detect_language(audio16, candidates)
        if probability >= self._sure_probability:
            result = self._model_for(language).transcribe_pcm16(
                audio16, WHISPER_SAMPLE_RATE, language=language
            )
            return result.model_copy(update={"language_probability": probability})

        # Specialists first: a confident one saves decoding with the large model.
        order = sorted(candidates, key=lambda c: c not in self._specialists)
        results = []
        for candidate in order:
            result = self._model_for(candidate).transcribe_pcm16(
                audio16, WHISPER_SAMPLE_RATE, language=candidate
            )
            results.append(result)
            if (
                candidate in self._specialists
                and result.text
                and result.avg_logprob is not None
                and result.avg_logprob >= self._accept_logprob
            ):
                break
        general_heard_speech = any(
            r.text for candidate, r in zip(order, results) if candidate not in self._specialists
        )
        general_asked = len(results) == len(order)
        floor = self._min_logprob
        if general_asked and not general_heard_speech:
            floor = max(floor, self._alone_min_logprob)
        heard = [
            r for r in results
            if r.text and r.avg_logprob is not None and r.avg_logprob >= floor
        ]
        if not heard:
            # Nothing believable: report the detector's pick with no text.
            fallback = next((r for r in results if r.language == language), results[0])
            return fallback.model_copy(update={"text": "", "language_probability": probability})
        best = max(heard, key=lambda r: r.avg_logprob)
        # Report how sure the decision is rather than the detector's weak guess.
        return best.model_copy(update={"language_probability": float(np.exp(best.avg_logprob))})

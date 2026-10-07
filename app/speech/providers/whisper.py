from pathlib import Path

import numpy as np

from app.audio.resample import resample
from app.speech.base import SpeechToText
from app.speech.models import TranscriptionResult

WHISPER_SAMPLE_RATE = 16000

# Whisper's language ID barely knows Uzbek: on real Uzbek calls large-v3 gives
# "uz" ~0.0 and answers Azerbaijani, Kazakh or Turkish instead. Probability on
# a related language therefore counts toward the allowed language of its family.
LANGUAGE_FAMILIES = {
    "uz": ("az", "kk", "tr", "tk", "tt", "ba"),
    "ru": ("uk", "be"),
}


def _family_score(code: str, probabilities: dict[str, float]) -> float:
    relatives = LANGUAGE_FAMILIES.get(code, ())
    return probabilities.get(code, 0.0) + sum(probabilities.get(r, 0.0) for r in relatives)


def pick_language(probabilities: dict[str, float], allowed: tuple[str, ...]) -> tuple[str, float]:
    """Best allowed language by family score, with that score as its probability."""
    scores = {code: _family_score(code, probabilities) for code in allowed}
    best = max(allowed, key=scores.__getitem__)
    return best, min(1.0, scores[best])


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

    def detect_language(self, audio16: np.ndarray, allowed: tuple[str, ...]) -> tuple[str, float]:
        """Pick one of ``allowed`` for 16 kHz audio without transcribing it."""
        _, _, probabilities = self._model.detect_language(audio16)
        return pick_language(dict(probabilities), allowed)

    def _run(
        self,
        audio,
        language: str | None,
        vad_filter: bool,
        allowed_languages: tuple[str, ...] = (),
    ) -> TranscriptionResult:
        segments, info = self._transcribe(audio, language, vad_filter)
        language_probability = float(info.language_probability)

        # Language detection runs eagerly; decoding only starts when segments are
        # iterated. If Whisper picked a language the caller cannot speak, redo it
        # with the allowed language whose family got the most probability.
        if language is None and allowed_languages and info.language not in allowed_languages:
            fallback, language_probability = pick_language(
                dict(info.all_language_probs or ()), allowed_languages
            )
            # A forced language reports 1.0; keep what detection actually thought.
            segments, info = self._transcribe(audio, fallback, vad_filter)

        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=language_probability,
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

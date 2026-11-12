import wave
from pathlib import Path

from app.tts.base import TextToSpeech
from app.tts.models import SpeechSynthesisResult


class VoskTextToSpeech(TextToSpeech):
    """Offline Russian voices from vosk-tts (Apache-2.0), several speakers in one model.

    Needs ``pip install vosk-tts`` and a model such as vosk-model-tts-ru-0.10-multi
    from https://alphacephei.com/vosk/models.
    """

    def __init__(self, model_path: str, speaker_id: int = 0, speech_rate: float | None = None) -> None:
        if not model_path:
            raise ValueError("TTS_VOSK_MODEL_PATH is not set")
        self._model_path = model_path
        self._speaker_id = speaker_id
        self._speech_rate = speech_rate
        self._synth = None

    def _engine(self):
        if self._synth is None:
            # Imported here: only the Russian phone prompts need it.
            from vosk_tts import Model, Synth

            self._synth = Synth(Model(model_path=self._model_path))
        return self._synth

    def synthesize(self, text: str, output_path: Path, language: str = "ru") -> SpeechSynthesisResult:
        if language != "ru":
            raise ValueError("vosk-tts voices are Russian only")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine().synth(text, str(output_path), speaker_id=self._speaker_id, speech_rate=self._speech_rate)
        with wave.open(str(output_path), "rb") as wav_file:
            duration_seconds = wav_file.getnframes() / wav_file.getframerate()
        return SpeechSynthesisResult(
            audio_path=str(output_path),
            format="wav",
            language=language,
            duration_seconds=duration_seconds,
        )

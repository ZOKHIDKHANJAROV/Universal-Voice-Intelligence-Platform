from pathlib import Path

from app.core.config import get_settings
from app.intent.service import get_intent_service
from app.speech.service import transcribe
from app.tts.service import synthesize
from app.voice.models import VoicePipelineResult


class VoicePipelineService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._intent = get_intent_service()

    def process(self, audio_path: Path, call_id: str, language: str = "uz") -> VoicePipelineResult:
        transcription = transcribe(audio_path, language=language)
        intent = self._intent.resolve(transcription.text)

        scenario = None
        if intent.scenario_id:
            from app.services.scenario_service import ScenarioService
            scenario = ScenarioService().get_scenario(intent.scenario_id)

        if scenario is None:
            response_text = "Kechirasiz, muammoingizni aniqlay olmadim. Operator bilan bog'lanish uchun 0 ni bosing."
        else:
            response_text = scenario.steps[0].message if scenario.steps else "Operator bilan bog'lanish uchun 0 ni bosing."

        output_path = self._settings.tts_output_dir / f"{call_id}.wav"
        tts_result = synthesize(response_text, output_path, language=language)

        return VoicePipelineResult(
            transcription=transcription.text,
            language=transcription.language,
            intent=intent,
            response_text=response_text,
            audio_path=tts_result.audio_path,
            audio_format=tts_result.format,
            duration_seconds=tts_result.duration_seconds,
        )

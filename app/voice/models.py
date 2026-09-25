from pydantic import BaseModel, Field

from app.intent.models import IntentResult


class VoicePipelineResult(BaseModel):
    transcription: str
    language: str
    intent: IntentResult
    response_text: str
    audio_path: str
    audio_format: str
    duration_seconds: float = Field(ge=0.0)

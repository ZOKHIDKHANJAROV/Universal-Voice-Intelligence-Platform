from pydantic import BaseModel, Field


class SpeechSynthesisResult(BaseModel):
    audio_path: str
    format: str
    language: str
    duration_seconds: float = Field(ge=0.0)

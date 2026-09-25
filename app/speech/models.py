from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    text: str
    language: str
    language_probability: float = Field(ge=0.0, le=1.0)
    duration_seconds: float = Field(ge=0.0)

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    text: str
    language: str
    language_probability: float = Field(ge=0.0, le=1.0)
    duration_seconds: float = Field(ge=0.0)
    # Mean token log-probability of the decoded text: how sure the model is of
    # what it heard (0 = certain). None when nothing was decoded.
    avg_logprob: float | None = None

from pydantic import BaseModel, Field


class IntentResolveRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = None


class IntentResult(BaseModel):
    intent: str | None
    scenario_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    source: str

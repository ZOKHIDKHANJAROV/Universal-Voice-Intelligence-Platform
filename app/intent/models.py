from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    intent: str | None
    scenario_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    source: str

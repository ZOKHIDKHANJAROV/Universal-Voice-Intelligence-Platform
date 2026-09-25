from pydantic import BaseModel, Field


class ScenarioStep(BaseModel):
    id: str
    message: str
    next_scenario: str | None = None


class Scenario(BaseModel):
    id: str
    title: str
    description: str
    language: str = "uz"
    keywords: list[str] = Field(default_factory=list)
    steps: list[ScenarioStep]
    enabled: bool = True


class ScenarioResolveRequest(BaseModel):
    text: str = Field(min_length=1)


class ScenarioResolveResponse(BaseModel):
    scenario_id: str | None
    matched: bool
    confidence: float

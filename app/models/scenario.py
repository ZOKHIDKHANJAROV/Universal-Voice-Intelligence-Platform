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
    # Words that state the problem; a scenario needs at least one to match.
    # A trailing "*" marks a stem: "chiqma*" covers chiqmayapti, chiqmadi, ...
    keywords: list[str] = Field(default_factory=list)
    # Words naming the subject (water, money, machine). They raise confidence
    # but never select a scenario on their own: "suv" alone is not a complaint.
    context_keywords: list[str] = Field(default_factory=list)
    steps: list[ScenarioStep]
    enabled: bool = True


class ScenarioResolveRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = None


class ScenarioResolveResponse(BaseModel):
    scenario_id: str | None
    matched: bool
    confidence: float

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
    # Problem words too general to trust alone ("ishlama*", "течет": a lift or
    # the internet can be broken too). They count only next to a context keyword.
    weak_keywords: list[str] = Field(default_factory=list)
    # Words that put the problem somewhere else (a cash machine, a flat's
    # kitchen, hot water); any of them rules the scenario out.
    exclude_keywords: list[str] = Field(default_factory=list)
    # Beats matching scenarios of lower priority whatever their score: "it took
    # my money and gave no water" is a payment problem, not a water one.
    priority: int = 0
    # "transfer": hand the call to an operator instead of answering.
    action: str | None = None
    steps: list[ScenarioStep]
    enabled: bool = True


class ScenarioUpdate(BaseModel):
    """Fields the web console may edit; omitted fields stay as they are."""

    keywords: list[str] | None = None
    context_keywords: list[str] | None = None
    message: str | None = Field(default=None, min_length=1, max_length=1000)
    enabled: bool | None = None


class ScenarioResolveRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = None


class ScenarioResolveResponse(BaseModel):
    scenario_id: str | None
    matched: bool
    confidence: float

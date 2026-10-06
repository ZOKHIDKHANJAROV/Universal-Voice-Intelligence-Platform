from fastapi import APIRouter, HTTPException

from app.models.scenario import (
    Scenario,
    ScenarioResolveRequest,
    ScenarioResolveResponse,
)
from app.services.scenario_service import get_scenario_service

router = APIRouter(prefix="/api/v1/scenarios", tags=["scenarios"])


@router.get("", response_model=list[Scenario])
def list_scenarios() -> list[Scenario]:
    return get_scenario_service().list_scenarios()


@router.get("/{scenario_id}", response_model=Scenario)
def get_scenario(scenario_id: str) -> Scenario:
    scenario = get_scenario_service().get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


@router.post("/resolve", response_model=ScenarioResolveResponse)
def resolve_scenario(payload: ScenarioResolveRequest) -> ScenarioResolveResponse:
    scenario, confidence = get_scenario_service().resolve(payload.text, payload.language)
    return ScenarioResolveResponse(
        scenario_id=scenario.id if scenario else None,
        matched=scenario is not None,
        confidence=confidence,
    )

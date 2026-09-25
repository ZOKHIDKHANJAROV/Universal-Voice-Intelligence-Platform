from fastapi import APIRouter, HTTPException

from app.models.scenario import (
    Scenario,
    ScenarioResolveRequest,
    ScenarioResolveResponse,
)
from app.services.scenario_service import ScenarioService

router = APIRouter(prefix="/api/v1/scenarios", tags=["scenarios"])
service = ScenarioService()


@router.get("", response_model=list[Scenario])
def list_scenarios() -> list[Scenario]:
    return service.list_scenarios()


@router.get("/{scenario_id}", response_model=Scenario)
def get_scenario(scenario_id: str) -> Scenario:
    scenario = service.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


@router.post("/resolve", response_model=ScenarioResolveResponse)
def resolve_scenario(payload: ScenarioResolveRequest) -> ScenarioResolveResponse:
    scenario, confidence = service.resolve(payload.text)
    return ScenarioResolveResponse(
        scenario_id=scenario.id if scenario else None,
        matched=scenario is not None,
        confidence=confidence,
    )

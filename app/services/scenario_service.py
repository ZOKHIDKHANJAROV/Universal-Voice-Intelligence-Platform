import json
from pathlib import Path

from app.models.scenario import Scenario


class ScenarioService:
    def __init__(self, data_path: Path | None = None) -> None:
        self._data_path = data_path or (
            Path(__file__).resolve().parents[1] / "data" / "scenarios.json"
        )
        self._scenarios = self._load()

    def _load(self) -> list[Scenario]:
        with self._data_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return [Scenario.model_validate(item) for item in raw]

    def list_scenarios(self) -> list[Scenario]:
        return [scenario for scenario in self._scenarios if scenario.enabled]

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        for scenario in self._scenarios:
            if scenario.enabled and scenario.id == scenario_id:
                return scenario
        return None

    def resolve(self, text: str) -> tuple[Scenario | None, float]:
        normalized = text.casefold().strip()
        best_scenario: Scenario | None = None
        best_score = 0

        for scenario in self.list_scenarios():
            matches = sum(
                1 for keyword in scenario.keywords if keyword.casefold() in normalized
            )
            if matches > best_score:
                best_score = matches
                best_scenario = scenario

        if best_scenario is None or best_score == 0:
            return None, 0.0

        confidence = min(1.0, 0.5 + 0.15 * best_score)
        return best_scenario, round(confidence, 2)

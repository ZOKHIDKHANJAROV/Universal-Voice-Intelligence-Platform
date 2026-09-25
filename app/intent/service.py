import json
import urllib.error
import urllib.request
from functools import lru_cache

from app.core.config import get_settings
from app.intent.models import IntentResult
from app.services.scenario_service import ScenarioService


class IntentService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._scenarios = ScenarioService()

    def resolve(self, text: str) -> IntentResult:
        scenario = self._scenarios.resolve(text)[0]
        if scenario is None:
            return IntentResult(intent=None, scenario_id=None, confidence=0.0, source="scenario")

        if not self._settings.llm_enabled:
            return IntentResult(
                intent=scenario.id,
                scenario_id=scenario.id,
                confidence=0.8,
                source="scenario",
            )

        try:
            result = self._ollama(text)
            if result.scenario_id and self._scenarios.get_scenario(result.scenario_id):
                return result
        except (OSError, ValueError, urllib.error.URLError):
            pass

        return IntentResult(
            intent=scenario.id,
            scenario_id=scenario.id,
            confidence=0.8,
            source="scenario-fallback",
        )

    def _ollama(self, text: str) -> IntentResult:
        scenarios = [s.model_dump(include={"id", "title", "description", "keywords"}) for s in self._scenarios.list_scenarios()]
        prompt = (
            "Select exactly one scenario from the supplied list. Do not invent a scenario. "
            "Return JSON only with intent, scenario_id, confidence. "
            f"Scenarios: {json.dumps(scenarios, ensure_ascii=False)}\n"
            f"User text: {text}"
        )
        payload = json.dumps({
            "model": self._settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }).encode()
        request = urllib.request.Request(
            f"{self._settings.llm_base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._settings.llm_timeout_seconds) as response:
            data = json.loads(response.read().decode())
        parsed = json.loads(data["response"])
        return IntentResult(
            intent=parsed.get("intent"),
            scenario_id=parsed.get("scenario_id"),
            confidence=float(parsed.get("confidence", 0.0)),
            source="ollama",
        )


@lru_cache(maxsize=1)
def get_intent_service() -> IntentService:
    return IntentService()

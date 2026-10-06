import json
import logging
import urllib.error
import urllib.request
from functools import lru_cache

from app.core.config import get_settings
from app.intent.models import IntentResult
from app.services.scenario_service import get_scenario_service

LOGGER = logging.getLogger("univoice.intent")


class IntentService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._scenarios = get_scenario_service()

    def resolve(self, text: str, language: str | None = None) -> IntentResult:
        scenario, confidence = self._scenarios.resolve(text, language)
        keyword_result = IntentResult(
            intent=scenario.id if scenario else None,
            scenario_id=scenario.id if scenario else None,
            confidence=confidence,
            source="scenario",
        )

        # The LLM is only worth its latency when keywords are unsure.
        if self._settings.llm_enabled and confidence < self._settings.llm_skip_confidence:
            try:
                result = self._ollama(text, language)
                if result.scenario_id and self._scenarios.get_scenario(result.scenario_id):
                    return self._apply_threshold(result)
            except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
                LOGGER.warning("LLM intent failed, using keywords: %s", exc)
            keyword_result = keyword_result.model_copy(update={"source": "scenario-fallback"})

        return self._apply_threshold(keyword_result)

    def _apply_threshold(self, result: IntentResult) -> IntentResult:
        # Below the threshold it is safer to hand the caller to an operator than
        # to read out the wrong instructions.
        if result.scenario_id and result.confidence < self._settings.intent_min_confidence:
            return IntentResult(
                intent=None,
                scenario_id=None,
                confidence=result.confidence,
                source=result.source,
            )
        return result

    def _ollama(self, text: str, language: str | None) -> IntentResult:
        scenarios = [
            s.model_dump(include={"id", "title", "description", "keywords"})
            for s in self._scenarios.list_scenarios()
            if language is None or s.language == language
        ]
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
            "options": {"num_gpu": self._settings.llm_num_gpu},
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
            confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.0)))),
            source="ollama",
        )


@lru_cache(maxsize=1)
def get_intent_service() -> IntentService:
    return IntentService()

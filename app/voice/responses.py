from app.services.scenario_service import get_scenario_service
from app.tts.prompts import prompt_text


def build_response(scenario_id: str | None, caller_language: str) -> tuple[str, str]:
    """Return (text, language) the bot should say for a resolved intent.

    A matched scenario is answered in its own language; the fallback uses the
    language the caller spoke.
    """
    scenario = get_scenario_service().get_scenario(scenario_id) if scenario_id else None
    if scenario is None:
        return prompt_text("not_understood", caller_language)
    if scenario.steps:
        return scenario.steps[0].message, scenario.language
    return prompt_text("operator", scenario.language)

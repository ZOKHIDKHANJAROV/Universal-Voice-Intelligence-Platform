"""Fixed phrases the voice bot can say: system prompts plus scenario messages."""

import json
from functools import lru_cache
from pathlib import Path

from app.services.scenario_service import get_scenario_service

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts.json"


@lru_cache(maxsize=1)
def get_prompts() -> dict[str, dict[str, str]]:
    with PROMPTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def prompt_text(key: str, language: str) -> tuple[str, str]:
    """Return (text, language), falling back to any language the prompt has."""
    variants = get_prompts()[key]
    if language in variants:
        return variants[language], language
    fallback_language, text = next(iter(variants.items()))
    return text, fallback_language


def all_phrases() -> list[tuple[str, str]]:
    """Every (text, language) pair that can be played on a call, deduplicated."""
    phrases: dict[tuple[str, str], None] = {}
    for variants in get_prompts().values():
        for language, text in variants.items():
            phrases[(text, language)] = None
    for scenario in get_scenario_service().list_scenarios():
        for step in scenario.steps:
            phrases[(step.message, scenario.language)] = None
    return list(phrases)

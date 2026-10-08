import json
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.intent.service import IntentService
from app.services.scenario_service import ScenarioService, keyword_matches, normalize_text, to_uzbek_latin
from app.voice.responses import build_response

CASES = json.loads((Path(__file__).parent / "data" / "intent_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scenarios() -> ScenarioService:
    return ScenarioService()


@pytest.mark.parametrize(("text", "language", "expected"), CASES["complaints"])
def test_complaints_reach_their_scenario(scenarios, text, language, expected) -> None:
    scenario, confidence = scenarios.resolve(text, language)
    assert scenario is not None and scenario.id == expected
    assert confidence >= 0.65


@pytest.mark.parametrize(("text", "language"), CASES["not_complaints"])
def test_mentioning_water_or_money_is_not_a_complaint(scenarios, text, language) -> None:
    # Before problem keywords were required, 13 of these 15 matched a scenario
    # and 64 of 380 real FLEURS / podcast sentences did.
    assert scenarios.resolve(text, language) == (None, 0.0)


@pytest.mark.parametrize("text", ["Привет, как дела", "Salom, yaxshimisiz", "Hello"])
def test_unrelated_text_does_not_match(scenarios, text) -> None:
    assert scenarios.resolve(text) == (None, 0.0)


def test_falls_back_to_other_language_when_needed(scenarios) -> None:
    # Whisper said "ru" but the words are Uzbek: still route somewhere sensible.
    scenario, _ = scenarios.resolve("Pulimni qaytarib bermadi", "ru")
    assert scenario is not None and scenario.id == "vending_payment_uz"


def test_normalizes_apostrophes_and_yo() -> None:
    assert normalize_text("Toʻxtab QOLDI!") == "to'xtab qoldi"
    assert normalize_text("Течёт") == "течет"


def test_normalizes_turkish_letters_whisper_leaks_into_uzbek() -> None:
    # Real large-v3 output for "Internet ishlamayapti" forced to Uzbek.
    assert normalize_text("İnternet işlemeyabdi") == "internet ishlemeyabdi"


def test_uzbek_cyrillic_initial_ye() -> None:
    assert to_uzbek_latin("Пулимни еб қўйди") == "pulimni yeb qo'ydi"


def test_short_keywords_need_exact_tokens() -> None:
    assert keyword_matches("не идет", ["не", "идет"])
    assert not keyword_matches("не идет", ["нет", "идет"])


def test_stems_are_explicit() -> None:
    assert keyword_matches("chiqma*", ["chiqmayapti"])
    assert keyword_matches("chiqma*", ["chiqmavotti"])
    assert not keyword_matches("chiqma*", ["chiqyapti"])  # comes out: no complaint


def test_stems_skip_first_and_second_person() -> None:
    assert keyword_matches("ishlama*", ["ishlamayapti"])
    assert not keyword_matches("ishlama*", ["ishlamayman"])
    assert not keyword_matches("chiqma*", ["chiqmadingizmi"])


def test_short_plain_keywords_take_only_short_endings() -> None:
    assert keyword_matches("pul", ["pulim"])
    assert not keyword_matches("loy", ["loyiha"])
    assert keyword_matches("вода", ["воды"])
    assert not keyword_matches("вода", ["водитель"])


def test_no_fuzzy_match_between_short_lookalikes() -> None:
    # oqib (flowing) vs o'qib (reading); oqyapti (flows) vs oqmayapti (does not).
    assert not keyword_matches("oqib", ["o'qib"])
    assert not keyword_matches("oqyapti", ["oqmayapti"])


def test_intent_reports_keyword_confidence() -> None:
    # One problem keyword (0.65) plus one context keyword (+0.05).
    result = IntentService().resolve("Pulimni yeb qo'ydi", "uz")
    assert result.scenario_id == "vending_payment_uz"
    assert result.confidence == 0.7


def test_intent_below_threshold_hands_off(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "intent_min_confidence", 0.9)
    result = IntentService().resolve("Suv chiqmadi", "uz")
    assert result.scenario_id is None
    assert result.confidence == 0.7


def test_response_uses_scenario_language() -> None:
    text, language = build_response("vending_no_water_ru", "uz")
    assert language == "ru"
    assert text.startswith("Приносим извинения")


def test_fallback_response_uses_caller_language() -> None:
    assert build_response(None, "ru")[1] == "ru"
    assert build_response(None, "uz")[1] == "uz"

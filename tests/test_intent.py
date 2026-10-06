import pytest

from app.core.config import get_settings
from app.intent.service import IntentService
from app.services.scenario_service import ScenarioService, keyword_matches, normalize_text
from app.voice.responses import build_response


@pytest.fixture(scope="module")
def scenarios() -> ScenarioService:
    return ScenarioService()


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        # Inflected forms that plain substring matching missed or misrouted.
        ("Suv chiqmadi, tugmani bosdim", "uz", "vending_no_water_uz"),
        ("apparatdan suvi chiqmayobdi", "uz", "vending_no_water_uz"),
        ("Pulimni yeb qoʻydi", "uz", "vending_payment_uz"),
        ("Suvning ta’mi yomon", "uz", "vending_taste_uz"),
        # Uzbek transcribed in Cyrillic, as Whisper often does.
        ("Сув чиқмаяпти", "uz", "vending_no_water_uz"),
        ("Пулимни еб қўйди", "uz", "vending_payment_uz"),
        ("Воды нет, ничего не льётся", "ru", "vending_no_water_ru"),
        ("Аппарат сломался, всё течёт", "ru", "vending_leak_ru"),
        ("Автомат съел мои деньги", "ru", "vending_payment_ru"),
        ("Вода горькая и мутная", "ru", "vending_taste_ru"),
    ],
)
def test_resolves_inflected_phrasings(scenarios, text, language, expected) -> None:
    scenario, confidence = scenarios.resolve(text, language)
    assert scenario is not None and scenario.id == expected
    assert confidence >= 0.65


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


def test_short_keywords_need_exact_tokens() -> None:
    assert keyword_matches("не идет", ["не", "идет"])
    assert not keyword_matches("не идет", ["нет", "идет"])


def test_intent_reports_keyword_confidence() -> None:
    result = IntentService().resolve("Pulimni yeb qo'ydi", "uz")
    assert result.scenario_id == "vending_payment_uz"
    assert result.confidence == 0.95


def test_intent_below_threshold_hands_off(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "intent_min_confidence", 0.9)
    result = IntentService().resolve("Suv chiqmadi", "uz")
    assert result.scenario_id is None
    assert result.confidence == 0.65


def test_response_uses_scenario_language() -> None:
    text, language = build_response("vending_no_water_ru", "uz")
    assert language == "ru"
    assert text.startswith("Приносим извинения")


def test_fallback_response_uses_caller_language() -> None:
    assert build_response(None, "ru")[1] == "ru"
    assert build_response(None, "uz")[1] == "uz"

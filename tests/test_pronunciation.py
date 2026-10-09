from app.tts.prompt_cache import PromptAudioCache
from app.tts.pronunciation import spoken_text


def test_operator_is_spelled_for_navoiy_with_suffixes_and_case() -> None:
    assert spoken_text("Operatorga ulayapman.", "uz") == "Aperatorga ulayapman."
    assert spoken_text("barcha operatorlar band", "uz") == "barcha aperatorlar band"


def test_other_languages_and_words_are_untouched() -> None:
    assert spoken_text("Соединяю с оператором.", "ru") == "Соединяю с оператором."
    assert spoken_text("Kooperativ", "uz") == "Kooperativ"


def test_lexicon_is_part_of_the_cache_key(tmp_path) -> None:
    cache = PromptAudioCache(tmp_path)
    # Same spoken text, same audio file.
    assert cache.path_for("Operatorga", "uz") == cache.path_for("Aperatorga", "uz")
    assert cache.path_for("Suv", "uz") != cache.path_for("Suv", "ru")

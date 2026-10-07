import json
import math
import re
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz

from app.models.scenario import Scenario

# Uzbek Latin uses several look-alike apostrophes (o', o‘, oʻ, o`); STT output
# and hand-written keywords rarely agree on which one. Whisper also leaks
# Turkish letters into Uzbek transcripts ("işlemeyabdi"), so map those to the
# Uzbek Latin spelling.
_APOSTROPHES = str.maketrans(
    {c: "'" for c in "‘’ʻʼ`´"} | {"ş": "sh", "ç": "ch", "ı": "i", "ğ": "g'", "ö": "o'", "ü": "u"}
    # casefold() turns Turkish "İ" into "i" plus a combining dot.
    | {"\u0307": ""}
)
# Whisper writes Uzbek in Cyrillic about as often as in Latin, while scenario
# keywords are Latin. Used only when matching Uzbek scenarios.
_UZ_CYRILLIC_TO_LATIN = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "ғ": "g'", "д": "d", "е": "e",
    "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "қ": "q",
    "л": "l", "м": "m", "н": "n", "о": "o", "ў": "o'", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ҳ": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "ъ": "'", "ь": "", "ы": "i", "э": "e", "ю": "yu",
    "я": "ya",
    # Kazakh letters: Whisper often writes Uzbek speech in Kazakh Cyrillic.
    "ә": "a", "ө": "o'", "ұ": "u", "ү": "u", "ң": "ng", "і": "i", "һ": "h",
})
_NON_WORD = re.compile(r"[^\w']+")
_FUZZY_MIN_RATIO = 85


def normalize_text(text: str) -> str:
    text = text.casefold().translate(_APOSTROPHES).replace("ё", "е")
    return " ".join(_NON_WORD.sub(" ", text).split())


def to_uzbek_latin(text: str) -> str:
    """Normalize and transliterate Uzbek Cyrillic to Latin."""
    return normalize_text(text).translate(_UZ_CYRILLIC_TO_LATIN)


def _common_prefix(a: str, b: str) -> int:
    size = 0
    for left, right in zip(a, b):
        if left != right:
            break
        size += 1
    return size


def _token_matches(keyword: str, token: str) -> bool:
    """Match one keyword word against one spoken word, tolerating inflection.

    Uzbek is agglutinative (chiqmayapti / chiqmadi / chiqmayobdi) and Russian
    changes endings (вода / воду / воды), so exact substrings miss most real
    phrasings. A shared stem plus a typo-tolerant ratio covers both cheaply.
    """
    if len(keyword) <= 2:
        return token == keyword
    if token.startswith(keyword):
        return True
    prefix = _common_prefix(keyword, token)
    if len(keyword) <= 5:
        if prefix >= max(3, len(keyword) - 1):
            return True
    elif prefix >= max(5, math.ceil(len(keyword) * 0.55)):
        return True
    return fuzz.ratio(keyword, token) >= _FUZZY_MIN_RATIO


def keyword_matches(keyword: str, tokens: list[str]) -> bool:
    """Every word of a (possibly multi-word) keyword must match a distinct token."""
    remaining = list(tokens)
    for word in normalize_text(keyword).split():
        for index, token in enumerate(remaining):
            if _token_matches(word, token):
                del remaining[index]
                break
        else:
            return False
    return True


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

    def resolve(self, text: str, language: str | None = None) -> tuple[Scenario | None, float]:
        """Pick the scenario with the most matching keywords.

        Scenarios in ``language`` are tried first; the rest only if none of them
        match, so a Russian caller is not routed to the Uzbek twin of a scenario.
        """
        normalized = normalize_text(text)
        tokens = normalized.split()
        latin_tokens = to_uzbek_latin(text).split()
        candidates = self.list_scenarios()
        if language:
            preferred = [s for s in candidates if s.language == language]
            others = [s for s in candidates if s.language != language]
            groups = [preferred, others]
        else:
            groups = [candidates]

        for group in groups:
            best_scenario, best_score = self._best_match(tokens, latin_tokens, group)
            if best_scenario is not None:
                confidence = min(1.0, 0.5 + 0.15 * best_score)
                return best_scenario, round(confidence, 2)
        return None, 0.0

    @staticmethod
    def _best_match(
        tokens: list[str], latin_tokens: list[str], scenarios: list[Scenario]
    ) -> tuple[Scenario | None, int]:
        best_scenario: Scenario | None = None
        best_score = 0
        for scenario in scenarios:
            words = latin_tokens if scenario.language == "uz" else tokens
            score = sum(1 for keyword in scenario.keywords if keyword_matches(keyword, words))
            if score > best_score:
                best_score = score
                best_scenario = scenario
        return best_scenario, best_score


@lru_cache(maxsize=1)
def get_scenario_service() -> ScenarioService:
    return ScenarioService()

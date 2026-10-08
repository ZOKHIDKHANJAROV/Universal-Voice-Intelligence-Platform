import json
import math
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz

from app.models.scenario import Scenario, ScenarioUpdate

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
# Uzbek Cyrillic "е" starts a word as "ye": еб -> yeb, ер -> yer.
_UZ_INITIAL_YE = re.compile(r"(?<!\w)е")
# Typo tolerance only for long words: on short ones a single edit is a different
# word (oqib "flowing" vs o'qib "reading") or a negation (oqyapti / oqmayapti).
_FUZZY_MIN_LENGTH = 7
# Complaints describe the machine in the third person (chiqmayapti, ishlamadi).
# First and second person forms of the same verbs are about the speaker:
# "chiqmayapman" (I am not going out), "ishlamayman" (I do not work).
_UZ_PERSONAL_ENDINGS = (
    "man", "manmi", "miz", "mizmi", "san", "sanmi", "siz", "sizmi",
    "dim", "dik", "ding", "dingiz", "dingizmi",
)
_FUZZY_MIN_RATIO = 90


def normalize_text(text: str) -> str:
    text = text.casefold().translate(_APOSTROPHES).replace("ё", "е")
    return " ".join(_NON_WORD.sub(" ", text).split())


def to_uzbek_latin(text: str) -> str:
    """Normalize and transliterate Uzbek Cyrillic to Latin."""
    return _UZ_INITIAL_YE.sub("ye", normalize_text(text)).translate(_UZ_CYRILLIC_TO_LATIN)


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
    changes endings (вода / воду / воды). Stems are marked explicitly with "*";
    plain keywords only accept a short ending, so "pul" matches pulim but not
    an unrelated longer word, and "loy" (mud) does not match loyiha (project).
    """
    if keyword.endswith("*"):
        stem = keyword[:-1]
        if not token.startswith(stem):
            return False
        is_uzbek_latin = stem.isascii()
        return not (is_uzbek_latin and token[len(stem):].endswith(_UZ_PERSONAL_ENDINGS))
    if token == keyword:
        return True
    if len(keyword) <= 2:
        # Particles like "не" must not match "нет".
        return False
    extra = len(token) - len(keyword)
    if len(keyword) <= 3:
        return token.startswith(keyword) and extra <= 2
    if len(keyword) <= 5:
        # вода -> воды / водой, but not водитель.
        return extra <= 3 and _common_prefix(keyword, token) >= len(keyword) - 1
    if token.startswith(keyword):
        return True
    if _common_prefix(keyword, token) >= max(5, math.ceil(len(keyword) * 0.55)):
        return True
    return len(keyword) >= _FUZZY_MIN_LENGTH and fuzz.ratio(keyword, token) >= _FUZZY_MIN_RATIO


def _bare_matches(bare: str, token: str) -> bool:
    """Strict match for a keyword with its apostrophe dropped: no stem guessing."""
    if bare.endswith("*"):
        return _token_matches(bare, token)
    return token.startswith(bare) and len(token) - len(bare) <= 2


def keyword_matches(keyword: str, tokens: list[str]) -> bool:
    """Every word of a (possibly multi-word) keyword must match a distinct token."""
    remaining = list(tokens)
    for raw in keyword.split():
        word = normalize_text(raw) + ("*" if raw.endswith("*") else "")
        # Uzbek Latin is often typed without apostrophes (qoydi for qo'ydi), so
        # a keyword's apostrophe is optional. The reverse is not: "oqib"
        # (flowing) must not match "o'qib" (reading).
        bare = word.replace("'", "")
        for index, token in enumerate(remaining):
            if _token_matches(word, token) or (bare != word and _bare_matches(bare, token)):
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
        self._lock = threading.Lock()

    def _load(self) -> list[Scenario]:
        with self._data_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return [Scenario.model_validate(item) for item in raw]

    def list_scenarios(self) -> list[Scenario]:
        return [scenario for scenario in self._scenarios if scenario.enabled]

    def all_scenarios(self) -> list[Scenario]:
        return list(self._scenarios)

    def update(self, scenario_id: str, changes: ScenarioUpdate) -> Scenario:
        """Apply console edits in memory and persist them to the JSON file.

        Everything that resolves intents shares this instance, so edits take
        effect on the next call without a restart.
        """
        with self._lock:
            index = next(
                (i for i, s in enumerate(self._scenarios) if s.id == scenario_id), None
            )
            if index is None:
                raise KeyError(scenario_id)
            scenario = self._scenarios[index]
            update: dict = {}
            for field in ("keywords", "context_keywords"):
                words = getattr(changes, field)
                if words is not None:
                    update[field] = [w.strip() for w in words if w.strip()]
            if changes.enabled is not None:
                update["enabled"] = changes.enabled
            if changes.message is not None:
                if not scenario.steps:
                    raise ValueError("Scenario has no step to hold a message")
                first = scenario.steps[0].model_copy(update={"message": changes.message.strip()})
                update["steps"] = [first, *scenario.steps[1:]]
            updated = scenario.model_copy(update=update)
            if updated.enabled and not updated.keywords:
                raise ValueError("An enabled scenario needs at least one problem keyword")
            self._scenarios[index] = updated
            self._save()
            return updated

    def _save(self) -> None:
        data = [s.model_dump() for s in self._scenarios]
        temporary = self._data_path.with_suffix(".tmp")
        temporary.write_text(_dump(data), encoding="utf-8")
        os.replace(temporary, self._data_path)

    def get_scenario(self, scenario_id: str) -> Scenario | None:
        for scenario in self._scenarios:
            if scenario.enabled and scenario.id == scenario_id:
                return scenario
        return None

    def resolve(self, text: str, language: str | None = None) -> tuple[Scenario | None, float]:
        """Pick the scenario whose problem keywords match best.

        A scenario needs at least one problem keyword; context keywords only add
        confidence and break ties. Scenarios in ``language`` are tried first and
        the rest only if none match, so a Russian caller is not routed to the
        Uzbek twin of a scenario.
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
            match = self._best_match(tokens, latin_tokens, group)
            if match is not None:
                scenario, problems, context = match
                confidence = min(1.0, 0.5 + 0.15 * problems + 0.05 * context)
                return scenario, round(confidence, 2)
        return None, 0.0

    @staticmethod
    def _best_match(
        tokens: list[str], latin_tokens: list[str], scenarios: list[Scenario]
    ) -> tuple[Scenario, int, int] | None:
        best: tuple[Scenario, int, int] | None = None
        best_score = 0.0
        for scenario in scenarios:
            words = latin_tokens if scenario.language == "uz" else tokens
            problems = sum(keyword_matches(k, words) for k in scenario.keywords)
            if not problems:
                continue
            context = sum(keyword_matches(k, words) for k in scenario.context_keywords)
            score = problems + 0.25 * context
            if score > best_score:
                best_score = score
                best = (scenario, problems, context)
        return best


def _dump(scenarios: list[dict]) -> str:
    """JSON with one keyword list per line, as the file is written by hand."""
    blocks = []
    for scenario in scenarios:
        block = json.dumps(scenario, ensure_ascii=False, indent=2)
        for key in ("keywords", "context_keywords"):
            expanded = json.dumps(scenario[key], ensure_ascii=False, indent=2).replace("\n", "\n  ")
            block = block.replace(f'"{key}": {expanded}', f'"{key}": {json.dumps(scenario[key], ensure_ascii=False)}')
        blocks.append("\n".join("  " + line for line in block.split("\n")))
    return "[\n" + ",\n".join(blocks) + "\n]\n"


@lru_cache(maxsize=1)
def get_scenario_service() -> ScenarioService:
    return ScenarioService()

"""Spelling fixes applied only when a phrase is synthesized.

Navoiy mispronounces some loanwords: "Operatorga" comes out as "apetarga".
Written the way Uzbeks say it, "aperatorga", it is heard correctly. The text
shown in the console and the call monitor keeps the normal spelling.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

LEXICON_PATH = Path(__file__).resolve().parents[1] / "data" / "pronunciation.json"


@lru_cache(maxsize=1)
def _patterns() -> dict[str, list[tuple[re.Pattern[str], str]]]:
    with LEXICON_PATH.open("r", encoding="utf-8") as file:
        lexicon: dict[str, dict[str, str]] = json.load(file)
    # Word starts only, so suffixes follow: operatorga, operatorlar.
    return {
        language: [(re.compile(rf"\b{re.escape(word)}", re.IGNORECASE), spoken)
                   for word, spoken in words.items()]
        for language, words in lexicon.items()
    }


def _same_case(spoken: str, written: str) -> str:
    return spoken[:1].upper() + spoken[1:] if written[:1].isupper() else spoken


def spoken_text(text: str, language: str) -> str:
    """The text as the TTS should read it."""
    for pattern, spoken in _patterns().get(language, []):
        text = pattern.sub(lambda match, s=spoken: _same_case(s, match.group(0)), text)
    return text

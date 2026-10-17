"""Knowledge base the maximum mode answers from, with BM25 search.

Entries are short texts an operator maintains in the admin panel. A new base
starts from the scenarios, so the LLM knows at least what the pre-recorded
answers say. Search is lexical (BM25 over normalized word stems): no model to
download, milliseconds per query, and it handles Uzbek Cyrillic/Latin and
word endings ("chiqmayapti", "chiqmadi") through stem truncation.
"""

import json
import logging
import math
import os
import threading
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.scenario_service import get_scenario_service, to_uzbek_latin

LOGGER = logging.getLogger("univoice.knowledge")

# Word stems: Uzbek and Russian inflect by suffix, so the first letters carry
# the meaning ("chiqmayapti" / "chiqmadi" -> "chiqm", "наливает" -> "налив").
STEM_LENGTH = 5
MIN_TOKEN_LENGTH = 3
_K1, _B = 1.5, 0.75
RELATIVE_MIN_SCORE = 0.5


class KnowledgeEntry(BaseModel):
    id: str
    title: str = Field(min_length=1, max_length=200)
    # "any" serves callers in every language.
    language: Literal["uz", "ru", "any"] = "any"
    text: str = Field(min_length=1, max_length=4000)
    # Extra words that should find this entry; searched, never shown to the LLM.
    keywords: str = Field(default="", max_length=2000)
    enabled: bool = True
    # "scenario:<id>" for entries imported from scenarios, else "manual".
    source: str = "manual"


class KnowledgeEntryInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    language: Literal["uz", "ru", "any"] = "any"
    text: str = Field(min_length=1, max_length=4000)
    keywords: str = Field(default="", max_length=2000)
    enabled: bool = True


class Passage(BaseModel):
    id: str
    title: str
    language: str
    text: str
    score: float


def tokenize(text: str) -> list[str]:
    """Word stems of ``text`` in a script-independent form."""
    # Uzbek Cyrillic becomes Latin; Russian stays Cyrillic (no Uzbek letters to map).
    words = to_uzbek_latin(text).replace("'", "").split()
    return [w[:STEM_LENGTH] for w in words if len(w) >= MIN_TOKEN_LENGTH]


def _scenario_entry(scenario) -> KnowledgeEntry:
    # A small model copies keyword lists into its answer; keep them for search only.
    keywords = ", ".join(k.rstrip("*") for k in scenario.keywords + scenario.context_keywords)
    answer = scenario.steps[0].message if scenario.steps else ""
    return KnowledgeEntry(
        id=f"scenario-{scenario.id}",
        title=scenario.title,
        language=scenario.language if scenario.language in ("uz", "ru") else "any",
        text=f"Situation: {scenario.description}\nTell the caller: {answer}",
        keywords=keywords,
        source=f"scenario:{scenario.id}",
    )


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        if path.exists():
            self._entries = self._read()
        else:
            self._entries = []
            self.import_scenarios()

    def _read(self) -> list[KnowledgeEntry]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [KnowledgeEntry(**item) for item in data]
        except (OSError, ValueError) as exc:
            LOGGER.error("Knowledge base %s unreadable, starting empty: %s", self._path, exc)
            return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        payload = [entry.model_dump() for entry in self._entries]
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self._path)

    def entries(self) -> list[KnowledgeEntry]:
        with self._lock:
            return list(self._entries)

    def add(self, item: KnowledgeEntryInput) -> KnowledgeEntry:
        entry = KnowledgeEntry(id=uuid.uuid4().hex[:12], **item.model_dump())
        with self._lock:
            self._entries.append(entry)
            self._save()
        return entry

    def update(self, entry_id: str, item: KnowledgeEntryInput) -> KnowledgeEntry | None:
        with self._lock:
            for index, entry in enumerate(self._entries):
                if entry.id == entry_id:
                    self._entries[index] = entry.model_copy(update=item.model_dump())
                    self._save()
                    return self._entries[index]
        return None

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.id != entry_id]
            if len(self._entries) == before:
                return False
            self._save()
            return True

    def import_scenarios(self) -> int:
        """Add or refresh one entry per answering scenario; return how many."""
        imported = [
            _scenario_entry(s)
            for s in get_scenario_service().all_scenarios()
            if s.action is None and s.steps
        ]
        with self._lock:
            by_id = {e.id: e for e in self._entries}
            for entry in imported:
                old = by_id.get(entry.id)
                # Keep an operator's on/off choice when refreshing.
                by_id[entry.id] = entry.model_copy(update={"enabled": old.enabled}) if old else entry
            self._entries = list(by_id.values())
            self._save()
        return len(imported)

    def search(self, query: str, language: str | None, top_k: int = 3) -> list[Passage]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        entries = [e for e in self.entries() if e.enabled]
        same_language = [e for e in entries if e.language in (language, "any")] if language else entries
        passages = self._rank(query_terms, same_language, top_k)
        # A Russian question about an Uzbek-only topic still deserves an answer.
        return passages or self._rank(query_terms, entries, top_k)

    @staticmethod
    def _rank(query_terms: set[str], entries: list[KnowledgeEntry], top_k: int) -> list[Passage]:
        if not entries:
            return []
        documents = [Counter(tokenize(f"{e.title} {e.text} {e.keywords}")) for e in entries]
        average = sum(sum(d.values()) for d in documents) / len(documents) or 1.0
        frequency = Counter(term for d in documents for term in d)
        count = len(documents)
        scored = []
        for entry, terms in zip(entries, documents):
            length = sum(terms.values())
            score = 0.0
            for term in query_terms:
                tf = terms.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * length / average))
            if score > 0:
                scored.append(Passage(id=entry.id, title=entry.title, language=entry.language,
                                      text=entry.text, score=round(score, 3)))
        scored.sort(key=lambda p: p.score, reverse=True)
        # Entries far below the best one only distract a small model.
        best = scored[0].score if scored else 0.0
        return [p for p in scored if p.score >= best * RELATIVE_MIN_SCORE][:top_k]


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(get_settings().knowledge_path)

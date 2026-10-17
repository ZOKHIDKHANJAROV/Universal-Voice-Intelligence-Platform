"""Maximum mode: find knowledge for the caller's words and let the LLM answer from it."""

import re
import time

from pydantic import BaseModel

from app.assistant.knowledge import Passage, get_knowledge_base
from app.assistant.llm import LlmError, chat
from app.assistant.runtime import AssistantSettings

LANGUAGE_RULE = {
    "uz": "Answer in Uzbek, Latin script (o', g', sh, ch), as people speak in Tashkent.",
    "ru": "Answer in Russian.",
}

SYSTEM_PROMPT = """You are the voice assistant of a customer support phone line for water vending machines.
You are speaking with a caller on the phone; your words are read aloud.

Rules:
- Find the KNOWLEDGE entry whose situation matches the caller's problem and tell the caller what it says, in short spoken words. Keep its concrete instructions (what to check, what to press, where to send).
- Use only facts from KNOWLEDGE. Never invent prices, addresses, working hours, phone numbers or promises.
- If no entry matches the caller's problem, do not guess: say briefly that you cannot answer that and that the caller can press zero to talk to an operator.
- {language_rule}
- At most {max_sentences} short sentences. Plain spoken text: no lists, no markdown, no emojis, no headings.
- Do not repeat the caller's words back. Do not greet: the call has already started.

KNOWLEDGE:
{knowledge}"""

_MARKUP = re.compile(r"[*#_`>|]+|^\s*[-•]\s*|^\s*\d+[.)]\s+", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[a-zA-Z]")


class LiveAnswer(BaseModel):
    text: str
    language: str
    passages: list[Passage]
    model: str
    retrieve_ms: int
    llm_ms: int


def _knowledge_block(passages: list[Passage]) -> str:
    if not passages:
        return "(nothing relevant found)"
    return "\n\n".join(f"[{i}] {p.title}\n{p.text}" for i, p in enumerate(passages, 1))


def clean_reply(text: str, max_sentences: int) -> str:
    """Spoken form: no markup, at most ``max_sentences`` sentences."""
    text = " ".join(_MARKUP.sub(" ", text).split())
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    text = " ".join(sentences[:max_sentences]).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def in_expected_script(text: str, language: str) -> bool:
    """Uzbek must come in Latin script for Navoiy; Russian in Cyrillic for Piper."""
    cyrillic, latin = len(_CYRILLIC.findall(text)), len(_LATIN.findall(text))
    if language == "uz":
        return latin > cyrillic
    if language == "ru":
        return cyrillic > latin
    return True


def generate_answer(
    question: str,
    language: str,
    settings: AssistantSettings,
    history: list[tuple[str, str]] | None = None,
) -> LiveAnswer:
    started = time.perf_counter()
    passages = get_knowledge_base().search(question, language, settings.top_k)
    retrieved = time.perf_counter()

    system = SYSTEM_PROMPT.format(
        language_rule=LANGUAGE_RULE.get(language, "Answer in the caller's language."),
        max_sentences=settings.max_sentences,
        knowledge=_knowledge_block(passages),
    )
    messages = [{"role": "system", "content": system}]
    for asked, answered in (history or [])[-2:]:
        messages += [{"role": "user", "content": asked}, {"role": "assistant", "content": answered}]
    messages.append({"role": "user", "content": question})

    reply = chat(settings.llm_base_url, settings.llm_model, messages, settings.llm_timeout_seconds,
                 gpu_layers=settings.llm_gpu_layers)
    answered = time.perf_counter()
    text = clean_reply(reply, settings.max_sentences)
    if not text:
        raise LlmError("Empty answer")
    if not in_expected_script(text, language):
        raise LlmError(f"Answer not in {language}: {text[:80]!r}")
    return LiveAnswer(
        text=text,
        language=language,
        passages=passages,
        model=settings.llm_model,
        retrieve_ms=round((retrieved - started) * 1000),
        llm_ms=round((answered - retrieved) * 1000),
    )

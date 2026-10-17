"""Admin panel API: answer mode, readiness of its parts, knowledge base, test box."""

import base64
import json
import shutil
import time
import urllib.request
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.assistant import llm
from app.assistant import runtime as assistant
from app.assistant.answer import generate_answer
from app.assistant.knowledge import KnowledgeEntry, KnowledgeEntryInput, get_knowledge_base
from app.audio.resample import pcm16_to_wav
from app.core.config import get_settings
from app.speech.service import get_stt
from app.tts.live import TELEPHONY_RATE, live_pcm8
from app.tts.prompt_cache import get_prompt_cache
from app.tts.prompts import all_phrases

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _tts_status(language: str) -> dict:
    settings = get_settings()
    provider = settings.tts_provider_ru if language == "ru" and settings.tts_provider_ru else settings.tts_provider
    if provider == "navoiy-http":
        url = f"{settings.tts_base_url.rstrip('/')}/health"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                ready = bool(json.loads(response.read().decode("utf-8")).get("ready", True))
            return {"provider": provider, "ready": ready, "detail": settings.tts_base_url}
        except (OSError, ValueError) as exc:
            return {"provider": provider, "ready": False,
                    "detail": f"{settings.tts_base_url} недоступен ({exc.__class__.__name__})"}
    if provider == "piper":
        model = settings.tts_model_path_ru if language == "ru" else settings.tts_model_path
        binary = shutil.which(settings.tts_binary)
        ready = bool(binary) and bool(model) and Path(model).exists()
        return {"provider": provider, "ready": ready,
                "detail": f"{settings.tts_binary} · {model or 'модель не задана'}"}
    return {"provider": provider, "ready": True, "detail": ""}


@router.get("/mode")
def get_mode() -> assistant.AssistantSettings:
    return assistant.current()


@router.put("/mode")
def set_mode(changes: assistant.AssistantSettingsUpdate) -> assistant.AssistantSettings:
    return assistant.get_assistant_store().update(changes)


@router.get("/status")
def status() -> dict:
    """What each mode needs and whether it is there."""
    settings = assistant.current()
    phrases = all_phrases()
    cache = get_prompt_cache()
    entries = get_knowledge_base().entries()
    return {
        "mode": settings.mode,
        "stt": {"ready": get_stt.cache_info().currsize > 0, "model": get_settings().stt_model},
        "prompts": {
            "rendered": sum(cache.path_for(t, lang).exists() for t, lang in phrases),
            "total": len(phrases),
        },
        "llm": llm.status(settings.llm_base_url, settings.llm_model),
        "tts_live": {language: _tts_status(language) for language in get_settings().stt_realtime_languages},
        "knowledge": {"entries": len(entries), "enabled": sum(e.enabled for e in entries)},
    }


@router.get("/knowledge")
def knowledge() -> list[KnowledgeEntry]:
    return get_knowledge_base().entries()


@router.post("/knowledge")
def add_knowledge(item: KnowledgeEntryInput) -> KnowledgeEntry:
    return get_knowledge_base().add(item)


@router.put("/knowledge/{entry_id}")
def update_knowledge(entry_id: str, item: KnowledgeEntryInput) -> KnowledgeEntry:
    entry = get_knowledge_base().update(entry_id, item)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@router.delete("/knowledge/{entry_id}")
def delete_knowledge(entry_id: str) -> dict:
    if not get_knowledge_base().delete(entry_id):
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"deleted": entry_id}


@router.post("/knowledge/import-scenarios")
def import_scenarios() -> dict:
    return {"imported": get_knowledge_base().import_scenarios()}


class TestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    language: str = "uz"
    speak: bool = True


@router.post("/test")
def test(request: TestRequest) -> dict:
    """Answer a typed question the way maximum mode would, whatever the current mode."""
    if request.language not in get_settings().stt_realtime_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    started = time.perf_counter()
    try:
        answer = generate_answer(request.text, request.language, assistant.current())
    except llm.LlmError as exc:
        raise HTTPException(status_code=502, detail=f"LLM: {exc}") from exc
    audio = None
    tts_ms = None
    warning = None
    if request.speak:
        spoken = time.perf_counter()
        try:
            wav = pcm16_to_wav(live_pcm8(answer.text, answer.language), TELEPHONY_RATE)
            audio = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
        except (OSError, ValueError) as exc:
            warning = f"Живой голос недоступен: {exc}"
        tts_ms = round((time.perf_counter() - spoken) * 1000)
    return {
        "answer": answer.text,
        "language": answer.language,
        "model": answer.model,
        "passages": [p.model_dump() for p in answer.passages],
        "timings_ms": {"rag": answer.retrieve_ms, "llm": answer.llm_ms, "tts": tts_ms,
                       "total": round((time.perf_counter() - started) * 1000)},
        "audio": audio,
        "warning": warning,
    }

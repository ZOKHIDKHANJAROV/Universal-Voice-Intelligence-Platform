import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.speech.service import get_stt
from app.tts.prompt_cache import get_prompt_cache
from app.tts.prompts import all_phrases
from app.voice.analyze import CallAnalysis, analyze_recording

router = APIRouter(prefix="/api/v1/console", tags=["console"])

# Only names produced by PromptAudioCache.path_for can be served.
_PROMPT_NAME = re.compile(r"^[a-z]{2}-[0-9a-f]{16}\.wav$")


@router.post("/analyze", response_model=CallAnalysis)
def analyze(
    audio: UploadFile = File(...),
    telephone: bool = Form(True),
    language: str = Form(""),
) -> CallAnalysis:
    settings = get_settings()
    if language and language not in settings.stt_realtime_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    content = audio.file.read(settings.stt_max_upload_bytes + 1)
    if len(content) > settings.stt_max_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio file is too large")

    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete_on_close=False) as temp_file:
        temp_file.write(content)
        temp_file.flush()
        try:
            return analyze_recording(Path(temp_file.name), telephone, language or None)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}") from exc


@router.get("/prompts")
def prompts() -> list[dict]:
    cache = get_prompt_cache()
    return [
        {"text": text, "language": language, "rendered": cache.path_for(text, language).exists()}
        for text, language in all_phrases()
    ]


@router.get("/prompts/audio/{name}")
def prompt_audio(name: str) -> FileResponse:
    if not _PROMPT_NAME.match(name):
        raise HTTPException(status_code=404, detail="Not found")
    path = get_settings().tts_prompt_cache_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not rendered")
    return FileResponse(path, media_type="audio/wav")


@router.get("/system")
def system() -> dict:
    settings = get_settings()
    phrases = all_phrases()
    cache = get_prompt_cache()
    return {
        "stt_model": settings.stt_model,
        "stt_model_uz": settings.stt_model_uz or None,
        "stt_device": settings.stt_device,
        "stt_compute_type": settings.stt_compute_type,
        "stt_loaded": get_stt.cache_info().currsize > 0,
        "languages": list(settings.stt_realtime_languages),
        "tts_provider": settings.tts_provider,
        "prompts_rendered": sum(cache.path_for(t, lang).exists() for t, lang in phrases),
        "prompts_total": len(phrases),
        "intent_min_confidence": settings.intent_min_confidence,
        "llm_enabled": settings.llm_enabled,
    }

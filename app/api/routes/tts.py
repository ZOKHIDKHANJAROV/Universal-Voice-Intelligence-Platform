import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from app.core.config import get_settings
from app.tts.service import synthesize

router = APIRouter(prefix="/api/v1/tts", tags=["text-to-speech"])


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: str = "uz"


@router.post("/synthesize")
def synthesize_audio(payload: TTSRequest):
    settings = get_settings()
    if payload.language not in settings.tts_supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    # One file per request; a shared path let concurrent requests overwrite
    # each other's audio.
    output_path = Path(tempfile.gettempdir()) / f"univoice-tts-{uuid4().hex}.wav"
    try:
        result = synthesize(payload.text, output_path, payload.language)
    except (OSError, ValueError) as exc:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FileResponse(
        result.audio_path,
        media_type="audio/wav",
        filename="response.wav",
        background=BackgroundTask(output_path.unlink, missing_ok=True),
    )

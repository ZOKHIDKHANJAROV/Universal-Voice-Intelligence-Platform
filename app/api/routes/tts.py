import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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
    output_path = Path(tempfile.gettempdir()) / "univoice-tts.wav"
    try:
        result = synthesize(payload.text, output_path, payload.language)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FileResponse(result.audio_path, media_type="audio/wav", filename="response.wav")

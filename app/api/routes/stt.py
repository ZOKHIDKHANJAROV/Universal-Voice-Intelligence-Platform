import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.speech.service import transcribe

router = APIRouter(prefix="/api/v1/stt", tags=["speech-to-text"])


@router.post("/transcribe")
def transcribe_audio(audio: UploadFile = File(...), language: str = "uz") -> dict:
    settings = get_settings()
    if language not in settings.stt_supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    content = audio.file.read(settings.stt_max_upload_bytes + 1)
    if len(content) > settings.stt_max_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio file is too large")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(content)
        temp_file.flush()
        result = transcribe(Path(temp_file.name), language=language)
    return result.model_dump()

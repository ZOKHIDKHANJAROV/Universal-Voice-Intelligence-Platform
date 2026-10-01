import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.voice.service import VoicePipelineService

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


@router.post("/process")
def process_voice(audio: UploadFile = File(...), language: str = "uz"):
    settings = get_settings()
    if language not in settings.stt_supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    content = audio.file.read(settings.stt_max_upload_bytes + 1)
    if len(content) > settings.stt_max_upload_bytes:
        raise HTTPException(status_code=413, detail="Audio file is too large")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(content)
        temp_file.flush()
        try:
            result = VoicePipelineService().process(
                Path(temp_file.name), call_id=f"voice-{uuid4().hex}", language=language
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return FileResponse(
        result.audio_path,
        media_type="audio/wav",
        filename="response.wav",
        headers={
            "X-Intent": result.intent.intent or "",
            "X-Scenario": result.intent.scenario_id or "",
        },
    )

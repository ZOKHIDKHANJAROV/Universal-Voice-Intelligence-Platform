import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from app.api.routes.console import router as console_router
from app.api.routes.health import router as health_router
from app.api.routes.intent import router as intent_router
from app.api.routes.scenarios import router as scenarios_router
from app.api.routes.stt import router as stt_router
from app.api.routes.voice import router as voice_router
from app.api.routes.tts import router as tts_router
from app.core.config import get_settings
from app.realtime.audio_socket import AudioSocketServer
from app.speech.service import warm_up
from app.tts.prompt_cache import get_prompt_cache
from app.tts.prompts import all_phrases

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())
LOGGER = logging.getLogger("univoice")

realtime_server = AudioSocketServer()


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    missing = await asyncio.to_thread(get_prompt_cache().preload, all_phrases())
    if missing:
        LOGGER.warning(
            "%d phrase(s) are not pre-rendered and will be synthesized live on first use. "
            "Run: python -m scripts.render_prompts",
            len(missing),
        )
    await realtime_server.start()
    warm_up_task = asyncio.create_task(_warm_up()) if settings.stt_preload else None
    try:
        yield
    finally:
        if warm_up_task:
            warm_up_task.cancel()
        await realtime_server.stop()


async def _warm_up() -> None:
    # In the background so the API answers health checks while models load.
    try:
        seconds = await asyncio.to_thread(warm_up)
        LOGGER.info("STT models loaded and warmed up in %.1fs", seconds)
    except Exception:
        LOGGER.exception("STT warm-up failed; models will load on the first call")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Universal Voice Intelligence Platform API",
    lifespan=lifespan,
)

protected = [Depends(verify_api_key)]
app.include_router(health_router)
app.include_router(scenarios_router, dependencies=protected)
app.include_router(stt_router, dependencies=protected)
app.include_router(voice_router, dependencies=protected)
app.include_router(intent_router, dependencies=protected)
app.include_router(tts_router, dependencies=protected)
app.include_router(console_router, dependencies=protected)

CONSOLE_PAGE = Path(__file__).parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    # Static page; every action it takes goes through the protected API.
    return FileResponse(CONSOLE_PAGE, media_type="text/html")

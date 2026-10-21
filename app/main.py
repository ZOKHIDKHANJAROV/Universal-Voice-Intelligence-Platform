import asyncio
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.routes.admin import router as admin_router
from app.api.routes.calls import router as calls_router
from app.api.routes.console import router as console_router
from app.api.routes.health import router as health_router
from app.api.routes.intent import router as intent_router
from app.api.routes.monitor import router as monitor_router
from app.api.routes.monitor import stream_router as monitor_stream_router
from app.api.routes.scenarios import router as scenarios_router
from app.api.routes.stt import router as stt_router
from app.api.routes.voice import router as voice_router
from app.api.routes.tts import router as tts_router
from app.assistant import runtime as assistant
from app.core import metrics as m
from app.core.config import get_settings
from app.core.logs import setup_logging
from app.realtime.audio_socket import AudioSocketServer
from app.speech.service import warm_up
from app.tts.prompt_cache import get_prompt_cache
from app.tts.prompts import all_phrases

settings = get_settings()
setup_logging(settings.log_level, settings.log_format)
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
    if assistant.current().mode == assistant.MAXIMUM:
        assistant.warm_up_in_background(assistant.current())
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
app.include_router(calls_router)
app.include_router(scenarios_router, dependencies=protected)
app.include_router(stt_router, dependencies=protected)
app.include_router(voice_router, dependencies=protected)
app.include_router(intent_router, dependencies=protected)
app.include_router(tts_router, dependencies=protected)
app.include_router(console_router, dependencies=protected)
app.include_router(monitor_router, dependencies=protected)
app.include_router(admin_router, dependencies=protected)
app.include_router(monitor_stream_router)  # checks ?key= itself

WEB = Path(__file__).parent / "web"
CONSOLE_PAGE = WEB / "index.html"
MONITOR_PAGE = WEB / "monitor.html"
ADMIN_PAGE = WEB / "admin.html"


# Browsers must revalidate, or an update keeps showing the old page.
_NO_CACHE = {"Cache-Control": "no-cache"}


def _route_template(request: Request) -> str:
    # The template (/api/v1/monitor/calls/{call_id}), not the path, keeps the
    # number of series small; scanners' random paths all count as "unmatched".
    # Routing leaves the matched route in the scope.
    return getattr(request.scope.get("route"), "path", "unmatched")


@app.middleware("http")
async def count_requests(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = _route_template(request)
        m.HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
        m.HTTP_SECONDS.labels(request.method, route).observe(time.perf_counter() - started)


@app.get("/metrics", include_in_schema=False)
def metrics(request: Request) -> Response:
    """Prometheus metrics. With API_KEY set, send it as X-API-Key or a Bearer token."""
    expected = get_settings().api_key
    if expected:
        given = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
        if not secrets.compare_digest(given, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    # Static page; every action it takes goes through the protected API.
    return FileResponse(CONSOLE_PAGE, media_type="text/html", headers=_NO_CACHE)


@app.get("/monitor", include_in_schema=False)
def monitor() -> FileResponse:
    return FileResponse(MONITOR_PAGE, media_type="text/html", headers=_NO_CACHE)


@app.get("/admin", include_in_schema=False)
def admin() -> FileResponse:
    return FileResponse(ADMIN_PAGE, media_type="text/html", headers=_NO_CACHE)

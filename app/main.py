from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.scenarios import router as scenarios_router
from app.api.routes.stt import router as stt_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Universal Voice Intelligence Platform API",
)

app.include_router(health_router)
app.include_router(scenarios_router)
app.include_router(stt_router)

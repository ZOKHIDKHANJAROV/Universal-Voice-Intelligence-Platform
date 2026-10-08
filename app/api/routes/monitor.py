import asyncio
import json
import secrets
import time

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from app.calls import hub
from app.calls.log import get_call_log
from app.core.config import get_settings

router = APIRouter(prefix="/api/v1/monitor", tags=["monitor"])

# EventSource and <audio> cannot send headers, so the stream and recordings
# live outside the routers protected by X-API-Key and check ?key= themselves.
stream_router = APIRouter(prefix="/api/v1/monitor", tags=["monitor"])


def _check_key(key: str) -> None:
    expected = get_settings().api_key
    if expected and not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


_HEARTBEAT_SECONDS = 15


@router.get("/calls")
def calls(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return get_call_log().list_calls(limit)


@router.get("/calls/{call_id}")
def call(call_id: str) -> dict:
    log = get_call_log()
    summary = log.summary(call_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return {"call": summary, "events": log.events(call_id)}


@stream_router.get("/calls/{call_id}/audio/{event_id}")
def audio(call_id: str, event_id: int, key: str = "") -> FileResponse:
    _check_key(key)
    path = get_call_log().audio_path(call_id, event_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No recording")
    return FileResponse(path, media_type="audio/wav")


@router.get("/stats")
def stats(hours: int = Query(24, ge=1, le=24 * 31)) -> dict:
    return {
        **get_call_log().stats(time.time() - hours * 3600),
        "hours": hours,
        "recording": get_settings().call_record_audio,
    }


@stream_router.get("/stream")
async def stream(request: Request, key: str = "") -> StreamingResponse:
    _check_key(key)

    queue = hub.subscribe()

    async def events():
        try:
            yield "retry: 3000\n\n"
            while not await request.is_disconnected():
                try:
                    message = await asyncio.wait_for(queue.get(), _HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keeps proxies from closing an idle stream
                    continue
                yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
        finally:
            hub.unsubscribe(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

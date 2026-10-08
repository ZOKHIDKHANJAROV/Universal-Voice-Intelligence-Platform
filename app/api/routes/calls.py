import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.realtime import call_actions

# Called by the Asterisk dialplan (CURL) right after the bot ends its
# AudioSocket session. Not under /api/v1: the dialplan sends no API key, and
# the answer reveals nothing beyond "operator" or "hangup" for a random UUID.
router = APIRouter(prefix="/internal/calls", tags=["internal"], include_in_schema=False)


def _check(call_uuid: str) -> None:
    try:
        uuid.UUID(call_uuid)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Unknown call") from exc


@router.get("/{call_uuid}/next", response_class=PlainTextResponse)
def next_action(call_uuid: str) -> str:
    _check(call_uuid)
    return call_actions.take(call_uuid)


@router.get("/{call_uuid}/operator-unavailable", response_class=PlainTextResponse)
def operator_unavailable(call_uuid: str) -> str:
    """No operator answered; the dialplan is sending the caller back to the bot."""
    _check(call_uuid)
    call_actions.mark_operator_unavailable(call_uuid)
    return "ok"

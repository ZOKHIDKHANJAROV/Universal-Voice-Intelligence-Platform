"""What Asterisk should do with a call once the bot lets go of it.

The bot ends its AudioSocket session to hand a caller over; the dialplan then
asks the API (GET /internal/calls/{uuid}/next) whether to transfer to an
operator or hang up. If no operator answers, the dialplan reports it
(GET /internal/calls/{uuid}/operator-unavailable) and reconnects the caller to
the bot, which then apologises instead of greeting again.
"""

import threading
import time
from dataclasses import dataclass

OPERATOR = "operator"
HANGUP = "hangup"

# Asterisk asks within milliseconds, or after a 45 s dial attempt; anything
# older belongs to a call that went away.
_TTL_SECONDS = 600.0


@dataclass
class _Record:
    action: str = HANGUP
    language: str | None = None
    operator_unavailable: bool = False
    expires: float = 0.0


_lock = threading.Lock()
_calls: dict[str, _Record] = {}


def _record(call_uuid: str) -> _Record:
    now = time.monotonic()
    for uuid, record in list(_calls.items()):
        if record.expires < now:
            del _calls[uuid]
    record = _calls.setdefault(call_uuid, _Record())
    record.expires = now + _TTL_SECONDS
    return record


def request(call_uuid: str, action: str, language: str | None = None) -> None:
    with _lock:
        record = _record(call_uuid)
        record.action = action
        record.language = language


def take(call_uuid: str) -> str:
    """Return the requested action once; later calls (and unknown calls) hang up."""
    with _lock:
        record = _calls.get(call_uuid)
        if record is None or record.expires < time.monotonic():
            return HANGUP
        action, record.action = record.action, HANGUP
        return action


def mark_operator_unavailable(call_uuid: str) -> None:
    with _lock:
        _record(call_uuid).operator_unavailable = True


def take_operator_unavailable(call_uuid: str) -> tuple[bool, str | None]:
    """Whether this call is coming back from a failed transfer, and its language."""
    with _lock:
        record = _calls.get(call_uuid)
        if record is None or not record.operator_unavailable:
            return False, None
        record.operator_unavailable = False
        return True, record.language

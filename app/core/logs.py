"""Log setup: one JSON object per line, tagged with the call it belongs to.

The call is kept in a context variable. asyncio tasks and asyncio.to_thread
copy the context, so everything a call starts (playback, STT in a worker
thread, LLM requests) logs under that call without passing ids around. The
context holds a mutable object because Asterisk sends the call UUID only after
the connection is set up and tasks already started must see it too.
"""

import contextvars
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

# Loggers uvicorn configures with its own handlers before the app is imported.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

# Attributes every LogRecord has; anything else came in through extra={...}.
# uvicorn adds color_message, an ANSI-coloured copy of the message.
_STANDARD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName", "color_message"}


@dataclass
class CallContext:
    # Our id for the connection (phone) or the console request.
    call_id: str
    # The Asterisk call UUID, the key in the call log and /monitor.
    call_uuid: str | None = None


_current: contextvars.ContextVar[CallContext | None] = contextvars.ContextVar("univoice_call", default=None)


def bind_call(call_id: str, call_uuid: str | None = None) -> CallContext:
    """Tag every log line of the current task (and what it starts) with this call."""
    context = CallContext(call_id, call_uuid)
    _current.set(context)
    return context


def current_call() -> CallContext | None:
    return _current.get()


class _CallFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = _current.get()
        record.call_id = context.call_id if context else None
        record.call_uuid = context.call_uuid if context else None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if getattr(record, "call_uuid", None):
            entry["call_uuid"] = record.call_uuid
        # Console calls use one id for both.
        if getattr(record, "call_id", None) and record.call_id != entry.get("call_uuid"):
            entry["call_id"] = record.call_id
        for key, value in vars(record).items():
            if key not in _STANDARD and key not in entry and key not in ("call_id", "call_uuid"):
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        call = getattr(record, "call_uuid", None) or getattr(record, "call_id", None)
        return f"{line} [call={call}]" if call else line


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Send the app's and uvicorn's logs to stdout, as JSON or as plain text."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_CallFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in _UVICORN_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

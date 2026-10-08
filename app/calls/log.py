"""Call log: every call as a chat of caller, bot and system events.

Stored in SQLite so the monitor needs no extra service. Writes are tiny and
guarded by a lock; logging failures never interrupt a call.
"""

import json
import logging
import shutil
import sqlite3
import threading
import time
from functools import lru_cache
from pathlib import Path

from app.audio.resample import pcm16_to_wav
from app.calls import hub
from app.core.config import get_settings

LOGGER = logging.getLogger("univoice.calls")

# Roles in the chat view.
CALLER = "caller"
BOT = "bot"
SYSTEM = "system"

# Call statuses.
ACTIVE = "active"
ENDED = "ended"
TRANSFERRED = "transferred"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    status TEXT NOT NULL,
    language TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id TEXT NOT NULL REFERENCES calls(id),
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT,
    data TEXT,
    audio TEXT
);
CREATE INDEX IF NOT EXISTS events_by_call ON events(call_id, id);
CREATE INDEX IF NOT EXISTS calls_by_start ON calls(started_at);
"""


class CallLog:
    def __init__(self, db_path: Path, audio_dir: Path, record_audio: bool) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._audio_dir = audio_dir
        self._record_audio = record_audio
        self._lock = threading.Lock()

    # -- writing -----------------------------------------------------------

    def start_call(self, call_id: str, channel: str) -> None:
        """Open a call, or reopen it when the caller comes back to the bot."""
        try:
            with self._lock:
                row = self._db.execute("SELECT id FROM calls WHERE id = ?", (call_id,)).fetchone()
                if row is None:
                    self._db.execute(
                        "INSERT INTO calls (id, channel, started_at, status) VALUES (?, ?, ?, ?)",
                        (call_id, channel, time.time(), ACTIVE),
                    )
                else:
                    self._db.execute(
                        "UPDATE calls SET status = ?, ended_at = NULL WHERE id = ?", (ACTIVE, call_id)
                    )
                self._db.commit()
            hub.publish({"type": "call", "call": self.summary(call_id)})
        except Exception:
            LOGGER.exception("Could not open call %s", call_id)

    def add_event(
        self,
        call_id: str,
        role: str,
        kind: str,
        text: str | None = None,
        data: dict | None = None,
        audio_pcm16: bytes | None = None,
        sample_rate: int = 8000,
    ) -> dict | None:
        try:
            now = time.time()
            with self._lock:
                cursor = self._db.execute(
                    "INSERT INTO events (call_id, ts, role, kind, text, data) VALUES (?, ?, ?, ?, ?, ?)",
                    (call_id, now, role, kind, text, json.dumps(data or {}, ensure_ascii=False)),
                )
                event_id = cursor.lastrowid
                audio = None
                if audio_pcm16 and self._record_audio:
                    audio = f"{event_id}.wav"
                    path = self._audio_dir / call_id / audio
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(pcm16_to_wav(audio_pcm16, sample_rate))
                    self._db.execute("UPDATE events SET audio = ? WHERE id = ?", (audio, event_id))
                language = (data or {}).get("language") if role == CALLER else None
                if language:
                    self._db.execute("UPDATE calls SET language = ? WHERE id = ?", (language, call_id))
                self._db.commit()
            event = {
                "id": event_id, "call_id": call_id, "ts": now, "role": role, "kind": kind,
                "text": text, "data": data or {}, "audio": audio,
            }
            hub.publish({"type": "event", "event": event, "call": self.summary(call_id)})
            return event
        except Exception:
            LOGGER.exception("Could not log %s event for call %s", kind, call_id)
            return None

    def end_call(self, call_id: str, status: str = ENDED) -> None:
        try:
            with self._lock:
                self._db.execute(
                    "UPDATE calls SET status = ?, ended_at = ? WHERE id = ?", (status, time.time(), call_id)
                )
                self._db.commit()
            hub.publish({"type": "call", "call": self.summary(call_id)})
        except Exception:
            LOGGER.exception("Could not close call %s", call_id)

    def prune(self, days: int) -> int:
        """Delete calls older than ``days`` together with their audio."""
        cutoff = time.time() - days * 86400
        with self._lock:
            old = [r["id"] for r in self._db.execute(
                "SELECT id FROM calls WHERE started_at < ?", (cutoff,)
            )]
            for call_id in old:
                self._db.execute("DELETE FROM events WHERE call_id = ?", (call_id,))
                self._db.execute("DELETE FROM calls WHERE id = ?", (call_id,))
                shutil.rmtree(self._audio_dir / call_id, ignore_errors=True)
            self._db.commit()
        return len(old)

    # -- reading -----------------------------------------------------------

    def summary(self, call_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(_SUMMARY_SQL + " WHERE c.id = ?", (call_id,)).fetchone()
        return dict(row) if row else None

    def list_calls(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                _SUMMARY_SQL + " ORDER BY c.started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def events(self, call_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events WHERE call_id = ? ORDER BY id", (call_id,)
            ).fetchall()
        return [
            {**dict(row), "data": json.loads(row["data"] or "{}")} for row in rows
        ]

    def audio_path(self, call_id: str, event_id: int) -> Path | None:
        with self._lock:
            row = self._db.execute(
                "SELECT audio FROM events WHERE call_id = ? AND id = ?", (call_id, event_id)
            ).fetchone()
        if not row or not row["audio"]:
            return None
        path = self._audio_dir / call_id / row["audio"]
        return path if path.is_file() else None

    def stats(self, since: float) -> dict:
        with self._lock:
            # A transfer counts even when nobody answered and the caller
            # returned to the bot, so look for the event, not the final status.
            calls = self._db.execute(
                "SELECT COUNT(*) AS n, SUM(status = 'active') AS active, "
                "SUM(EXISTS (SELECT 1 FROM events e WHERE e.call_id = c.id AND e.kind = 'transfer' "
                "AND e.role = 'system')) AS transferred "
                "FROM calls c WHERE started_at >= ?", (since,)
            ).fetchone()
            answers = self._db.execute(
                "SELECT data FROM events e JOIN calls c ON c.id = e.call_id "
                "WHERE e.kind = 'answer' AND c.started_at >= ?", (since,)
            ).fetchall()
        parsed = [json.loads(r["data"] or "{}") for r in answers]
        latencies = sorted(d["total_ms"] for d in parsed if d.get("total_ms") is not None)
        understood = sum(1 for d in parsed if d.get("scenario_id"))
        return {
            "calls": calls["n"] or 0,
            "active": calls["active"] or 0,
            "transferred": calls["transferred"] or 0,
            "answers": len(parsed),
            "understood": understood,
            "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
        }


_SUMMARY_SQL = """
SELECT c.*,
       (SELECT COUNT(*) FROM events e WHERE e.call_id = c.id AND e.role = 'caller') AS turns,
       (SELECT text FROM events e WHERE e.call_id = c.id AND e.role != 'system' AND e.text != ''
        ORDER BY e.id DESC LIMIT 1) AS last_text
FROM calls c
"""


@lru_cache(maxsize=1)
def get_call_log() -> CallLog:
    settings = get_settings()
    log = CallLog(settings.call_log_path, settings.call_audio_dir, settings.call_record_audio)
    removed = log.prune(settings.call_log_retention_days)
    if removed:
        LOGGER.info("Removed %d calls older than %d days", removed, settings.call_log_retention_days)
    return log

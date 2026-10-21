"""Call log: every call as a chat of caller, bot and system events.

Stored in SQLite by default so the monitor needs no extra service, or in
PostgreSQL when DATABASE_URL is set (see app/calls/storage.py). Writes are
tiny; logging failures never interrupt a call.
"""

import json
import logging
import shutil
import time
from functools import lru_cache
from pathlib import Path

from app.audio.resample import pcm16_to_wav
from app.calls import hub
from app.calls.storage import Database, open_database
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


class CallLog:
    def __init__(self, db: Database, audio_dir: Path, record_audio: bool) -> None:
        self._db = db
        self._audio_dir = audio_dir
        self._record_audio = record_audio

    # -- writing -----------------------------------------------------------

    def start_call(self, call_id: str, channel: str) -> None:
        """Open a call, or reopen it when the caller comes back to the bot."""
        try:
            with self._db.transaction() as tx:
                if not tx.execute("SELECT id FROM calls WHERE id = ?", (call_id,)):
                    tx.execute(
                        "INSERT INTO calls (id, channel, started_at, status) VALUES (?, ?, ?, ?)",
                        (call_id, channel, time.time(), ACTIVE),
                    )
                else:
                    tx.execute("UPDATE calls SET status = ?, ended_at = NULL WHERE id = ?", (ACTIVE, call_id))
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
            with self._db.transaction() as tx:
                event_id = tx.insert(
                    "INSERT INTO events (call_id, ts, role, kind, text, data) VALUES (?, ?, ?, ?, ?, ?)",
                    (call_id, now, role, kind, text, json.dumps(data or {}, ensure_ascii=False)),
                )
                audio = None
                if audio_pcm16 and self._record_audio:
                    audio = f"{event_id}.wav"
                    path = self._audio_dir / call_id / audio
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(pcm16_to_wav(audio_pcm16, sample_rate))
                    tx.execute("UPDATE events SET audio = ? WHERE id = ?", (audio, event_id))
                language = (data or {}).get("language") if role == CALLER else None
                if language:
                    tx.execute("UPDATE calls SET language = ? WHERE id = ?", (language, call_id))
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
            with self._db.transaction() as tx:
                tx.execute(
                    "UPDATE calls SET status = ?, ended_at = ? WHERE id = ?", (status, time.time(), call_id)
                )
            hub.publish({"type": "call", "call": self.summary(call_id)})
        except Exception:
            LOGGER.exception("Could not close call %s", call_id)

    def prune(self, days: int) -> int:
        """Delete calls older than ``days`` together with their audio."""
        cutoff = time.time() - days * 86400
        with self._db.transaction() as tx:
            old = [r["id"] for r in tx.execute("SELECT id FROM calls WHERE started_at < ?", (cutoff,))]
            for call_id in old:
                tx.execute("DELETE FROM events WHERE call_id = ?", (call_id,))
                tx.execute("DELETE FROM calls WHERE id = ?", (call_id,))
        for call_id in old:
            shutil.rmtree(self._audio_dir / call_id, ignore_errors=True)
        return len(old)

    # -- reading -----------------------------------------------------------

    def summary(self, call_id: str) -> dict | None:
        rows = self._db.query(_SUMMARY_SQL + " WHERE c.id = ?", (call_id,))
        return rows[0] if rows else None

    def list_calls(self, limit: int = 50) -> list[dict]:
        return self._db.query(_SUMMARY_SQL + " ORDER BY c.started_at DESC LIMIT ?", (limit,))

    def events(self, call_id: str) -> list[dict]:
        rows = self._db.query("SELECT * FROM events WHERE call_id = ? ORDER BY id", (call_id,))
        return [{**row, "data": json.loads(row["data"] or "{}")} for row in rows]

    def audio_path(self, call_id: str, event_id: int) -> Path | None:
        rows = self._db.query("SELECT audio FROM events WHERE call_id = ? AND id = ?", (call_id, event_id))
        if not rows or not rows[0]["audio"]:
            return None
        path = self._audio_dir / call_id / rows[0]["audio"]
        return path if path.is_file() else None

    def stats(self, since: float) -> dict:
        # A transfer counts even when nobody answered and the caller
        # returned to the bot, so look for the event, not the final status.
        # CASE rather than SUM(condition): PostgreSQL cannot sum booleans.
        calls = self._db.query(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active, "
            "SUM(CASE WHEN EXISTS (SELECT 1 FROM events e WHERE e.call_id = c.id AND e.kind = 'transfer' "
            "AND e.role = 'system') THEN 1 ELSE 0 END) AS transferred "
            "FROM calls c WHERE started_at >= ?", (since,)
        )[0]
        answers = self._db.query(
            "SELECT data FROM events e JOIN calls c ON c.id = e.call_id "
            "WHERE e.kind = 'answer' AND c.started_at >= ?", (since,)
        )
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
    db = open_database(settings.database_url, settings.call_log_path)
    log = CallLog(db, settings.call_audio_dir, settings.call_record_audio)
    try:
        removed = log.prune(settings.call_log_retention_days)
    except Exception:
        # PostgreSQL may be down at start; calls go on and each write retries.
        LOGGER.exception("Could not remove old calls")
    else:
        if removed:
            LOGGER.info("Removed %d calls older than %d days", removed, settings.call_log_retention_days)
    return log

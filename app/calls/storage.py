"""Where the call log lives: SQLite (default) or PostgreSQL.

Both speak the same small interface: ``query`` for reads and ``transaction``
for writes. The call log writes its SQL once with ``?`` placeholders; each
backend adapts it to its own dialect.

SQLite is a single file and needs no service, which suits one laptop.
PostgreSQL (``DATABASE_URL=postgresql://...``) suits a shared server, backups
and reporting from other tools.
"""

import logging
import sqlite3
import threading
from contextlib import AbstractContextManager, contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Protocol

LOGGER = logging.getLogger("univoice.calls")

# Timestamps are Unix seconds; ``data`` holds the event's JSON.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    started_at {real} NOT NULL,
    ended_at {real},
    status TEXT NOT NULL,
    language TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id {serial},
    call_id TEXT NOT NULL REFERENCES calls(id),
    ts {real} NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT,
    data TEXT,
    audio TEXT
);
CREATE INDEX IF NOT EXISTS events_by_call ON events(call_id, id);
CREATE INDEX IF NOT EXISTS calls_by_start ON calls(started_at);
"""


class Transaction(Protocol):
    def execute(self, sql: str, params: tuple = ()) -> list[dict]: ...

    def insert(self, sql: str, params: tuple = ()) -> int:
        """Run an INSERT and return the new row's ``id``."""
        ...


class Database(Protocol):
    def query(self, sql: str, params: tuple = ()) -> list[dict]: ...

    def transaction(self) -> AbstractContextManager[Transaction]:
        """Writes that commit together, or not at all."""
        ...


class SQLiteDatabase:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA.format(real="REAL", serial="INTEGER PRIMARY KEY AUTOINCREMENT"))
        self._lock = threading.Lock()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._db.execute(sql, params).fetchall()]

    @contextmanager
    def transaction(self) -> Iterator["_SQLiteTransaction"]:
        with self._lock:
            try:
                yield _SQLiteTransaction(self._db)
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise


class _SQLiteTransaction:
    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db

    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(row) for row in self._db.execute(sql, params).fetchall()]

    def insert(self, sql: str, params: tuple = ()) -> int:
        return self._db.execute(sql, params).lastrowid


class PostgresDatabase:
    """One connection shared under a lock, like SQLite; reconnects when it drops.

    The call log writes a few small rows per utterance, so a pool would only
    add moving parts.
    """

    def __init__(self, url: str) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'DATABASE_URL points at PostgreSQL; install the driver: pip install -e ".[postgres]"'
            ) from exc
        self._url = url
        self._conn = None
        self._lock = threading.Lock()

    def _connection(self):
        import psycopg
        from psycopg.rows import dict_row

        if self._conn is None or self._conn.closed or self._conn.broken:
            self._conn = psycopg.connect(self._url, row_factory=dict_row, connect_timeout=5)
            with self._conn.transaction():
                # Several API workers may start at once.
                self._conn.execute("SELECT pg_advisory_xact_lock(724100)")
                for statement in _SCHEMA.format(real="DOUBLE PRECISION", serial="BIGSERIAL PRIMARY KEY").split(";"):
                    if statement.strip():
                        self._conn.execute(statement)
        return self._conn

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            conn = self._connection()
            with conn.transaction():
                return _PostgresTransaction(conn).execute(sql, params)

    @contextmanager
    def transaction(self) -> Iterator["_PostgresTransaction"]:
        with self._lock:
            conn = self._connection()
            with conn.transaction():
                yield _PostgresTransaction(conn)


class _PostgresTransaction:
    def __init__(self, conn) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        cursor = self._conn.execute(_to_postgres(sql), params)
        return [_plain(row) for row in cursor.fetchall()] if cursor.description else []

    def insert(self, sql: str, params: tuple = ()) -> int:
        return self._conn.execute(_to_postgres(sql) + " RETURNING id", params).fetchone()["id"]


def _to_postgres(sql: str) -> str:
    # The call log's SQL has no literal '?' or '%'.
    return sql.replace("?", "%s")


def _plain(row: dict) -> dict:
    # SUM() comes back as Decimal; the API returns JSON numbers.
    return {key: int(value) if isinstance(value, Decimal) else value for key, value in row.items()}


def open_database(database_url: str, sqlite_path: Path) -> Database:
    """``DATABASE_URL`` set to postgresql://... picks PostgreSQL, empty keeps SQLite."""
    if not database_url:
        return SQLiteDatabase(sqlite_path)
    if database_url.startswith(("postgresql://", "postgres://")):
        return PostgresDatabase(database_url)
    if database_url.startswith("sqlite:///"):
        return SQLiteDatabase(Path(database_url.removeprefix("sqlite:///")))
    raise ValueError("DATABASE_URL must start with postgresql:// or sqlite:///")

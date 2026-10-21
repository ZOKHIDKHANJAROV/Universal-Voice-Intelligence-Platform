import os

import pytest
from prometheus_client import REGISTRY

import app.assistant.knowledge as knowledge
import app.assistant.runtime as assistant_runtime
import app.calls.log as call_log
from app.core.config import get_settings


def metric(name: str, **labels: str) -> float:
    """Current value of a Prometheus sample (0 when the series does not exist yet)."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture(autouse=True)
def _isolated_call_log(tmp_path, monkeypatch):
    """Each test gets its own call log, answer mode and knowledge base instead of data/."""
    settings = get_settings()
    monkeypatch.setattr(settings, "database_url", "")  # SQLite, even if .env sets PostgreSQL
    monkeypatch.setattr(settings, "call_log_path", tmp_path / "calls.db")
    monkeypatch.setattr(settings, "call_audio_dir", tmp_path / "calls")
    monkeypatch.setattr(settings, "assistant_settings_path", tmp_path / "assistant.json")
    monkeypatch.setattr(settings, "knowledge_path", tmp_path / "knowledge.json")
    # Switching to maximum mode would load the LLM over the network.
    monkeypatch.setattr(assistant_runtime, "warm_up_in_background", lambda settings: None)
    for cached in (call_log.get_call_log, assistant_runtime.get_assistant_store, knowledge.get_knowledge_base):
        cached.cache_clear()
    yield
    for cached in (call_log.get_call_log, assistant_runtime.get_assistant_store, knowledge.get_knowledge_base):
        cached.cache_clear()


@pytest.fixture(params=["sqlite", "postgres"])
def call_log_db(request, monkeypatch) -> str:
    """Run a call-log test on SQLite and on PostgreSQL; returns the DATABASE_URL.

    PostgreSQL runs only with TEST_DATABASE_URL set (e.g.
    postgresql://univoice:univoice@localhost:5432/univoice_test) and is skipped
    otherwise. Its call tables are dropped before each test.
    """
    url = ""
    if request.param == "postgres":
        url = os.environ.get("TEST_DATABASE_URL", "")
        if not url:
            pytest.skip("set TEST_DATABASE_URL=postgresql://... to test the PostgreSQL call log")
        psycopg = pytest.importorskip("psycopg")
        try:
            with psycopg.connect(url, connect_timeout=3, autocommit=True) as conn:
                conn.execute("DROP TABLE IF EXISTS events, calls")
        except psycopg.OperationalError as exc:
            pytest.skip(f"PostgreSQL is not reachable: {exc}")
    monkeypatch.setattr(get_settings(), "database_url", url)
    call_log.get_call_log.cache_clear()
    return url

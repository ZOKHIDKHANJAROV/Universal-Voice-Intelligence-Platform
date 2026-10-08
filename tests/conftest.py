import pytest

import app.calls.log as call_log
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_call_log(tmp_path, monkeypatch):
    """Each test gets its own call log instead of writing to data/calls.db."""
    settings = get_settings()
    monkeypatch.setattr(settings, "call_log_path", tmp_path / "calls.db")
    monkeypatch.setattr(settings, "call_audio_dir", tmp_path / "calls")
    call_log.get_call_log.cache_clear()
    yield
    call_log.get_call_log.cache_clear()

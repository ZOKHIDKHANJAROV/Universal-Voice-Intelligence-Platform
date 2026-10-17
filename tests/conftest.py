import pytest

import app.assistant.knowledge as knowledge
import app.assistant.runtime as assistant_runtime
import app.calls.log as call_log
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_call_log(tmp_path, monkeypatch):
    """Each test gets its own call log, answer mode and knowledge base instead of data/."""
    settings = get_settings()
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

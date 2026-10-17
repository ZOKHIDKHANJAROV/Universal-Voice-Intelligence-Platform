"""Answer mode and its settings, changed from the admin panel while calls run.

Kept in a small JSON file next to the call log so a restart keeps the
operator's choice; every call turn reads the current value.
"""

import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings

LOGGER = logging.getLogger("univoice.assistant")

MINIMUM = "minimum"
MAXIMUM = "maximum"


class AssistantSettings(BaseModel):
    # minimum: language detection + pre-recorded scenario answers.
    # maximum: language detection + knowledge base search (RAG) + an LLM
    # answer spoken by live TTS; falls back to minimum when a part fails.
    mode: Literal["minimum", "maximum"] = MINIMUM
    llm_base_url: str = ""
    llm_model: str = ""
    # Knowledge entries handed to the LLM per question.
    top_k: int = Field(default=3, ge=1, le=8)
    # Longest answer the LLM may give; phone answers must be short.
    max_sentences: int = Field(default=2, ge=1, le=5)
    # Past this the caller hears the pre-recorded answer instead of more silence.
    llm_timeout_seconds: float = Field(default=8.0, ge=2.0, le=120.0)
    # Say "one moment" while the answer is generated.
    filler: bool = True
    # LLM layers on the GPU (Ollama num_gpu); None = Ollama decides, 0 = CPU
    # only. Whisper and Navoiy already hold most of an 8 GB card, and Windows
    # spills an over-full card into system RAM, which is far slower than the CPU.
    llm_gpu_layers: int | None = Field(default=None, ge=0, le=99)


class AssistantSettingsUpdate(BaseModel):
    mode: Literal["minimum", "maximum"] | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=8)
    max_sentences: int | None = Field(default=None, ge=1, le=5)
    llm_timeout_seconds: float | None = Field(default=None, ge=2.0, le=120.0)
    filler: bool | None = None
    # -1 in the admin form means "automatic" (stored as None).
    llm_gpu_layers: int | None = Field(default=None, ge=-1, le=99)


class AssistantStore:
    """Defaults from .env plus the fields changed in the admin panel.

    Only the changed fields are saved, so a new default (say, a better model
    in .env) still applies to everything the operator never touched.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._overrides = self._load()
        self._settings = self._defaults().model_copy(update=self._overrides)

    def _defaults(self) -> AssistantSettings:
        settings = get_settings()
        return AssistantSettings(
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
        )

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
            return AssistantSettingsUpdate(**saved).model_dump(exclude_none=True)
        except (OSError, ValueError) as exc:
            LOGGER.error("Ignoring unreadable %s: %s", self._path, exc)
            return {}

    def get(self) -> AssistantSettings:
        with self._lock:
            return self._settings

    def update(self, changes: AssistantSettingsUpdate) -> AssistantSettings:
        with self._lock:
            self._overrides.update(changes.model_dump(exclude_none=True))
            if self._overrides.get("llm_gpu_layers") == -1:
                del self._overrides["llm_gpu_layers"]
            self._settings = self._defaults().model_copy(update=self._overrides)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._overrides, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self._path)
            LOGGER.info("Assistant settings changed: %s", changes.model_dump(exclude_none=True))
            settings = self._settings
        if settings.mode == MAXIMUM and (
            changes.mode or changes.llm_model or changes.llm_base_url or changes.llm_gpu_layers is not None
        ):
            warm_up_in_background(settings)
        return settings


def warm_up_in_background(settings: AssistantSettings) -> None:
    """Load the LLM now; a cold model takes longer than a caller should wait."""
    from app.assistant import llm

    def run() -> None:
        try:
            llm.warm_up(settings.llm_base_url, settings.llm_model, settings.llm_gpu_layers)
            LOGGER.info("LLM %s loaded", settings.llm_model)
        except llm.LlmError as exc:
            LOGGER.warning("LLM %s could not be loaded: %s", settings.llm_model, exc)

    threading.Thread(target=run, name="llm-warm-up", daemon=True).start()


@lru_cache(maxsize=1)
def get_assistant_store() -> AssistantStore:
    return AssistantStore(get_settings().assistant_settings_path)


def current() -> AssistantSettings:
    return get_assistant_store().get()

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "UniVoice AI"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_supported_languages: tuple[str, ...] = ("uz", "ru", "en")
    stt_max_upload_bytes: int = 25 * 1024 * 1024

    llm_enabled: bool = False
    llm_base_url: str = "http://ollama:11434"
    llm_model: str = "qwen3:4b"
    llm_timeout_seconds: int = 30

    tts_provider: str = "piper"
    tts_binary: str = "piper"
    tts_model_path: str = ""
    tts_python_binary: str = "python"
    tts_navoiy_inference_script: str = ""
    tts_navoiy_cosyvoice_dir: str = ""
    tts_navoiy_base_model_dir: str = ""
    tts_navoiy_checkpoint: str = ""
    tts_navoiy_reference_audio: str = ""
    tts_navoiy_emotion: str = "warm"
    tts_output_dir: Path = Path("/var/lib/univoice/audio")
    tts_supported_languages: tuple[str, ...] = ("uz",)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

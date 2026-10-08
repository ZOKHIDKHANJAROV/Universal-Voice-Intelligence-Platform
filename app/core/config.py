from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "UniVoice AI"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    # When set, every /api/v1 route requires the X-API-Key header.
    api_key: str = ""

    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1
    # Re-decode at higher temperatures when output looks wrong. Off by default:
    # it rarely rescues a short phone utterance and can multiply latency.
    stt_temperature_fallback: bool = False
    stt_max_concurrency: int = 1
    # Load and warm up the STT models when the service starts. Otherwise the
    # first caller after a restart waits ~12 s while they load.
    stt_preload: bool = True
    # Optional decoder prompt for STT_MODEL. Empty by default: all measured
    # configurations (docs/stt-evaluation.md) ran without one.
    stt_initial_prompt: str = ""
    # Optional Uzbek fine-tune (faster-whisper format). Uzbek speech goes to it,
    # forced to "uz"; STT_MODEL keeps detecting the language and handles Russian.
    stt_model_uz: str = ""
    # Fine-tunes trained without prompts continue the prompt instead of
    # transcribing, so the Uzbek model gets none by default.
    stt_initial_prompt_uz: str = ""
    # Fine-tunes trained on plain text drop the first words when asked for
    # timestamps (75% -> 20% WER on affected clips without them).
    stt_without_timestamps_uz: bool = True
    stt_supported_languages: tuple[str, ...] = ("uz", "ru", "en")
    # Languages a phone caller is expected to speak; detection is clamped to these.
    stt_realtime_languages: tuple[str, ...] = ("uz", "ru")
    stt_max_upload_bytes: int = 25 * 1024 * 1024

    intent_min_confidence: float = 0.6

    llm_enabled: bool = False
    llm_base_url: str = "http://ollama:11434"
    llm_model: str = "qwen3:4b"
    llm_timeout_seconds: int = 30
    # Ollama layers to offload to GPU; 0 keeps the LLM on CPU so it does not
    # compete with Whisper for VRAM. -1 lets Ollama decide.
    llm_num_gpu: int = 0
    # Keyword matches at or above this confidence skip the LLM call.
    llm_skip_confidence: float = 0.8

    tts_provider: str = "navoiy-http"
    # Optional separate provider for Russian, e.g. "piper". Empty = tts_provider.
    tts_provider_ru: str = ""
    tts_base_url: str = "http://navoiy-tts:8100"
    tts_timeout_seconds: int = 120
    tts_binary: str = "piper"
    tts_model_path: str = ""
    tts_model_path_ru: str = ""
    tts_python_binary: str = "python"
    tts_navoiy_inference_script: str = ""
    tts_navoiy_cosyvoice_dir: str = ""
    tts_navoiy_base_model_dir: str = ""
    tts_navoiy_checkpoint: str = ""
    tts_navoiy_reference_audio: str = ""
    tts_navoiy_emotion: str = "warm"
    tts_output_dir: Path = Path("/var/lib/univoice/audio")
    # Pre-rendered 8 kHz phrases played on calls (see scripts/render_prompts.py).
    tts_prompt_cache_dir: Path = Path("/var/lib/univoice/audio/prompts")
    tts_supported_languages: tuple[str, ...] = ("uz", "ru")

    # Call log behind the /monitor page (SQLite, no extra service).
    call_log_path: Path = Path("data/calls.db")
    call_log_retention_days: int = 30
    # Keep each caller utterance as 8 kHz WAV to replay it in the monitor.
    # Off by default: callers must be told when they are recorded.
    call_record_audio: bool = False
    call_audio_dir: Path = Path("data/calls")

    # Realtime telephony audio bridge
    realtime_audio_host: str = "0.0.0.0"
    realtime_audio_port: int = 9019
    realtime_vad_mode: int = 2
    realtime_silence_ms: int = 700
    realtime_min_utterance_ms: int = 400
    realtime_max_utterance_ms: int = 12000
    # Consecutive 20 ms speech frames needed to interrupt playback.
    realtime_barge_in_frames: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

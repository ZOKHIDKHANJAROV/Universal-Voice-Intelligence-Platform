# UniVoice AI

Universal Voice Intelligence Platform for voice automation and AI-assisted call handling.

## Current scope

- Scenario Engine for controlled business flows
- Asterisk / SIP voice gateway
- Speech-to-Text (STT)
- Intent detection and LLM routing
- Text-to-Speech (TTS)
- PostgreSQL persistence
- Call history and analytics

## Architecture

Caller -> Asterisk / SIP -> STT -> Intent / LLM -> Scenario Engine -> TTS -> Caller

## STT

The current STT layer uses faster-whisper with a provider abstraction. The default configuration is CPU-based small Whisper with int8 compute type and Uzbek (uz) as the default language.

The STT API accepts an audio file and returns transcription text, detected language, language probability, and duration.

### Endpoint

POST /api/v1/stt/transcribe?language=uz

Example:

    curl -X POST \
      -F "audio=@sample.wav" \
      "http://127.0.0.1:8000/api/v1/stt/transcribe?language=uz"

The first transcription initializes/downloads the configured Whisper model. Model files should be cached in the runtime environment.

## Run locally

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    uvicorn app.main:app --reload

## Run with Docker

    docker compose up --build

## Test

    pytest

## API

- GET /health
- GET /api/v1/scenarios
- GET /api/v1/scenarios/{scenario_id}
- POST /api/v1/scenarios/resolve
- POST /api/v1/stt/transcribe

## Roadmap

1. Scenario Engine
2. Telephony gateway / Asterisk
3. STT abstraction and Uzbek speech recognition
4. Intent detection and LLM routing
5. TTS abstraction and Uzbek speech synthesis
6. PostgreSQL persistence
7. Call history and analytics
8. Admin panel
9. Production deployment and observability

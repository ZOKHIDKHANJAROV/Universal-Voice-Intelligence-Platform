# UniVoice AI

Universal Voice Intelligence Platform for voice automation and AI-assisted call handling.

## Current scope

Phase 1 of the platform focuses on a deterministic Scenario Engine that can later be connected to:

- Asterisk / SIP telephony
- Speech-to-Text (STT)
- LLM-based intent detection
- Text-to-Speech (TTS)
- PostgreSQL
- Redis

The first API is intentionally runnable without external AI providers.

## Architecture

```
Caller / Telephony
       |
       v
   Voice Gateway
       |
       v
      STT
       |
       v
 Intent / LLM
       |
       v
 Scenario Engine
       |
       v
      TTS
       |
       v
     Caller
```

The current repository implements the **Scenario Engine + REST API**.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload
```

Open:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

## Run with Docker

```bash
docker compose up --build
```

## Test

```bash
pytest
```

## API

- `GET /health`
- `GET /api/v1/scenarios`
- `GET /api/v1/scenarios/{scenario_id}`
- `POST /api/v1/scenarios/resolve`

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

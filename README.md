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

Pressing **0**, or asking for an operator, ends the bot's part of the call:
the dialplan asks the API what to do next (`/internal/calls/{uuid}/next`) and
dials `UNIVOICE_OPERATOR`; if nobody answers, the caller returns to the bot.

Asterisk streams call audio to the API over AudioSocket (port 9019). The API
segments speech with webrtcvad, transcribes it with Whisper, matches a scenario,
and plays the answer back. Answers are fixed scenario texts, so their audio is
rendered once ahead of time and played from disk; the GPU TTS model does not
run during calls.

## Hardware budget

The default `.env.example` is sized for a single 8 GB GPU (e.g. an RTX 4060
Laptop) that also drives a desktop:

| Component | Where | VRAM |
|---|---|---|
| Whisper `large-v3`, `int8_float16`, beam 1 | API container, during calls | ~2.0 GB (measured) |
| Uzbek Whisper medium fine-tune (`STT_MODEL_UZ`) | API container, during calls | ~0.8 GB |
| Navoiy / CosyVoice2 TTS | only while running `make render-prompts` | ~3 GB |
| Ollama LLM (optional) | CPU by default (`LLM_NUM_GPU=0`) | 0 |

Do not drop to a smaller Whisper model to save resources: `small` does not
recognize Uzbek at all (on the sample recordings it gives `uz` a detection
probability of 0.000 and transcribes Uzbek speech as Turkish or Russian). Without
a GPU, keep `large-v3` with `STT_DEVICE=cpu`, `STT_COMPUTE_TYPE=int8` (slower
replies) and build with `--build-arg INSTALL_GPU=0`.

## STT

The STT layer uses faster-whisper with a provider abstraction. Code defaults
are CPU-based small Whisper with int8, which is only good enough for tests;
`.env.example` switches to GPU `large-v3`, which Uzbek needs.

On calls, 8 kHz telephony audio is resampled to the 16 kHz Whisper expects,
and language detection is limited to `STT_REALTIME_LANGUAGES` (Uzbek and
Russian by default).

Stock Whisper barely understands Uzbek over the phone (88% word error rate).
`make uz-model` downloads and converts an Uzbek fine-tune; with
`STT_MODEL_UZ` set, Uzbek speech goes to it and `STT_MODEL` keeps handling
Russian. That brings Uzbek to 22% WER with Russian unchanged at 4.5%.
`make uz-finetune` further adapts it to conversational Tashkent-dialect
speech (73.5% -> 34.5% WER on podcasts, read speech unchanged). That model is
published as [ZOKHID/univoice-stt-uz-ct2](https://huggingface.co/ZOKHID/univoice-stt-uz-ct2)
(faster-whisper, the `.env.example` default) and
[ZOKHID/univoice-stt-uz](https://huggingface.co/ZOKHID/univoice-stt-uz)
(transformers weights and LoRA adapter). See
[docs/stt-evaluation.md](docs/stt-evaluation.md) for the measurements and
how to re-run them on your own recordings.

The STT API accepts an audio file and returns transcription text, detected language, language probability, and duration.

### Endpoint

POST /api/v1/stt/transcribe?language=uz

Example:

    curl -X POST \
      -F "audio=@sample.wav" \
      "http://127.0.0.1:8000/api/v1/stt/transcribe?language=uz"

The first transcription initializes/downloads the configured Whisper model. In
Docker the model is kept in the `whisper-models` volume, or in the host cache
when `HF_CACHE_DIR` is set.

## Call audio (pre-rendered prompts)

Everything the bot says comes from `app/data/scenarios.json` and
`app/data/prompts.json`. Render it to 8 kHz WAV after changing either file:

    make render-prompts

This starts the Navoiy TTS container, renders missing phrases into
`TTS_PROMPT_CACHE_DIR`, and stops the container again to free the GPU. Files
are named by a hash of their text, so only edited phrases are re-rendered. Use
`python -m scripts.render_prompts --dry-run` to see what is missing. A phrase
that was never rendered is synthesized live on first use, which only works
while the TTS service is running.

## Call flow

1. Greeting in Uzbek (Navoiy voice), then Russian (Piper voice).
2. The caller describes the problem; the bot answers from the matching
   scenario, then asks "anything else?". Every prompt and answer plays to the
   end: speech during it is not heard (`REALTIME_BARGE_IN=true` allows
   interrupting).
3. Pressing 0 or asking for an operator transfers the call at any time.

The bot hangs up after a goodbye when:

| Situation | Setting (0 = never) |
|---|---|
| Silence after the greeting: asks once more, then goodbye | `REALTIME_NO_INPUT_SECONDS=15` |
| Silence after an answer and "anything else?" | `REALTIME_FOLLOWUP_SECONDS=8` |
| The call reaches its length limit | `REALTIME_MAX_CALL_SECONDS=300` |

Asterisk also hangs up when no audio arrives for 30 s (`rtp_timeout` in
`infrastructure/asterisk/pjsip.conf`), e.g. when a mobile caller loses
coverage. Phones that send no audio during silence are handled: missing audio
counts as silence.

## Web console

Open `http://localhost:8000/` once the API runs. It is a single static page
(`app/web/index.html`, no build step) for testing and tuning the bot:

- **Речь** — record from the microphone or upload a file; it goes through the
  call path (8 kHz G.711 simulation, language detection, STT, scenario) and
  shows the transcript, scenario, confidence, per-stage timings and plays the
  bot's answer.
- **Текст** — type a caller phrase and see which scenario fires, live.
- **Сценарии** — edit problem keywords, context keywords and the answer text;
  changes apply immediately and are saved to `app/data/scenarios.json`.
  New answer text needs `make render-prompts`.
- **Доступ** — the `API_KEY` to send, if one is set.

Browsers only allow the microphone on `localhost` or over HTTPS.

## Call monitor

`http://localhost:8000/monitor` shows every call as a chat, live (Server-Sent
Events): what the caller said (transcript, language and confidence, STT time,
optional recording), what the bot answered (scenario, confidence, end-to-end
latency) and system events (greeting, barge-in, key presses, transfer,
"operators busy", hangup). Console tests appear as calls marked "тест". The
header shows the last 24 hours: calls, active now, share of answers that hit
a scenario, transfers and median answer latency.

The log is SQLite (`CALL_LOG_PATH`, default `data/calls.db`), kept for
`CALL_LOG_RETENTION_DAYS` (30). `CALL_RECORD_AUDIO=true` also keeps each caller
utterance as 8 kHz WAV for replay; it is off by default because callers must be
told when they are recorded.

## Run locally

On a single Windows laptop with a GPU, see [docs/local-run.md](docs/local-run.md)
(Asterisk in Docker, API on the host, one-command start).

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -e ".[dev]"          # add ,gpu for CUDA: ".[dev,gpu]"
    uvicorn app.main:app --reload

## Run with Docker

    docker compose up --build        # Uzbek STT model downloads on first start
    make render-prompts              # once, and after editing phrases

The `api` container serves the console at <http://localhost:8000/> and the
call monitor at <http://localhost:8000/monitor>, uses the GPU, and mounts
`./models` and `./data` so the same `.env` paths work on the host and in the
container. To reuse host downloads on a slow network (`HF_CACHE_DIR`,
`CUDA_LIBS_IMAGE`), see [docs/local-run.md](docs/local-run.md#everything-in-docker).

## Test

    pytest

## API

Set `API_KEY` to require an `X-API-Key` header on all `/api/v1` routes.

- GET /health
- GET /api/v1/scenarios
- GET /api/v1/scenarios/{scenario_id}
- POST /api/v1/scenarios/resolve
- POST /api/v1/intent/resolve
- POST /api/v1/stt/transcribe
- POST /api/v1/tts/synthesize
- POST /api/v1/voice/process
- PUT /api/v1/scenarios/{scenario_id}
- POST /api/v1/console/analyze
- GET /api/v1/console/prompts
- GET /api/v1/console/system

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

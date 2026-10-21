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

## Answer modes and admin panel

<http://localhost:8000/admin> switches how the bot answers; the change applies
to the next caller phrase, no restart.

| | Minimum | Maximum |
|---|---|---|
| Pipeline | language detection → scenario → pre-recorded answer | language detection → knowledge base search (RAG) → LLM answer → live voice |
| Answer time | ~1 s | a few seconds; "one moment" plays meanwhile |
| Needs | Whisper, rendered phrases | also Ollama with the model, Navoiy (uz) and Piper (ru) running |

Maximum mode answers only from the knowledge base (edited in the panel; it
starts from the scenarios), in the caller's language, in at most two short
sentences, and offers an operator when the base has no answer. If the LLM is
down, times out, answers in the wrong language, or the voice fails, the bot
answers from the recordings as in minimum mode. Requests for an operator
transfer immediately in both modes. The panel also shows what each mode needs
and has a test box that answers typed questions.

Setup for maximum mode on one laptop: [docs/local-run.md](docs/local-run.md#maximum-mode-rag--llm--live-voice).

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

## Observability

`/health` says the API is up; the call monitor (above) shows individual calls.
For production there are Prometheus metrics, JSON logs and an optional
Prometheus + Grafana stack.

### Metrics: `GET /metrics`

Prometheus text format. With `API_KEY` set, send it as `X-API-Key` or
`Authorization: Bearer <key>`. Phone calls are labelled `source="phone"`, web
console tests `source="console"`.

| Metric | What it counts |
|---|---|
| `univoice_active_calls` | phone calls connected to the bot now |
| `univoice_calls_total{source}` | calls started (a caller back from a failed transfer is not a new call) |
| `univoice_call_ends_total{reason}` | bot sessions ended: `caller_hangup`, `transferred`, `done`, `silence`, `time_limit`, `error` |
| `univoice_stage_seconds{stage,source}` | histogram per caller turn: `vad` (last voiced frame -> utterance to STT, the end-of-speech wait), `stt`, `intent`, `rag` and `llm` (maximum mode), `tts` (reply audio ready), `end_to_end` (caller stops talking -> reply ready) |
| `univoice_answers_total{source,outcome}` | turns answered: `scenario` hit, `not_recognized`, `llm` (maximum mode) |
| `univoice_scenario_hits_total{scenario}` | answers per scenario |
| `univoice_unheard_utterances_total{source}` | utterances STT turned into no text |
| `univoice_operator_transfers_total{trigger}` | hand-overs: `dtmf` (pressed 0) or `intent` (asked for an operator) |
| `univoice_operator_unavailable_total` | transfers no operator answered |
| `univoice_llm_answers_total{source,result}` | maximum mode: `ok`, or `fallback` to the pre-recorded answer |
| `univoice_stt_language_total{source,language}` | language each utterance was transcribed as |
| `univoice_stt_route_total{route,language}` | with `STT_MODEL_UZ`: `forced`, `detector`, `specialist` (Uzbek fine-tune sure), `compared`, `no_speech` |
| `univoice_errors_total{stage}` | failures: `session`, `utterance`, `prompt`, `keepalive` |
| `univoice_http_requests_total{method,route,status}`, `univoice_http_request_seconds` | API requests by route template |
| `univoice_gpu_memory_used_bytes`, `univoice_gpu_memory_total_bytes` | per GPU, from NVML (`nvidia-ml-py`, in the `gpu` extra); absent without an NVIDIA GPU |

Process metrics (`process_*`, `python_*`) come with the Prometheus client.

To check it: start the API, send a recording from the console
(<http://localhost:8000/>), then

    curl -s localhost:8000/metrics | grep -E '^univoice_(calls|answers|stage_seconds_(sum|count))'

### Logs

One JSON object per line on stdout (`LOG_FORMAT=json`, the default;
`LOG_FORMAT=text` for reading in a terminal). Every line written while a call
is handled carries `call_uuid`: the Asterisk call UUID, the same key as in
`/monitor`, including lines from STT worker threads and faster-whisper itself.
Lines before Asterisk sends the UUID, and per-connection details, have
`call_id` (our id for the AudioSocket connection). Console tests use one id for
both. uvicorn's access and error lines go through the same handler.

    {"ts": "2026-10-21T04:08:07.098+00:00", "level": "INFO", "logger": "univoice.realtime", "message": "STT call_id=... text='...' lang=uz confidence=0.833", "call_uuid": "61383f58-466a-4d28-930d-4a861de88e3a", "call_id": "2c596ffb-..."}

To follow one call: `docker compose logs api | grep <call_uuid>`.

### Prometheus and Grafana (optional)

    docker compose --profile monitoring up -d

- Prometheus <http://localhost:9090> scrapes `/metrics` every 15 s, keeps 30
  days and evaluates the alert rules in
  [infrastructure/monitoring/alerts.yml](infrastructure/monitoring/alerts.yml).
- Grafana <http://localhost:3000> (`admin` / `GRAFANA_ADMIN_PASSWORD`, default
  `admin`) opens on the provisioned "UniVoice AI" dashboard: active calls,
  calls and answers per hour, scenario hit rate, top scenarios, transfers, LLM
  fallbacks, latency per stage (p50/p90/p99), STT languages and routing,
  errors, HTTP 5xx and GPU memory. Edit
  [the dashboard JSON](infrastructure/monitoring/grafana/dashboards/univoice.json)
  in git; UI changes are not saved.
- Prometheus sends `API_KEY` from `.env` as a Bearer token. When the API runs
  on the host (`scripts/start-local.ps1`) set
  `METRICS_TARGET=host.docker.internal:8000`.

Alert rules (thresholds are starting points):

| Alert | Fires when |
|---|---|
| `UniVoiceDown` | `/metrics` unreachable for 2 min |
| `UniVoiceCallsNotAnswered` | 3+ phone calls in 30 min and not one answered turn |
| `UniVoiceSlowAnswers` | p90 end-to-end (end of speech -> reply) above 3.5 s for 10 min |
| `UniVoiceErrors` | over 10% of call steps fail (15 min) |
| `UniVoiceHttpErrors` | over 5% of API requests return 5xx (10 min) |
| `UniVoiceLlmFallbacks` | most maximum-mode answers fall back to pre-recorded ones |
| `UniVoiceGpuMemoryFull` | GPU memory above 95% for 10 min |

No Alertmanager is bundled: alerts show in Prometheus and Grafana. To get
Telegram or e-mail messages, add an Alertmanager and point Prometheus at it.

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
- GET /metrics (Prometheus; see [Observability](#observability))
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

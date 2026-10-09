# Running UniVoice on one Windows laptop

Tested on an RTX 4060 Laptop (8 GB). Asterisk runs in Docker; the API runs on
the host so it can use the GPU without rebuilding the Docker image.

```
softphone ──SIP/RTP──> Asterisk (Docker, :5060) ──AudioSocket──> API (host, :9019)
                                                                 ├─ STT  large-v3 + Uzbek fine-tune (GPU)
browser ──HTTP──> http://localhost:8000/ (console) ──────────────┤─ scenarios
                                                                 └─ pre-rendered answers (data/prompts)
```

## One-time setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,gpu,tts-ru]"
make piper-ru            # Russian voice for the bot's Russian answers
```

`.env` for this machine:

```env
STT_MODEL=large-v3
STT_DEVICE=cuda
STT_COMPUTE_TYPE=int8_float16
# Downloads once from Hugging Face (~0.74 GB).
STT_MODEL_UZ=ZOKHID/univoice-stt-uz-ct2
STT_INITIAL_PROMPT=
TTS_PROMPT_CACHE_DIR=data/prompts
TTS_PROVIDER_RU=piper
TTS_BINARY=.venv/Scripts/piper.exe
TTS_MODEL_PATH_RU=models/piper/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx
```

Render the bot's phrases once, and again after editing answer texts. Uzbek
needs the Navoiy GPU service (stop the API first; both together are tight in
8 GB), Russian uses Piper on CPU:

```powershell
docker compose -p universal-voice-intelligence-platform --profile tts up -d --wait navoiy-tts
$env:TTS_BASE_URL = "http://127.0.0.1:8100"
.\.venv\Scripts\python.exe -m scripts.render_prompts --language uz
docker compose -p universal-voice-intelligence-platform --profile tts stop navoiy-tts
.\.venv\Scripts\python.exe -m scripts.render_prompts --language ru
```

## Every day

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1
```

This starts Asterisk and the API. Then:

- **Console:** <http://localhost:8000/>. Test speech or text, edit scenarios.
- **Phone call:** in a softphone (e.g. MicroSIP) add an account with server
  `127.0.0.1`, user `1000`, password `change-me`, UDP, and call `1000`.
- **Operator:** register a second softphone (or a second account) as `1001`
  with the same password. Pressing **0** or saying "operator" / "оператор"
  during a call rings `1001`; if nobody answers within 45 s the caller goes
  back to the bot, which says all operators are busy and keeps helping.
  With a provider trunk, point `UNIVOICE_OPERATOR` at a real number, e.g.
  `PJSIP/998901234567@trunk`.
  For a softphone on another device in the same network, set
  `external_media_address` and `external_signaling_address` in
  `infrastructure/asterisk/pjsip.conf` to the laptop's LAN IP and use that IP
  as the server. A wrong signaling address shows up as calls that drop after
  exactly 32 s: the phone's ACK goes to an address it cannot reach.
- After restarting Asterisk, re-register the softphone (or wait for it to do
  so) before calling.

Stop with Ctrl+C, then
`docker compose -p universal-voice-intelligence-platform stop asterisk`.

## Everything in Docker

The API and web UI (console and `/monitor`) also run as the `api` container
with the GPU, so no `.venv` is needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-local.ps1 -Docker
```

The container uses the same `.env`. `./models` and `./data` are mounted at
`/app/models` and `/app/data`, so relative paths such as
`models/whisper-medium-uzbek-podcasts-ct2` and `data/prompts` work unchanged,
and the call log is the same one the host API writes. Add to `.env` to avoid
large downloads on a slow network:

```env
# Whisper models already downloaded on the host (~3 GB for large-v3).
HF_CACHE_DIR=C:/Users/<you>/.cache/huggingface
HF_HUB_DISABLE_XET=1
# cuBLAS/cuDNN from the local Navoiy image instead of ~1.2 GB of pip wheels.
INSTALL_GPU=0
CUDA_LIBS_IMAGE=universal-voice-intelligence-platform-navoiy-tts
```

Only one API can listen on port 8000: stop the host one (Ctrl+C) before
starting the container, and the container
(`docker compose -p universal-voice-intelligence-platform stop api`) before
going back to the host. Logs:
`docker compose -p universal-voice-intelligence-platform logs -f api`.

## Notes

- `-p universal-voice-intelligence-platform` reuses the images and volumes
  built from the main checkout, including the 6 GB of Navoiy models.
- The first ~10 s after start the models load and warm up; the console shows
  "модели загружены" when ready.
- The in-app terminal of the Claude desktop app drops capital letters from
  pasted text; use a regular PowerShell window for these commands.
- `change-me` is a development password. Change it before exposing port 5060.

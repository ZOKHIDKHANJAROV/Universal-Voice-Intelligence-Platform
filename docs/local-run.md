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
  For a softphone on another device in the same network, set
  `external_media_address` in `infrastructure/asterisk/pjsip.conf` to the
  laptop's LAN IP and use that IP as the server.

Stop with Ctrl+C, then
`docker compose -p universal-voice-intelligence-platform stop asterisk`.

## Notes

- `-p universal-voice-intelligence-platform` reuses the images and volumes
  built from the main checkout, including the 6 GB of Navoiy models.
- The first ~10 s after start the models load and warm up; the console shows
  "модели загружены" when ready.
- The in-app terminal of the Claude desktop app drops capital letters from
  pasted text; use a regular PowerShell window for these commands.
- `change-me` is a development password. Change it before exposing port 5060.

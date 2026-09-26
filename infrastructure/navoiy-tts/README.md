# Navoiy TTS service

This service runs the Uzbek Navoiy TTS checkpoint on a CUDA GPU through the
CosyVoice2-0.5B runtime.

## Reference voice

Place a short, clean WAV recording from a speaker who has explicitly consented
to voice cloning at:

`infrastructure/navoiy-tts/reference.wav`

The file is intentionally not committed to the repository.

The Navoiy model release requires CUDA and uses the CosyVoice2-0.5B base model.
The service downloads both model artifacts into the `navoiy-models` Docker
volume on first startup.

## Start

```powershell
docker compose up -d --build
docker compose logs -f navoiy-tts
```

The first startup downloads several gigabytes of model data and initializes the
GPU model. Subsequent starts reuse the Docker volume.

## Test

```powershell
Invoke-RestMethod http://localhost:8100/health
```

Then test the full voice pipeline through the API:

```powershell
curl.exe -X POST -F "audio=@D:\Universal-Voice-Intelligence-Platform\test_uz.wav" "http://localhost:8000/api/v1/voice/process?language=uz" -o "voice_result.wav"
```

# Navoiy TTS service

This service runs the Uzbek Navoiy TTS checkpoint on a CUDA GPU through the
CosyVoice2-0.5B runtime.

## Reference voice

Place a short, clean Uzbek WAV recording from a speaker who has explicitly
consented to voice cloning at:

`infrastructure/navoiy-tts/reference.wav`

Recommended reference:

- 5-15 seconds
- Uzbek speech
- one speaker
- clean voice without music or background noise
- WAV
- mono
- 16 kHz or 24 kHz

The reference audio is intentionally not committed to Git. It is a runtime
asset and may contain a person's biometric voice data.

### Generate a test reference from UzbekVoice

For development/testing, the repository includes a helper that selects clips
from the Apache-2.0 licensed UzbekVoice dataset and creates a single-speaker
16 kHz mono WAV:

```bash
python scripts/create_navoiy_reference.py
```

Install the helper dependencies first:

```bash
python -m pip install datasets soundfile numpy librosa
```

The script creates:

`infrastructure/navoiy-tts/reference.wav`

Review the resulting audio before using it for voice cloning.

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

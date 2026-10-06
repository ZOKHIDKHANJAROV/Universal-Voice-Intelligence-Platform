# Uzbek TTS

The TTS layer supports replaceable providers.

## How TTS is used on calls

Calls never wait on TTS. Every phrase is fixed text (scenario messages in
`app/data/scenarios.json`, system prompts in `app/data/prompts.json`), so
`scripts/render_prompts.py` synthesizes each one once, resamples it to 8 kHz
telephony audio with an anti-aliasing filter, and stores it in
`TTS_PROMPT_CACHE_DIR`. The realtime bridge loads these files into memory at
startup. See "Call audio" in the README for the workflow.

## Provider per language

`TTS_PROVIDER` is used for every language unless `TTS_PROVIDER_RU` is set.
CosyVoice2 does not officially list Russian, and Navoiy is fine-tuned on
Uzbek, so listen to the rendered Russian prompts. If they are poor, render
Russian with a CPU Piper voice instead:

```env
TTS_PROVIDER=navoiy-http
TTS_PROVIDER_RU=piper
TTS_BINARY=piper
TTS_MODEL_PATH_RU=/models/piper/ru_RU-irina-medium.onnx
```

Since rendering happens offline, this choice costs nothing at call time.

## Navoiy

Navoiy TTS is an Uzbek checkpoint built on CosyVoice2-0.5B. It requires the CosyVoice runtime, the Navoiy checkpoint, CUDA, and a reference WAV.

Model card: https://huggingface.co/aisha-org/navoiy-tts

The Navoiy model generates 24 kHz audio. The upstream setup requires CUDA, so this provider is intended for a GPU deployment.

The Docker service in `infrastructure/navoiy-tts` (`TTS_PROVIDER=navoiy-http`)
accepts `uz` and `ru`. Uzbek text goes through the `uztts` normalizer;
Russian text is passed to the model unchanged. The service is in the `tts`
compose profile and is only started for rendering.

To run the direct provider instead, install the pinned CosyVoice revision and
Navoiy runtime according to the model card, then configure:

```env
TTS_PROVIDER=navoiy
TTS_PYTHON_BINARY=python
TTS_NAVOIY_INFERENCE_SCRIPT=/opt/navoiy-tts/inference.py
TTS_NAVOIY_COSYVOICE_DIR=/opt/CosyVoice
TTS_NAVOIY_BASE_MODEL_DIR=/opt/CosyVoice/pretrained_models/CosyVoice2-0.5B
TTS_NAVOIY_CHECKPOINT=/opt/navoiy-tts/emotion_600h_joint.pt
TTS_NAVOIY_REFERENCE_AUDIO=/models/reference.wav
TTS_NAVOIY_EMOTION=warm
```

The reference voice must be used with the speaker's permission.

# Uzbek TTS

The TTS layer supports replaceable providers.

## Navoiy

Navoiy TTS is an Uzbek checkpoint built on CosyVoice2-0.5B. It requires the CosyVoice runtime, the Navoiy checkpoint, CUDA, and a reference WAV.

Model card: https://huggingface.co/aisha-org/navoiy-tts

The Navoiy model generates 24 kHz audio. The upstream setup requires CUDA, so this provider is intended for a GPU deployment.

Install the pinned CosyVoice revision and Navoiy runtime according to the model card, then configure:

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

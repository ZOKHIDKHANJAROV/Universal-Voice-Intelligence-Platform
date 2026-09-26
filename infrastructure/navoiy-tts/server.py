import os
import random
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


COSYVOICE_DIR = Path("/opt/CosyVoice")
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))

BASE_MODEL_DIR = Path(os.getenv("NAVOIY_BASE_MODEL_DIR", "/models/CosyVoice2-0.5B"))
NAVOIY_DIR = Path(os.getenv("NAVOIY_MODEL_DIR", "/models/navoiy"))
CHECKPOINT = Path(
    os.getenv("NAVOIY_CHECKPOINT", str(NAVOIY_DIR / "emotion_600h_joint.pt"))
)
EMOTIONS_FILE = Path(
    os.getenv("NAVOIY_EMOTIONS_FILE", str(NAVOIY_DIR / "emotions_40h.json"))
)
REFERENCE_AUDIO = Path(
    os.getenv("NAVOIY_REFERENCE_AUDIO", "/models/reference.wav")
)
MODEL_CACHE = Path(os.getenv("NAVOIY_MODEL_CACHE", "/models/hf-cache"))
BASE_REVISION = os.getenv(
    "NAVOIY_BASE_REVISION",
    "eec1ae6c79877dbd9379285cf8789c9e0879293d",
)
NAVOIY_REVISION = os.getenv(
    "NAVOIY_MODEL_REVISION",
    "912e6bb6663f11e5dec5de830ca06ea8dd4c985f",
)
DEFAULT_EMOTION = os.getenv("NAVOIY_EMOTION", "warm")
DEFAULT_SPEED = float(os.getenv("NAVOIY_SPEED", "1.0"))

app = FastAPI(title="UniVoice Navoiy TTS", version="0.1.0")
_engine = None
_engine_lock = threading.Lock()
_init_error: Exception | None = None


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    language: str = "uz"
    emotion: str | None = None
    speed: float | None = Field(default=None, gt=0.0, le=2.0)


def _ensure_models() -> None:
    from huggingface_hub import snapshot_download

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "FunAudioLLM/CosyVoice2-0.5B",
        revision=BASE_REVISION,
        local_dir=BASE_MODEL_DIR,
        cache_dir=MODEL_CACHE,
    )
    snapshot_download(
        "aisha-org/navoiy-tts",
        revision=NAVOIY_REVISION,
        local_dir=NAVOIY_DIR,
        allow_patterns=[
            "emotion_600h_joint.pt",
            "emotions_40h.json",
            "uztts/*",
        ],
        cache_dir=MODEL_CACHE,
    )


def _load_engine():
    global _engine, _init_error

    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine

        try:
            _ensure_models()

            import torch
            from cosyvoice.cli.cosyvoice import CosyVoice2
            from uztts.normalize import normalize

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for Navoiy TTS")

            import json

            emotions = json.loads(EMOTIONS_FILE.read_text(encoding="utf-8"))
            emotion_map = {}
            for entry in emotions:
                emotion_map[entry["uz"].lower()] = entry
                for tag in entry["tag"].replace("[", " ").replace("]", " ").split():
                    emotion_map[tag.lower()] = entry

            model = CosyVoice2(
                str(BASE_MODEL_DIR.resolve()),
                load_jit=False,
                load_trt=False,
                fp16=True,
            )

            state = torch.load(
                CHECKPOINT,
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(state, dict):
                for key in ("state_dict", "model", "llm"):
                    nested = state.get(key)
                    if isinstance(nested, dict):
                        state = nested
                        break

            incompatible = model.model.llm.load_state_dict(state, strict=False)
            if incompatible.missing_keys:
                print(
                    f"Warning: {len(incompatible.missing_keys)} missing checkpoint keys",
                    flush=True,
                )
            if incompatible.unexpected_keys:
                print(
                    f"Warning: {len(incompatible.unexpected_keys)} unexpected checkpoint keys",
                    flush=True,
                )

            model.model.llm.eval()
            _engine = (model, normalize, emotion_map)
            _init_error = None
            return _engine
        except Exception as exc:
            _init_error = exc
            raise


@app.on_event("startup")
def startup() -> None:
    try:
        _load_engine()
    except Exception as exc:
        print(f"Navoiy TTS initialization failed: {exc}", flush=True)


@app.get("/health")
def health():
    if _engine is not None:
        return {"ok": True, "ready": True, "gpu": True}
    return {
        "ok": True,
        "ready": False,
        "error": str(_init_error) if _init_error else "initializing",
    }


@app.post("/v1/synthesize")
def synthesize(request: SynthesizeRequest):
    if request.language != "uz":
        raise HTTPException(status_code=400, detail="Only Uzbek TTS is supported")

    if not REFERENCE_AUDIO.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Reference audio is missing. Put a consented Uzbek reference WAV at "
                "infrastructure/navoiy-tts/reference.wav"
            ),
        )

    try:
        model, normalize, emotions = _load_engine()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TTS initialization failed: {exc}") from exc

    emotion_name = (request.emotion or DEFAULT_EMOTION).lower().strip("[]")
    emotion = emotions.get(emotion_name)
    if emotion is None:
        raise HTTPException(status_code=400, detail=f"Unknown emotion: {emotion_name}")

    speed = request.speed or DEFAULT_SPEED
    text = normalize(request.text, mode="infer")
    instruction = emotion["instruct"].strip() + "<|endofprompt|>"

    import torch
    import torchaudio

    random.seed(1986)
    torch.manual_seed(1986)
    torch.cuda.manual_seed_all(1986)

    output_dir = Path("/tmp/univoice-tts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"response-{random.getrandbits(64):016x}.wav"

    chunks = []
    with torch.inference_mode():
        for result in model.inference_instruct2(
            text,
            instruction,
            str(REFERENCE_AUDIO.resolve()),
            stream=False,
            speed=speed,
        ):
            chunks.append(result["tts_speech"].detach().cpu())

    if not chunks:
        raise HTTPException(status_code=500, detail="Navoiy TTS returned no audio")

    audio = torch.cat(chunks, dim=1)
    torchaudio.save(str(output_path), audio, 24000)

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="response.wav",
    )

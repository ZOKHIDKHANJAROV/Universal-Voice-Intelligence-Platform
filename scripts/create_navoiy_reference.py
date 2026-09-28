#!/usr/bin/env python3
"""
Create infrastructure/navoiy-tts/reference.wav from UzbekVoice.

The generated reference is intentionally kept out of Git. It is a local
runtime asset for Navoiy/CosyVoice voice cloning.

The script disables Hugging Face audio decoding while streaming. This avoids
the PyTorch/TorchCodec dependency in the Windows development environment.
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

OUTPUT = Path("infrastructure/navoiy-tts/reference.wav")
DATASET = "DavronSherbaev/uzbekvoice-filtered"

MIN_TOTAL_SECONDS = 7.0
MAX_TOTAL_SECONDS = 12.0
MIN_CLIP_SECONDS = 2.0
MAX_CLIP_SECONDS = 8.0
TARGET_SAMPLE_RATE = 16_000


def is_uzbek(item: dict) -> bool:
    value = str(item.get("native_language", "")).lower()
    return "o'zbek" in value or "uzbek" in value or "ўзбек" in value


def decode_audio(audio: dict) -> tuple[np.ndarray, int]:
    """Decode raw audio bytes/path with soundfile, not TorchCodec."""
    audio_bytes = audio.get("bytes")
    if audio_bytes:
        samples, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False,
        )
        return np.asarray(samples, dtype=np.float32), int(sample_rate)

    audio_path = audio.get("path")
    if audio_path:
        path = Path(audio_path)
        if path.exists():
            samples, sample_rate = sf.read(
                path,
                dtype="float32",
                always_2d=False,
            )
            return np.asarray(samples, dtype=np.float32), int(sample_rate)

    raise RuntimeError(
        "The dataset returned audio without decodable bytes or a local path."
    )


def main() -> None:
    print(f"Loading {DATASET} in streaming mode...")

    dataset = load_dataset(
        DATASET,
        split="train",
        streaming=True,
    )

    # Critical: decode(False) is the supported streaming API for exposing
    # media paths/bytes without constructing TorchCodec audio decoders.
    dataset = dataset.decode(False)

    speakers: dict[str, list[dict]] = defaultdict(list)
    checked = 0

    for item in dataset:
        checked += 1
        duration = float(item.get("duration", 0.0))

        if not MIN_CLIP_SECONDS <= duration <= MAX_CLIP_SECONDS:
            continue
        if not is_uzbek(item):
            continue

        speaker = item.get("client_id")
        if not speaker:
            continue

        speakers[str(speaker)].append(item)

        if len(speakers) >= 100:
            break

    print(f"Checked approximately {checked} records.")
    print(f"Candidate speakers: {len(speakers)}")

    selected: tuple[str, list[dict], float] | None = None

    for speaker, clips in speakers.items():
        total = 0.0
        chosen: list[dict] = []

        for clip in sorted(
            clips,
            key=lambda x: float(x.get("duration", 0.0)),
        ):
            duration = float(clip.get("duration", 0.0))

            if total + duration > MAX_TOTAL_SECONDS:
                continue

            chosen.append(clip)
            total += duration

            if total >= MIN_TOTAL_SECONDS:
                break

        if MIN_TOTAL_SECONDS <= total <= MAX_TOTAL_SECONDS:
            selected = (speaker, chosen, total)
            break

    if selected is None:
        raise RuntimeError(
            "No suitable single-speaker Uzbek reference was found."
        )

    speaker, clips, _ = selected
    parts: list[np.ndarray] = []

    print(f"Selected speaker: {speaker}")

    for index, clip in enumerate(clips, 1):
        samples, sample_rate = decode_audio(clip["audio"])

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        if sample_rate != TARGET_SAMPLE_RATE:
            import librosa

            samples = librosa.resample(
                samples,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE,
            )

        parts.append(samples.astype(np.float32, copy=False))

        print(
            f"  {index}. {float(clip['duration']):.2f}s "
            f"{clip.get('sentence', '')}"
        )

    final_audio = np.concatenate(parts)
    peak = float(np.max(np.abs(final_audio))) if final_audio.size else 0.0

    if peak > 0:
        final_audio = final_audio / peak * 0.95

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        OUTPUT,
        final_audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )

    print()
    print(f"Created: {OUTPUT}")
    print(f"Duration: {len(final_audio) / TARGET_SAMPLE_RATE:.2f}s")
    print("Format: WAV PCM_16, mono, 16 kHz")
    print()
    print("Review the recording before using it for voice cloning.")


if __name__ == "__main__":
    main()

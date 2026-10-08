#!/usr/bin/env python3
"""
Export clips from a Hugging Face audio parquet shard to WAV + eval_stt manifest.

Used to hold out one shard of a training corpus as a test set in the same
domain (e.g. conversational Tashkent-dialect podcasts).

    python -m scripts.prepare_parquet data/train/podcasts_tashkent/data/train-00003-of-00026.parquet \
        data/eval/podcasts_uz --language uz --limit 100
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import wave
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def decode(audio: dict) -> np.ndarray:
    from faster_whisper import decode_audio

    return decode_audio(io.BytesIO(audio["bytes"]), sampling_rate=16000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("shard", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", default="uz")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()

    rows = [row for row in read_rows(args.shard) if (row.get("text") or "").strip()]
    random.Random(args.seed).shuffle(rows)
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    for row in rows:
        if len(kept) >= args.limit:
            break
        samples = decode(row["audio"])
        if not 0.5 <= len(samples) / 16000 <= args.max_seconds:
            continue
        name = f"{Path(args.shard).stem}-{row.get('id', len(kept))}.wav"
        with wave.open(str(audio_dir / name), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        kept.append((f"audio/{name}", " ".join(row["text"].split())))

    with (args.output / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["audio", "text", "language", "scenario"])
        writer.writerows([audio, text, args.language, ""] for audio, text in kept)
    print(f"{len(kept)} clips -> {args.output / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

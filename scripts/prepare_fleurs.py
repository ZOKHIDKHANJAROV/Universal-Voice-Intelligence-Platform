#!/usr/bin/env python3
"""
Turn a downloaded Google FLEURS split into an eval_stt manifest.

FLEURS is read speech on general topics, so it measures how well a model
knows the language, not our call vocabulary. A fixed random subset keeps CPU
evaluation affordable and comparable between models.

    python -m scripts.prepare_fleurs data/eval/fleurs_uz --language uz --limit 60
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tarfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="folder holding raw/data/<lang>/ from the HF repo")
    parser.add_argument("--language", default="uz")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=60, help="clips to keep, 0 = all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = next((args.directory / "raw" / "data").glob(f"{args.language}_*"))
    audio_dir = args.directory / "audio"
    if not audio_dir.exists():
        with tarfile.open(raw / "audio" / f"{args.split}.tar.gz") as archive:
            archive.extractall(audio_dir, filter="data")

    with (raw / f"{args.split}.tsv").open(encoding="utf-8") as file:
        # id, file name, raw transcription, normalized transcription, ...
        rows = [line.rstrip("\n").split("\t") for line in file if line.strip()]

    # FLEURS repeats each sentence with several speakers; keep one per sentence.
    by_sentence: dict[str, list[str]] = {}
    for row in rows:
        by_sentence.setdefault(row[0], row)
    clips = sorted(by_sentence.values(), key=lambda row: row[1])
    if args.limit:
        clips = random.Random(args.seed).sample(clips, min(args.limit, len(clips)))

    manifest = args.directory / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["audio", "text", "language", "scenario"])
        for row in clips:
            path = next(audio_dir.rglob(row[1]))
            writer.writerow([path.relative_to(args.directory).as_posix(), row[2], args.language, ""])
    print(f"{len(clips)} clips -> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

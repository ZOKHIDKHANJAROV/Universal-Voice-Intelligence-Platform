#!/usr/bin/env python3
"""
Convert a Hugging Face Whisper checkpoint into a faster-whisper model.

Needs `pip install torch transformers` (the CPU build of torch is enough).
The result is what STT_MODEL should point to.

    python -m scripts.convert_whisper models/_src/whisper-medium-uzbek models/whisper-medium-uzbek-ct2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="local folder or HF repo id with a transformers Whisper model")
    parser.add_argument("output", type=Path)
    # int8_float16 weights run as int8 on CPU and int8/float16 on GPU.
    parser.add_argument("--quantization", default="int8_float16")
    args = parser.parse_args()

    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperTokenizerFast

    source = Path(args.source)
    # faster-whisper reads tokenizer.json; some fine-tunes only ship vocab/merges,
    # and silently falling back to the stock tokenizer would be wrong if the
    # vocabulary was changed.
    if source.is_dir() and not (source / "tokenizer.json").exists():
        WhisperTokenizerFast.from_pretrained(source).save_pretrained(source)

    copy_files = [name for name in ("tokenizer.json", "preprocessor_config.json")
                  if not source.is_dir() or (source / name).exists()]
    if args.output.exists():
        shutil.rmtree(args.output)
    TransformersConverter(args.source, copy_files=copy_files).convert(
        str(args.output), quantization=args.quantization
    )
    size = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file())
    print(f"{args.output}: {size / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

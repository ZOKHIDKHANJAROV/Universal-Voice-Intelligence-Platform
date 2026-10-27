#!/usr/bin/env python3
"""
Convert a Hugging Face Whisper checkpoint to a quantized whisper.cpp model.

The offline Android app runs whisper.cpp, which reads ggml files. This turns
the Uzbek fine-tune (transformers weights) into one:

    python -m scripts.convert_whisper_ggml --model ZOKHID/univoice-stt-uz \\
        --quantize-bin path/to/whisper-quantize

Steps: download the checkpoint, run whisper.cpp's own converter
(models/convert-h5-to-ggml.py from the mobile/android/third_party/whisper.cpp
submodule) to an f16 ggml file, then quantize it (q5_0 by default, ~0.5 GB for
medium). The converter wants the openai/whisper repository only for its mel
filter bank; that bank is written here from the checkpoint's own feature
extractor instead, so no extra clone is needed.

`whisper-quantize` is built from the same submodule:

    cmake -S mobile/android/third_party/whisper.cpp -B build/whisper -DCMAKE_BUILD_TYPE=Release
    cmake --build build/whisper --target whisper-quantize -j

Needs `pip install transformers torch numpy` (CPU torch is enough).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHISPER_CPP = ROOT / "mobile/android/third_party/whisper.cpp"


def fetch(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.json", "*.txt", "*.safetensors"]))


def write_mel_filters(model_dir: Path, whisper_root: Path) -> None:
    """whisper/assets/mel_filters.npz as openai/whisper ships it: mel_<n> = (n_mels, n_fft/2+1)."""
    import numpy as np
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    filters = np.asarray(extractor.mel_filters, dtype=np.float32).T
    assets = whisper_root / "whisper" / "assets"
    assets.mkdir(parents=True)
    np.savez_compressed(assets / "mel_filters.npz", **{f"mel_{filters.shape[0]}": filters})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="ZOKHID/univoice-stt-uz", help="HF repo id or local directory")
    parser.add_argument("--out", type=Path, default=ROOT / "models/ggml")
    parser.add_argument("--name", default="univoice-stt-uz")
    parser.add_argument("--quantize", default="q5_0", help="ggml type, or 'none' to keep f16")
    parser.add_argument("--quantize-bin", type=Path, help="whisper.cpp's whisper-quantize executable")
    parser.add_argument("--whisper-cpp", type=Path, default=WHISPER_CPP)
    args = parser.parse_args()

    converter = args.whisper_cpp / "models" / "convert-h5-to-ggml.py"
    if not converter.exists():
        sys.exit(f"{converter} not found: run `git submodule update --init mobile/android/third_party/whisper.cpp`")
    if args.quantize != "none" and not (args.quantize_bin and args.quantize_bin.exists()):
        sys.exit("--quantize-bin is required (build whisper-quantize, see the script header) or pass --quantize none")

    model_dir = fetch(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    f16 = args.out / f"ggml-{args.name}-f16.bin"
    with tempfile.TemporaryDirectory() as temp:
        whisper_root, converted = Path(temp) / "openai-whisper", Path(temp) / "out"
        converted.mkdir()
        write_mel_filters(model_dir, whisper_root)
        subprocess.run([sys.executable, str(converter), str(model_dir), str(whisper_root), str(converted)], check=True)
        # move, not replace: the temp directory may be on another disk.
        shutil.move(converted / "ggml-model.bin", f16)
    print(f"f16: {f16} ({f16.stat().st_size / 1e6:.0f} MB)")

    final = f16
    if args.quantize != "none":
        final = args.out / f"ggml-{args.name}-{args.quantize}.bin"
        subprocess.run([str(args.quantize_bin), str(f16), str(final), args.quantize], check=True)

    print(f"model:  {final}\nsize:   {final.stat().st_size}\nsha256: {sha256(final)}")
    print("Put size and sha256 into Models.UZ (mobile/android/.../stt/ModelStore.kt) once the file is published.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

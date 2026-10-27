#!/usr/bin/env python3
"""
Render the bot's fixed answers for the offline Android app.

Writes one WAV per scenario answer and per phrase in app/data/prompts.json to
mobile/android/app/src/main/assets/prompts, plus manifest.json that maps
(scenario id or prompt key, language) to its file. The phone plays these
instead of synthesizing speech.

Phones play wideband audio, so phrases are rendered at 24 kHz with the same
TTS services as `make render-prompts` (Navoiy for Uzbek, Piper for Russian;
they must be running). Without them, --from-cache reuses the 8 kHz call
recordings from TTS_PROMPT_CACHE_DIR (telephone quality, but it works).

    python -m scripts.export_mobile_prompts
    python -m scripts.export_mobile_prompts --from-cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from app.audio.resample import pcm16_to_wav, wav_to_pcm16
from app.services.scenario_service import ScenarioService
from app.tts.prompt_cache import TELEPHONY_RATE, get_prompt_cache
from app.tts.prompts import get_prompts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "mobile/android/app/src/main/assets/prompts"


def phrases() -> list[tuple[str, str, str]]:
    """(key, language, text) for everything the phone can say."""
    items = []
    for scenario in ScenarioService().list_scenarios():
        if scenario.steps:
            items.append((scenario.id, scenario.language, scenario.steps[0].message))
    for key, by_language in get_prompts().items():
        for language, text in by_language.items():
            items.append((key, language, text))
    return items


def render(text: str, language: str, rate: int, from_cache: bool) -> bytes:
    """PCM16 mono at ``rate``."""
    if from_cache:
        cached = get_prompt_cache().get_pcm8(text, language)
        if cached is None:
            raise FileNotFoundError("not in the call prompt cache; run `make render-prompts` first")
        return wav_to_pcm16(pcm16_to_wav(cached, TELEPHONY_RATE), rate)
    # Imported here so --from-cache works without the TTS dependencies.
    from app.tts.pronunciation import spoken_text
    from app.tts.service import synthesize_bytes

    return wav_to_pcm16(synthesize_bytes(spoken_text(text, language), language), rate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rate", type=int, default=24000, help="sample rate of the phone audio")
    parser.add_argument("--from-cache", action="store_true", help="reuse the 8 kHz call recordings")
    parser.add_argument("--force", action="store_true", help="re-render files that already exist")
    parser.add_argument("--language", help="with --force, re-render only this language, e.g. ru")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    entries, failures = [], 0
    for key, language, text in phrases():
        digest = hashlib.sha1(f"{language}\n{text}\n{args.rate}".encode("utf-8")).hexdigest()[:16]
        name = f"{language}-{digest}.wav"
        path = args.out / name
        label = f"[{language}] {key}"
        force = args.force and args.language in (None, language)
        if path.exists() and not force:
            print(f"  ok      {label}")
        else:
            try:
                path.write_bytes(pcm16_to_wav(render(text, language, args.rate, args.from_cache), args.rate))
                print(f"  render  {label}")
            except (OSError, ValueError) as exc:
                failures += 1
                print(f"  FAILED  {label}: {exc}")
                continue
        entries.append({"key": key, "language": language, "text": text, "file": name})

    used = {entry["file"] for entry in entries}
    for stale in args.out.glob("*.wav"):
        if stale.name not in used:
            stale.unlink()
            print(f"  pruned  {stale.name}")
    manifest = {"sample_rate": args.rate, "entries": entries}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in args.out.glob("*.wav"))
    print(f"{len(entries)} phrase(s), {size / 1e6:.1f} MB -> {args.out}")
    if failures:
        print(f"{failures} phrase(s) failed; are the TTS services running? (or use --from-cache)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

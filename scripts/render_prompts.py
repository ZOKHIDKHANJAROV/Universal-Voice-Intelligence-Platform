#!/usr/bin/env python3
"""
Pre-render every phrase the voice bot can say to 8 kHz telephony WAV.

Calls then play audio from disk instead of waiting on the GPU TTS model, so
the TTS service only needs to run while this script does. Re-run it after
editing app/data/scenarios.json or app/data/prompts.json; unchanged phrases
are skipped because files are keyed by their text.

    python -m scripts.render_prompts            # render what is missing
    python -m scripts.render_prompts --dry-run  # list status only
    python -m scripts.render_prompts --force    # re-render everything
    python -m scripts.render_prompts --prune    # also delete stale files
"""

from __future__ import annotations

import argparse
import sys
import time

from app.core.config import get_settings
from app.tts.prompt_cache import get_prompt_cache
from app.tts.prompts import all_phrases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-render existing files")
    parser.add_argument("--dry-run", action="store_true", help="only report what is missing")
    parser.add_argument("--prune", action="store_true", help="delete files no phrase uses")
    parser.add_argument("--language", help="only phrases in this language, e.g. ru")
    args = parser.parse_args()

    cache = get_prompt_cache()
    phrases = all_phrases()
    if args.language:
        phrases = [(text, lang) for text, lang in phrases if lang == args.language]
    directory = get_settings().tts_prompt_cache_dir
    print(f"{len(phrases)} phrase(s), cache: {directory}")

    failures = 0
    for text, language in phrases:
        path = cache.path_for(text, language)
        exists = path.exists()
        label = f"[{language}] {text[:60]}{'...' if len(text) > 60 else ''}"
        if args.dry_run or (exists and not args.force):
            print(f"  {'ok     ' if exists else 'MISSING'} {label}")
            continue
        started = time.perf_counter()
        try:
            cache.render(text, language, force=args.force)
        except (OSError, ValueError) as exc:
            failures += 1
            print(f"  FAILED  {label}\n          {exc}")
            continue
        print(f"  render  {label} ({time.perf_counter() - started:.1f}s)")

    if args.prune and not args.dry_run and directory.exists():
        # Every phrase counts as used, even with --language: pruning by the
        # filtered list would delete the other languages' audio.
        used = {cache.path_for(text, language).name for text, language in all_phrases()}
        for stale in directory.glob("*.wav"):
            if stale.name not in used:
                stale.unlink()
                print(f"  pruned  {stale.name}")

    if failures:
        print(f"{failures} phrase(s) failed; is the TTS service running?")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

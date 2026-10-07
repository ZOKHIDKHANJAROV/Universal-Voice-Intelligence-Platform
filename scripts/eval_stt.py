#!/usr/bin/env python3
"""
Measure STT quality the way callers experience it.

Every clip is degraded like a phone call (8 kHz + G.711 mu-law), run through
the same realtime path as live calls, and scored on word/character error rate
and on whether the right scenario was picked. Use it to compare Whisper
models before switching STT_MODEL, and before and after any fine-tuning.

Manifest: a CSV with columns  audio,text,language[,scenario]
  audio     path relative to the manifest (any format ffmpeg/PyAV reads)
  text      what was really said (Uzbek in Latin script)
  language  uz or ru
  scenario  expected scenario id; empty means "should not match"

    python -m scripts.eval_stt data/eval/manifest.csv
    python -m scripts.eval_stt data/eval/manifest.csv --model navai-uz/whisper-medium-uzbek-ct2
    python -m scripts.eval_stt data/eval/manifest.csv --language uz   # Uzbek-only models
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from app.audio.resample import resample
from app.core.config import get_settings
from app.intent.service import get_intent_service
from app.services.scenario_service import normalize_text, to_uzbek_latin

TELEPHONY_RATE = 8000
_MU = 255.0


def telephony(audio16: np.ndarray, mu_law: bool = True) -> np.ndarray:
    """Downsample to 8 kHz and round-trip through 8-bit mu-law like G.711."""
    audio = np.clip(resample(audio16, 16000, TELEPHONY_RATE), -1.0, 1.0)
    if not mu_law:
        return audio
    compressed = np.sign(audio) * np.log1p(_MU * np.abs(audio)) / np.log1p(_MU)
    quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
    return (np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(_MU)) / _MU).astype(np.float32)


def comparable(text: str, language: str) -> str:
    return to_uzbek_latin(text) if language == "uz" else normalize_text(text)


def error_rates(reference: str, hypothesis: str, language: str) -> tuple[float, float]:
    ref, hyp = comparable(reference, language), comparable(hypothesis, language)
    ref_words = ref.split()
    wer = Levenshtein.distance(ref_words, hyp.split()) / max(1, len(ref_words))
    cer = Levenshtein.distance(ref, hyp) / max(1, len(ref))
    return wer, cer


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    missing = {"audio", "text", "language"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    for row in rows:
        row["audio"] = str((path.parent / row["audio"]).resolve())
    return rows


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", default=settings.stt_model, help="name, HF repo or local CTranslate2 dir")
    parser.add_argument(
        "--model-uz",
        default=settings.stt_model_uz,
        help="Uzbek specialist; --model then detects the language and handles the rest",
    )
    parser.add_argument("--device", default=settings.stt_device)
    parser.add_argument("--compute-type", default=settings.stt_compute_type)
    parser.add_argument("--beam-size", type=int, default=settings.stt_beam_size)
    parser.add_argument("--language", help="force one language instead of detecting it")
    parser.add_argument(
        "--initial-prompt",
        default=settings.stt_initial_prompt,
        help='decoder prompt; "" disables it (fine-tunes may continue the prompt instead of transcribing)',
    )
    parser.add_argument("--no-mu-law", action="store_true", help="skip G.711 simulation")
    parser.add_argument("--output", type=Path, help="write per-clip results as CSV")
    args = parser.parse_args()

    from faster_whisper import decode_audio

    from app.speech.providers.routed import LanguageRoutedSpeechToText
    from app.speech.providers.whisper import FasterWhisperSpeechToText

    rows = load_manifest(args.manifest)
    started = time.perf_counter()

    def load(name: str, prompt: str, without_timestamps: bool = False) -> FasterWhisperSpeechToText:
        return FasterWhisperSpeechToText(
            name,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            initial_prompt=prompt,
            without_timestamps=without_timestamps,
        )

    stt = load(args.model, args.initial_prompt)
    if args.model_uz:
        uzbek = load(args.model_uz, settings.stt_initial_prompt_uz, settings.stt_without_timestamps_uz)
        stt = LanguageRoutedSpeechToText(stt, {"uz": uzbek})
    print(f"prompt: {args.initial_prompt!r}")
    print(f"model {args.model}{' + uz ' + args.model_uz if args.model_uz else ''} "
          f"on {args.device}/{args.compute_type}: loaded in {time.perf_counter() - started:.1f}s")
    intent = get_intent_service()
    allowed = settings.stt_realtime_languages

    results = []
    for row in rows:
        phone = telephony(decode_audio(row["audio"], sampling_rate=16000), not args.no_mu_law)
        started = time.perf_counter()
        result = stt.transcribe_pcm16(phone, TELEPHONY_RATE, language=args.language, allowed_languages=allowed)
        seconds = time.perf_counter() - started

        language = result.language if result.language in allowed else allowed[0]
        wer, cer = error_rates(row["text"], result.text, row["language"])
        expected = (row.get("scenario") or "").strip() or None
        scenario = intent.resolve(result.text, language).scenario_id
        results.append({
            "audio": Path(row["audio"]).name,
            "language": row["language"],
            "detected": result.language,
            "wer": wer,
            "cer": cer,
            "seconds": seconds,
            "audio_seconds": len(phone) / TELEPHONY_RATE,
            "expected": expected or "",
            "scenario": scenario or "",
            "intent_ok": scenario == expected if "scenario" in row else "",
            "reference": row["text"],
            "hypothesis": result.text,
        })
        mark = {True: "ok ", False: "BAD", "": "   "}[results[-1]["intent_ok"]]
        print(f"{mark} WER {wer:6.1%}  {result.language}  {seconds:5.2f}s  {Path(row['audio']).name}\n"
              f"      ref: {row['text']}\n      hyp: {result.text}")

    print()
    for language in sorted({r["language"] for r in results}):
        group = [r for r in results if r["language"] == language]
        words = sum(len(comparable(r["reference"], language).split()) for r in group)
        wer = sum(r["wer"] * len(comparable(r["reference"], language).split()) for r in group) / max(1, words)
        detected_ok = sum(r["detected"] == language for r in group)
        print(f"[{language}] {len(group)} clips  WER {wer:.1%}  "
              f"CER {statistics.mean(r['cer'] for r in group):.1%}  "
              f"language right {detected_ok}/{len(group)}")
    scored = [r for r in results if r["intent_ok"] != ""]
    if scored:
        print(f"scenario right {sum(r['intent_ok'] for r in scored)}/{len(scored)}")
    rtf = sum(r["seconds"] for r in results) / max(1e-9, sum(r["audio_seconds"] for r in results))
    latencies = sorted(r["seconds"] for r in results)
    print(f"latency median {statistics.median(latencies):.2f}s  "
          f"max {latencies[-1]:.2f}s  real-time factor {rtf:.2f}")

    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

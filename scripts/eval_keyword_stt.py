#!/usr/bin/env python3
"""
Compare speech engines by the scenario they lead to, not by WER.

For the offline phone app only the chosen scenario matters: the answer is a
pre-recorded file. This runs every clip of a manifest through

  - vosk-full:    a small Vosk model with its whole vocabulary;
  - vosk-grammar: the same model limited to the words the scenarios use
                  (every vocabulary word a scenario keyword matches, the rest
                  becomes [unk]);
  - vosk-auto:    both models listen, the more confident scenario wins (what
                  the phone does when it does not know the caller's language);
  - server:       optional, the running API's /console/analyze (Whisper),

then through ScenarioService with the intent threshold, and prints how many
clips reached the expected scenario (or no scenario, for expected = empty).

    python -m scripts.eval_keyword_stt data/eval/scenario_tts/phrases.json \\
        --vosk-uz models/vosk/vosk-model-small-uz-0.22 \\
        --vosk-ru models/vosk/vosk-model-small-ru-0.22 --server http://localhost:8000

Needs `pip install vosk` and the models unpacked from alphacephei.com/vosk/models.

The manifest is phrases.json (id, text, language, scenario; audio = <id>.wav
next to it) or a CSV with audio,text,language,scenario like eval_stt.py reads.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import struct
import sys
import time
import wave
from pathlib import Path

from app.services.scenario_service import ScenarioService, _token_matches, normalize_text, to_uzbek_latin

MIN_CONFIDENCE = 0.6


def load_manifest(path: Path) -> list[dict]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [{**r, "audio": path.parent / f"{r['id']}.wav"} for r in rows]
    with path.open(encoding="utf-8") as file:
        return [
            {**r, "audio": path.parent / r["audio"], "scenario": r.get("scenario") or None}
            for r in csv.DictReader(file)
        ]


def pcm16k(path: Path) -> bytes:
    """Mono 16-bit PCM at 16 kHz, resampled with the app's own resampler."""
    from app.audio.resample import wav_to_pcm16

    return wav_to_pcm16(path.read_bytes(), 16000)


def _fst_symbols(path: Path) -> list[str]:
    """Output symbol table of an OpenFST file (small Vosk models keep their words only there)."""
    data = path.read_bytes()
    position = data.find(struct.pack("<i", 2125658996))  # SymbolTable magic
    if position < 0:
        raise ValueError(f"no symbol table in {path}")

    def string(at: int) -> tuple[str, int]:
        size = struct.unpack_from("<i", data, at)[0]
        return data[at + 4:at + 4 + size].decode("utf-8", "replace"), at + 4 + size

    _, at = string(position + 4)
    _, count = struct.unpack_from("<qq", data, at)
    at += 16
    words = []
    for _ in range(count):
        word, at = string(at)
        at += 8  # key
        words.append(word)
    return words


def vocabulary(model_dir: Path) -> list[str]:
    for name in ("graph/words.txt", "words.txt"):
        path = model_dir / name
        if path.exists():
            words = [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            break
    else:
        words = _fst_symbols(model_dir / "graph" / "Gr.fst")
    return [w for w in words if w.replace("'", "").replace("-", "").isalpha()]


def grammar_words(service: ScenarioService, language: str, vocab: list[str]) -> list[str]:
    """Vocabulary words that some keyword of a ``language`` scenario matches."""
    keyword_words = set()
    for scenario in service.list_scenarios():
        if scenario.language != language:
            continue
        for keyword in (scenario.keywords + scenario.context_keywords
                        + scenario.weak_keywords + scenario.exclude_keywords):
            for raw in keyword.split():
                keyword_words.add(normalize_text(raw.rstrip("*")) + ("*" if raw.endswith("*") else ""))
    convert = to_uzbek_latin if language == "uz" else normalize_text
    chosen = []
    for word in vocab:
        token = convert(word)
        if any(_token_matches(k, token) or _token_matches(k.replace("'", ""), token) for k in keyword_words):
            chosen.append(word)
    return chosen


class Vosk:
    def __init__(self, model_dir: Path, words: list[str] | None) -> None:
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        self.model = Model(str(model_dir))
        self.grammar = json.dumps(words + ["[unk]"], ensure_ascii=False) if words else None

    def transcribe(self, pcm: bytes) -> str:
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self.model, 16000, self.grammar) if self.grammar else KaldiRecognizer(self.model, 16000)
        recognizer.AcceptWaveform(pcm)
        text = json.loads(recognizer.FinalResult()).get("text", "")
        return " ".join(w for w in text.split() if w != "[unk]")


class Auto:
    """Both Vosk models listen; the answer with the higher confidence wins.

    What the phone does when the caller's language is not known (an incoming
    call): a tie goes to the model whose own language the scenario is in.
    """

    def __init__(self, service: ScenarioService, models: dict[str, Vosk]) -> None:
        self.service = service
        self.models = models

    def resolve(self, pcm: bytes) -> tuple[str, str | None, float]:
        best = ("", None, 0.0, -1)
        for language, model in self.models.items():
            text = model.transcribe(pcm)
            scenario, confidence = self.service.resolve(text, language)
            own = 1 if scenario and scenario.language == language else 0
            if scenario and (confidence, own) > (best[2], best[3]):
                best = (text, scenario.id, confidence, own)
            elif not best[0]:
                best = (text, None, 0.0, -1)
        return best[0], best[1], best[2]


def server_transcribe(url: str, path: Path, language: str) -> str:
    import httpx

    with path.open("rb") as audio:
        response = httpx.post(f"{url}/api/v1/console/analyze", files={"audio": audio},
                              data={"language": language, "telephone": "false"}, timeout=300)
    response.raise_for_status()
    return response.json()["transcription"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--vosk-uz", type=Path)
    parser.add_argument("--vosk-ru", type=Path)
    parser.add_argument("--server", help="API base URL, e.g. http://localhost:8000")
    parser.add_argument("--pad", action="store_true",
                        help="add 0.5 s of silence before and 1 s after every clip, as on a real call")
    parser.add_argument("--show", choices=("errors", "all", "none"), default="errors")
    args = parser.parse_args()

    service = ScenarioService()
    rows = load_manifest(args.manifest)
    engines: dict[str, dict[str, object]] = {}
    for language, model_dir in (("uz", args.vosk_uz), ("ru", args.vosk_ru)):
        if model_dir:
            words = grammar_words(service, language, vocabulary(model_dir))
            print(f"[{language}] grammar: {len(words)} words")
            engines.setdefault("vosk-full", {})[language] = Vosk(model_dir, None)
            engines.setdefault("vosk-grammar", {})[language] = Vosk(model_dir, words)
    auto = None
    if args.vosk_uz and args.vosk_ru:
        auto = Auto(service, {"uz": engines["vosk-full"]["uz"], "ru": engines["vosk-full"]["ru"]})
        engines["vosk-auto"] = {"uz": "auto", "ru": "auto"}
    if args.server:
        engines["server"] = {"uz": "server", "ru": "server"}

    totals = {name: {"ok": 0, "n": 0, "seconds": 0.0, "audio": 0.0} for name in engines}
    for row in rows:
        pcm = pcm16k(row["audio"])
        if args.pad:
            pcm = bytes(16000) + pcm + bytes(32000)
        duration = len(pcm) / 32000
        for name, by_language in engines.items():
            engine = by_language.get(row["language"])
            if engine is None:
                continue
            start = time.perf_counter()
            if engine == "auto":
                text, scenario_id, confidence = auto.resolve(pcm)
                seconds = time.perf_counter() - start
                got = scenario_id if confidence >= MIN_CONFIDENCE else None
            else:
                text = (server_transcribe(args.server, row["audio"], row["language"]) if engine == "server"
                        else engine.transcribe(pcm))
                seconds = time.perf_counter() - start
                scenario, confidence = service.resolve(text, row["language"])
                got = scenario.id if scenario and confidence >= MIN_CONFIDENCE else None
            ok = got == (row["scenario"] or None)
            total = totals[name]
            total["ok"] += ok; total["n"] += 1; total["seconds"] += seconds; total["audio"] += duration
            if args.show == "all" or (args.show == "errors" and not ok):
                print(f"{'ok ' if ok else 'BAD'} {name:12} {row['language']} want {str(row['scenario']):22} "
                      f"got {str(got):22} | {row['text'][:45]!r} -> {text[:60]!r}")

    print()
    for name, total in totals.items():
        if total["n"]:
            print(f"{name:12} scenario right {total['ok']:3}/{total['n']:<3} "
                  f"({100 * total['ok'] / total['n']:.0f}%)  speed x{total['seconds'] / total['audio']:.3f} of audio length")
    return 0


if __name__ == "__main__":
    sys.exit(main())

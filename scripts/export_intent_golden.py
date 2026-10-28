#!/usr/bin/env python3
"""
Write the scenario matcher's answers to a golden file for the Android app.

The offline app ports app/services/scenario_service.py to Kotlin; its
ScenarioParityTest replays this file and must pick the same scenario with the
same confidence. Re-run after changing the matcher or app/data/scenarios.json:

    python -m scripts.export_intent_golden
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.scenario_service import ScenarioService, normalize_text, to_uzbek_latin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "mobile/android/app/src/test/resources/intent_golden.json"

# Spellings that exercise the normalizer: Cyrillic and Kazakh-Cyrillic Uzbek,
# Turkish letters, look-alike apostrophes, personal verb forms, short words.
EXTRA = [
    "SUV CHIQMAYAPTI!!!", "suv chiqmayapti aka, iltimos", "Сув чиқмаяпти", "Су шықмаяпти",
    "işlemeyabdi apparat", "Apparat İşlamayapti", "pulimni yeb qo‘ydi", "pulimni yeb qoʻydi",
    "pulimni yeb qo`ydi", "pulimni yeb qoydi", "pulim qaytmadi", "Пулимни еб қўйди",
    "men uyga chiqmayapman", "suv oqib ketyapti", "kitob o'qib o'tiraman", "loyiha haqida gapiraman",
    "suv loyqa", "suvning ta'mi yomon", "Вода грязная", "вода не идёт", "ВОДА НЕ ИДЁТ",
    "деньги списались, а вода не пошла", "водитель приехал", "нет воды", "не льется вода",
    "оператор", "соедините с оператором пожалуйста", "operatorga ulang", "jonli odam bilan gaplashmoqchiman",
    "автомат протекает", "течёт вода из автомата", "apparatdan suv oqyapti", "suv oqmayapti",
    "", "   ", "123", "ok", "Hello", "Привет, как дела", "Salom, yaxshimisiz",
    "Straße", "naïve café",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    service = ScenarioService()
    cases = json.loads((ROOT / "tests/data/intent_cases.json").read_text(encoding="utf-8"))
    texts = [c[0] for group in cases.values() for c in group] + EXTRA
    for scenario in service.list_scenarios():
        texts += scenario.keywords + scenario.context_keywords + scenario.weak_keywords + scenario.exclude_keywords

    seen: set[str] = set()
    rows = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        for language in ("uz", "ru", None):
            scenario, confidence = service.resolve(text, language)
            rows.append({
                "text": text,
                "language": language,
                "scenario_id": scenario.id if scenario else None,
                "confidence": confidence,
            })
        rows.append({"text": text, "normalized": normalize_text(text), "latin": to_uzbek_latin(text)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

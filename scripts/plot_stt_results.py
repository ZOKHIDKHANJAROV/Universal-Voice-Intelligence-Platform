#!/usr/bin/env python3
"""
Draw the README charts for the STT measurements in docs/stt-evaluation.md.

Standard library only; writes SVG files that follow the viewer's light/dark
theme. The numbers below are copied from docs/stt-evaluation.md: update both
together after a new measurement.

    python -m scripts.plot_stt_results            # -> docs/images/*.svg
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

# Fine-tune on Tashkent-dialect podcasts: navai-uz base, 12 h pilot, ~95 h full run.
FINETUNE = {
    "series": ["navai-uz (base)", "pilot, 12 h", "full run, ~95 h"],
    "groups": [
        ("Podcasts WER", [73.5, 48.8, 34.5]),
        ("Podcasts CER", [45.2, 23.0, 15.8]),
        ("FLEURS WER", [17.5, 18.7, 17.3]),
        ("FLEURS CER", [6.4, 6.3, 5.0]),
    ],
}

# Uzbek WER on FLEURS through the simulated phone channel, per STT setup.
SETUPS = [
    ("small alone", 106.7),
    ("large-v3 alone", 87.7),
    ("small + navai-uz", 26.8),
    ("callcenter medium, forced uz", 25.3),
    ("navai-uz alone, forced uz", 24.2),
    ("large-v3 + navai-uz", 22.0),
]

# Share of clips sent to the right language model.
LANGUAGE = {
    "series": ["detector only", "detector + confidence"],
    "groups": [
        ("Uzbek, 2.5 s", [46, 96]),
        ("Russian, 2.5 s", [100, 98]),
        ("Uzbek, up to 8 s", [88, 99]),
        ("Russian, up to 8 s", [100, 100]),
    ],
}

# Reference categorical palette, slots 1-3 (light, dark).
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70")]

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def _style() -> str:
    light = "".join(f"--s{i}:{c[0]};" for i, c in enumerate(SERIES))
    dark = "".join(f"--s{i}:{c[1]};" for i, c in enumerate(SERIES))
    return (
        "<style>"
        f"svg{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;{light}}}"
        "@media (prefers-color-scheme: dark){"
        f"svg{{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#3a3936;{dark}}}}}"
        f"text{{font-family:{FONT};fill:var(--ink2);font-size:12px}}"
        ".title{fill:var(--ink);font-size:15px;font-weight:600}"
        ".value{fill:var(--ink);font-size:11px}"
        ".grid{stroke:var(--grid);stroke-width:1}"
        "</style>"
    )


def _bar(x: float, y: float, w: float, h: float, fill: str, horizontal: bool = False) -> str:
    """A bar with 4 px rounded data end and a square baseline end."""
    r = min(4.0, w / 2, h / 2) if h > 0 and w > 0 else 0
    if horizontal:  # baseline on the left
        d = (f"M{x},{y} h{w - r} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r} "
             f"a{r},{r} 0 0 1 {-r},{r} h{-(w - r)} z")
    else:  # baseline at the bottom
        d = (f"M{x},{y + h} v{-(h - r)} a{r},{r} 0 0 1 {r},{-r} h{w - 2 * r} "
             f"a{r},{r} 0 0 1 {r},{r} v{h - r} z")
    return f'<path d="{d}" fill="{fill}"/>'


def _legend(series: list[str], x: float, y: float) -> str:
    out = []
    for i, name in enumerate(series):
        out.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="var(--s{i})"/>')
        out.append(f'<text x="{x + 15}" y="{y}">{escape(name)}</text>')
        x += 25 + 7 * len(name)
    return "".join(out)


def grouped_bars(data: dict, title: str, subtitle: str, ymax: float, unit: str = "%") -> str:
    width, height = 720, 360
    left, right, top, bottom = 48, 16, 86, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    groups, series = data["groups"], data["series"]
    group_w = plot_w / len(groups)
    bar_w = min(36, (group_w * 0.7) / len(series))
    gap = 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(title)}">',
        _style(),
        f'<rect width="{width}" height="{height}" fill="var(--bg)" rx="8"/>',
        f'<text class="title" x="{left - 32}" y="26">{escape(title)}</text>',
        f'<text x="{left - 32}" y="46">{escape(subtitle)}</text>',
        _legend(series, left - 32, 70),
    ]
    step = 20 if ymax <= 100 else 25
    for v in range(0, int(ymax) + 1, step):
        y = top + plot_h - v / ymax * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y}" y2="{y}"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4}" text-anchor="end">{v}{unit}</text>')

    for g, (label, values) in enumerate(groups):
        cluster = len(series) * bar_w + (len(series) - 1) * gap
        x0 = left + g * group_w + (group_w - cluster) / 2
        for s, value in enumerate(values):
            h = value / ymax * plot_h
            x = x0 + s * (bar_w + gap)
            y = top + plot_h - h
            parts.append(_bar(x, y, bar_w, h, f"var(--s{s})"))
            parts.append(f'<text class="value" x="{x + bar_w / 2}" y="{y - 5}" '
                         f'text-anchor="middle">{value:g}</text>')
        parts.append(f'<text x="{left + g * group_w + group_w / 2}" y="{height - bottom + 20}" '
                     f'text-anchor="middle">{escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def horizontal_bars(rows: list[tuple[str, float]], title: str, subtitle: str,
                    highlight: str, xmax: float) -> str:
    width = 720
    left, right, top, row_h = 200, 56, 66, 34
    height = top + row_h * len(rows) + 30
    plot_w = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(title)}">',
        _style(),
        f'<rect width="{width}" height="{height}" fill="var(--bg)" rx="8"/>',
        f'<text class="title" x="16" y="26">{escape(title)}</text>',
        f'<text x="16" y="46">{escape(subtitle)}</text>',
    ]
    bottom_y = top + row_h * len(rows)
    for v in range(0, int(xmax) + 1, 20):
        x = left + v / xmax * plot_w
        parts.append(f'<line class="grid" x1="{x}" x2="{x}" y1="{top - 4}" y2="{bottom_y}"/>')
        parts.append(f'<text x="{x}" y="{bottom_y + 18}" text-anchor="middle">{v}%</text>')
    for i, (label, value) in enumerate(rows):
        y = top + i * row_h + 6
        h = row_h - 12
        w = min(value, xmax) / xmax * plot_w
        fill = "var(--s0)" if label == highlight else "var(--ink2)"
        opacity = "" if label == highlight else ' opacity="0.45"'
        parts.append(_bar(left, y, w, h, fill, horizontal=True).replace("/>", f"{opacity}/>"))
        weight = ' style="font-weight:600;fill:var(--ink)"' if label == highlight else ""
        parts.append(f'<text x="{left - 8}" y="{y + h / 2 + 4}" text-anchor="end"{weight}>'
                     f'{escape(label)}</text>')
        parts.append(f'<text class="value" x="{left + w + 6}" y="{y + h / 2 + 4}">{value:g}%</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    charts = {
        "stt-finetune.svg": grouped_bars(
            FINETUNE, "Fine-tuning on conversational Uzbek",
            "Error rate, lower is better. 8 kHz G.711 phone audio, forced uz, greedy.", 80),
        "stt-setups.svg": horizontal_bars(
            SETUPS, "Uzbek word error rate by STT setup",
            "FLEURS uz, 60 clips through the simulated phone channel. Lower is better.",
            "large-v3 + navai-uz", 110),
        "stt-language.svg": grouped_bars(
            LANGUAGE, "Clips routed to the right language model",
            "150 FLEURS ru, 100 podcast and 100 FLEURS uz clips, phone channel. Higher is better.",
            100),
    }
    for name, svg in charts.items():
        (args.out / name).write_text(svg, encoding="utf-8")
        print(args.out / name)


if __name__ == "__main__":
    main()

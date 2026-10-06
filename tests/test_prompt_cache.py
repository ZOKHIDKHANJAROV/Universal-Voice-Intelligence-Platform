import wave

import numpy as np

import app.tts.prompt_cache as prompt_cache
from app.audio.resample import pcm16_to_wav
from app.tts.prompts import all_phrases, prompt_text


def _fake_tts(calls: list):
    def synthesize_bytes(text: str, language: str) -> bytes:
        calls.append((text, language))
        tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)
        return pcm16_to_wav((tone * 32767).astype("<i2").tobytes(), 24000)

    return synthesize_bytes


def test_renders_once_to_telephony_wav(tmp_path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(prompt_cache, "synthesize_bytes", _fake_tts(calls))
    cache = prompt_cache.PromptAudioCache(tmp_path)

    pcm8 = cache.get_or_render_pcm8("Salom", "uz")
    assert len(pcm8) == 2 * 8000  # one second at 8 kHz, 16-bit

    with wave.open(str(cache.path_for("Salom", "uz")), "rb") as wav:
        assert wav.getframerate() == 8000
        assert wav.getnchannels() == 1

    # A fresh cache instance reads from disk instead of synthesizing again.
    again = prompt_cache.PromptAudioCache(tmp_path).get_or_render_pcm8("Salom", "uz")
    assert again == pcm8
    assert calls == [("Salom", "uz")]


def test_editing_text_invalidates_audio(tmp_path) -> None:
    cache = prompt_cache.PromptAudioCache(tmp_path)
    assert cache.path_for("Salom", "uz") != cache.path_for("Salom!", "uz")
    assert cache.path_for("Salom", "uz") != cache.path_for("Salom", "ru")


def test_preload_reports_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(prompt_cache, "synthesize_bytes", _fake_tts([]))
    cache = prompt_cache.PromptAudioCache(tmp_path)
    cache.render("Bir", "uz")
    assert cache.preload([("Bir", "uz"), ("Ikki", "uz")]) == [("Ikki", "uz")]


def test_all_phrases_cover_prompts_and_scenarios() -> None:
    phrases = all_phrases()
    assert prompt_text("greeting", "uz") in phrases
    assert prompt_text("not_understood", "ru") in phrases
    assert any(language == "ru" and text.startswith("Приносим") for text, language in phrases)
    assert len(phrases) == len(set(phrases))

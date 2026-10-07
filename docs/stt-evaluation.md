# STT evaluation

How well each Whisper setup understands callers, measured the way calls sound:
every clip is downsampled to 8 kHz and passed through G.711 mu-law before
going through the realtime transcription path (`scripts/eval_stt.py`).

## Results

Google FLEURS read speech: 60 Uzbek test sentences, 30 Russian dev sentences,
one speaker per sentence, fixed random subset (`--seed 0`). CPU, `int8`, beam 1,
no decoder prompt. Uzbek is scored after Cyrillic-to-Latin transliteration, so
the script Whisper picks does not count as an error.

| Setup | Uzbek WER | Uzbek CER | Russian WER | Uzbek detected as Uzbek | VRAM |
|---|---|---|---|---|---|
| `large-v3` (previous default) | 87.7% | 29.6% | 4.5% | 56/60 | ~2.0 GB (measured, `int8_float16`) |
| **`large-v3` + `navai-uz/whisper-medium-uzbek`** | **22.0%** | **11.0%** | **4.5%** | 56/60 | ~2.8 GB (estimate) |
| `small` + `navai-uz/whisper-medium-uzbek` | 26.8% | 12.1% | 10.1% | 53/60 | ~1.3 GB (estimate) |
| `navai-uz/whisper-medium-uzbek` alone, forced `uz` | 24.2% | 12.4% | 99.6% | n/a | ~0.8 GB (estimate) |
| `Abduqayum/whisper-uzbek-medium-callcenter` alone, forced `uz` | 25.3% | 10.2% | not run | n/a | ~0.8 GB (estimate) |
| `small` alone | 106.7% | 49.1% | 10.1% | 53/60 | ~0.5 GB (estimate) |

Notes:

- The single-model rows for the Uzbek fine-tunes were run before the
  timestamp fix below; the routed `+ navai` rows were run after it.
- Uzbek fine-tunes have forgotten Russian (they write Russian speech in Uzbek
  Latin) and misdetect Uzbek as Russian on 16/60 clips, hence the routing:
  `STT_MODEL` detects the language and transcribes Russian, `STT_MODEL_UZ`
  transcribes Uzbek.
- Stock Whisper gives Uzbek a detection probability of about 0 and answers
  Azerbaijani or Kazakh, so detection sums probabilities by language family
  (`LANGUAGE_FAMILIES` in `app/speech/providers/whisper.py`).
- The two Uzbek fine-tunes are within noise of each other on this set. The
  call-centre one was trained with simulated phone audio; compare both on
  real call recordings before settling.
- Remaining Uzbek "errors" include numbers written as words
  ("sakkiz yuz ikki" for "802"), which is a spelling difference, not a
  recognition error.

## What changed because of these measurements

| Finding | Effect | Fix |
|---|---|---|
| The fine-tune continued the decoder prompt instead of transcribing ("ga o'tkaziladi") | Uzbek WER 52% instead of 24% | No prompt for the Uzbek model (`STT_INITIAL_PROMPT_UZ` empty) |
| With timestamps the fine-tune dropped the first words ("ishlamayapti" instead of "internet ishlamayapti") | WER 75% instead of 20% on affected clips | `STT_WITHOUT_TIMESTAMPS_UZ=true` |
| Picking the best of uz/ru chose Russian for Uzbek speech | Uzbek written as Cyrillic transliteration | Family scoring |

## Not measured yet

- GPU latency: the dev machine lacks `cublas64_12.dll`; the Docker image
  installs it through the `gpu` extra. CPU latency (median 11 s per
  utterance for `large-v3` + `navai`) is not usable for calls.
- Real call audio and vending vocabulary. FLEURS is read news-style speech.
- Scenario false triggers: on this non-complaint speech 8 of 60 Uzbek clips
  matched a scenario through short keywords such as `loy`, `pul`, `hid`.

## Reproduce

    pip install -e ".[dev,convert]"
    make uz-model
    # FLEURS (HF_HUB_DISABLE_XET=1 if downloads stall)
    python -c "from huggingface_hub import hf_hub_download as d; [d('google/fleurs', f, repo_type='dataset', local_dir='data/eval/fleurs_uz/raw') for f in ['data/uz_uz/test.tsv', 'data/uz_uz/audio/test.tar.gz']]"
    python -m scripts.prepare_fleurs data/eval/fleurs_uz --language uz --limit 60
    python -m scripts.eval_stt data/eval/fleurs_uz/manifest.csv \
        --model large-v3 --model-uz models/whisper-medium-uzbek-ct2 \
        --device cpu --compute-type int8 --initial-prompt ""

Real call recordings go in a manifest of the same shape (`audio,text,language,scenario`)
under `data/eval/`, which is ignored by Git because it holds callers' voices.

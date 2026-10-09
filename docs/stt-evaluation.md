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

## GPU latency

RTX 4060 Laptop, `int8_float16`, `large-v3` + `navai` loaded together,
greedy decoding only (`STT_TEMPERATURE_FALLBACK=false`):

| Utterance length | Median | Max |
|---|---|---|
| under 8 s | 1.2 s | 1.6 s |
| 8–15 s | 1.5 s | 2.3 s |
| 15–30 s | 1.8 s | 2.3 s |

Peak VRAM with both models: ~4.1 GB. With temperature fallback on, one
Uzbek clip misrouted to Russian produced "Субтитры добавил DimaTorzok" and
took 16 s; such subtitle hallucinations are now treated as silence.

## Fine-tuning on conversational speech

`navai-uz` was trained on read speech only. On conversational
Tashkent-dialect podcasts it scores 73.5% WER, against 17.5% on FLEURS.
A LoRA pilot (`scripts/finetune_whisper.py`) on 3 shards (12.2 h) of
`islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset`
(Apache-2.0, Gemini 2.5 Pro transcripts), half the clips degraded to phone
audio, 2 epochs, rank 32, lr 1e-4:

| Test set (GPU, forced `uz`, no timestamps) | `navai-uz` | pilot |
|---|---|---|
| Podcasts, held-out shard, 100 clips | 73.5% WER / 45.2% CER | **48.8% / 23.0%** |
| FLEURS read speech, 60 clips | 17.5% / 6.4% | 18.7% / 6.3% |

Held-out loss fell from 2.31 to 0.90 over 105 steps (52 minutes, 2.96 GiB
peak VRAM).

The full run used all 25 training shards (~14,000 clips, ~95 h), one epoch,
same settings, training while the corpus downloaded (`--expect-shards 25`):
438 steps in 3.2 h on the RTX 4060 Laptop, 2.97 GiB peak VRAM, best held-out
loss 0.628.

| Test set (GPU, forced `uz`, no timestamps) | `navai-uz` | pilot (12 h) | **full (~95 h)** |
|---|---|---|---|
| Podcasts, held-out shard, 100 clips: WER | 73.5% | 48.8% | **34.5%** |
| Podcasts: CER | 45.2% | 23.0% | **15.8%** |
| FLEURS read speech, 60 clips: WER | 17.5% | 18.7% | **17.3%** |
| FLEURS: CER | 6.4% | 6.3% | **5.0%** |

In the production setup (`large-v3` detecting the language and handling
Russian, the full fine-tune for Uzbek): Uzbek FLEURS 19.6% WER, Russian
unchanged at 4.5%, median latency 1.4 s (Uzbek) and 1.6 s (Russian).

Caveats:

- Test transcripts come from the same labeller (Gemini) as the training
  ones, and the held-out shard may share podcast channels and speakers with
  training, so the podcast gain is likely optimistic. Real call recordings
  are the test that matters.
- Gemini labels are cased and punctuated with ASCII apostrophes;
  `normalize_label` rewrites them in the model's own style first (initial
  loss 7.2 -> 3.9 on raw vs rewritten labels), so training adapts to the
  audio rather than to a new spelling style.
- Labels keep numbers as digits while the model writes them as words.

## Scenario false triggers

Better STT made keyword matching fire on ordinary speech: 64 of 380 real
non-complaint sentences (FLEURS and podcast references plus STT output)
matched a scenario, mostly through subject words such as `pul` (money),
`suv` / `вода` (water) or `yomon` (bad), and through two matching bugs:
`oqib` (flowing) fuzzy-matched `oʻqib` (reading), and `loy` (mud) matched
`loyiha` (project). Subject words also pulled other complaints into the
"no water" scenario ("Вода грязная", "Suv loyqa").

Scenarios now separate `keywords` (the problem; at least one must match)
from `context_keywords` (the subject; they only add confidence), stems are
explicit (`chiqma*`), Uzbek stems skip first/second person forms
(`chiqmayapman`, "I am not going out"), short keywords take only short
endings, and fuzzy matching is limited to words of 7+ letters.

| | before | after |
|---|---|---|
| Hand-written complaints routed correctly (`tests/data/intent_cases.json`) | 36/50 | 52/52 |
| Hand-written non-complaints matched | 13/15 | 0/18 |
| Real non-complaint sentences matched | 64/380 | 2/380 |

The two remaining real matches use `loyqa` in its literal sense (silty
ground). The hand-written cases were written together with the keywords;
the 380 real sentences were not looked at while choosing them.

## Choosing the language on short phrases

Real calls showed Uzbek going to the wrong model: on 1-3 s phrases
large-v3's language detector answers Indonesian, English or Dutch, and
Russian then won the uz/ru choice with scores of 0.03-0.3. Large-v3 heard
"suv chiqmayapti" as nothing or "Черт, что я делаю", while the Uzbek model
transcribed the same recordings correctly with mean log-probabilities of
-0.00 to -0.07. On Russian speech and noise its log-probability is -0.5 to
-1.7, and stock Whisper is the reverse.

Below a detector score of 0.9 the models' own confidence now decides: an
Uzbek transcription at -0.25 or better wins outright, otherwise both models
transcribe and the surer one wins; text below -1.0 counts as noise.
Measured on 150 FLEURS ru, 100 podcast and 100 FLEURS uz clips through the
simulated phone channel (`STT_LANGUAGE_SURE`, `STT_ACCEPT_LOGPROB`):

| Correct language | detector only | detector + confidence |
|---|---|---|
| Uzbek, 2.5 s phrase | 92/200 (46%) | 192/200 (96%) |
| Russian, 2.5 s phrase | 150/150 | 147/150 (98%) |
| Uzbek, up to 8 s | 175/200 (88%) | 198/200 (99%) |
| Russian, up to 8 s | 150/150 | 150/150 |

On the 11 recorded utterances of the first real test calls, all six Uzbek
phrases are now transcribed correctly (none were before), the Russian ones
stay Russian, and coughs stay empty. Confident Uzbek skips the large-model
decode: ~0.5 s instead of ~1 s per phrase.

## Not measured yet

- Real call audio and vending vocabulary. FLEURS is read news-style speech.

## Reproduce the fine-tune

Needs a CUDA build of torch, `peft` and `pyarrow` (see the script header).

    # 26 shards, ~10.4 GB; shard 3 is held out for evaluation
    python -c "from huggingface_hub import snapshot_download as s; s('islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset', repo_type='dataset', local_dir='data/train/podcasts_tashkent')"
    make uz-finetune

`make uz-finetune` trains from `models/_src/whisper-medium-uzbek` (left there
by `make uz-model`) and writes `models/whisper-medium-uzbek-podcasts-ct2`;
point `STT_MODEL_UZ` at it.

The result of this run is published:
[ZOKHID/univoice-stt-uz-ct2](https://huggingface.co/ZOKHID/univoice-stt-uz-ct2)
for faster-whisper and
[ZOKHID/univoice-stt-uz](https://huggingface.co/ZOKHID/univoice-stt-uz)
with transformers weights and the LoRA adapter.

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

#!/usr/bin/env python3
"""
LoRA fine-tuning of a Whisper checkpoint that fits an 8 GB laptop GPU.

The base model is frozen in fp16 and only low-rank adapters train, with
gradient checkpointing. Half the clips are degraded like a phone call
(8 kHz + G.711 mu-law) so the model adapts to the channel callers use.
Labels carry <|notimestamps|>, matching how the Uzbek model runs in production.

Needs a CUDA build of torch plus `pip install peft pyarrow`.

    python -m scripts.finetune_whisper \\
        --base models/_src/whisper-medium-uzbek \\
        --train "data/train/podcasts_tashkent/data/train-0000[0-2]-*.parquet" \\
        --eval data/train/podcasts_tashkent/data/train-00003-of-00026.parquet \\
        --output models/ft/podcasts-pilot --epochs 2

Then convert the merged model with scripts/convert_whisper.py and compare it
with scripts/eval_stt.py.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from app.audio.resample import resample
from scripts.eval_stt import telephony

SAMPLE_RATE = 16000
MAX_LABEL_TOKENS = 440  # Whisper decoder holds 448 positions

_APOSTROPHES = "'\u2018\u2019`\u00b4\u02bc\u02bb"


def normalize_label(text: str) -> str:
    """Rewrite a transcript in the style navai-uz was trained on.

    Gemini labels are cased, punctuated and use ASCII apostrophes; the model
    writes lowercase, unpunctuated Uzbek with oʻ/gʻ (U+02BB) and ʼ (U+02BC).
    Training on raw labels spends the update on style: initial loss 7.2 vs
    3.9 after this rewrite, on 30 podcast clips.
    """
    text = " ".join(text.split()).lower()
    text = re.sub(f"([og])[{_APOSTROPHES}]", r"\1" + "\u02bb", text)
    text = re.sub(f"[{_APOSTROPHES.replace(chr(0x2BB), '')}]", "\u02bc", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s\u02bb\u02bc]", " ", text)
    return " ".join(text.split())


def resolve_paths(patterns: list[str]) -> list[str]:
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No parquet files match {patterns}")
    return paths


def _read_rows(paths: list[str]):
    import pyarrow.parquet as pq

    for path in paths:
        yield from pq.read_table(path, columns=["audio", "text"]).to_pylist()


def _usable(row: dict, max_seconds: float, tokenizer):
    """Decode a parquet row into a training clip, or nothing if unusable."""
    from faster_whisper import decode_audio

    text = normalize_label(row["text"] or "")
    if not text or len(tokenizer(text).input_ids) > MAX_LABEL_TOKENS:
        return
    audio = decode_audio(io.BytesIO(row["audio"]["bytes"]), sampling_rate=SAMPLE_RATE)
    # Whisper sees at most 30 s; longer clips would lose audio but keep text.
    if 0.5 <= len(audio) / SAMPLE_RATE <= max_seconds:
        yield {"audio": audio, "text": text}


def iter_clips(paths: list[str], max_seconds: float, tokenizer):
    for row in _read_rows(paths):
        yield from _usable(row, max_seconds, tokenizer)


def stream_epoch(
    patterns: list[str],
    exclude: set[str],
    max_seconds: float,
    tokenizer,
    buffer_shards: int,
    rng: random.Random,
    expect_shards: int = 0,
):
    """One pass over all shards with bounded memory.

    The full corpus does not fit in RAM (25 shards are ~10 GB of compressed
    audio), so shards are visited in random order a few at a time and clips
    are shuffled within that buffer.

    With ``expect_shards`` the glob is re-read before each group, so training
    can start while the corpus is still downloading: it waits only when every
    finished shard has been used and more are expected.
    """
    visited: set[str] = set()
    while True:
        # A shard file appears only once its download has finished.
        matches = {path for pattern in patterns for path in glob.glob(pattern)}
        ready = sorted(matches - visited - exclude)
        if not ready:
            if len(visited) >= expect_shards:
                return
            print(f"waiting for shards ({len(visited)}/{expect_shards} used)", flush=True)
            time.sleep(30)
            continue
        rng.shuffle(ready)
        group = ready[:buffer_shards]
        visited.update(group)
        rows = list(_read_rows(group))
        rng.shuffle(rows)
        for row in rows:
            yield from _usable(row, max_seconds, tokenizer)


def featurize(clip: dict, feature_extractor, tokenizer, telephony_prob: float, train: bool) -> dict:
    audio = clip["audio"]
    # Evaluation always uses the phone channel; training mixes both.
    if not train or random.random() < telephony_prob:
        audio = resample(telephony(audio), 8000, SAMPLE_RATE)
    if train:
        audio = np.clip(audio * random.uniform(0.5, 1.5), -1.0, 1.0)
    features = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
    return {"input_features": features, "labels": tokenizer(clip["text"]).input_ids}


def make_collate(decoder_start_token_id: int):
    import torch

    def collate(batch: list[dict]) -> dict:
        features = torch.tensor(np.stack([item["input_features"] for item in batch]))
        labels = [item["labels"] for item in batch]
        # The model prepends the start token itself when shifting labels.
        labels = [seq[1:] if seq and seq[0] == decoder_start_token_id else seq for seq in labels]
        width = max(len(seq) for seq in labels)
        padded = torch.full((len(labels), width), -100, dtype=torch.long)
        for row, seq in enumerate(labels):
            padded[row, : len(seq)] = torch.tensor(seq)
        return {"input_features": features, "labels": padded}

    return collate


def evaluate_loss(model, batches, device) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for batch in batches:
            out = model(
                input_features=batch["input_features"].to(device, torch.float16),
                labels=batch["labels"].to(device),
            )
            total += out.loss.item() * len(batch["labels"])
            count += len(batch["labels"])
    model.train()
    return total / max(1, count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, help="transformers Whisper checkpoint (local dir or HF id)")
    parser.add_argument("--train", nargs="+", required=True, help="parquet glob(s) with audio + text")
    parser.add_argument("--eval", nargs="+", required=True, help="held-out parquet glob(s)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="uzbek")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--telephony-prob", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--eval-clips", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=25, help="optimizer steps")
    parser.add_argument("--max-minutes", type=float, default=0, help="stop after this long (0 = no limit)")
    parser.add_argument("--buffer-shards", type=int, default=2, help="shards held in RAM for shuffling")
    parser.add_argument(
        "--expect-shards",
        type=int,
        default=0,
        help="total train shards; train while the rest are still downloading",
    )
    parser.add_argument("--init-adapter", type=Path, help="continue from a saved adapter")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    import pyarrow.parquet as pq
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA build of torch is required for training")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    tokenizer = WhisperTokenizer.from_pretrained(args.base, language=args.language, task="transcribe")
    tokenizer.set_prefix_tokens(language=args.language, task="transcribe", predict_timestamps=False)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base)

    eval_paths = resolve_paths(args.eval)
    exclude = set(eval_paths)  # a broad train glob may also match the held-out shard
    train_paths = [p for p in resolve_paths(args.train) if p not in exclude]
    rows_per_shard = [pq.ParquetFile(path).metadata.num_rows for path in train_paths]
    expect_shards = max(args.expect_shards, len(train_paths))
    # Shards still downloading are assumed to be as large as the ones on disk.
    train_clips = round(sum(rows_per_shard) / len(rows_per_shard) * expect_shards)
    eval_clips = []
    for clip in iter_clips(eval_paths, args.max_seconds, tokenizer):
        eval_clips.append(clip)
        if len(eval_clips) >= args.eval_clips:
            break
    print(f"train {len(train_paths)}/{expect_shards} shards on disk, ~{train_clips} clips; "
          f"eval {len(eval_clips)} clips", flush=True)

    model = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float16)
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    # SpecAugment, as in the base model's own training.
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # With a frozen base, checkpointed blocks need an input that requires grad.
    model.model.encoder.conv1.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))

    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="none",
        ))
    # Adapters train in fp32 for stable updates; the frozen base stays fp16.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()
    model.to(device)

    collate = make_collate(model.config.decoder_start_token_id)
    eval_examples = [featurize(c, feature_extractor, tokenizer, 1.0, train=False) for c in eval_clips]
    eval_batches = [
        collate(eval_examples[i : i + args.batch_size])
        for i in range(0, len(eval_examples), args.batch_size)
    ]
    rng = random.Random(args.seed)

    def train_batches():
        epoch = 0
        while True:
            pending = []
            for clip in stream_epoch(
                args.train, exclude, args.max_seconds, tokenizer, args.buffer_shards, rng, expect_shards
            ):
                pending.append(featurize(clip, feature_extractor, tokenizer, args.telephony_prob, train=True))
                if len(pending) == args.batch_size:
                    yield collate(pending)
                    pending = []
            epoch += 1
            print(f"epoch {epoch} finished", flush=True)

    total_steps = math.ceil(train_clips * args.epochs / (args.batch_size * args.grad_accum))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup_steps)
        * max(0.0, 1 - step / max(1, total_steps)),
    )
    scaler = torch.amp.GradScaler("cuda")

    args.output.mkdir(parents=True, exist_ok=True)
    best = evaluate_loss(model, eval_batches, device)
    history = [{"step": 0, "eval_loss": best}]
    print(f"step 0/{total_steps}  eval loss {best:.4f}", flush=True)
    model.save_pretrained(args.output / "adapter")

    step, micro, running = 0, 0, 0.0
    started = time.time()
    model.train()
    for batch in train_batches():
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(
                input_features=batch["input_features"].to(device, torch.float16),
                labels=batch["labels"].to(device),
            ).loss / args.grad_accum
        scaler.scale(loss).backward()
        running += loss.item()
        micro += 1
        if micro % args.grad_accum:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1

        elapsed = time.time() - started
        if step % 5 == 0:
            print(f"step {step}/{total_steps}  loss {running / 5:.4f}  "
                  f"{elapsed / step:.1f}s/step  "
                  f"VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            running = 0.0
        out_of_time = args.max_minutes and elapsed > args.max_minutes * 60
        if step % args.eval_every == 0 or step >= total_steps or out_of_time:
            current = evaluate_loss(model, eval_batches, device)
            history.append({"step": step, "eval_loss": current})
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            improved = current < best
            print(f"step {step}  eval loss {current:.4f}{'  (best, saved)' if improved else ''}", flush=True)
            if improved:
                best = current
                model.save_pretrained(args.output / "adapter")
        if step >= total_steps or out_of_time:
            break

    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best eval loss {best:.4f}; merging adapter", flush=True)

    # Merge the best adapter into a clean fp16 copy of the base for conversion.
    from peft import PeftModel

    del model
    torch.cuda.empty_cache()
    base = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, args.output / "adapter").merge_and_unload()
    merged.generation_config.forced_decoder_ids = None
    merged.save_pretrained(args.output / "merged")
    tokenizer.save_pretrained(args.output / "merged")
    feature_extractor.save_pretrained(args.output / "merged")
    print(f"merged model -> {args.output / 'merged'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

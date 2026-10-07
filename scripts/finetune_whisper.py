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


def load_rows(patterns: list[str], max_seconds: float, tokenizer, limit: int = 0) -> list[dict]:
    import pyarrow.parquet as pq
    from faster_whisper import decode_audio

    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No parquet files match {patterns}")
    rows = []
    for path in paths:
        for row in pq.read_table(path, columns=["audio", "text"]).to_pylist():
            text = normalize_label(row["text"] or "")
            if not text:
                continue
            seconds = len(decode_audio(io.BytesIO(row["audio"]["bytes"]), sampling_rate=SAMPLE_RATE)) / SAMPLE_RATE
            # Whisper sees at most 30 s; longer clips would lose audio but keep text.
            if not 0.5 <= seconds <= max_seconds:
                continue
            if len(tokenizer(text).input_ids) > MAX_LABEL_TOKENS:
                continue
            rows.append({"bytes": row["audio"]["bytes"], "text": text, "seconds": seconds})
            if limit and len(rows) >= limit:
                return rows
    return rows


class ClipDataset:
    def __init__(self, rows, feature_extractor, tokenizer, telephony_prob: float, train: bool) -> None:
        self.rows = rows
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.telephony_prob = telephony_prob
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        from faster_whisper import decode_audio

        row = self.rows[index]
        audio = decode_audio(io.BytesIO(row["bytes"]), sampling_rate=SAMPLE_RATE)
        # Evaluation always uses the phone channel; training mixes both.
        if not self.train or random.random() < self.telephony_prob:
            audio = resample(telephony(audio), 8000, SAMPLE_RATE)
        if self.train:
            audio = np.clip(audio * random.uniform(0.5, 1.5), -1.0, 1.0)
        features = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
        return {"input_features": features, "labels": self.tokenizer(row["text"]).input_ids}


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


def evaluate_loss(model, loader, device) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for batch in loader:
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA build of torch is required for training")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    tokenizer = WhisperTokenizer.from_pretrained(args.base, language=args.language, task="transcribe")
    tokenizer.set_prefix_tokens(language=args.language, task="transcribe", predict_timestamps=False)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base)

    started = time.time()
    train_rows = load_rows(args.train, args.max_seconds, tokenizer)
    eval_rows = load_rows(args.eval, args.max_seconds, tokenizer, limit=args.eval_clips)
    hours = sum(r["seconds"] for r in train_rows) / 3600
    print(f"train {len(train_rows)} clips ({hours:.1f} h), eval {len(eval_rows)} clips, "
          f"loaded in {time.time() - started:.0f}s", flush=True)

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
    train_loader = torch.utils.data.DataLoader(
        ClipDataset(train_rows, feature_extractor, tokenizer, args.telephony_prob, train=True),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        ClipDataset(eval_rows, feature_extractor, tokenizer, 1.0, train=False),
        batch_size=args.batch_size, collate_fn=collate,
    )

    total_steps = math.ceil(len(train_loader) * args.epochs / args.grad_accum)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup_steps)
        * max(0.0, 1 - step / max(1, total_steps)),
    )
    scaler = torch.amp.GradScaler("cuda")

    args.output.mkdir(parents=True, exist_ok=True)
    best = evaluate_loss(model, eval_loader, device)
    history = [{"step": 0, "eval_loss": best}]
    print(f"step 0/{total_steps}  eval loss {best:.4f}", flush=True)
    model.save_pretrained(args.output / "adapter")

    step, micro, running = 0, 0, 0.0
    started = time.time()
    model.train()
    done = False
    while not done:
        for batch in train_loader:
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
                current = evaluate_loss(model, eval_loader, device)
                history.append({"step": step, "eval_loss": current})
                improved = current < best
                print(f"step {step}  eval loss {current:.4f}{'  (best, saved)' if improved else ''}", flush=True)
                if improved:
                    best = current
                    model.save_pretrained(args.output / "adapter")
            if step >= total_steps or out_of_time:
                done = True
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

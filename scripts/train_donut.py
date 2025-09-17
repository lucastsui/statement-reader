#!/usr/bin/env python3
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from PIL import Image
from transformers import DonutProcessor, DonutImageProcessor, XLMRobertaTokenizer, VisionEncoderDecoderModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm


DATA_DIR = "data/donut"
OUT_DIR = "outputs/donut"
MODEL_NAME = "naver-clova-ix/donut-base"


@dataclass
class TrainConfig:
    epochs: int = 12
    lr: float = 5e-5
    batch_size: int = 2
    grad_accum: int = 8
    max_target_len: int = 256
    warmup_steps: int = 200


class DonutDataset(Dataset):
    def __init__(self, path: str, processor: DonutProcessor, max_target_len: int):
        self.records = [json.loads(l) for l in open(path)]
        self.processor = processor
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        # Build training text (Donut style: task prompt + target JSON)
        label = rec["label"]
        text = "<s_rental_statement>" + json.dumps(label, separators=(",", ":"))
        # Ensure EOS token at end for stable loss alignment
        eos = self.processor.tokenizer.eos_token or "</s>"
        if not text.endswith(eos):
            text += eos
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=self.max_target_len,
            truncation=True,
        ).input_ids.squeeze(0)
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate(batch, pad_token_id: int):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    max_lab = max(b["labels"].size(0) for b in batch)
    lab = torch.full((len(batch), max_lab), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        la = b["labels"]
        lab[i, : la.size(0)] = la
    return {
        "pixel_values": pixel_values,
        "labels": lab,
    }


def main():
    # Ensure output directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cfg = TrainConfig()
    # Reproducibility
    seed = 42
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if device.type == "mps":
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass
    # Force slow tokenizer (SentencePiece) and construct processor explicitly to avoid fast-path
    image_processor = DonutImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    # Register task-specific special token and resize embeddings
    added = tokenizer.add_special_tokens({"additional_special_tokens": ["<s_rental_statement>"]})
    processor = DonutProcessor(tokenizer=tokenizer, image_processor=image_processor)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    if added:
        try:
            model.decoder.resize_token_embeddings(len(tokenizer))
        except Exception:
            model.resize_token_embeddings(len(tokenizer))
    # Ensure pad token ids are set to avoid training warnings/errors
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # Ensure decoder start/eos and vocab size are set correctly
    # Start generation from our task token if possible, else BOS
    tok_id = processor.tokenizer.convert_tokens_to_ids("<s_rental_statement>")
    if tok_id is not None and tok_id != processor.tokenizer.unk_token_id:
        model.config.decoder_start_token_id = tok_id
    elif getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = len(processor.tokenizer)
    model.to(device)

    train_ds = DonutDataset(os.path.join(DATA_DIR, "train.jsonl"), processor, cfg.max_target_len)
    val_ds = DonutDataset(os.path.join(DATA_DIR, "val.jsonl"), processor, cfg.max_target_len)

    pad_token_id = processor.tokenizer.pad_token_id
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad_token_id))
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda b: collate(b, pad_token_id))

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    total_steps = (len(train_dl) * cfg.epochs) // cfg.grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

    global_step = 0
    best_val = float("inf")
    writer = SummaryWriter(log_dir="outputs/tb")

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg.epochs}")):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            # Mixed precision on MPS for speed
            use_amp = device.type in ("cuda", "mps")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / cfg.grad_accum
            loss.backward()
            running += loss.item()
            if (i + 1) % cfg.grad_accum == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    writer.add_scalar("train/loss", running, global_step)
                    print(f"step {global_step}: train_loss={running:.4f}")
                    running = 0.0
        # Validation (token-level loss proxy)
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_dl:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
                val_steps += 1
        val_loss = val_loss / max(1, val_steps)
        print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}")
        writer.add_scalar("val/loss", val_loss, epoch + 1)
        # Save checkpoint
        ckpt_dir = os.path.join(OUT_DIR, f"epoch_{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        # Track best
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(os.path.join(OUT_DIR, "best"))
            processor.save_pretrained(os.path.join(OUT_DIR, "best"))


if __name__ == "__main__":
    main()



"""Smoke test for Gemma4 E4B Vision LoRA SFT via Unsloth.

Runs a few gradient steps on a slice of the trajectory dataset to
verify wiring (model loads, vision data flows, loss decreases). Uses
Unsloth's FastVisionModel which handles QLoRA + flash attention +
the head_dim 512 quirks of gemma4's global attention layers.

Usage:
    HF_HOME=/mnt/storage/models/huggingface CUDA_VISIBLE_DEVICES=0 \\
    .venv/bin/python -m train.sft_smoke
"""

from __future__ import annotations

# Unsloth must be imported BEFORE transformers/trl/peft so it can
# monkey-patch them. Imports must stay at top-of-file.
import os
os.environ.setdefault("HF_HOME", "/mnt/storage/models/huggingface")

from unsloth import FastVisionModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sft_smoke")


MODEL_ID = "unsloth/gemma-4-E4B-it"
SHARD_PATH = Path("data/sft_dataset/v1/shard_00000.jsonl")
OUTPUT_DIR = Path("train_runs/smoke")


def load_records(
    shard_path: Path,
    max_records: int | None,
    longest_first: bool = False,
) -> List[dict]:
    """Read exporter JSONL into Unsloth's expected message format.

    Each row becomes a conversation with one user turn (image + the
    full prompt) and one assistant turn (the teacher's raw response).
    Unsloth's data collator handles the tokenization and image
    embedding when SFTTrainer iterates this dataset.

    If ``longest_first`` is True, the records are sorted by raw
    char length of (prompt + response) descending. Use this for
    worst-case memory tests — the smoke script's default sample
    misses the long-tail records that actually stress the budget.
    """
    raw_rows: List[dict] = []
    with shard_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            raw_rows.append(json.loads(line))

    if longest_first:
        raw_rows.sort(
            key=lambda r: len(r.get("prompt", "")) + len(r.get("raw_response", "")),
            reverse=True,
        )

    records: List[dict] = []
    for r in raw_rows:
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except FileNotFoundError:
            logger.warning("missing image: %s", r["image_path"])
            continue
        records.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": r["prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": r["raw_response"]},
                    ],
                },
            ],
        })
        if max_records and len(records) >= max_records:
            break
    logger.info("loaded %d records from %s (longest_first=%s)",
                len(records), shard_path, longest_first)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=int, default=50)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=64,
                    help="LoRA rank — Unsloth notebook defaults to 32. "
                         "We start higher for the smoke test, the full run "
                         "will set this from --lora-rank.")
    ap.add_argument("--max-length", type=int, default=16384,
                    help="Sequence length. Unsloth's flash-attn integration "
                         "handles gemma4's head_dim 512 layers, so we can "
                         "go to the full 14k Pokemon prompt length here "
                         "without OOM.")
    ap.add_argument("--finetune-vision", action="store_true",
                    help="Also LoRA-finetune the vision encoder. Default off "
                         "since the Pokemon screenshots don't need a "
                         "specialized visual encoder — the language model "
                         "is doing the planning.")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable NF4 quantization of the base model. Use "
                         "this for pure bf16 LoRA (more accurate, more "
                         "memory hungry). Default is QLoRA (NF4 base).")
    ap.add_argument("--longest-first", action="store_true",
                    help="Sort records by length descending. For worst-case "
                         "memory tests — the default sample order misses "
                         "the long-tail records that actually stress GPU "
                         "memory.")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("loading %s (4bit=%s, unsloth checkpointing, max_seq_length=%d)",
                MODEL_ID, not args.no_4bit, args.max_length)
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=not args.no_4bit,
        use_gradient_checkpointing="unsloth",
        # CRITICAL: this defaults to 2048 in Unsloth's FastVisionModel.
        # If you don't override it here, the data collator silently
        # truncates every sequence to 2048 tokens regardless of what
        # SFTConfig.max_length says, because UnslothVisionDataCollator
        # pulls its truncation length from `model.max_seq_length`.
        max_seq_length=args.max_length,
    )

    logger.info("attaching LoRA (rank=%d)", args.lora_rank)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        target_modules="all-linear",
    )
    FastVisionModel.for_training(model)

    records = load_records(
        SHARD_PATH,
        max_records=args.records,
        longest_first=args.longest_first,
    )
    if not records:
        logger.error("no records loaded — aborting")
        return 1

    sft_cfg = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.001,
        logging_steps=1,
        save_strategy="no",
        seed=3407,
        output_dir=str(OUTPUT_DIR),
        report_to="none",
        # Vision-finetuning required flags from the Unsloth notebook
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=records,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=sft_cfg,
    )

    logger.info("starting training: %d steps × accum=%d, max_length=%d, rank=%d",
                args.steps, args.grad_accum, args.max_length, args.lora_rank)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n>>> PEAK GPU MEMORY: {peak:.2f} GB", flush=True)
    logger.info("=== smoke run complete ===")
    save_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    logger.info("saved adapter to %s", save_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Full Gemma4 E4B Vision LoRA SFT run via Unsloth.

Targets a single 5090 with QLoRA + max_length 8192. Trains on the
filtered ≤8k token dataset (559 records produced by
``data/filter_by_length.py``). Saves checkpoints every N steps and
streams per-step losses to a jsonl file so the run can be resumed
or evaluated mid-flight.

Default config (overnight-friendly, ~3-5h on 1× 5090):

    qlora    : True (NF4 base, ~10 GB savings)
    rank     : 256 (660M trainable, 7.6% of total)
    max_len  : 8192 (the practical 5090 cap for gemma4 head_dim 512)
    epochs   : 10
    batch    : 1 × grad_accum 4 = effective batch 4
    optim    : adamw_8bit
    save     : every 100 steps + final adapter

Usage:
    HF_HOME=/mnt/storage/models/huggingface CUDA_VISIBLE_DEVICES=0 \\
    .venv/bin/python -m train.sft_run \\
        --shard data/sft_dataset/v1_8k/shard_00000.jsonl \\
        --output train_runs/v1 \\
        --epochs 10
"""

from __future__ import annotations

# Unsloth must be imported BEFORE transformers/trl/peft so it can
# monkey-patch them. Imports must stay at top-of-file.
import os

# Auto-discover the HuggingFace cache across machines and force
# offline mode when a cache is found. SLURM compute nodes on della
# have no internet, so we MUST use a pre-staged cache.
_hf_candidates = [
    os.environ.get("HF_HOME"),
    "/mnt/storage/models/huggingface",                   # local 5090 box
    "/scratch/gpfs/CHIJ/milkkarten/huggingface",         # della-gpu sk9014 (CHIJ)
    "/data1/milkkarten/.cache/huggingface",              # Cynthia / jin-gpu-3
]
for _c in _hf_candidates:
    if _c and os.path.isdir(_c):
        os.environ["HF_HOME"] = _c
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        break

from unsloth import FastVisionModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers.trainer_callback import TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sft_run")


MODEL_ID = "unsloth/gemma-4-E4B-it"


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------


def load_records(
    shard_paths: List[Path],
    max_records: int | None = None,
    data_root: Path | None = None,
) -> List[dict]:
    """Read exporter JSONL shards into Unsloth's chat format.

    ``data_root`` is prefixed to each record's ``image_path`` so the
    same shard can be used from different working directories. On
    della we stage shards + run_data into a single dir and pass the
    dir as data_root.
    """
    records: List[dict] = []
    for sp in shard_paths:
        with sp.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                img_path = r["image_path"]
                if data_root is not None and not Path(img_path).is_absolute():
                    img_path = str(data_root / img_path)
                try:
                    img = Image.open(img_path).convert("RGB")
                except FileNotFoundError:
                    logger.warning("missing image: %s", img_path)
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
        if max_records and len(records) >= max_records:
            break
    logger.info("loaded %d records from %d shard(s)", len(records), len(shard_paths))
    return records


# ----------------------------------------------------------------------
# JSONL loss logger callback
# ----------------------------------------------------------------------


class JsonlLossLogger(TrainerCallback):
    """Append every training metric dict to a jsonl file.

    Lets us read losses live during the run and post-mortem after.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)  # line buffered

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = dict(logs)
        row["step"] = state.global_step
        row["epoch"] = state.epoch
        row["timestamp"] = time.time()
        self._fh.write(json.dumps(row) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._fh.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=Path, nargs="+",
                    default=[Path("data/sft_dataset/v1_8k/shard_00000.jsonl")],
                    help="One or more JSONL shards from the exporter.")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Optional prefix prepended to each record's "
                         "image_path. Use this on della where the dataset "
                         "is staged at an absolute path.")
    ap.add_argument("--output", type=Path,
                    default=Path("train_runs/v1"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=256)
    ap.add_argument("--lora-alpha", type=int, default=256)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--max-length", type=int, default=8192,
                    help="Practical cap on a single 5090 — gemma4's "
                         "head_dim 512 global attention layers fall back "
                         "to MATH SDPA which OOMs above ~8k.")
    ap.add_argument("--save-steps", type=int, default=100,
                    help="Save a checkpoint every N optimizer steps.")
    ap.add_argument("--max-records", type=int, default=None,
                    help="Cap dataset size (for debugging).")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Disable QLoRA (NF4 base). Required for bf16 LoRA "
                         "or full FT.")
    ap.add_argument("--finetune-vision", action="store_true",
                    help="Also LoRA the vision tower.")
    ap.add_argument("--model-id", type=str, default=MODEL_ID,
                    help="Unsloth model ID. Choices: "
                         "unsloth/gemma-4-E2B-it, unsloth/gemma-4-E4B-it, "
                         "unsloth/gemma-4-26B-A4B-it, unsloth/gemma-4-31B-it.")
    ap.add_argument("--full-ft", action="store_true",
                    help="Full finetuning (all params trainable) instead of "
                         "LoRA. Implies --no-4bit. Only feasible for "
                         "smaller models on a single H200.")
    args = ap.parse_args()
    if args.full_ft:
        args.no_4bit = True

    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps({
        "model_id": args.model_id,
        "full_ft": args.full_ft,
        "shard": [str(s) for s in args.shard],
        "data_root": str(args.data_root) if args.data_root else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_length": args.max_length,
        "save_steps": args.save_steps,
        "load_in_4bit": not args.no_4bit,
        "finetune_vision": args.finetune_vision,
    }, indent=2))
    logger.info("config saved to %s", config_path)

    logger.info("loading %s (4bit=%s, full_ft=%s, max_seq_length=%d)",
                args.model_id, not args.no_4bit, args.full_ft, args.max_length)
    model, processor = FastVisionModel.from_pretrained(
        args.model_id,
        load_in_4bit=not args.no_4bit and not args.full_ft,
        full_finetuning=args.full_ft,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_length,
        # NOTE: Unsloth's `float32_mixed_precision=True` is actually
        # FULL fp32 (not the standard "bf16 fwd + fp32 master weights"
        # mixed precision). It doubles VRAM. Leave it OFF; use
        # adamw_torch for full FT — adamw_torch already keeps an
        # internal fp32 master copy of bf16 params for the optimizer
        # update, which is the real "mixed precision" we want.
    )

    if args.full_ft:
        logger.info("FULL FINETUNING — skipping LoRA injection")
    else:
        logger.info("attaching LoRA (rank=%d, alpha=%d, dropout=%.2f)",
                    args.lora_rank, args.lora_alpha, args.lora_dropout)
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=args.finetune_vision,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=3407,
            target_modules="all-linear",
        )
    FastVisionModel.for_training(model)

    records = load_records(
        args.shard,
        max_records=args.max_records,
        data_root=args.data_root,
    )
    if not records:
        logger.error("no records loaded — aborting")
        return 1

    # Optimizer choice:
    #   - LoRA path: adamw_8bit. Trainable params are tiny LoRA matrices
    #     so the bnb 8bit kernel handles them fine.
    #   - Full FT path: adamw_torch. Standard. The bnb 8bit kernel
    #     fails on huge embedding/lm_head tensors so we can't use
    #     adamw_8bit. adamw_torch keeps an internal fp32 master copy
    #     of bf16 params for the optimizer step (~64 GB state for 8B
    #     params), which puts E4B at ~146 GB total at 24k context —
    #     borderline on H200's 141 GB. The smoke test will reveal
    #     whether it fits in practice. If E4B OOMs, fall back to LoRA.
    optim_name = "adamw_torch" if args.full_ft else "adamw_8bit"
    optim_args = None
    # Full FT needs a much lower LR than LoRA. The default 2e-4 is
    # LoRA-tuned and would diverge full FT immediately. 5e-6 is
    # standard for full SFT of an instruction-tuned base.
    effective_lr = 5e-6 if (args.full_ft and args.lr == 2e-4) else args.lr
    if effective_lr != args.lr:
        logger.info("auto-lowering LR for full FT: %.0e -> %.0e", args.lr, effective_lr)
    logger.info("optimizer: %s  lr: %.0e  optim_args: %s",
                optim_name, effective_lr, optim_args)
    sft_cfg_kwargs = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=effective_lr,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        optim=optim_name,
        weight_decay=0.001,
    )
    if optim_args:
        sft_cfg_kwargs["optim_args"] = optim_args
    sft_cfg = SFTConfig(
        **sft_cfg_kwargs,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,  # keep 5 most recent checkpoints
        seed=3407,
        output_dir=str(args.output),
        report_to="none",
        # Vision-finetuning required flags from the Unsloth notebook
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
    )

    loss_log = args.output / "losses.jsonl"
    trainer = SFTTrainer(
        model=model,
        train_dataset=records,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=sft_cfg,
        callbacks=[JsonlLossLogger(loss_log)],
    )

    n_records = len(records)
    steps_per_epoch = max(n_records // (args.batch_size * args.grad_accum), 1)
    total_steps = steps_per_epoch * args.epochs
    logger.info(
        "starting training: epochs=%d, records=%d, steps/epoch≈%d, "
        "total steps≈%d, max_length=%d, rank=%d",
        args.epochs, n_records, steps_per_epoch, total_steps,
        args.max_length, args.lora_rank,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    logger.info("=== training complete ===")
    logger.info("elapsed: %.0f sec (%.2f hr)", elapsed, elapsed / 3600)
    logger.info("peak GPU memory: %.2f GB", peak)

    final_dir = args.output / "lora_adapter_final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("saved final adapter to %s", final_dir)

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps({
        "elapsed_sec": elapsed,
        "elapsed_hr": elapsed / 3600,
        "peak_gpu_gb": peak,
        "total_steps": trainer.state.global_step,
        "final_loss": trainer.state.log_history[-1].get("train_loss")
                      if trainer.state.log_history else None,
        "final_adapter": str(final_dir),
        "loss_log": str(loss_log),
    }, indent=2))
    logger.info("summary at %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

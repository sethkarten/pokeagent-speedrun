"""Offline GRPO training for Gemma4 26B-A4B via Unsloth + TRL.

Trains the SFT adapter to produce Gemini-like reasoning traces using
Group Relative Policy Optimization.  The model generates completions
for each prompt, and four reward functions score them against the Gemini
teacher's reference response.

Default config (della 2xH200, 24h):

    base model : unsloth/gemma-4-26B-A4B-it
    adapter    : adapters/26b_emerald_v3 (SFT LoRA)
    bf16       : True (no quantization on H200)
    num_gen    : 4 completions per prompt
    max_prompt : 16384 tokens
    max_compl  : 2048 tokens
    epochs     : 2
    batch      : 1 x grad_accum 4 x 2 GPUs = effective batch 8
    optim      : adamw_torch (fp32 master weights)
    save       : every 200 steps + final adapter

Usage (della 2xH200):
    accelerate launch --num_processes 2 \\
        -m train.grpo_offline \\
        --shard data/sft_dataset/emerald_v3/*.jsonl \\
        --adapter adapters/26b_emerald_v3 \\
        --output train_runs/grpo_emerald_v1 \\
        --epochs 2
"""
from __future__ import annotations

import importlib.machinery
import json
import logging
import os
import signal
import sys
import time
import types
from pathlib import Path

# Reduce CUDA memory fragmentation for GRPO's variable-length prompts.
# GRPO batches can have very different lengths (our prompts range 4K-18K
# tokens), so the default allocator fragments and OOMs on otherwise
# feasible batches. expandable_segments lets CUDA grow/shrink segments.
# NOTE: the error message says "PYTORCH_ALLOC_CONF" but the real env
# var PyTorch reads is "PYTORCH_CUDA_ALLOC_CONF".
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

# ---------------------------------------------------------------------------
# HF cache auto-discovery — must run before any HF import.
# ---------------------------------------------------------------------------
_HF_CANDIDATES = [
    os.environ.get("HF_HOME"),
    "/mnt/storage/models/huggingface",
    "/scratch/gpfs/CHIJ/milkkarten/huggingface",
    "/data1/milkkarten/.cache/huggingface",
]
for _c in _HF_CANDIDATES:
    if _c and os.path.isdir(_c):
        os.environ["HF_HOME"] = _c
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        break

# ---------------------------------------------------------------------------
# TRL 0.24 import workaround
#
# GRPOTrainer eagerly imports through a chain:
#   grpo_trainer -> callbacks -> mergekit_utils (needs mergekit)
#                             -> judges (needs llm_blender)
#                             -> weave
#
# These packages are not installed / incompatible with transformers 5.5.
# TRL's is_*_available() returns (False, None) tuples which are truthy
# in Python, so conditional imports fire even when the package is absent.
#
# Fix: stub missing packages before TRL's import chain runs.
# ---------------------------------------------------------------------------


def _stub_package(
    name: str,
    submodules: dict[str, dict[str, object]] | None = None,
) -> None:
    """Register a fake package with submodules in ``sys.modules``."""
    pkg = types.ModuleType(name)
    pkg.__path__ = []  # type: ignore[attr-defined]
    pkg.__package__ = name
    pkg.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    sys.modules[name] = pkg
    for sub, attrs in (submodules or {}).items():
        fqn = f"{name}.{sub}"
        mod = types.ModuleType(fqn)
        mod.__package__ = name
        mod.__spec__ = importlib.machinery.ModuleSpec(fqn, None)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[fqn] = mod
        setattr(pkg, sub, mod)


_STUBS = {
    "mergekit": {
        "config": {"MergeConfiguration": type("MergeConfiguration", (), {})},
        "merge": {
            "MergeOptions": type("MergeOptions", (), {}),
            "run_merge": lambda *a, **k: None,
        },
    },
    "llm_blender": {},
    "weave": {},
}
for _name, _subs in _STUBS.items():
    if _name not in sys.modules:
        _stub_package(_name, _subs)

# weave.trace.context submodules needed by callbacks.py
for _fqn in ("weave.trace", "weave.trace.context"):
    if _fqn not in sys.modules:
        _m = types.ModuleType(_fqn)
        _m.__package__ = "weave"
        _m.__path__ = []  # type: ignore[attr-defined]
        _m.__spec__ = importlib.machinery.ModuleSpec(_fqn, None)
        _m.weave_client_context = None  # type: ignore[attr-defined]
        sys.modules[_fqn] = _m
sys.modules["weave"].EvaluationLogger = type(  # type: ignore[attr-defined]
    "EvaluationLogger", (), {}
)

# ---------------------------------------------------------------------------
# Now safe to import Unsloth / TRL / reward functions.
# Unsloth must be imported BEFORE transformers/trl/peft.
# ---------------------------------------------------------------------------
# Unsloth is always needed (Gemma4 has ClippableLinear that stock PEFT
# can't handle). Use Unsloth's compiled GRPOTrainer for both single-GPU
# and DDP. For DDP: monkey-patch DistributedDataParallel to expose
# model.config and other attrs the compiled trainer needs.
# ---------------------------------------------------------------------------
_IS_DDP = os.environ.get("LOCAL_RANK") is not None

# For DDP: pin each process to its assigned GPU BEFORE any torch/unsloth
# import touches CUDA. Otherwise FastVisionModel with device_map='auto'
# spreads the model across all visible GPUs and the 4 DDP processes
# collide on GPU 0 → OOM.
if _IS_DDP:
    _local_rank = os.environ["LOCAL_RANK"]
    os.environ["CUDA_VISIBLE_DEVICES"] = _local_rank

from unsloth import FastVisionModel, PatchFastRL  # noqa: E402  isort:skip

import argparse  # noqa: E402

import torch  # noqa: E402, F401
from datasets import Dataset  # noqa: E402
from PIL import Image  # noqa: E402
from transformers.trainer_callback import TrainerCallback  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from train.reward_functions import (  # noqa: E402
    action_similarity_reward,
    format_reward,
    state_accuracy_reward,
    tool_match_reward,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("grpo_offline")

MODEL_ID = "unsloth/gemma-4-26B-A4B-it"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_grpo_dataset(
    shard_paths: list[Path],
    data_root: Path | None = None,
    max_records: int | None = None,
) -> Dataset:
    """Convert exporter JSONL shards into an HF Dataset for GRPOTrainer.

    Columns produced:
    - ``prompt``:  conversational format with image placeholder
    - ``images``:  list of PIL Images (one per example)
    - ``gemini_response``:  teacher's raw output (for reward functions)
    - ``pre_state``:  JSON-encoded game state dict (for reward functions)
    """
    records: list[dict] = []
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
                    "prompt": [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": r["prompt"]},
                        ],
                    }],
                    "images": [img],
                    "gemini_response": r["raw_response"],
                    "pre_state": json.dumps(r.get("pre_state", {})),
                })
                if max_records and len(records) >= max_records:
                    break
        if max_records and len(records) >= max_records:
            break

    logger.info("loaded %d records from %d shard(s)", len(records), len(shard_paths))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# JSONL loss logger (same pattern as sft_run.py)
# ---------------------------------------------------------------------------


class JsonlLossLogger(TrainerCallback):
    """Stream per-step metrics to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._f = open(path, "a", buffering=1)  # noqa: SIM115

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if logs:
            row: dict = {"step": state.global_step, "epoch": state.epoch}
            row.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
            self._f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--shard", type=Path, nargs="+",
        default=[Path("data/sft_dataset/emerald_v3/shard_00000.jsonl")],
        help="One or more JSONL shards from the exporter.",
    )
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Path to SFT LoRA adapter directory.")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Prefix prepended to each record's image_path.")
    ap.add_argument("--output", type=Path, default=Path("train_runs/grpo_v1"))
    ap.add_argument("--model-id", type=str, default=MODEL_ID)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--num-generations", type=int, default=4,
                    help="Completions per prompt (min 2).")
    ap.add_argument("--max-prompt-length", type=int, default=16384)
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--max-records", type=int, default=None,
                    help="Cap dataset size (for debugging).")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--4bit", dest="load_4bit", action="store_true",
                    help="QLoRA for smoke tests on smaller GPUs.")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps({
        "model_id": args.model_id,
        "adapter": str(args.adapter),
        "shard": [str(s) for s in args.shard],
        "data_root": str(args.data_root) if args.data_root else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "load_4bit": args.load_4bit,
    }, indent=2))
    logger.info("config saved to %s", config_path)

    # -- Load model with SFT adapter --
    logger.info(
        "loading adapter %s (ddp=%s, 4bit=%s)",
        args.adapter, _IS_DDP, args.load_4bit,
    )
    model, processor = FastVisionModel.from_pretrained(
        str(args.adapter),
        load_in_4bit=args.load_4bit,
        max_seq_length=args.max_prompt_length + args.max_completion_length,
    )
    FastVisionModel.for_training(model)
    # Always use Unsloth's compiled trainer (stock TRL has vision
    # compatibility issues with Unsloth-loaded Gemma4 models).
    # For DDP: Unsloth's compiled GRPOTrainer does model.config which
    # fails on DistributedDataParallel. Monkey-patch DDP to expose it.
    PatchFastRL("grpo", FastVisionModel)
    if _IS_DDP:
        import torch.nn.parallel

        _orig_ddp_getattr = torch.nn.parallel.DistributedDataParallel.__getattr__

        def _ddp_getattr_with_config(self, name):
            if name in ("config", "warnings_issued", "generation_config",
                        "can_generate", "get_base_model"):
                return getattr(self.module, name)
            return _orig_ddp_getattr(self, name)

        torch.nn.parallel.DistributedDataParallel.__getattr__ = _ddp_getattr_with_config

    # -- Load dataset --
    dataset = load_grpo_dataset(
        args.shard, data_root=args.data_root, max_records=args.max_records,
    )
    if len(dataset) == 0:
        logger.error("no records loaded — aborting")
        return 1

    # -- GRPOConfig --
    grpo_config = GRPOConfig(
        output_dir=str(args.output),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        # Temperature tradeoff for GRPO:
        # - Too low (0.3): all num_gen samples identical → reward_std=0 →
        #   zero advantage → no gradient. Observed empirically.
        # - Too high (0.7): generations drift from SFT distribution.
        # 0.5 with top_p=0.9 keeps diversity while staying reasonable.
        temperature=0.5,
        top_p=0.9,
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
        mask_truncated_completions=True,
        # Use Liger fused linear+CE loss to avoid materializing the full
        # logits tensor (vocab 262144 x seq_len x fp32 = ~35 GB at our
        # sizes, which caused deterministic OOM on Red at step 10).
        use_liger_loss=True,
        reward_weights=[2.0, 1.5, 1.0, 0.5],
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        logging_steps=1,
        log_completions=True,
        report_to="none",
        remove_unused_columns=False,
        shuffle_dataset=True,
        seed=3407,
    )

    # -- Build trainer --
    loss_log = args.output / "losses.jsonl"
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            tool_match_reward,
            action_similarity_reward,
            state_accuracy_reward,
            format_reward,
        ],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=None,
        callbacks=[JsonlLossLogger(loss_log)],
    )

    n_records = len(dataset)
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(n_records // effective_batch, 1)
    total_steps = steps_per_epoch * args.epochs
    logger.info(
        "starting GRPO: epochs=%d, records=%d, num_gen=%d, "
        "steps/epoch~%d, total~%d, max_prompt=%d, max_compl=%d",
        args.epochs, n_records, args.num_generations,
        steps_per_epoch, total_steps,
        args.max_prompt_length, args.max_completion_length,
    )

    # -- SIGTERM handler (SLURM wall-time safety) --
    def _sigterm_handler(signum, frame):
        logger.warning("SIGTERM received — saving emergency checkpoint")
        emergency_dir = args.output / "checkpoint_emergency"
        try:
            model.save_pretrained(emergency_dir)
            processor.save_pretrained(emergency_dir)
            logger.info("emergency checkpoint saved to %s", emergency_dir)
        except Exception:
            logger.exception("emergency save failed")
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        pass

    # -- Train --
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    logger.info("=== GRPO training complete ===")
    logger.info("elapsed: %.0f sec (%.2f hr)", elapsed, elapsed / 3600)
    logger.info("peak GPU memory: %.2f GB", peak)

    # -- Save final adapter --
    final_dir = args.output / "grpo_adapter_final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("saved final adapter to %s", final_dir)

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps({
        "elapsed_sec": elapsed,
        "elapsed_hr": elapsed / 3600,
        "peak_gpu_gb": peak,
        "total_steps": trainer.state.global_step,
        "final_loss": (
            trainer.state.log_history[-1].get("loss")
            if trainer.state.log_history else None
        ),
        "final_adapter": str(final_dir),
        "loss_log": str(loss_log),
        "num_records": n_records,
        "num_generations": args.num_generations,
    }, indent=2))
    logger.info("summary at %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Probe a trained Gemma4 LoRA adapter on a held-out trajectory record.

Compares student output to the recorded teacher response. No metrics —
just side-by-side text so you can eyeball whether the student is
imitating the teacher's reasoning style and tool-call format.

Usage:
    HF_HOME=/mnt/storage/models/huggingface CUDA_VISIBLE_DEVICES=0 \\
    .venv/bin/python -m train.probe \\
        --adapter train_runs/v1/lora_adapter_final \\
        --shard data/sft_dataset/v1_8k/shard_00000.jsonl \\
        --record-index -1 \\
        --max-new-tokens 512

You can also probe an intermediate checkpoint:
    --adapter train_runs/v1/checkpoint-500
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HOME", "/mnt/storage/models/huggingface")

from unsloth import FastVisionModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_ID = "unsloth/gemma-4-E4B-it"


def load_record(shard: Path, index: int) -> dict:
    rows = [json.loads(l) for l in shard.open() if l.strip()]
    if index < 0:
        index = len(rows) + index
    if not 0 <= index < len(rows):
        raise IndexError(f"record {index} out of range (shard has {len(rows)} rows)")
    return rows[index]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Directory containing the trained LoRA adapter.")
    ap.add_argument("--shard", type=Path,
                    default=Path("data/sft_dataset/v1_8k/shard_00000.jsonl"))
    ap.add_argument("--record-index", type=int, default=-1,
                    help="Which record from the shard to probe (-1 = last).")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-base-comparison", action="store_true",
                    help="Skip running the un-adapted base model. By default "
                         "we run it too so you can see student-vs-base-vs-teacher.")
    ap.add_argument("--no-4bit", action="store_true",
                    help="Load base in bf16 instead of NF4 (slower, more memory).")
    args = ap.parse_args()

    record = load_record(args.shard, args.record_index)
    image = Image.open(record["image_path"]).convert("RGB")
    user_prompt = record["prompt"]
    teacher_response = record["raw_response"]

    print("=" * 80)
    print(f"PROBING ADAPTER: {args.adapter}")
    print(f"RECORD: {args.shard.name}[{args.record_index}]")
    print(f"  step={record.get('step')}  role={record.get('role')}")
    print(f"  interaction_type={record.get('interaction_type')}")
    print(f"  prompt_chars={len(user_prompt)}  image={record['image_path']}")
    print("=" * 80)

    base_response = None
    if not args.no_base_comparison:
        print("\n--- LOADING BASE MODEL (no adapter) ---")
        base_model, base_processor = FastVisionModel.from_pretrained(
            MODEL_ID,
            load_in_4bit=not args.no_4bit,
            use_gradient_checkpointing=False,
            max_seq_length=8192,
        )
        FastVisionModel.for_inference(base_model)
        print("\n--- GENERATING WITH BASE MODEL ---")
        base_response = _generate(base_model, base_processor, image, user_prompt, args)
        # free base before loading student
        del base_model, base_processor
        torch.cuda.empty_cache()

    print("\n--- LOADING STUDENT (base + LoRA via Unsloth) ---")
    # Pass the adapter dir directly as model_name — Unsloth's loader
    # knows how to merge the LoRA into the base without hitting the
    # Gemma4ClippableLinear issue that transformers' load_adapter
    # would crash on.
    model, processor = FastVisionModel.from_pretrained(
        str(args.adapter),
        load_in_4bit=not args.no_4bit,
        use_gradient_checkpointing=False,
        max_seq_length=8192,
    )
    FastVisionModel.for_inference(model)

    print("\n--- GENERATING WITH STUDENT ---")
    student_response = _generate(model, processor, image, user_prompt, args)

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if base_response is not None:
        print("\n[BASE MODEL — no training]")
        print("-" * 80)
        print(base_response)
        print()

    print("\n[STUDENT — base + LoRA adapter]")
    print("-" * 80)
    print(student_response)
    print()

    print("\n[TEACHER — recorded ground truth from trajectory]")
    print("-" * 80)
    print(teacher_response)
    print()

    print("=" * 80)
    print("Done. Things to look for:")
    print("  1. Does the student use the same ANALYZE/PLAN format?")
    print("  2. Does it call the same tool with similar args?")
    print("  3. Does the reasoning reference what's actually on the screen?")
    print("  4. Is it noticeably different from the base model output?")
    print("=" * 80)
    return 0


def _generate(model, processor, image, prompt_text, args) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            use_cache=True,
        )
    new_tokens = out[0, input_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    sys.exit(main())

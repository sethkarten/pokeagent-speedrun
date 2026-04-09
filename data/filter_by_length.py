"""Filter exporter JSONL shards to records that fit within a token cap.

For training Gemma4 E4B Vision on a single 5090, the practical
sequence-length cap is ~8192 tokens (the head_dim 512 global
attention layers fall back to PyTorch MATH SDPA on Blackwell, which
is O(seq²) and OOMs above ~8k). Rather than truncating long records
mid-prompt — which destroys the most recent context, the part the
agent's decision actually depends on — we drop them entirely.

Usage:
    .venv/bin/python -m data.filter_by_length \\
        --input  data/sft_dataset/v1/shard_00000.jsonl \\
        --output data/sft_dataset/v1_8k/shard_00000.jsonl \\
        --max-tokens 8192
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

# HF cache lives on /mnt/storage; default to that if not set.
os.environ.setdefault("HF_HOME", "/mnt/storage/models/huggingface")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("filter_by_length")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="Source JSONL shard from data/export_trajectories.py.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination JSONL shard for kept records.")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="Drop records whose tokenized full conversation "
                         "(image + prompt + response) exceeds this many tokens.")
    ap.add_argument("--processor", default="google/gemma-4-E4B-it",
                    help="HF processor ID used to tokenize for length check.")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    logger.info("loading processor %s", args.processor)
    processor = AutoProcessor.from_pretrained(args.processor)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    role_in: Counter = Counter()
    role_out: Counter = Counter()
    role_dropped: Counter = Counter()

    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            n_in += 1
            role = r.get("role", "?")
            role_in[role] += 1

            try:
                img = Image.open(r["image_path"]).convert("RGB")
            except FileNotFoundError:
                logger.warning("missing image: %s — dropping", r["image_path"])
                role_dropped[role] += 1
                continue

            msgs = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": r["prompt"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": r["raw_response"]},
                ]},
            ]
            ids = processor.apply_chat_template(
                [msgs],
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"]
            n_tokens = int(ids.size(1))

            if n_tokens > args.max_tokens:
                role_dropped[role] += 1
                continue

            r["_token_length"] = n_tokens
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
            role_out[role] += 1

            if n_in % 100 == 0:
                logger.info("processed %d in / %d kept", n_in, n_out)

    print()
    print("=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    print(f"  input records:  {n_in}")
    print(f"  kept records:   {n_out}  ({n_out/max(n_in,1)*100:.1f}%)")
    print(f"  dropped:        {n_in - n_out}")
    print(f"  cap:            {args.max_tokens} tokens")
    print()
    print("PER-ROLE:")
    for role in sorted(set(role_in) | set(role_out)):
        kept = role_out.get(role, 0)
        dropped = role_dropped.get(role, 0)
        total = role_in.get(role, 0)
        rate = kept / max(total, 1) * 100
        print(f"  {role:14s}  in={total:5d}  kept={kept:5d}  dropped={dropped:5d}  ({rate:.1f}%)")
    print()
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

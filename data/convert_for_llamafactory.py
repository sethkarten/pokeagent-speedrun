"""Convert our exporter v2 shards to LlamaFactory ShareGPT vision format.

LlamaFactory expects multimodal SFT data as ShareGPT-style JSON:

    [
      {
        "conversations": [
          {"from": "human", "value": "<image>user instruction"},
          {"from": "gpt",   "value": "model response"}
        ],
        "images": ["abs/or/rel/path/to/image.png"]
      },
      ...
    ]

Plus a `dataset_info.json` entry pointing at the JSON file with the
sharegpt formatting + columns mapping.

Our exporter shards have one record per LLM call with:
    image_path  : relative path to a screenshot
    prompt      : the full LLM prompt (already includes instructions, state, history)
    raw_response: the LLM's raw output (the SFT target)

The conversion is straightforward — wrap prompt as the human turn
(prefixed with `<image>` so LF inserts the image token), wrap
raw_response as the gpt turn, point images at the absolute screenshot
path.

Usage:
    .venv/bin/python -m data.convert_for_llamafactory \\
        --shards data/sft_dataset/v2/shard_00000.jsonl \\
                 data/sft_dataset/v2/shard_00001.jsonl \\
                 data/sft_dataset/v2/shard_00002.jsonl \\
                 data/sft_dataset/v2/shard_00003.jsonl \\
        --image-root /scratch/gpfs/CHIJ/milkkarten/pokeagent-distill-data \\
        --output data/sft_dataset/v2_lf/pokeagent_v2.json \\
        --dataset-info data/sft_dataset/v2_lf/dataset_info.json \\
        --dataset-name pokeagent_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger("convert_for_llamafactory")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=Path, nargs="+", required=True,
                    help="Source JSONL shards from data/export_trajectories.py")
    ap.add_argument("--image-root", type=Path, required=True,
                    help="Absolute prefix for image paths. Each record's "
                         "image_path is joined onto this and verified to "
                         "exist before writing.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination JSON file (LLaMA-Factory ShareGPT format).")
    ap.add_argument("--dataset-info", type=Path, required=True,
                    help="Destination dataset_info.json with the entry "
                         "for this dataset.")
    ap.add_argument("--dataset-name", default="pokeagent_v2",
                    help="Name LF will refer to this dataset by.")
    ap.add_argument("--max-records", type=int, default=None,
                    help="Optional cap (for smoke test).")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_info.parent.mkdir(parents=True, exist_ok=True)

    out_records: List[dict] = []
    n_in = 0
    n_missing = 0
    for shard in args.shards:
        with shard.open() as f:
            for line in f:
                if not line.strip():
                    continue
                n_in += 1
                r = json.loads(line)
                rel = r.get("image_path")
                if not rel:
                    n_missing += 1
                    continue
                abs_path = (args.image_root / rel).resolve()
                if not abs_path.exists():
                    n_missing += 1
                    continue
                out_records.append({
                    "conversations": [
                        {"from": "human", "value": "<image>" + r["prompt"]},
                        {"from": "gpt",   "value": r["raw_response"]},
                    ],
                    "images": [str(abs_path)],
                })
                if args.max_records and len(out_records) >= args.max_records:
                    break
        if args.max_records and len(out_records) >= args.max_records:
            break

    logger.info("input records: %d", n_in)
    logger.info("kept records:  %d", len(out_records))
    logger.info("missing image: %d", n_missing)

    with args.output.open("w") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s (%d records)", args.output, len(out_records))

    # Build / merge dataset_info.json. LlamaFactory expects this to live
    # next to the data files; multiple datasets share one file.
    if args.dataset_info.exists():
        info = json.loads(args.dataset_info.read_text())
    else:
        info = {}
    info[args.dataset_name] = {
        "file_name": args.output.name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
    args.dataset_info.write_text(json.dumps(info, indent=2))
    logger.info("updated %s", args.dataset_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

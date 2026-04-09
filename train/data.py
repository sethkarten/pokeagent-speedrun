"""Shared dataset + collator for Gemma4 VLM SFT.

Used by both ``train/sft_smoke.py`` and ``train/sft_run.py``. Reads
JSONL shards produced by ``data/export_trajectories.py`` and yields
batches ready for the model forward.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("train.data")


class SFTRecordDataset(Dataset):
    """Reads exporter JSONL shards and yields (image, prompt, response)."""

    def __init__(self, shard_paths: List[Path], max_records: int | None = None):
        self.records: List[dict] = []
        for sp in shard_paths:
            with sp.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.records.append(json.loads(line))
                    if max_records and len(self.records) >= max_records:
                        break
            if max_records and len(self.records) >= max_records:
                break
        logger.info("loaded %d records from %d shard(s)", len(self.records), len(shard_paths))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "image_path": r["image_path"],
            "prompt": r["prompt"],
            "response": r["raw_response"],
            "role": r.get("role", "orchestrator"),
            "interaction_type": r.get("interaction_type", ""),
            "step": r.get("step", -1),
            "run_id": r.get("run_id", ""),
        }


@dataclass
class Gemma4VLCollator:
    """Build batched (input_ids, pixel_values, labels) from raw records.

    Each record turns into a single multi-turn chat:
        user:    [image] + prompt
        assistant: response

    Loss is computed only on the assistant tokens — prompt + image
    tokens are masked to -100 in the labels tensor.
    """

    processor: Any  # Gemma4Processor
    max_length: int = 16384

    def __call__(self, batch: List[dict]) -> dict:
        all_messages: List[List[dict]] = []
        prompt_only_messages: List[List[dict]] = []

        for ex in batch:
            img = Image.open(ex["image_path"]).convert("RGB")
            user_content = [
                {"type": "image", "image": img},
                {"type": "text", "text": ex["prompt"]},
            ]
            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant",
                 "content": [{"type": "text", "text": ex["response"]}]},
            ]
            prompt_messages = [
                {"role": "user", "content": user_content},
            ]
            all_messages.append(full_messages)
            prompt_only_messages.append(prompt_messages)

        proc_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
        }
        full_inputs = self.processor.apply_chat_template(
            all_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=proc_kwargs,
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_only_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=proc_kwargs,
        )

        input_ids = full_inputs["input_ids"]
        labels = input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        for i in range(input_ids.size(0)):
            prompt_len = int((prompt_inputs["input_ids"][i] != pad_id).sum().item())
            full_len = int((input_ids[i] != pad_id).sum().item())
            mask_until = min(prompt_len, full_len)
            labels[i, :mask_until] = -100
        labels[input_ids == pad_id] = -100
        # If a row has no unmasked tokens (assistant fully truncated),
        # unmask the final token so loss stays finite.
        for i in range(labels.size(0)):
            if (labels[i] != -100).sum() == 0:
                labels[i, -1] = input_ids[i, -1]

        out = {
            "input_ids": input_ids,
            "attention_mask": full_inputs["attention_mask"],
            "labels": labels,
        }
        for k, v in full_inputs.items():
            if k in out:
                continue
            out[k] = v
        return out

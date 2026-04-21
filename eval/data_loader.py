"""Sample loader for eval.

Reads JSONL shards from ``data/sft_dataset/<name>/shard_*.jsonl`` and returns a
list of sample dicts. Each sample has:
- ``idx`` (int) — position within the sampled subset
- ``prompt`` (str) — full real prompt (for ``prompt_mode="real"``)
- ``simplified_prompt`` (str) — short single-shot prompt (for ``prompt_mode="simplified"``)
- ``image_b64`` (str or None) — base64 PNG of screenshot
- ``image_path`` (str or None) — resolved absolute path to screenshot
- ``pre_state`` (dict)
- ``raw_response`` (str) — teacher's raw output
- ``completion`` (dict) — structured {reasoning, tool_calls}
- ``state_type`` (str) — "battle" or "overworld" (derived from pre_state)
- ``location`` (str)
- ``source`` (str) — shard filename + line index
"""

from __future__ import annotations

import base64
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "sft_dataset"
RUN_DATA_DIR = REPO_ROOT  # image_path is relative to repo root (run_data/...)


@dataclass
class Sample:
    idx: int
    prompt: str
    simplified_prompt: str
    image_b64: Optional[str]
    image_path: Optional[str]
    pre_state: dict
    raw_response: str
    completion: dict
    state_type: str
    location: str
    source: str


def _state_type_from_pre_state(pre_state: dict) -> str:
    if pre_state.get("is_in_battle"):
        return "battle"
    return "overworld"


def _build_simplified_prompt(example: dict) -> str:
    """Build a short single-shot prompt (state + screenshot + 'choose best action').

    The simplified prompt style was used in ``eval_full_model_comparison.md``
    (Base vs SFT, short prompt). The idea is: no 40-55K-char harness context,
    just 'here's what the player sees, pick a tool'.
    """
    ps = example.get("pre_state") or {}
    loc = ps.get("location", "?")
    coords = ps.get("player_coords")
    ctx = ps.get("context", "?")
    is_battle = ps.get("is_in_battle", False)
    dialog = ps.get("dialog_active", False)

    lines = [
        "You are playing Pokemon. Below is the current game state.",
        "",
        "## Game State",
        f"- Location: {loc}",
    ]
    if coords:
        lines.append(f"- Coordinates: {coords}")
    lines.append(f"- Context: {ctx}")
    lines.append(f"- In battle: {is_battle}")
    lines.append(f"- Dialog active: {dialog}")
    lines.append("")
    lines.append("## Available tools")
    lines.append(
        "- press_buttons(buttons, reasoning): press one or more GameBoy buttons in sequence.\n"
        "- navigate_to_coords(x, y, reasoning): auto-walk to (x, y) on the current map.\n"
        "- execute_custom_subagent(reasoning): delegate to a subagent (e.g. battle handler).\n"
        "- replan_objectives(edits, reasoning): revise the current objective list.\n"
        "- process_memory(reasoning): read/write agent memory store."
    )
    lines.append("")
    lines.append("## Your task")
    lines.append(
        "Describe what you see on the screen, pick the best tool, and give its arguments. "
        "Use the format: `[tool_name] ANALYZE: ... PLAN: ...` (or `Calling tool_name(args)`)."
    )
    return "\n".join(lines)


def _resolve_image(example: dict) -> tuple[Optional[str], Optional[str]]:
    """Return (image_b64, absolute_image_path) if the screenshot is available.

    image_b64 is None in the majority of shards — we load it from image_path
    lazily. If both fail, return (None, None) and the caller should skip.
    """
    b64 = example.get("image_b64")
    rel_path = example.get("image_path")
    if not rel_path and not b64:
        return None, None

    abs_path = None
    if rel_path:
        candidate = (RUN_DATA_DIR / rel_path).resolve()
        if candidate.exists():
            abs_path = str(candidate)

    if b64:
        return b64, abs_path

    if abs_path:
        try:
            with open(abs_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode("ascii"), abs_path
        except OSError:
            return None, abs_path
    return None, None


def iter_shards(dataset: str) -> Iterator[tuple[str, int, dict]]:
    """Yield (shard_name, line_idx, example) tuples from all shards of a dataset."""
    shards_dir = DATASET_DIR / dataset
    if not shards_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {shards_dir}")
    for shard_path in sorted(shards_dir.glob("shard_*.jsonl")):
        with open(shard_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield shard_path.name, i, json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_samples(
    dataset: str,
    n_samples: int,
    *,
    seed: int = 42,
    require_image: bool = True,
    role_filter: str = "orchestrator",
) -> list[Sample]:
    """Load a diverse subset of samples.

    Diversity strategy:
    1. Collect all examples matching role_filter with resolvable images.
    2. Partition by (state_type, location).
    3. Round-robin pick across buckets until n_samples reached, shuffle within bucket.
    """
    rng = random.Random(seed)

    buckets: dict[tuple[str, str], list[dict]] = {}
    for shard_name, line_idx, ex in iter_shards(dataset):
        if role_filter and ex.get("role") != role_filter:
            continue
        ps = ex.get("pre_state") or {}
        state_type = _state_type_from_pre_state(ps)
        location = ps.get("location") or "?"
        key = (state_type, location)
        ex["_shard"] = shard_name
        ex["_line_idx"] = line_idx
        buckets.setdefault(key, []).append(ex)

    # shuffle each bucket
    for v in buckets.values():
        rng.shuffle(v)

    # Want ~half overworld, ~half battle if available.
    ow_buckets = [k for k in buckets if k[0] == "overworld"]
    bt_buckets = [k for k in buckets if k[0] == "battle"]
    rng.shuffle(ow_buckets)
    rng.shuffle(bt_buckets)

    target_ow = n_samples // 2 + (n_samples % 2)  # slight bias to overworld if odd
    target_bt = n_samples - target_ow

    picked: list[dict] = []
    picked.extend(_round_robin_take(buckets, ow_buckets, target_ow))
    picked.extend(_round_robin_take(buckets, bt_buckets, target_bt))

    # If one category was under-filled (e.g. no battles), fill from the other.
    if len(picked) < n_samples:
        remaining = n_samples - len(picked)
        other_order = ow_buckets + bt_buckets
        picked.extend(_round_robin_take(buckets, other_order, remaining))

    samples: list[Sample] = []
    for i, ex in enumerate(picked[:n_samples]):
        b64, abs_path = _resolve_image(ex)
        if require_image and not b64:
            # try replacement from same bucket
            continue
        ps = ex.get("pre_state") or {}
        samples.append(
            Sample(
                idx=i,
                prompt=ex.get("prompt", ""),
                simplified_prompt=_build_simplified_prompt(ex),
                image_b64=b64,
                image_path=abs_path,
                pre_state=ps,
                raw_response=ex.get("raw_response", "") or "",
                completion=ex.get("completion", {}) or {},
                state_type=_state_type_from_pre_state(ps),
                location=ps.get("location") or "?",
                source=f"{ex.get('_shard','?')}:{ex.get('_line_idx','?')}",
            )
        )
    return samples


def _round_robin_take(
    buckets: dict[tuple[str, str], list[dict]],
    bucket_keys: list[tuple[str, str]],
    n: int,
) -> list[dict]:
    """Round-robin across buckets taking one at a time until n total collected."""
    picked: list[dict] = []
    if not bucket_keys:
        return picked
    # Work on shallow copies since we'll pop from them
    local = {k: list(buckets[k]) for k in bucket_keys}
    while len(picked) < n:
        progress = False
        for k in list(local.keys()):
            if not local[k]:
                del local[k]
                continue
            picked.append(local[k].pop())
            progress = True
            if len(picked) >= n:
                break
        if not progress:
            break
    return picked

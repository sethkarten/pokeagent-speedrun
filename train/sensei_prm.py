"""SENSEI-style PRM for DAgger.

Three changes vs the absolute-rubric PRM in `openclaw_judge._score_one`:

1. Goal context is cached once per iteration (not baked into per-step prompt):
   a single Gemini call at the start of a window asks, given the first
   screenshot + state, what are the NEXT 3-5 observable sub-goals. That text
   is then loaded as a fixed preamble into every pairwise comparison and
   carried into the teacher-relabel `teacher_hint`.

2. Per-step reward comes from a pairwise preference (SENSEI / MOTIF style)
   rather than an absolute rubric. For each step i we ask Gemini to compare
   state_i vs an anchor state (the rollout's first step if i < stride, else
   step i-stride). "B (current) closer to goal" → 1.0, "A (anchor)" → 0.0,
   "equal" → 0.5. This removes the reward channel where hard-SFT can
   exploit "good reasoning while stuck" to accumulate high PRM scores
   without moving (observed in B_v1 iter 2: 0.313 reward at (3,6) 64/64).

3. Position-delta hard gate: if `pre_state.player_coords` has not changed
   across a rolling window, reward is multiplied by a floor (default 0.2).
   Cheap sanity check that catches the "stuck but articulate" failure mode.

The function output matches `_score_traj_records` — each record gets
`_reward` in [0,1], `_teacher_hint`, `_image_path_abs` — so
`score_and_relabel` downstream doesn't change.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("sensei_prm")


# ── goal context (one-shot per window) ────────────────────────────────

_GOAL_PROMPT_TEMPLATE = """You are scoring a Pokémon {game} playing agent.

Given the attached screenshot and current game state, list the next 3-5 OBSERVABLE sub-goals the agent should accomplish. Focus on the immediate frontier, not the whole game.

Rules:
- Each sub-goal must describe a SPECIFIC, visually verifiable event (entering a named building, talking to a specific NPC, defeating a specific trainer, obtaining an item).
- Order by the expected completion sequence.
- No meta-strategy or reasoning tips; just the next observable events.

Format as a numbered list. Example:
1. Exit the player's house to Pallet Town.
2. Enter Professor Oak's Lab.
3. Receive starter Pokemon.
4. Defeat rival in first battle.
5. Leave Pallet Town via Route 1.

Current state:
{state}
"""


def get_or_build_goal_context(
    run_root: Path,
    first_image_path: Optional[str],
    first_pre_state: dict,
    game: str,
    judge_model: str,
    refresh_iter: int = 3,
    iter_idx: int = 0,
) -> str:
    """Return the cached goal context for this run; build it on first call
    (or refresh every `refresh_iter` iterations to follow the agent's
    progress, matching SENSEI Suppl. E.9's re-annotation step).
    """
    cache_path = run_root / "goal_context.txt"
    if cache_path.exists() and (iter_idx % refresh_iter) != 0:
        return cache_path.read_text()

    from train.openclaw_judge import _get_model, _load_image_bytes

    prompt = _GOAL_PROMPT_TEMPLATE.format(
        game=game,
        state=json.dumps({
            "location": first_pre_state.get("location"),
            "player_coords": first_pre_state.get("player_coords"),
            "is_in_battle": first_pre_state.get("is_in_battle"),
            "dialog_active": first_pre_state.get("dialog_active"),
        }, indent=2),
    )
    parts: list[Any] = [prompt]
    img = _load_image_bytes(first_image_path, None) if first_image_path else None
    if img:
        parts.append(img)

    m = _get_model(judge_model)
    for attempt in range(3):
        raw = m.generate(parts, timeout_s=60.0)
        if raw:
            break
        time.sleep(1 + attempt * 2)
    else:
        logger.warning("goal context call returned nothing; using fallback")
        raw = (f"1. Make progress past the current area ({first_pre_state.get('location')})."
               " 2. Explore new areas. 3. Engage any NPCs or trainers visible.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(raw)
    logger.info("[sensei] goal context (iter %d):\n%s", iter_idx, raw[:500])
    return raw


# ── pairwise preference over (anchor, current) ────────────────────────

_PAIR_PROMPT_TEMPLATE = """You are scoring progress in Pokémon {game}.

The agent's NEXT sub-goals are:
{goal_context}

Two screenshots are attached:
- SCREENSHOT A: state from earlier in the trajectory (anchor).
- SCREENSHOT B: state from later in the trajectory (current).

Which screenshot is CLOSER to accomplishing the next sub-goal?

Answer with EXACTLY one JSON object: {{"pref": "A"}} or {{"pref": "B"}} or {{"pref": "tie"}}.
No other text.
"""


def _pairwise_preference(
    anchor_image: Optional[bytes],
    current_image: Optional[bytes],
    goal_context: str,
    game: str,
    judge_model: str,
) -> float:
    """Return the "current is better" reward in [0, 1].
    A → 0.0, tie → 0.5, B → 1.0.
    """
    if anchor_image is None or current_image is None:
        return 0.5

    from train.openclaw_judge import _get_model

    prompt = _PAIR_PROMPT_TEMPLATE.format(game=game, goal_context=goal_context[:1500])
    parts: list[Any] = [prompt, anchor_image, current_image]
    m = _get_model(judge_model)
    for attempt in range(3):
        raw = m.generate(parts, timeout_s=45.0)
        if raw:
            break
        time.sleep(1 + attempt * 2)
    else:
        return 0.5

    text = raw.strip().lower()
    # tolerant parse: look for literal "a", "b", "tie"
    m_ = re.search(r'"pref"\s*:\s*"([abABTtIiEe]+)"', raw)
    pref = (m_.group(1).lower() if m_ else text[:16])
    if "tie" in pref:
        return 0.5
    if pref.startswith("b"):
        return 1.0
    if pref.startswith("a"):
        return 0.0
    return 0.5


# ── position gate ─────────────────────────────────────────────────────

def _position_gate_multipliers(records: list[dict], window: int = 5,
                               floor: float = 0.2) -> list[float]:
    """Per-step multiplier that damps reward when position is frozen.

    If the player_coords *and* location are identical for the trailing
    `window` steps and nothing event-like fired (battle/dialog transition),
    multiply the reward by `floor`. Scales linearly back up as soon as
    motion resumes.
    """
    coords = []
    locs = []
    in_batl = []
    in_dial = []
    for r in records:
        ps = r.get("pre_state", {}) or {}
        coords.append(tuple(ps.get("player_coords") or ()))
        locs.append(ps.get("location"))
        in_batl.append(bool(ps.get("is_in_battle")))
        in_dial.append(bool(ps.get("dialog_active")))

    mults = [1.0] * len(records)
    for i in range(len(records)):
        lo = max(0, i - window + 1)
        trail_coords = set(coords[lo:i + 1])
        trail_locs = set(locs[lo:i + 1])
        # any state-change event in the window releases the gate
        batl_change = len(set(in_batl[lo:i + 1])) > 1
        dial_change = len(set(in_dial[lo:i + 1])) > 1
        if (i - lo) >= (window - 1) \
                and len(trail_coords) == 1 \
                and len(trail_locs) == 1 \
                and not batl_change \
                and not dial_change:
            mults[i] = floor
    return mults


# ── main entrypoint: replaces _score_traj_records in dagger_prm ──────

def _resolve_screenshot(run_dir: Path, step_n: int, record: dict) -> Optional[str]:
    """Mirror of dagger_prm._resolve_screenshot."""
    for k in ("image_path", "image", "screenshot"):
        p = record.get(k)
        if not p:
            continue
        if not Path(p).is_absolute():
            p = str(run_dir / p)
        if Path(p).exists():
            return p
    cand = run_dir / "screenshots" / f"step_{int(step_n):05d}.png"
    if cand.exists():
        return str(cand)
    cand = run_dir / "screenshots" / f"step_{int(step_n)}.png"
    return str(cand) if cand.exists() else None


def score_traj_records_pairwise(
    records: list[dict],
    run_dir: Path,
    run_root: Path,
    judge_model: str,
    game: str,
    stride: int = 8,
    iter_idx: int = 0,
    max_parallel: int = 8,
    objective_block: str = "",
) -> list[dict]:
    """SENSEI-style scoring. Attaches `_reward`, `_teacher_hint`,
    `_image_path_abs` to every record. Public shape matches
    `_score_traj_records` so score_and_relabel can use this drop-in.

    When `objective_block` is provided (from the objective_context module
    reading the server's persisted objectives.json), we use the authoritative
    story objective as the goal for pairwise preference. Otherwise we fall
    back to the original ask-Gemini-from-first-screenshot goal_context.
    """
    from train.openclaw_judge import _load_image_bytes

    if not records:
        return records

    # 1. image paths + bytes
    for i, t in enumerate(records):
        t["_image_path_abs"] = _resolve_screenshot(run_dir, t.get("step", i), t)

    # 2. goal context — prefer the authoritative DirectObjective block if given.
    if objective_block.strip():
        goal_ctx = objective_block
    else:
        first_img = records[0].get("_image_path_abs")
        first_ps = records[0].get("pre_state", {}) or {}
        goal_ctx = get_or_build_goal_context(
            run_root=run_root,
            first_image_path=first_img,
            first_pre_state=first_ps,
            game=game,
            judge_model=judge_model,
            iter_idx=iter_idx,
        )

    # 3. image bytes cache
    img_bytes: dict[int, Optional[bytes]] = {}
    for i, t in enumerate(records):
        img_bytes[i] = _load_image_bytes(t["_image_path_abs"], None) if t["_image_path_abs"] else None

    # 4. pairwise jobs: step i vs anchor=max(0, i-stride)
    jobs: list[tuple[int, int]] = []   # (i, anchor_index)
    for i in range(len(records)):
        anchor = max(0, i - stride)
        if anchor == i:
            continue  # step 0 has no anchor
        jobs.append((i, anchor))

    logger.info("[sensei] dispatching %d pairwise pref calls (stride=%d, parallel=%d)",
                len(jobs), stride, max_parallel)

    def _run(job):
        i, a = job
        r = _pairwise_preference(
            anchor_image=img_bytes[a],
            current_image=img_bytes[i],
            goal_context=goal_ctx,
            game=game,
            judge_model=judge_model,
        )
        return i, r

    rewards: list[float] = [0.5] * len(records)  # step 0 gets 0.5 (no anchor)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        for i, r in ex.map(_run, jobs):
            rewards[i] = r

    # 5. position gate
    gate = _position_gate_multipliers(records)
    gated = [rewards[i] * gate[i] for i in range(len(records))]

    # 6. attach
    for i, t in enumerate(records):
        t["_reward"] = float(gated[i])
        t["_reward_detail"] = {
            "pairwise_pref": float(rewards[i]),
            "position_gate": float(gate[i]),
            "mode": "sensei_pairwise",
        }
        t["_teacher_hint"] = (
            f"Next sub-goals for this run:\n{goal_ctx.strip()[:1200]}\n\n"
            f"Current step {i} @ {t.get('pre_state', {}).get('location', '?')}. "
            f"Choose the action that advances the next sub-goal."
        )

    reward_mean = sum(gated) / max(1, len(gated))
    n_gated = sum(1 for g in gate if g < 1.0)
    logger.info("[sensei] reward_mean=%.3f, gated=%d/%d, stride=%d, iter=%d",
                reward_mean, n_gated, len(gate), stride, iter_idx)
    return records

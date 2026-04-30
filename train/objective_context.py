"""Objective-aware context for PRM + teacher prompts.

Loads the harness's 78-objective Red storyline (via
`DirectObjectiveManager.auto_save()` which writes to
`.pokeagent_cache/<run_id>/objectives.json`), exposes the current +
upcoming objectives to the PRM/teacher prompt builders, and runs a
Gemini "completion judge" after each rollout to advance the story
index persistently across DAgger iterations.

Flow:
1. Before iter N's rollout: dagger_prm reads `<output>/objectives_index.txt`
   (persisted from iter N-1) and passes `--direct-objectives-start <idx>`
   to run.py. The server loads the full 78-objective sequence and starts
   pointing at `idx`.
2. After iter N's rollout: we load `.pokeagent_cache/<run_id>/objectives.json`
   (the server's latest state), then call Gemini to check whether any of
   the next few objectives were completed this window; if so, advance
   the index. Persist the new index back to `<output>/objectives_index.txt`
   for iter N+1.
3. During PRM scoring + teacher relabel: inject `current_objective`
   (description + target_location + navigation_hint + completion_condition)
   and a preview of the next 2 objectives into every prompt. Replaces the
   rubric's ad-hoc `teacher_hint` and the SENSEI pairwise's
   screenshot-derived goal context.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("objective_context")

INDEX_FILE = "objectives_index.txt"


# ── load / persist story index across iters ──────────────────────────

def load_persisted_index(run_root: Path) -> int:
    """Read the story index saved after the previous iter (or 0 if first iter)."""
    p = run_root / INDEX_FILE
    if p.exists():
        try:
            return int(p.read_text().strip())
        except Exception:
            pass
    return 0


def save_persisted_index(run_root: Path, index: int) -> None:
    """Write the current story index for the next iter to pick up."""
    (run_root / INDEX_FILE).write_text(str(index))


# ── read objectives.json from cache ──────────────────────────────────

def load_objectives_state(run_cache_dir: Path) -> Optional[dict]:
    """Read the server's persisted objectives.json for this run's cache dir."""
    p = run_cache_dir / "objectives.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("objectives.json parse failed at %s: %s", p, e)
        return None


def get_objective_context(objectives_state: Optional[dict],
                          upcoming: int = 2,
                          completed_lookback: int = 5) -> dict:
    """Extract the currently-active story objective + upcoming + completed objectives.

    Returns a dict shaped for prompt injection:
        {
            "has_objective": bool,
            "current": {id, description, target_location, navigation_hint, completion_condition} | None,
            "upcoming": [ {id, description, target_location} ],
            "completed": [ {id, description, target_location} ],   # most-recent first
            "story_index": int,
            "story_total": int,
        }
    """
    if not objectives_state:
        return {"has_objective": False, "current": None, "upcoming": [],
                "completed": [], "story_index": 0, "story_total": 0}

    story = objectives_state.get("story", {})
    seq = story.get("sequence", [])
    idx = int(story.get("index", 0))

    def _obj_brief(o: dict) -> dict:
        return {
            "id": o.get("id"),
            "description": o.get("description", "")[:200],
            "target_location": o.get("target_location"),
            "navigation_hint": o.get("navigation_hint", "")[:400],
            "completion_condition": o.get("completion_condition"),
        }

    current = _obj_brief(seq[idx]) if 0 <= idx < len(seq) else None
    next_ones = [
        {k: v for k, v in _obj_brief(seq[j]).items()
         if k in ("id", "description", "target_location")}
        for j in range(idx + 1, min(idx + 1 + upcoming, len(seq)))
    ]
    # Most recent N completed objectives (story[idx-N..idx-1]), reverse chronological.
    completed = [
        {k: v for k, v in _obj_brief(seq[j]).items()
         if k in ("id", "description", "target_location")}
        for j in range(max(0, idx - completed_lookback), idx)
    ][::-1]
    return {
        "has_objective": current is not None,
        "current": current,
        "upcoming": next_ones,
        "completed": completed,
        "story_index": idx,
        "story_total": len(seq),
    }


def render_objective_block(ctx: dict) -> str:
    """Flat text block to inject into prompts (PRM / teacher).

    The teacher uses the COMPLETED list as a do-not-revisit signal: if the
    student walks back to a `target_location` in the completed list, the
    teacher should relabel toward the CURRENT objective instead. This
    mirrors how live ContinualHarness Gemini uses its working-memory
    achievement log to avoid regressing.
    """
    if not ctx.get("has_objective"):
        return ""
    cur = ctx["current"]
    out = []
    if ctx.get("completed"):
        out.append("## ALREADY COMPLETED — DO NOT WALK BACK TO THESE LOCATIONS")
        for c in ctx["completed"]:
            out.append(f"- {c['id']}: {c['description']} @ {c['target_location']}")
        out.append("")  # blank line
    out.extend([
        f"## CURRENT STORY OBJECTIVE ({ctx['story_index'] + 1}/{ctx['story_total']})",
        f"- id: {cur['id']}",
        f"- description: {cur['description']}",
        f"- target_location: {cur['target_location']}",
        f"- navigation_hint: {cur['navigation_hint']}",
        f"- completion_condition: {cur['completion_condition']}",
    ])
    if ctx["upcoming"]:
        out.append("## UPCOMING (for context)")
        for u in ctx["upcoming"]:
            out.append(f"- {u['id']}: {u['description']} @ {u['target_location']}")
    return "\n".join(out) + "\n"


# ── completion judge: advance the story index after a rollout ────────

_COMPLETION_PROMPT = """You are verifying whether a Pokemon Red agent has completed specific story objectives during a rollout window.

Given:
- CURRENT_OBJECTIVE and the next {lookahead} upcoming objectives
- A TRAJECTORY summary (per-step location, coords, and any state events from the last rollout window)

Decide how many objectives (starting from CURRENT_OBJECTIVE and going forward) were plausibly completed during this window. Completion means the `completion_condition` would likely be satisfied given the evidence in the trajectory.

Be conservative: if unsure, return 0. If the agent clearly reached a target_location AND triggered an event matching the completion_condition, advance.

Return JSON ONLY: {{"advance_by": <int 0-{max_advance}>, "reason": "<one sentence evidence>"}}

## OBJECTIVES
{objective_block}

## TRAJECTORY ({n_steps} steps)
{trajectory_summary}
"""


def _summarize_trajectory(records: list[dict], max_steps: int = 300) -> str:
    """Compact per-step summary for the completion judge.

    The previous default of 40 was a critical bug: a 64-step rollout that
    transitioned to a new map at step 50+ would have those transition
    records dropped, and the judge would say "no evidence of exit" while
    the rollout's actual shard showed 17 PalletTown steps. With max_steps
    300 we always cover 64- and 128-step rollouts in full; if a future
    config exceeds that, we keep the first half AND the last half so the
    endpoint is always visible.
    """
    n = len(records)
    if n <= max_steps:
        sample = list(enumerate(records))
    else:
        head = max_steps // 2
        tail = max_steps - head
        sample = (
            list(enumerate(records[:head]))
            + [(-1, None)]  # gap marker
            + list(enumerate(records[-tail:], start=n - tail))
        )

    lines = []
    for i, t in sample:
        if t is None:
            lines.append(f"... ({n - max_steps} mid-window steps truncated)")
            continue
        ps = t.get("pre_state", {}) or {}
        loc = ps.get("location", "?")
        coords = ps.get("player_coords") or ()
        batl = "B" if ps.get("is_in_battle") else ""
        dial = "D" if ps.get("dialog_active") else ""
        flags = (batl + dial) or "-"
        lines.append(f"{i:3d} {loc} {tuple(coords)} [{flags}]")
    return "\n".join(lines)


def judge_completion_advance(
    records: list[dict],
    objectives_state: dict,
    judge_model: str,
    lookahead: int = 4,
) -> tuple[int, str]:
    """Ask Gemini how many consecutive story objectives were completed this
    window. Returns (advance_by, reason)."""
    from train.openclaw_judge import _get_model

    ctx = get_objective_context(objectives_state, upcoming=lookahead - 1)
    if not ctx["has_objective"]:
        return 0, "no active objective"

    block = render_objective_block(ctx)
    prompt = _COMPLETION_PROMPT.format(
        lookahead=lookahead,
        max_advance=lookahead,
        objective_block=block,
        n_steps=len(records),
        trajectory_summary=_summarize_trajectory(records),
    )
    m = _get_model(judge_model)
    raw = None
    for attempt in range(3):
        raw = m.generate([prompt], timeout_s=45.0)
        if raw:
            break
        time.sleep(1 + attempt * 2)
    if not raw:
        return 0, "judge unreachable"

    # tolerant JSON parse
    m_obj = re.search(r"\{.*?\}", raw, re.DOTALL)
    try:
        parsed = json.loads(m_obj.group(0)) if m_obj else {}
    except Exception:
        parsed = {}
    advance = int(parsed.get("advance_by", 0) or 0)
    advance = max(0, min(advance, lookahead))
    reason = str(parsed.get("reason", ""))[:200]
    return advance, reason


def apply_advance_and_persist(
    objectives_state: dict,
    advance_by: int,
    run_cache_dir: Path,
    run_root: Path,
) -> int:
    """Advance the story_index by `advance_by`, write updated objectives.json
    back to the cache, and save the index into the run's output dir for the
    next iter. Returns the new index."""
    if advance_by <= 0:
        idx = int(objectives_state.get("story", {}).get("index", 0))
        save_persisted_index(run_root, idx)
        return idx

    story = objectives_state.setdefault("story", {})
    seq = story.get("sequence", [])
    idx = int(story.get("index", 0))
    new_idx = min(idx + advance_by, len(seq))

    # mark skipped objectives completed for transparency
    for j in range(idx, new_idx):
        if j < len(seq):
            seq[j]["completed"] = True
            seq[j]["completed_timestamp"] = seq[j].get("completed_timestamp") or ""
    story["index"] = new_idx

    # write back to the cache
    cache_path = run_cache_dir / "objectives.json"
    try:
        cache_path.write_text(json.dumps(objectives_state, indent=2))
    except Exception as e:
        logger.warning("could not write %s: %s", cache_path, e)

    save_persisted_index(run_root, new_idx)
    logger.info("[objective] advanced %d -> %d (gap=%d)", idx, new_idx, advance_by)
    return new_idx

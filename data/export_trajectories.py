"""Pokemon SFT trajectory exporter.

Joins everything we record per agent step into a single SFT-ready
JSONL row, applies per-trajectory and per-step filters, and writes
parquet shards optimized for Hugging Face ``datasets.load_dataset``.

Inputs (per run directory ``run_data/run_<id>/``):

  - ``trajectory_history.jsonl``    one entry per agent step:
                                    reasoning, action.tool_calls,
                                    pre_state, post_state, milestones
  - ``screenshots/step_NNNNN.png``  per-step screenshot, written by
                                    PokeAgent.run() (browser games
                                    write the same path naturally;
                                    Pokemon needs the patch in
                                    agents/PokeAgent.py to populate it)
  - ``prompt_evolution/llm_traces/llm_log.jsonl``
                                    full prompt + raw response per
                                    LLM call, with ``agent_step``
                                    linkage and model_info
  - ``end_state/game_state/milestones.json``
                                    final milestone state
  - ``prompt_evolution/meta_prompts/steps_X_to_Y_metadata.json``
                                    autoevolve directive output for
                                    each evolution window — what the
                                    evolver decided to add/update/keep

Output (one JSONL row per kept LLM call, plus a sidecar parquet):

Iteration unit is one LLM call, NOT one trajectory entry. A run with
N orchestrator steps and M subagent invocations produces (N + M) rows
(subject to filtering). Each row has a ``role`` of "orchestrator" or
"subagent" — the same model produced both, and the student should
learn to imitate either.

  {
    "schema_version":    2,
    "run_id":            "20260408_195739_autonomous_..._autoevolve",
    "step":              29,
    "role":              "subagent",                # or "orchestrator"
    "interaction_type":  "gemini_Custom_Combat_Handler",
    "screenshot_step_offset": -1,                   # 0 = exact, -k = walked back
    "image_path":        "run_data/.../screenshots/step_00028.png",
    "image_b64":         "<screenshot>",
    "prompt":            "...",        # full LLM prompt as logged
    "raw_response":      "...",        # full LLM response as logged
    "completion": {
      "reasoning":       "ANALYZE: ...",  # from trajectory entry (orchestrator)
      "tool_calls": [
        {"name": "press_buttons", "args": {...}}
      ]
    },
    "pre_state": {
      "location":        "ROUTE 101",
      "player_coords":   [12, 7],
      "context":         "overworld",
      "is_in_battle":    false,
      "dialog_active":   false,
      "milestones":      ["entered_lab", "talked_to_birch"]
    },
    "post_state": {
      "milestones":      ["entered_lab", "talked_to_birch", "left_lab"],
      "milestones_added": ["left_lab"]
    },
    "directive_window": {
      "window":          [110, 160],
      "directives":      [{"type": "skill_update", "id": "navigate_lab", ...}],
      "evolved":         true   # at least one skill/subagent created or updated
    },
    "model_info": {
      "model":           "gemini-3.1-pro-preview",
      "backend":         "gemini",
      "prompt_tokens":   8123,
      "completion_tokens": 142
    },
    "weight":            3.0,            # quality scoring
    "weight_reasons":    ["preceded_milestone", "evolved_skill_used"],
    "filter_status":     "kept"          # kept | dropped:<reason>
  }

Usage::

    .venv/bin/python -m data.export_trajectories \
        --runs run_data/run_20260408_032015 \
        --output data/sft_dataset/v1 \
        --teacher-model gemini-3.1-pro-preview

Run with ``--dry-run`` to print stats without writing the dataset.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("export_trajectories")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


SCHEMA_VERSION = 2

# ----------------------------------------------------------------------
# Interaction-type → role classifier
# ----------------------------------------------------------------------
#
# Each LLM call is logged with an `interaction_type` like
# `gemini_auto-evolve_orchestrator` or `gemini_Custom_Combat_Handler`.
# We classify these into coarse `role` buckets:
#
#   orchestrator : main planner/decider — every step
#   subagent     : Combat_Handler, Pokemon_Center_Healer, Custom_Subagent,
#                  Run_From_Battle, etc. ("it is all the same model at heart")
#   meta         : PromptOptimizer, HarnessEvolver_* — these are the
#                  evolution loop itself; we DO NOT want the student to
#                  learn to evolve its own harness, so they're excluded
#                  from training data by default.
#   unknown      : everything else


ROLE_ORCHESTRATOR = "orchestrator"
ROLE_SUBAGENT = "subagent"
ROLE_META = "meta"
ROLE_UNKNOWN = "unknown"

DEFAULT_INCLUDE_ROLES = (ROLE_ORCHESTRATOR, ROLE_SUBAGENT, ROLE_META)


def _classify_interaction_type(itype: str) -> str:
    """Map a raw interaction_type string into one of the role buckets."""
    if not itype:
        return ROLE_UNKNOWN
    s = itype.lower()
    if "orchestrator" in s or "autonomous_cli" in s or "cli_agent" in s:
        return ROLE_ORCHESTRATOR
    if "harnessevolver" in s or "promptoptimizer" in s or "process_trajectory" in s:
        return ROLE_META
    # Subagent naming is inconsistent — match the common patterns:
    # gemini_Custom_Combat_Handler, gemini_Custom_Pokemon_Center_Healer,
    # gemini_Custom_Custom_Subagent, gemini_Custom_Run_From_Battle, etc.
    if "custom_" in s or "subagent" in s or "_handler" in s or "_healer" in s:
        return ROLE_SUBAGENT
    return ROLE_UNKNOWN


# ----------------------------------------------------------------------
# Filter configuration
# ----------------------------------------------------------------------


@dataclass
class FilterConfig:
    """Knobs for the trajectory and step filters.

    Defaults are calibrated against the Pokemon Emerald autoevolve runs
    we collected pre-distillation. Tighten the trajectory filters for
    final dataset, loosen them for early calibration.
    """

    # Trajectory-level
    min_milestones_reached: int = 3
    max_no_tool_call_fraction: float = 0.30
    max_loop_window: int = 50

    # Step-level
    drop_failed_tool_calls: bool = True
    drop_repeated_calls_window: int = 3        # identical to last N steps
    regret_window: int = 5                     # next M steps undid the action
    drop_no_groundings: bool = True            # reasoning lacks game-state ref
    min_reasoning_chars: int = 80

    # Quality boosts (sample weight multipliers, applied multiplicatively)
    weight_preceded_milestone: float = 3.0
    weight_evolved_skill_used: float = 2.0
    weight_grounded_reasoning: float = 1.5
    weight_default: float = 1.0

    # Image filtering — drop transition frames where the agent saw a
    # half-rendered screen. The std check catches all-black / blank
    # frames. The pixel-delta check was originally a browser-game
    # filter (where frame-to-frame deltas of >30 indicate fade or
    # loading), but Pokemon emulator frames are DISCRETE — pressing a
    # button can change the entire screen (battle start, menu open,
    # warping into a new map) and those are legitimate state changes.
    # Default it to None and only re-enable for browser games.
    image_min_std: Optional[float] = 3.0
    image_max_pixel_delta: Optional[float] = None


# ----------------------------------------------------------------------
# Data sources — readers for each on-disk artifact
# ----------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("skipping malformed line in %s: %s", path, e)
    return out


def _load_milestones_final(run_dir: Path) -> List[str]:
    p = run_dir / "end_state" / "game_state" / "milestones.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return [k for k, v in data.items() if v]
        if isinstance(data, list):
            return list(data)
    except Exception as e:
        logger.warning("could not parse %s: %s", p, e)
    return []


def _load_directive_windows(run_dir: Path) -> List[Tuple[int, int, dict]]:
    """Read all autoevolve directive metadata files for a run.

    Each file is named ``steps_<lo>_to_<hi>_metadata.json`` and contains
    the evolver's output for that window. We return them as
    ``(lo, hi, payload)`` so we can attach to each step the directive
    window it falls inside.
    """
    out = []
    meta_dir = run_dir / "prompt_evolution" / "meta_prompts"
    if not meta_dir.exists():
        return out
    for f in sorted(meta_dir.glob("steps_*_to_*_metadata.json")):
        # parse filename
        try:
            stem = f.stem  # steps_-39_to_10_metadata
            parts = stem.replace("_metadata", "").split("_")
            # ['steps', '-39', 'to', '10']
            lo = int(parts[1])
            hi = int(parts[3])
        except (IndexError, ValueError):
            logger.debug("skipping unparseable directive file: %s", f.name)
            continue
        try:
            payload = json.loads(f.read_text())
        except Exception as e:
            logger.warning("could not parse %s: %s", f, e)
            continue
        out.append((lo, hi, payload))
    return out


def _candidate_llm_log_paths(run_dir: Path) -> List[Path]:
    """Find every llm_log file that could belong to this run.

    Pokemon and browser runs use different conventions:

    1. Per-run mirror at ``<run_dir>/prompt_evolution/llm_traces/llm_log.jsonl``
       — written by `prompt_evolution/llm_traces/` infrastructure.
    2. Global file at ``llm_logs/llm_log_<session_id>.jsonl`` —
       always written by `utils.data_persistence.llm_logger`. The
       session_id is the leading ``YYYYMMDD_HHMMSS`` of the run dir
       name.

    We return both and let the indexer merge them. Last-writer-wins
    handles overlap.
    """
    out: List[Path] = []
    per_run = run_dir / "prompt_evolution" / "llm_traces" / "llm_log.jsonl"
    if per_run.exists():
        out.append(per_run)

    # Extract YYYYMMDD_HHMMSS prefix from run_dir name (handles both
    # "run_<ts>" and "<ts>_<suffix>" naming conventions).
    name = run_dir.name
    if name.startswith("run_"):
        name = name[4:]
    parts = name.split("_")
    if len(parts) >= 2:
        session_id = f"{parts[0]}_{parts[1]}"
        gp = Path("llm_logs") / f"llm_log_{session_id}.jsonl"
        if gp.exists():
            out.append(gp)
    return out


def _load_llm_traces(
    run_dir: Path,
    include_roles: Iterable[str],
) -> List[dict]:
    """Return all trainable LLM interactions for a run, chronological.

    Reads from every candidate llm_log path (per-run mirror + global
    file). Each kept entry has a synthetic ``_role`` field attached
    classifying it as orchestrator/subagent/meta. Entries whose role
    isn't in ``include_roles`` are dropped here so downstream code
    can iterate the list without re-checking.

    Sort order is (agent_step, timestamp). Multiple entries can share
    the same agent_step — that's normal when an orchestrator decision
    triggers a subagent within the same step. The orchestrator entry
    will sort first because it logs before the subagent runs.

    De-dup: a single (run_dir, per-run mirror, global log) entry can
    appear in multiple files. We key by (timestamp, interaction_type,
    agent_step) and keep the first instance.
    """
    include = set(include_roles)
    seen_keys: set = set()
    out: List[dict] = []
    for p in _candidate_llm_log_paths(run_dir):
        for entry in _read_jsonl(p):
            if entry.get("type") != "interaction":
                continue
            role = _classify_interaction_type(entry.get("interaction_type", ""))
            if role not in include:
                continue
            step = entry.get("agent_step")
            if step is None:
                continue
            key = (
                entry.get("timestamp"),
                entry.get("interaction_type"),
                int(step),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            entry = dict(entry)
            entry["_role"] = role
            out.append(entry)
    out.sort(key=lambda e: (int(e.get("agent_step") or 0), e.get("timestamp") or ""))
    return out


# ----------------------------------------------------------------------
# Per-step record builder
# ----------------------------------------------------------------------


def _extract_reasoning(entry: dict) -> str:
    """Pull the real reasoning text out of a trajectory entry.

    The trajectory's top-level `reasoning` field is unreliable — it
    often contains placeholder strings like "Executed press_buttons"
    or is empty entirely. Gemini's actual ANALYZE/PLAN reasoning lives
    inside ``action.tool_calls[0].args.reasoning`` (or `.reason`),
    populated when the model fills in the structured tool call.
    Fall back to the top-level field only if the tool_call args
    don't have anything substantive.
    """
    action = entry.get("action") or {}
    tool_calls = action.get("tool_calls") or []
    candidates: List[str] = []
    for tc in tool_calls:
        args = (tc or {}).get("args") or {}
        for key in ("reasoning", "reason", "rationale"):
            v = args.get(key)
            if v and isinstance(v, str):
                candidates.append(v)
    # Concatenate all tool_call reasonings (rare but possible).
    if candidates:
        return "\n".join(candidates)
    top = entry.get("reasoning")
    return top if isinstance(top, str) else ""


def _grounded_reasoning(text: str, pre_state: dict) -> bool:
    """Heuristic: does the reasoning text reference specific game state?

    Looks for location names, player coords, specific buttons,
    Pokemon-game-specific keywords (Birch, Pokeball, etc), or
    structural ANALYZE/PLAN markers that indicate the model is
    actually doing chain-of-thought rather than generating boilerplate.
    """
    if not text:
        return False
    text_lower = text.lower()

    # Structural markers from PokeAgent's prompt template — Gemini
    # follows this pattern strongly, so its presence is a strong
    # signal of grounded reasoning even if the specific tokens we
    # check below don't appear.
    if "analyze:" in text_lower and "plan:" in text_lower:
        return True

    location = (pre_state.get("location") or "").lower()
    if location and any(tok in text_lower for tok in location.split() if len(tok) > 3):
        return True
    coords = pre_state.get("player_coords") or []
    if coords and any(f"({c}" in text_lower or f", {c}" in text_lower for c in coords):
        return True

    # Pokemon-specific grounding tokens — recognising in-game proper
    # nouns means the model is paying attention to the screen.
    pokemon_keywords = (
        # Emerald-specific
        "birch", "mom", "may", "brendan", "treecko", "torchic", "mudkip",
        # Red-specific
        "oak", "blue", "gary", "charmander", "bulbasaur", "squirtle",
        "pikachu", "brock", "misty", "surge", "pewter", "cerulean",
        "vermilion", "pallet",
        # Generic Pokemon
        "pokeball", "pokémon", "pokemon", "wild", "battle", "tackle",
        "potion", "trainer", "gym", "rival", "dialogue", "menu",
        "bag", "party", "map", "town", "route", "lab",
    )
    for kw in pokemon_keywords:
        if kw in text_lower:
            return True

    for btn in ("up", "down", "left", "right", "select", "start",
                "press a", "press b"):
        if btn in text_lower:
            return True
    return False


def _image_qa(image_bytes: bytes, prev_image_bytes: Optional[bytes],
              cfg: FilterConfig) -> Optional[str]:
    """Cheap image quality check. Returns drop reason or None."""
    if cfg.image_min_std is None and cfg.image_max_pixel_delta is None:
        return None
    try:
        from io import BytesIO
        from PIL import Image
        import numpy as np
        img = np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))
        if cfg.image_min_std is not None and float(img.std()) < cfg.image_min_std:
            return f"low_image_std<{cfg.image_min_std}"
        if cfg.image_max_pixel_delta is not None and prev_image_bytes is not None:
            prev = np.array(Image.open(BytesIO(prev_image_bytes)).convert("RGB"))
            if prev.shape == img.shape:
                delta = float(np.abs(img.astype("int16") - prev.astype("int16")).mean())
                if delta > cfg.image_max_pixel_delta:
                    return f"frame_transition_delta>{cfg.image_max_pixel_delta}"
    except Exception as e:
        logger.debug("image qa failed: %s", e)
    return None


def _filter_step(
    entry: dict,
    prev_entries: List[dict],
    next_entries: List[dict],
    image_bytes: Optional[bytes],
    prev_image_bytes: Optional[bytes],
    cfg: FilterConfig,
) -> Tuple[bool, str]:
    """Apply step-level filters. Returns (kept, reason)."""
    action = entry.get("action") or {}
    tool_calls = action.get("tool_calls") or []
    if not tool_calls:
        return False, "no_tool_call"
    if cfg.drop_failed_tool_calls:
        outcome = entry.get("outcome") or {}
        if outcome.get("success") is False:
            return False, "tool_call_failed"

    # Looping: identical to recent N steps
    sig = json.dumps([{"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls],
                     sort_keys=True)
    for prev in prev_entries[-cfg.drop_repeated_calls_window:]:
        prev_sig = json.dumps(
            [{"name": tc.get("name"), "args": tc.get("args")}
             for tc in (prev.get("action") or {}).get("tool_calls") or []],
            sort_keys=True,
        )
        if prev_sig == sig:
            return False, "repeat_of_recent_action"

    # Regret detection: did next M steps undo this one?
    pre_coords = entry.get("player_coords")
    if pre_coords:
        for nxt in next_entries[: cfg.regret_window]:
            nxt_coords = nxt.get("player_coords")
            if nxt_coords == pre_coords:
                # Same coords M steps later — only "regret" if there
                # was movement in between, otherwise the agent is
                # just stuck in a dialog.
                intermediate = next_entries[: cfg.regret_window]
                moved = any(
                    e.get("player_coords") != pre_coords
                    for e in intermediate
                )
                if moved:
                    return False, "regret_returned_to_origin"
                break

    # Directional no-op: agent pressed ONLY directional buttons but
    # coords didn't change AND not in battle/dialog. This is
    # wall-pressing — bad training signal. A/B/START presses that
    # don't change coords are legitimate (NPC interaction, menus).
    DIRECTIONAL = {"UP", "DOWN", "LEFT", "RIGHT"}
    if tool_calls and tool_calls[0].get("name") == "press_buttons":
        btns = (tool_calls[0].get("args") or {}).get("buttons") or []
        if btns and all(b.upper() in DIRECTIONAL for b in btns):
            pre_state = entry.get("pre_state") or {}
            coords = entry.get("player_coords")
            in_battle = pre_state.get("is_in_battle", False)
            dialog = pre_state.get("dialog_active", False)
            if not in_battle and not dialog and coords:
                # Check if coords changed in the NEXT entry
                if next_entries:
                    next_coords = next_entries[0].get("player_coords")
                    next_loc = next_entries[0].get("location")
                    cur_loc = entry.get("location")
                    if (next_coords == coords and next_loc == cur_loc):
                        return False, "directional_noop"

    # Consecutive no-op streak: if this step AND the previous N steps
    # all have the same coords + location + not in battle/dialog,
    # the agent is stuck. Drop after 5 consecutive static steps.
    if len(prev_entries) >= 5:
        coords = entry.get("player_coords")
        loc = entry.get("location")
        pre_state = entry.get("pre_state") or {}
        if (coords and loc
                and not pre_state.get("is_in_battle")
                and not pre_state.get("dialog_active")):
            all_same = all(
                p.get("player_coords") == coords
                and p.get("location") == loc
                and not (p.get("pre_state") or {}).get("is_in_battle")
                and not (p.get("pre_state") or {}).get("dialog_active")
                for p in prev_entries[-5:]
            )
            if all_same:
                return False, "consecutive_static_streak"

    # Reasoning grounding — use the *real* reasoning from tool_call
    # args, not the trajectory's top-level placeholder field.
    reasoning = _extract_reasoning(entry)
    if cfg.drop_no_groundings:
        pre_state = entry.get("pre_state") or {}
        if not _grounded_reasoning(reasoning, pre_state):
            if len(reasoning) < cfg.min_reasoning_chars:
                return False, "ungrounded_short_reasoning"

    # Image quality
    if image_bytes:
        drop = _image_qa(image_bytes, prev_image_bytes, cfg)
        if drop:
            return False, drop

    return True, "kept"


def _filter_subagent_call(
    llm: dict,
    image_bytes: Optional[bytes],
    prev_image_bytes: Optional[bytes],
    cfg: FilterConfig,
) -> Tuple[bool, str]:
    """Lighter-weight filter for subagent LLM calls.

    Subagents are short-lived tools (Combat_Handler picks an attack,
    Pokemon_Center_Healer navigates the heal flow). They intentionally
    repeat actions and don't have an enclosing trajectory entry, so
    the loop / regret / no-tool-call filters don't apply. We just
    sanity-check the response and the screenshot.
    """
    response = llm.get("response") or ""
    if not response.strip():
        return False, "subagent_empty_response"
    # Failed/short responses (no function call AND no substantive text)
    md = llm.get("metadata") or {}
    has_fn = bool(md.get("has_function_call"))
    if not has_fn and len(response.strip()) < cfg.min_reasoning_chars:
        return False, "subagent_short_no_function_call"
    if image_bytes:
        drop = _image_qa(image_bytes, prev_image_bytes, cfg)
        if drop:
            return False, drop
    return True, "kept"


def _filter_trajectory(entries: List[dict], cfg: FilterConfig) -> Tuple[bool, str]:
    """Apply trajectory-level filters. Returns (keep, reason)."""
    if not entries:
        return False, "empty_trajectory"

    n_steps = len(entries)
    no_tool = sum(1 for e in entries if not (e.get("action", {}).get("tool_calls")))
    if no_tool / n_steps > cfg.max_no_tool_call_fraction:
        return False, f"too_many_no_tool_call_steps_{no_tool}/{n_steps}"

    # Loop window: longest run of identical actions
    cur_sig = None
    cur_len = 0
    max_len = 0
    for e in entries:
        sig = json.dumps(e.get("action") or {}, sort_keys=True)
        if sig == cur_sig:
            cur_len += 1
        else:
            cur_sig = sig
            cur_len = 1
        max_len = max(max_len, cur_len)
    if max_len > cfg.max_loop_window:
        return False, f"long_loop_{max_len}"

    return True, "kept"


def _compute_weight(entry: dict, next_entries: List[dict],
                    directives: List[dict], cfg: FilterConfig) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    weight = cfg.weight_default

    # Preceded a milestone increment?
    pre_ms = set((entry.get("pre_state") or {}).get("milestones") or [])
    for nxt in next_entries[:5]:
        post_ms = set((nxt.get("pre_state") or {}).get("milestones") or [])
        if post_ms - pre_ms:
            weight *= cfg.weight_preceded_milestone
            reasons.append("preceded_milestone")
            break

    # Came from an evolved skill — was a process_skill or evolved
    # skill referenced in the directive payload?
    if directives:
        for d in directives:
            if d.get("type") in ("skill_update", "skill_create"):
                weight *= cfg.weight_evolved_skill_used
                reasons.append("evolved_skill_used")
                break

    # Reasoning is grounded
    pre_state = entry.get("pre_state") or {}
    if _grounded_reasoning(entry.get("reasoning", ""), pre_state):
        weight *= cfg.weight_grounded_reasoning
        reasons.append("grounded_reasoning")

    return weight, reasons


def _screenshot_dirs(run_dir: Path) -> List[Path]:
    """Candidate ``screenshots/`` dirs to probe for a given run.

    Pokemon runs split data: trajectory_history.jsonl ends up in
    ``.pokeagent_cache/<run_id>/`` while screenshots end up in
    ``run_data/<run_id>/screenshots/``. Browser runs put both under
    ``run_data/<run_id>/``. So we probe both locations from whichever
    one the caller passed in as ``run_dir``.
    """
    out = [run_dir / "screenshots"]
    name = run_dir.name
    if ".pokeagent_cache" in str(run_dir):
        out.append(Path("run_data") / name / "screenshots")
    else:
        out.append(Path(".pokeagent_cache") / name / "screenshots")
    return [d for d in out if d.exists()]


def _step_screenshot(run_dir: Path, step: int) -> Optional[Path]:
    for d in _screenshot_dirs(run_dir):
        for pattern in ("step_{:05d}.png", "step_{:04d}.png"):
            candidate = d / pattern.format(step)
            if candidate.exists():
                return candidate
    return None


def _step_screenshot_with_fallback(
    run_dir: Path,
    step: int,
    max_gap: int = 50,
) -> Tuple[Optional[Path], int]:
    """Look up the per-step screenshot, falling back to recent steps.

    Subagent steps don't always have a screenshot of their own —
    only the orchestrator path saves screenshots in PokeAgent. When
    a subagent is invoked the agent is still looking at the most
    recent orchestrator screenshot, so falling back to step-1 / step-2
    is the right semantic match.

    Returns (path, offset) where offset is 0 for an exact match or
    a negative integer indicating how many steps backward we walked.
    Returns (None, 0) if nothing is found within ``max_gap``.
    """
    p = _step_screenshot(run_dir, step)
    if p is not None:
        return p, 0
    for delta in range(1, max_gap + 1):
        p = _step_screenshot(run_dir, step - delta)
        if p is not None:
            return p, -delta
    return None, 0


def _directive_window_for_step(step: int, windows: List[Tuple[int, int, dict]]) -> Optional[dict]:
    """Find the directive window covering this step (if any)."""
    for lo, hi, payload in windows:
        if lo <= step <= hi:
            return {
                "window": [lo, hi],
                "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
                "evolved": bool(payload.get("evolved") or payload.get("changes")
                                if isinstance(payload, dict) else False),
                "raw": payload if isinstance(payload, dict) else None,
            }
    return None


# ----------------------------------------------------------------------
# Main processor
# ----------------------------------------------------------------------


def _resolve_trajectory_path(run_dir: Path) -> Optional[Path]:
    """Find trajectory_history.jsonl for a run.

    Two conventions in the wild:

    - ``run_data/<run_id>/trajectory_history.jsonl`` — current standard,
      used by browser runs and the post-2026-04-08 PokeAgent path.
    - ``.pokeagent_cache/<run_id>/trajectory_history.jsonl`` — legacy
      Pokemon convention; the cache mirrors the run dir but lives at
      the repo root. Some Pokemon runs still write here.

    We probe both and return whichever exists. If both exist, prefer
    the run_data copy (it's newer / canonical).
    """
    candidates = [
        run_dir / "trajectory_history.jsonl",
        Path(".pokeagent_cache") / run_dir.name / "trajectory_history.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def process_run(
    run_dir: Path,
    cfg: FilterConfig,
    embed_image: bool = True,
    teacher_filter: Optional[str] = None,
    include_roles: Iterable[str] = DEFAULT_INCLUDE_ROLES,
    screenshot_max_gap: int = 50,
) -> Tuple[List[dict], Dict[str, int]]:
    """Process one run directory. Returns (kept_records, stats).

    Iteration unit is one LLM call (orchestrator + subagents), not one
    trajectory entry. Trajectory entries are loaded for state context
    only — they're indexed by step number and looked up per call.
    Subagent calls share their orchestrator's screenshot (with up to
    ``screenshot_max_gap`` step fallback for races).
    """
    include_roles = tuple(include_roles)
    stats = {
        "trajectory_kept": 0,
        "trajectory_dropped": 0,
        "interactions_total": 0,
        "interactions_kept": 0,
        "interactions_dropped": 0,
        # Per-role keep counts (filled in below)
    }
    drop_reasons: Dict[str, int] = {}

    traj_path = _resolve_trajectory_path(run_dir)
    if traj_path is None:
        logger.info("[%s] no trajectory_history.jsonl found in run dir or "
                    ".pokeagent_cache — skipping", run_dir.name)
        stats["trajectory_dropped"] += 1
        drop_reasons["no_trajectory_file"] = 1
        return [], {**stats, "drop_reasons": drop_reasons}
    entries = _read_jsonl(traj_path)

    if not entries:
        logger.info("[%s] no trajectory entries — skipping", run_dir.name)
        stats["trajectory_dropped"] += 1
        drop_reasons["empty_trajectory"] = drop_reasons.get("empty_trajectory", 0) + 1
        return [], {**stats, "drop_reasons": drop_reasons}

    keep, reason = _filter_trajectory(entries, cfg)
    if not keep:
        logger.info("[%s] dropped trajectory: %s", run_dir.name, reason)
        stats["trajectory_dropped"] += 1
        drop_reasons[f"trajectory:{reason}"] = drop_reasons.get(f"trajectory:{reason}", 0) + 1
        return [], {**stats, "drop_reasons": drop_reasons}

    stats["trajectory_kept"] += 1

    # Trajectory entries indexed by step → state context for any role.
    traj_by_step: Dict[int, dict] = {}
    for i, e in enumerate(entries):
        s = int(e.get("step", i + 1))
        traj_by_step[s] = e

    llm_traces = _load_llm_traces(run_dir, include_roles=include_roles)
    directive_windows = _load_directive_windows(run_dir)

    # Skill audit: build a set of skills that were created (via
    # process_skill) and then later used with a FAILED outcome. We
    # exclude the evolver traces that CREATED those broken skills —
    # training on them teaches the student to produce buggy code.
    # Also exclude skills that were created but NEVER used (no signal
    # about whether they work).
    _skill_created_at: Dict[str, int] = {}  # skill_id → step
    _skill_used_ok: set = set()             # skill_ids that ran successfully
    _skill_used_fail: set = set()           # skill_ids that failed
    for e in entries:
        tc_list = (e.get("action") or {}).get("tool_calls") or []
        for tc in tc_list:
            args = tc.get("args") or {}
            if tc.get("name") == "process_skill":
                for ent in (args.get("entries") or []):
                    sid = (ent.get("id") or ent.get("name") or "") if isinstance(ent, dict) else ""
                    if sid:
                        _skill_created_at[sid] = int(e.get("step", 0))
            elif tc.get("name") == "run_skill":
                sid = args.get("skill_id") or ""
                outcome = (e.get("outcome") or {}).get("success")
                if outcome is True:
                    _skill_used_ok.add(sid)
                elif outcome is False:
                    _skill_used_fail.add(sid)
    _bad_skill_steps: set = set()
    for sid, step in _skill_created_at.items():
        if sid in _skill_used_fail and sid not in _skill_used_ok:
            _bad_skill_steps.add(step)
            logger.info("[%s] skill audit: %s created at step %d failed on use — "
                        "excluding evolver trace", run_dir.name, sid, step)
    if _bad_skill_steps:
        logger.info("[%s] skill audit: %d bad skill creation steps flagged",
                    run_dir.name, len(_bad_skill_steps))
    stats["interactions_total"] = len(llm_traces)

    if teacher_filter:
        # Drop the run if any kept interaction used a different model.
        seen = set()
        for v in llm_traces:
            mi = v.get("model_info") or {}
            seen.add(mi.get("model", ""))
        if seen and teacher_filter not in seen:
            logger.info(
                "[%s] dropped trajectory: teacher mismatch (saw %s, want %s)",
                run_dir.name, sorted(seen), teacher_filter,
            )
            stats["trajectory_kept"] -= 1
            stats["trajectory_dropped"] += 1
            drop_reasons["teacher_mismatch"] = drop_reasons.get("teacher_mismatch", 0) + 1
            return [], {**stats, "drop_reasons": drop_reasons}

    kept_records: List[dict] = []
    role_kept: Dict[str, int] = {}
    role_dropped: Dict[str, int] = {}
    prev_image_bytes: Optional[bytes] = None

    for llm in llm_traces:
        role = llm.get("_role", ROLE_UNKNOWN)
        step = int(llm.get("agent_step") or 0)
        entry = traj_by_step.get(step) or {}
        # For subagent calls without a matching trajectory entry, fall
        # back to the closest preceding entry so state context still
        # makes sense.
        if not entry and role == ROLE_SUBAGENT:
            for s in range(step, max(step - screenshot_max_gap, 0) - 1, -1):
                if s in traj_by_step:
                    entry = traj_by_step[s]
                    break

        # Locate trajectory neighbors by step for filter context.
        # The neighbors are sequential trajectory entries (orchestrator
        # decisions), not other LLM calls.
        sorted_steps = sorted(traj_by_step.keys())
        try:
            j = sorted_steps.index(int(entry.get("step", step)))
        except ValueError:
            j = -1
        prev_entries = [traj_by_step[s] for s in sorted_steps[:j]] if j >= 0 else []
        next_entries = [traj_by_step[s] for s in sorted_steps[j + 1:]] if j >= 0 else []

        ss_path, ss_offset = _step_screenshot_with_fallback(
            run_dir, step, max_gap=screenshot_max_gap,
        )
        image_bytes = ss_path.read_bytes() if ss_path else None

        if image_bytes is None:
            stats["interactions_dropped"] += 1
            drop_reasons["missing_screenshot"] = (
                drop_reasons.get("missing_screenshot", 0) + 1
            )
            role_dropped[role] = role_dropped.get(role, 0) + 1
            continue

        # Filter:
        #   - Orchestrator: full trajectory-context filters (loops,
        #     regret, no_tool_call, image qa, reasoning grounding).
        #   - Subagent: image qa + raw-response sanity. Subagents are
        #     short-lived and intentionally repetitive (e.g. selecting
        #     the same attack), so the loop / regret filters don't
        #     apply. Reasoning grounding only applies if a trajectory
        #     entry exists for context.
        # Skill audit: drop evolver traces that created broken skills
        if role == ROLE_META and step in _bad_skill_steps:
            keep_step, drop_reason = False, "bad_skill_creation"
        elif role == ROLE_ORCHESTRATOR and entry:
            keep_step, drop_reason = _filter_step(
                entry, prev_entries, next_entries,
                image_bytes, prev_image_bytes, cfg,
            )
        elif role == ROLE_SUBAGENT:
            keep_step, drop_reason = _filter_subagent_call(
                llm, image_bytes, prev_image_bytes, cfg,
            )
        else:
            keep_step, drop_reason = True, "kept"

        if not keep_step:
            stats["interactions_dropped"] += 1
            drop_reasons[drop_reason] = drop_reasons.get(drop_reason, 0) + 1
            role_dropped[role] = role_dropped.get(role, 0) + 1
            prev_image_bytes = image_bytes
            continue

        prompt = llm.get("prompt", "")
        raw_response = llm.get("response", "")
        model_info = llm.get("model_info", {})
        token_usage = (llm.get("metadata") or {}).get("token_usage") or {}
        interaction_type = llm.get("interaction_type", "")

        directive_window = _directive_window_for_step(step, directive_windows)

        # Milestones diff from this step's entry to the next trajectory entry
        pre_ms = (entry.get("pre_state") or {}).get("milestones") or []
        next_idx = j + 1 if (j >= 0 and j + 1 < len(sorted_steps)) else None
        next_entry = traj_by_step[sorted_steps[next_idx]] if next_idx is not None else {}
        next_ms = (next_entry.get("pre_state") or {}).get("milestones") or []
        added_ms = sorted(set(next_ms) - set(pre_ms))

        weight, weight_reasons = _compute_weight(entry, next_entries, [], cfg)

        # For orchestrator records the completion comes from the
        # trajectory entry's tool_calls (which is the actually-executed
        # action chain). For subagent records the trajectory has no
        # corresponding entry, so use the raw response — the student
        # will learn the subagent format from the response text.
        if role == ROLE_ORCHESTRATOR and entry:
            completion = {
                "reasoning": _extract_reasoning(entry),
                "tool_calls": (entry.get("action") or {}).get("tool_calls") or [],
            }
        else:
            completion = {
                "reasoning": _extract_reasoning(entry) if entry else "",
                "tool_calls": (entry.get("action") or {}).get("tool_calls") or [] if entry else [],
            }

        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "step": step,
            "role": role,
            "interaction_type": interaction_type,
            "screenshot_step_offset": ss_offset,
            "image_path": str(ss_path) if ss_path else None,
            "image_b64": base64.b64encode(image_bytes).decode() if (image_bytes and embed_image) else None,
            "prompt": prompt,
            "raw_response": raw_response,
            "completion": completion,
            "pre_state": entry.get("pre_state") or {},
            "post_state": {
                "milestones": next_ms,
                "milestones_added": added_ms,
            },
            "directive_window": directive_window,
            "model_info": {
                "model": model_info.get("model", ""),
                "backend": model_info.get("backend", ""),
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            },
            "weight": weight,
            "weight_reasons": weight_reasons,
            "filter_status": "kept",
        }
        kept_records.append(record)
        stats["interactions_kept"] += 1
        role_kept[role] = role_kept.get(role, 0) + 1
        prev_image_bytes = image_bytes

    stats["role_kept"] = role_kept
    stats["role_dropped"] = role_dropped
    return kept_records, {**stats, "drop_reasons": drop_reasons}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _resolve_runs(args_runs: List[str]) -> List[Path]:
    """Walk the user's --runs args and return run dirs to process.

    A run dir is anything with a trajectory_history.jsonl in either
    its own root OR the matching .pokeagent_cache mirror. We accept
    individual run dirs and parent dirs (which we glob into).
    """
    def _has_trajectory(d: Path) -> bool:
        if (d / "trajectory_history.jsonl").exists():
            return True
        cache_mirror = Path(".pokeagent_cache") / d.name / "trajectory_history.jsonl"
        return cache_mirror.exists()

    out: List[Path] = []
    for entry in args_runs:
        p = Path(entry)
        if not p.is_dir():
            logger.warning("skipping unknown path: %s", entry)
            continue
        if _has_trajectory(p):
            out.append(p)
            continue
        # Treat as a parent dir — glob run-shaped children.
        for sub in sorted(list(p.glob("run_*")) + list(p.glob("2026*"))):
            if sub.is_dir() and _has_trajectory(sub):
                out.append(sub)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run dirs to export. Can be individual run_<id> "
                         "dirs or a parent like 'run_data'.")
    ap.add_argument("--output", required=True,
                    help="Output directory for sharded JSONL.")
    ap.add_argument("--shard-size", type=int, default=2000,
                    help="Examples per shard.")
    ap.add_argument("--teacher-model", default=None,
                    help="Filter to runs that used this teacher model "
                         "(matched against model_info.model).")
    ap.add_argument("--no-embed-image", action="store_true",
                    help="Don't inline screenshot bytes; reference by path only.")
    ap.add_argument("--include-roles", default=",".join(DEFAULT_INCLUDE_ROLES),
                    help="Comma-separated roles to include. Available: "
                         f"{ROLE_ORCHESTRATOR}, {ROLE_SUBAGENT}, {ROLE_META}. "
                         "Default excludes meta (PromptOptimizer/HarnessEvolver).")
    ap.add_argument("--screenshot-max-gap", type=int, default=50,
                    help="For interactions without an exact-step screenshot, "
                         "fall back to the most recent within this many steps. "
                         "Subagents like Combat_Handler can run 30+ calls "
                         "between orchestrator steps without a fresh "
                         "screenshot, so the gap needs to be large enough to "
                         "reach the triggering orchestrator's screenshot.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats without writing the dataset.")
    args = ap.parse_args()

    include_roles = tuple(r.strip() for r in args.include_roles.split(",") if r.strip())

    cfg = FilterConfig()
    runs = _resolve_runs(args.runs)
    logger.info("processing %d runs", len(runs))

    out_dir = Path(args.output)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_stats: Dict[str, int] = {
        "runs_seen": 0,
        "runs_with_kept_steps": 0,
        "trajectory_kept": 0,
        "trajectory_dropped": 0,
        "interactions_total": 0,
        "interactions_kept": 0,
        "interactions_dropped": 0,
    }
    aggregate_drops: Dict[str, int] = {}
    aggregate_role_kept: Dict[str, int] = {}
    aggregate_role_dropped: Dict[str, int] = {}

    shard_idx = 0
    shard_writer = None
    shard_count = 0
    total_kept = 0

    def _open_shard():
        nonlocal shard_writer, shard_count
        if args.dry_run:
            return
        shard_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
        shard_writer = shard_path.open("w")
        shard_count = 0
        logger.info("opening %s", shard_path)

    if not args.dry_run:
        _open_shard()

    for run in runs:
        aggregate_stats["runs_seen"] += 1
        records, stats = process_run(
            run, cfg,
            embed_image=not args.no_embed_image,
            teacher_filter=args.teacher_model,
            include_roles=include_roles,
            screenshot_max_gap=args.screenshot_max_gap,
        )
        for k, v in stats.items():
            if k == "drop_reasons":
                for kk, vv in v.items():
                    aggregate_drops[kk] = aggregate_drops.get(kk, 0) + vv
            elif k == "role_kept":
                for kk, vv in v.items():
                    aggregate_role_kept[kk] = aggregate_role_kept.get(kk, 0) + vv
            elif k == "role_dropped":
                for kk, vv in v.items():
                    aggregate_role_dropped[kk] = aggregate_role_dropped.get(kk, 0) + vv
            else:
                aggregate_stats[k] = aggregate_stats.get(k, 0) + v
        if records:
            aggregate_stats["runs_with_kept_steps"] += 1
        for r in records:
            total_kept += 1
            if args.dry_run:
                continue
            shard_writer.write(json.dumps(r, ensure_ascii=False) + "\n")
            shard_count += 1
            if shard_count >= args.shard_size:
                shard_writer.close()
                shard_idx += 1
                _open_shard()

    if not args.dry_run and shard_writer is not None:
        shard_writer.close()

    print()
    print("=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    for k, v in aggregate_stats.items():
        print(f"  {k}: {v}")
    print()
    print("PER-ROLE KEPT / DROPPED")
    all_roles = sorted(set(aggregate_role_kept) | set(aggregate_role_dropped))
    for r in all_roles:
        kept = aggregate_role_kept.get(r, 0)
        dropped = aggregate_role_dropped.get(r, 0)
        total = kept + dropped
        rate = (kept / total * 100) if total else 0
        print(f"  {r:14s}  kept={kept:6d}  dropped={dropped:6d}  keep_rate={rate:5.1f}%")
    print()
    print("DROP REASONS")
    for k, v in sorted(aggregate_drops.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    if aggregate_stats["interactions_total"]:
        keep_rate = aggregate_stats["interactions_kept"] / aggregate_stats["interactions_total"] * 100
        print(f"\ninteraction keep rate: {keep_rate:.1f}%")
    print(f"total examples kept: {total_kept}")
    if not args.dry_run:
        print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

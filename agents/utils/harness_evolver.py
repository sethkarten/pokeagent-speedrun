#!/usr/bin/env python3
"""
HarnessEvolver — Full harness evolution for the AutoEvolve scaffold.

Subsumes PromptOptimizer and adds subagent, skill, and memory evolution.
Runs periodically (every N steps, after a minimum warmup) to analyze
recent trajectories and improve all harness components mid-episode.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.prompts.paths import (
    AUTOEVOLVE_BASE_ORCHESTRATOR_POLICY_PATH,
    GAME_NAME,
    AUTOEVOLVE_SYSTEM_PROMPT_PATH,
    resolve_repo_path,
)
from agents.utils.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)

# Evolution cadence — adaptive: frequent early to bootstrap the harness,
# then backs off once capabilities stabilize. Same shape across game types
# now: every 10 steps for the first 200 (intense bootstrap), then every 25
# steps for the rest of the run. The wider early window gives the autoevolver
# many chances to build up the skill library and subagent registry while the
# agent is still figuring out the game; once the toolkit stabilizes, we
# back off to one evolution per ~25 steps to keep cost reasonable on long
# runs without losing the ability to react to new situations.
MIN_WARMUP_STEPS = 10
EARLY_PHASE_CUTOFF = 200
EARLY_FREQUENCY = 10
STABLE_FREQUENCY = 25

# Tools that exist for every scaffold (used to validate subagent tool lists)
# Tools available in the autoevolve scaffold (no navigate_to, no walkthrough, no wiki)
_COMMON_EVOLVER_TOOLS = frozenset({
    "complete_direct_objective",
    "get_game_state",
    "process_memory",
    "process_skill",
    "run_skill",
    "run_code",
    "process_subagent",
    "execute_custom_subagent",
    "process_trajectory_history",
    "replan_objectives",
})

def _slugify_id(raw: str) -> str:
    """Normalize an LLM-supplied id into a stable, URL-safe slug.

    The LLM is prompted to provide descriptive ids like "open_door" or
    "battle_handler" but sometimes returns "Open Door!" or "Battle
    Handler v2". Lowercase, strip punctuation, collapse runs of underscores,
    cap length so the id stays usable as a tree-overview leaf.
    """
    import re as _re
    s = (raw or "").strip().lower()
    s = _re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = _re.sub(r"_+", "_", s).strip("_")
    return s[:48] or "unnamed"


_IS_BROWSER_GAME = os.environ.get("GAME_TYPE", "emerald").lower() == "browser"

if _IS_BROWSER_GAME:
    # Browser games use Playwright-driven input primitives instead of the
    # Pokemon press_buttons / navigate_to / get_map_data tools.
    _ALWAYS_AVAILABLE_TOOLS = _COMMON_EVOLVER_TOOLS | frozenset({
        "press_keys",
        "mouse_click",
        "double_click",
        "hold_key",
        "mouse_move",
        "mouse_drag",
        "key_down",
        "key_up",
        "wait_ms",
    })
else:
    _ALWAYS_AVAILABLE_TOOLS = _COMMON_EVOLVER_TOOLS | frozenset({
        "press_buttons",
        "get_map_data",
    })


# Per-game text inserted into the LLM prompts that drive skill / subagent
# evolution. These have to match what the agent's run_skill sandbox actually
# exposes (see ``BrowserGameAgent._execute_run_skill`` /
# ``PokeAgent._execute_run_skill``); otherwise the LLM emits code that crashes
# at exec() time.
_BROWSER_SKILL_API_BLOCK = """## Skill Code API (MUST follow this exactly)

Skill code runs as inline Python (NOT a function definition). It has access to:
- `tools['press_keys'](keys=['Space'], reasoning='...')` — press one or more keys in sequence
- `tools['mouse_click'](x=480, y=300, reasoning='...')` — click the canvas at (x, y)
- `tools['double_click'](x=50, y=200, reasoning='...')` — double-click the canvas
- `tools['hold_key'](key='ArrowRight', duration_ms=500, reasoning='...')` — hold a key for N ms (single-step)
- `tools['mouse_move'](x=480, y=300, steps=8, reasoning='...')` — move cursor without clicking (hover, paddle-follow, mouse-look)
- `tools['mouse_drag'](x1=100, y1=100, x2=300, y2=200, steps=12, reasoning='...')` — press, drag, release (drag-to-aim, sliders, drawing)
- `tools['key_down'](key='ArrowRight', reasoning='...')` — press without releasing (held across steps; use for hold-to-flap, continuous run, mouse-look)
- `tools['key_up'](key='ArrowRight', reasoning='...')` — release a previously-held key (always pair with key_down)
- `tools['wait_ms'](duration_ms=500, reasoning='...')` — let game time pass (animations / fall timers / dialogue)
- `tools['get_game_state']()` — returns dict with `screenshot_base64`, `page_text`, `game_info` (canvas dims, mouse_x/y, held_keys, last action), and `success`
- `tools['process_memory'](edits=[...])` — read/write the agent's memory store
- `args` — dict of arguments passed by the caller (e.g. `args['x']`, `args['target']`)
- `result` — set this variable to return data
- Libraries: `collections`, `heapq`, `numpy`/`np`, `json`, `re`, `math`, `random`, `time`, `base64`
- Image processing: `Image`/`PIL` (Pillow), `ImageDraw`, `ImageFilter`, `cv2` (OpenCV — may be `None`), `BytesIO`, plus `decode_screenshot(b64_str)` helper that returns a PIL Image directly from `tools['get_game_state']()['screenshot_base64']`. Use these for pixel scans, template matching, and contour detection to find UI elements visually.

CRITICAL: Use `tools['function_name'](...)` syntax. Do NOT call bare `press_keys()` or `mouse_click()`.
Do NOT write `def skill_name():` — write inline code that reads `args` and sets `result`.

NEVER reference Pokemon-only tools like `press_buttons`, `get_map_data`, `navigate_to`, or fields like `player_position`/`player_coords`/`grid`/`warps` — they do not exist in the browser sandbox and the skill will crash at exec() time."""

_POKEMON_SKILL_API_BLOCK = """## Skill Code API (MUST follow this exactly)

Skill code runs as inline Python (NOT a function definition). It has access to:
- `tools['press_buttons'](buttons=['UP'], reasoning='...')` — press buttons (waits for emulator)
- `tools['get_game_state']()` — returns dict with `player_position`, `location`, `state_text`
- `tools['get_map_data']()` — returns dict with `grid` (list of strings), `player`, `dimensions`, `warps`, `objects`
- `args` — dict of arguments passed by the caller (e.g. `args['x']`, `args['y']`)
- `result` — set this variable to return data
- Libraries: `collections`, `heapq`, `numpy`/`np`, `json`, `re`, `math`, `random`

CRITICAL: Use `tools['function_name'](...)` syntax. Do NOT call bare `get_game_state()` or `press_buttons()`.
Do NOT write `def skill_name():` — write inline code that reads `args` and sets `result`."""

SKILL_API_BLOCK = _BROWSER_SKILL_API_BLOCK if _IS_BROWSER_GAME else _POKEMON_SKILL_API_BLOCK

# Hint shown alongside an UNDERPERFORMING skill the LLM is asked to rewrite.
# Pokemon hint pushes toward grid pathfinding; browser hint pushes toward
# pixel-based UI calibration since browser games have no map data.
_BROWSER_REWRITE_HINT = (
    "`tools['get_game_state']()` returns the current screenshot as base64 "
    "and the canvas dimensions, so the rewritten code can decode it with PIL "
    "and inspect pixel regions to find UI elements before clicking. All standard "
    "libraries (collections, heapq, numpy, etc) are available."
)
_POKEMON_REWRITE_HINT = (
    "`tools['get_map_data']()` which returns structured grid/position/warp data, "
    "and all standard libraries (collections, heapq, numpy, etc)."
)
SKILL_REWRITE_HINT = _BROWSER_REWRITE_HINT if _IS_BROWSER_GAME else _POKEMON_REWRITE_HINT

# Patterns the skills evolver suggests when proposing rewrites. Browser games
# don't have grids/A*, they have UI calibration and timed input sequences.
_BROWSER_SKILL_PATTERNS = """   - **UI calibration**: Decode `tools['get_game_state']()['screenshot_base64']` with PIL and scan for distinctive RGB regions to locate buttons/icons before clicking
   - **Timed input sequences**: Chain `press_keys` / `mouse_click` calls with `time.sleep` between them to perform combo moves
   - **Wait-and-check loops**: Poll `get_game_state` until `page_text` or pixel state matches an expected condition, then act
   - **Coordinate templates**: Parameterise click targets via `args` so one skill can open any of N similar icons"""

_POKEMON_SKILL_PATTERNS = """   - **Pathfinding**: BFS (collections.deque), A* (heapq) on `tools['get_map_data']()['grid']`
   - **State machines**: Track game mode transitions and act accordingly
   - **Decision logic**: Evaluate options (damage, resources)
   - **Parsing**: Extract info from game state text (regex, string splitting)"""

SKILL_PATTERNS_BLOCK = _BROWSER_SKILL_PATTERNS if _IS_BROWSER_GAME else _POKEMON_SKILL_PATTERNS

# Example tool list shown in the subagent recommendation JSON template — must
# match the actual tools available so the LLM emits valid recommendations.
SUBAGENT_EXAMPLE_TOOLS = (
    '["press_keys", "mouse_click", "key_down", "key_up", "wait_ms", "get_game_state"]'
    if _IS_BROWSER_GAME
    else '["press_buttons", "navigate_to", ...]'
)


class HarnessEvolver:
    """Evolves all harness components: prompt, subagents, skills, memory."""

    def __init__(
        self,
        vlm,
        run_data_manager,
        base_prompt_path: str = AUTOEVOLVE_BASE_ORCHESTRATOR_POLICY_PATH,
        system_prompt_path: str = AUTOEVOLVE_SYSTEM_PROMPT_PATH,
    ):
        # Compose the existing PromptOptimizer for prompt-level evolution
        self.prompt_optimizer = PromptOptimizer(
            vlm, run_data_manager, base_prompt_path, system_prompt_path
        )
        # Reuse the text-only VLM created by the optimizer
        self.text_vlm = self.prompt_optimizer.vlm
        self.run_manager = run_data_manager

        self.generation = 0
        self.evolution_log: List[Dict[str, Any]] = []
        # Track previous evolution results for before/after comparison
        self._prev_skill_stats: Dict[str, Dict[str, int]] = {}
        self._prev_changes_summary: str = ""

        logger.info("HarnessEvolver initialized (warmup=%d steps)", MIN_WARMUP_STEPS)

    # ------------------------------------------------------------------
    # Store accessors (lazy — imported only when evolution actually runs)
    # ------------------------------------------------------------------

    def _get_memory_store(self):
        from utils.stores.memory import get_memory_store
        return get_memory_store()

    def _get_skill_store(self):
        from utils.stores.skills import get_skill_store
        return get_skill_store()

    def _get_subagent_store(self):
        from utils.stores.subagents import get_subagent_store
        return get_subagent_store()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_evolve(self, current_step: int, frequency: int) -> bool:
        """Return True if evolution should fire at this step.

        Uses an adaptive frequency that depends on game type — see the
        module-level MIN_WARMUP_STEPS / EARLY_PHASE_CUTOFF / EARLY_FREQUENCY
        / STABLE_FREQUENCY constants. Browser games evolve more aggressively
        (every 10 steps for the first 100, then every 25) since they tend
        to be shorter and need tighter feedback. Pokemon games keep the
        original 25/200/100 schedule.
        The caller's ``frequency`` arg is ignored in favor of the adaptive schedule.
        """
        if current_step < MIN_WARMUP_STEPS or current_step <= 0:
            return False
        effective_freq = EARLY_FREQUENCY if current_step <= EARLY_PHASE_CUTOFF else STABLE_FREQUENCY
        return current_step % effective_freq == 0

    def get_current_prompt(self) -> str:
        """Delegate to the inner PromptOptimizer."""
        return self.prompt_optimizer.get_current_prompt()

    def _compute_skill_stats(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Compute per-skill call/stuck stats from trajectories."""
        stats: Dict[str, Dict[str, int]] = {}
        for traj in trajectories:
            action = traj.get("action", {})
            if not isinstance(action, dict):
                continue
            for tc in action.get("tool_calls", []):
                if tc.get("name") != "run_skill":
                    continue
                sid = tc.get("args", {}).get("skill_id", "")
                if not sid:
                    continue
                if sid not in stats:
                    stats[sid] = {"calls": 0, "stuck": 0}
                stats[sid]["calls"] += 1
                pre = traj.get("pre_state", {}).get("player_coords")
                post = traj.get("post_state", {}).get("player_coords", pre)
                if pre == post:
                    stats[sid]["stuck"] += 1
        return stats

    def _auto_revert_degraded_skills(self, current_stats: Dict[str, Dict[str, int]]) -> List[str]:
        """Auto-revert skills that got worse after the last evolution.

        Uses mutation_history to restore the previous code version.
        Returns list of reverted skill IDs.
        """
        if not self._prev_skill_stats:
            return []

        reverted = []
        store = self._get_skill_store()

        for sid, cur in current_stats.items():
            prev = self._prev_skill_stats.get(sid)
            if not prev or prev["calls"] < 3:
                continue
            if cur["calls"] < 3:
                continue

            prev_stuck_pct = 100 * prev["stuck"] // prev["calls"]
            cur_stuck_pct = 100 * cur["stuck"] // cur["calls"]

            if cur_stuck_pct - prev_stuck_pct > 15:
                # Skill got significantly worse — try to revert code
                entry = store.get(sid)
                if not entry or not getattr(entry, "mutation_history", []):
                    continue

                # Find the last mutation that changed 'code'
                for mut in reversed(entry.mutation_history):
                    old_code = mut.get("fields", {}).get("code", {}).get("old")
                    if old_code is not None:
                        logger.info(
                            "Auto-reverting skill %s: stuck %d%% -> %d%%, restoring previous code (%d chars)",
                            sid, prev_stuck_pct, cur_stuck_pct, len(old_code),
                        )
                        store.update(sid, code=old_code, effectiveness="medium")
                        reverted.append(sid)
                        break

        return reverted

    def _build_changes_feedback(self, current_stats: Dict[str, Dict[str, int]]) -> str:
        """Compare current skill performance to previous evolution's stats."""
        if not self._prev_skill_stats:
            return ""
        lines = ["## Impact of Previous Evolution Changes"]
        for sid, cur in current_stats.items():
            prev = self._prev_skill_stats.get(sid)
            if not prev or prev["calls"] == 0:
                continue
            prev_stuck_pct = 100 * prev["stuck"] // prev["calls"]
            cur_stuck_pct = 100 * cur["stuck"] // cur["calls"] if cur["calls"] > 0 else 0
            delta = cur_stuck_pct - prev_stuck_pct
            if delta > 10:
                lines.append(f"- **{sid} GOT WORSE**: stuck rate {prev_stuck_pct}% -> {cur_stuck_pct}% (+{delta}pp). Code was auto-reverted to previous version.")
            elif delta < -10:
                lines.append(f"- **{sid} IMPROVED**: stuck rate {prev_stuck_pct}% -> {cur_stuck_pct}% ({delta}pp). Keep this approach.")
            else:
                lines.append(f"- {sid}: stuck rate {prev_stuck_pct}% -> {cur_stuck_pct}% (no significant change)")
        if self._prev_changes_summary:
            lines.append(f"\nPrevious changes were: {self._prev_changes_summary}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def evolve(self, current_step: int, num_trajectory_steps: int = 50) -> Dict[str, Any]:
        """Run all evolution passes and return a summary.

        Each pass is independent — if one fails, the others still run and
        the evolution log is always saved.
        """
        logger.info(
            "=== HarnessEvolver generation %d at step %d ===",
            self.generation, current_step,
        )

        trajectories = self.prompt_optimizer.get_recent_trajectories(num_trajectory_steps)
        if not trajectories:
            logger.warning("No trajectories — skipping evolution")
            return {"skipped": True, "reason": "no_trajectories"}

        # Compute current skill stats for before/after comparison
        current_stats = self._compute_skill_stats(trajectories)

        # Auto-revert skills that got worse after the last evolution
        reverted = self._auto_revert_degraded_skills(current_stats)

        results: Dict[str, Any] = {}
        if reverted:
            results["reverted_skills"] = reverted

        # Each pass is wrapped independently so failures don't block others
        for name, fn in [
            ("prompt", lambda: self._evolve_prompt(current_step, num_trajectory_steps)),
            ("subagents", lambda: self._evolve_subagents(trajectories, current_step)),
            ("skills", lambda: self._evolve_skills(trajectories, current_step)),
            ("memory", lambda: self._evolve_memory(trajectories, current_step)),
        ]:
            try:
                results[name] = fn()
            except Exception as e:
                logger.error("Evolution pass '%s' failed: %s", name, e, exc_info=True)
                results[name] = {"error": str(e)}

        # Save stats for next evolution's before/after comparison
        self._prev_skill_stats = current_stats
        changes = []
        for pass_name in ["subagents", "skills", "memory"]:
            data = results.get(pass_name, {})
            if isinstance(data, dict):
                for k in ["created", "updated", "retired"]:
                    if data.get(k):
                        changes.append(f"{pass_name}_{k}={data[k]}")
        self._prev_changes_summary = ", ".join(changes) if changes else "prompt only"

        self.generation += 1
        self._save_evolution_log(current_step, results)
        return results

    # ------------------------------------------------------------------
    # Evolution passes
    # ------------------------------------------------------------------

    def _evolve_prompt(self, current_step: int, num_trajectory_steps: int) -> Dict[str, Any]:
        """Evolve the orchestrator base prompt via PromptOptimizer."""
        try:
            new_prompt = self.prompt_optimizer.optimize_prompt(
                current_step=current_step,
                num_trajectory_steps=num_trajectory_steps,
            )
            return {"rewritten": True, "length": len(new_prompt)}
        except Exception as e:
            logger.error("Prompt evolution failed: %s", e, exc_info=True)
            return {"rewritten": False, "error": str(e)}

    def _extract_tool_failures(self, trajectories: List[Dict[str, Any]]) -> str:
        """Extract tool failure patterns from trajectories for evolution analysis."""
        failures = []
        for traj in trajectories:
            action = traj.get("action", {})
            if not isinstance(action, dict):
                continue
            for tc in action.get("tool_calls", []):
                result = tc.get("result", "")
                result_str = str(result) if result else ""
                if not result_str:
                    continue
                # Check for failure indicators
                is_failure = False
                if isinstance(result, dict):
                    is_failure = result.get("success") is False or "error" in result
                elif '"success": false' in result_str.lower() or '"error"' in result_str.lower():
                    is_failure = True
                if is_failure:
                    failures.append({
                        "step": traj.get("step"),
                        "tool": tc.get("name"),
                        "args": str(tc.get("args", {}))[:200],
                        "error": result_str[:300],
                    })
        if not failures:
            return ""
        lines = ["## Tool Failures Detected"]
        for f in failures:
            lines.append(f"- Step {f['step']}: `{f['tool']}` args={f['args']} => {f['error']}")
        return "\n".join(lines)

    def _evolve_subagents(
        self, trajectories: List[Dict[str, Any]], current_step: int
    ) -> Dict[str, Any]:
        """Analyze trajectories and create/update/retire subagents."""
        store = self._get_subagent_store()
        registry_overview = store.get_tree_overview()

        trajectory_summary = self.prompt_optimizer._format_trajectories_for_analysis(
            trajectories
        )
        tool_failures = self._extract_tool_failures(trajectories)

        prompt = f"""You are a harness evolution system for an AI agent playing {GAME_NAME}.
The agent has NO walkthrough or wiki access — it learns entirely through gameplay.

Your job: analyze recent trajectories and recommend changes to the agent's subagent library.

## Current Subagent Registry
{registry_overview}

## Recent Trajectories (last {len(trajectories)} steps)
{trajectory_summary}

{tool_failures}

## Analysis Tasks

1. **Identify missing subagents**: Look for repeated multi-step patterns the agent does manually that would benefit from a dedicated subagent. Common needs:
   - Handlers for recurring game modes (e.g., combat, puzzles, menus)
   - Planning/reflection subagent (if agent gets stuck repeatedly)
   - Navigation specialist (if agent struggles with complex routes)

2. **Evaluate existing subagents**: If any custom subagents were used, check if they hit safety caps, failed repeatedly, or could be improved.

3. **Retire ineffective subagents**: If a subagent was used multiple times with poor results, recommend removal.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{{
  "analysis": "Brief summary of what you observed",
  "create": [
    {{
      "id": "descriptive_snake_case_id",
      "name": "string",
      "description": "string",
      "handler_type": "one_step or looping",
      "max_turns": 25,
      "available_tools": {SUBAGENT_EXAMPLE_TOOLS},
      "system_instructions": "Detailed instructions for the subagent (what it does, how to behave)",
      "directive": "Default task directive",
      "return_condition": "Condition to signal completion back to orchestrator"
    }}
  ],
  "update": [
    {{
      "id": "existing_id",
      "system_instructions": "improved instructions",
      "directive": "improved directive"
    }}
  ],
  "retire": ["existing_id"]
}}

**id MUST be a short descriptive snake_case slug** (e.g. `door_opener`, `combat_handler`, `popup_dismisser`) — NOT a generic placeholder. The id appears in every prompt and the orchestrator picks subagents by matching its current intent against ids and descriptions, so non-descriptive ids make the subagent registry unusable.

Only include sections with actual recommendations. Empty arrays are fine.
Available tools the subagent can use: {sorted(_ALWAYS_AVAILABLE_TOOLS)}
"""

        try:
            response = self.text_vlm.get_text_query(prompt, "HarnessEvolver_Subagents")
            recommendations = self._parse_json_response(response)
            if recommendations is None:
                return {"error": "failed_to_parse_response"}

            result = {"created": [], "updated": [], "retired": [], "analysis": recommendations.get("analysis", "")}

            # Create new subagents
            for spec in recommendations.get("create", []):
                try:
                    # Validate tool names
                    tools = spec.get("available_tools", [])
                    valid_tools = [t for t in tools if t in _ALWAYS_AVAILABLE_TOOLS]
                    if not valid_tools:
                        # Pick a sensible default per game type
                        valid_tools = (
                            ["press_keys"]
                            if "press_keys" in _ALWAYS_AVAILABLE_TOOLS
                            else ["press_buttons"]
                        )

                    # BaseStore.add returns the entry ID (string), not the
                    # entry object — look up the entry afterwards.
                    add_kwargs = dict(
                        path=f"evolved/{spec.get('name', 'unnamed').lower().replace(' ', '_')}",
                        name=spec.get("name", "Unnamed"),
                        description=spec.get("description", ""),
                        handler_type=spec.get("handler_type", "looping"),
                        max_turns=min(spec.get("max_turns", 25), 50),
                        available_tools=valid_tools,
                        system_instructions=spec.get("system_instructions", "")[:12000],
                        directive=spec.get("directive", "")[:12000],
                        return_condition=spec.get("return_condition", "Task completed"),
                        importance=3,
                        source="evolved",
                    )
                    # Forward LLM-supplied descriptive id if present so the
                    # registry isn't all sa_NNNN — the LLM was previously
                    # working around this by stuffing the auto id into the
                    # name field (e.g. "sa_0008: UI_Parser").
                    custom_id = spec.get("id")
                    if isinstance(custom_id, str) and custom_id.strip():
                        add_kwargs["id"] = _slugify_id(custom_id)
                    entry_id = store.add(**add_kwargs)
                    entry = store.get(entry_id)
                    result["created"].append(entry_id)
                    logger.info(
                        "Created evolved subagent: %s (%s)",
                        entry_id,
                        getattr(entry, "name", spec.get("name", "")),
                    )
                except Exception as e:
                    logger.error("Failed to create subagent %s: %s", spec.get("name"), e)

            # Update existing subagents
            for upd in recommendations.get("update", []):
                sid = upd.get("id")
                if not sid:
                    continue
                try:
                    fields = {k: v for k, v in upd.items() if k != "id" and v}
                    if fields:
                        store.update(sid, **fields)
                        result["updated"].append(sid)
                        logger.info("Updated evolved subagent: %s", sid)
                except Exception as e:
                    logger.error("Failed to update subagent %s: %s", sid, e)

            # Retire subagents
            for sid in recommendations.get("retire", []):
                try:
                    store.remove(sid)
                    result["retired"].append(sid)
                    logger.info("Retired evolved subagent: %s", sid)
                except Exception as e:
                    logger.error("Failed to retire subagent %s: %s", sid, e)

            return result

        except Exception as e:
            logger.error("Subagent evolution failed: %s", e, exc_info=True)
            return {"error": str(e)}

    def _evolve_skills(
        self, trajectories: List[Dict[str, Any]], current_step: int
    ) -> Dict[str, Any]:
        """Extract successful patterns as skills and update effectiveness."""
        store = self._get_skill_store()
        skill_overview = store.get_tree_overview()

        trajectory_summary = self.prompt_optimizer._format_trajectories_for_analysis(
            trajectories
        )
        tool_failures = self._extract_tool_failures(trajectories)

        # Analyze run_skill performance to find underperforming skills
        skill_stats: Dict[str, Dict[str, int]] = {}
        for traj in trajectories:
            action = traj.get("action", {})
            if not isinstance(action, dict):
                continue
            for tc in action.get("tool_calls", []):
                if tc.get("name") != "run_skill":
                    continue
                sid = tc.get("args", {}).get("skill_id", "")
                if not sid:
                    continue
                if sid not in skill_stats:
                    skill_stats[sid] = {"calls": 0, "stuck": 0}
                skill_stats[sid]["calls"] += 1
                # Check if position changed (stuck detection)
                pre = traj.get("pre_state", {}).get("player_coords")
                post = traj.get("post_state", {}).get("player_coords", pre)
                if pre == post:
                    skill_stats[sid]["stuck"] += 1

        underperforming_skills = ""
        for sid, stats in skill_stats.items():
            if stats["calls"] >= 3 and stats["stuck"] / stats["calls"] >= 0.5:
                entry = store.get(sid)
                if entry and getattr(entry, "code", ""):
                    underperforming_skills += (
                        f"\n### UNDERPERFORMING: [{sid}] {getattr(entry, 'name', sid)}\n"
                        f"Stats: {stats['calls']} calls, {stats['stuck']} stuck ({100*stats['stuck']//stats['calls']}% failure)\n"
                        f"Current code:\n```python\n{entry.code}\n```\n"
                        f"Rewrite this skill with a better algorithm. The code has access to "
                        f"{SKILL_REWRITE_HINT}\n"
                    )

        # Detect antipattern: using run_code repeatedly instead of saving as skills
        run_code_count = sum(
            1 for t in trajectories
            for tc in (t.get("action", {}).get("tool_calls", []) if isinstance(t.get("action", {}), dict) else [])
            if tc.get("name") == "run_code"
        )
        run_skill_count = sum(
            1 for t in trajectories
            for tc in (t.get("action", {}).get("tool_calls", []) if isinstance(t.get("action", {}), dict) else [])
            if tc.get("name") == "run_skill"
        )
        antipattern_warning = ""
        if run_code_count >= 3 and run_skill_count == 0:
            antipattern_warning = f"""
## CRITICAL ANTIPATTERN DETECTED
The agent called run_code {run_code_count} times but run_skill 0 times in the last {len(trajectories)} steps. This means the agent is writing disposable scripts instead of saving reusable skills. You MUST create executable skills from the patterns in these run_code calls. Extract the common code, save it as a skill with process_skill, so the agent can call run_skill instead.
"""

        changes_feedback = self._build_changes_feedback(skill_stats)

        prompt = f"""You are a harness evolution system analyzing an AI agent's recent gameplay in {GAME_NAME}.

Your job: identify reusable behavioral patterns (skills) from successful actions and evaluate existing skills.

{changes_feedback}

## Current Skill Library
{skill_overview}

## Recent Trajectories (last {len(trajectories)} steps)
{trajectory_summary}

{tool_failures}
{antipattern_warning}
{underperforming_skills}

{SKILL_API_BLOCK}

## Analysis Tasks

1. **Rewrite underperforming executable skills**: If any skills are shown as UNDERPERFORMING above, you MUST rewrite their code. Include the FULL replacement `code` in your update. Common patterns:
{SKILL_PATTERNS_BLOCK}

2. **Fix broken skills**: If skills error or use wrong API fields, fix the code.

3. **Record new skills**: Look for successful action sequences. Include `code` for skills that automate multi-step actions. Make skills game-specific and useful. ALWAYS include a `code` field when proposing an executable skill — note-only skills (no code) cannot be invoked via run_skill.

4. **Update effectiveness ratings** based on trajectory outcomes.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{{
  "analysis": "Brief summary",
  "add": [
    {{
      "id": "descriptive_snake_case_id",
      "name": "string",
      "path": "category/subcategory",
      "description": "What the skill does",
      "code": "optional Python code for executable skills",
      "effectiveness": "low|medium|high",
      "importance": 3
    }}
  ],
  "update": [
    {{
      "id": "existing_id",
      "effectiveness": "low|medium|high",
      "description": "optional updated description",
      "code": "optional: FULL replacement Python code if improving an executable skill"
    }}
  ]
}}

**id MUST be a short descriptive snake_case slug** (e.g. `pathfind_bfs`, `dismiss_popup`, `attack_combo`) — NOT a generic placeholder. The id appears in every prompt's SKILL LIBRARY and the orchestrator picks skills by matching its current intent against ids and descriptions, so non-descriptive ids make the library unusable.
"""

        try:
            response = self.text_vlm.get_text_query(prompt, "HarnessEvolver_Skills")
            recommendations = self._parse_json_response(response)
            if recommendations is None:
                return {"error": "failed_to_parse_response"}

            result = {"created": [], "updated": [], "analysis": recommendations.get("analysis", "")}

            for spec in recommendations.get("add", []):
                try:
                    add_kwargs = dict(
                        path=spec.get("path", "general"),
                        name=spec.get("name", "Unnamed Skill"),
                        description=spec.get("description", ""),
                        # Forward `code` so executable skills the LLM produces
                        # actually become invocable via run_skill instead of
                        # silently turning into note-only entries.
                        code=spec.get("code", ""),
                        effectiveness=spec.get("effectiveness", "medium"),
                        importance=spec.get("importance", 3),
                        source="evolved",
                    )
                    custom_id = spec.get("id")
                    if isinstance(custom_id, str) and custom_id.strip():
                        add_kwargs["id"] = _slugify_id(custom_id)
                    entry_id = store.add(**add_kwargs)
                    entry = store.get(entry_id)
                    has_code = bool(getattr(entry, "code", ""))
                    result["created"].append(entry_id)
                    logger.info(
                        "Created evolved skill: %s (%s)%s",
                        entry_id,
                        getattr(entry, "name", spec.get("name", "")),
                        " [executable]" if has_code else " [note-only]",
                    )
                except Exception as e:
                    logger.error("Failed to create skill %s: %s", spec.get("name"), e)

            for upd in recommendations.get("update", []):
                sid = upd.get("id")
                if not sid:
                    continue
                try:
                    fields = {k: v for k, v in upd.items() if k != "id" and v}
                    if fields:
                        store.update(sid, **fields)
                        result["updated"].append(sid)
                except Exception as e:
                    logger.error("Failed to update skill %s: %s", sid, e)

            return result

        except Exception as e:
            logger.error("Skill evolution failed: %s", e, exc_info=True)
            return {"error": str(e)}

    def _evolve_memory(
        self, trajectories: List[Dict[str, Any]], current_step: int
    ) -> Dict[str, Any]:
        """Lightweight memory curation: fill gaps and rebalance importance."""
        store = self._get_memory_store()
        memory_overview = store.get_tree_overview()

        # Collect locations and events from trajectories for gap analysis
        locations_visited = set()
        for traj in trajectories:
            pre = traj.get("pre_state", {})
            loc = pre.get("location") or traj.get("location")
            if loc:
                locations_visited.add(loc)

        trajectory_summary = self.prompt_optimizer._format_trajectories_for_analysis(
            trajectories[-20:]  # Use last 20 for memory curation (lighter)
        )

        prompt = f"""You are a memory curator for an AI agent playing {GAME_NAME}.

Your job: review the agent's memory store and recent trajectories, then recommend targeted improvements. Be conservative — the agent itself writes memory during gameplay. You fill gaps and clean up.

## Current Memory Overview
{memory_overview}

## Locations Visited Recently
{sorted(locations_visited)}

## Recent Trajectories (last {min(20, len(trajectories))} steps)
{trajectory_summary}

## Curation Tasks

1. **Fill knowledge gaps**: If the agent visited important locations or encountered key events but has no memory entry for them, recommend adding one.
2. **Update stale entries**: If memory content contradicts what trajectories show (e.g., team composition changed), recommend updates.
3. **Rebalance importance**: Entries about areas the agent has moved past can have importance lowered (1-2). Active area knowledge stays at 3-5.

Keep recommendations minimal (max 3-5 changes). Do NOT duplicate what the agent already stored.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{{
  "analysis": "Brief summary of memory state",
  "add": [
    {{
      "id": "descriptive_snake_case_id",
      "path": "category/subcategory",
      "title": "string",
      "content": "string",
      "importance": 3
    }}
  ],
  "update": [
    {{
      "id": "existing_id",
      "content": "optional updated content",
      "importance": 2
    }}
  ]
}}

**id MUST be a short descriptive snake_case slug** (e.g. `slime_attack_pattern`, `door_unlocks_at_lvl3`, `inventory_x_button`) — NOT a generic placeholder. The id appears in every prompt's MEMORY OVERVIEW and the orchestrator recalls memories by matching its current intent against ids and titles.
"""

        try:
            response = self.text_vlm.get_text_query(prompt, "HarnessEvolver_Memory")
            recommendations = self._parse_json_response(response)
            if recommendations is None:
                return {"error": "failed_to_parse_response"}

            result = {"created": [], "updated": [], "analysis": recommendations.get("analysis", "")}

            for spec in recommendations.get("add", []):
                try:
                    add_kwargs = dict(
                        path=spec.get("path", "general"),
                        title=spec.get("title", "Untitled"),
                        content=spec.get("content", ""),
                        importance=spec.get("importance", 3),
                        source="evolved",
                    )
                    custom_id = spec.get("id")
                    if isinstance(custom_id, str) and custom_id.strip():
                        add_kwargs["id"] = _slugify_id(custom_id)
                    entry_id = store.add(**add_kwargs)
                    entry = store.get(entry_id)
                    result["created"].append(entry_id)
                    logger.info(
                        "Created evolved memory: %s (%s)",
                        entry_id,
                        getattr(entry, "title", spec.get("title", "")),
                    )
                except Exception as e:
                    logger.error("Failed to create memory: %s", e)

            for upd in recommendations.get("update", []):
                mid = upd.get("id")
                if not mid:
                    continue
                try:
                    fields = {k: v for k, v in upd.items() if k != "id" and v is not None}
                    if fields:
                        store.update(mid, **fields)
                        result["updated"].append(mid)
                except Exception as e:
                    logger.error("Failed to update memory %s: %s", mid, e)

            return result

        except Exception as e:
            logger.error("Memory evolution failed: %s", e, exc_info=True)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from VLM response, handling markdown fences."""
        text = response.strip()
        # Strip markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.error("Failed to parse JSON from evolution response: %s", text[:200])
            return None

    def _save_evolution_log(self, current_step: int, results: Dict[str, Any]):
        """Persist evolution log entry to cache."""
        from utils.data_persistence.run_data_manager import get_cache_path

        entry = {
            "generation": self.generation,
            "step": current_step,
            "timestamp": datetime.now().isoformat(),
            "prompt_rewritten": results.get("prompt", {}).get("rewritten", False),
            "subagents_created": results.get("subagents", {}).get("created", []),
            "subagents_updated": results.get("subagents", {}).get("updated", []),
            "subagents_retired": results.get("subagents", {}).get("retired", []),
            "skills_created": results.get("skills", {}).get("created", []),
            "skills_updated": results.get("skills", {}).get("updated", []),
            "memory_created": results.get("memory", {}).get("created", []),
            "memory_updated": results.get("memory", {}).get("updated", []),
        }

        # Append store counts
        try:
            entry["store_counts"] = {
                "memory": len(self._get_memory_store()._entries),
                "skills": len(self._get_skill_store()._entries),
                "subagents": len(self._get_subagent_store()._entries),
            }
        except Exception:
            pass

        self.evolution_log.append(entry)

        try:
            log_file = get_cache_path("evolution_log.jsonl")
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info("Saved evolution log: generation=%d step=%d", self.generation, current_step)
        except Exception as e:
            logger.error("Failed to save evolution log: %s", e)


def create_harness_evolver(
    vlm,
    run_data_manager,
    base_prompt_path: str = AUTOEVOLVE_BASE_ORCHESTRATOR_POLICY_PATH,
    system_prompt_path: str = AUTOEVOLVE_SYSTEM_PROMPT_PATH,
) -> HarnessEvolver:
    """Factory function to create a HarnessEvolver instance."""
    return HarnessEvolver(vlm, run_data_manager, base_prompt_path, system_prompt_path)

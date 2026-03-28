#!/usr/bin/env python3
"""
HarnessEvolver — Full harness evolution for the AutoEvolve scaffold.

Subsumes PromptOptimizer and adds subagent, skill, and memory evolution.
Runs periodically (every N steps, after a minimum warmup) to analyze
recent trajectories and improve all harness components mid-episode.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.prompts.paths import (
    AUTOEVOLVE_BASE_SYSTEM_PROMPT_PATH,
    POKEAGENT_BASE_PROMPT_PATH,
    resolve_repo_path,
)
from agents.utils.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)

# Minimum steps before the first evolution fires.
MIN_WARMUP_STEPS = 25

# Evolution frequency adapts: frequent early (every 25 steps for first 200),
# then backs off (every 100 steps) once the harness stabilizes.
EARLY_PHASE_CUTOFF = 200
EARLY_FREQUENCY = 25
STABLE_FREQUENCY = 100

# Tools that exist for every scaffold (used to validate subagent tool lists)
# Tools available in the autoevolve scaffold (no navigate_to, no walkthrough, no wiki)
_ALWAYS_AVAILABLE_TOOLS = frozenset({
    "press_buttons",
    "complete_direct_objective",
    "get_game_state",
    "get_map_data",
    "process_memory",
    "process_skill",
    "run_skill",
    "run_code",
    "process_subagent",
    "execute_custom_subagent",
    "process_trajectory_history",
    "get_progress_summary",
    "replan_objectives",
})


class HarnessEvolver:
    """Evolves all harness components: prompt, subagents, skills, memory."""

    def __init__(
        self,
        vlm,
        run_data_manager,
        base_prompt_path: str = POKEAGENT_BASE_PROMPT_PATH,
        system_prompt_path: str = AUTOEVOLVE_BASE_SYSTEM_PROMPT_PATH,
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

        Uses adaptive frequency: evolve more often early (every 25 steps
        for the first 200 steps) to bootstrap the harness quickly, then
        back off to every 100 steps once capabilities stabilize.
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
                lines.append(f"- **{sid} GOT WORSE**: stuck rate {prev_stuck_pct}% -> {cur_stuck_pct}% (+{delta}pp). REVERT or fix the last code change.")
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

        results: Dict[str, Any] = {}

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

        prompt = f"""You are a harness evolution system for an AI agent playing Pokemon Emerald.
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
      "name": "string",
      "description": "string",
      "handler_type": "one_step or looping",
      "max_turns": 25,
      "available_tools": ["press_buttons", "navigate_to", ...],
      "system_instructions": "Detailed instructions for the subagent (what it does, how to behave)",
      "directive": "Default task directive",
      "return_condition": "Condition to signal completion back to orchestrator"
    }}
  ],
  "update": [
    {{
      "id": "sa_XXXX",
      "system_instructions": "improved instructions",
      "directive": "improved directive"
    }}
  ],
  "retire": ["sa_XXXX"]
}}

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
                        valid_tools = ["press_buttons"]

                    entry = store.add(
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
                    result["created"].append(entry.id)
                    logger.info("Created evolved subagent: %s (%s)", entry.id, entry.name)
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
                        f"`tools['get_map_data']()` which returns structured grid/position/warp data, "
                        f"and all standard libraries (collections, heapq, numpy, etc).\n"
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

        prompt = f"""You are a harness evolution system analyzing an AI agent's recent gameplay in Pokemon Emerald.

Your job: identify reusable behavioral patterns (skills) from successful actions and evaluate existing skills.

{changes_feedback}

## Current Skill Library
{skill_overview}

## Recent Trajectories (last {len(trajectories)} steps)
{trajectory_summary}

{tool_failures}
{antipattern_warning}
{underperforming_skills}

## Skill Code API (MUST follow this exactly)

Skill code runs as inline Python (NOT a function definition). It has access to:
- `tools['press_buttons'](buttons=['UP'], reasoning='...')` — press buttons (waits for emulator)
- `tools['get_game_state']()` — returns dict with `player_position`, `location`, `state_text`
- `tools['get_map_data']()` — returns dict with `grid` (list of strings), `player`, `dimensions`, `warps`, `objects`
- `args` — dict of arguments passed by the caller (e.g. `args['x']`, `args['y']`)
- `result` — set this variable to return data
- Libraries: `collections`, `heapq`, `numpy`/`np`, `json`, `re`, `math`, `random`

CRITICAL: Use `tools['function_name'](...)` syntax. Do NOT call bare `get_game_state()` or `press_buttons()`.
Do NOT write `def skill_name():` — write inline code that reads `args` and sets `result`.

## Analysis Tasks

1. **Rewrite underperforming executable skills**: If any skills are shown as UNDERPERFORMING above, you MUST rewrite their code. Include the FULL replacement `code` in your update. Common patterns:
   - **Pathfinding**: BFS (collections.deque), A* (heapq) on `tools['get_map_data']()['grid']`
   - **State machines**: Track game mode transitions and act accordingly
   - **Decision logic**: Evaluate options (damage, resources)
   - **Parsing**: Extract info from game state text (regex, string splitting)

2. **Fix broken skills**: If skills error or use wrong API fields, fix the code.

3. **Record new skills**: Look for successful action sequences. Include `code` for skills that automate multi-step actions. Make skills game-specific and useful.

4. **Update effectiveness ratings** based on trajectory outcomes.

## Output Format

Respond with ONLY a JSON object (no markdown fences):
{{
  "analysis": "Brief summary",
  "add": [
    {{
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
      "id": "skill_XXXX",
      "effectiveness": "low|medium|high",
      "description": "optional updated description",
      "code": "optional: FULL replacement Python code if improving an executable skill"
    }}
  ]
}}
"""

        try:
            response = self.text_vlm.get_text_query(prompt, "HarnessEvolver_Skills")
            recommendations = self._parse_json_response(response)
            if recommendations is None:
                return {"error": "failed_to_parse_response"}

            result = {"created": [], "updated": [], "analysis": recommendations.get("analysis", "")}

            for spec in recommendations.get("add", []):
                try:
                    entry = store.add(
                        path=spec.get("path", "general"),
                        name=spec.get("name", "Unnamed Skill"),
                        description=spec.get("description", ""),
                        effectiveness=spec.get("effectiveness", "medium"),
                        importance=spec.get("importance", 3),
                        source="evolved",
                    )
                    result["created"].append(entry.id)
                    logger.info("Created evolved skill: %s (%s)", entry.id, entry.name)
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

        prompt = f"""You are a memory curator for an AI agent playing Pokemon Emerald.

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
      "path": "category/subcategory",
      "title": "string",
      "content": "string",
      "importance": 3
    }}
  ],
  "update": [
    {{
      "id": "mem_XXXX",
      "content": "optional updated content",
      "importance": 2
    }}
  ]
}}
"""

        try:
            response = self.text_vlm.get_text_query(prompt, "HarnessEvolver_Memory")
            recommendations = self._parse_json_response(response)
            if recommendations is None:
                return {"error": "failed_to_parse_response"}

            result = {"created": [], "updated": [], "analysis": recommendations.get("analysis", "")}

            for spec in recommendations.get("add", []):
                try:
                    entry = store.add(
                        path=spec.get("path", "general"),
                        title=spec.get("title", "Untitled"),
                        content=spec.get("content", ""),
                        importance=spec.get("importance", 3),
                        source="evolved",
                    )
                    result["created"].append(entry.id)
                    logger.info("Created evolved memory: %s (%s)", entry.id, entry.title)
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
    base_prompt_path: str = POKEAGENT_BASE_PROMPT_PATH,
    system_prompt_path: str = AUTOEVOLVE_BASE_SYSTEM_PROMPT_PATH,
) -> HarnessEvolver:
    """Factory function to create a HarnessEvolver instance."""
    return HarnessEvolver(vlm, run_data_manager, base_prompt_path, system_prompt_path)

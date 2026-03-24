"""Prompt construction for the local reflection subagent."""

from __future__ import annotations

from typing import Any, Dict, List


def _format_objective_summary(objective_state: Dict[str, Any]) -> str:
    if objective_state.get("mode") == "categorized":
        categories = objective_state.get("categories", {})
        parts: List[str] = []
        story = categories.get("story", {})
        story_obj = story.get("current_objective")
        parts.append(
            f"Story ({story.get('index', 0) + 1}/{story.get('total', 0)}): "
            f"{story_obj.get('description') if isinstance(story_obj, dict) else 'None'}"
        )

        battling = categories.get("battling", {})
        battling_group = battling.get("current_objective") or []
        if battling_group:
            parts.append(
                "Battling group: "
                + "; ".join(obj.get("description", "Unknown") for obj in battling_group if isinstance(obj, dict))
            )
        else:
            parts.append("Battling group: None")

        dynamics = categories.get("dynamics", {})
        dynamics_obj = dynamics.get("current_objective")
        parts.append(
            f"Dynamics ({dynamics.get('index', 0) + 1}/{dynamics.get('total', 0)}): "
            f"{dynamics_obj.get('description') if isinstance(dynamics_obj, dict) else 'None'}"
        )
        return "\n".join(parts)

    objective = objective_state.get("objective")
    if isinstance(objective, dict):
        desc = objective.get("description", "None")
        hint = objective.get("navigation_hint")
        return f"{desc}\nHint: {hint}" if hint else desc
    return str(objective or "None")


def _format_progress(progress: Dict[str, Any], objective_state: Dict[str, Any]) -> str:
    if objective_state.get("mode") == "categorized":
        categories = objective_state.get("categories", {})
        return "\n".join(
            [
                f"Milestones completed: {progress.get('total_milestones_completed', 0)}",
                (
                    f"Story objectives: {categories.get('story', {}).get('completed', 0)}"
                    f"/{categories.get('story', {}).get('total', 0)}"
                ),
                (
                    f"Battling objectives: {categories.get('battling', {}).get('completed', 0)}"
                    f"/{categories.get('battling', {}).get('total', 0)}"
                ),
                (
                    f"Dynamics objectives: {categories.get('dynamics', {}).get('completed', 0)}"
                    f"/{categories.get('dynamics', {}).get('total', 0)}"
                ),
            ]
        )

    direct = progress.get("direct_objectives", {})
    return "\n".join(
        [
            f"Milestones completed: {progress.get('total_milestones_completed', 0)}",
            (
                f"Objectives completed in current sequence: "
                f"{direct.get('objectives_completed_in_current_sequence', 0)}"
                f"/{direct.get('total_in_current_sequence', 0)}"
            ),
        ]
    )


def build_reflect_prompt(*, situation: str, context: Dict[str, Any], last_n_steps: int) -> str:
    """Build the prompt for the local reflection subagent."""
    current_state = context.get("current_state", {})
    objective_state = context.get("objective_state", {})
    memory_summary = context.get("memory_summary") or "No memories recorded yet."

    return f"""You are the reflection subagent for a Pokemon Emerald speedrun agent.
Your job is to diagnose whether the orchestrator is stuck, pursuing the wrong objective, or missing a simpler next move.

The current screenshot is attached when available. Use it as CURRENT visual evidence only.
Do not assume historical screenshots exist. Historical context comes only from the text trajectory window below.

AGENT CONCERN:
{situation}

CURRENT STATE:
Location: {current_state.get('location')}
Coordinates: ({current_state.get('coordinates', {}).get('x')}, {current_state.get('coordinates', {}).get('y')})
{current_state.get('state_text', '')}

CURRENT OBJECTIVE STATE:
Mode: {objective_state.get('mode')}
{_format_objective_summary(objective_state)}
Status: {objective_state.get('status')}

PROGRESS SUMMARY:
{_format_progress(context.get('progress', {}), objective_state)}

LONG-TERM MEMORY:
{memory_summary}

RECENT TRAJECTORY WINDOW (last {last_n_steps} steps):
{context.get('trajectory_summary', 'No prior trajectories recorded.')}

GROUND TRUTH PRIORITY:
1. Current game state and screenshot
2. Long-term memory
3. Trajectory window
4. Current objectives

TASKS:
1. Identify whether the agent is looping, misreading state, or following the wrong objective.
2. Check whether the current objective still matches the evidence.
3. Point out the most important mistake or blind spot.
4. Recommend the next concrete action or verification step.

OUTPUT FORMAT:
**ASSESSMENT**:
[2-3 sentences]

**ISSUES**:
- [Issue 1]
- [Issue 2]

**RECOMMENDATIONS**:
1. [Concrete next action]
2. [If needed, what to verify or change]

**SHOULD_REALIGN**: [YES or NO]

Be decisive, concise, and evidence-driven."""

"""Prompt helpers for the gym-puzzle local subagent."""

from __future__ import annotations

import re
from typing import Any, Dict

from .utils.puzzle_solver import GYM_PUZZLES


def resolve_gym_name(arguments: Dict[str, Any], state_text: str) -> str:
    gym_name = arguments.get("gym_name")
    if gym_name:
        return str(gym_name)

    location_match = re.search(r"Current Location: ([^\n]+)", state_text or "")
    if location_match:
        return location_match.group(1).strip()
    return "Unknown"


def get_gym_puzzle_info(gym_name: str) -> Dict[str, Any]:
    return GYM_PUZZLES.get(
        gym_name,
        {
            "type": "unknown",
            "description": "Unknown gym - no specific puzzle guidance available",
            "strategy": "Navigate through the gym and defeat trainers to reach the gym leader.",
        },
    )


def build_gym_puzzle_prompt(
    *,
    gym_name: str,
    gym_info: Dict[str, Any],
    state_text: str,
    action_history: str,
    function_results: str,
) -> str:
    return f"""You are analyzing a Pokemon Emerald gym puzzle to help the agent solve it.

GYM: {gym_name}
TYPE: {gym_info.get("type", "unknown")}
DESCRIPTION: {gym_info.get("description", "")}

GENERAL STRATEGY:
{gym_info.get("strategy", "")}

RECENT ACTION HISTORY:
{action_history}

{function_results}

CURRENT GAME STATE:
{state_text}

Provide your analysis in this format:

**PUZZLE ANALYSIS**:
[Explain how this specific puzzle works based on the map and your current position]

**WHAT WE'VE TRIED**:
[Based on the action history above, summarize what approaches have been attempted and what worked/didn't work]

**SPECIFIC SOLUTION STEPS**:
1. [First concrete action with coordinates if applicable]
2. [Second action]
3. [Continue...]

**NAVIGATION TIPS**:
[Any important details about tile types, warps, or obstacles to watch for]

**IMPORTANT**:
- Look at the porymap ground truth map in the game state. Tiles marked '#' are walls, '.' are walkable, 'D' are doors/warps, 'S' are stairs.
- Review the action history to avoid repeating failed attempts.
- Learn from previous outputs and function results to refine your strategy.
- Use `press_buttons()` for puzzle gyms. Avoid `navigate_to()` for rotating doors, switches, or moving platforms.
Be specific and actionable. Reference actual coordinates from the porymap when possible."""

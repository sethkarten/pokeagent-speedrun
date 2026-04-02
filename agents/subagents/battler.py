"""Prompt helpers and policy for the battler local subagent."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.prompts.paths import GAME_NAME
from .utils.registry import BATTLE_ALLOWED_TOOL_NAMES


def allowed_battler_tool_names() -> set[str]:
    return set(BATTLE_ALLOWED_TOOL_NAMES)


def format_battler_history(history: List[Dict[str, Any]]) -> str:
    """Format the battler's inner turn history, analogous to the orchestrator's short-term memory."""
    if not history:
        return "No previous battle actions recorded."

    lines: List[str] = []
    for entry in history[-10:]:
        step = entry.get("step", "?")
        reasoning = (entry.get("reasoning") or "").strip()
        lines.append(f"[Battle Turn {step}]")
        if reasoning:
            lines.append(f"  THINKING: {reasoning[:300]}")
        for tc in entry.get("tool_calls", []):
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            result = tc.get("result", "")
            lines.append(f"  TOOL: {name}")
            lines.append(f"    args: {json.dumps(args, ensure_ascii=False)}")
            if result:
                try:
                    lines.append(f"    result: {json.dumps(result, ensure_ascii=False)[:300]}")
                except TypeError:
                    lines.append(f"    result: {str(result)[:300]}")
        lines.append("")
    return "\n".join(lines).strip()


def build_battler_prompt(
    *,
    current_state_text: str,
    location: str,
    objective_state: Dict[str, Any],
    progress: Dict[str, Any],
    memory_summary: str,
    skill_overview: str = "",
    handoff_summary: str,
    battle_history: str,
    turn_index: int,
) -> str:
    return f"""You are the {GAME_NAME} battle subagent.
You are responsible only for resolving the active battle as efficiently and safely as possible.

BATTLE TURN: {turn_index}

LOCATION: {location}

CURRENT BATTLE-FOCUSED GAME STATE:
{current_state_text}

OBJECTIVE STATE:
{objective_state}

PROGRESS SUMMARY:
{progress}

LONG-TERM MEMORY OVERVIEW:
{memory_summary or "No memories recorded yet."}

SKILL LIBRARY:
{skill_overview or "No skills learned yet."}

PRE-BATTLE CONTEXT HANDOFF:
{handoff_summary}

BATTLE HISTORY (recent turns):
{battle_history}

DECISION PROCESS:
1. Analyze the current battle state, menu state, and visible options.
2. Review your battle history to avoid repeating mistakes.
3. Choose the highest-value action for this turn.
4. If useful, store battle-relevant information in memory before the final action.
5. Finish with one tool call."""


def extract_key_events_from_summary(summary_text: str) -> List[str]:
    events: List[str] = []
    for line in (summary_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            events.append(stripped[2:].strip())
    return events[:6]

"""Prompt helpers for the summarize local subagent."""

from __future__ import annotations

from typing import Any, Dict

from agents.subagents.utils.trajectory_window import MAX_TRAJECTORY_WINDOW

DEFAULT_SUMMARY_WINDOW = 25


def build_summarize_prompt(*, context: Dict[str, Any], last_n_steps: int, reasoning: str) -> str:
    current_state = context.get("current_state", {})
    objective_state = context.get("objective_state", {})
    knowledge_summary = context.get("knowledge_summary") or "No knowledge recorded yet."
    progress = context.get("progress", {})

    return f"""You are the summarize subagent for a Pokemon Emerald speedrun agent.
Your job is to produce a detailed, unbiased handoff summary of the latest trajectory window.

FOCUS:
{reasoning or "Summarize the important recent events, current blockers, and best next step."}

CURRENT STATE:
Location: {current_state.get('location')}
Coordinates: ({current_state.get('coordinates', {}).get('x')}, {current_state.get('coordinates', {}).get('y')})
{current_state.get('state_text', '')}

OBJECTIVE STATE:
{objective_state}

PROGRESS SUMMARY:
{progress}

KNOWLEDGE SUMMARY:
{knowledge_summary}

RECENT TRAJECTORY WINDOW (last {last_n_steps} steps):
{context.get('trajectory_summary', 'No prior trajectories recorded.')}

OUTPUT FORMAT:
**SUMMARY**:
[5-10 sentences covering the most important recent developments]

**KEY_EVENTS**:
- [Event 1]
- [Event 2]
- The number of key events should scale to the number of steps in the trajectory window. Note the max number of n steps in the trajectory window is {last_n_steps}.

Be specific, factual, and avoid overclaiming beyond the evidence."""

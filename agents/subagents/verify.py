"""Verification prompt and parsing helpers for local subagents."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.prompts.paths import GAME_NAME


def resolve_verification_target(objective_state: Dict[str, Any], category: Optional[str] = None) -> Dict[str, Any]:
    """Resolve the authoritative objective that verify should judge."""
    mode = objective_state.get("mode", "legacy")
    if mode != "categorized":
        objective = objective_state.get("objective")
        return {
            "mode": "legacy",
            "category": "legacy",
            "objective": objective if isinstance(objective, dict) else {"description": str(objective or "None")},
        }

    selected_category = category or "story"
    category_state = (objective_state.get("categories") or {}).get(selected_category, {})
    objective = category_state.get("current_objective")

    if selected_category == "battling":
        objectives = objective if isinstance(objective, list) else []
        description = "; ".join(obj.get("description", "Unknown") for obj in objectives if isinstance(obj, dict)) or "None"
        completion_conditions = [
            obj.get("completion_condition") for obj in objectives if isinstance(obj, dict) and obj.get("completion_condition")
        ]
        return {
            "mode": "categorized",
            "category": selected_category,
            "objective": {
                "description": description,
                "completion_condition": "; ".join(completion_conditions) if completion_conditions else None,
                "group_size": len(objectives),
            },
        }

    if not isinstance(objective, dict):
        objective = {"description": "None"}

    return {
        "mode": "categorized",
        "category": selected_category,
        "objective": objective,
    }


def build_verify_prompt(
    *,
    context: Dict[str, Any],
    target: Dict[str, Any],
    last_n_steps: int,
    reasoning: str,
) -> str:
    """Build the verify-subagent prompt."""
    current_state = context.get("current_state", {})
    objective = target.get("objective", {})
    memory_summary = context.get("memory_summary") or "No memories recorded yet."
    skill_overview = context.get("skill_overview") or "No skills learned yet."

    return f"""You are the verify subagent for a {GAME_NAME} speedrun agent.
Your only job is to decide whether the current objective has already been completed.

The current screenshot is attached when available. Use it as CURRENT visual evidence only.
Historical context comes from the text trajectory window below, not historical screenshots.
If the evidence is mixed or insufficient, return is_complete=false.

REQUESTED VERIFICATION FOCUS:
{reasoning or "Check whether the current objective is complete."}

AUTHORITATIVE OBJECTIVE TO VERIFY:
Mode: {target.get('mode')}
Category: {target.get('category')}
Description: {objective.get('description')}
Target location: {objective.get('target_location')}
Completion condition: {objective.get('completion_condition')}
Navigation hint: {objective.get('navigation_hint')}

CURRENT STATE:
Location: {current_state.get('location')}
Coordinates: ({current_state.get('coordinates', {}).get('x')}, {current_state.get('coordinates', {}).get('y')})
{current_state.get('state_text', '')}

LONG-TERM MEMORY OVERVIEW:
{memory_summary}

SKILL LIBRARY:
{skill_overview}

RECENT TRAJECTORY WINDOW (last {last_n_steps} steps):
{context.get('trajectory_summary', 'No prior trajectories recorded.')}

RULES:
1. Use only evidence present in current state, screenshot, memory, and trajectory window.
2. Do not mark the objective complete just because progress seems likely.
3. If the objective is ambiguous, incomplete, or contradicted by evidence, set is_complete to false.
4. Keep evidence bullets short and specific.

Return STRICT JSON only with this exact schema:
{{
  "objective_category": "{target.get('category')}",
  "objective_description": "{objective.get('description', '')}",
  "is_complete": false,
  "confidence": "low|medium|high",
  "evidence_for": ["..."],
  "evidence_against": ["..."],
  "recommended_next_action": "...",
  "reasoning_summary": "..."
}}"""


def _strip_markdown_json_fence(text: str) -> str:
    """Remove leading ```json / ``` fences so models can still return fenced JSON."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if not lines:
        return text
    # Drop first fence line (``` or ```json)
    lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_markdown_json_fence(text)
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_verify_response(raw_text: str, *, target: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and normalize the verifier response."""
    cleaned = _strip_markdown_json_fence(raw_text)
    parsed = _extract_json_object(cleaned) or {}
    objective = target.get("objective", {})

    evidence_for = parsed.get("evidence_for")
    if not isinstance(evidence_for, list):
        evidence_for = [str(evidence_for)] if evidence_for else []

    evidence_against = parsed.get("evidence_against")
    if not isinstance(evidence_against, list):
        evidence_against = [str(evidence_against)] if evidence_against else []

    confidence = str(parsed.get("confidence", "low")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    return {
        "objective_category": parsed.get("objective_category") or target.get("category"),
        "objective_description": parsed.get("objective_description") or objective.get("description", "Unknown"),
        "is_complete": bool(parsed.get("is_complete", False)),
        "confidence": confidence,
        "evidence_for": [str(item) for item in evidence_for],
        "evidence_against": [str(item) for item in evidence_against],
        "recommended_next_action": str(parsed.get("recommended_next_action", "")).strip(),
        "reasoning_summary": str(parsed.get("reasoning_summary", "")).strip(),
        # Prefer cleaned text so logs / history are not double-wrapped in ```json fences
        "raw_response": cleaned if cleaned else raw_text.strip(),
    }

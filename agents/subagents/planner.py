"""Prompt helpers, formatters, and tool declarations for the objectives-planning subagent."""

from __future__ import annotations

import json
from typing import Any, Dict, List


PLANNER_ALLOWED_TOOL_NAMES = (
    "get_progress_summary",
    "get_walkthrough",
    "search_knowledge",
    "get_knowledge_summary",
    "add_knowledge",
    "lookup_pokemon_info",
    "subagent_summarize",
    "subagent_verify",
    "subagent_reflect",
    "subagent_gym_puzzle",
    "replan_objectives",
)

PLANNER_SAFETY_CAP = 25
PLANNER_HISTORY_CAP = 20


def allowed_planner_tool_names() -> set[str]:
    return set(PLANNER_ALLOWED_TOOL_NAMES)


REPLAN_OBJECTIVES_TOOL_DECLARATION: Dict[str, Any] = {
    "name": "replan_objectives",
    "description": (
        "Modify the current objective sequence for a single category. Max 5 edits per call. "
        "Each edit specifies an index and either a new objective (create/modify) or null (delete). "
        "Set return_to_orchestrator=true on the FINAL call to exit the planning session and "
        "return to the orchestrator. You may call this tool multiple times (once per category) "
        "before setting return_to_orchestrator=true."
    ),
    "parameters": {
        "type_": "OBJECT",
        "properties": {
            "category": {
                "type_": "STRING",
                "enum": ["story", "battling", "dynamics"],
                "description": "Which category to replan. Only one category per call.",
            },
            "edits": {
                "type_": "ARRAY",
                "items": {
                    "type_": "OBJECT",
                    "properties": {
                        "index": {
                            "type_": "INTEGER",
                            "description": (
                                "Per-category index to modify/create/delete. Must be >= "
                                "the current index for this category."
                            ),
                        },
                        "objective": {
                            "type_": "OBJECT",
                            "description": (
                                "New objective data. Omit or set to null to delete the "
                                "objective at this index."
                            ),
                            "properties": {
                                "id": {"type_": "STRING"},
                                "description": {"type_": "STRING"},
                                "action_type": {
                                    "type_": "STRING",
                                    "enum": [
                                        "navigate",
                                        "interact",
                                        "battle",
                                        "wait",
                                        "create_new_objectives",
                                    ],
                                },
                                "target_location": {"type_": "STRING"},
                                "navigation_hint": {"type_": "STRING"},
                                "completion_condition": {"type_": "STRING"},
                                "optional": {"type_": "BOOLEAN"},
                            },
                            "required": ["id", "description", "action_type"],
                        },
                    },
                    "required": ["index"],
                },
                "description": "Up to 5 edits. Omitting 'objective' or setting it null = delete.",
            },
            "return_to_orchestrator": {
                "type_": "BOOLEAN",
                "description": (
                    "Set true on the final replan call to exit the planning session "
                    "and return to the orchestrator."
                ),
            },
            "rationale": {
                "type_": "STRING",
                "description": (
                    "5-10 sentence explanation of what changed, why, and what the "
                    "orchestrator should do next."
                ),
            },
        },
        "required": ["category", "edits", "return_to_orchestrator", "rationale"],
    },
}


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_full_objective_sequence(snapshot: Dict[str, Any]) -> str:
    """Format the full objective sequence snapshot for inclusion in the planner prompt."""
    if not snapshot:
        return "No objective sequence loaded."

    parts: List[str] = []
    for cat_name in ("story", "battling", "dynamics"):
        cat = snapshot.get(cat_name, {})
        seq = cat.get("sequence", [])
        cur_idx = cat.get("current_index", 0)
        total = cat.get("total", 0)
        completed = cat.get("completed", 0)

        header = (
            f"=== {cat_name.upper()} ({completed}/{total} completed, "
            f"current_index={cur_idx}) ==="
        )
        parts.append(header)

        if not seq:
            parts.append("  (empty)")
            parts.append("")
            continue

        for obj in seq:
            idx = obj.get("index", "?")
            marker = ">>>" if idx == cur_idx else "   "
            status = "[DONE]" if obj.get("completed") else "[    ]"
            opt = " (optional)" if obj.get("optional") else ""
            line = f"  {marker} [{idx}] {status} {obj.get('id', '?')}: {obj.get('description', '')}{opt}"
            parts.append(line)

            details: List[str] = []
            if obj.get("action_type"):
                details.append(f"action_type={obj['action_type']}")
            if obj.get("target_location"):
                details.append(f"target={obj['target_location']}")
            if obj.get("completion_condition"):
                details.append(f"condition={obj['completion_condition']}")
            if obj.get("navigation_hint"):
                details.append(f"hint={obj['navigation_hint'][:120]}")
            if obj.get("prerequisite_story_objective"):
                details.append(f"prereq={obj['prerequisite_story_objective']}")
            if details:
                parts.append(f"        {' | '.join(details)}")

        parts.append("")

    return "\n".join(parts).strip()


def format_planner_history(history: List[Dict[str, Any]]) -> str:
    """Format the planner's short-term memory, analogous to format_battler_history."""
    if not history:
        return "No previous planning actions recorded."

    lines: List[str] = []
    for entry in history[-PLANNER_HISTORY_CAP:]:
        step = entry.get("step", "?")
        reasoning = (entry.get("reasoning") or "").strip()
        lines.append(f"[Planning Turn {step}]")
        if reasoning:
            lines.append(f"  THINKING: {reasoning[:400]}")
        for tc in entry.get("tool_calls", []):
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            result = tc.get("result", "")
            lines.append(f"  TOOL: {name}")
            lines.append(f"    args: {json.dumps(args, ensure_ascii=False)[:300]}")
            if result:
                try:
                    lines.append(f"    result: {json.dumps(result, ensure_ascii=False)[:400]}")
                except TypeError:
                    lines.append(f"    result: {str(result)[:400]}")
        lines.append("")
    return "\n".join(lines).strip()


def build_planner_prompt(
    *,
    reason: str,
    objective_sequence: str,
    current_state_text: str,
    location: str,
    progress: Dict[str, Any],
    knowledge_summary: str,
    planner_history: str,
    handoff_summary: str,
    turn_index: int,
) -> str:
    return f"""You are the objective-planning subagent for a Pokemon Emerald speedrun agent.
Your job is to review, create, modify, or delete objectives in the agent's objective sequence
so the orchestrator always has clear, achievable next steps.

PLANNING TURN: {turn_index}

ORCHESTRATOR'S REASON FOR CALLING YOU:
{reason}

LOCATION: {location}

CURRENT GAME STATE:
{current_state_text}

PROGRESS SUMMARY:
{progress}

KNOWLEDGE SUMMARY:
{knowledge_summary or "No knowledge recorded yet."}

PRE-PLANNING CONTEXT HANDOFF:
{handoff_summary}

PLANNING HISTORY (recent turns):
{planner_history}

FULL OBJECTIVE SEQUENCE (all categories):
{objective_sequence}

INSTRUCTIONS:
1. Review the orchestrator's reason and the current objective sequence.
2. Use research tools (get_walkthrough, get_progress_summary, search_knowledge, lookup_pokemon_info)
   as necessary to gather information about what objectives should come next.
3. Use subagent tools (subagent_verify, subagent_reflect, subagent_summarize) if you need
   analysis of the agent's recent trajectory or verification of objective completion.
4. When you have a clear plan, call replan_objectives() with your edits for ONE category.
   You may call it multiple times (once per category) if you need to modify multiple categories.
5. On your FINAL replan_objectives call, set return_to_orchestrator=true.
6. If no changes are needed, call replan_objectives with an empty edits list and
   return_to_orchestrator=true, explaining why no changes were necessary.

RULES:
- Max 5 edits per replan_objectives call.
- One category per call.
- You can only modify objectives at or after the current index for each category.
- Specify objective=null to explicitly delete an objective.
- Appended objectives must have contiguous indices starting from the current sequence end.
- Include a detailed rationale (5-10 sentences) explaining what changed and why.
- Your rationale should also tell the orchestrator what to do next after returning."""

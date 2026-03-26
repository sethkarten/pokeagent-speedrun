"""Registry metadata for local PokeAgent subagents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Sequence


@dataclass(frozen=True)
class LocalSubagentSpec:
    tool_name: str
    handler_type: str
    interaction_name: str
    handler_method: str
    description: str
    parameters: Dict[str, Any]
    allowed_tool_names: Sequence[str] = ()


BATTLE_ALLOWED_TOOL_NAMES = (
    "press_buttons",
    "process_memory",
    "process_skill",
    "add_memory",
    "search_memory",
    "get_memory_summary",
    "get_progress_summary",
    "lookup_pokemon_info",
)

PLANNER_ALLOWED_TOOL_NAMES = (
    "get_progress_summary",
    "get_walkthrough",
    "process_memory",
    "process_skill",
    "search_memory",
    "get_memory_summary",
    "add_memory",
    "lookup_pokemon_info",
    "subagent_summarize",
    "subagent_verify",
    "subagent_reflect",
    "subagent_gym_puzzle",
    "replan_objectives",
)

# Tools that all subagents are never allowed to call (prevents recursion).
SUBAGENT_FORBIDDEN_TOOLS: FrozenSet[str] = frozenset({"execute_custom_subagent"})

# Tool names of built-in subagents (toggled off in experimental settings).
BUILTIN_SUBAGENT_TOOL_NAMES: FrozenSet[str] = frozenset({
    "subagent_reflect",
    "subagent_verify",
    "subagent_gym_puzzle",
    "subagent_summarize",
    "subagent_battler",
    "subagent_plan_objectives",
})


LOCAL_SUBAGENT_SPECS = (
    LocalSubagentSpec(
        tool_name="subagent_reflect",
        handler_type="one_step",
        interaction_name="Subagent_Reflect",
        handler_method="_execute_subagent_reflect",
        description=(
            "Use this when you feel stuck, uncertain, or suspect your current "
            "approach/objectives are wrong. This local subagent reviews the "
            "latest trajectory window, current state, and current screenshot "
            "to diagnose what should change next."
        ),
        parameters={
            "type_": "OBJECT",
            "properties": {
                "situation": {
                    "type_": "STRING",
                    "description": (
                        "Describe what you've been trying to do and why you think "
                        "something might be wrong. Include recent actions, lack "
                        "of progress, confusion about objectives, or any "
                        "observations that seem off."
                    ),
                },
                "last_n_steps": {
                    "type_": "INTEGER",
                    "description": (
                        "Optional. Number of recent trajectory steps to review. "
                        "Defaults to 10 and is capped at 50."
                    ),
                },
            },
            "required": ["situation"],
        },
    ),
    LocalSubagentSpec(
        tool_name="subagent_verify",
        handler_type="one_step",
        interaction_name="Subagent_Verify",
        handler_method="_execute_subagent_verify",
        description=(
            "Use this before completing an objective when you want an explicit "
            "verdict on whether the current objective is actually finished. "
            "This local subagent checks the current state, current screenshot, "
            "and recent trajectory window against the authoritative objective."
        ),
        parameters={
            "type_": "OBJECT",
            "properties": {
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "Why you want verification right now. Mention the "
                        "evidence you think might show the objective is complete."
                    ),
                },
                "category": {
                    "type_": "STRING",
                    "enum": ["story", "battling", "dynamics"],
                    "description": (
                        "Optional. In categorized mode, which current objective "
                        "category to verify. Defaults to story if omitted."
                    ),
                },
                "last_n_steps": {
                    "type_": "INTEGER",
                    "description": (
                        "Optional. Number of recent trajectory steps to review. "
                        "Defaults to 10 and is capped at 50."
                    ),
                },
            },
            "required": ["reasoning"],
        },
    ),
    LocalSubagentSpec(
        tool_name="subagent_gym_puzzle",
        handler_type="one_step",
        interaction_name="Gym_Puzzle_Analysis",
        handler_method="_execute_subagent_gym_puzzle",
        description=(
            "Get expert guidance on solving gym puzzles. Use this when you're "
            "in a gym and need help understanding the puzzle mechanics or "
            "finding the solution."
        ),
        parameters={
            "type_": "OBJECT",
            "properties": {
                "gym_name": {
                    "type_": "STRING",
                    "description": (
                        "Gym / map identifier from the current game state, e.g. "
                        "'LAVARIDGE_TOWN_GYM_1F' or 'MOSSDEEP_CITY_GYM'."
                    ),
                }
            },
            "required": [],
        },
    ),
    LocalSubagentSpec(
        tool_name="subagent_summarize",
        handler_type="one_step",
        interaction_name="Subagent_Summarize",
        handler_method="_execute_subagent_summarize",
        description=(
            "Summarize the latest trajectory window into a detailed, unbiased "
            "handoff. Useful for context compaction, battle handoff, or broad "
            "situation review."
        ),
        parameters={
            "type_": "OBJECT",
            "properties": {
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "What you want emphasized in the summary, such as "
                        "battle handoff, progress recap, or uncertainty review."
                    ),
                },
                "last_n_steps": {
                    "type_": "INTEGER",
                    "description": (
                        "Optional. Number of recent trajectory steps to review. "
                        "Defaults to 25 and is capped at 50."
                    ),
                },
            },
            "required": [],
        },
    ),
    LocalSubagentSpec(
        tool_name="subagent_battler",
        handler_type="looping",
        interaction_name="Subagent_Battler",
        handler_method="_execute_subagent_battler",
        description=(
            "Delegate the current battle to a specialized local battler loop. "
            "Consumes real global steps while in battle and returns a compacted "
            "battle summary to the main orchestrator."
        ),
        parameters={
            "type_": "OBJECT",
            "properties": {
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "Why battle delegation is appropriate right now and what "
                        "the battler should prioritize."
                    ),
                },
                "last_n_steps": {
                    "type_": "INTEGER",
                    "description": (
                        "Optional. Number of recent trajectory steps to summarize "
                        "for the battle handoff. Defaults to 25 and is capped at 50."
                    ),
                },
            },
            "required": [],
        },
        allowed_tool_names=BATTLE_ALLOWED_TOOL_NAMES,
    ),
    LocalSubagentSpec(
        tool_name="subagent_plan_objectives",
        handler_type="looping",
        interaction_name="Subagent_Plan_Objectives",
        handler_method="_execute_subagent_plan_objectives",
        description=(
            "Delegate objective planning/replanning to a specialized subagent. "
            "This agent reviews the full objective sequence, uses research tools "
            "and other subagents, then modifies objectives via replan_objectives. "
            "Call this when you need new objectives, when you are stuck and believe "
            "replanning is needed, or when the current objective sequence is exhausted."
        ),
        parameters={
            "type_": "OBJECT",
            "properties": {
                "reason": {
                    "type_": "STRING",
                    "description": (
                        "Why planning is needed. Include context about what is stuck, "
                        "what just changed, or why new objectives are needed."
                    ),
                },
                "last_n_steps": {
                    "type_": "INTEGER",
                    "description": (
                        "Optional. Number of recent trajectory steps for the initial "
                        "summary. Defaults to 25 and is capped at 50."
                    ),
                },
            },
            "required": ["reason"],
        },
        allowed_tool_names=PLANNER_ALLOWED_TOOL_NAMES,
    ),
    # ----- M3: Generalized subagent primitives -----
    LocalSubagentSpec(
        tool_name="execute_custom_subagent",
        handler_type="looping",
        interaction_name="Execute_Custom_Subagent",
        handler_method="_execute_custom_subagent",
        description=(
            "Launch an autonomous subagent that runs multiple actions on its own, "
            "receiving fresh game state each turn. The subagent runs until it "
            "signals return_to_orchestrator or hits max_steps. You do NOT need to "
            "call it multiple times — it loops internally. Use subagent_id for "
            "registry entries or config for ad-hoc. Cannot nest subagents."
        ),
        parameters={
            "type_": "OBJECT",
            "properties": {
                "subagent_id": {
                    "type_": "STRING",
                    "description": (
                        "ID of a subagent in the registry (e.g. 'sa_0007'). "
                        "Mutually exclusive with 'config'."
                    ),
                },
                "config": {
                    "type_": "OBJECT",
                    "description": (
                        "Inline subagent config for ad-hoc execution. Fields: "
                        "max_turns (int), available_tools (list of tool names), "
                        "system_instructions (str), directive (str), "
                        "return_condition (str)."
                    ),
                    "properties": {
                        "max_turns": {"type_": "INTEGER"},
                        "available_tools": {
                            "type_": "ARRAY",
                            "items": {"type_": "STRING"},
                        },
                        "system_instructions": {"type_": "STRING"},
                        "directive": {"type_": "STRING"},
                        "return_condition": {"type_": "STRING"},
                    },
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "Why you are launching this subagent now.",
                },
                "max_steps": {
                    "type_": "INTEGER",
                    "description": (
                        "Maximum number of actions the subagent can take before "
                        "returning control. Overrides the registry's max_turns. "
                        "The subagent runs autonomously, receiving fresh game state "
                        "each turn, until it signals return_to_orchestrator or hits "
                        "this limit. Default: registry value or 25."
                    ),
                },
            },
            "required": ["reasoning"],
        },
    ),
    LocalSubagentSpec(
        tool_name="process_trajectory_history",
        handler_type="one_step",
        interaction_name="Process_Trajectory_History",
        handler_method="_execute_process_trajectory_history",
        description=(
            "Analyze a window of trajectory history with a custom directive. "
            "Reads trajectory entries in the requested step range and passes "
            "them with your directive to a VLM for analysis. Use this for "
            "custom reflection, pattern detection, or behaviour review."
        ),
        parameters={
            "type_": "OBJECT",
            "properties": {
                "window_range": {
                    "type_": "ARRAY",
                    "items": {"type_": "INTEGER"},
                    "description": (
                        "Two-element array [start_step, end_step] defining "
                        "the range of trajectory steps to analyze. The range "
                        "is clipped to available data."
                    ),
                },
                "directive": {
                    "type_": "STRING",
                    "description": (
                        "Your analysis directive — what should the VLM look "
                        "for or reason about in this trajectory window?"
                    ),
                },
            },
            "required": ["window_range", "directive"],
        },
    ),
)


LOCAL_SUBAGENT_SPEC_BY_NAME = {spec.tool_name: spec for spec in LOCAL_SUBAGENT_SPECS}


def get_local_subagent_spec(tool_name: str) -> Optional[LocalSubagentSpec]:
    return LOCAL_SUBAGENT_SPEC_BY_NAME.get(tool_name)


def is_local_subagent_tool(tool_name: str) -> bool:
    return tool_name in LOCAL_SUBAGENT_SPEC_BY_NAME


def build_local_subagent_tool_declarations(
    include_builtins: bool = True,
) -> list[Dict[str, Any]]:
    """Build tool declarations for all registered local subagents.

    When *include_builtins* is ``False``, built-in subagents (reflect,
    verify, summarize, gym_puzzle, battler, planner) are excluded so that
    only the generic primitives (execute_custom_subagent,
    process_trajectory_history) are exposed.  This is used for
    experimental settings where the AutoEvolve loop must learn subagent
    usage from scratch.
    """
    declarations: list[Dict[str, Any]] = []
    for spec in LOCAL_SUBAGENT_SPECS:
        if not include_builtins and spec.tool_name in BUILTIN_SUBAGENT_TOOL_NAMES:
            continue
        declarations.append(
            {
                "name": spec.tool_name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
        )
    return declarations

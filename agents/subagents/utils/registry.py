"""Registry metadata for local PokeAgent subagents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


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
    "add_knowledge",
    "search_knowledge",
    "get_knowledge_summary",
    "get_progress_summary",
    "lookup_pokemon_info",
)


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
)


LOCAL_SUBAGENT_SPEC_BY_NAME = {spec.tool_name: spec for spec in LOCAL_SUBAGENT_SPECS}


def get_local_subagent_spec(tool_name: str) -> Optional[LocalSubagentSpec]:
    return LOCAL_SUBAGENT_SPEC_BY_NAME.get(tool_name)


def is_local_subagent_tool(tool_name: str) -> bool:
    return tool_name in LOCAL_SUBAGENT_SPEC_BY_NAME


def build_local_subagent_tool_declarations() -> list[Dict[str, Any]]:
    return [
        {
            "name": spec.tool_name,
            "description": spec.description,
            "parameters": spec.parameters,
        }
        for spec in LOCAL_SUBAGENT_SPECS
    ]

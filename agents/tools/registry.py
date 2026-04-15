"""Centralized tool declaration registry with per-scaffold availability.

Each entry in ``TOOL_REGISTRY`` carries a ``scaffolds`` field that is either
the sentinel string ``"all"`` (available to every scaffold) or a ``set`` of
scaffold names where the tool should be declared.

``build_tools_for_scaffold(scaffold)`` returns the filtered list of Gemini-
compatible tool declarations for the requested scaffold.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from agents.subagents.utils.registry import (
    BUILTIN_SUBAGENT_TOOL_NAMES,
    LOCAL_SUBAGENT_SPECS,
)
from agents.subagents.planner import REPLAN_OBJECTIVES_TOOL_DECLARATION

# ---------------------------------------------------------------------------
# Scaffold tier constants
# ---------------------------------------------------------------------------

ALL_SCAFFOLDS = "all"

# Scaffolds that get the expert-level tools (pathfinding, walkthrough, wiki)
EXPERT_SCAFFOLDS = frozenset({"pokeagent", "autonomous_cli"})

# Scaffolds where built-in subagents are excluded, but generic primitives
# (skills, code, subagent CRUD, custom subagents, etc.) are available.
NO_BUILTINS_SCAFFOLDS = frozenset({"simple", "autoevolve"})

# Union of all scaffolds except simplest — used for tools that every scaffold
# except the bare-minimum one should have.
_ALL_EXCEPT_SIMPLEST = frozenset(
    {"pokeagent", "autonomous_cli", "simple", "autoevolve", "simplest"}
)
_STANDARD_SCAFFOLDS = frozenset(
    {"pokeagent", "autonomous_cli", "simple", "autoevolve"}
)


def _press_buttons_description() -> str:
    gt = os.environ.get("GAME_TYPE", "emerald").lower()
    if gt == "red":
        return (
            "Press Game Boy buttons to interact with the game. Available "
            "buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, WAIT "
            "(Game Boy has no L/R shoulder buttons). Use WAIT to observe "
            "without pressing any button."
        )
    return (
        "Press Game Boy Advance buttons to interact with the game. "
        "Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, "
        "L, R, WAIT. Use WAIT to observe without pressing any button."
    )


# ---------------------------------------------------------------------------
# Registry: each entry is {name, description, parameters, scaffolds}
# ---------------------------------------------------------------------------

TOOL_REGISTRY: List[Dict[str, Any]] = [
    # -- press_buttons: every scaffold --
    {
        "name": "press_buttons",
        "scaffolds": ALL_SCAFFOLDS,
        "description": _press_buttons_description(),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "buttons": {
                    "type_": "ARRAY",
                    "items": {"type_": "STRING"},
                    "description": "List of buttons to press (e.g., ['A'], ['UP'])",
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "REQUIRED FORMAT: Must include 'ANALYZE: [game screen, "
                        "location, objective, situation]' and 'PLAN: [action, "
                        "reason, expected result]'. Example: 'ANALYZE: Dialogue "
                        "box visible. Location: Route 101 (5,8). Objective: Talk "
                        "to Prof Birch. Situation: Birch asking for help. PLAN: "
                        "Action: Press A. Reason: Advance dialogue. Expected: "
                        "Dialogue continues or battle starts.'"
                    ),
                },
            },
            "required": ["buttons", "reasoning"],
        },
    },
    # -- complete_direct_objective: all except simplest --
    {
        "name": "complete_direct_objective",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Complete the current direct objective and advance to the next "
            "one. In CATEGORIZED mode, you must specify which category "
            "objective to complete (story, battling, or dynamics). In "
            "LEGACY mode, category is ignored."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "REQUIRED FORMAT: Must include 'ANALYZE: [current state, "
                        "objective requirements, completion evidence]' and "
                        "'PLAN: [confirm completion, next objective]'. Example: "
                        "'ANALYZE: Objective was to reach Route 101. Current "
                        "location shows Route 101 at (15,5). Evidence: Game text "
                        "shows \"Route 101\". PLAN: Objective complete, marking "
                        "as done. Next: Talk to Prof Birch.'"
                    ),
                },
                "category": {
                    "type_": "STRING",
                    "enum": ["story", "battling", "dynamics"],
                    "description": (
                        "Which category objective to complete (required in "
                        "CATEGORIZED mode). 'story' for narrative objectives, "
                        "'battling' for team objectives, 'dynamics' for "
                        "agent-created objectives."
                    ),
                },
            },
            "required": ["reasoning"],
        },
    },
    # -- process_memory: every scaffold --
    {
        "name": "process_memory",
        "scaffolds": ALL_SCAFFOLDS,
        "description": (
            "Manage long-term memory. The LONG-TERM MEMORY OVERVIEW in your "
            "prompt shows all entry IDs. Use 'read' to get full details, "
            "'add' to create, 'update' to modify, 'delete' to remove."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "action": {
                    "type_": "STRING",
                    "enum": ["read", "add", "update", "delete"],
                    "description": "The action to perform on memory entries.",
                },
                "entries": {
                    "type_": "ARRAY",
                    "items": {
                        "type_": "OBJECT",
                        "properties": {
                            "id": {
                                "type_": "STRING",
                                "description": (
                                    "Entry ID or name (required for read/update/"
                                    "delete). For add, optionally specify a "
                                    "descriptive ID (e.g. 'route_101_map'). You "
                                    "can look up entries by ID or title."
                                ),
                            },
                            "path": {
                                "type_": "STRING",
                                "description": (
                                    "Category path, e.g. 'navigation/routes'. "
                                    "Defaults to 'uncategorized'."
                                ),
                            },
                            "title": {
                                "type_": "STRING",
                                "description": "Short descriptive title (required for add).",
                            },
                            "content": {
                                "type_": "STRING",
                                "description": "Detailed content text (required for add).",
                            },
                            "importance": {
                                "type_": "INTEGER",
                                "description": "1-5 importance rating (default 3).",
                            },
                            "location": {
                                "type_": "STRING",
                                "description": "Game location this memory relates to.",
                            },
                        },
                    },
                    "description": (
                        "For read: [{id}]. For add: [{path, title, content}] — "
                        "title AND content required. For update: [{id, ...fields}]. "
                        "For delete: [{id}]."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "Required. Brief justification for this memory operation "
                        "(what you are trying to learn or change and why)."
                    ),
                },
            },
            "required": ["action", "entries", "reasoning"],
        },
    },
    # -- process_skill: all except simplest --
    {
        "name": "process_skill",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Manage the skill library. The SKILL LIBRARY section in your "
            "prompt shows all skill IDs. Use 'read' to get full details, "
            "'add' to create, 'update' to modify, 'delete' to remove."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "action": {
                    "type_": "STRING",
                    "enum": ["read", "add", "update", "delete"],
                    "description": "The action to perform on skill entries.",
                },
                "entries": {
                    "type_": "ARRAY",
                    "items": {
                        "type_": "OBJECT",
                        "properties": {
                            "id": {
                                "type_": "STRING",
                                "description": (
                                    "Entry ID (required for read/update/delete). "
                                    "For add, optionally specify a descriptive ID "
                                    "(e.g. 'move_to_coords'). You can look up "
                                    "entries by ID or name."
                                ),
                            },
                            "name": {
                                "type_": "STRING",
                                "description": "Skill name (required for add).",
                            },
                            "description": {
                                "type_": "STRING",
                                "description": (
                                    "What the skill does and when to use it "
                                    "(required for add)."
                                ),
                            },
                            "path": {
                                "type_": "STRING",
                                "description": (
                                    "Category path, e.g. 'battle/type_matchups'. "
                                    "Defaults to 'general'."
                                ),
                            },
                            "effectiveness": {
                                "type_": "STRING",
                                "description": "'low', 'medium', or 'high'.",
                            },
                            "code": {
                                "type_": "STRING",
                                "description": (
                                    "Optional executable Python code for the skill. "
                                    "The code receives a `tools` dict with callable "
                                    "functions (e.g. tools['press_buttons'](buttons="
                                    "[...], reasoning=...), tools['get_game_state']"
                                    "()). Return value is sent back to you."
                                ),
                            },
                        },
                    },
                    "description": (
                        "For read: [{id}]. For add: [{name, description}] — both "
                        "required; optionally include 'code' for executable skills. "
                        "For update: [{id, ...fields}]. For delete: [{id}]."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "Required. Brief justification for this skill operation "
                        "(what strategy you are recording or updating and why)."
                    ),
                },
            },
            "required": ["action", "entries", "reasoning"],
        },
    },
    # -- run_skill: all except simplest --
    {
        "name": "run_skill",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Execute a skill's code. The skill must have a 'code' field "
            "containing Python. The code runs in a sandbox with access to "
            "game tools via a `tools` dict (e.g. tools['press_buttons']"
            "(buttons=['A'], reasoning='...'), tools['get_game_state']()). "
            "Use this for executable skills like custom pathfinding or "
            "battle routines."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "skill_id": {
                    "type_": "STRING",
                    "description": "ID of the skill to execute (e.g. 'skill_0001')",
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "Why you are running this skill now",
                },
                "args": {
                    "type_": "OBJECT",
                    "description": (
                        "REQUIRED for executable skills. Key-value arguments "
                        "passed to the skill code. Example for move_to_coords: "
                        '{"x": 5, "y": 12}. Check the skill\'s description for '
                        "what args it expects."
                    ),
                    "properties": {
                        "x": {
                            "type_": "INTEGER",
                            "description": "Target X coordinate (for navigation skills)",
                        },
                        "y": {
                            "type_": "INTEGER",
                            "description": "Target Y coordinate (for navigation skills)",
                        },
                    },
                },
            },
            "required": ["skill_id", "reasoning", "args"],
        },
    },
    # -- run_code: all except simplest --
    {
        "name": "run_code",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Execute arbitrary Python code in the game sandbox. Use this to "
            "prototype, debug, and test code BEFORE saving it as a skill. "
            "The code has access to the same `tools` dict as run_skill "
            "(press_buttons, get_game_state, etc.) plus `args`. Set `result` "
            "to return data. Use this to: inspect get_game_state() output, "
            "test map parsing, prototype pathfinding, debug skill code. "
            "Stdout from print() is captured."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "code": {
                    "type_": "STRING",
                    "description": (
                        "Python code to execute. Has access to: "
                        "tools['press_buttons'](), tools['get_game_state'](), "
                        "args, random, collections, heapq, numpy/np, json, re, "
                        "math. Set 'result' variable to return data."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "What you are testing or prototyping",
                },
                "args": {
                    "type_": "OBJECT",
                    "description": "Optional arguments passed as the `args` dict",
                    "properties": {},
                },
            },
            "required": ["code", "reasoning"],
        },
    },
    # -- process_subagent: all except simplest --
    {
        "name": "process_subagent",
        "scaffolds": _STANDARD_SCAFFOLDS,
        "description": (
            "Manage the subagent registry. The SUBAGENT REGISTRY section in "
            "your prompt shows all subagent IDs. Use 'read' to get full "
            "config, 'add' to create, 'update' to modify, 'delete' to "
            "remove (built-ins cannot be deleted)."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "action": {
                    "type_": "STRING",
                    "enum": ["read", "add", "update", "delete"],
                    "description": "The action to perform on subagent entries.",
                },
                "entries": {
                    "type_": "ARRAY",
                    "items": {
                        "type_": "OBJECT",
                        "properties": {
                            "id": {
                                "type_": "STRING",
                                "description": (
                                    "Entry ID or name (required for read/update/"
                                    "delete). For add, optionally specify a "
                                    "descriptive ID (e.g. 'battle_handler'). You "
                                    "can look up entries by ID or name."
                                ),
                            },
                            "name": {
                                "type_": "STRING",
                                "description": "Subagent name (required for add).",
                            },
                            "description": {
                                "type_": "STRING",
                                "description": "What the subagent does (required for add).",
                            },
                            "system_instructions": {
                                "type_": "STRING",
                                "description": "System prompt for the subagent (max 12000 chars).",
                            },
                            "directive": {
                                "type_": "STRING",
                                "description": "Per-invocation directive (max 12000 chars).",
                            },
                            "handler_type": {
                                "type_": "STRING",
                                "description": "'one_step' or 'looping' (default 'looping').",
                            },
                            "max_turns": {
                                "type_": "INTEGER",
                                "description": "Max turns for looping subagents (default 25).",
                            },
                            "return_condition": {
                                "type_": "STRING",
                                "description": "When the subagent should return control.",
                            },
                        },
                    },
                    "description": (
                        "For read: [{id}]. For add: [{name, description, "
                        "system_instructions?, directive?}] — name AND description "
                        "required. For update: [{id, ...fields}]. For delete: [{id}]."
                    ),
                },
                "reasoning": {
                    "type_": "STRING",
                    "description": "Required. Brief justification for this subagent operation.",
                },
            },
            "required": ["action", "entries", "reasoning"],
        },
    },
    # -- get_progress_summary: expert scaffolds only --
    {
        "name": "get_progress_summary",
        "scaffolds": EXPERT_SCAFFOLDS,
        "description": (
            "Get progress: milestones, current location/coords, "
            "direct-objective status, completed objectives history, memory "
            "tree overview, and run directory. Call with no arguments."
        ),
        "parameters": {"type_": "OBJECT", "properties": {}, "required": []},
    },
    # -- navigate_to: expert scaffolds only --
    {
        "name": "navigate_to",
        "scaffolds": EXPERT_SCAFFOLDS,
        "description": (
            "Automatically navigate to specific coordinates using A* "
            "pathfinding. IMPORTANT: Always specify the variance parameter. "
            "If you get blocked repeatedly at the same position, increase "
            "variance to 'medium', 'high', or 'extreme' to explore "
            "alternative paths."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "x": {"type_": "INTEGER", "description": "Target X coordinate"},
                "y": {"type_": "INTEGER", "description": "Target Y coordinate"},
                "variance": {
                    "type_": "STRING",
                    "description": (
                        "REQUIRED. Path variance level: 'none' (optimal path, "
                        "use first), 'low' (1-step variation), 'medium' (3-step "
                        "variation, use if blocked), 'high' (5-step variation, "
                        "use if very stuck), 'extreme' (8-step variation, use as "
                        "last resort). Default: 'none'"
                    ),
                    "enum": ["none", "low", "medium", "high", "extreme"],
                },
                "reason": {
                    "type_": "STRING",
                    "description": (
                        "REQUIRED FORMAT: Must include 'ANALYZE: [current "
                        "location, objective, destination details]' and "
                        "'PLAN: [why navigating here, what to do when arrive]'. "
                        "Example: 'ANALYZE: Currently at Littleroot (8,10). "
                        "Objective: Meet Prof Birch. Destination: Route 101 "
                        "entrance at (15,5). PLAN: Navigate to encounter Birch "
                        "being attacked. Will save him to progress story.'"
                    ),
                },
                "consider_npcs": {
                    "type_": "BOOLEAN",
                    "description": (
                        "Whether to treat NPCs as obstacles during pathfinding. "
                        "Set to true to avoid NPCs (recommended for most "
                        "navigation). Set to false only if NPCs are "
                        "wandering/moving and you want to ignore them."
                    ),
                },
            },
            "required": ["x", "y", "variance", "reason", "consider_npcs"],
        },
    },
    # -- get_walkthrough: expert scaffolds only --
    {
        "name": "get_walkthrough",
        "scaffolds": EXPERT_SCAFFOLDS,
        "description": (
            "Get official Emerald walkthrough (Parts 1-21). Part 1: "
            "Littleroot, Part 6: Roxanne, Part 21: Elite Four."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "part": {
                    "type_": "INTEGER",
                    "description": "Walkthrough part 1-21",
                },
            },
            "required": ["part"],
        },
    },
    # -- lookup_pokemon_info: expert scaffolds only --
    {
        "name": "lookup_pokemon_info",
        "scaffolds": EXPERT_SCAFFOLDS,
        "description": (
            "Look up Pokemon information from Bulbapedia (stats, moves, "
            "evolution, locations)."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "topic": {
                    "type_": "STRING",
                    "description": "Pokemon name or topic to look up",
                },
                "source": {
                    "type_": "STRING",
                    "description": "Wiki source (default: bulbapedia)",
                },
            },
            "required": ["topic"],
        },
    },
    # -- replan_objectives: simple + autoevolve only (not expert, not simplest) --
    {
        **REPLAN_OBJECTIVES_TOOL_DECLARATION,
        "scaffolds": NO_BUILTINS_SCAFFOLDS,
    },
    # -- evolve_harness: autoevolve only --
    {
        "name": "evolve_harness",
        "scaffolds": frozenset({"autoevolve"}),
        "description": (
            "Trigger an evolution pass NOW to improve skills, subagents, and "
            "memory based on recent performance. Use this when you notice a "
            "skill or subagent is underperforming and needs improvement, "
            "rather than waiting for the automatic evolution cycle."
        ),
        "parameters": {
            "type_": "OBJECT",
            "properties": {
                "reasoning": {
                    "type_": "STRING",
                    "description": (
                        "What needs improvement and why (e.g., 'navigate_to "
                        "skill gets stuck 80% of the time, needs obstacle "
                        "avoidance')"
                    ),
                },
                "num_steps": {
                    "type_": "INTEGER",
                    "description": (
                        "Number of recent trajectory steps to analyze (default 50)"
                    ),
                },
            },
            "required": ["reasoning"],
        },
    },
]


def _tool_available(entry: Dict[str, Any], scaffold: str) -> bool:
    """Return True if the tool entry is available for *scaffold*."""
    s = entry.get("scaffolds", ALL_SCAFFOLDS)
    if s == ALL_SCAFFOLDS:
        return True
    return scaffold in s


def build_tools_for_scaffold(scaffold: str) -> List[Dict[str, Any]]:
    """Return the filtered list of Gemini tool declarations for *scaffold*.

    Combines the static ``TOOL_REGISTRY`` with the dynamically-assembled
    local-subagent declarations from
    ``agents.subagents.utils.registry.build_local_subagent_tool_declarations``.
    """
    # 1) Static registry tools
    tools: List[Dict[str, Any]] = []
    for entry in TOOL_REGISTRY:
        if not _tool_available(entry, scaffold):
            continue
        tools.append(
            {
                "name": entry["name"],
                "description": entry["description"],
                "parameters": entry["parameters"],
            }
        )

    # 2) Local subagent tools (builtins vs generic primitives)
    #    simplest gets none of these at all.
    if scaffold != "simplest":
        include_builtins = scaffold in EXPERT_SCAFFOLDS
        for spec in LOCAL_SUBAGENT_SPECS:
            if not include_builtins and spec.tool_name in BUILTIN_SUBAGENT_TOOL_NAMES:
                continue
            tools.append(
                {
                    "name": spec.tool_name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                }
            )

    return tools

#!/usr/bin/env python3
"""
BrowserGameAgent — Auto-evolve style agent for browser-based games.

Uses the same harness evolution pattern as PokeAgent's autoevolve scaffold
but adapted for browser games played via Playwright.  The agent starts with
no pre-built skills, subagents, or game knowledge and discovers everything
through observation and experimentation.
"""

import os
import sys
import time
import json
import logging
import requests
import traceback
import base64
import io
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from collections import OrderedDict

import PIL.Image as PILImage
import numpy as np
import google.generativeai.types as genai_types
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metric_tracking.server_metrics import update_server_metrics
from utils.data_persistence.llm_logger import get_llm_logger
from utils.agent_infrastructure.vlm_backends import VLM
from utils.data_persistence.run_data_manager import (
    get_run_data_manager,
    initialize_run_data_manager,
)
from agents.subagents import (
    PokeAgentRuntime,
    build_local_subagent_tool_declarations,
    get_local_subagent_spec,
    is_local_subagent_tool,
)
from agents.subagents.utils.executor import SubagentExecutor
from agents.subagents.planner import REPLAN_OBJECTIVES_TOOL_DECLARATION
from utils.json_utils import convert_protobuf_value, convert_protobuf_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ACTION_HISTORY_WINDOW = 20

# ---------------------------------------------------------------------------
# MCP adapter — maps tool names to server HTTP endpoints
# ---------------------------------------------------------------------------

class BrowserMCPToolAdapter:
    """HTTP adapter for browser game MCP endpoints."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        endpoint_map = {
            # Browser game actions
            "get_game_state": "/mcp/get_game_state",
            "press_keys": "/mcp/press_keys",
            "mouse_click": "/mcp/mouse_click",
            "click_element": "/mcp/click_element",
            "double_click": "/mcp/double_click",
            "hold_key": "/mcp/hold_key",
            "mouse_move": "/mcp/mouse_move",
            "mouse_drag": "/mcp/mouse_drag",
            "key_down": "/mcp/key_down",
            "key_up": "/mcp/key_up",
            "wait_ms": "/mcp/wait_ms",
            # Stores (generic — work for any game type)
            "process_memory": "/mcp/process_memory",
            "get_memory_overview": "/mcp/get_memory_overview",
            "process_skill": "/mcp/process_skill",
            "get_skill_overview": "/mcp/get_skill_overview",
            "process_subagent": "/mcp/process_subagent",
            "get_subagent_overview": "/mcp/get_subagent_overview",
            # Objectives (optional)
            "replan_objectives": "/mcp/replan_objectives",
            "get_full_objective_sequence": "/mcp/get_full_objective_sequence",
        }

        endpoint = endpoint_map.get(tool_name)
        if not endpoint:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        url = f"{self.server_url}{endpoint}"
        logger.info(f"Calling MCP tool: {tool_name}")

        # Convert protobuf args
        converted = {}
        for k, v in arguments.items():
            converted[k] = convert_protobuf_value(v)

        try:
            response = requests.post(url, json=converted, timeout=90)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Tool {tool_name} completed")
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# BrowserGameAgent
# ---------------------------------------------------------------------------

class BrowserGameAgent:
    """Auto-evolve style agent for browser-based games."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model: str = "gemini-2.5-flash",
        backend: str = "gemini",
        max_steps: Optional[int] = None,
        game_url: str = "",
        enable_prompt_optimization: bool = True,
        optimization_frequency: int = 25,
    ):
        logger.info(
            f"Initializing BrowserGameAgent backend={backend} model={model} server={server_url}"
        )
        self.server_url = server_url
        self.model = model
        self.backend = backend
        self.max_steps = max_steps
        self.game_url = game_url
        self.step_count = 0
        self.optimization_enabled = enable_prompt_optimization
        self.optimization_frequency = optimization_frequency

        self.conversation_history: List[Dict] = []
        self.recent_function_results: List[Dict] = []
        self._subagent_vlm_cache: OrderedDict = OrderedDict()
        self._VLM_CACHE_CAP = 10
        self._local_subagent_vlm = None

        self.runtime = PokeAgentRuntime(
            initial_step=self.step_count,
            publish_history=self._add_to_history,
            publish_function_result=self._store_function_result_for_context,
            on_step_change=lambda step: setattr(self, "step_count", step),
        )

        # Load system instructions
        self.system_instructions = self._load_system_instructions()

        # MCP adapter
        self.mcp_adapter = BrowserMCPToolAdapter(server_url)

        # Tool declarations
        self.tools = self._create_tool_declarations()

        # VLM
        self.vlm = VLM(
            backend=self.backend,
            model_name=self.model,
            tools=self.tools,
            system_instruction=self.system_instructions,
        )
        logger.info(f"VLM initialized: {self.backend}/{self.model}, {len(self.tools)} tools")

        # LLM logger
        self.llm_logger = get_llm_logger()

        # Subagent executor
        self.executor = SubagentExecutor(
            runtime=self.runtime,
            mcp_adapter=self.mcp_adapter,
            get_run_data_manager_fn=get_run_data_manager,
            server_url=self.server_url,
            get_subagent_vlm_fn=self._get_subagent_vlm,
            handle_vlm_function_calls_fn=self._handle_vlm_function_calls,
            extract_text_fn=self._extract_text_from_response,
            log_trajectory_fn=self._log_trajectory_for_step,
            llm_logger=self.llm_logger,
            wait_for_actions_fn=lambda **kw: None,  # browser actions are synchronous
        )

        # Harness evolver
        self.prompt_optimizer = None
        self.harness_evolver = None
        if self.optimization_enabled:
            run_manager = get_run_data_manager()
            if run_manager:
                from agents.utils.harness_evolver import create_harness_evolver

                base_prompt_path = "agents/prompts/browser-game-directives/BASE_ORCHESTRATOR_POLICY.md"
                system_prompt_path = "agents/prompts/browser-game-directives/SYSTEM_PROMPT.md"
                self.harness_evolver = create_harness_evolver(
                    vlm=self.vlm,
                    run_data_manager=run_manager,
                    base_prompt_path=base_prompt_path,
                    system_prompt_path=system_prompt_path,
                )
                self.prompt_optimizer = self.harness_evolver.prompt_optimizer
                logger.info("HarnessEvolver enabled")

    # ------------------------------------------------------------------
    # System instructions
    # ------------------------------------------------------------------

    def _load_system_instructions(self) -> str:
        prompt_path = (
            Path(__file__).parent / "prompts" / "browser-game-directives" / "SYSTEM_PROMPT.md"
        )
        if prompt_path.exists():
            return prompt_path.read_text()
        logger.warning(f"System prompt not found: {prompt_path}")
        return "You are an AI agent playing a browser-based game. Use the available tools to explore and play."

    def _load_base_prompt(self) -> str:
        if self.prompt_optimizer:
            return self.prompt_optimizer.get_current_prompt()
        base_path = (
            Path(__file__).parent
            / "prompts"
            / "browser-game-directives"
            / "BASE_ORCHESTRATOR_POLICY.md"
        )
        if base_path.exists():
            return base_path.read_text()
        return "# Strategic Guidance\nExplore the game, learn mechanics, store discoveries in memory."

    # ------------------------------------------------------------------
    # Tool declarations
    # ------------------------------------------------------------------

    def _create_tool_declarations(self) -> list:
        tools = [
            {
                "name": "press_keys",
                "description": (
                    "Press keyboard keys to interact with the game. "
                    "Common keys: ArrowUp, ArrowDown, ArrowLeft, ArrowRight, w, a, s, d, "
                    "Space, Enter, Escape, Tab, Shift, e, q, r, f, 1-9. "
                    "Pass multiple keys to press them in sequence."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "keys": {
                            "type_": "ARRAY",
                            "items": {"type_": "STRING"},
                            "description": "List of keys to press in sequence",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are pressing these keys",
                        },
                    },
                    "required": ["keys", "reasoning"],
                },
            },
            {
                "name": "mouse_click",
                "description": (
                    "Click at (x, y) coordinates on the game canvas. "
                    "Use for menu buttons, UI elements, or point-and-click interactions. "
                    "Coordinates are relative to the game canvas (0,0 = top-left). "
                    "PREFER click_element when the target has a name — "
                    "click_element is much more accurate because it uses a "
                    "vision model to find the element for you. Only use raw "
                    "mouse_click when click_element fails or when you already "
                    "know the exact coordinates from a prior step."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "x": {
                            "type_": "INTEGER",
                            "description": "X coordinate (pixels from left edge)",
                        },
                        "y": {
                            "type_": "INTEGER",
                            "description": "Y coordinate (pixels from top edge)",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are clicking here",
                        },
                    },
                    "required": ["x", "y", "reasoning"],
                },
            },
            {
                "name": "click_element",
                "description": (
                    "Click an on-screen element identified by a "
                    "natural-language description. The harness sends the "
                    "current canvas screenshot plus your description to a "
                    "vision model (MolmoWeb) which returns the pixel "
                    "coordinates of the matching element, then dispatches "
                    "the click for you. PREFER this over mouse_click(x, y) "
                    "whenever you know what to click but not the exact "
                    "coordinates — it's much more accurate than guessing "
                    "pixel positions in canvases.\n"
                    "\n"
                    "Good descriptions are imperative and visually specific:\n"
                    "  - 'the START button at the bottom of the title screen'\n"
                    "  - 'the leftmost folder icon on the desktop'\n"
                    "  - 'the FOLDER DUNGEON title text at the top'\n"
                    "  - 'the red triangle in the upper right'\n"
                    "  - 'the inventory tab labeled WEAPONS'\n"
                    "\n"
                    "If click_element fails (returns success=false because "
                    "MolmoWeb couldn't locate the element), reword your "
                    "description with more visual detail or fall back to "
                    "mouse_click(x, y) with your best estimate."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "description": {
                            "type_": "STRING",
                            "description": (
                                "Natural-language description of the element "
                                "to click. Be visually specific."
                            ),
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are clicking this element",
                        },
                    },
                    "required": ["description", "reasoning"],
                },
            },
            {
                "name": "double_click",
                "description": (
                    "Double-click at (x, y) coordinates on the game canvas. "
                    "Use for opening folders, files, or any element that requires "
                    "a double-click (common in desktop/OS-themed games)."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "x": {
                            "type_": "INTEGER",
                            "description": "X coordinate (pixels from left edge)",
                        },
                        "y": {
                            "type_": "INTEGER",
                            "description": "Y coordinate (pixels from top edge)",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are double-clicking here",
                        },
                    },
                    "required": ["x", "y", "reasoning"],
                },
            },
            {
                "name": "hold_key",
                "description": (
                    "Hold a key down for a specified duration. "
                    "Useful for continuous movement or charging actions."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "key": {
                            "type_": "STRING",
                            "description": "Key to hold (e.g., ArrowRight, w, Space)",
                        },
                        "duration_ms": {
                            "type_": "INTEGER",
                            "description": "How long to hold in milliseconds",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are holding this key",
                        },
                    },
                    "required": ["key", "duration_ms", "reasoning"],
                },
            },
            {
                "name": "mouse_move",
                "description": (
                    "Move the mouse cursor to (x, y) on the game canvas WITHOUT clicking. "
                    "Use for hover-driven UI: tooltips, paddle/cursor-following games, "
                    "mouse-look in 3D games, or any game that reacts to mousemove events. "
                    "Coordinates are relative to the game canvas (0,0 = top-left)."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "x": {
                            "type_": "INTEGER",
                            "description": "X coordinate (pixels from left edge)",
                        },
                        "y": {
                            "type_": "INTEGER",
                            "description": "Y coordinate (pixels from top edge)",
                        },
                        "steps": {
                            "type_": "INTEGER",
                            "description": (
                                "Number of intermediate mousemove events (default 8). "
                                "Higher = smoother motion for games that animate during the move."
                            ),
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are moving the cursor here",
                        },
                    },
                    "required": ["x", "y", "reasoning"],
                },
            },
            {
                "name": "mouse_drag",
                "description": (
                    "Press at (x1, y1), drag to (x2, y2), release. "
                    "Use for drag-to-aim, dragging items in inventories, sliders, "
                    "drawing, or any drag interaction. Coordinates are canvas-relative."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "x1": {"type_": "INTEGER", "description": "Start X (pixels from left)"},
                        "y1": {"type_": "INTEGER", "description": "Start Y (pixels from top)"},
                        "x2": {"type_": "INTEGER", "description": "End X (pixels from left)"},
                        "y2": {"type_": "INTEGER", "description": "End Y (pixels from top)"},
                        "steps": {
                            "type_": "INTEGER",
                            "description": (
                                "Number of intermediate mousemove events during drag (default 12). "
                                "Higher = smoother for games that sample cursor continuously."
                            ),
                        },
                        "hold_ms": {
                            "type_": "INTEGER",
                            "description": "Delay between mousedown and the first move (default 50ms)",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are performing this drag",
                        },
                    },
                    "required": ["x1", "y1", "x2", "y2", "reasoning"],
                },
            },
            {
                "name": "key_down",
                "description": (
                    "Press a keyboard key WITHOUT releasing it. The key stays "
                    "held across agent steps until you call key_up with the "
                    "same key (or BrowserEnv stops / navigation recovers). "
                    "Use for games where holding direction keys is the natural "
                    "input model — Flappy Bird's hold-to-flap, racing games, "
                    "platformers with continuous run, etc. The set of "
                    "currently held keys is shown in your prompt every step."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "key": {
                            "type_": "STRING",
                            "description": "Key to hold (e.g. ArrowRight, w, Space)",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are pressing and holding this key",
                        },
                    },
                    "required": ["key", "reasoning"],
                },
            },
            {
                "name": "key_up",
                "description": (
                    "Release a key that was previously held with key_down. "
                    "Always release keys you no longer need — leaving them "
                    "held will affect every subsequent step until you do."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "key": {
                            "type_": "STRING",
                            "description": "Key to release (must match a prior key_down)",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are releasing this key now",
                        },
                    },
                    "required": ["key", "reasoning"],
                },
            },
            {
                "name": "wait_ms",
                "description": (
                    "Let game time pass without taking any other action. "
                    "In virtual-time mode this advances the game's internal "
                    "clock by duration_ms — animations play, timers fire, "
                    "physics ticks, then everything re-pauses for your next "
                    "decision. Use this when you need to wait on something "
                    "(animation finishing, falling platform reaching the "
                    "right spot, enemy moving into range, dialogue auto-"
                    "advancing) WITHOUT consuming a step on a primitive "
                    "action. The reasoning field MUST explain what you are "
                    "waiting for."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "duration_ms": {
                            "type_": "INTEGER",
                            "description": (
                                "How many ms of game time to let pass. Cap "
                                "30000 (30s of game time). Typical values: "
                                "100-500 for animations, 1000-3000 for "
                                "longer waits like dialogue or fall timers."
                            ),
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": (
                                "What you are waiting for — an animation, "
                                "an enemy approach, a fall timer, etc."
                            ),
                        },
                    },
                    "required": ["duration_ms", "reasoning"],
                },
            },
        ]

        # Memory / Skill / Subagent CRUD tools.
        # These were previously left undeclared and worked only because Gemini
        # hallucinated their names from training data and the unvalidated
        # dispatcher silently routed them. Now they're declared properly so
        # the new allow-list reject in _execute_function_call doesn't block
        # them. Required for the autoevolve loop and for any agent that
        # builds its own toolkit.
        tools.append({
            "name": "process_memory",
            "description": (
                "Read, add, update, or delete entries in long-term memory. "
                "Use this to record game controls, mechanics, level info, "
                "enemy patterns, item locations, and anything else you "
                "discover through gameplay. The current memory tree is "
                "shown in the LONG-TERM MEMORY OVERVIEW section of your "
                "prompt — pick an action based on what you need to do."
            ),
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "action": {
                        "type_": "STRING",
                        "enum": ["read", "add", "update", "delete"],
                        "description": "What to do with the entries",
                    },
                    "entries": {
                        "type_": "ARRAY",
                        "items": {"type_": "OBJECT"},
                        "description": (
                            "Per-action shape: read=[{id}], "
                            "add=[{path,title,content,importance,id?}], "
                            "update=[{id,title?,content?,path?,importance?}], "
                            "delete=[{id}]."
                        ),
                    },
                    "reasoning": {
                        "type_": "STRING",
                        "description": "Why you are reading/writing this memory",
                    },
                },
                "required": ["action", "entries", "reasoning"],
            },
        })
        tools.append({
            "name": "process_skill",
            "description": (
                "Read, add, update, or delete entries in the skill library. "
                "Skills can be either text guidance or executable Python (set "
                "the 'code' field). The current skill tree is shown in the "
                "SKILL LIBRARY section of your prompt — entries with the "
                "[run_skill] tag are executable and you should call them via "
                "the run_skill tool instead of writing the same code again."
            ),
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "action": {
                        "type_": "STRING",
                        "enum": ["read", "add", "update", "delete"],
                        "description": "What to do with the entries",
                    },
                    "entries": {
                        "type_": "ARRAY",
                        "items": {"type_": "OBJECT"},
                        "description": (
                            "Per-action shape: read=[{id}], "
                            "add=[{path,name,description,code?,effectiveness?,importance?,id?}], "
                            "update=[{id,name?,description?,code?,effectiveness?,path?}], "
                            "delete=[{id}]. Always include the 'code' field "
                            "when adding an executable skill or run_skill will "
                            "reject it."
                        ),
                    },
                    "reasoning": {
                        "type_": "STRING",
                        "description": "Why you are creating/updating this skill",
                    },
                },
                "required": ["action", "entries", "reasoning"],
            },
        })
        tools.append({
            "name": "process_subagent",
            "description": (
                "Read, add, update, or delete entries in the subagent registry. "
                "Use this to register named multi-step routines (combat handlers, "
                "exploration loops, UI sequences) that you can later invoke via "
                "execute_custom_subagent. The current registry is shown in the "
                "SUBAGENT REGISTRY section of your prompt."
            ),
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "action": {
                        "type_": "STRING",
                        "enum": ["read", "add", "update", "delete"],
                        "description": "What to do with the entries",
                    },
                    "entries": {
                        "type_": "ARRAY",
                        "items": {"type_": "OBJECT"},
                        "description": (
                            "Per-action shape: read=[{id}], "
                            "add=[{path,name,description,handler_type,max_turns,available_tools,system_instructions,directive,return_condition,importance?,id?}], "
                            "update=[{id,...}], delete=[{id}]."
                        ),
                    },
                    "reasoning": {
                        "type_": "STRING",
                        "description": "Why you are creating/updating this subagent",
                    },
                },
                "required": ["action", "entries", "reasoning"],
            },
        })
        tools.append({
            "name": "run_skill",
            "description": (
                "Execute a saved skill's code in a sandbox. Use this BEFORE "
                "reaching for primitive actions if any skill in the SKILL "
                "LIBRARY matches what you want to do. The sandbox exposes "
                "tools['press_keys']/['mouse_click']/['double_click']/['hold_key']"
                "/['mouse_move']/['mouse_drag']/['get_game_state']/['process_memory'] "
                "and the args dict you pass in."
            ),
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "skill_id": {
                        "type_": "STRING",
                        "description": "ID of the skill to run (from SKILL LIBRARY)",
                    },
                    "args": {
                        "type_": "OBJECT",
                        "description": (
                            "Arguments passed to the skill code as the 'args' "
                            "dict. Inspect the skill's description to know "
                            "what fields it expects."
                        ),
                    },
                    "reasoning": {
                        "type_": "STRING",
                        "description": "Why you are running this skill now",
                    },
                },
                "required": ["skill_id", "reasoning"],
            },
        })
        tools.append({
            "name": "run_code",
            "description": (
                "Execute Python code in a read-only sandbox to inspect game "
                "state and prototype skill code. Has access to "
                "tools['get_game_state']() but NOT to action tools — use this "
                "only for development/debugging. Once code works, save it as "
                "a skill via process_skill (with the 'code' field set) and "
                "invoke it via run_skill going forward."
            ),
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "code": {
                        "type_": "STRING",
                        "description": "Python code to execute (sets a 'result' variable)",
                    },
                    "args": {
                        "type_": "OBJECT",
                        "description": "Optional args dict passed as 'args' to the code",
                    },
                    "reasoning": {
                        "type_": "STRING",
                        "description": "Why you are running this code",
                    },
                },
                "required": ["code", "reasoning"],
            },
        })

        # Local subagent tools (from shared declarations).
        # Exclude process_trajectory_history — it runs without screenshots and
        # causes the VLM to hallucinate.  The agent already sees the screenshot
        # in its main prompt each step.
        tools.extend(
            t for t in build_local_subagent_tool_declarations(include_builtins=False)
            if t["name"] != "process_trajectory_history"
        )

        # Replan objectives — used sparingly (the prompt enforces this).
        tools.append(REPLAN_OBJECTIVES_TOOL_DECLARATION)

        # Evolve harness
        tools.append(
            {
                "name": "evolve_harness",
                "description": (
                    "Trigger an evolution pass NOW to improve skills, subagents, and memory "
                    "based on recent performance."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "reasoning": {
                            "type_": "STRING",
                            "description": "What needs improvement and why",
                        },
                        "num_steps": {
                            "type_": "INTEGER",
                            "description": "Number of recent trajectory steps to analyze (default 50)",
                        },
                    },
                    "required": ["reasoning"],
                },
            }
        )

        logger.info(f"Created {len(tools)} tool declarations")
        return tools

    # ------------------------------------------------------------------
    # Function execution
    # ------------------------------------------------------------------

    def _execute_function_call_by_name(self, function_name: str, arguments: dict) -> str:
        if is_local_subagent_tool(function_name):
            spec = get_local_subagent_spec(function_name)
            return getattr(self, spec.handler_method)(arguments)

        if function_name == "run_skill":
            result_json = self._execute_run_skill(arguments)
            self._store_function_result_for_context("run_skill", result_json)
            return result_json
        if function_name == "run_code":
            result_json = self._execute_run_code(arguments)
            self._store_function_result_for_context("run_code", result_json)
            return result_json
        if function_name == "evolve_harness":
            result_json = self._execute_evolve_harness(arguments)
            self._store_function_result_for_context("evolve_harness", result_json)
            return result_json

        result = self.mcp_adapter.call_tool(function_name, arguments)
        return json.dumps(result, indent=2, default=str)

    def _execute_function_call(self, function_call, allowed_tool_names=None):
        name = function_call.name
        if allowed_tool_names and name not in allowed_tool_names:
            return json.dumps({"success": False, "error": f"Tool {name} not allowed in this context"})

        # Allow-list against declared tools so the model can't hallucinate
        # API names like "get_game_state" and have them silently succeed via
        # the unvalidated MCP dispatch below. Without this guard, gemma4 was
        # emitting ~0.4 get_game_state calls per step on top of the screenshot
        # we already include in the prompt — pure waste.
        declared_names = {t["name"] for t in (self.tools or [])}
        if declared_names and name not in declared_names:
            err = (
                f"Tool '{name}' is not a declared tool. "
                f"Available tools: {sorted(declared_names)}. "
                f"The current screenshot is already in your prompt — you do "
                f"not need to fetch state separately."
            )
            logger.warning(f"Rejected hallucinated tool call: {name}")
            return json.dumps({"success": False, "error": err})
        args = convert_protobuf_args(function_call.args) if hasattr(function_call, "args") else {}
        return self._execute_function_call_by_name(name, args)

    def _convert_protobuf_args(self, args) -> dict:
        return convert_protobuf_args(args)

    def _execute_run_skill(self, arguments: dict) -> str:
        skill_id = arguments.get("skill_id", "")
        reasoning = arguments.get("reasoning", "")
        skill_args = arguments.get("args", {})

        from utils.stores.skills import get_skill_store

        store = get_skill_store()
        entry = store.get(skill_id)
        if entry is None:
            return json.dumps({"success": False, "error": f"Skill {skill_id} not found"})

        code = getattr(entry, "code", "")
        if not code or not code.strip():
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill {skill_id} has no executable code.",
                }
            )

        if not skill_args:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Skill {skill_id} requires arguments in the 'args' field.",
                }
            )

        def _tool_caller(tool_name):
            def call(**kwargs):
                return self.mcp_adapter.call_tool(tool_name, kwargs)
            return call

        sandbox_tools = {}
        for tool_name in (
            "press_keys", "mouse_click", "click_element", "double_click",
            "hold_key", "mouse_move", "mouse_drag",
            "key_down", "key_up", "wait_ms",
            "get_game_state", "process_memory",
        ):
            sandbox_tools[tool_name] = _tool_caller(tool_name)

        import random, collections, math, re as _re_mod, heapq, itertools, functools, base64 as _base64
        from PIL import Image as _PILImage, ImageDraw as _PILImageDraw, ImageFilter as _PILImageFilter
        try:
            import cv2 as _cv2
        except ImportError:
            _cv2 = None
        from io import BytesIO as _BytesIO

        def _decode_screenshot(b64_str):
            if not b64_str:
                return None
            return _PILImage.open(_BytesIO(_base64.b64decode(b64_str))).convert("RGB")

        sandbox_globals = {
            "__builtins__": {
                "range": range, "len": len, "int": int, "float": float,
                "str": str, "list": list, "dict": dict, "tuple": tuple,
                "set": set, "frozenset": frozenset, "type": type,
                "bool": bool, "print": print, "abs": abs, "min": min,
                "max": max, "sum": sum, "enumerate": enumerate, "zip": zip,
                "sorted": sorted, "reversed": reversed, "isinstance": isinstance,
                "map": map, "filter": filter, "any": any, "all": all,
                "__import__": __import__,
                "True": True, "False": False, "None": None,
            },
            "random": random,
            "collections": collections,
            "math": math,
            "json": json,
            "re": _re_mod,
            "heapq": heapq,
            "itertools": itertools,
            "functools": functools,
            "time": time,
            "np": np,
            "numpy": np,
            "base64": _base64,
            # Image libs — see _execute_run_code for the same set
            "Image": _PILImage,
            "ImageDraw": _PILImageDraw,
            "ImageFilter": _PILImageFilter,
            "PIL": _PILImage,
            "cv2": _cv2,
            "BytesIO": _BytesIO,
            "decode_screenshot": _decode_screenshot,
            "tools": sandbox_tools,
            "args": skill_args or {},
        }

        logger.info(f"Running skill {skill_id}: {reasoning}")
        try:
            exec(code, sandbox_globals)  # noqa: S102
            result = sandbox_globals.get("result", "Skill executed successfully")
            return json.dumps({"success": True, "skill_id": skill_id, "result": result})
        except Exception as e:
            logger.error(f"Skill {skill_id} execution FAILED: {e}", exc_info=True)
            return json.dumps({"success": False, "skill_id": skill_id, "error": str(e)})

    def _execute_run_code(self, arguments: dict) -> str:
        code = arguments.get("code", "")
        reasoning = arguments.get("reasoning", "")
        user_args = arguments.get("args", {})

        if not code.strip():
            return json.dumps({"success": False, "error": "No code provided"})

        def _read_only_tool(tool_name):
            def call(**kwargs):
                return self.mcp_adapter.call_tool(tool_name, kwargs)
            return call

        sandbox_tools = {
            "get_game_state": _read_only_tool("get_game_state"),
        }

        import random, collections, math, re as _re_mod, heapq, itertools, functools, base64 as _base64
        # Image processing libs for visual analysis. PIL and cv2 (opencv)
        # let the agent decode the screenshot it gets from
        # tools['get_game_state']()['screenshot_base64'] and run real
        # vision code — pixel scans, template matching, contour
        # detection — to find UI elements that the VLM can't always
        # locate by eye alone.
        from PIL import Image as _PILImage, ImageDraw as _PILImageDraw, ImageFilter as _PILImageFilter
        try:
            import cv2 as _cv2
        except ImportError:
            _cv2 = None
        from io import BytesIO as _BytesIO

        captured_stdout = []
        def _print(*args, **kwargs):
            captured_stdout.append(" ".join(str(a) for a in args))

        def _decode_screenshot(b64_str):
            """Helper: decode the base64 PNG returned by get_game_state
            into a PIL Image. Skill code can call this to avoid the
            base64 boilerplate every time."""
            if not b64_str:
                return None
            return _PILImage.open(_BytesIO(_base64.b64decode(b64_str))).convert("RGB")

        sandbox_globals = {
            "__builtins__": {
                "range": range, "len": len, "int": int, "float": float,
                "str": str, "list": list, "dict": dict, "tuple": tuple,
                "set": set, "frozenset": frozenset, "type": type,
                "bool": bool, "print": _print, "abs": abs, "min": min,
                "max": max, "sum": sum, "enumerate": enumerate, "zip": zip,
                "sorted": sorted, "reversed": reversed, "isinstance": isinstance,
                "map": map, "filter": filter, "any": any, "all": all,
                "__import__": __import__,
                "True": True, "False": False, "None": None,
            },
            "random": random,
            "collections": collections,
            "math": math,
            "json": json,
            "re": _re_mod,
            "heapq": heapq,
            "itertools": itertools,
            "functools": functools,
            "time": time,
            "np": np,
            "numpy": np,
            "base64": _base64,
            # Image libs for visual analysis
            "Image": _PILImage,
            "ImageDraw": _PILImageDraw,
            "ImageFilter": _PILImageFilter,
            "PIL": _PILImage,  # convenience alias
            "cv2": _cv2,  # may be None if opencv-headless not installed
            "BytesIO": _BytesIO,
            "decode_screenshot": _decode_screenshot,
            "tools": sandbox_tools,
            "args": user_args or {},
        }

        logger.info(f"Running code: {reasoning}")
        try:
            exec(code, sandbox_globals)  # noqa: S102
            result = sandbox_globals.get("result", None)
            return json.dumps({
                "success": True,
                "result": result,
                "stdout": "\n".join(captured_stdout) if captured_stdout else "",
            }, default=str)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": "\n".join(captured_stdout) if captured_stdout else "",
            })

    def _execute_evolve_harness(self, arguments: dict) -> str:
        reasoning = arguments.get("reasoning", "")
        num_steps = int(arguments.get("num_steps", 50))

        if not self.harness_evolver:
            return json.dumps({"success": False, "error": "HarnessEvolver not available"})

        logger.info(f"Orchestrator requested evolution: {reasoning}")
        try:
            results = self.harness_evolver.evolve(
                current_step=self.step_count,
                num_trajectory_steps=num_steps,
            )
            return json.dumps({"success": True, "evolution_results": results}, default=str)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    # ------------------------------------------------------------------
    # Subagent VLM management
    # ------------------------------------------------------------------

    def _get_subagent_vlm(
        self,
        tool_names: Optional[set] = None,
        supplemental_tools: Optional[list] = None,
    ):
        """Lazily create cached VLMs for subagents.

        - Tool-less subagents (reflect/verify/summarize): cached bare VLM
          via ``self._local_subagent_vlm``.
        - ``execute_custom_subagent``: VLM with the requested tool subset
          (intersected with the orchestrator's declared tools) plus any
          ``supplemental_tools`` injected by the executor (e.g. the
          synthetic ``return_to_orchestrator`` tool). Cached by
          (tool_names, supplemental tool names) tuple with LRU eviction.

        Mirrors PokeAgent's signature so the shared SubagentExecutor can
        call this without branching on agent type. Previously the browser
        agent had two clashing definitions of this method (the second one
        silently shadowed the first via Python's late-binding) and neither
        accepted supplemental_tools, so every execute_custom_subagent
        call crashed with TypeError before this fix.
        """
        normalized = tuple(sorted(tool_names or ()))
        supp_key = tuple(t["name"] for t in (supplemental_tools or []))

        # Bare VLM for tool-less one-shot subagents
        if not normalized and not supp_key:
            if self._local_subagent_vlm is None:
                self._local_subagent_vlm = VLM(
                    backend=self.backend,
                    model_name=self.model,
                    tools=None,
                )
            return self._local_subagent_vlm

        cache_key = (normalized, supp_key)
        cached = self._subagent_vlm_cache.get(cache_key)
        if cached is not None:
            self._subagent_vlm_cache.move_to_end(cache_key)
            return cached

        allowed = [t for t in self.tools if t.get("name") in set(normalized)]
        if supplemental_tools:
            allowed.extend(supplemental_tools)

        cached = VLM(
            backend=self.backend,
            model_name=self.model,
            tools=allowed or None,
            system_instruction=self.system_instructions or None,
        )
        self._subagent_vlm_cache[cache_key] = cached
        self._subagent_vlm_cache.move_to_end(cache_key)
        while len(self._subagent_vlm_cache) > self._VLM_CACHE_CAP:
            self._subagent_vlm_cache.popitem(last=False)
        return cached

    # ------------------------------------------------------------------
    # Local subagent handlers (called via registry dispatch)
    # ------------------------------------------------------------------

    def _run_one_step_subagent(
        self,
        *,
        prompt: str,
        interaction_name: str,
        current_image=None,
        orchestrator_args=None,
        metrics_arg_keys=None,
    ) -> tuple:
        """Run a one-step local subagent with its own claimed global step."""
        step_number = self.runtime.claim_step(owner="subagent", interaction_name=interaction_name)
        subagent_vlm = self._get_subagent_vlm()
        if current_image is not None:
            response = subagent_vlm.get_query(current_image, prompt, interaction_name)
        else:
            response = subagent_vlm.get_text_query(prompt, interaction_name)
        text = self._extract_text_from_response(response)
        if text:
            return step_number, text
        raise RuntimeError(f"{interaction_name} did not return text output")

    def _execute_custom_subagent(self, arguments: dict) -> str:
        """Launch a custom subagent (from registry or inline config)."""
        return self.executor.execute_custom_subagent(arguments)

    def _execute_process_trajectory_history(self, arguments: dict) -> str:
        """One-step VLM analysis on a trajectory window with a directive."""
        return self.executor.process_trajectory_history(arguments)

    # ------------------------------------------------------------------
    # History & context helpers
    # ------------------------------------------------------------------

    def _add_to_history(
        self,
        step: int,
        llm_response: str,
        tool_calls: list,
        start_coords: tuple = None,
        end_coords: tuple = None,
    ):
        entry = {
            "step": step,
            "llm_response": llm_response,
            "timestamp": time.time(),
            "tool_calls": tool_calls,
            "start_coords": start_coords,
            "end_coords": end_coords,
        }
        self.conversation_history.append(entry)
        # Keep window
        if len(self.conversation_history) > ACTION_HISTORY_WINDOW * 2:
            self.conversation_history = self.conversation_history[-ACTION_HISTORY_WINDOW * 2 :]

    def _store_function_result_for_context(self, function_name: str, result_json: str):
        self.recent_function_results.append(
            {"function_name": function_name, "result": result_json, "timestamp": time.time()}
        )
        if len(self.recent_function_results) > 3:
            self.recent_function_results = self.recent_function_results[-3:]

    def _get_function_results_context(self) -> str:
        if not self.recent_function_results:
            return ""
        lines = ["\n" + "=" * 70, "RESULTS FROM PREVIOUS STEP:", "=" * 70]
        for entry in self.recent_function_results:
            lines.append(f"\nFunction: {entry['function_name']}")
            result = entry["result"]
            if len(result) > 10000:
                lines.append(result[:10000] + "\n... (truncated)")
            else:
                lines.append(result)
            lines.append("")
        lines.append("=" * 70)
        self.recent_function_results = []
        return "\n".join(lines)

    def _get_memory_context(self) -> str:
        try:
            result = self.mcp_adapter.call_tool("get_memory_overview", {})
            if result.get("success"):
                overview = result.get("overview", "")
                if overview and overview.strip() not in ("No memory entries yet.", "No knowledge entries yet."):
                    return overview
            return "No memory entries yet."
        except Exception:
            return "Long-term memory temporarily unavailable."

    def _get_skill_context(self) -> str:
        try:
            result = self.mcp_adapter.call_tool("get_skill_overview", {})
            if result.get("success"):
                overview = result.get("overview", "")
                if overview and overview.strip() != "No skills learned yet.":
                    return overview
            return "No skills learned yet."
        except Exception:
            return "No skills learned yet."

    def _get_subagent_context(self) -> str:
        try:
            result = self.mcp_adapter.call_tool("get_subagent_overview", {})
            if result.get("success"):
                overview = result.get("overview", "")
                if overview and overview.strip() != "No subagents registered yet.":
                    return overview
            return "No subagents registered yet."
        except Exception:
            return "No subagents registered yet."

    def _format_action_history(self) -> str:
        """Render the agent's recent steps with both calls AND results.

        Critical: this MUST include the ``result`` field of each tool call,
        not just the call. Without results in the persistent history, the
        agent can't tell what its own previous actions accomplished, and
        loops forever re-trying the same thing because every step looks
        identical to the previous one. We learned this the hard way on
        the first 1k Folder Dungeon run — the agent clicked the same
        chest icon 26 times in a row because the click_element results
        weren't visible past one step.

        Result rendering is per-tool because different tools need
        different summaries:
          - click_element / mouse_click → coordinates clicked + the
            molmoweb thought (what the click oracle thought it was
            clicking on)
          - run_code → captured stdout + the ``result`` variable, which
            is the only way the agent gets to inspect game state from
            inside its sandbox
          - process_memory / process_skill / process_subagent → success
            flag + summary line
          - everything else → first ~250 chars of the result JSON

        Total per-tool budget: ~400 chars. With ACTION_HISTORY_WINDOW=10
        and ~2 tools per step, that's ~8 KB of action history in the
        prompt. Comfortable for our 32K context budget.
        """
        if not self.conversation_history:
            return "No previous actions recorded."
        recent = self.conversation_history[-ACTION_HISTORY_WINDOW:]
        lines = []
        for entry in recent:
            step = entry.get("step", "?")
            llm_response = entry.get("llm_response", "").strip()
            lines.append(f"[Step {step}]")
            if llm_response:
                # Truncate llm_response to ~300 chars so a verbose
                # gemma "thinking" turn doesn't dominate the budget.
                trimmed = llm_response[:300]
                if len(llm_response) > 300:
                    trimmed += "..."
                lines.append(f"  THINKING: {trimmed}")
            tool_calls = entry.get("tool_calls", [])
            if tool_calls:
                lines.append("  TOOLS:")
                for tc in tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    result = tc.get("result", "")
                    lines.append(f"    - {name}")
                    lines.append(
                        f"      args: {json.dumps(args, ensure_ascii=False)[:300]}"
                    )
                    summary = self._summarize_tool_result(name, result)
                    if summary:
                        lines.append(f"      result: {summary}")
            lines.append("")
        return "\n".join(lines).strip()

    def _summarize_tool_result(self, name: str, result) -> str:
        """Render a per-tool result summary for the action history view.

        Tools have different "interesting" fields. We extract the most
        actionable subset per tool so the agent can see what changed
        without us pasting full JSON dumps into every history entry.
        Falls back to a 250-char prefix of the JSON for unknown tools.
        """
        if not result:
            return ""

        # Coerce dicts and JSON strings to a uniform dict view.
        if isinstance(result, str):
            try:
                obj = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                # Not JSON — treat as opaque string and truncate.
                return result[:250].replace("\n", " ")
        elif isinstance(result, dict):
            obj = result
        else:
            return str(result)[:250].replace("\n", " ")

        if not isinstance(obj, dict):
            return str(obj)[:250].replace("\n", " ")

        success = obj.get("success", True)
        success_marker = "OK" if success else "FAIL"

        if name == "click_element":
            if not success:
                err = obj.get("error", "")
                thought = obj.get("molmoweb_thought", "")
                return f"FAIL — {err[:100]}" + (f" | thought: {thought[:100]}" if thought else "")
            clicked = obj.get("clicked", {})
            thought = obj.get("molmoweb_thought", "")
            x = clicked.get("x") if isinstance(clicked, dict) else None
            y = clicked.get("y") if isinstance(clicked, dict) else None
            parts = [f"OK clicked at ({x},{y})"]
            if thought:
                parts.append(f"oracle: {thought[:150]}")
            return " | ".join(parts)

        if name == "mouse_click":
            if not success:
                return f"FAIL — {obj.get('error', '')[:200]}"
            clicked = obj.get("clicked", {})
            x = clicked.get("x") if isinstance(clicked, dict) else None
            y = clicked.get("y") if isinstance(clicked, dict) else None
            return f"OK clicked at ({x},{y})"

        if name == "press_keys":
            if not success:
                return f"FAIL — {obj.get('error', '')[:200]}"
            keys = obj.get("keys_pressed") or obj.get("keys") or []
            return f"OK pressed {keys}"

        if name == "double_click":
            if not success:
                return f"FAIL — {obj.get('error', '')[:200]}"
            dc = obj.get("double_clicked", {})
            x = dc.get("x") if isinstance(dc, dict) else None
            y = dc.get("y") if isinstance(dc, dict) else None
            return f"OK double-clicked at ({x},{y})"

        if name == "hold_key":
            if not success:
                return f"FAIL — {obj.get('error', '')[:200]}"
            key = obj.get("key_held") or obj.get("key", "?")
            dur = obj.get("duration_ms", "?")
            return f"OK held {key} for {dur}ms"

        if name == "mouse_move":
            if not success:
                return f"FAIL — {obj.get('error', '')[:200]}"
            mt = obj.get("moved_to", {})
            x = mt.get("x") if isinstance(mt, dict) else None
            y = mt.get("y") if isinstance(mt, dict) else None
            return f"OK moved to ({x},{y})"

        if name == "mouse_drag":
            if not success:
                return f"FAIL — {obj.get('error', '')[:200]}"
            d = obj.get("dragged", {})
            return f"OK dragged {d.get('from')} -> {d.get('to')}" if isinstance(d, dict) else "OK"

        if name in ("key_down", "key_up"):
            if not success:
                return f"FAIL — {obj.get('error', '')[:200]}"
            # Server uses key_down/key_up as response field names
            key = obj.get(name) or obj.get("key", "?")
            return f"OK {name}({key})"

        if name == "wait_ms":
            return f"OK waited {obj.get('duration_ms', '?')}ms"

        if name == "run_code":
            # The most important result type. Show captured stdout +
            # the `result` variable so the agent can actually use
            # what it computed.
            if not success:
                return f"FAIL — {obj.get('error', '')[:300]}"
            stdout = obj.get("stdout", "") or obj.get("output", "")
            res_var = obj.get("result")
            parts = ["OK"]
            if stdout:
                stdout_trim = stdout[:400]
                if len(stdout) > 400:
                    stdout_trim += "...[truncated]"
                parts.append(f"stdout: {stdout_trim}")
            if res_var is not None:
                rv_str = json.dumps(res_var, default=str)[:300]
                parts.append(f"result var: {rv_str}")
            return " | ".join(parts)

        if name == "run_skill":
            if not success:
                return f"FAIL — {obj.get('error', '')[:300]}"
            stdout = obj.get("stdout", "") or obj.get("output", "")
            res_var = obj.get("result")
            parts = ["OK"]
            if stdout:
                parts.append(f"stdout: {stdout[:300]}")
            if res_var is not None:
                parts.append(f"result: {json.dumps(res_var, default=str)[:200]}")
            return " | ".join(parts)

        if name in ("process_memory", "process_skill", "process_subagent", "replan_objectives"):
            if not success:
                return f"FAIL — {obj.get('error', '')[:200]}"
            # These have action-specific summaries; just confirm they
            # took effect with a short note.
            action = obj.get("action") or obj.get("operation") or ""
            count = obj.get("count") or obj.get("entries_count") or ""
            extra = f"action={action} count={count}".strip()
            return f"OK {extra}".strip()

        if name in ("get_game_state", "get_memory_overview", "get_skill_overview", "get_subagent_overview"):
            # These are read-only and run automatically each step;
            # surfacing their results in history is noise.
            return ""

        # Fallback: 250-char prefix of the raw JSON.
        try:
            return f"{success_marker} " + json.dumps(obj, default=str)[:250]
        except Exception:
            return f"{success_marker} <unserializable>"

    def _calculate_context_size(self) -> int:
        size = 0
        for entry in self.conversation_history:
            size += len(entry.get("llm_response", ""))
            for tc in entry.get("tool_calls", []):
                size += len(json.dumps(tc.get("args", {})))
                size += len(str(tc.get("result", "")))
        return size

    def _log_trajectory_for_step(self, step, pre_state, tool_calls, reasoning, **kwargs):
        """Append one trajectory entry to trajectory_history.jsonl.

        HarnessEvolver / PromptOptimizer read this file each evolution cycle.
        Without it, auto-evolve fires but immediately bails with
        "No trajectories — skipping evolution", and the prompt/skill/subagent/
        memory passes never run. Format must match the schema in
        ``RunDataManager.log_trajectory`` so the optimizer can parse it.
        """
        run_manager = get_run_data_manager()
        if not run_manager:
            return
        try:
            def _strip(value):
                # Trim screenshots and oversize blobs from anything we persist —
                # the trajectory file is read every evolution cycle, so keeping
                # it small matters for both disk and prompt budgets.
                if isinstance(value, dict):
                    return {
                        k: _strip(v)
                        for k, v in value.items()
                        if k not in ("screenshot_base64", "screenshot")
                    }
                if isinstance(value, list):
                    return [_strip(v) for v in value]
                if isinstance(value, str) and len(value) > 4000:
                    return value[:4000] + "...[truncated]"
                return value

            sanitized_calls = [
                {
                    "name": tc.get("name"),
                    "args": _strip(tc.get("args") or {}),
                    "result": _strip(tc.get("result", "")),
                }
                for tc in (tool_calls or [])
            ]
            action = {
                "type": "tool_calls",
                "tool_calls": sanitized_calls,
                "total_tool_calls": len(sanitized_calls),
            }
            run_manager.log_trajectory(
                step=step,
                reasoning=reasoning or "",
                action=action,
                pre_state=pre_state or {},
                outcome={"success": True, "objectives_completed": []},
            )
        except Exception as e:
            logger.warning(f"Trajectory save error: {e}", exc_info=True)

    def _summarize_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        """One-line human-readable summary of a tool call for logging.

        Goal: when a call fails, the operator should be able to see what
        the agent was actually trying to do without grepping the prompt.
        For execute_custom_subagent / run_skill we resolve the id to a
        store name; for everything else we show the most informative
        args (reasoning, x/y, keys, etc.) plus a count of any extras.
        """
        if not isinstance(args, dict):
            return f"{name}({args!r})"

        reasoning = args.get("reasoning") or args.get("reason") or ""

        # Resolve store IDs to names so the log isn't all "sa_0008".
        if name == "execute_custom_subagent":
            sid = args.get("subagent_id") or args.get("id") or "?"
            sa_name = self._lookup_subagent_name(sid)
            label = f"{sid} ({sa_name})" if sa_name else sid
            directive = args.get("directive") or args.get("config", {}).get("directive", "")
            extra = f" directive={directive[:60]!r}" if directive else ""
            r = f" — {reasoning[:80]}" if reasoning else ""
            return f"execute_custom_subagent({label}){extra}{r}"

        if name == "run_skill":
            sid = args.get("skill_id", "?")
            sk_name = self._lookup_skill_name(sid)
            label = f"{sid} ({sk_name})" if sk_name else sid
            sk_args = args.get("args", {})
            r = f" — {reasoning[:80]}" if reasoning else ""
            return f"run_skill({label}, args={sk_args}){r}"

        # Default: show the most useful 2-3 fields
        useful_keys = [
            k for k in ("x", "y", "x1", "y1", "x2", "y2", "key", "keys",
                        "duration_ms", "skill_id", "subagent_id",
                        "action", "code")
            if k in args
        ][:3]
        kv = ", ".join(
            f"{k}={self._truncate_arg(args[k])}" for k in useful_keys
        )
        if reasoning:
            kv = f"{kv}{', ' if kv else ''}reasoning={reasoning[:80]!r}"
        if not kv:
            # Fall back to generic dump
            kv = ", ".join(f"{k}={self._truncate_arg(v)}"
                           for k, v in list(args.items())[:3])
        return f"{name}({kv})"

    def _truncate_arg(self, value: Any) -> str:
        """Compact repr of a tool arg for log lines."""
        if isinstance(value, str):
            return repr(value if len(value) < 60 else value[:57] + "...")
        if isinstance(value, (list, tuple)):
            inner = ", ".join(self._truncate_arg(v) for v in value[:5])
            extra = f", +{len(value)-5}" if len(value) > 5 else ""
            return f"[{inner}{extra}]"
        if isinstance(value, dict):
            return f"{{...{len(value)} keys}}"
        return str(value)

    def _lookup_subagent_name(self, sid: str) -> Optional[str]:
        try:
            from utils.stores.subagents import get_subagent_store
            entry = get_subagent_store().get(sid)
            return getattr(entry, "name", None) or getattr(entry, "title", None) if entry else None
        except Exception:
            return None

    def _lookup_skill_name(self, sid: str) -> Optional[str]:
        try:
            from utils.stores.skills import get_skill_store
            entry = get_skill_store().get(sid)
            return getattr(entry, "name", None) or getattr(entry, "title", None) if entry else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # VLM response handling
    # ------------------------------------------------------------------

    def _handle_vlm_function_calls(
        self,
        response,
        tool_calls_made,
        tool_call_count,
        max_tool_calls,
        allowed_tool_names=None,
    ):
        if not hasattr(response, "candidates") or not response.candidates:
            return False

        candidate = response.candidates[0]
        if not hasattr(candidate, "content") or not candidate.content:
            return False
        content = candidate.content
        if not hasattr(content, "parts"):
            return False

        function_calls_found = False
        for part in content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_call_count += 1

                # Pre-extract args once so we can log them on success AND
                # failure paths without re-parsing.
                try:
                    fc_args = convert_protobuf_args(fc.args) if hasattr(fc, "args") else {}
                except Exception:
                    fc_args = {}

                # Build a one-line summary of the call for the log so the
                # operator can see what the agent was *trying* to do, not
                # just the bare tool name. Includes reasoning if present
                # and resolves subagent_id / skill_id to a name when we
                # can look one up locally.
                summary = self._summarize_tool_call(fc.name, fc_args)
                logger.info(
                    f"Tool call ({tool_call_count}/{max_tool_calls}): {summary}"
                )

                try:
                    result = self._execute_function_call(
                        fc, allowed_tool_names=allowed_tool_names
                    )
                except Exception as e:
                    logger.error(
                        f"Tool {fc.name} FAILED ({type(e).__name__}: {e}) — call: {summary}"
                    )
                    tool_calls_made.append(
                        {
                            "name": fc.name,
                            "args": fc_args,
                            "result": json.dumps({"success": False, "error": str(e)}),
                        }
                    )
                    raise

                tool_calls_made.append(
                    {
                        "name": fc.name,
                        "args": convert_protobuf_args(fc.args),
                        "result": result,
                    }
                )
                function_calls_found = True

        return function_calls_found and len(tool_calls_made) > 0

    def _extract_text_from_response(self, response):
        if isinstance(response, str):
            return response.strip()
        try:
            if hasattr(response, "text"):
                return response.text.strip()
            if hasattr(response, "candidates") and response.candidates:
                parts = getattr(
                    getattr(response.candidates[0], "content", None), "parts", None
                )
                if parts:
                    texts = [p.text for p in parts if hasattr(p, "text") and p.text]
                    if texts:
                        return "\n".join(texts).strip()
            return ""
        except Exception:
            return ""

    def _is_black_frame(self, image) -> bool:
        """Detect a *broken* screenshot (pipeline failure), not a dim game.

        A real broken frame is uniform — usually all-zero bytes from a
        screenshot taken before the canvas had any content. A dim game
        scene (Flappy Bird at night, a fade-to-black, a dark dungeon)
        has high pixel variance and at least a few bright pixels even
        when the average is low.

        Heuristic: low std AND no bright pixels. Either signal alone is
        not enough — a uniform gray loading screen has low std but the
        mean isn't zero, and a dark game scene has bright pixels
        (HUD, sky highlights) so max stays well above the threshold.
        """
        arr = np.array(image)
        return arr.std() < 2.0 and arr.max() < 10

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, game_state_result: str, step_count: int) -> str:
        try:
            if isinstance(game_state_result, dict):
                data = game_state_result
            else:
                data = json.loads(game_state_result)
        except Exception:
            data = {}

        state_text = data.get("state_text", "Game state unavailable.")
        action_history = self._format_action_history()
        function_results = self._get_function_results_context()
        memory_ctx = self._get_memory_context()
        skill_ctx = self._get_skill_context()
        subagent_ctx = self._get_subagent_context()
        base_prompt = self._load_base_prompt()

        # Show the agent how long the session has been running. Useful
        # for time-aware decisions ("we've been at this 30 minutes,
        # try a different approach") and for the playtest report.
        elapsed_s = max(0.0, time.time() - getattr(self, "_session_start_time", time.time()))
        if elapsed_s < 60:
            elapsed_str = f"{elapsed_s:.0f}s"
        elif elapsed_s < 3600:
            elapsed_str = f"{int(elapsed_s // 60)}m {int(elapsed_s % 60)}s"
        else:
            h = int(elapsed_s // 3600); m = int((elapsed_s % 3600) // 60)
            elapsed_str = f"{h}h {m}m"
        max_steps_str = f"/{self.max_steps}" if self.max_steps else ""

        prompt = f"""# Current Step: {step_count}{max_steps_str}   Session runtime: {elapsed_str}

{base_prompt}

## CONTEXT FOR THIS STEP

### ACTION HISTORY (last {ACTION_HISTORY_WINDOW} steps):
{action_history}
{function_results}

### CURRENT GAME STATE:
{state_text}

### LONG-TERM MEMORY OVERVIEW
{memory_ctx}

### SKILL LIBRARY
{skill_ctx}

### SUBAGENT REGISTRY
{subagent_ctx}

Step {step_count}"""

        logger.info(f"Prompt: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")
        return prompt

    # ------------------------------------------------------------------
    # Prerequisites
    # ------------------------------------------------------------------

    def check_prerequisites(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            if resp.status_code == 200:
                logger.info("Server is healthy")
                return True
        except Exception:
            pass
        logger.error("Server not reachable")
        return False

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def run_step(
        self,
        prompt: str,
        max_tool_calls: int = 5,
        screenshot_b64: str = None,
        step_number: Optional[int] = None,
        game_info: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        try:
            preview_step = step_number if step_number is not None else self.runtime.peek_next_step()

            # Pre-state for trajectory.
            # Re-use the game_info that run() already fetched at the top of
            # the step instead of calling get_game_state a second time —
            # avoids ~1 redundant Playwright screenshot per step.
            run_manager = get_run_data_manager()
            pre_state = None
            if run_manager and game_info is not None:
                pre_state = {"game_info": game_info}

            tool_calls_made = []
            reasoning_text = ""

            start_time = time.time()
            interaction_name = "browser_autoevolve_orchestrator"

            claimed_step = None

            if screenshot_b64:
                image_data = base64.b64decode(screenshot_b64)
                image = PILImage.open(io.BytesIO(image_data))

                if self._is_black_frame(image):
                    # Should never happen — black frames mean the screenshot
                    # pipeline is broken. Fail loudly so we can debug.
                    logger.error("Black frame received from server — screenshot pipeline broken")
                    return False, "Black frame received"

                claimed_step = self.runtime.claim_step(
                    owner="orchestrator",
                    interaction_name=interaction_name,
                )

                max_retries = 3
                # VLM call timeout. Local Ollama with the patched
                # 1120-token vision encoder can take ~30-60s on a cold
                # model load (first agent step after daemon start) and
                # ~3-10s once warm. The previous 60s cap was tight
                # enough to fire on every cold load. 180s gives plenty
                # of headroom for the cold path while still catching
                # genuine hangs.
                vlm_timeout_s = int(os.environ.get("VLM_CALL_TIMEOUT_S", "180"))
                response = None
                for attempt in range(max_retries):
                    try:
                        executor = ThreadPoolExecutor(max_workers=1)
                        future = executor.submit(
                            self.vlm.get_query, image, prompt, interaction_name
                        )
                        response = future.result(timeout=vlm_timeout_s)
                        break
                    except FutureTimeoutError:
                        logger.warning(
                            f"VLM timeout after {vlm_timeout_s}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt == max_retries - 1:
                            return False, f"VLM timeout after {vlm_timeout_s}s"
                    except Exception as e:
                        logger.error(f"VLM error: {e}")
                        if attempt == max_retries - 1:
                            return False, str(e)
                        time.sleep(2 ** attempt)
                    finally:
                        executor.shutdown(wait=False)
            else:
                claimed_step = self.runtime.claim_step(
                    owner="orchestrator",
                    interaction_name=interaction_name,
                )
                response = self.vlm.get_text_query(prompt, interaction_name)

            vlm_duration = time.time() - start_time

            # Extract reasoning text
            reasoning_text = self._extract_text_from_response(response)

            # Handle function calls
            tool_call_count = 0
            has_calls = self._handle_vlm_function_calls(
                response, tool_calls_made, tool_call_count, max_tool_calls
            )

            if not has_calls:
                logger.warning("No function calls in response")

            # Log trajectory
            if run_manager:
                self._log_trajectory_for_step(
                    step=claimed_step if claimed_step is not None else preview_step,
                    pre_state=pre_state,
                    tool_calls=tool_calls_made,
                    reasoning=reasoning_text,
                    vlm_duration=vlm_duration,
                )

            # Add to history
            self._add_to_history(
                step=claimed_step if claimed_step is not None else preview_step,
                llm_response=reasoning_text,
                tool_calls=tool_calls_made,
            )

            # Auto-evolve check
            if self.harness_evolver and self.harness_evolver.should_evolve(
                self.step_count, self.optimization_frequency
            ):
                logger.info("Auto-evolve triggered")
                try:
                    self.harness_evolver.evolve(self.step_count)
                except Exception as e:
                    logger.warning(f"Auto-evolve failed: {e}")

            return True, reasoning_text

        except Exception as e:
            logger.error(f"run_step error: {e}")
            traceback.print_exc()
            return False, str(e)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> int:
        self.conversation_history = []
        logger.info("=" * 70)
        logger.info("Browser Game Agent (AutoEvolve)")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model}")
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Tools: {len(self.tools)}")
        if self.max_steps:
            logger.info(f"Max steps: {self.max_steps}")
        logger.info("=" * 70)

        if not self.check_prerequisites():
            return 1

        # Create screenshot log directory for this run
        run_manager = get_run_data_manager()
        self._screenshot_dir = None
        if run_manager:
            ss_dir = Path(str(run_manager.get_run_directory())) / "screenshots"
            ss_dir.mkdir(parents=True, exist_ok=True)
            self._screenshot_dir = ss_dir
            logger.info(f"Screenshots will be saved to {ss_dir}")

        logger.info("Starting autonomous agent loop...")
        self._session_start_time = time.time()

        try:
            while True:
                if self.max_steps and self.step_count >= self.max_steps:
                    logger.info(f"Reached max steps ({self.max_steps})")
                    break

                next_step = self.runtime.peek_next_step()
                logger.info(f"\n{'=' * 70}")
                logger.info(f"Step {next_step}")
                logger.info(f"{'=' * 70}")

                # Fetch game state once per step. Both the prompt and the
                # trajectory pre_state are derived from this single call.
                gs_result = self._execute_function_call_by_name("get_game_state", {})

                # Extract screenshot + game_info from the same payload.
                gs_data: Dict[str, Any] = {}
                screenshot_b64 = None
                game_info = None
                try:
                    gs_data = json.loads(gs_result)
                    screenshot_b64 = gs_data.get("screenshot_base64")
                    game_info = gs_data.get("game_info")
                except Exception:
                    pass

                # Save screenshot for post-run analysis
                if screenshot_b64 and self._screenshot_dir:
                    try:
                        img_data = base64.b64decode(screenshot_b64)
                        img = PILImage.open(io.BytesIO(img_data))
                        img.save(self._screenshot_dir / f"step_{next_step:04d}.png")
                    except Exception:
                        pass

                # Build prompt (uses gs_result for state_text; doesn't refetch)
                prompt = self._build_prompt(gs_result, next_step)

                # Run step — pass game_info so run_step doesn't refetch
                success, output = self.run_step(
                    prompt,
                    screenshot_b64=screenshot_b64,
                    step_number=next_step,
                    game_info=game_info,
                )

                if not success:
                    logger.warning("Step failed, waiting 5s...")
                    time.sleep(5)
                    continue

                # Update server metrics
                try:
                    update_server_metrics(self.server_url)
                except Exception:
                    pass

                # Checkpoint
                try:
                    requests.post(
                        f"{self.server_url}/checkpoint",
                        json={"step_count": self.step_count},
                        timeout=10,
                    )
                except Exception:
                    pass

                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\nAgent stopped by user")
            return 0
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            traceback.print_exc()
            return 1

        logger.info(f"Agent completed {self.step_count} steps")
        return 0

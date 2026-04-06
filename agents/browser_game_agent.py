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
            "double_click": "/mcp/double_click",
            "hold_key": "/mcp/hold_key",
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
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Why you are clicking here",
                        },
                    },
                    "required": ["x", "y", "reasoning"],
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
        ]

        # Memory / Skill / Subagent tools (from shared declarations)
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
            "press_keys", "mouse_click", "double_click", "hold_key",
            "get_game_state", "process_memory",
        ):
            sandbox_tools[tool_name] = _tool_caller(tool_name)

        import random, collections, math, re as _re_mod, heapq, itertools, functools

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

        import random, collections, math, re as _re_mod, heapq, itertools, functools

        captured_stdout = []
        def _print(*args, **kwargs):
            captured_stdout.append(" ".join(str(a) for a in args))

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

    def _get_subagent_vlm(self, tool_subset, system_instruction="", interaction_name="subagent"):
        cache_key = (frozenset(tool_subset) if tool_subset else frozenset(), system_instruction[:200])
        if cache_key in self._subagent_vlm_cache:
            self._subagent_vlm_cache.move_to_end(cache_key)
            return self._subagent_vlm_cache[cache_key]

        tool_decls = [t for t in self.tools if t["name"] in (tool_subset or set())]
        vlm = VLM(
            backend=self.backend,
            model_name=self.model,
            tools=tool_decls or None,
            system_instruction=system_instruction or None,
        )
        self._subagent_vlm_cache[cache_key] = vlm
        if len(self._subagent_vlm_cache) > self._VLM_CACHE_CAP:
            self._subagent_vlm_cache.popitem(last=False)
        return vlm

    # ------------------------------------------------------------------
    # Local subagent handlers (called via registry dispatch)
    # ------------------------------------------------------------------

    def _get_subagent_vlm(self, tool_subset=None, system_instruction="", interaction_name="subagent"):
        """Get or create a VLM for one-step subagents (no tools needed)."""
        if self._local_subagent_vlm is None:
            self._local_subagent_vlm = VLM(
                backend=self.backend,
                model_name=self.model,
            )
        return self._local_subagent_vlm

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
        if not self.conversation_history:
            return "No previous actions recorded."
        recent = self.conversation_history[-ACTION_HISTORY_WINDOW:]
        lines = []
        for entry in recent:
            step = entry.get("step", "?")
            llm_response = entry.get("llm_response", "").strip()
            lines.append(f"[Step {step}]")
            if llm_response:
                lines.append(f"  THINKING: {llm_response}")
            tool_calls = entry.get("tool_calls", [])
            if tool_calls:
                lines.append("  TOOLS:")
                for tc in tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    lines.append(f"    - {name}")
                    lines.append(f"      args: {json.dumps(args, ensure_ascii=False)}")
            lines.append("")
        return "\n".join(lines).strip()

    def _calculate_context_size(self) -> int:
        size = 0
        for entry in self.conversation_history:
            size += len(entry.get("llm_response", ""))
            for tc in entry.get("tool_calls", []):
                size += len(json.dumps(tc.get("args", {})))
                size += len(str(tc.get("result", "")))
        return size

    def _log_trajectory_for_step(self, step, pre_state, tool_calls, reasoning, **kwargs):
        run_manager = get_run_data_manager()
        if not run_manager:
            return
        try:
            run_manager.save_trajectory_entry(
                step=step,
                pre_state=pre_state,
                action={"tool_calls": tool_calls, "reasoning": reasoning},
                **kwargs,
            )
        except Exception as e:
            logger.debug(f"Trajectory save error: {e}")

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
                logger.info(f"Tool call: {fc.name} ({tool_call_count}/{max_tool_calls})")

                try:
                    result = self._execute_function_call(
                        fc, allowed_tool_names=allowed_tool_names
                    )
                except Exception as e:
                    logger.error(f"Tool {fc.name} failed: {e}")
                    tool_calls_made.append(
                        {
                            "name": fc.name,
                            "args": convert_protobuf_args(fc.args) if hasattr(fc, "args") else {},
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
        arr = np.array(image)
        return arr.mean() < 5

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

        prompt = f"""# Current Step: {step_count}

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
    ) -> tuple:
        try:
            preview_step = step_number if step_number is not None else self.runtime.peek_next_step()

            # Pre-state for trajectory
            run_manager = get_run_data_manager()
            pre_state = None
            if run_manager:
                try:
                    gs = self.mcp_adapter.call_tool("get_game_state", {})
                    if isinstance(gs, str):
                        gs = json.loads(gs)
                    if gs.get("success"):
                        pre_state = {"game_info": gs.get("game_info", {})}
                except Exception:
                    pass

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
                response = None
                for attempt in range(max_retries):
                    try:
                        executor = ThreadPoolExecutor(max_workers=1)
                        future = executor.submit(
                            self.vlm.get_query, image, prompt, interaction_name
                        )
                        response = future.result(timeout=60)
                        break
                    except FutureTimeoutError:
                        logger.warning(f"VLM timeout (attempt {attempt + 1}/{max_retries})")
                        if attempt == max_retries - 1:
                            if claimed_step is not None:
                                self.runtime.release_step(claimed_step)
                            return False, "VLM timeout"
                    except Exception as e:
                        logger.error(f"VLM error: {e}")
                        if attempt == max_retries - 1:
                            if claimed_step is not None:
                                self.runtime.release_step(claimed_step)
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
            if claimed_step is not None:
                try:
                    self.runtime.release_step(claimed_step)
                except Exception:
                    pass
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

        try:
            while True:
                if self.max_steps and self.step_count >= self.max_steps:
                    logger.info(f"Reached max steps ({self.max_steps})")
                    break

                next_step = self.runtime.peek_next_step()
                logger.info(f"\n{'=' * 70}")
                logger.info(f"Step {next_step}")
                logger.info(f"{'=' * 70}")

                # Fetch game state
                gs_result = self._execute_function_call_by_name("get_game_state", {})

                # Extract screenshot
                try:
                    gs_data = json.loads(gs_result)
                    screenshot_b64 = gs_data.get("screenshot_base64")
                except Exception:
                    screenshot_b64 = None

                # Save screenshot for post-run analysis
                if screenshot_b64 and self._screenshot_dir:
                    try:
                        img_data = base64.b64decode(screenshot_b64)
                        img = PILImage.open(io.BytesIO(img_data))
                        img.save(self._screenshot_dir / f"step_{next_step:04d}.png")
                    except Exception:
                        pass

                # Build prompt
                prompt = self._build_prompt(gs_result, next_step)

                # Run step
                success, output = self.run_step(
                    prompt, screenshot_b64=screenshot_b64, step_number=next_step
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

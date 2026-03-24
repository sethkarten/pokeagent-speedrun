"""
SubagentExecutor — generic loop and execution engine for subagents.

Extracts the shared boilerplate from the battler / planner loops and
provides ``execute_custom_subagent`` and ``process_trajectory_history``.
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Set

from agents.subagents.utils.context import load_subagent_context
from agents.subagents.utils.registry import SUBAGENT_FORBIDDEN_TOOLS
from agents.subagents.utils.trajectory_window import (
    format_trajectory_window,
    load_trajectory_range,
)
from utils.stores.subagents import get_subagent_store

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3


# ------------------------------------------------------------------
# Callback protocols (all provided by PokeAgent at construction)
# ------------------------------------------------------------------

class SubagentExecutor:
    """Runs generic and custom subagent loops on behalf of PokeAgent.

    Parameters
    ----------
    runtime : PokeAgentRuntime
    mcp_adapter : MCPToolAdapter
    get_run_data_manager_fn : callable returning RunDataManager
    server_url : str
    get_subagent_vlm_fn : callable(tool_names, supplemental_tools) -> VLM
    handle_vlm_function_calls_fn : callable matching _handle_vlm_function_calls
    extract_text_fn : callable(response) -> str
    log_trajectory_fn : callable(run_manager, step_num, pre_state, prompt,
                                 reasoning, tool_calls, response) -> None
    llm_logger : LLMLogger
    wait_for_actions_fn : callable()
    """

    def __init__(
        self,
        *,
        runtime,
        mcp_adapter,
        get_run_data_manager_fn: Callable,
        server_url: str,
        get_subagent_vlm_fn: Callable,
        handle_vlm_function_calls_fn: Callable,
        extract_text_fn: Callable,
        log_trajectory_fn: Callable,
        llm_logger,
        wait_for_actions_fn: Callable,
    ):
        self.runtime = runtime
        self.mcp_adapter = mcp_adapter
        self._get_run_data_manager = get_run_data_manager_fn
        self.server_url = server_url
        self._get_subagent_vlm = get_subagent_vlm_fn
        self._handle_vlm_function_calls = handle_vlm_function_calls_fn
        self._extract_text = extract_text_fn
        self._log_trajectory = log_trajectory_fn
        self.llm_logger = llm_logger
        self._wait_for_actions = wait_for_actions_fn

    # ------------------------------------------------------------------
    # Generic subagent loop
    # ------------------------------------------------------------------

    def run_generic_loop(
        self,
        *,
        vlm,
        prompt_builder: Callable[[Dict[str, Any], int, List[Dict[str, Any]]], str],
        allowed_tool_names: Set[str],
        safety_cap: int,
        owner: str,
        interaction_name: str,
        should_continue_fn: Optional[Callable[[Dict[str, Any], int], bool]] = None,
        on_turn_complete_fn: Optional[
            Callable[[int, str, List[Dict[str, Any]]], None]
        ] = None,
    ) -> Dict[str, Any]:
        """Core subagent loop used by battler, planner, and custom subagents.

        Parameters
        ----------
        vlm : VLM instance (with or without tools)
        prompt_builder : ``(context_dict, turn_index_1based, history) -> prompt_str``
        allowed_tool_names : tool allow-list (enforced by _handle_vlm_function_calls)
        safety_cap : max iterations
        owner : step ownership label (e.g. ``"subagent_battler"``)
        interaction_name : VLM interaction label (e.g. ``"Subagent_Battler"``)
        should_continue_fn : ``(context, turn_index) -> bool``; defaults to always True
        on_turn_complete_fn : ``(turn_index, reasoning, tool_calls_made) -> None``

        Returns
        -------
        dict with ``turns_taken``, ``history``, ``final_reasoning``, ``hit_safety_cap``.
        """
        run_manager = self._get_run_data_manager()
        history: List[Dict[str, Any]] = []
        turns_taken = 0
        consecutive_failures = 0
        final_reasoning = ""

        while turns_taken < safety_cap:
            # --- pre-turn continuity check ---
            context = load_subagent_context(
                self.mcp_adapter,
                run_manager,
                last_n_steps=1,
                include_current_image=True,
            )
            if should_continue_fn is not None and not should_continue_fn(context, turns_taken):
                break

            current_image = context.get("current_image")

            step_number = self.runtime.claim_step(
                owner=owner, interaction_name=interaction_name
            )

            try:
                prompt = prompt_builder(context, turns_taken + 1, history)
            except Exception:
                logger.error("prompt_builder raised: %s", traceback.format_exc())
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error("Hit %d consecutive failures — aborting loop.", MAX_CONSECUTIVE_FAILURES)
                    break
                continue

            # --- VLM query ---
            try:
                if current_image is not None:
                    response = vlm.get_query(current_image, prompt, interaction_name)
                else:
                    response = vlm.get_text_query(prompt, interaction_name)
                reasoning_text = self._extract_text(response)
                final_reasoning = reasoning_text
                consecutive_failures = 0
            except Exception:
                logger.error("VLM error on turn %d: %s", turns_taken + 1, traceback.format_exc())
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error("Hit %d consecutive VLM failures — aborting loop.", MAX_CONSECUTIVE_FAILURES)
                    break
                continue

            # --- tool execution ---
            tool_calls_made: List[Dict[str, Any]] = []
            try:
                used_tools = self._handle_vlm_function_calls(
                    response,
                    tool_calls_made,
                    0,
                    4,
                    allowed_tool_names=allowed_tool_names,
                )
            except Exception:
                logger.error("Tool execution error: %s", traceback.format_exc())
                used_tools = False

            if not used_tools or not tool_calls_made:
                logger.warning("%s produced no tool calls on turn %d — breaking.", interaction_name, turns_taken + 1)
                break

            # --- logging ---
            self.llm_logger.add_step_tool_calls(step_number, tool_calls_made)
            if run_manager:
                game_state_result = context.get("game_state_result", {})
                raw_state = game_state_result.get("raw_state", {}) or {}
                pre_state = run_manager.create_state_snapshot(raw_state) if raw_state else None
                if pre_state:
                    self._log_trajectory(
                        run_manager=run_manager,
                        step_num=step_number,
                        pre_state=pre_state,
                        prompt=prompt,
                        reasoning=reasoning_text,
                        tool_calls=tool_calls_made,
                        response=reasoning_text,
                    )

            try:
                from utils.metric_tracking.server_metrics import update_server_metrics
                update_server_metrics(self.server_url)
            except Exception:
                pass

            history.append({
                "step": turns_taken + 1,
                "reasoning": reasoning_text,
                "tool_calls": tool_calls_made,
            })

            if on_turn_complete_fn is not None:
                on_turn_complete_fn(turns_taken + 1, reasoning_text, tool_calls_made)

            turns_taken += 1

        return {
            "turns_taken": turns_taken,
            "history": history,
            "final_reasoning": final_reasoning,
            "hit_safety_cap": turns_taken >= safety_cap,
        }

    # ------------------------------------------------------------------
    # execute_custom_subagent
    # ------------------------------------------------------------------

    def execute_custom_subagent(self, arguments: dict) -> str:
        """Launch a custom subagent from the registry or with an inline config."""
        try:
            return self._execute_custom_subagent_inner(arguments)
        except Exception as e:
            logger.error("execute_custom_subagent failed: %s", traceback.format_exc())
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _execute_custom_subagent_inner(self, arguments: dict) -> str:
        subagent_id = arguments.get("subagent_id")
        inline_config = arguments.get("config")

        if not subagent_id and not inline_config:
            return json.dumps({
                "success": False,
                "error": "Provide either 'subagent_id' or 'config'.",
            }, indent=2)

        if subagent_id and inline_config:
            return json.dumps({
                "success": False,
                "error": "'subagent_id' and 'config' are mutually exclusive.",
            }, indent=2)

        # --- resolve config ---
        if subagent_id:
            store = get_subagent_store()
            entry = store.get(subagent_id)
            if entry is None:
                return json.dumps({
                    "success": False,
                    "error": f"Subagent '{subagent_id}' not found in registry.",
                }, indent=2)
            if getattr(entry, "is_builtin", False):
                return json.dumps({
                    "success": False,
                    "error": (
                        f"'{subagent_id}' is a built-in subagent. "
                        "Use the dedicated tool (e.g. subagent_battler) instead."
                    ),
                }, indent=2)
            config = {
                "max_turns": entry.max_turns,
                "available_tools": list(entry.available_tools),
                "system_instructions": entry.system_instructions,
                "directive": entry.directive,
                "return_condition": entry.return_condition,
                "name": entry.name,
            }
        else:
            config = dict(inline_config)

        # --- validate tools ---
        requested_tools = set(config.get("available_tools") or [])
        forbidden = requested_tools & SUBAGENT_FORBIDDEN_TOOLS
        if forbidden:
            return json.dumps({
                "success": False,
                "error": f"Forbidden tools in config: {sorted(forbidden)}. "
                         "Custom subagents cannot spawn other subagents.",
            }, indent=2)

        max_turns = int(config.get("max_turns", 25))
        instructions = config.get("system_instructions", "")
        directive = config.get("directive", "")
        return_condition = config.get("return_condition", "")
        name = config.get("name", "Custom_Subagent")
        interaction_name = f"Custom_{name.replace(' ', '_')}"

        vlm = self._get_subagent_vlm(requested_tools or None)

        def prompt_builder(
            context: Dict[str, Any],
            turn_index: int,
            history: List[Dict[str, Any]],
        ) -> str:
            parts = []
            if instructions:
                parts.append(f"### SYSTEM INSTRUCTIONS\n{instructions}")
            parts.append(f"### DIRECTIVE\n{directive}")
            if return_condition:
                parts.append(f"### RETURN CONDITION\n{return_condition}")

            state_text = context.get("current_state", {}).get("state_text", "")
            location = context.get("current_state", {}).get("location", "Unknown")
            parts.append(f"### CURRENT STATE\nLocation: {location}\n{state_text}")

            mem = context.get("memory_summary", "")
            if mem:
                parts.append(f"### MEMORY OVERVIEW\n{mem}")
            skill = context.get("skill_overview", "")
            if skill:
                parts.append(f"### SKILL OVERVIEW\n{skill}")

            if history:
                hist_lines = []
                for h in history[-10:]:
                    tc_names = [tc.get("name", "?") for tc in h.get("tool_calls", [])]
                    hist_lines.append(
                        f"Turn {h['step']}: {', '.join(tc_names)}"
                    )
                parts.append(
                    f"### SUBAGENT HISTORY (turn {turn_index})\n"
                    + "\n".join(hist_lines)
                )
            else:
                parts.append(f"### TURN {turn_index} — No previous actions.")

            parts.append(
                "### INSTRUCTIONS\n"
                "Call tools to accomplish your directive. "
                "When finished, include return_to_orchestrator in your final "
                "tool call args to hand control back."
            )
            return "\n\n".join(parts)

        should_return = [False]

        def on_turn_complete(turn: int, reasoning: str, tool_calls: List[Dict]) -> None:
            for tc in tool_calls:
                if tc.get("args", {}).get("return_to_orchestrator"):
                    should_return[0] = True
                    return

        def should_continue(context: Dict, turn: int) -> bool:
            return not should_return[0]

        result = self.run_generic_loop(
            vlm=vlm,
            prompt_builder=prompt_builder,
            allowed_tool_names=requested_tools,
            safety_cap=max_turns,
            owner=f"custom_subagent_{name}",
            interaction_name=interaction_name,
            should_continue_fn=should_continue,
            on_turn_complete_fn=on_turn_complete,
        )

        return json.dumps({
            "success": True,
            "subagent_name": name,
            "turns_taken": result["turns_taken"],
            "hit_safety_cap": result["hit_safety_cap"],
            "final_reasoning": result["final_reasoning"],
        }, indent=2)

    # ------------------------------------------------------------------
    # process_trajectory_history
    # ------------------------------------------------------------------

    def process_trajectory_history(self, arguments: dict) -> str:
        """One-step VLM analysis on a trajectory window with a custom directive."""
        try:
            return self._process_trajectory_history_inner(arguments)
        except Exception as e:
            logger.error("process_trajectory_history failed: %s", traceback.format_exc())
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _process_trajectory_history_inner(self, arguments: dict) -> str:
        window_range = arguments.get("window_range", [])
        directive = arguments.get("directive", "")

        if not directive:
            return json.dumps({
                "success": False,
                "error": "A non-empty 'directive' is required.",
            }, indent=2)

        if not isinstance(window_range, (list, tuple)) or len(window_range) < 2:
            return json.dumps({
                "success": False,
                "error": "'window_range' must be a two-element list [start, end].",
            }, indent=2)

        start, end = int(window_range[0]), int(window_range[1])
        if start > end:
            start, end = end, start

        run_manager = self._get_run_data_manager()
        entries, actual_min, actual_max = load_trajectory_range(run_manager, start, end)

        trajectory_text = format_trajectory_window(entries)
        prompt = (
            f"### DIRECTIVE\n{directive}\n\n"
            f"### TRAJECTORY WINDOW (requested steps {start}–{end}, "
            f"available range {actual_min}–{actual_max}, "
            f"{len(entries)} entries returned)\n\n"
            f"{trajectory_text}\n\n"
            "Analyze the trajectory above according to the directive. "
            "Be specific, cite step numbers, and provide actionable insights."
        )

        step_number = self.runtime.claim_step(
            owner="subagent", interaction_name="Process_Trajectory_History"
        )
        vlm = self._get_subagent_vlm(None)
        response = vlm.get_text_query(prompt, "Process_Trajectory_History")
        analysis = self._extract_text(response)

        self.llm_logger.add_step_tool_calls(
            step_number,
            [{"name": "Process_Trajectory_History", "args": {
                "window_range": [start, end],
                "directive": directive[:800],
            }}],
        )

        return json.dumps({
            "success": True,
            "analysis": analysis,
            "steps_analyzed": len(entries),
            "actual_range": [actual_min, actual_max],
            "requested_range": [start, end],
            "step_number": step_number,
        }, indent=2)

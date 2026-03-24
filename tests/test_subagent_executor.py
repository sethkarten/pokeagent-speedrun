"""Tests for agents.subagents.utils.executor — SubagentExecutor generic loop,
execute_custom_subagent, and process_trajectory_history."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys_path = Path(__file__).parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from agents.subagents.utils.executor import MAX_CONSECUTIVE_FAILURES, SubagentExecutor
from agents.subagents.utils.registry import SUBAGENT_FORBIDDEN_TOOLS
from agents.subagents.utils.runtime import PokeAgentRuntime


# ---------------------------------------------------------------------------
# Helpers / Stubs
# ---------------------------------------------------------------------------

_step_counter = 0


def _make_runtime():
    """Minimal PokeAgentRuntime with a simple step counter."""
    global _step_counter
    _step_counter = 0
    return PokeAgentRuntime(
        initial_step=0,
        publish_history=lambda *a, **kw: None,
        publish_function_result=lambda *a, **kw: None,
        on_step_change=lambda step: None,
    )


def _make_mock_mcp():
    """MCPToolAdapter mock — returns a minimal game state on every call."""
    adapter = MagicMock()
    adapter.call_tool.return_value = {
        "success": True,
        "state_text": "Littleroot Town — Overworld",
        "screenshot_base64": "",
        "raw_state": {
            "location": {"map_name": "LITTLEROOT_TOWN"},
            "player": {"x": 5, "y": 5},
        },
    }
    return adapter


def _make_run_data_manager(tmp_path):
    mgr = MagicMock()
    mgr.create_state_snapshot.return_value = {"step": 1}
    mgr.trajectory_dir = str(tmp_path / "trajectory")
    Path(mgr.trajectory_dir).mkdir(parents=True, exist_ok=True)
    return mgr


class StubVLM:
    """VLM stub that returns canned text responses and optionally function calls."""

    def __init__(self, responses=None, *, with_tools=False):
        self.responses = responses or ["Stub reasoning."]
        self._idx = 0
        self.with_tools = with_tools

    def get_query(self, image, prompt, interaction_name):
        return self._next()

    def get_text_query(self, prompt, interaction_name):
        return self._next()

    def _next(self):
        resp = self.responses[min(self._idx, len(self.responses) - 1)]
        self._idx += 1
        return resp


# ---------------------------------------------------------------------------
# Executor factory
# ---------------------------------------------------------------------------

def _make_executor(tmp_path, *, vlm_stub=None, tool_calls_side_effect=None):
    runtime = _make_runtime()
    mcp = _make_mock_mcp()
    run_mgr = _make_run_data_manager(tmp_path)
    llm_logger = MagicMock()

    handle_fn = MagicMock(return_value=True)
    if tool_calls_side_effect is not None:
        handle_fn.side_effect = tool_calls_side_effect

    executor = SubagentExecutor(
        runtime=runtime,
        mcp_adapter=mcp,
        get_run_data_manager_fn=lambda: run_mgr,
        server_url="http://localhost:8000",
        get_subagent_vlm_fn=lambda *a, **kw: vlm_stub or StubVLM(),
        handle_vlm_function_calls_fn=handle_fn,
        extract_text_fn=lambda resp: resp if isinstance(resp, str) else str(resp),
        log_trajectory_fn=lambda **kw: None,
        llm_logger=llm_logger,
        wait_for_actions_fn=lambda: None,
    )
    return executor, handle_fn, run_mgr


# ---------------------------------------------------------------------------
# run_generic_loop tests
# ---------------------------------------------------------------------------


class TestRunGenericLoop:
    def test_basic_loop_runs_to_safety_cap(self, tmp_path):
        cap = 3
        tool_calls_made_log = []

        def side_effect(response, tool_calls_made, depth, max_depth, **kw):
            tool_calls_made.append({"name": "press_buttons", "args": {}})
            return True

        executor, handle_fn, _ = _make_executor(
            tmp_path, tool_calls_side_effect=side_effect,
        )

        result = executor.run_generic_loop(
            vlm=StubVLM(),
            prompt_builder=lambda ctx, turn, hist: f"Turn {turn}",
            allowed_tool_names={"press_buttons"},
            safety_cap=cap,
            owner="test",
            interaction_name="TestLoop",
        )
        assert result["turns_taken"] == cap
        assert result["hit_safety_cap"] is True

    def test_should_continue_fn_can_break(self, tmp_path):
        def side_effect(response, tool_calls_made, depth, max_depth, **kw):
            tool_calls_made.append({"name": "press_buttons", "args": {}})
            return True

        executor, _, _ = _make_executor(
            tmp_path, tool_calls_side_effect=side_effect,
        )

        result = executor.run_generic_loop(
            vlm=StubVLM(),
            prompt_builder=lambda ctx, turn, hist: f"Turn {turn}",
            allowed_tool_names={"press_buttons"},
            safety_cap=100,
            owner="test",
            interaction_name="TestLoop",
            should_continue_fn=lambda ctx, turn: turn < 2,
        )
        assert result["turns_taken"] == 2
        assert result["hit_safety_cap"] is False

    def test_no_tool_calls_breaks(self, tmp_path):
        """If the VLM produces no tool calls, the loop breaks immediately."""
        def no_tools(response, tool_calls_made, depth, max_depth, **kw):
            return False

        executor, _, _ = _make_executor(
            tmp_path, tool_calls_side_effect=no_tools,
        )

        result = executor.run_generic_loop(
            vlm=StubVLM(),
            prompt_builder=lambda ctx, turn, hist: "prompt",
            allowed_tool_names=set(),
            safety_cap=10,
            owner="test",
            interaction_name="TestLoop",
        )
        assert result["turns_taken"] == 0

    def test_consecutive_vlm_failures_abort(self, tmp_path):
        """After MAX_CONSECUTIVE_FAILURES VLM errors, loop aborts."""
        failing_vlm = MagicMock()
        failing_vlm.get_query.side_effect = RuntimeError("VLM crashed")
        failing_vlm.get_text_query.side_effect = RuntimeError("VLM crashed")

        executor, _, _ = _make_executor(tmp_path, vlm_stub=failing_vlm)

        result = executor.run_generic_loop(
            vlm=failing_vlm,
            prompt_builder=lambda ctx, turn, hist: "prompt",
            allowed_tool_names={"press_buttons"},
            safety_cap=100,
            owner="test",
            interaction_name="TestLoop",
        )
        assert result["turns_taken"] == 0
        assert result["hit_safety_cap"] is False

    def test_on_turn_complete_callback(self, tmp_path):
        turns_completed = []

        def side_effect(response, tool_calls_made, depth, max_depth, **kw):
            tool_calls_made.append({"name": "press_buttons", "args": {}})
            return True

        executor, _, _ = _make_executor(
            tmp_path, tool_calls_side_effect=side_effect,
        )

        result = executor.run_generic_loop(
            vlm=StubVLM(),
            prompt_builder=lambda ctx, turn, hist: "prompt",
            allowed_tool_names={"press_buttons"},
            safety_cap=3,
            owner="test",
            interaction_name="TestLoop",
            on_turn_complete_fn=lambda turn, reasoning, tc: turns_completed.append(turn),
        )
        assert turns_completed == [1, 2, 3]


# ---------------------------------------------------------------------------
# execute_custom_subagent tests
# ---------------------------------------------------------------------------


class TestExecuteCustomSubagent:
    def test_missing_both_id_and_config(self, tmp_path):
        executor, _, _ = _make_executor(tmp_path)
        result = json.loads(executor.execute_custom_subagent({"reasoning": "test"}))
        assert result["success"] is False
        assert "subagent_id" in result["error"] or "config" in result["error"]

    def test_both_id_and_config(self, tmp_path):
        executor, _, _ = _make_executor(tmp_path)
        result = json.loads(executor.execute_custom_subagent({
            "subagent_id": "sa_0001",
            "config": {"name": "test"},
            "reasoning": "test",
        }))
        assert result["success"] is False
        assert "mutually exclusive" in result["error"]

    def test_forbidden_tools_rejected(self, tmp_path):
        executor, _, _ = _make_executor(tmp_path)
        result = json.loads(executor.execute_custom_subagent({
            "config": {
                "name": "Bad",
                "available_tools": ["press_buttons", "execute_custom_subagent"],
                "directive": "do stuff",
            },
            "reasoning": "test",
        }))
        assert result["success"] is False
        assert "Forbidden" in result["error"]

    def test_inline_config_runs(self, tmp_path):
        def side_effect(response, tool_calls_made, depth, max_depth, **kw):
            tool_calls_made.append({
                "name": "press_buttons",
                "args": {"return_to_orchestrator": True},
            })
            return True

        executor, _, _ = _make_executor(
            tmp_path, tool_calls_side_effect=side_effect,
        )
        result = json.loads(executor.execute_custom_subagent({
            "config": {
                "name": "Inline",
                "available_tools": ["press_buttons"],
                "directive": "test directive",
                "max_turns": 5,
            },
            "reasoning": "testing inline",
        }))
        assert result["success"] is True
        assert result["subagent_name"] == "Inline"
        assert result["turns_taken"] >= 1

    def test_builtin_subagent_rejected(self, tmp_path):
        """Attempting to launch a built-in subagent via execute_custom_subagent
        should be rejected (user must use the dedicated tool)."""
        from utils.stores.subagents import SubagentStore

        store = SubagentStore(cache_dir=str(tmp_path / "sa"))
        builtin_id = next(
            eid for eid, e in store.entries.items() if e.is_builtin
        )

        executor, _, _ = _make_executor(tmp_path)
        with patch("agents.subagents.utils.executor.get_subagent_store", return_value=store):
            result = json.loads(executor.execute_custom_subagent({
                "subagent_id": builtin_id,
                "reasoning": "test",
            }))
        assert result["success"] is False
        assert "built-in" in result["error"]

    def test_registry_lookup_not_found(self, tmp_path):
        from utils.stores.subagents import SubagentStore

        store = SubagentStore(cache_dir=str(tmp_path / "sa"))
        executor, _, _ = _make_executor(tmp_path)
        with patch("agents.subagents.utils.executor.get_subagent_store", return_value=store):
            result = json.loads(executor.execute_custom_subagent({
                "subagent_id": "sa_9999",
                "reasoning": "test",
            }))
        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# process_trajectory_history tests
# ---------------------------------------------------------------------------


class TestProcessTrajectoryHistory:
    def test_missing_directive(self, tmp_path):
        executor, _, _ = _make_executor(tmp_path)
        result = json.loads(executor.process_trajectory_history({
            "window_range": [1, 10],
        }))
        assert result["success"] is False
        assert "directive" in result["error"]

    def test_invalid_window_range(self, tmp_path):
        executor, _, _ = _make_executor(tmp_path)
        result = json.loads(executor.process_trajectory_history({
            "window_range": [5],
            "directive": "analyze",
        }))
        assert result["success"] is False
        assert "window_range" in result["error"]

    def test_successful_analysis(self, tmp_path):
        executor, _, run_mgr = _make_executor(tmp_path)

        fake_entries = [
            {"step": i, "action": "press_buttons", "outcome": "ok"}
            for i in range(1, 6)
        ]
        with patch(
            "agents.subagents.utils.executor.load_trajectory_range",
            return_value=(fake_entries, 1, 5),
        ):
            result = json.loads(executor.process_trajectory_history({
                "window_range": [1, 5],
                "directive": "What happened?",
            }))
        assert result["success"] is True
        assert result["steps_analyzed"] == 5
        assert result["actual_range"] == [1, 5]
        assert "analysis" in result

    def test_swapped_range_is_normalized(self, tmp_path):
        executor, _, _ = _make_executor(tmp_path)
        with patch(
            "agents.subagents.utils.executor.load_trajectory_range",
            return_value=([], 10, 5),
        ):
            result = json.loads(executor.process_trajectory_history({
                "window_range": [10, 5],
                "directive": "analyze",
            }))
        assert result["success"] is True

"""Tests for PokeAgent system-instruction selection, prompt optimization, and scaffold tool declarations."""

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest

from agents.PokeAgent import PokeAgent
from agents.prompts.paths import POKEAGENT_PROMPT_PATH, SIMPLE_PROMPT_PATH, SIMPLEST_PROMPT_PATH
from agents.tools.registry import build_tools_for_scaffold

_POKE_MODULE = importlib.import_module("agents.PokeAgent")


def _filename_arg(load_mock):
    """Bound-method mock may record (filename,) or (self, filename)."""
    args = load_mock.call_args.args
    return args[1] if len(args) > 1 else args[0]


def test_auto_system_uses_pokeagent_without_optimization():
    load_mock = MagicMock(return_value="SYS_BODY")
    with patch.object(_POKE_MODULE, "MCPToolAdapter"), patch.object(_POKE_MODULE, "VLM"), patch.object(
        _POKE_MODULE, "create_prompt_optimizer"
    ) as m_cop, patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = None
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            PokeAgent(server_url="http://localhost:8000", enable_prompt_optimization=False)
    load_mock.assert_called_once()
    assert _filename_arg(load_mock) == POKEAGENT_PROMPT_PATH
    m_cop.assert_not_called()


def test_auto_system_uses_pokeagent_with_optimization_and_run_manager():
    load_mock = MagicMock(return_value="SYS_BODY")
    fake_opt = MagicMock()
    with patch.object(_POKE_MODULE, "MCPToolAdapter"), patch.object(_POKE_MODULE, "VLM"), patch.object(
        _POKE_MODULE, "create_prompt_optimizer"
    ) as m_cop, patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = MagicMock()
        m_cop.return_value = fake_opt
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            agent = PokeAgent(server_url="http://localhost:8000", enable_prompt_optimization=True)
    load_mock.assert_called_once()
    assert _filename_arg(load_mock) == POKEAGENT_PROMPT_PATH
    m_cop.assert_called_once()
    assert agent.prompt_optimizer is fake_opt


def test_optimization_requested_no_run_manager_raises():
    """Prompt optimization without run_data_manager is a configuration error."""
    load_mock = MagicMock(return_value="SYS_BODY")
    with patch.object(_POKE_MODULE, "MCPToolAdapter"), patch.object(_POKE_MODULE, "VLM") as m_vlm, patch.object(
        _POKE_MODULE, "create_prompt_optimizer"
    ) as m_cop, patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = None
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            with pytest.raises(RuntimeError, match="run_data_manager"):
                PokeAgent(server_url="http://localhost:8000", enable_prompt_optimization=True)
    load_mock.assert_called_once()
    assert _filename_arg(load_mock) == POKEAGENT_PROMPT_PATH
    assert m_cop.call_count == 0
    assert m_vlm.call_count == 1


def test_explicit_system_path_still_requires_run_manager_for_optimization():
    load_mock = MagicMock(return_value="CUSTOM")
    with patch.object(_POKE_MODULE, "MCPToolAdapter"), patch.object(_POKE_MODULE, "VLM") as m_vlm, patch.object(
        _POKE_MODULE, "create_prompt_optimizer"
    ) as m_cop, patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = None
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            with pytest.raises(RuntimeError, match="run_data_manager"):
                PokeAgent(
                    server_url="http://localhost:8000",
                    enable_prompt_optimization=True,
                    system_instructions_file=POKEAGENT_PROMPT_PATH,
                )
    load_mock.assert_called_once()
    assert _filename_arg(load_mock) == POKEAGENT_PROMPT_PATH
    assert m_vlm.call_count == 1
    assert m_cop.call_count == 0


def test_simple_scaffold_uses_simple_prompt():
    load_mock = MagicMock(return_value="SYS_BODY")
    with patch.object(_POKE_MODULE, "MCPToolAdapter"), patch.object(_POKE_MODULE, "VLM"), patch.object(
        _POKE_MODULE, "create_prompt_optimizer"
    ) as m_cop, patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = None
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            PokeAgent(server_url="http://localhost:8000", scaffold="simple")
    load_mock.assert_called_once()
    assert _filename_arg(load_mock) == SIMPLE_PROMPT_PATH
    m_cop.assert_not_called()


def test_simplest_scaffold_uses_simplest_prompt():
    load_mock = MagicMock(return_value="SYS_BODY")
    with patch.object(_POKE_MODULE, "MCPToolAdapter"), patch.object(_POKE_MODULE, "VLM"), patch.object(
        _POKE_MODULE, "create_prompt_optimizer"
    ) as m_cop, patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = None
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            PokeAgent(server_url="http://localhost:8000", scaffold="simplest")
    load_mock.assert_called_once()
    assert _filename_arg(load_mock) == SIMPLEST_PROMPT_PATH
    m_cop.assert_not_called()


# ---------------------------------------------------------------------------
# Tool declarations per scaffold (registry integration)
# ---------------------------------------------------------------------------

EXPECTED_TOOLS_PER_SCAFFOLD = {
    "simplest": {"press_buttons", "process_memory"},
    "simple": {
        "press_buttons", "complete_direct_objective", "process_memory",
        "process_skill", "run_skill", "run_code", "process_subagent",
        "execute_custom_subagent", "process_trajectory_history",
        "replan_objectives",
    },
    "autoevolve": {
        "press_buttons", "complete_direct_objective", "process_memory",
        "process_skill", "run_skill", "run_code", "process_subagent",
        "execute_custom_subagent", "process_trajectory_history",
        "replan_objectives", "evolve_harness",
    },
    "pokeagent": {
        "press_buttons", "complete_direct_objective", "process_memory",
        "process_skill", "run_skill", "run_code", "process_subagent",
        "get_progress_summary", "navigate_to", "get_walkthrough",
        "lookup_pokemon_info",
        "subagent_reflect", "subagent_verify", "subagent_gym_puzzle",
        "subagent_summarize", "subagent_battler", "subagent_plan_objectives",
        "subagent_cleanup_run_artifacts",
        "execute_custom_subagent", "process_trajectory_history",
    },
}


@pytest.mark.parametrize("scaffold", list(EXPECTED_TOOLS_PER_SCAFFOLD.keys()))
def test_tool_declarations_per_scaffold(scaffold):
    """Each scaffold gets exactly the expected set of tools from the registry."""
    tools = build_tools_for_scaffold(scaffold)
    actual_names = {t["name"] for t in tools}
    expected = EXPECTED_TOOLS_PER_SCAFFOLD[scaffold]
    assert actual_names == expected, (
        f"scaffold={scaffold}: extra={actual_names - expected}, missing={expected - actual_names}"
    )


@pytest.mark.parametrize("scaffold", list(EXPECTED_TOOLS_PER_SCAFFOLD.keys()))
def test_tool_declarations_via_pokeagent(scaffold):
    """PokeAgent._create_tool_declarations delegates to the registry correctly."""
    load_mock = MagicMock(return_value="SYS_BODY")
    with patch.object(_POKE_MODULE, "MCPToolAdapter"), patch.object(_POKE_MODULE, "VLM"), \
         patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = None
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            agent = PokeAgent(server_url="http://localhost:8000", scaffold=scaffold)
    actual_names = {t["name"] for t in agent.tools}
    expected = EXPECTED_TOOLS_PER_SCAFFOLD[scaffold]
    assert actual_names == expected


# ---------------------------------------------------------------------------
# Simplest prompt content gating
# ---------------------------------------------------------------------------

def test_simplest_prompt_excludes_stores_and_objectives():
    """The simplest scaffold prompt must omit OBJECTIVES, SKILL LIBRARY, and SUBAGENT REGISTRY."""
    load_mock = MagicMock(return_value="SYS_BODY")
    with patch.object(_POKE_MODULE, "MCPToolAdapter") as mock_mcp_cls, \
         patch.object(_POKE_MODULE, "VLM"), \
         patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = None
        mock_adapter = MagicMock()
        mock_mcp_cls.return_value = mock_adapter
        mock_adapter.call_tool.return_value = {
            "success": True,
            "overview": "mem_0001 | some memory",
        }
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            agent = PokeAgent(server_url="http://localhost:8000", scaffold="simplest")

    game_state = json.dumps({
        "state_text": "Location: Littleroot Town",
        "objectives_mode": "categorized",
        "categorized_objectives": {"story": {"id": "s1", "description": "Go north"}},
        "categorized_status": {"story": {"current_index": 0, "total": 5, "completed": 0}},
    })

    prompt = agent._build_structured_prompt(game_state, step_count=1)

    assert "### OBJECTIVES" not in prompt
    assert "### SKILL LIBRARY" not in prompt
    assert "### SUBAGENT REGISTRY" not in prompt
    assert "### LONG-TERM MEMORY OVERVIEW" in prompt
    assert "### STATE" in prompt
    assert "### SHORT-TERM MEMORY" in prompt


def test_simple_prompt_includes_all_sections():
    """The simple scaffold prompt should still include all standard sections."""
    load_mock = MagicMock(return_value="SYS_BODY")
    with patch.object(_POKE_MODULE, "MCPToolAdapter") as mock_mcp_cls, \
         patch.object(_POKE_MODULE, "VLM"), \
         patch.object(_POKE_MODULE, "get_run_data_manager") as m_rm:
        m_rm.return_value = None
        mock_adapter = MagicMock()
        mock_mcp_cls.return_value = mock_adapter
        mock_adapter.call_tool.return_value = {
            "success": True,
            "overview": "mem_0001 | some memory",
        }
        with patch.object(PokeAgent, "_load_system_instructions", load_mock):
            agent = PokeAgent(server_url="http://localhost:8000", scaffold="simple")

    game_state = json.dumps({
        "state_text": "Location: Littleroot Town",
        "objectives_mode": "legacy",
        "direct_objective": "Go north",
        "direct_objective_status": "",
        "direct_objective_context": "",
    })

    prompt = agent._build_structured_prompt(game_state, step_count=1)

    assert "### OBJECTIVES" in prompt
    assert "### SKILL LIBRARY" in prompt
    assert "### SUBAGENT REGISTRY" in prompt
    assert "### LONG-TERM MEMORY OVERVIEW" in prompt

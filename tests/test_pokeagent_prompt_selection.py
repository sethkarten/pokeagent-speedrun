"""Tests for PokeAgent system-instruction selection and prompt optimization requirements."""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from agents.PokeAgent import PokeAgent
from agents.prompts.paths import POKEAGENT_PROMPT_PATH, SIMPLE_PROMPT_PATH

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

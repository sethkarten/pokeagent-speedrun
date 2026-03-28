#!/usr/bin/env python3
"""Smoke tests for the agents package layout."""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_agents_package_imports():
    import agents
    from agents import PokeAgent, VisionOnlyAgent
    from agents.objectives import DirectObjectiveManager

    assert PokeAgent is not None
    assert VisionOnlyAgent is not None
    assert DirectObjectiveManager is not None
    assert hasattr(agents, "PokeAgent")
    assert hasattr(agents, "VisionOnlyAgent")


def test_pokeagent_adapter_no_create_direct_objectives_mapping():
    from agents.PokeAgent import MCPToolAdapter

    adapter = MCPToolAdapter(server_url="http://localhost:8000")
    result = adapter.call_tool("create_direct_objectives", {"objectives": []})
    assert result["success"] is False
    assert "Unknown tool" in result["error"]


def test_vision_only_adapter_no_create_direct_objectives_mapping():
    from agents.vision_only_agent import MCPToolAdapter

    adapter = MCPToolAdapter(server_url="http://localhost:8000")
    result = adapter.call_tool("create_direct_objectives", {"objectives": []})
    assert result["success"] is False
    assert "Unknown tool" in result["error"]

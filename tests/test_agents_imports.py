#!/usr/bin/env python3
"""Smoke tests for the post-refactor agents package layout."""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_agents_package_imports():
    import agents
    from agents import Agent
    from agents.custom.PokeAgent import PokeAgent
    from agents.custom.vision_only_agent import VisionOnlyAgent
    from agents.objectives import DirectObjectiveManager
    from agents.simple.claude_plays import ClaudePlaysAgent
    from agents.simple.gemini_plays import GeminiPlaysAgent

    assert Agent is not None
    assert PokeAgent is not None
    assert VisionOnlyAgent is not None
    assert DirectObjectiveManager is not None
    assert ClaudePlaysAgent is not None
    assert GeminiPlaysAgent is not None
    assert hasattr(agents, "Agent")

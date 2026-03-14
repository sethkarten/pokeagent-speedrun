#!/usr/bin/env python3
"""Smoke tests for the refactored utils package layout."""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_utils_package_imports():
    from agents.custom.utils.prompt_optimizer import PromptOptimizer
    from utils.agent_infrastructure.cli_agent_backends import CodexCliBackend, get_backend
    from utils.agent_infrastructure.vlm_backends import VLM
    from utils.data_persistence.llm_logger import LLMLogger
    from utils.data_persistence.run_data_manager import initialize_run_data_manager
    from utils.mapping.pathfinding import Pathfinder
    from utils.metric_tracking.server_metrics import update_server_metrics

    assert PromptOptimizer is not None
    assert CodexCliBackend is not None
    assert get_backend is not None
    assert VLM is not None
    assert LLMLogger is not None
    assert initialize_run_data_manager is not None
    assert Pathfinder is not None
    assert update_server_metrics is not None

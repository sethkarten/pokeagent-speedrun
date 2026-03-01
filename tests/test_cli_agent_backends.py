#!/usr/bin/env python3
"""Tests for CLI agent backends and LLMLogger integration."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cli_agent_backends import (
    CliSessionMetrics,
    ClaudeCodeBackend,
    get_backend,
    log_session_to_llm_logger,
)


class TestCliSessionMetrics:
    """Test CliSessionMetrics dataclass."""

    def test_defaults(self):
        m = CliSessionMetrics()
        assert m.session_id == ""
        assert m.total_cost_usd == 0.0
        assert m.tool_use_count == 0

    def test_populated(self):
        m = CliSessionMetrics(
            session_id="s1",
            model="claude-sonnet-4-6",
            total_cost_usd=0.05,
            input_tokens=100,
            output_tokens=50,
            num_turns=2,
            tool_use_count=3,
        )
        assert m.input_tokens == 100
        assert m.output_tokens == 50
        assert m.total_cost_usd == 0.05
        assert m.tool_use_count == 3


class TestDevcontainerBuildContext:
    """Verify devcontainer_build_context points to an existing directory."""

    def test_claude_devcontainer_build_context_exists(self):
        backend = get_backend("claude")
        ctx = backend.devcontainer_build_context
        assert ctx, "devcontainer_build_context must be non-empty"
        project_root = Path(__file__).resolve().parent.parent
        ctx_path = project_root / ctx
        assert ctx_path.is_dir(), f"devcontainer_build_context must be a directory: {ctx_path}"
        dockerfile = ctx_path / "Dockerfile"
        assert dockerfile.exists(), f"Dockerfile must exist in build context: {dockerfile}"


class TestGetBackend:
    def test_claude_returns_claude_code_backend(self):
        backend = get_backend("claude")
        assert isinstance(backend, ClaudeCodeBackend)
        assert backend.name == "claude"

    def test_codex_raises(self):
        with pytest.raises(NotImplementedError, match="Codex CLI"):
            get_backend("codex")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown CLI type"):
            get_backend("unknown")


class TestClaudeCodeBackendBuildLaunchCmd:
    def test_returns_cmd_env_bootstrap_temp_path(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Play Pokemon.")
        backend = ClaudeCodeBackend()
        cmd, env, bootstrap, temp_path = backend.build_launch_cmd(
            str(directive),
            "http://localhost:8000",
            str(tmp_path),
            dangerously_skip_permissions=True,
        )
        assert "claude" in cmd
        assert "--print" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd  # required by Claude Code with stream-json
        assert "--dangerously-skip-permissions" in cmd
        assert "--mcp-config" in cmd
        assert env.get("POKEMON_SERVER_URL") == "http://localhost:8000"
        assert "Play Pokemon." in bootstrap
        assert "Runtime context" in bootstrap
        assert temp_path is not None
        assert temp_path.endswith(".json")
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_handle_stream_event_result_updates_metrics(self):
        backend = ClaudeCodeBackend()
        metrics = CliSessionMetrics()
        event = {
            "type": "result",
            "total_cost_usd": 0.01,
            "num_turns": 2,
            "duration_ms": 5000,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        backend.handle_stream_event(event, metrics)
        assert metrics.total_cost_usd == 0.01
        assert metrics.num_turns == 2
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50


class TestLogSessionToLLMLogger:
    """Verify CLI session metrics are passed to LLMLogger correctly."""

    @patch("utils.llm_logger.log_llm_interaction")
    def test_log_session_to_llm_logger_calls_log_llm_interaction_with_expected_shape(
        self, mock_log
    ):
        metrics = CliSessionMetrics(
            model="claude-sonnet-4-6",
            total_cost_usd=0.02,
            input_tokens=200,
            output_tokens=80,
            num_turns=3,
            tool_use_count=5,
            duration_ms=10_000,
        )
        log_session_to_llm_logger(metrics, session_number=1, backend_name="claude")
        mock_log.assert_called_once()
        call_kw = mock_log.call_args
        assert call_kw.kwargs["interaction_type"] == "cli_claude"
        assert call_kw.kwargs["step_number"] == 1
        assert call_kw.kwargs["model_info"]["model"] == "claude-sonnet-4-6"
        assert call_kw.kwargs["model_info"]["backend"] == "claude"
        meta = call_kw.kwargs["metadata"]
        assert "token_usage" in meta
        assert meta["token_usage"]["prompt_tokens"] == 200
        assert meta["token_usage"]["completion_tokens"] == 80
        assert call_kw.kwargs["duration"] == 10.0

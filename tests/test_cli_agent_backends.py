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
    CodexCliBackend,
    GeminiCliBackend,
    get_backend,
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

    @pytest.mark.parametrize("cli_type", ["claude", "gemini", "codex"])
    def test_devcontainer_build_context_exists(self, cli_type):
        backend = get_backend(cli_type)
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

    def test_gemini_returns_gemini_cli_backend(self):
        backend = get_backend("gemini")
        assert isinstance(backend, GeminiCliBackend)
        assert backend.name == "gemini"

    def test_codex_returns_codex_cli_backend(self):
        backend = get_backend("codex")
        assert isinstance(backend, CodexCliBackend)
        assert backend.name == "codex"

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


class TestClaudeCodeBackendStreamEvent:
    """Verify handle_stream_event correctly updates CliSessionMetrics."""

    def test_handle_stream_event_tool_use_increments_count(self):
        backend = ClaudeCodeBackend()
        metrics = CliSessionMetrics()
        event = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}}
        backend.handle_stream_event(event, metrics)
        assert metrics.tool_use_count == 1

    def test_handle_stream_event_result_stores_model(self):
        backend = ClaudeCodeBackend()
        metrics = CliSessionMetrics()
        event = {
            "type": "result",
            "total_cost_usd": 0.05,
            "num_turns": 5,
            "duration_ms": 8000,
            "usage": {"input_tokens": 300, "output_tokens": 100},
        }
        backend.handle_stream_event(event, metrics)
        assert metrics.total_cost_usd == 0.05
        assert metrics.num_turns == 5


class TestCodexCliBackendBuildLaunchCmd:
    def test_returns_cmd_with_exec_json(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Play Pokemon.")
        backend = CodexCliBackend()
        cmd, env, bootstrap, temp_path = backend.build_launch_cmd(
            str(directive),
            "http://localhost:8000",
            str(tmp_path),
            dangerously_skip_permissions=True,
        )
        assert "codex" in " ".join(cmd)
        assert "exec" in " ".join(cmd)
        assert "--json" in " ".join(cmd)
        assert "Play Pokemon." in bootstrap
        assert temp_path is None

    def test_resume_session_id_appended(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Play Pokemon.")
        backend = CodexCliBackend()
        cmd, _, _, _ = backend.build_launch_cmd(
            str(directive),
            "http://localhost:8000",
            str(tmp_path),
            dangerously_skip_permissions=True,
            resume_session_id="abc-123",
        )
        cmd_str = " ".join(cmd)
        assert "resume" in cmd_str
        assert "abc-123" in cmd_str

    def test_resume_last_appended(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Play Pokemon.")
        backend = CodexCliBackend()
        cmd, _, _, _ = backend.build_launch_cmd(
            str(directive),
            "http://localhost:8000",
            str(tmp_path),
            dangerously_skip_permissions=True,
            resume_session_id="--last",
        )
        cmd_str = " ".join(cmd)
        assert "resume" in cmd_str
        assert "--last" in cmd_str


class TestCodexCliBackendGetResumeSessionId:
    def test_empty_dir_returns_none(self, tmp_path):
        backend = CodexCliBackend()
        assert backend.get_resume_session_id(tmp_path) is None

    def test_sessions_dir_with_file_returns_stem(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "abc-123.jsonl").write_text("{}")
        backend = CodexCliBackend()
        result = backend.get_resume_session_id(tmp_path)
        assert result == "abc-123"

    def test_returns_most_recent_session(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "old.jsonl").write_text("{}")
        (sessions_dir / "new.jsonl").write_text("{}")
        backend = CodexCliBackend()
        result = backend.get_resume_session_id(tmp_path)
        assert result in ("old", "new")


class TestCodexCliBackendHandleStreamEvent:
    def test_thread_started_sets_session_id(self):
        backend = CodexCliBackend()
        metrics = CliSessionMetrics()
        event = {"type": "thread.started", "thread_id": "tid-123"}
        backend.handle_stream_event(event, metrics)
        assert metrics.session_id == "tid-123"

    def test_turn_completed_updates_metrics(self):
        backend = CodexCliBackend()
        metrics = CliSessionMetrics()
        event = {
            "type": "turn.completed",
            "usage": {"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 10},
        }
        backend.handle_stream_event(event, metrics)
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50

    def test_mcp_tool_call_increments_count(self):
        backend = CodexCliBackend()
        metrics = CliSessionMetrics()
        event = {
            "type": "item.completed",
            "item": {"type": "mcp_tool_call", "tool": "read_file", "arguments": {}},
        }
        backend.handle_stream_event(event, metrics)
        assert metrics.tool_use_count == 1

    def test_turn_failed_sets_error(self):
        backend = CodexCliBackend()
        metrics = CliSessionMetrics()
        event = {"type": "turn.failed", "error": {"message": "model failed"}}
        backend.handle_stream_event(event, metrics)
        assert metrics.is_error is True
        assert "failed" in metrics.error

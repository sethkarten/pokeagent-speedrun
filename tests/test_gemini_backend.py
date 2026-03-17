#!/usr/bin/env python3
"""Tests for GeminiCliBackend."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.agent_infrastructure.cli_agent_backends import (
    CliSessionMetrics,
    GeminiCliBackend,
    get_backend,
)
# ── GeminiCliBackend basic properties ─────────────────────────────────────────


class TestGeminiBackendProperties:
    def test_get_backend_returns_gemini(self):
        backend = get_backend("gemini")
        assert isinstance(backend, GeminiCliBackend)
        assert backend.name == "GeminiCLI"

    def test_agent_memory_subdir(self):
        backend = GeminiCliBackend()
        assert backend.agent_memory_subdir == "gemini_memory"

    def test_container_image(self):
        backend = GeminiCliBackend()
        assert backend.container_image == "gemini-agent-devcontainer"

    def test_devcontainer_build_context_exists(self):
        backend = GeminiCliBackend()
        project_root = Path(__file__).resolve().parent.parent
        ctx_path = project_root / backend.devcontainer_build_context
        assert ctx_path.is_dir(), f"devcontainer dir must exist: {ctx_path}"
        assert (ctx_path / "Dockerfile").exists()
        assert (ctx_path / "init-firewall.sh").exists()


# ── build_launch_cmd ──────────────────────────────────────────────────────────


class TestGeminiBuildLaunchCmd:
    def test_local_cmd_structure(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Play Pokemon.")
        backend = GeminiCliBackend()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            cmd, env, bootstrap, temp_path = backend.build_launch_cmd(
                str(directive),
                "http://localhost:8000",
                str(tmp_path),
            )
        assert "gemini" in cmd
        assert "--yolo" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "-p" in cmd
        assert "Play Pokemon." in bootstrap
        assert temp_path is None

    def test_local_writes_gemini_settings(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Test.")
        backend = GeminiCliBackend()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            backend.build_launch_cmd(
                str(directive),
                "http://localhost:8000",
                str(tmp_path),
            )
        settings_path = tmp_path / ".gemini" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "mcpServers" in settings

    def test_thinking_effort_adds_model_config_overrides(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Test.")
        backend = GeminiCliBackend()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            backend.build_launch_cmd(
                str(directive),
                "http://localhost:8000",
                str(tmp_path),
                thinking_effort="high",
            )
        settings_path = tmp_path / ".gemini" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "modelConfigs" in settings
        overrides = settings["modelConfigs"].get("customOverrides", [])
        assert len(overrides) > 0
        first = overrides[0]
        assert "generateContentConfig" in first
        assert "thinkingConfig" in first["generateContentConfig"]

    def test_resume_session_id_appended(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Test.")
        backend = GeminiCliBackend()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            cmd, *_ = backend.build_launch_cmd(
                str(directive),
                "http://localhost:8000",
                str(tmp_path),
                resume_session_id="abc-123",
            )
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"

    def test_containerized_requires_run_id(self, tmp_path):
        backend = GeminiCliBackend()
        with pytest.raises(ValueError, match="run_id"):
            backend.build_launch_cmd(
                "", "http://localhost:8000", str(tmp_path),
                containerized=True, mcp_sse_port=8002,
            )

    def test_containerized_cmd_has_docker(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Test.")
        agent_mem = tmp_path / "gemini_memory"
        agent_mem.mkdir()
        backend = GeminiCliBackend()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            cmd, env, bootstrap, _ = backend.build_launch_cmd(
                str(directive),
                "http://localhost:8000",
                str(tmp_path),
                containerized=True,
                run_id="test_run",
                agent_memory_dir=str(agent_mem),
                mcp_sse_port=8002,
            )
        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert "gemini-agent-devcontainer" in cmd
        assert any("GEMINI_API_KEY" in c for c in cmd)

    def test_containerized_writes_settings_to_agent_memory(self, tmp_path):
        directive = tmp_path / "directive.md"
        directive.write_text("Test.")
        agent_mem = tmp_path / "gemini_memory"
        agent_mem.mkdir()
        backend = GeminiCliBackend()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            backend.build_launch_cmd(
                str(directive),
                "http://localhost:8000",
                str(tmp_path),
                containerized=True,
                run_id="test_run",
                agent_memory_dir=str(agent_mem),
                mcp_sse_port=8002,
            )
        settings_path = agent_mem / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert settings["mcpServers"]["pokemon-emerald"]["url"] == "http://host.docker.internal:8002/sse"


# ── handle_stream_event ───────────────────────────────────────────────────────


class TestGeminiStreamEvents:
    def test_init_event_sets_session_and_model(self):
        backend = GeminiCliBackend()
        metrics = CliSessionMetrics()
        backend.handle_stream_event(
            {"type": "init", "session_id": "s1", "model": "gemini-2.5-pro"},
            metrics,
        )
        assert metrics.session_id == "s1"
        assert metrics.model == "gemini-2.5-pro"

    def test_tool_use_increments_count(self):
        backend = GeminiCliBackend()
        metrics = CliSessionMetrics()
        backend.handle_stream_event(
            {"type": "tool_use", "name": "read_file", "arguments": {"path": "/foo"}},
            metrics,
        )
        assert metrics.tool_use_count == 1

    def test_tool_use_gemini_format_tool_name_parameters(self):
        """Gemini stream-json uses tool_name and parameters."""
        backend = GeminiCliBackend()
        metrics = CliSessionMetrics()
        backend.handle_stream_event(
            {"type": "tool_use", "tool_name": "get_game_state", "parameters": {}},
            metrics,
        )
        assert metrics.tool_use_count == 1

    def test_tool_use_posts_reasoning_to_server(self):
        """Tool use with reasoning in parameters is posted for UI streaming (like Claude)."""
        backend = GeminiCliBackend()
        metrics = CliSessionMetrics()
        with patch.object(backend, "_post_thinking") as mock_post:
            backend.handle_stream_event(
                {
                    "type": "tool_use",
                    "tool_name": "press_buttons",
                    "parameters": {
                        "buttons": ["A", "B"],
                        "reasoning": "Moving right to exit the moving van",
                        "speed": "normal",
                    },
                },
                metrics,
                server_url="http://localhost:8118",
            )
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][1] == "[press_buttons] Moving right to exit the moving van"

    def test_result_event_updates_metrics(self):
        backend = GeminiCliBackend()
        metrics = CliSessionMetrics()
        backend.handle_stream_event(
            {
                "type": "result",
                "stats": {"input_tokens": 500, "output_tokens": 200, "duration_ms": 3000},
            },
            metrics,
        )
        assert metrics.input_tokens == 500
        assert metrics.output_tokens == 200
        assert metrics.duration_ms == 3000

    def test_error_event_does_not_crash(self):
        backend = GeminiCliBackend()
        metrics = CliSessionMetrics()
        backend.handle_stream_event(
            {"type": "error", "message": "quota exceeded"},
            metrics,
        )

    def test_message_event_does_not_crash(self):
        backend = GeminiCliBackend()
        metrics = CliSessionMetrics()
        backend.handle_stream_event(
            {"type": "message", "role": "model", "content": "Hello!"},
            metrics,
        )


# ── get_resume_session_id ─────────────────────────────────────────────────────


class TestGeminiResumeSession:
    def test_returns_none_for_empty_dir(self, tmp_path):
        backend = GeminiCliBackend()
        assert backend.get_resume_session_id(tmp_path) is None

    def test_finds_most_recent_session(self, tmp_path):
        chats = tmp_path / "tmp" / "workspace1" / "chats"
        chats.mkdir(parents=True)
        import time
        (chats / "old-session-id.json").write_text("{}")
        time.sleep(0.05)
        (chats / "new-session-id.json").write_text("{}")
        backend = GeminiCliBackend()
        result = backend.get_resume_session_id(tmp_path)
        assert result == "new-session-id"

    def test_ignores_non_json_files(self, tmp_path):
        chats = tmp_path / "tmp" / "ws" / "chats"
        chats.mkdir(parents=True)
        (chats / "not-a-session.txt").write_text("random")
        backend = GeminiCliBackend()
        assert backend.get_resume_session_id(tmp_path) is None


# ── run_login ─────────────────────────────────────────────────────────────────


class TestGeminiRunLogin:
    def test_always_returns_true(self):
        backend = GeminiCliBackend()
        assert backend.run_login() is True


# ── log_cli_interaction integration ───────────────────────────────────────────


class TestGeminiLogCliInteraction:
    def _make_session_with_messages(self, *messages):
        return {"sessionId": "test-session", "messages": list(messages)}

    def _make_gemini_message(self, msg_id, input_t=100, output_t=50):
        return {
            "id": msg_id,
            "type": "gemini",
            "timestamp": "2026-03-09T12:00:00Z",
            "model": "gemini-2.5-pro",
            "tokens": {
                "input": input_t,
                "output": output_t,
                "cached": 0,
                "thoughts": 0,
                "tool": 0,
                "total": input_t + output_t,
            },
            "toolCalls": [],
        }

    def test_appends_steps_to_llm_logger(self, tmp_path, monkeypatch):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        chats_dir.mkdir(parents=True)
        session_file = chats_dir / "session-2026-03-09T12-00-test.json"
        session = self._make_session_with_messages(
            self._make_gemini_message("msg-1", 100, 50),
            self._make_gemini_message("msg-2", 200, 80),
        )
        session_file.write_text(json.dumps(session))

        mock_logger = MagicMock()
        monkeypatch.setattr(
            "utils.agent_infrastructure.cli_agent_backends.GeminiCliBackend._sync_metrics_to_server",
            lambda *a, **kw: None,
        )

        with patch("utils.data_persistence.llm_logger.get_llm_logger", return_value=mock_logger):
            backend = GeminiCliBackend()
            hashes, last_step = backend.log_cli_interaction(
                tmp_path, set(), -1, server_url="http://localhost:8000"
            )

        assert last_step == 1  # started at -1, two entries → 0, 1
        assert mock_logger.append_cli_step.call_count == 2
        assert len(hashes) == 2

    def test_no_entries_returns_unchanged(self, tmp_path):
        backend = GeminiCliBackend()
        hashes, last_step = backend.log_cli_interaction(tmp_path, set(), 5)
        assert last_step == 5
        assert hashes == set()

#!/usr/bin/env python3
"""Tests for GeminiCliBackend and gemini_telemetry_reader."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cli_agent_backends import (
    CliSessionMetrics,
    GeminiCliBackend,
    get_backend,
)
from utils.metric_tracking.gemini_telemetry_reader import (
    find_telemetry_files,
    load_new_gemini_usage,
    _extract_api_response,
    _make_dedup_hash,
    API_RESPONSE_EVENT,
)


# ── GeminiCliBackend basic properties ─────────────────────────────────────────


class TestGeminiBackendProperties:
    def test_get_backend_returns_gemini(self):
        backend = get_backend("gemini")
        assert isinstance(backend, GeminiCliBackend)
        assert backend.name == "gemini"

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
        assert "telemetry" in settings
        assert settings["telemetry"]["enabled"] is True
        assert "pokemon-emerald" in settings["mcpServers"]
        assert settings["mcpServers"]["pokemon-emerald"]["trust"] is True

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


# ── gemini_telemetry_reader ───────────────────────────────────────────────────


class TestFindTelemetryFiles:
    def test_finds_jsonl_and_log_files(self, tmp_path):
        (tmp_path / "telemetry.jsonl").write_text("")
        (tmp_path / "other.log").write_text("")
        (tmp_path / "not_telemetry.txt").write_text("")
        files = find_telemetry_files(tmp_path)
        assert len(files) == 2

    def test_empty_dir_returns_empty(self, tmp_path):
        assert find_telemetry_files(tmp_path) == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        assert find_telemetry_files(tmp_path / "nonexistent") == []

    def test_recursive_search(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        (nested / "deep.jsonl").write_text("")
        files = find_telemetry_files(tmp_path)
        assert len(files) == 1


class TestExtractApiResponse:
    def _make_record(self, **attrs):
        return {"body": API_RESPONSE_EVENT, "attributes": attrs}

    def test_extracts_token_counts(self):
        record = self._make_record(
            model="gemini-2.5-pro",
            input_token_count=100,
            output_token_count=50,
            cached_content_token_count=20,
            thoughts_token_count=10,
            tool_token_count=5,
            total_token_count=185,
            duration_ms=1500,
            prompt_id="p1",
        )
        entry = _extract_api_response(record)
        assert entry is not None
        assert entry["prompt"] == 100
        assert entry["completion"] == 65  # 50 + 10 + 5
        assert entry["cached"] == 20
        assert entry["total"] == 185
        assert entry["_model"] == "gemini-2.5-pro"
        assert entry["_duration_ms"] == 1500

    def test_computes_total_when_zero(self):
        record = self._make_record(
            input_token_count=100,
            output_token_count=50,
            total_token_count=0,
        )
        entry = _extract_api_response(record)
        assert entry["total"] == 150

    def test_returns_none_for_non_api_response(self):
        record = {"body": "gemini_cli.tool_call", "attributes": {"model": "x"}}
        assert _extract_api_response(record) is None

    def test_returns_none_for_empty_attributes(self):
        record = {"body": API_RESPONSE_EVENT}
        assert _extract_api_response(record) is None

    def test_handles_resource_as_non_dict(self):
        """Regression: resource can be int/other; must not call .get() on it."""
        record = {"body": API_RESPONSE_EVENT, "resource": 123}
        assert _extract_api_response(record) is None

    def test_handles_non_dict_record(self):
        """Regression: JSON line can be non-dict (number, array)."""
        assert _extract_api_response(42) is None
        assert _extract_api_response([1, 2]) is None


class TestMakeDedupHash:
    def test_same_input_same_hash(self):
        entry = {"_prompt_id": "p1", "_timestamp": "t1", "_model": "m1", "prompt": 100, "completion": 50}
        h1 = _make_dedup_hash(entry)
        h2 = _make_dedup_hash(entry)
        assert h1 == h2

    def test_different_input_different_hash(self):
        e1 = {"_prompt_id": "p1", "_timestamp": "t1", "_model": "m1", "prompt": 100, "completion": 50}
        e2 = {"_prompt_id": "p2", "_timestamp": "t2", "_model": "m1", "prompt": 100, "completion": 50}
        assert _make_dedup_hash(e1) != _make_dedup_hash(e2)


class TestLoadNewGeminiUsage:
    def _write_telemetry(self, path: Path, records: list[dict]):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _make_api_response(self, prompt_id="p1", input_t=100, output_t=50, model="gemini-2.5-pro"):
        return {
            "body": API_RESPONSE_EVENT,
            "timestamp": "2026-03-09T12:00:00Z",
            "attributes": {
                "model": model,
                "input_token_count": input_t,
                "output_token_count": output_t,
                "total_token_count": input_t + output_t,
                "duration_ms": 1000,
                "prompt_id": prompt_id,
            },
        }

    def test_loads_api_response_entries(self, tmp_path):
        self._write_telemetry(
            tmp_path / "telemetry.jsonl",
            [self._make_api_response("p1"), self._make_api_response("p2")],
        )
        entries, hashes, offsets = load_new_gemini_usage(tmp_path, set())
        assert len(entries) == 2
        assert len(hashes) == 2

    def test_skips_non_api_response_events(self, tmp_path):
        records = [
            {"body": "gemini_cli.tool_call", "attributes": {"function_name": "read_file"}},
            self._make_api_response("p1"),
            {"body": "gemini_cli.config", "attributes": {"model": "gemini-pro"}},
        ]
        self._write_telemetry(tmp_path / "telemetry.jsonl", records)
        entries, _, _ = load_new_gemini_usage(tmp_path, set())
        assert len(entries) == 1
        assert entries[0]["_model"] == "gemini-2.5-pro"

    def test_deduplication_across_polls(self, tmp_path):
        self._write_telemetry(
            tmp_path / "telemetry.jsonl",
            [self._make_api_response("p1")],
        )
        entries1, hashes1, offsets1 = load_new_gemini_usage(tmp_path, set())
        assert len(entries1) == 1

        entries2, hashes2, offsets2 = load_new_gemini_usage(tmp_path, hashes1, offsets1)
        assert len(entries2) == 0
        assert hashes2 == hashes1

    def test_incremental_read_via_offsets(self, tmp_path):
        tfile = tmp_path / "telemetry.jsonl"
        self._write_telemetry(tfile, [self._make_api_response("p1")])
        entries1, hashes1, offsets1 = load_new_gemini_usage(tmp_path, set())
        assert len(entries1) == 1

        # Append a new entry
        with open(tfile, "a") as f:
            f.write(json.dumps(self._make_api_response("p2")) + "\n")

        entries2, hashes2, offsets2 = load_new_gemini_usage(tmp_path, hashes1, offsets1)
        assert len(entries2) == 1
        assert entries2[0]["_model"] == "gemini-2.5-pro"

    def test_handles_malformed_lines(self, tmp_path):
        tfile = tmp_path / "telemetry.jsonl"
        with open(tfile, "w") as f:
            f.write("not json\n")
            f.write(json.dumps(self._make_api_response("p1")) + "\n")
            f.write("{\n")
        entries, _, _ = load_new_gemini_usage(tmp_path, set())
        assert len(entries) == 1

    def test_handles_file_truncation(self, tmp_path):
        tfile = tmp_path / "telemetry.jsonl"
        self._write_telemetry(tfile, [self._make_api_response(f"p{i}") for i in range(5)])
        _, _, offsets = load_new_gemini_usage(tmp_path, set())

        # Simulate truncation (new file smaller than offset)
        tfile.write_text(json.dumps(self._make_api_response("pnew")) + "\n")
        entries, _, _ = load_new_gemini_usage(tmp_path, set(), offsets)
        assert len(entries) == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        entries, hashes, offsets = load_new_gemini_usage(tmp_path, set())
        assert entries == []
        assert hashes == set()

    def test_skips_zero_total_entries(self, tmp_path):
        record = {
            "body": API_RESPONSE_EVENT,
            "attributes": {
                "model": "gemini-2.5-pro",
                "input_token_count": 0,
                "output_token_count": 0,
                "total_token_count": 0,
                "prompt_id": "empty",
            },
        }
        self._write_telemetry(tmp_path / "telemetry.jsonl", [record])
        entries, _, _ = load_new_gemini_usage(tmp_path, set())
        assert len(entries) == 0


# ── log_cli_interaction integration ───────────────────────────────────────────


class TestGeminiLogCliInteraction:
    def _make_api_response(self, prompt_id, input_t=100, output_t=50):
        return {
            "body": API_RESPONSE_EVENT,
            "timestamp": "2026-03-09T12:00:00Z",
            "attributes": {
                "model": "gemini-2.5-pro",
                "input_token_count": input_t,
                "output_token_count": output_t,
                "total_token_count": input_t + output_t,
                "duration_ms": 1000,
                "prompt_id": prompt_id,
            },
        }

    def test_appends_steps_to_llm_logger(self, tmp_path, monkeypatch):
        tfile = tmp_path / "telemetry.jsonl"
        with open(tfile, "w") as f:
            f.write(json.dumps(self._make_api_response("p1")) + "\n")
            f.write(json.dumps(self._make_api_response("p2")) + "\n")

        mock_logger = MagicMock()
        monkeypatch.setattr(
            "utils.cli_agent_backends.GeminiCliBackend._sync_metrics_to_server",
            lambda *a, **kw: None,
        )

        with patch("utils.llm_logger.get_llm_logger", return_value=mock_logger):
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

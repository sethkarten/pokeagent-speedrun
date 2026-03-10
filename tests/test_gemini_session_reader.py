#!/usr/bin/env python3
"""Tests for utils/metric_tracking/gemini_session_reader.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metric_tracking.gemini_session_reader import (
    extract_tokens_from_message,
    extract_tool_calls_from_message,
    find_session_files,
    load_new_usage_entries,
)


# ---------------------------------------------------------------------------
# Fixtures – representative Gemini session message shapes
# ---------------------------------------------------------------------------

GEMINI_MSG_FULL = {
    "id": "msg-abc123",
    "type": "gemini",
    "timestamp": "2026-03-09T12:00:00Z",
    "model": "gemini-2.5-pro",
    "tokens": {
        "input": 100,
        "output": 50,
        "cached": 200,
        "thoughts": 10,
        "tool": 5,
        "total": 365,
    },
    "toolCalls": [
        {"name": "read_file", "args": {"path": "/foo.py"}},
        {"name": "run_terminal_cmd", "args": {}},
    ],
}

GEMINI_MSG_MINIMAL = {
    "id": "msg-min",
    "type": "gemini",
    "timestamp": "2026-03-09T12:01:00Z",
    "tokens": {"input": 50, "output": 20, "total": 70},
    "toolCalls": [],
}

GEMINI_MSG_COMPUTED_TOTAL = {
    "id": "msg-computed",
    "type": "gemini",
    "tokens": {
        "input": 10,
        "output": 5,
        "thoughts": 2,
        "tool": 1,
        "cached": 0,
        "total": 0,
    },
    "toolCalls": [],
}


# ---------------------------------------------------------------------------
# Unit tests for token extraction
# ---------------------------------------------------------------------------

class TestExtractTokensFromMessage:
    def test_full_entry_returns_all_buckets(self):
        tokens = extract_tokens_from_message(GEMINI_MSG_FULL)
        assert tokens is not None
        assert tokens["prompt"] == 100
        assert tokens["completion"] == 50 + 10 + 5  # output + thoughts + tool
        assert tokens["cached"] == 200
        assert tokens["total"] == 365
        # Gemini uses implicit caching; cache_write not in session format → None
        assert tokens["cache_write"] is None

    def test_cache_write_none_when_absent(self):
        """Gemini session format does not expose cache_write; expect None."""
        msg = {"id": "x", "type": "gemini", "tokens": {"input": 10, "output": 5, "total": 15}}
        tokens = extract_tokens_from_message(msg)
        assert tokens is not None
        assert tokens["cache_write"] is None

    def test_cache_write_from_session_when_present(self):
        """If session adds cache_write/cacheWrite in future, we read it."""
        msg = {
            "id": "x",
            "type": "gemini",
            "tokens": {"input": 10, "output": 5, "cache_write": 100, "total": 115},
        }
        tokens = extract_tokens_from_message(msg)
        assert tokens is not None
        assert tokens["cache_write"] == 100

    def test_minimal_entry(self):
        tokens = extract_tokens_from_message(GEMINI_MSG_MINIMAL)
        assert tokens is not None
        assert tokens["prompt"] == 50
        assert tokens["completion"] == 20
        assert tokens["total"] == 70

    def test_computes_total_when_zero(self):
        tokens = extract_tokens_from_message(GEMINI_MSG_COMPUTED_TOTAL)
        assert tokens is not None
        assert tokens["total"] == 10 + 5 + 2 + 1

    def test_no_tokens_dict_returns_none(self):
        msg = {"id": "x", "type": "gemini", "tokens": None}
        assert extract_tokens_from_message(msg) is None

    def test_tokens_not_dict_returns_none(self):
        msg = {"id": "x", "type": "gemini", "tokens": "invalid"}
        assert extract_tokens_from_message(msg) is None


# ---------------------------------------------------------------------------
# Unit tests for tool call extraction
# ---------------------------------------------------------------------------

class TestExtractToolCallsFromMessage:
    def test_extracts_tool_calls(self):
        calls = extract_tool_calls_from_message(GEMINI_MSG_FULL)
        assert len(calls) == 2
        assert calls[0] == {"name": "read_file", "args": {"path": "/foo.py"}}
        assert calls[1] == {"name": "run_terminal_cmd", "args": {}}

    def test_empty_tool_calls(self):
        assert extract_tool_calls_from_message(GEMINI_MSG_MINIMAL) == []

    def test_no_tool_calls_key(self):
        msg = {"id": "x", "type": "gemini"}
        assert extract_tool_calls_from_message(msg) == []

    def test_tool_calls_not_list(self):
        msg = {"id": "x", "type": "gemini", "toolCalls": "invalid"}
        assert extract_tool_calls_from_message(msg) == []

    def test_skips_invalid_tool_call_entries(self):
        msg = {
            "id": "x",
            "type": "gemini",
            "toolCalls": [{"name": "ok", "args": {}}, None, {}, {"args": {}}],
        }
        calls = extract_tool_calls_from_message(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "ok"


# ---------------------------------------------------------------------------
# Unit tests for find_session_files
# ---------------------------------------------------------------------------

class TestFindSessionFiles:
    def test_finds_session_files_sorted_by_mtime(self, tmp_path):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        chats_dir.mkdir(parents=True)
        a = chats_dir / "session-a.json"
        b = chats_dir / "session-b.json"
        a.write_text("{}")
        b.write_text("{}")
        import time
        time.sleep(0.01)
        a.touch()
        files = find_session_files(tmp_path)
        assert len(files) == 2
        assert files[-1] == a  # most recently modified last

    def test_empty_dir_returns_empty(self, tmp_path):
        assert find_session_files(tmp_path) == []

    def test_nonexistent_chats_dir_returns_empty(self, tmp_path):
        assert find_session_files(tmp_path) == []

    def test_ignores_non_session_files(self, tmp_path):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        chats_dir.mkdir(parents=True)
        (chats_dir / "other.json").write_text("{}")
        (chats_dir / "session-valid.json").write_text("{}")
        files = find_session_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "session-valid.json"


# ---------------------------------------------------------------------------
# Integration tests for load_new_usage_entries
# ---------------------------------------------------------------------------

class TestLoadNewUsageEntries:
    def _write_session(self, path: Path, session: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session))

    def test_loads_gemini_messages(self, tmp_path):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        session = {
            "sessionId": "s1",
            "messages": [
                {
                    "id": "m1",
                    "type": "gemini",
                    "tokens": {"input": 10, "output": 5, "total": 15},
                    "toolCalls": [],
                },
            ],
        }
        self._write_session(chats_dir / "session-1.json", session)
        entries, hashes = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 1
        assert entries[0]["_tokens"]["total"] == 15
        assert "m1" in hashes

    def test_deduplication_across_polls(self, tmp_path):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        session = {
            "sessionId": "s1",
            "messages": [
                {"id": "m1", "type": "gemini", "tokens": {"input": 10, "output": 5, "total": 15}, "toolCalls": []},
            ],
        }
        self._write_session(chats_dir / "session-1.json", session)
        _, hashes1 = load_new_usage_entries(tmp_path, set())
        entries2, hashes2 = load_new_usage_entries(tmp_path, hashes1)
        assert entries2 == []
        assert hashes2 == hashes1

    def test_skips_non_gemini_messages(self, tmp_path):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        session = {
            "sessionId": "s1",
            "messages": [
                {"id": "u1", "type": "user", "content": "hi"},
                {"id": "m1", "type": "gemini", "tokens": {"input": 10, "output": 5, "total": 15}, "toolCalls": []},
            ],
        }
        self._write_session(chats_dir / "session-1.json", session)
        entries, _ = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 1

    def test_skips_zero_total_entries(self, tmp_path):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        session = {
            "sessionId": "s1",
            "messages": [
                {"id": "m1", "type": "gemini", "tokens": {"input": 0, "output": 0, "total": 0}, "toolCalls": []},
            ],
        }
        self._write_session(chats_dir / "session-1.json", session)
        entries, _ = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 0

    def test_skips_messages_without_id(self, tmp_path):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        session = {
            "sessionId": "s1",
            "messages": [
                {"type": "gemini", "tokens": {"input": 10, "output": 5, "total": 15}, "toolCalls": []},
            ],
        }
        self._write_session(chats_dir / "session-1.json", session)
        entries, _ = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 0

    def test_handles_malformed_json(self, tmp_path):
        chats_dir = tmp_path / "tmp" / "workspace" / "chats"
        chats_dir.mkdir(parents=True)
        (chats_dir / "session-bad.json").write_text("{ invalid json")
        entries, _ = load_new_usage_entries(tmp_path, set())
        assert entries == []

    def test_empty_dir_returns_empty(self, tmp_path):
        entries, hashes = load_new_usage_entries(tmp_path, set())
        assert entries == []
        assert hashes == set()

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        entries, hashes = load_new_usage_entries(tmp_path / "nonexistent", set())
        assert entries == []
        assert hashes == set()


# ---------------------------------------------------------------------------
# Cost calculation: Gemini subset style (cached within prompt)
# ---------------------------------------------------------------------------

class TestGeminiCostCalculationSubsetStyle:
    """Verify append_cli_step correctly computes cost when cached is a subset of prompt."""

    def test_gemini_subset_cost_uses_uncached_plus_cached_rates(self, tmp_path):
        """Gemini: prompt=total input, cached=subset. Cost = uncached*prompt_rate + cached*cached_rate."""
        from unittest.mock import patch
        from utils.llm_logger import LLMLogger

        metrics_file = tmp_path / "cumulative_metrics.json"
        with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
            ll = LLMLogger(log_dir=str(tmp_path), session_id="cost_test")

        # Real Gemini step shape: prompt=12630 (total input), cached=7703 (subset), completion=10
        token_usage = {
            "prompt": 12630,
            "completion": 10,
            "cached": 7703,
            "cache_write": None,
            "total": 12640,
        }
        ll.append_cli_step(
            step_number=0,
            token_usage=token_usage,
            duration=0.0,
            timestamp=0.0,
            model_info={"model": "gemini-3-flash"},
        )

        # uncached = 12630 - 7703 = 4927; cost = 4927*0.0005 + 7703*0.00005 + 10*0.003
        cost = ll.cumulative_metrics["total_cost"]
        expected = (4927 / 1000) * 0.0005 + (7703 / 1000) * 0.00005 + (10 / 1000) * 0.003
        assert abs(cost - expected) < 1e-9, f"Cost {cost} != expected {expected}"

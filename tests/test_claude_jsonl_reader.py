#!/usr/bin/env python3
"""Tests for utils/claude_jsonl_reader.py (Issues 9, 10, 11)."""

import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.claude_jsonl_reader import (
    _create_unique_hash,
    _parse_timestamp,
    extract_tokens_from_entry,
    extract_tool_calls_from_entry,
    find_jsonl_files,
    load_new_usage_entries,
)


# ---------------------------------------------------------------------------
# Fixtures – representative JSONL entry shapes from real Claude Code sessions
# ---------------------------------------------------------------------------

ASSISTANT_ENTRY_FULL = {
    "type": "assistant",
    "message": {
        "id": "msg_01ABC",
        "model": "claude-sonnet-4-6",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_001",
                "name": "Read",
                "input": {"file_path": "/workspace/foo.py"},
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 1500,
        },
    },
    "requestId": "req_XXYY",
    "uuid": "uuid-aaa",
    "timestamp": "2026-01-19T10:00:00.000Z",
}

ASSISTANT_ENTRY_NO_CACHE = {
    "type": "assistant",
    "message": {
        "id": "msg_02DEF",
        "model": "claude-sonnet-4-6",
        "role": "assistant",
        "content": [],
        "usage": {
            "input_tokens": 50,
            "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
    "requestId": "req_AABB",
    "uuid": "uuid-bbb",
    "timestamp": "2026-01-19T10:00:05.000Z",
}

USER_ENTRY = {
    "type": "user",
    "message": {"role": "user", "content": "Hello"},
    "uuid": "uuid-ccc",
    "timestamp": "2026-01-19T09:59:00.000Z",
}

MALFORMED_USAGE_ENTRY = {
    "type": "assistant",
    "message": {"id": "msg_03", "usage": None},
    "requestId": "req_CC",
    "uuid": "uuid-ddd",
    "timestamp": "2026-01-19T10:00:10.000Z",
}


# ---------------------------------------------------------------------------
# Issue 9: Unit tests for token extraction
# ---------------------------------------------------------------------------

class TestExtractTokensFromEntry:
    def test_full_entry_returns_all_buckets(self):
        tokens = extract_tokens_from_entry(ASSISTANT_ENTRY_FULL)
        assert tokens is not None
        assert tokens["prompt"] == 10
        assert tokens["completion"] == 5
        assert tokens["cache_write"] == 200
        assert tokens["cached"] == 1500
        assert tokens["total"] == 10 + 5 + 200 + 1500

    def test_no_cache_entry(self):
        tokens = extract_tokens_from_entry(ASSISTANT_ENTRY_NO_CACHE)
        assert tokens is not None
        assert tokens["prompt"] == 50
        assert tokens["completion"] == 20
        assert tokens["cache_write"] == 0
        assert tokens["cached"] == 0
        assert tokens["total"] == 70

    def test_user_entry_returns_none(self):
        assert extract_tokens_from_entry(USER_ENTRY) is None

    def test_malformed_usage_returns_none(self):
        assert extract_tokens_from_entry(MALFORMED_USAGE_ENTRY) is None

    def test_top_level_usage_fallback(self):
        entry = {
            "type": "assistant",
            "usage": {"input_tokens": 3, "output_tokens": 7},
            "requestId": "req_TL",
            "uuid": "uuid-tl",
        }
        tokens = extract_tokens_from_entry(entry)
        assert tokens is not None
        assert tokens["prompt"] == 3
        assert tokens["completion"] == 7
        assert tokens["total"] == 10

    def test_missing_message_key_falls_back_to_top_level(self):
        entry = {
            "type": "assistant",
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "uuid": "x",
        }
        tokens = extract_tokens_from_entry(entry)
        assert tokens is not None


class TestExtractToolCalls:
    def test_extracts_tool_use_blocks(self):
        tools = extract_tool_calls_from_entry(ASSISTANT_ENTRY_FULL)
        assert len(tools) == 1
        assert tools[0]["name"] == "Read"
        assert tools[0]["args"]["file_path"] == "/workspace/foo.py"

    def test_no_tools_returns_empty_list(self):
        tools = extract_tool_calls_from_entry(ASSISTANT_ENTRY_NO_CACHE)
        assert tools == []

    def test_user_entry_returns_empty_list(self):
        tools = extract_tool_calls_from_entry(USER_ENTRY)
        assert tools == []


class TestCreateUniqueHash:
    def test_uses_message_id_and_request_id(self):
        uid = _create_unique_hash(ASSISTANT_ENTRY_FULL)
        assert uid == "msg_01ABC:req_XXYY"

    def test_falls_back_to_uuid(self):
        entry = {"type": "assistant", "uuid": "uuid-fallback"}
        uid = _create_unique_hash(entry)
        assert uid == "uuid-fallback"

    def test_returns_none_when_no_ids(self):
        assert _create_unique_hash({"type": "assistant"}) is None


class TestParseTimestamp:
    def test_iso_z_string(self):
        ts = _parse_timestamp("2026-01-19T10:00:00.000Z")
        assert ts is not None
        assert ts.tzinfo is not None
        assert ts.year == 2026

    def test_unix_float(self):
        now = time.time()
        ts = _parse_timestamp(now)
        assert ts is not None
        assert abs(ts.timestamp() - now) < 1

    def test_none_input(self):
        assert _parse_timestamp(None) is None

    def test_malformed_string(self):
        assert _parse_timestamp("not-a-date") is None


# ---------------------------------------------------------------------------
# Issue 10: Integration test – mocked JSONL on disk, poller logic
# ---------------------------------------------------------------------------

class TestLoadNewUsageEntries:
    def _write_jsonl(self, path: Path, entries: list[dict]) -> None:
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def test_returns_only_assistant_entries_with_usage(self, tmp_path):
        jsonl = tmp_path / "session.jsonl"
        self._write_jsonl(jsonl, [ASSISTANT_ENTRY_FULL, USER_ENTRY, ASSISTANT_ENTRY_NO_CACHE])
        entries, hashes = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 2
        assert all(e["type"] == "assistant" for e in entries)

    def test_deduplication_on_second_poll(self, tmp_path):
        jsonl = tmp_path / "session.jsonl"
        self._write_jsonl(jsonl, [ASSISTANT_ENTRY_FULL])
        entries1, hashes1 = load_new_usage_entries(tmp_path, set())
        assert len(entries1) == 1
        entries2, hashes2 = load_new_usage_entries(tmp_path, hashes1)
        assert len(entries2) == 0

    def test_new_entries_picked_up_after_first_poll(self, tmp_path):
        jsonl = tmp_path / "session.jsonl"
        self._write_jsonl(jsonl, [ASSISTANT_ENTRY_FULL])
        entries1, hashes1 = load_new_usage_entries(tmp_path, set())
        assert len(entries1) == 1
        # Append a second entry
        with open(jsonl, "a") as f:
            f.write(json.dumps(ASSISTANT_ENTRY_NO_CACHE) + "\n")
        entries2, hashes2 = load_new_usage_entries(tmp_path, hashes1)
        assert len(entries2) == 1
        assert entries2[0]["uuid"] == "uuid-bbb"

    def test_tokens_and_tool_calls_injected(self, tmp_path):
        jsonl = tmp_path / "session.jsonl"
        self._write_jsonl(jsonl, [ASSISTANT_ENTRY_FULL])
        entries, _ = load_new_usage_entries(tmp_path, set())
        e = entries[0]
        assert "_tokens" in e
        assert "_tool_calls" in e
        assert "_parsed_timestamp" in e
        assert e["_tokens"]["prompt"] == 10
        assert e["_tool_calls"][0]["name"] == "Read"

    def test_malformed_lines_skipped(self, tmp_path):
        jsonl = tmp_path / "session.jsonl"
        with open(jsonl, "w") as f:
            f.write("this is not json\n")
            f.write(json.dumps(ASSISTANT_ENTRY_FULL) + "\n")
        entries, _ = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 1

    def test_nonexistent_path_returns_empty(self, tmp_path):
        entries, hashes = load_new_usage_entries(tmp_path / "nonexistent", set())
        assert entries == []
        assert hashes == set()

    def test_multi_file_rglob(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        self._write_jsonl(tmp_path / "a.jsonl", [ASSISTANT_ENTRY_FULL])
        self._write_jsonl(subdir / "b.jsonl", [ASSISTANT_ENTRY_NO_CACHE])
        entries, _ = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Issue 10 (continued): Integration test for append_cli_step via poller
# ---------------------------------------------------------------------------

class TestAppendCliStepIntegration:
    """Test that the poller correctly updates cumulative_metrics via LLMLogger."""

    def test_steps_appended_to_cumulative_metrics(self, tmp_path):
        from utils.llm_logger import LLMLogger

        metrics_file = tmp_path / "cumulative_metrics.json"
        with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
            llm_logger = LLMLogger(log_dir=str(tmp_path), session_id="test_cli")

        # Write a JSONL file
        jsonl = tmp_path / "session.jsonl"
        with open(jsonl, "w") as f:
            f.write(json.dumps(ASSISTANT_ENTRY_FULL) + "\n")
            f.write(json.dumps(ASSISTANT_ENTRY_NO_CACHE) + "\n")

        from utils.claude_jsonl_reader import load_new_usage_entries

        new_entries, processed = load_new_usage_entries(tmp_path, set())
        assert len(new_entries) == 2

        new_entries.sort(key=lambda e: (e["_parsed_timestamp"] or datetime.min.replace(tzinfo=None)))
        step = -1
        for entry in new_entries:
            step += 1
            llm_logger.append_cli_step(
                step_number=step,
                token_usage=entry["_tokens"],
                duration=1.5,
                timestamp=time.time(),
                model_info={"model": "claude-sonnet-4-6"},
                tool_calls=entry["_tool_calls"],
            )

        steps = llm_logger.cumulative_metrics["steps"]
        assert len(steps) == 2
        assert steps[0]["step"] == 0
        assert steps[1]["step"] == 1
        # First entry has cache tokens
        assert steps[0]["cached_tokens"] == 1500
        assert steps[0]["cache_write_tokens"] == 200
        # Tool calls attached
        assert "tool_calls" in steps[0]
        assert steps[0]["tool_calls"][0]["name"] == "Read"
        # Cost should be positive
        assert llm_logger.cumulative_metrics["total_cost"] > 0


# ---------------------------------------------------------------------------
# Issue 11: Backward compatibility – cumulative_metrics.json schema
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = {
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "total_cost",
    "total_actions",
    "start_time",
    "total_llm_calls",
    "total_run_time",
    "last_update_time",
    "metadata",
    "steps",
    "milestones",
    "objectives",
}

REQUIRED_STEP_KEYS = {
    "step",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "total_tokens",
    "time_taken",
    "timestamp",
}


class TestCumulativeMetricsSchema:
    """Ensure LLMLogger always produces a cumulative_metrics.json with the required schema."""

    def test_fresh_logger_has_required_keys(self, tmp_path):
        from utils.llm_logger import LLMLogger

        metrics_file = tmp_path / "cumulative_metrics.json"
        with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
            ll = LLMLogger(log_dir=str(tmp_path), session_id="schema_test")

        missing = REQUIRED_TOP_LEVEL_KEYS - set(ll.cumulative_metrics.keys())
        assert not missing, f"Missing top-level keys: {missing}"

    def test_append_cli_step_produces_valid_step_shape(self, tmp_path):
        from utils.llm_logger import LLMLogger

        metrics_file = tmp_path / "cumulative_metrics.json"
        with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
            ll = LLMLogger(log_dir=str(tmp_path), session_id="schema_test2")

        ll.append_cli_step(
            step_number=0,
            token_usage={"prompt": 10, "completion": 5, "cached": 0, "cache_write": 0, "total": 15},
            duration=1.0,
            timestamp=time.time(),
            model_info={"model": "claude-sonnet-4-6"},
        )

        assert len(ll.cumulative_metrics["steps"]) == 1
        step = ll.cumulative_metrics["steps"][0]
        missing = REQUIRED_STEP_KEYS - set(step.keys())
        assert not missing, f"Missing step keys: {missing}"

    def test_persisted_file_has_correct_schema(self, tmp_path):
        from utils.llm_logger import LLMLogger

        metrics_file = tmp_path / "cumulative_metrics.json"
        with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
            ll = LLMLogger(log_dir=str(tmp_path), session_id="schema_test3")
            ll.append_cli_step(
                step_number=0,
                token_usage={"prompt": 1, "completion": 1, "cached": 0, "cache_write": 0, "total": 2},
                duration=0.5,
                timestamp=time.time(),
            )

        assert metrics_file.exists()
        data = json.loads(metrics_file.read_text())
        missing_top = REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
        assert not missing_top, f"Persisted file missing top-level keys: {missing_top}"
        assert len(data["steps"]) == 1
        missing_step = REQUIRED_STEP_KEYS - set(data["steps"][0].keys())
        assert not missing_step, f"Persisted step missing keys: {missing_step}"

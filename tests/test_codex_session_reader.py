#!/usr/bin/env python3
"""Tests for Codex session reader."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metric_tracking.codex_session_reader import (
    find_session_files,
    load_new_usage_entries,
)


class TestFindSessionFiles:
    def test_empty_dir_returns_empty(self, tmp_path):
        assert find_session_files(tmp_path) == []

    def test_sessions_subdir_found(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "a.jsonl").write_text("{}")
        files = find_session_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "a.jsonl"

    def test_fallback_to_data_path_jsonl(self, tmp_path):
        (tmp_path / "x.jsonl").write_text("{}")
        files = find_session_files(tmp_path)
        assert len(files) == 1


class TestLoadNewUsageEntries:
    def test_empty_dir_returns_empty(self, tmp_path):
        entries, hashes = load_new_usage_entries(tmp_path, set())
        assert entries == []
        assert hashes == set()

    def test_event_msg_token_count_extracted(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        line = json.dumps({
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "input_tokens": 100,
                "output_tokens": 50,
                "cached_input_tokens": 10,
            },
            "timestamp": "2025-01-15T12:00:00Z",
        })
        (sessions_dir / "s1.jsonl").write_text(line + "\n")

        entries, hashes = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 1
        assert entries[0]["_tokens"]["prompt"] == 100
        assert entries[0]["_tokens"]["completion"] == 50
        assert entries[0]["_tokens"]["cached"] == 10
        # total = input + output (cached is subset of input; no double-count)
        assert entries[0]["_tokens"]["total"] == 150
        assert len(hashes) == 1

    def test_usage_object_extracted(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        line = json.dumps({
            "type": "turn",
            "usage": {"input_tokens": 200, "output_tokens": 80, "cached_input_tokens": 0},
            "timestamp": "2025-01-15T12:01:00Z",
        })
        (sessions_dir / "s2.jsonl").write_text(line + "\n")

        entries, hashes = load_new_usage_entries(tmp_path, set())
        assert len(entries) == 1
        assert entries[0]["_tokens"]["prompt"] == 200
        assert entries[0]["_tokens"]["completion"] == 80
        assert entries[0]["_tokens"]["total"] == 280

    def test_dedup_by_processed_hashes(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        line = json.dumps({
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"input_tokens": 1, "output_tokens": 1},
                    "total_token_usage": {"total_tokens": 2},
                },
            },
            "timestamp": "2025-01-15T12:00:00Z",
        })
        (sessions_dir / "s1.jsonl").write_text(line + "\n")

        entries1, hashes1 = load_new_usage_entries(tmp_path, set())
        assert len(entries1) == 1

        entries2, hashes2 = load_new_usage_entries(tmp_path, hashes1)
        assert len(entries2) == 0
        assert hashes2 == hashes1

    def test_malformed_line_skipped(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "s1.jsonl").write_text(
            '{"valid": true}\n'
            "not json\n"
            '{"payload":{"type":"token_count","input_tokens":5,"output_tokens":5}}\n'
        )

        entries, _ = load_new_usage_entries(tmp_path, set())
        # First line has no token data; second is malformed; third has token_count
        assert len(entries) == 1
        assert entries[0]["_tokens"]["total"] == 10

    def test_duplicate_token_snapshots_collapsed_and_tool_call_attached(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        lines = [
            {
                "type": "turn_context",
                "payload": {"model": "openai/gpt-5.3-codex"},
                "timestamp": "2025-01-15T12:00:00Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "mcp__pokemon-emerald__get_game_state",
                    "arguments": "{}",
                },
                "timestamp": "2025-01-15T12:00:01Z",
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 10,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 5,
                        },
                        "total_token_usage": {"total_tokens": 135},
                    },
                },
                "timestamp": "2025-01-15T12:00:02Z",
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 10,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 5,
                        },
                        "total_token_usage": {"total_tokens": 135},
                    },
                },
                "timestamp": "2025-01-15T12:00:03Z",
            },
        ]
        (sessions_dir / "s1.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")

        entries, hashes = load_new_usage_entries(tmp_path, set())

        assert len(entries) == 1
        assert entries[0]["_tokens"]["prompt"] == 100
        assert entries[0]["_tokens"]["completion"] == 25
        assert entries[0]["_tool_calls"] == [{"name": "mcp__pokemon-emerald__get_game_state", "args": {}}]
        assert entries[0]["_model"] == "openai/gpt-5.3-codex"
        assert len(hashes) == 1

    def test_multiple_rollouts_do_not_dedup_each_other(self, tmp_path):
        sessions_dir = tmp_path / "sessions" / "2025" / "01" / "15"
        sessions_dir.mkdir(parents=True)
        line = {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"input_tokens": 10, "output_tokens": 2},
                    "total_token_usage": {"total_tokens": 12},
                },
            },
            "timestamp": "2025-01-15T12:00:00Z",
        }
        (sessions_dir / "rollout-a.jsonl").write_text(json.dumps(line) + "\n")
        (sessions_dir / "rollout-b.jsonl").write_text(json.dumps(line) + "\n")

        entries, hashes = load_new_usage_entries(tmp_path, set())

        assert len(entries) == 2
        assert len(hashes) == 2

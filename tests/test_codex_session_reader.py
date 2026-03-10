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
        assert entries[0]["_tokens"]["total"] == 160
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
            "payload": {"type": "token_count", "input_tokens": 1, "output_tokens": 1},
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

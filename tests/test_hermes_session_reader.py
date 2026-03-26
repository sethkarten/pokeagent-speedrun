#!/usr/bin/env python3
"""Tests for Hermes session reader utilities."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metric_tracking.hermes_session_reader import (
    _normalize_tool_calls,
    get_latest_session_id,
    load_new_usage_entries,
)


def _write_session_log(tmp_path: Path, payload: dict) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    session_id = payload["session_id"]
    (sessions_dir / f"session_{session_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_usage_events(tmp_path: Path, *events: dict) -> None:
    lines = [json.dumps(event) for event in events]
    (tmp_path / "usage_events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestHermesSessionReader:
    def test_get_latest_session_id_reads_session_logs(self, tmp_path):
        _write_session_log(
            tmp_path,
            {
                "session_id": "hermes-session-2",
                "messages": [],
            },
        )
        assert get_latest_session_id(tmp_path) == "hermes-session-2"

    def test_load_new_usage_entries_reads_session_json_and_usage_events(self, tmp_path):
        _write_session_log(
            tmp_path,
            {
                "session_id": "hermes-session-1",
                "model": "google/gemini-3-flash-preview",
                "session_start": "2026-03-15T20:02:47.565245+00:00",
                "messages": [
                    {"role": "user", "content": "start"},
                    {
                        "role": "assistant",
                        "finish_reason": "tool_calls",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "mcp_pokemon_emerald_get_game_state",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "finish_reason": "stop",
                        "content": "done",
                    },
                ],
            },
        )
        _write_usage_events(
            tmp_path,
            {
                "timestamp": "2026-03-15T20:02:48.000000+00:00",
                "session_id": "hermes-session-1",
                "api_call_index": 1,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cached_tokens": 25,
                "cache_write_tokens": 10,
            },
            {
                "timestamp": "2026-03-15T20:02:49.000000+00:00",
                "session_id": "hermes-session-1",
                "api_call_index": 2,
                "prompt_tokens": 60,
                "completion_tokens": 15,
                "total_tokens": 75,
            },
        )

        entries, hashes, state = load_new_usage_entries(tmp_path, set(), {})

        assert len(entries) == 2
        assert len(hashes) == 2
        assert entries[0]["_model"] == "google/gemini-3-flash-preview"
        assert entries[0]["_tokens"]["total"] == 150
        assert entries[0]["_tokens"]["cached"] == 25
        assert entries[0]["_tokens"]["cache_write"] == 10
        assert entries[0]["_tool_calls"] == [{"name": "get_game_state", "args": {}}]
        assert entries[1]["_tokens"]["total"] == 75
        assert entries[1]["_tool_calls"] == []
        assert state["hermes-session-1"]["assistant_index"] == 2

    def test_second_poll_with_same_state_returns_no_entries(self, tmp_path):
        _write_session_log(
            tmp_path,
            {
                "session_id": "hermes-session-1",
                "messages": [
                    {"role": "assistant", "tool_calls": []},
                ],
            },
        )
        _write_usage_events(
            tmp_path,
            {
                "timestamp": "2026-03-15T20:02:48.000000+00:00",
                "session_id": "hermes-session-1",
                "api_call_index": 1,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )

        entries, hashes, state = load_new_usage_entries(tmp_path, set(), {})
        assert entries

        second_entries, second_hashes, second_state = load_new_usage_entries(tmp_path, hashes, state)
        assert second_entries == []
        assert second_hashes == hashes
        assert second_state == state

    def test_normalize_tool_calls_reads_nested_function_shape(self):
        raw = [
            {
                "id": "tool_1",
                "type": "function",
                "function": {
                    "name": "mcp_pokemon_emerald_get_game_state",
                    "arguments": "{}",
                },
            }
        ]

        assert _normalize_tool_calls(raw) == [
            {"name": "get_game_state", "args": {}},
        ]

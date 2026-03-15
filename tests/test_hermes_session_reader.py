#!/usr/bin/env python3
"""Tests for Hermes session reader utilities."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metric_tracking.hermes_session_reader import (
    STATE_DB_NAME,
    get_latest_session_id,
    load_new_usage_entries,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DB = FIXTURES_DIR / "hermes_state.db"


def _copy_fixture_db(tmp_path: Path) -> Path:
    target = tmp_path / STATE_DB_NAME
    shutil.copy2(FIXTURE_DB, target)
    return target


class TestHermesSessionReader:
    def test_get_latest_session_id_reads_state_db(self, tmp_path):
        _copy_fixture_db(tmp_path)
        assert get_latest_session_id(tmp_path) == "hermes-session-2"

    def test_load_new_usage_entries_returns_delta_entries(self, tmp_path):
        _copy_fixture_db(tmp_path)

        entries, hashes, state = load_new_usage_entries(tmp_path, set(), {})

        assert len(entries) == 2
        assert len(hashes) == 2
        assert set(state) == {"hermes-session-1", "hermes-session-2"}
        assert entries[0]["_tokens"]["total"] == 150
        assert entries[1]["_tokens"]["total"] == 300
        assert entries[0]["_tool_calls"][0]["name"] == "mcp_pokemon-emerald_get_game_state"

    def test_second_poll_with_same_state_returns_no_entries(self, tmp_path):
        _copy_fixture_db(tmp_path)

        entries, hashes, state = load_new_usage_entries(tmp_path, set(), {})
        assert entries

        second_entries, second_hashes, second_state = load_new_usage_entries(tmp_path, hashes, state)
        assert second_entries == []
        assert second_hashes == hashes
        assert second_state == state

"""Tests for trajectory window reading, writing, and formatting."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from agents.subagents.utils.trajectory_window import (
    MAX_TRAJECTORY_WINDOW,
    _trajectory_file_for_run,
    format_trajectory_window,
    load_recent_trajectories,
    read_last_jsonl_lines,
    resolve_trajectory_path,
)


class _RunManagerStub:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir


def _write_trajectories_cache(cache_dir: Path, count: int) -> Path:
    """Write trajectory entries to cache-style path."""
    trajectory_file = cache_dir / "trajectory_history.jsonl"
    trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_file.open("w", encoding="utf-8") as handle:
        for step in range(1, count + 1):
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "reasoning": f"reasoning-{step}",
                        "action": {"tool": "press_buttons", "buttons": ["A"]},
                        "pre_state": {"location": "Route 101", "player_coords": [step, step]},
                        "outcome": {"success": True},
                        "location": "Route 101",
                        "player_coords": [step, step],
                    }
                )
                + "\n"
            )
    return trajectory_file


# ---- Read from cache path ----

_CACHE_MOCK_TARGET = "utils.data_persistence.run_data_manager.get_cache_path"


def test_load_recent_trajectories_from_cache(tmp_path):
    cache_dir = tmp_path / ".pokeagent_cache" / "test_run"
    _write_trajectories_cache(cache_dir, 8)

    with mock.patch(_CACHE_MOCK_TARGET, side_effect=lambda rel: cache_dir / rel):
        loaded = load_recent_trajectories(None, last_n_steps=3)
    assert [entry["step"] for entry in loaded] == [6, 7, 8]


def test_load_recent_trajectories_caps_window_at_max(tmp_path):
    cache_dir = tmp_path / ".pokeagent_cache" / "test_run"
    n_entries = MAX_TRAJECTORY_WINDOW + 30
    _write_trajectories_cache(cache_dir, n_entries)

    with mock.patch(_CACHE_MOCK_TARGET, side_effect=lambda rel: cache_dir / rel):
        loaded = load_recent_trajectories(None, last_n_steps=999)
    assert len(loaded) == MAX_TRAJECTORY_WINDOW
    assert loaded[0]["step"] == n_entries - MAX_TRAJECTORY_WINDOW + 1
    assert loaded[-1]["step"] == n_entries


def test_trajectory_file_resolves_run_data_copy_when_cache_missing(tmp_path):
    run_dir = tmp_path / "run_data" / "test_run"
    synced = run_dir / "trajectory_history.jsonl"
    synced.parent.mkdir(parents=True, exist_ok=True)
    synced.write_text('{"step": 1}\n', encoding="utf-8")

    with mock.patch(_CACHE_MOCK_TARGET, side_effect=lambda rel: tmp_path / "nonexistent" / rel):
        result = resolve_trajectory_path(_RunManagerStub(run_dir))
    assert result == synced


# ---- Missing file handling ----

def test_load_recent_trajectories_handles_missing_or_empty_files(tmp_path):
    stub = _RunManagerStub(tmp_path)

    with mock.patch(_CACHE_MOCK_TARGET, side_effect=lambda rel: tmp_path / "nonexistent" / rel):
        assert load_recent_trajectories(stub, last_n_steps=5) == []


# ---- Format compat: new schema (no post_state) ----

def test_format_trajectory_window_without_post_state():
    entries = [
        {
            "step": 1,
            "action": {"tool": "press_buttons", "buttons": ["A"]},
            "pre_state": {"location": "Route 101", "player_coords": [5, 10]},
            "outcome": {"success": True},
            "location": "Route 101",
            "player_coords": [5, 10],
            "reasoning": "Testing",
        }
    ]
    text = format_trajectory_window(entries)
    assert "Route 101" in text
    assert "(5, 10)" in text
    assert "->" not in text


def test_format_trajectory_window_with_legacy_post_state():
    entries = [
        {
            "step": 1,
            "action": {"tool": "press_buttons", "buttons": ["A"]},
            "pre_state": {"location": "Route 101", "player_coords": [5, 10]},
            "post_state": {"location": "Petalburg City", "player_coords": [12, 20]},
            "outcome": {"success": True},
            "reasoning": "Walking",
        }
    ]
    text = format_trajectory_window(entries)
    assert "->" in text
    assert "Petalburg City" in text


def test_format_trajectory_window_with_objective_context():
    entries = [
        {
            "step": 1,
            "action": {"tool": "press_buttons"},
            "pre_state": {},
            "outcome": {"success": True},
            "location": "Littleroot",
            "player_coords": [3, 4],
            "objective_context": "story_01",
            "reasoning": "",
        }
    ]
    text = format_trajectory_window(entries)
    assert "Obj: story_01" in text


# ---- Sync test ----

def test_sync_trajectories_to_run_data(tmp_path):
    """sync_trajectories_to_run_data copies JSONL from cache to run_data."""
    from utils.data_persistence.run_data_manager import RunDataManager

    cache_dir = tmp_path / "cache"
    cache_file = cache_dir / "trajectory_history.jsonl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text('{"step":1}\n', encoding="utf-8")

    run_dir = tmp_path / "run_data" / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    mgr = RunDataManager.__new__(RunDataManager)
    mgr.run_dir = run_dir
    mgr.run_id = "test_run"
    mgr.base_dir = tmp_path / "run_data"
    mgr.trajectory_step = 0

    with mock.patch("utils.data_persistence.run_data_manager.get_cache_path",
                     side_effect=lambda rel: cache_dir / rel):
        mgr.sync_trajectories_to_run_data()

    dest = run_dir / "trajectory_history.jsonl"
    assert dest.exists()
    assert dest.read_text(encoding="utf-8").strip() == '{"step":1}'


def test_read_last_jsonl_lines_matches_full_scan_suffix(tmp_path):
    trajectory_file = tmp_path / "trajectory_history.jsonl"
    total = 250
    with trajectory_file.open("w", encoding="utf-8") as handle:
        for idx in range(total):
            handle.write(f'{{"step": {idx + 1}}}\n')

    expected = trajectory_file.read_text(encoding="utf-8").splitlines()[-37:]
    actual = read_last_jsonl_lines(trajectory_file, 37)
    assert actual == expected

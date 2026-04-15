#!/usr/bin/env python3
"""Tests for LLMLogger action-count tracking, step attribution, and milestone/objective split calculations."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys_path = Path(__file__).parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from utils.data_persistence.llm_logger import LLMLogger


def _make_logger(tmpdir: str) -> LLMLogger:
    """Create an LLMLogger wired to a temp directory."""
    log_dir = Path(tmpdir) / "logs"
    log_dir.mkdir(exist_ok=True)
    metrics_file = Path(tmpdir) / "cumulative_metrics.json"
    with patch("utils.data_persistence.run_data_manager.get_cache_path", return_value=metrics_file):
        return LLMLogger(log_dir=str(log_dir), session_id="test")


def _add_step_entry(logger: LLMLogger, step: int):
    """Simulate a log_interaction that creates a step entry (minimal token usage)."""
    logger.log_interaction(
        "test",
        "prompt",
        "response",
        metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        step_number=step,
        model_info={"model": "test-model"},
        duration=0.1,
    )


class TestIncrementActionCount:

    def test_increment_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            assert logger.cumulative_metrics["total_actions"] == 0
            logger.increment_action_count()
            assert logger.cumulative_metrics["total_actions"] == 1

    def test_increment_by_n(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.increment_action_count(5)
            logger.increment_action_count(3)
            assert logger.cumulative_metrics["total_actions"] == 8


class TestAddStepToolCalls:

    def test_attaches_to_existing_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            _add_step_entry(logger, step=1)

            logger.add_step_tool_calls(1, [
                {"name": "press_buttons", "args": {"buttons": ["A", "B"]}},
                {"name": "get_game_state", "args": {}},
            ])

            entry = next(e for e in logger.cumulative_metrics["steps"] if e["step"] == 1)
            assert len(entry["tool_calls"]) == 2
            assert entry["tool_calls"][0]["name"] == "press_buttons"

    def test_creates_entry_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.add_step_tool_calls(42, [{"name": "navigate_to", "args": {"x": 5, "y": 10}}])

            entry = next(e for e in logger.cumulative_metrics["steps"] if e["step"] == 42)
            assert entry["tool_calls"][0]["name"] == "navigate_to"


class TestLogMilestoneWithActions:

    def test_first_milestone_cumulative_and_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.cumulative_metrics["total_actions"] = 50

            logger.log_milestone_completion("OLDALE_TOWN", step_number=10, timestamp=1000.0)

            ms = logger.cumulative_metrics["milestones"]
            assert len(ms) == 1
            assert ms[0]["cumulative_actions"] == 50
            assert ms[0]["split_actions"] == 50

    def test_second_milestone_split_is_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.cumulative_metrics["total_actions"] = 50
            logger.log_milestone_completion("OLDALE_TOWN", step_number=10, timestamp=1000.0)

            logger.cumulative_metrics["total_actions"] = 120
            logger.log_milestone_completion("PETALBURG_CITY", step_number=25, timestamp=2000.0)

            ms = logger.cumulative_metrics["milestones"]
            assert len(ms) == 2
            assert ms[1]["cumulative_actions"] == 120
            assert ms[1]["split_actions"] == 70  # 120 - 50

    def test_three_milestones_split_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            actions = [30, 80, 200]
            names = ["M1", "M2", "M3"]
            for i, (a, name) in enumerate(zip(actions, names)):
                logger.cumulative_metrics["total_actions"] = a
                logger.log_milestone_completion(name, step_number=i + 1, timestamp=1000.0 + i)

            ms = logger.cumulative_metrics["milestones"]
            assert ms[0]["split_actions"] == 30
            assert ms[1]["split_actions"] == 50   # 80 - 30
            assert ms[2]["split_actions"] == 120  # 200 - 80

    def test_milestone_preserves_token_fields(self):
        """Action tracking shouldn't break existing token fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.cumulative_metrics["total_tokens"] = 1000
            logger.cumulative_metrics["prompt_tokens"] = 800
            logger.cumulative_metrics["completion_tokens"] = 200
            logger.cumulative_metrics["total_actions"] = 10

            logger.log_milestone_completion("TEST", step_number=5, timestamp=1000.0)

            ms = logger.cumulative_metrics["milestones"][0]
            assert ms["cumulative_total_tokens"] == 1000
            assert ms["cumulative_prompt_tokens"] == 800
            assert ms["cumulative_actions"] == 10


class TestLogObjectiveWithActions:

    def test_first_objective_cumulative_and_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.cumulative_metrics["total_actions"] = 25

            logger.log_objective_completion(
                "story_01", "story", 0, step_number=3, timestamp=1000.0
            )

            obj = logger.cumulative_metrics["objectives"]
            assert len(obj) == 1
            assert obj[0]["cumulative_actions"] == 25
            assert obj[0]["split_actions"] == 25

    def test_second_objective_split_is_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.cumulative_metrics["total_actions"] = 25
            logger.log_objective_completion(
                "story_01", "story", 0, step_number=3, timestamp=1000.0
            )

            logger.cumulative_metrics["total_actions"] = 60
            logger.log_objective_completion(
                "story_02", "story", 1, step_number=8, timestamp=2000.0
            )

            obj = logger.cumulative_metrics["objectives"]
            assert len(obj) == 2
            assert obj[1]["cumulative_actions"] == 60
            assert obj[1]["split_actions"] == 35  # 60 - 25

    def test_objective_preserves_category_and_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.log_objective_completion(
                "battling_03", "battling", 2, step_number=10, timestamp=1000.0
            )
            obj = logger.cumulative_metrics["objectives"][0]
            assert obj["objective_id"] == "battling_03"
            assert obj["category"] == "battling"
            assert obj["objective_index"] == 2


class TestMilestoneSplitDeltas:
    """Verify that split metrics for milestones are correct deltas between consecutive milestones."""

    def test_split_steps_are_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.log_milestone_completion("M1", step_number=10, timestamp=1000.0)
            logger.log_milestone_completion("M2", step_number=30, timestamp=2000.0)
            logger.log_milestone_completion("M3", step_number=55, timestamp=3000.0)

            ms = logger.cumulative_metrics["milestones"]
            assert ms[0]["split_steps"] == 10
            assert ms[1]["split_steps"] == 20  # 30 - 10
            assert ms[2]["split_steps"] == 25  # 55 - 30

    def test_split_time_is_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.log_milestone_completion("M1", step_number=1, timestamp=1000.0)
            logger.log_milestone_completion("M2", step_number=2, timestamp=1500.5)

            ms = logger.cumulative_metrics["milestones"]
            assert ms[1]["split_time_seconds"] == 500.5

    def test_split_tokens_are_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)

            logger.cumulative_metrics["total_tokens"] = 100
            logger.cumulative_metrics["prompt_tokens"] = 80
            logger.log_milestone_completion("M1", step_number=1, timestamp=1000.0)

            logger.cumulative_metrics["total_tokens"] = 350
            logger.cumulative_metrics["prompt_tokens"] = 300
            logger.log_milestone_completion("M2", step_number=2, timestamp=2000.0)

            ms = logger.cumulative_metrics["milestones"]
            assert ms[1]["split_total_tokens"] == 250   # 350 - 100
            assert ms[1]["split_prompt_tokens"] == 220   # 300 - 80


class TestLoadRestoresActionTracking:
    """Verify that load_cumulative_metrics restores _last_*_actions from saved entries."""

    def test_restore_milestone_actions_after_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.cumulative_metrics["total_actions"] = 50
            logger.log_milestone_completion("M1", step_number=10, timestamp=1000.0)

            metrics_file = Path(tmpdir) / "cumulative_metrics.json"
            logger.save_cumulative_metrics(str(metrics_file))

            logger2 = _make_logger(tmpdir)
            logger2.load_cumulative_metrics(str(metrics_file))

            assert logger2.cumulative_metrics["_last_milestone_actions"] == 50

            logger2.cumulative_metrics["total_actions"] = 120
            logger2.log_milestone_completion("M2", step_number=25, timestamp=2000.0)
            ms = logger2.cumulative_metrics["milestones"]
            assert ms[1]["split_actions"] == 70  # 120 - 50

    def test_restore_objective_actions_after_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _make_logger(tmpdir)
            logger.cumulative_metrics["total_actions"] = 30
            logger.log_objective_completion("story_01", "story", 0, step_number=5, timestamp=1000.0)

            metrics_file = Path(tmpdir) / "cumulative_metrics.json"
            logger.save_cumulative_metrics(str(metrics_file))

            logger2 = _make_logger(tmpdir)
            logger2.load_cumulative_metrics(str(metrics_file))

            assert logger2.cumulative_metrics["_last_objective_actions"] == 30

            logger2.cumulative_metrics["total_actions"] = 80
            logger2.log_objective_completion("story_02", "story", 1, step_number=12, timestamp=2000.0)
            obj = logger2.cumulative_metrics["objectives"]
            assert obj[1]["split_actions"] == 50  # 80 - 30

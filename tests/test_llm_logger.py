#!/usr/bin/env python3
"""Tests for LLMLogger cumulative metrics and checkpoint persistence."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root for imports
sys_path = Path(__file__).parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from utils.llm_logger import LLMLogger


class TestCumulativeMetricsPersistence:
    """Test save/load of cumulative_metrics.json."""

    def test_load_cumulative_metrics_missing_file_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "cumulative_metrics.json"
            assert not metrics_file.exists()
            with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
                logger = LLMLogger(log_dir=tmpdir, session_id="test_session")
            result = logger.load_cumulative_metrics(str(metrics_file))
            assert result is False
            assert logger.cumulative_metrics["total_tokens"] == 0

    def test_save_and_load_cumulative_metrics_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_file = Path(tmpdir) / "cumulative_metrics.json"
            with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
                logger = LLMLogger(log_dir=tmpdir, session_id="test_session")
            logger.cumulative_metrics["total_tokens"] = 12345
            logger.cumulative_metrics["total_cost"] = 0.5
            metrics_file = Path(tmpdir) / "cumulative_metrics.json"
            logger.save_cumulative_metrics(str(metrics_file))
            assert metrics_file.exists()
            loaded = json.loads(metrics_file.read_text())
            assert loaded["total_tokens"] == 12345
            assert loaded["total_cost"] == 0.5
            # Load into fresh logger
            logger2 = LLMLogger(log_dir=tmpdir, session_id="test_session2")
            result = logger2.load_cumulative_metrics(str(metrics_file))
            assert result is True
            assert logger2.cumulative_metrics["total_tokens"] == 12345
            assert logger2.cumulative_metrics["total_cost"] == 0.5


class TestCheckpointNoCumulativeMetrics:
    """Test that checkpoint_llm.txt no longer embeds cumulative_metrics."""

    def test_save_checkpoint_does_not_include_cumulative_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()
            (log_dir / "llm_log_test.jsonl").write_text("")
            metrics_file = Path(tmpdir) / "cumulative_metrics.json"
            with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
                logger = LLMLogger(log_dir=str(log_dir), session_id="test")
            logger.cumulative_metrics["total_tokens"] = 999
            checkpoint_file = Path(tmpdir) / "checkpoint_llm.txt"
            logger.save_checkpoint(str(checkpoint_file), agent_step_count=42)
            assert checkpoint_file.exists()
            data = json.loads(checkpoint_file.read_text())
            assert "cumulative_metrics" not in data
            assert data["agent_step_count"] == 42
            assert "log_entries" in data

    def test_load_checkpoint_does_not_overlay_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()
            log_file = log_dir / "llm_log_test.jsonl"
            log_file.write_text("")
            metrics_file = Path(tmpdir) / "cumulative_metrics.json"
            with patch("utils.run_data_manager.get_cache_path", return_value=metrics_file):
                logger = LLMLogger(log_dir=str(log_dir), session_id="test")
            assert logger.cumulative_metrics["total_tokens"] == 0
            # Create checkpoint with old format (had cumulative_metrics) - we ignore it
            checkpoint_file = Path(tmpdir) / "checkpoint_llm.txt"
            checkpoint_data = {
                "agent_step_count": 100,
                "log_entries": [{"type": "step_start", "step_number": 1}],
                "cumulative_metrics": {"total_tokens": 50000},  # Should be ignored
            }
            checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
            result = logger.load_checkpoint(str(checkpoint_file))
            assert result == 100
            # Metrics should NOT be updated from checkpoint (we use cumulative_metrics.json only)
            assert logger.cumulative_metrics["total_tokens"] == 0


class TestStepsNoTruncation:
    """Test that steps array is no longer truncated at 1000."""

    def test_more_than_1000_steps_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()
            metrics_file = Path(tmpdir) / "cumulative_metrics.json"

            def _get_cache_path(relative_path):
                if "cumulative_metrics" in relative_path:
                    return metrics_file
                return Path(tmpdir) / relative_path

            with patch("utils.run_data_manager.get_cache_path", side_effect=_get_cache_path):
                logger = LLMLogger(log_dir=str(log_dir), session_id="test")
                # Simulate 1001 log_interaction calls with step_number
                for i in range(1001):
                    logger.log_interaction(
                        "test",
                        "prompt",
                        "response",
                        metadata={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
                        step_number=i + 1,
                        model_info={},
                        duration=0.01,
                    )
            assert len(logger.cumulative_metrics["steps"]) == 1001
            assert logger.cumulative_metrics["steps"][0]["step"] == 1
            assert logger.cumulative_metrics["steps"][-1]["step"] == 1001


class TestBackupRestoreWithoutMetricsFile:
    """Test restore from backup with checkpoint but no cumulative_metrics.json."""

    def test_no_metrics_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()
            (log_dir / "llm_log_test.jsonl").write_text("")
            metrics_path = Path(tmpdir) / "cumulative_metrics.json"
            assert not metrics_path.exists()
            with patch("utils.run_data_manager.get_cache_path", return_value=metrics_path):
                logger = LLMLogger(log_dir=str(log_dir), session_id="test")
                result = logger.load_cumulative_metrics(str(metrics_path))
            assert result is False
            assert logger.cumulative_metrics["total_tokens"] == 0
            assert logger.cumulative_metrics["total_llm_calls"] == 0

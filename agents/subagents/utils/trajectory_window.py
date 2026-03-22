"""Trajectory-window helpers for local subagents."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TRAJECTORY_WINDOW = 50
DEFAULT_TRAJECTORY_WINDOW = 10


def clamp_trajectory_window(last_n_steps: Optional[int]) -> int:
    """Clamp a requested trajectory window to the supported range."""
    if last_n_steps is None:
        return DEFAULT_TRAJECTORY_WINDOW
    try:
        requested = int(last_n_steps)
    except (TypeError, ValueError):
        return DEFAULT_TRAJECTORY_WINDOW
    return max(1, min(requested, MAX_TRAJECTORY_WINDOW))


def _trajectory_file_for_run(run_data_manager: Any) -> Optional[Path]:
    """Resolve the trajectories.jsonl path from the run manager."""
    if run_data_manager is None:
        return None

    run_dir = None
    if hasattr(run_data_manager, "get_run_directory"):
        run_dir = run_data_manager.get_run_directory()
    elif hasattr(run_data_manager, "run_dir"):
        run_dir = run_data_manager.run_dir

    if not run_dir:
        return None

    return Path(run_dir) / "prompt_evolution" / "trajectories" / "trajectories.jsonl"


def _read_last_jsonl_lines(path: Path, max_lines: int) -> List[str]:
    """Read the last non-empty lines from a JSONL file without scanning it twice."""
    if max_lines <= 0 or not path.exists():
        return []

    lines: List[str] = []
    block_size = 4096

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        if file_size == 0:
            return []

        buffer = b""
        position = file_size

        while position > 0 and len(lines) < max_lines:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            buffer = chunk + buffer

            parts = buffer.split(b"\n")
            if position > 0:
                buffer = parts[0]
                parts = parts[1:]
            else:
                buffer = b""

            for raw_line in reversed(parts):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                lines.append(line)
                if len(lines) >= max_lines:
                    break

        if position == 0 and buffer.strip() and len(lines) < max_lines:
            lines.append(buffer.decode("utf-8", errors="replace").strip())

    lines.reverse()
    return lines


def load_recent_trajectories(run_data_manager: Any, last_n_steps: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load the most recent trajectory entries, preserving chronological order."""
    window_size = clamp_trajectory_window(last_n_steps)
    trajectory_file = _trajectory_file_for_run(run_data_manager)
    if not trajectory_file or not trajectory_file.exists():
        return []

    trajectories: List[Dict[str, Any]] = []
    for line in _read_last_jsonl_lines(trajectory_file, window_size):
        try:
            trajectories.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed trajectory line from %s", trajectory_file)

    return trajectories


def _format_coords(snapshot: Dict[str, Any]) -> str:
    coords = snapshot.get("player_coords")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return f"({coords[0]}, {coords[1]})"
    return "(?)"


def _summarize_action(action: Any) -> str:
    if isinstance(action, dict):
        name = action.get("tool") or action.get("name") or action.get("action")
        if name:
            return json.dumps(action, ensure_ascii=True, sort_keys=True)
    return str(action) if action else "unknown action"


def _summarize_outcome(outcome: Any) -> str:
    if isinstance(outcome, dict):
        if "success" in outcome:
            return "success" if outcome.get("success") else "failure"
        if "status" in outcome:
            return str(outcome.get("status"))
    return str(outcome)[:120] if outcome else "unknown"


def format_trajectory_window(trajectories: List[Dict[str, Any]]) -> str:
    """Format trajectory entries into a compact, readable window."""
    if not trajectories:
        return "No prior trajectories recorded."

    lines: List[str] = []
    for entry in trajectories:
        step = entry.get("step", "?")
        action = _summarize_action(entry.get("action"))
        pre_state = entry.get("pre_state") or {}
        post_state = entry.get("post_state") or {}
        reasoning = (entry.get("reasoning") or "").strip()
        outcome = _summarize_outcome(entry.get("outcome"))

        lines.append(
            f"Step {step}: {action} | "
            f"{pre_state.get('location', 'Unknown')} {_format_coords(pre_state)} -> "
            f"{post_state.get('location', 'Unknown')} {_format_coords(post_state)} | "
            f"Outcome: {outcome}"
        )
        if reasoning:
            lines.append(f"  Reasoning: {reasoning[:280]}")

    return "\n".join(lines)

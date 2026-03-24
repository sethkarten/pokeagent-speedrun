"""Trajectory-window helpers for local subagents."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TRAJECTORY_WINDOW = 100
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


def _trajectory_file_for_run(run_data_manager: Any = None) -> Optional[Path]:
    """Resolve trajectory_history.jsonl, preferring the cache location."""
    try:
        from utils.data_persistence.run_data_manager import get_cache_path
        cache_file = get_cache_path("trajectory_history.jsonl")
        if cache_file.exists():
            return cache_file
    except Exception:
        pass

    # Fallback to legacy run_data path for older runs
    if run_data_manager is not None:
        run_dir = None
        if hasattr(run_data_manager, "get_run_directory"):
            run_dir = run_data_manager.get_run_directory()
        elif hasattr(run_data_manager, "run_dir"):
            run_dir = run_data_manager.run_dir
        if run_dir:
            legacy = Path(run_dir) / "prompt_evolution" / "trajectories" / "trajectories.jsonl"
            if legacy.exists():
                return legacy

    return None


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


def load_trajectory_range(
    run_data_manager: Any,
    start: int,
    end: int,
) -> tuple[List[Dict[str, Any]], int, int]:
    """Load trajectory entries whose ``step`` falls in [start, end].

    Returns ``(entries, actual_min_step, actual_max_step)`` where the
    actual values reflect the available data (the range is clipped).
    """
    trajectory_file = _trajectory_file_for_run(run_data_manager)
    if not trajectory_file or not trajectory_file.exists():
        return [], 0, 0

    all_lines = _read_last_jsonl_lines(trajectory_file, MAX_TRAJECTORY_WINDOW * 10)

    entries: List[Dict[str, Any]] = []
    min_step = float("inf")
    max_step = float("-inf")

    for line in all_lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = entry.get("step")
        if step is None:
            continue
        step_int = int(step)
        if step_int < min_step:
            min_step = step_int
        if step_int > max_step:
            max_step = step_int
        if start <= step_int <= end:
            entries.append(entry)

    actual_min = int(min_step) if min_step != float("inf") else 0
    actual_max = int(max_step) if max_step != float("-inf") else 0
    return entries, actual_min, actual_max


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
        reasoning = (entry.get("reasoning") or "").strip()
        outcome = _summarize_outcome(entry.get("outcome"))

        # New schema: location/player_coords are top-level fields
        location = entry.get("location") or pre_state.get("location", "Unknown")
        coords_raw = entry.get("player_coords") or pre_state.get("player_coords")
        if isinstance(coords_raw, (list, tuple)) and len(coords_raw) >= 2:
            coords_str = f"({coords_raw[0]}, {coords_raw[1]})"
        else:
            coords_str = _format_coords(pre_state)

        # Legacy compat: if post_state exists, show transition; otherwise just show location
        post_state = entry.get("post_state")
        if post_state:
            loc_info = (
                f"{location} {coords_str} -> "
                f"{post_state.get('location', 'Unknown')} {_format_coords(post_state)}"
            )
        else:
            loc_info = f"{location} {coords_str}"

        obj_context = entry.get("objective_context")
        obj_str = f" | Obj: {obj_context}" if obj_context else ""

        lines.append(f"Step {step}: {action} | {loc_info} | Outcome: {outcome}{obj_str}")
        if reasoning:
            lines.append(f"  Reasoning: {reasoning[:280]}")

    return "\n".join(lines)

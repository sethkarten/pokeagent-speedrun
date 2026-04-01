"""Context loading helpers for local PokeAgent subagents."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, Optional

import PIL.Image as PILImage

from .trajectory_window import format_trajectory_window, load_recent_trajectories


def decode_screenshot_base64(image_b64: Optional[str]):
    if not image_b64:
        return None
    try:
        return PILImage.open(io.BytesIO(base64.b64decode(image_b64)))
    except Exception:
        return None


def _extract_current_state(game_state_result: Dict[str, Any]) -> Dict[str, Any]:
    player_position = game_state_result.get("player_position") or {}
    raw_state = game_state_result.get("raw_state") or {}
    return {
        "location": raw_state.get("player", {}).get("location", "Unknown"),
        "coordinates": {"x": player_position.get("x"), "y": player_position.get("y")},
        "state_text": game_state_result.get("state_text", ""),
<<<<<<< HEAD
=======
        "inventory": raw_state.get("player", {}).get("inventory", {}),
>>>>>>> tu8435/tersoo-dev-2
    }


def _extract_objective_state(game_state_result: Dict[str, Any]) -> Dict[str, Any]:
    mode = game_state_result.get("objectives_mode", "legacy")

    if mode == "categorized":
        categorized = game_state_result.get("categorized_objectives") or {}
        status = game_state_result.get("categorized_status") or {}
        categories = {
            "story": {
                "current_objective": categorized.get("story"),
                "index": status.get("story", {}).get("current_index", 0),
                "total": status.get("story", {}).get("total", 0),
                "completed": status.get("story", {}).get("completed", 0),
            },
            "battling": {
                "current_objective": categorized.get("battling_group") or [],
                "recommended_objectives": categorized.get("recommended_battling_objectives") or [],
                "index": status.get("battling", {}).get("current_index", 0),
                "total": status.get("battling", {}).get("total", 0),
                "completed": status.get("battling", {}).get("completed", 0),
            },
            "dynamics": {
                "current_objective": categorized.get("dynamics"),
                "index": status.get("dynamics", {}).get("current_index", 0),
                "total": status.get("dynamics", {}).get("total", 0),
                "completed": status.get("dynamics", {}).get("completed", 0),
            },
        }
        is_complete = all(cat["index"] >= cat["total"] for cat in categories.values())
        return {
            "mode": "categorized",
            "sequence": "categorized_objectives",
            "categories": categories,
            "is_complete": is_complete,
            "status": "all_complete" if is_complete else "active",
        }

    direct_objective = game_state_result.get("direct_objective")
    direct_status = game_state_result.get("direct_objective_status") or {}
    return {
        "mode": "legacy",
        "sequence": direct_status.get("sequence_name"),
        "objective": direct_objective,
        "status": "complete" if direct_status.get("is_complete") else "active",
        "is_complete": bool(direct_status.get("is_complete")),
        "completed_count": direct_status.get("completed_count", 0),
        "total_objectives": direct_status.get("total_objectives", 0),
        "current_index": direct_status.get("current_index", 0),
    }


def load_subagent_context(
    mcp_adapter: Any,
    run_data_manager: Any,
    *,
    last_n_steps: int,
    include_current_image: bool = True,
) -> Dict[str, Any]:
    """Fetch normalized state for local one-step subagents."""
    game_state_result = mcp_adapter.call_tool("get_game_state", {})
    if not game_state_result.get("success"):
        raise RuntimeError(game_state_result.get("error", "Failed to load game state"))

    memory_result = mcp_adapter.call_tool("get_memory_overview", {})
    memory_summary = ""
    if memory_result.get("success"):
        memory_summary = memory_result.get("overview", "") or ""

    skill_result = mcp_adapter.call_tool("get_skill_overview", {})
    skill_overview = ""
    if skill_result.get("success"):
        skill_overview = skill_result.get("overview", "") or ""

    trajectory_window = load_recent_trajectories(run_data_manager, last_n_steps=last_n_steps)
    current_image = None
    if include_current_image:
        current_image = decode_screenshot_base64(game_state_result.get("screenshot_base64"))

    return {
        "current_state": _extract_current_state(game_state_result),
        "objective_state": _extract_objective_state(game_state_result),
        "memory_summary": memory_summary.strip(),
        "skill_overview": skill_overview.strip(),
        "trajectory_window": trajectory_window,
        "trajectory_summary": format_trajectory_window(trajectory_window),
        "current_image": current_image,
        "game_state_result": game_state_result,
    }

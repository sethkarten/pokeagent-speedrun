"""
Server-side game tool implementations (no HTTP calls).

These helpers operate directly on the EmeraldEmulator instance and are called
by the /mcp/* endpoint handlers in server/app.py.  The MCP server
(server/cli/pokemon_mcp_server.py) does NOT import this module; it routes
tool calls to app.py over HTTP instead.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from utils.json_utils import serialize_for_json
from utils.mapping.pathfinding import Pathfinder
from utils.knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)

# Module-level singletons (lazy on first import of this module)
pathfinder = Pathfinder()
knowledge_base = get_knowledge_base()


# ---------------------------------------------------------------------------
# Porymap helpers
# ---------------------------------------------------------------------------

def load_porymap_for_pathfinding(state: dict) -> tuple:
    """Load porymap data into *state* for pathfinding, with game-state-aware override support.

    Returns:
        (coord_offset, state) - coord_offset is (offset_x, offset_y) if using override map, else None
    """
    coord_offset = None
    location_name = state.get("player", {}).get("location", "Unknown")

    if not location_name or location_name in ("Unknown", "TITLE_SEQUENCE"):
        return coord_offset, state

    try:
        from utils.mapping.porymap_json_builder import build_json_map_for_llm
        from utils.state_formatter import ROM_TO_PORYMAP_MAP
        from utils.mapping.ascii_map_loader import get_effective_map_name, get_override

        badge_count = 0
        badges = state.get("game", {}).get("badges", [])
        if isinstance(badges, list):
            badge_count = len(badges)
        elif isinstance(badges, int):
            badge_count = badges

        from pokemon_env.porymap_paths import get_porymap_root
        pokeemerald_root = get_porymap_root()
        if not pokeemerald_root:
            return coord_offset, state

        porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)
        if not porymap_map_name:
            return coord_offset, state

        effective_map_name = get_effective_map_name(porymap_map_name, badge_count=badge_count)
        override = get_override(effective_map_name)
        if override and ("offset_x" in override or "offset_y" in override):
            offset_x = override.get("offset_x", 0)
            offset_y = override.get("offset_y", 0)
            coord_offset = (offset_x, offset_y)
            logger.info(
                f"Porymap pathfinding: Using override map '{effective_map_name}' "
                f"with coord offset ({offset_x}, {offset_y})"
            )

        try:
            json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root, badge_count=badge_count)
        except ValueError as e:
            logger.error(f"Failed to build porymap for '{porymap_map_name}': {e}")
            return coord_offset, state

        if json_map and "grid" in json_map:
            raw_tiles = json_map.get("raw_tiles")

            if "map" not in state:
                state["map"] = {}
            if "porymap" not in state["map"]:
                state["map"]["porymap"] = {}

            state["map"]["porymap"]["grid"] = json_map["grid"]
            state["map"]["porymap"]["objects"] = json_map.get("objects", [])
            state["map"]["porymap"]["dimensions"] = json_map.get("dimensions", {})
            state["map"]["porymap"]["warps"] = json_map.get("warps", [])
            state["map"]["porymap"]["raw_tiles"] = raw_tiles

            grid_dims = (
                f"{len(json_map['grid'][0])}x{len(json_map['grid'])}" if json_map.get("grid") else "None"
            )
            logger.info(
                f"Loaded porymap for pathfinding: '{effective_map_name}' "
                f"(ROM: '{location_name}'), grid: {grid_dims}, badges: {badge_count}"
            )

    except Exception as e:
        logger.warning(f"Failed to load porymap data for pathfinding: {e}")

    return coord_offset, state


# ---------------------------------------------------------------------------
# _direct helpers — called by app.py /mcp/* endpoint handlers
# ---------------------------------------------------------------------------

def get_game_state_direct(env, state_formatter, action_history=None, current_obs=None) -> dict:
    """Get game state without HTTP calls.

    Args:
        env: EmeraldEmulator instance
        state_formatter: format_state_for_llm function
        action_history: Optional list of recent actions with start/end positions
        current_obs: Optional numpy array of current frame (from game loop)

    Returns:
        Dictionary with success status and state data including screenshot
    """
    try:
        import base64
        import io
        from PIL import Image
        import numpy as np

        screenshot = None
        if current_obs is not None:
            screenshot = Image.fromarray(current_obs)
            logger.debug("Using current_obs (latest frame from game loop)")
        elif hasattr(env, "current_frame") and env.current_frame is not None:
            screenshot = Image.fromarray(env.current_frame)
            logger.debug("Using env.current_frame (background-polled)")
        elif hasattr(env, "get_screenshot"):
            screenshot = env.get_screenshot()
            logger.debug("Using env.get_screenshot() (direct video buffer - may be stale)")

        state = env.get_comprehensive_state(screenshot=screenshot)
        state_text = state_formatter(state, action_history=action_history)

        screenshot_b64 = None
        if screenshot is not None:
            buffered = io.BytesIO()
            screenshot.save(buffered, format="PNG")
            screenshot_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            logger.debug("Encoded fresh screenshot (caching disabled)")

        result = {
            "success": True,
            "state_text": state_text,
            "screenshot_base64": screenshot_b64,
            "player_position": state.get("player", {}).get("position", {}),
            "location": state.get("player", {}).get("location", "Unknown"),
            "raw_state": state,
        }

        return serialize_for_json(result)

    except Exception as e:
        logger.error(f"Failed to get game state: {e}")
        return {"success": False, "error": str(e)}


def navigate_to_direct(
    env,
    x,
    y,
    reason: str = "",
    variance: Optional[str] = None,
    consider_npcs: bool = True,
    blocked_coords: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """Calculate path to coordinates without HTTP calls.

    Returns buttons to be queued via take_action.
    """
    try:
        x = int(x)
        y = int(y)

        variance_level = None
        nav_reason = "" if reason is None else str(reason)

        if variance is not None:
            variance_str = str(variance).strip()
            variance_lower = variance_str.lower()
            if variance_lower in {"low", "medium", "high"}:
                variance_level = variance_lower
            elif variance_lower in {"", "none"}:
                variance_level = None
            elif not reason:
                nav_reason = variance_str
        elif nav_reason.lower() in {"low", "medium", "high"}:
            variance_level = nav_reason.lower()
            nav_reason = ""

        state = env.get_comprehensive_state()

        location_name = state.get("player", {}).get("location", "Unknown")
        coord_offset = None
        if location_name and location_name not in ("Unknown", "TITLE_SEQUENCE"):
            try:
                from utils.mapping.porymap_json_builder import build_json_map_for_llm
                from utils.mapping.ascii_map_loader import get_effective_map_name, get_override
                from utils.state_formatter import ROM_TO_PORYMAP_MAP

                badge_count = 0
                badges = state.get("game", {}).get("badges", [])
                if isinstance(badges, list):
                    badge_count = len(badges)
                elif isinstance(badges, int):
                    badge_count = badges

                from pokemon_env.porymap_paths import get_porymap_root
                pokeemerald_root = get_porymap_root()

                if pokeemerald_root:
                    porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)

                    if porymap_map_name:
                        effective_map_name = get_effective_map_name(porymap_map_name, badge_count=badge_count)
                        override = get_override(effective_map_name)
                        if override and ("offset_x" in override or "offset_y" in override):
                            offset_x = override.get("offset_x", 0)
                            offset_y = override.get("offset_y", 0)
                            coord_offset = (offset_x, offset_y)
                            logger.info(
                                f"Using override map '{effective_map_name}' with coord offset ({offset_x}, {offset_y})"
                            )

                        try:
                            json_map = build_json_map_for_llm(
                                porymap_map_name, pokeemerald_root, badge_count=badge_count
                            )
                        except ValueError as e:
                            logger.error(
                                f"Failed to build porymap for '{porymap_map_name}' due to corrupted tileset: {e}"
                            )
                            json_map = None

                        if json_map and "grid" in json_map:
                            raw_tiles = json_map.get("raw_tiles")
                            player_pos = state.get("player", {}).get("position", {})
                            px = player_pos.get("x", 0)
                            py = player_pos.get("y", 0)

                            if raw_tiles and 0 <= py < len(raw_tiles) and 0 <= px < len(raw_tiles[py]):
                                player_tile = raw_tiles[py][px]
                                if len(player_tile) >= 4:
                                    player_elevation = player_tile[3]
                                    logger.info(
                                        f"Skipping elevation filtering - pathfinding handles elevation "
                                        f"via E0 connectors (player at elevation {player_elevation})"
                                    )

                            if "map" not in state:
                                state["map"] = {}
                            if "porymap" not in state["map"]:
                                state["map"]["porymap"] = {}

                            state["map"]["porymap"]["grid"] = json_map["grid"]
                            state["map"]["porymap"]["objects"] = json_map.get("objects", [])
                            state["map"]["porymap"]["dimensions"] = json_map.get("dimensions", {})
                            state["map"]["porymap"]["warps"] = json_map.get("warps", [])
                            state["map"]["porymap"]["raw_tiles"] = raw_tiles

                            grid_dims = (
                                f"{len(json_map['grid'][0])}x{len(json_map['grid'])}"
                                if json_map.get("grid")
                                else "None"
                            )
                            logger.info(
                                f"Loaded UNFILTERED porymap for pathfinding: '{porymap_map_name}' "
                                f"(ROM: '{location_name}'), grid: {grid_dims}"
                            )
                        else:
                            logger.warning(f"Porymap data for '{porymap_map_name}' missing grid data")
                    else:
                        logger.warning(f"Could not map ROM location '{location_name}' to porymap map name")
                else:
                    logger.warning("Could not find pokeemerald root for porymap data")
            except Exception as porymap_err:
                logger.warning(f"Failed to load porymap data for pathfinding: {porymap_err}")

        player_pos = state.get("player", {}).get("position", {})
        rom_start_x = player_pos.get("x", 0)
        rom_start_y = player_pos.get("y", 0)

        start_x = rom_start_x
        start_y = rom_start_y
        goal_x = x
        goal_y = y
        if coord_offset:
            offset_x, offset_y = coord_offset
            start_x = rom_start_x - offset_x
            start_y = rom_start_y - offset_y
            logger.info(
                f"Coordinate translation: ROM ({rom_start_x}, {rom_start_y}) -> "
                f"local ({start_x}, {start_y}), goal ({goal_x}, {goal_y})"
            )

        start = (start_x, start_y)
        goal = (goal_x, goal_y)

        goal_was_blocked = False
        map_data = state.get("map", {}).get("porymap", {})
        if map_data.get("grid"):
            grid = map_data["grid"]
            if 0 <= goal_y < len(grid):
                row = grid[goal_y]
                if isinstance(row, (list, str)) and 0 <= goal_x < len(row):
                    cell = row[goal_x] if isinstance(row, str) else row[goal_x]
                    if cell == "#":
                        goal_was_blocked = True

        buttons = pathfinder.find_path(
            start, goal, state, variance=variance_level, consider_npcs=consider_npcs, blocked_coords=blocked_coords
        )

        if not buttons:
            error_msg = f"No path found from ({start_x}, {start_y}) to ({x}, {y})"

            map_data = state.get("map", {}).get("porymap", {})
            if map_data.get("grid"):
                grid = map_data["grid"]
                if 0 <= y < len(grid):
                    row = grid[y]
                    if isinstance(row, (list, str)) and 0 <= x < len(row):
                        cell = row[x] if isinstance(row, str) else row[x]
                        if cell == "#":
                            error_msg += " - Target is blocked by a wall or obstacle"
                        elif cell == "W":
                            error_msg += " - Target requires Surf (water)"
                        elif cell in ["X", "?"]:
                            error_msg += " - Target is out of bounds or unexplored"
                        else:
                            error_msg += f" - Target tile '{cell}' may be unreachable from current position"

            manhattan_dist = abs(x - start_x) + abs(y - start_y)
            if manhattan_dist > 100:
                error_msg += f" (distance: {manhattan_dist} tiles - may be too far)"

            return {"success": False, "error": error_msg, "target": f"({x}, {y})", "reason": "unreachable"}

        nav_reason_text = f"Navigating from ({start_x}, {start_y}) to ({x}, {y})"
        if nav_reason:
            nav_reason_text += f": {nav_reason}"
        if variance_level:
            nav_reason_text += f" (variance={variance_level})"

        logger.info(f"🗺️ {nav_reason_text} - Buttons calculated: {len(buttons)}")

        result = {
            "success": True,
            "buttons": buttons,
            "target": f"({x}, {y})",
            "path_length": len(buttons),
            "reason": nav_reason,
            "variance": variance_level or "none",
        }

        if goal_was_blocked:
            result["note"] = f"Requested target ({x}, {y}) was blocked; navigated to nearest reachable tile"

        return result
    except Exception as e:
        logger.error(f"Failed to navigate: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Knowledge base helpers
# ---------------------------------------------------------------------------

def add_knowledge_direct(category, title, content, location=None, coordinates=None, importance=3) -> dict:
    """Add knowledge to knowledge base - for use by server endpoints."""
    try:
        entry_id = knowledge_base.add(
            category=category,
            title=title,
            content=content,
            location=location,
            coordinates=coordinates,
            importance=importance,
        )
        logger.info(f"📝 Added knowledge: {title} ({category})")
        return {"success": True, "entry_id": entry_id, "message": f"Stored knowledge: {title}"}
    except Exception as e:
        logger.error(f"Failed to add knowledge: {e}")
        return {"success": False, "error": str(e)}


def search_knowledge_direct(category=None, query="", location="", min_importance=1) -> dict:
    """Search knowledge base - for use by server endpoints."""
    try:
        results = knowledge_base.search(
            category=category, location=location or None, query=query or None, min_importance=min_importance
        )
        return {"success": True, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Failed to search knowledge: {e}")
        return {"success": False, "error": str(e)}


def get_knowledge_summary_direct(min_importance=3) -> dict:
    """Get knowledge summary - for use by server endpoints."""
    try:
        summary = knowledge_base.get_summary(min_importance=min_importance)
        return {"success": True, "summary": summary}
    except Exception as e:
        logger.error(f"Failed to get knowledge summary: {e}")
        return {"success": False, "error": str(e)}

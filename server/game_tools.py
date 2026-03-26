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
from utils.stores.memory import get_memory_store
from utils.stores.skills import get_skill_store
from utils.stores.subagents import get_subagent_store

logger = logging.getLogger(__name__)

# Module-level singletons (lazy on first import of this module)
pathfinder = Pathfinder()
memory_store = get_memory_store()
skill_store = get_skill_store()
subagent_store = get_subagent_store()


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

        # For Red: inject red_whole_map so _format_red_map_info() can build the ASCII map
        # and write porymap back into state for movement preview. porymap is NOT pre-injected
        # here because _format_map_info() constructs it from red_whole_map automatically.
        game_type = os.environ.get("GAME_TYPE", "emerald")
        if game_type == "red":
            try:
                if hasattr(env, 'memory_reader') and hasattr(env.memory_reader, 'map_reader'):
                    whole_map = env.memory_reader.map_reader.get_whole_map_data()
                    if whole_map and whole_map.get("grid"):
                        state.setdefault("map", {})["red_whole_map"] = whole_map
                    # add back porymap field for movement preview
                    state["map"]["porymap"] = {
                            "grid": whole_map["grid"],
                            "objects": whole_map.get("objects", []),
                            "dimensions": whole_map.get("dimensions", {}),
                            "warps": whole_map.get("warps", []),
                            "raw_tiles": whole_map.get("raw_tiles"),
                        }
            except Exception as e:
                logger.warning(f"Failed to inject red_whole_map in get_game_state_direct: {e}")

        try:
            state_text = state_formatter(state, action_history=action_history)
        except Exception as formatter_err:
            logger.exception("State formatter failed; returning fallback text with screenshot preserved")
            game_state_name = state.get("game", {}).get("game_state") or "unknown"
            location = state.get("player", {}).get("location") or "Unknown"
            position = state.get("player", {}).get("position") or {}
            state_text = (
                "State text formatter unavailable for this screen.\n"
                f"Game State: {game_state_name}\n"
                f"Location: {location}\n"
                f"Position: X={position.get('x', 'unknown')}, Y={position.get('y', 'unknown')}\n"
                "Use the attached screenshot as the source of truth for the current UI."
            )

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

        # --- Red: load native map data as porymap-compatible format ---
        _game_type = os.environ.get("GAME_TYPE", "emerald")
        if _game_type == "red":
            try:
                if hasattr(env, "memory_reader") and hasattr(env.memory_reader, "map_reader"):
                    whole_map = env.memory_reader.map_reader.get_whole_map_data()
                    if whole_map and whole_map.get("grid"):
                        if "map" not in state:
                            state["map"] = {}
                        state["map"]["porymap"] = {
                            "grid": whole_map["grid"],
                            "objects": whole_map.get("objects", []),
                            "dimensions": whole_map.get("dimensions", {}),
                            "warps": whole_map.get("warp_events", []),
                            "raw_tiles": whole_map.get("raw_tiles"),
                        }
                        logger.info(
                            f"Loaded Red map for pathfinding: '{whole_map.get('location')}', "
                            f"grid: {whole_map['dimensions'].get('width')}x{whole_map['dimensions'].get('height')}"
                        )
            except Exception as e:
                logger.warning(f"Failed to load Red map data for pathfinding: {e}")

        elif location_name and location_name not in ("Unknown", "TITLE_SEQUENCE"):
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
# Memory (long-term) helpers — formerly "knowledge base"
# ---------------------------------------------------------------------------

def add_memory_direct(category=None, title="", content="", location=None, coordinates=None, importance=3, path=None) -> dict:
    """Add an entry to long-term memory."""
    try:
        title_s = (title or "").strip()
        content_s = (content or "").strip()
        if not title_s and not content_s:
            return {
                "success": False,
                "error": "title and content cannot both be empty (refusing useless memory rows)",
            }
        effective_path = path or category or "uncategorized"
        entry_id = memory_store.add(
            path=effective_path,
            title=title_s,
            content=content_s,
            location=location,
            coordinates=coordinates,
            importance=importance,
        )
        logger.info(f"📝 Added memory: {title_s} ({effective_path})")
        return {"success": True, "entry_id": entry_id, "message": f"Stored memory: {title_s}"}
    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return {"success": False, "error": str(e)}


def search_memory_direct(category=None, query="", location="", min_importance=1, path=None) -> dict:
    """Search long-term memory."""
    try:
        results = memory_store.search(
            path=path, category=category, location=location or None, query=query or None, min_importance=min_importance
        )
        return {"success": True, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Failed to search memory: {e}")
        return {"success": False, "error": str(e)}


def get_memory_summary_direct(min_importance=3) -> dict:
    """Get long-term memory summary (legacy content dump)."""
    try:
        summary = memory_store.get_summary(min_importance=min_importance)
        return {"success": True, "summary": summary}
    except Exception as e:
        logger.error(f"Failed to get memory summary: {e}")
        return {"success": False, "error": str(e)}


def get_memory_overview_direct() -> dict:
    """Get compact tree overview of long-term memory ([id] title grouped by path)."""
    try:
        overview = memory_store.get_tree_overview()
        return {"success": True, "overview": overview}
    except Exception as e:
        logger.error(f"Failed to get memory overview: {e}")
        return {"success": False, "error": str(e)}


def _normalize_process_reasoning(reasoning: object) -> Optional[str]:
    """Return stripped non-empty reasoning string, or None if missing/invalid."""
    if reasoning is None:
        return None
    text = str(reasoning).strip()
    return text if text else None


def _validate_process_entries_list(entries: object) -> Optional[str]:
    """Return an error message if *entries* is not a non-empty list."""
    if not isinstance(entries, list):
        return "entries must be a list"
    if len(entries) == 0:
        return "entries must be non-empty (provide at least one entry object for this action)"
    return None


def _batch_all_succeeded(results: list) -> bool:
    """True only if every per-entry result reports success (batch semantics)."""
    return bool(results) and all(r.get("success") for r in results)


def process_memory_direct(action: str, entries: list, reasoning: object) -> dict:
    """Unified CRUD dispatch for long-term memory.

    ``action`` is one of: read, add, update, delete.
    ``entries`` is a list of per-entry dicts with action-specific fields.
    ``reasoning`` is required (non-empty): why this memory operation is appropriate.
    """
    try:
        if _normalize_process_reasoning(reasoning) is None:
            return {
                "success": False,
                "error": "reasoning is required (non-empty string explaining why this memory operation is needed)",
            }
        ent_err = _validate_process_entries_list(entries)
        if ent_err:
            return {"success": False, "error": ent_err, "results": []}
        results = []
        for entry_data in entries:
            if action == "read":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for read"})
                    continue
                entry = memory_store.get(entry_id)
                if entry is None:
                    results.append({"success": False, "error": f"Entry {entry_id} not found"})
                else:
                    results.append({"success": True, "entry": memory_store.to_display_dict(entry)})

            elif action == "add":
                path = entry_data.get("path", entry_data.get("category", "uncategorized"))
                title = (entry_data.get("title") or "").strip()
                content = (entry_data.get("content") or "").strip()
                if not title and not content:
                    results.append(
                        {
                            "success": False,
                            "error": "add requires non-empty title and/or content (empty entries create useless memory rows)",
                        }
                    )
                    continue
                importance = int(entry_data.get("importance", 3))
                location = entry_data.get("location")
                coordinates = entry_data.get("coordinates")
                if isinstance(coordinates, str) and "," in coordinates:
                    parts = coordinates.split(",")
                    try:
                        coordinates = (int(parts[0].strip()), int(parts[1].strip()))
                    except (ValueError, IndexError):
                        coordinates = None
                add_kwargs = dict(
                    path=path, title=title or "", content=content or "",
                    importance=importance, location=location,
                    coordinates=coordinates,
                )
                custom_id = entry_data.get("id", "")
                if custom_id and isinstance(custom_id, str) and custom_id.strip():
                    add_kwargs["id"] = custom_id.strip()
                entry_id = memory_store.add(**add_kwargs)
                results.append({"success": True, "entry_id": entry_id})

            elif action == "update":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for update"})
                    continue
                update_fields = {k: v for k, v in entry_data.items() if k != "id" and v is not None}
                ok = memory_store.update(entry_id, **update_fields)
                results.append({"success": ok, "entry_id": entry_id})

            elif action == "delete":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for delete"})
                    continue
                ok = memory_store.remove(entry_id)
                results.append({"success": ok, "entry_id": entry_id})

            else:
                results.append({"success": False, "error": f"Unknown action: {action}"})

        return {"success": _batch_all_succeeded(results), "results": results}
    except Exception as e:
        logger.error(f"process_memory error: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Skill library helpers
# ---------------------------------------------------------------------------

def get_skill_overview_direct() -> dict:
    """Get compact tree overview of the skill library ([id] name grouped by path)."""
    try:
        overview = skill_store.get_tree_overview()
        return {"success": True, "overview": overview}
    except Exception as e:
        logger.error(f"Failed to get skill overview: {e}")
        return {"success": False, "error": str(e)}


def process_skill_direct(action: str, entries: list, reasoning: object) -> dict:
    """Unified CRUD dispatch for the skill library.

    ``reasoning`` is required (non-empty): why this skill operation is appropriate.
    """
    try:
        if _normalize_process_reasoning(reasoning) is None:
            return {
                "success": False,
                "error": "reasoning is required (non-empty string explaining why this skill operation is needed)",
            }
        ent_err = _validate_process_entries_list(entries)
        if ent_err:
            return {"success": False, "error": ent_err, "results": []}
        results = []
        for entry_data in entries:
            if action == "read":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for read"})
                    continue
                entry = skill_store.get(entry_id)
                if entry is None:
                    results.append({"success": False, "error": f"Skill {entry_id} not found"})
                else:
                    results.append({"success": True, "entry": skill_store.to_display_dict(entry)})

            elif action == "add":
                raw_name = entry_data.get("name", entry_data.get("title", ""))
                name = (raw_name or "").strip() if isinstance(raw_name, str) else str(raw_name or "").strip()
                raw_desc = entry_data.get("description", "")
                description = (
                    (raw_desc or "").strip() if isinstance(raw_desc, str) else str(raw_desc or "").strip()
                )
                if not name or not description:
                    results.append(
                        {
                            "success": False,
                            "error": (
                                "add requires non-empty name (or title) and description "
                                "(empty objects {} are rejected; do not batch placeholder rows)"
                            ),
                        }
                    )
                    continue
                path = entry_data.get("path", "general")
                effectiveness = entry_data.get("effectiveness", "medium")
                importance = int(entry_data.get("importance", 3))
                code = entry_data.get("code", "")
                # Allow agent to specify a custom ID (human-readable name)
                custom_id = entry_data.get("id", "")
                add_kwargs = dict(
                    path=path, name=name, title=name, description=description,
                    effectiveness=effectiveness, importance=importance,
                    code=code if isinstance(code, str) else "",
                )
                if custom_id and isinstance(custom_id, str) and custom_id.strip():
                    add_kwargs["id"] = custom_id.strip()
                entry_id = skill_store.add(**add_kwargs)
                results.append({"success": True, "entry_id": entry_id})

            elif action == "update":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for update"})
                    continue
                update_fields = {k: v for k, v in entry_data.items() if k != "id" and v is not None}
                if "name" in update_fields:
                    update_fields.setdefault("title", update_fields["name"])
                ok = skill_store.update(entry_id, **update_fields)
                results.append({"success": ok, "entry_id": entry_id})

            elif action == "delete":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for delete"})
                    continue
                ok = skill_store.remove(entry_id)
                results.append({"success": ok, "entry_id": entry_id})

            else:
                results.append({"success": False, "error": f"Unknown action: {action}"})

        return {"success": _batch_all_succeeded(results), "results": results}
    except Exception as e:
        logger.error(f"process_skill error: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Subagent registry helpers
# ---------------------------------------------------------------------------

def get_subagent_overview_direct() -> dict:
    """Get compact tree overview of the subagent registry ([id] name grouped by path)."""
    try:
        overview = subagent_store.get_tree_overview()
        return {"success": True, "overview": overview}
    except Exception as e:
        logger.error(f"Failed to get subagent overview: {e}")
        return {"success": False, "error": str(e)}


def process_subagent_direct(action: str, entries: list, reasoning: object) -> dict:
    """Unified CRUD dispatch for the subagent registry.

    ``reasoning`` is required (non-empty): why this subagent operation is appropriate.
    """
    try:
        if _normalize_process_reasoning(reasoning) is None:
            return {
                "success": False,
                "error": "reasoning is required (non-empty string explaining why this subagent operation is needed)",
            }
        ent_err = _validate_process_entries_list(entries)
        if ent_err:
            return {"success": False, "error": ent_err, "results": []}
        results = []
        for entry_data in entries:
            if action == "read":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for read"})
                    continue
                entry = subagent_store.get(entry_id)
                if entry is None:
                    results.append({"success": False, "error": f"Subagent {entry_id} not found"})
                else:
                    results.append({"success": True, "entry": subagent_store.to_display_dict(entry)})

            elif action == "add":
                raw_name = entry_data.get("name", entry_data.get("title", ""))
                name = (raw_name or "").strip() if isinstance(raw_name, str) else str(raw_name or "").strip()
                raw_desc = entry_data.get("description", "")
                description = (
                    (raw_desc or "").strip() if isinstance(raw_desc, str) else str(raw_desc or "").strip()
                )
                if not name or not description:
                    results.append(
                        {
                            "success": False,
                            "error": (
                                "add requires non-empty name (or title) and description "
                                "(empty objects {} are rejected; fill fields before calling)"
                            ),
                        }
                    )
                    continue
                path = entry_data.get("path", "custom")
                handler_type = entry_data.get("handler_type", "looping")
                max_turns = int(entry_data.get("max_turns", 25))
                available_tools = entry_data.get("available_tools", [])
                system_instructions = entry_data.get("system_instructions", "")
                directive = entry_data.get("directive", "")
                return_condition = entry_data.get("return_condition", "")
                importance = int(entry_data.get("importance", 3))
                try:
                    add_kwargs = dict(
                        path=path, name=name, title=name, description=description,
                        handler_type=handler_type, max_turns=max_turns,
                        available_tools=available_tools,
                        system_instructions=system_instructions,
                        directive=directive, return_condition=return_condition,
                        importance=importance,
                    )
                    custom_id = entry_data.get("id", "")
                    if custom_id and isinstance(custom_id, str) and custom_id.strip():
                        add_kwargs["id"] = custom_id.strip()
                    entry_id = subagent_store.add(**add_kwargs)
                    results.append({"success": True, "entry_id": entry_id})
                except ValueError as ve:
                    results.append({"success": False, "error": str(ve)})

            elif action == "update":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for update"})
                    continue
                update_fields = {k: v for k, v in entry_data.items() if k != "id" and v is not None}
                if "name" in update_fields:
                    update_fields.setdefault("title", update_fields["name"])
                try:
                    ok = subagent_store.update(entry_id, **update_fields)
                    results.append({"success": ok, "entry_id": entry_id})
                except ValueError as ve:
                    results.append({"success": False, "error": str(ve)})

            elif action == "delete":
                entry_id = entry_data.get("id")
                if not entry_id:
                    results.append({"success": False, "error": "Missing 'id' for delete"})
                    continue
                entry = subagent_store.get(entry_id)
                if entry is not None and getattr(entry, "is_builtin", False):
                    results.append({"success": False, "error": f"Cannot delete built-in subagent {entry_id}"})
                    continue
                ok = subagent_store.remove(entry_id)
                results.append({"success": ok, "entry_id": entry_id})

            else:
                results.append({"success": False, "error": f"Unknown action: {action}"})

        return {"success": _batch_all_succeeded(results), "results": results}
    except Exception as e:
        logger.error(f"process_subagent error: {e}")
        return {"success": False, "error": str(e)}


# Backward-compat aliases
add_knowledge_direct = add_memory_direct
search_knowledge_direct = search_memory_direct
get_knowledge_summary_direct = get_memory_summary_direct
get_knowledge_base = lambda: memory_store

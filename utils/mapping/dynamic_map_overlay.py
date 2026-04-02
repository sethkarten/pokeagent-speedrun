#!/usr/bin/env python3
"""
Dynamic map overlay helpers for runtime-changing maps.

For selected locations, this module reads LIVE metatiles from emulator memory
and rebuilds the pathfinding grid so dynamic collision changes (e.g. gym
barriers) are reflected in navigation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from utils.mapping.map_formatter import format_tile_to_symbol

logger = logging.getLogger(__name__)

# Expand this allowlist as additional dynamic maps are supported.
DYNAMIC_MAP_LOCATIONS = {
    "MAUVILLE CITY GYM",
    # "FORTREE CITY GYM",
}

# GBA map buffers include a 7-tile border around the real map.
GBA_MAP_BORDER_OFFSET = 7


def is_dynamic_location(location_name: str) -> bool:
    """Return True if the given location needs live metatile overlays."""
    return bool(location_name) and location_name.upper() in DYNAMIC_MAP_LOCATIONS


def _read_live_tiles(memory_reader: Any, width: int, height: int, location_name: str) -> List[List[Any]] | None:
    """Read live metatiles from the emulator, returning None on failure."""
    if memory_reader is None:
        logger.warning("Dynamic overlay skipped for %s: memory_reader unavailable", location_name)
        return None

    if not getattr(memory_reader, "_map_buffer_addr", None):
        find_buffer_fn = getattr(memory_reader, "_find_map_buffer_addresses", None)
        if not callable(find_buffer_fn) or not find_buffer_fn():
            logger.warning("Dynamic overlay skipped for %s: failed to locate map buffer", location_name)
            return None

    live_raw_tiles = memory_reader.read_map_metatiles(
        x_start=GBA_MAP_BORDER_OFFSET,
        y_start=GBA_MAP_BORDER_OFFSET,
        width=width,
        height=height,
    )
    if not live_raw_tiles:
        logger.warning("Dynamic overlay skipped for %s: live metatile read returned empty", location_name)
        return None
    return live_raw_tiles


def _build_live_grid(live_raw_tiles: List[List[Any]], location_name: str) -> List[List[str]]:
    """Convert raw metatile data into an ASCII symbol grid."""
    live_grid: List[List[str]] = []
    for y, row in enumerate(live_raw_tiles):
        grid_row: List[str] = []
        for x, tile in enumerate(row):
            grid_row.append(format_tile_to_symbol(tile, x=x, y=y, location_name=location_name))
        live_grid.append(grid_row)
    return live_grid


# ------------------------------------------------------------------
# Public API: overlay on a json_map dict (used inside _format_porymap_info)
# ------------------------------------------------------------------

def apply_live_overlay_to_json_map(json_map: Dict[str, Any], memory_reader: Any, location_name: str) -> bool:
    """Replace grid/raw_tiles/ascii in a json_map dict with live emulator data.

    This is the preferred entry-point for code paths that work with the
    ``json_map`` dictionary produced by ``build_json_map_for_llm``.

    Returns True if the overlay was applied.
    """
    if not is_dynamic_location(location_name):
        return False

    dims = json_map.get("dimensions") or {}
    width = int(dims.get("width", 0) or 0)
    height = int(dims.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        logger.warning("Dynamic overlay skipped for %s: missing json_map dimensions", location_name)
        return False

    live_raw_tiles = _read_live_tiles(memory_reader, width, height, location_name)
    if live_raw_tiles is None:
        return False

    live_grid = _build_live_grid(live_raw_tiles, location_name)

    json_map["raw_tiles"] = live_raw_tiles
    json_map["grid"] = live_grid
    json_map["ascii"] = "\n".join("".join(row) for row in live_grid)

    logger.info("Dynamic overlay applied to json_map for %s (%sx%s)", location_name, width, height)
    return True


# ------------------------------------------------------------------
# Public API: overlay on the full game state dict (used in navigate_to)
# ------------------------------------------------------------------

def apply_live_metatile_overlay(state: Dict[str, Any], env: Any, location_name: str) -> bool:
    """Replace porymap grid/raw_tiles in the game *state* dict with live data.

    Returns True if overlay was applied, otherwise False.
    """
    if not is_dynamic_location(location_name):
        return False

    porymap = state.get("map", {}).get("porymap", {})
    dimensions = porymap.get("dimensions", {}) or {}
    width = int(dimensions.get("width", 0) or 0)
    height = int(dimensions.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        logger.warning("Dynamic overlay skipped for %s: missing porymap dimensions", location_name)
        return False

    memory_reader = getattr(env, "memory_reader", None)
    live_raw_tiles = _read_live_tiles(memory_reader, width, height, location_name)
    if live_raw_tiles is None:
        return False

    live_grid = _build_live_grid(live_raw_tiles, location_name)

    state.setdefault("map", {}).setdefault("porymap", {})
    state["map"]["porymap"]["raw_tiles"] = live_raw_tiles
    state["map"]["porymap"]["grid"] = live_grid

    logger.info(
        "Dynamic overlay applied for %s using live metatiles (%sx%s)",
        location_name,
        width,
        height,
    )
    return True

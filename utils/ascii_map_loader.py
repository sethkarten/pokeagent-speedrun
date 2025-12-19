#!/usr/bin/env python3
"""
ASCII Map Loader - Load corrected ground truth ASCII maps directly
Bypasses map.bin binary files for specific corrected maps
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from pokemon_env.enums import MetatileBehavior


# Corrected ASCII ground truth maps (dimensions: 18 width x 21 height)
CORRECTED_MAPS = {
    "LavaridgeTown_Gym_B1F": """##################
##################
....#..S..#......#
..S.........S....#
##################
##################
S..S#..#.#...#...#
....#..#.#...#...#
....#..#.#...#...#
....#..#S#...#...#
S...#..#.#↓↓↓#↓↓↓#
....#↓↓#.#...#...#
↓↓↓↓#..#.#..S#...#
....#..#.#...#...#
.S..#S.#.#...#...#
....#..#.#↓↓↓#...#
....S....#...#...#
S........#.......#
....S....#S......#
##################
##################""",

    "LavaridgeTown_Gym_1F": """##################
##################
.....S#S...#....##
..S...#....#S...##
......#....#....##
......#....#....##
S....S#.S.S#..S.##
##################
##################
....#S.#S.##...###
S..S#..#..##...###
....#..#..#.....##
....#..#..#.S...##
....#..#..#↓↓↓↓↓##
.S.S#S.#..#.....##
..S.#↓↓#..#.....##
....#..#.........#
S......#.........#
.......#..S......#
#############DD###
##################"""
}


def ascii_to_metatiles(ascii_map: str, map_name: str = "") -> List[List[Tuple[int, Any, int, int]]]:
    """
    Convert ASCII map to metatile format compatible with porymap_json_builder.

    Returns:
        List[List[Tuple]] - 2D array where each tile is (tile_id, behavior, collision, elevation)
    """
    lines = ascii_map.strip().split('\n')
    height = len(lines)
    width = len(lines[0]) if height > 0 else 0

    metatiles = []

    for y, line in enumerate(lines):
        row = []
        for x, char in enumerate(line):
            # Map ASCII characters to metatile properties
            # Format: (tile_id, behavior, collision, elevation)

            if char == '#':
                # Blocked/wall tile
                tile = (1, MetatileBehavior.NORMAL, 1, 3)  # collision=1 (blocked)
            elif char == '.':
                # Walkable floor tile
                tile = (2, MetatileBehavior.NORMAL, 0, 3)  # collision=0 (walkable)
            elif char == 'S':
                # Stairs/warp tile (always walkable)
                # Use specific gym warp behavior based on map name
                if "Gym_1F" in map_name:
                    behavior = MetatileBehavior.LAVARIDGE_GYM_1F_WARP
                elif "Gym_B1F" in map_name:
                    behavior = MetatileBehavior.LAVARIDGE_GYM_B1F_WARP
                else:
                    behavior = MetatileBehavior.NORMAL
                tile = (3, behavior, 0, 3)
            elif char == 'D':
                # Door tile (always walkable)
                tile = (4, MetatileBehavior.ANIMATED_DOOR, 0, 3)
            elif char == 'P':
                # Player position (walkable)
                tile = (2, MetatileBehavior.NORMAL, 0, 3)
            elif char in ['↓', '↑', '←', '→']:
                # Ledge tiles (one-way movement)
                if char == '↓':
                    tile = (5, MetatileBehavior.JUMP_SOUTH, 0, 3)
                elif char == '↑':
                    tile = (6, MetatileBehavior.JUMP_NORTH, 0, 3)
                elif char == '←':
                    tile = (7, MetatileBehavior.JUMP_WEST, 0, 3)
                else:  # '→'
                    tile = (8, MetatileBehavior.JUMP_EAST, 0, 3)
            else:
                # Unknown character - treat as walkable
                tile = (2, MetatileBehavior.NORMAL, 0, 3)

            row.append(tile)
        metatiles.append(row)

    return metatiles


def load_corrected_map(map_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a corrected ASCII ground truth map if available.

    Args:
        map_name: Map name like "LavaridgeTown_Gym_B1F"

    Returns:
        Dict with 'raw_tiles' and 'dimensions', or None if no corrected map exists
    """
    if map_name not in CORRECTED_MAPS:
        return None

    ascii_map = CORRECTED_MAPS[map_name]
    metatiles = ascii_to_metatiles(ascii_map, map_name)

    height = len(metatiles)
    width = len(metatiles[0]) if height > 0 else 0

    return {
        'raw_tiles': metatiles,
        'dimensions': {'width': width, 'height': height}
    }


def has_corrected_map(map_name: str) -> bool:
    """Check if a corrected map exists for the given map name."""
    return map_name in CORRECTED_MAPS

#!/usr/bin/env python3
"""
ASCII Map Loader - Override porymap data with corrected/custom map data.

Supports selective overrides:
- ascii: Replace the ASCII grid representation
- warps: Replace warp definitions
- objects: Replace NPC/object definitions
- connections: Replace map connections

Any field not specified uses the original porymap data.
Conditional loading is supported via selector functions based on game state.
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from pokemon_env.enums import MetatileBehavior
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MAP OVERRIDES - Partial or full replacements for porymap data
# =============================================================================

# Each override can specify any subset of: ascii, warps, objects, connections, offset_x, offset_y
# Missing fields will use original porymap data
# offset_x/offset_y: Translate ROM coordinates to local map coordinates (local = rom - offset). This is used if the map dimensions change

MAP_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Petalburg City fixing map bug preventing correct pathfinding
    "PetalburgCity": {
        "ascii": """##############################
##############################
###################...WWWW####
###################WWWWWWW####
###################WWWWWWW####
####.##D###########WWWWWWW####
####......#########WWWWWWW####
####......#####D###WWWWWWW####
####......##..#.#.#WWWWWWW####
####....####......#.......####
##...............#......######
##......................######
........................#D####
............................##
##WWWWW##..........####.....##
##WWWWW##..........####.....##
##WWWWW##........#.#D##.......
##WWWWW######.................
##WWWWW######.................
##WWWWW###D##.................
##WWWWW...................####
##WWWWW...........############
##WWWWW...........############
##WWWWWWWWWW......##D#########
##WWWWWWWWWW##.....####..#####
##WWWWWWWWWW##...........#####
##WWWWWWWWWW##...........#####
##WWWWWWWWWW##............####
##...######.##################
##...######.##################""",
    },

    # Rustboro City - fixing map bug preventing automatic pathfinding to gym
    "RustboroCity": {
        "ascii": """
##WWWWWWWW#########.....################
##WWW##WWW#########.....################
WWWWW##WWW#########.....################
WWWWWWWWWW#########.....################
WWWWWWWWWW#########.....################
WWWWWW#....########.......##############
WW##W#.......######.......##############
WW##W#.############.....#############.##
WWWWW#############............#.........
WWWWW#############......................
WWWWW#############......................
##WWW#############......................
##WWW#############......................
WWWWW#############......###############.
WWWWW######DD#####......#...........###.
WWWWW##...#..#...#......#...........#.##
WWWWW##..........#.#..#.######.######.##
WWWWW##..........#......######.########.
WWWW#.#.#......#.#......###D##.########.
WWWW#.#.........##.....#..#.#..##D###.##
WWWW#.#..#....#..#..................#.##
WWWW#.#.............................###.
WWWW#.#.............................###.
WWW#..############..................#.##
WWW#..#............#..#.............#.##
WWW#..#.................###############.
WWW####..#########......#...#####...###.
WW#.###..#########......#.#######...#.##
WW#.###..#########......#.####D##...#.##
WW#.###..#########......#...........###.
W#..###..####D####......#...........###.
W#..###............#..#.#######...###.##
W######.................#######.....#.##
W######.................#######.....###.
..#####.................###D###.....###.
..##########.............#..........#.##
############...####.................#.##
############...####.................###.
#########D###..#D###..#....###......###.
#######....................###......#.##
#######....................###.####.#.##
###............................####.###.
###............................####.###.
###............####.....######.####.#.##
##########.....####.....######.####.#.##
##########.....#D###..#.######.####.###.
##########..............##D###.####.###.
##########..........................#.##
##########..........................#.##
##########.........#.........#......###.
##########.....................########.
#####D####.....................#.....##.
###............................#.######.
###............................#.####.##
############........############.##.####
##.........#..#..#..#............##.####
##########.#........#.##################
##########.#........#.##################
######.....#........#.............######
############........#############.######""",
    },

    # Lavaridge Gym B1F - ASCII correction only (warps/objects from porymap)
    "LavaridgeTown_Gym_B1F": {
        "ascii": """##################
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
    },

    # Lavaridge Gym 1F - ASCII correction only
    "LavaridgeTown_Gym_1F": {
        "ascii": """##################
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
##################""",
    },

    # Petalburg Gym Lobby - Early game (< 4 badges)
    # Full override: smaller map with Norman in lobby position
    # ROM coordinates: lobby starts at y=105, so offset_y=105 translates ROM y=110 to local y=5
    "PetalburgCity_Gym_Lobby": {
        "offset_x": 0,
        "offset_y": 105,
        "ascii": """#########
.........
.........
.........
.........
.#.....#.
.........
####DD###""",
        "warps": [
            {"x": 4, "y": 7, "elevation": 3, "dest_map": "MAP_PETALBURG_CITY", "dest_warp_id": "2"},
            {"x": 5, "y": 7, "elevation": 3, "dest_map": "MAP_PETALBURG_CITY", "dest_warp_id": "2"},
        ],
        "objects": [
            {"x": 4, "y": 3, "elevation": 3, "graphics_id": "OBJ_EVENT_GFX_NORMAN",
             "movement_type": "MOVEMENT_TYPE_FACE_DOWN", "movement_range_x": 0,
             "movement_range_y": 0, "trainer_type": "TRAINER_TYPE_NONE",
             "trainer_sight_or_berry_tree_id": "0"},
        ],
        "connections": [],
    },
}


# =============================================================================
# CONDITIONAL MAP SELECTION - Select map variant based on game state
# =============================================================================

def _select_petalburg_gym(badge_count: int = 0, **kwargs) -> Optional[str]:
    """Return lobby variant if < 4 badges, else None for default."""
    if badge_count < 4:
        logger.info(f"PetalburgCity_Gym: Using lobby (badges={badge_count})")
        return "PetalburgCity_Gym_Lobby"
    return None


# Map selectors: original_name -> function(badge_count, ...) -> alternate_name or None
CONDITIONAL_SELECTORS: Dict[str, Callable[..., Optional[str]]] = {
    "PetalburgCity_Gym": _select_petalburg_gym,
}


# =============================================================================
# PUBLIC API
# =============================================================================

def get_effective_map_name(map_name: str, badge_count: int = 0, **kwargs) -> str:
    """Get the effective map name after applying conditional selection."""
    if map_name in CONDITIONAL_SELECTORS:
        alternate = CONDITIONAL_SELECTORS[map_name](badge_count=badge_count, **kwargs)
        if alternate:
            return alternate
    return map_name


def get_override(map_name: str) -> Optional[Dict[str, Any]]:
    """Get override data for a map, if any exists."""
    return MAP_OVERRIDES.get(map_name)


def has_override(map_name: str) -> bool:
    """Check if any override exists for the map."""
    return map_name in MAP_OVERRIDES


def ascii_to_metatiles(ascii_map: str, map_name: str = "") -> List[List[Tuple[int, Any, int, int]]]:
    """Convert ASCII map to metatile format for pathfinding."""
    lines = ascii_map.strip().split('\n')
    metatiles = []

    for line in lines:
        row = []
        for char in line:
            if char == '#':
                tile = (1, MetatileBehavior.NORMAL, 1, 3) # collision 1 means blocked
            elif char == '.':
                tile = (2, MetatileBehavior.NORMAL, 0, 3) # collision 0 means walkable
            elif char == 'S':
                # use specific gym warp behavir based on map name
                if "Gym_1F" in map_name:
                    behavior = MetatileBehavior.LAVARIDGE_GYM_1F_WARP
                elif "Gym_B1F" in map_name:
                    behavior = MetatileBehavior.LAVARIDGE_GYM_B1F_WARP
                else:
                    behavior = MetatileBehavior.NORMAL
                tile = (3, behavior, 0, 3)
            elif char == 'D':
                tile = (4, MetatileBehavior.ANIMATED_DOOR, 0, 3)
            elif char == 'P':
                # Player position marker (walkable)
                tile = (2, MetatileBehavior.NORMAL, 0, 3)
            elif char == '↓':
                tile = (5, MetatileBehavior.JUMP_SOUTH, 0, 3)
            elif char == '↑':
                tile = (6, MetatileBehavior.JUMP_NORTH, 0, 3)
            elif char == '←':
                tile = (7, MetatileBehavior.JUMP_WEST, 0, 3)
            elif char == '→':
                tile = (8, MetatileBehavior.JUMP_EAST, 0, 3)
            else:
                # default to walkable tile
                tile = (2, MetatileBehavior.NORMAL, 0, 3)
            row.append(tile)
        metatiles.append(row)

    return metatiles

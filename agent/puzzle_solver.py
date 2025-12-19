#!/usr/bin/env python3
"""
Gym Puzzle Solver Agent

Analyzes gym puzzle states and provides guidance on how to solve them.
Similar to the reflect agent, this agent takes the current gym and game state
and returns step-by-step instructions for solving the puzzle.
"""

from pathlib import Path
import os
from typing import Dict, Any, Optional
import base64


# Gym puzzle knowledge base
GYM_PUZZLES = {
    "RUSTBORO_CITY_GYM": {
        "type": "trainer_gauntlet",
        "description": "Stone Badge gym - no puzzle, just defeat trainers",
        "strategy": "Navigate through trainers to reach Roxanne at the back"
    },
    "DEWFORD_TOWN_GYM": {
        "type": "darkness_maze",
        "description": "Knuckle Badge gym - dark room with trainers",
        "strategy": "Use Flash to light up the room, or navigate in darkness to Brawly"
    },
    "MAUVILLE_CITY_GYM": {
        "type": "electric_barriers",
        "description": "Dynamo Badge gym - electric barrier maze",
        "strategy": "Defeat trainers to deactivate barriers blocking the path to Wattson"
    },
    "LAVARIDGE_TOWN_GYM_1F": {
        "type": "hidden_floor_puzzle",
        "description": "Heat Badge gym - holes in floor with hidden tiles",
        "strategy": """Lavaridge Gym floor puzzle solution:

The gym has holes in the floor. You must step on the correct tiles to activate hidden platforms.
Some tiles look like normal floor but are actually warp tiles that drop you down.

Key mechanics:
- Tiles with collision=1 are WALLS (show as #) - cannot walk through
- Tiles with DOOR/WARP behavior and collision=0 are holes - will drop you down
- Some tiles appear as floor but are actually warps

General strategy:
1. The gym has multiple trainers on platforms
2. You need to navigate the platforms carefully
3. Some 'D' tiles in walls are hidden warps that activate when you step on certain tiles
4. Trial and error: if you fall, you restart from the entrance
5. Work your way through the trainers to reach Flannery at the back

Common path patterns:
- Start at the entrance
- Navigate the platforms avoiding the holes
- Defeat trainers to potentially open new paths
- Reach the back where Flannery is located

Use the knowledge base to write notes about which geysers led where and any possible routes or deadends"""
    },
    "PETALBURG_CITY_GYM": {
        "type": "door_puzzle",
        "description": "Balance Badge gym - rooms with different trainer types",
        "strategy": "Choose doors based on trainer types: Speed, Accuracy, Defense, Strength, Recovery, One-Hit KO"
    },
    "FORTREE_CITY_GYM": {
        "type": "rotating_doors",
        "description": "Feather Badge gym - rotating door puzzle",
        "strategy": "Navigate through rotating doors that change direction when stepped on"
    },
    "MOSSDEEP_CITY_GYM": {
        "type": "tile_puzzle",
        "description": "Mind Badge gym - floor tile puzzle with teleports",
        "strategy": "Step on specific floor tiles to activate teleport pads to reach Tate and Liza"
    },
    "SOOTOPOLIS_CITY_GYM_1F": {
        "type": "ice_puzzle",
        "description": "Rain Badge gym - ice floor sliding puzzle",
        "strategy": "Slide on ice floors, use rocks to stop movement and reach Juan"
    }
}
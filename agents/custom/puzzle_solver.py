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
        "strategy": """Fortree Gym rotating door (turnstile) puzzle solution:

=== HOW THE ROTATING GATES WORK ===
Each gate looks like a + or X shape with 4 arms radiating from a center pivot.
- You can ONLY walk through the OPEN sides (gaps between arms)
- When you walk through, the ENTIRE gate rotates 90° CLOCKWISE
- The arms that were blocking LEFT/RIGHT will now block UP/DOWN (and vice versa)

VISUAL GUIDE - Gate States:
State A (blocks left/right):    State B (blocks up/down):
    |                               -+-
   -+-                               |
    |

Walking through State A (entering from top or bottom) rotates it to State B.
Walking through State B (entering from left or right) rotates it to State A.

=== ROTATION RULE ===
ALWAYS rotates CLOCKWISE when you pass through:
- Enter from SOUTH (walking UP) → The gate rotates so the arm that was pointing SOUTH now points WEST
- Enter from NORTH (walking DOWN) → The gate rotates so the arm that was pointing NORTH now points EAST
- Enter from WEST (walking RIGHT) → The gate rotates so the arm that was pointing WEST now points NORTH
- Enter from EAST (walking LEFT) → The gate rotates so the arm that was pointing EAST now points SOUTH

=== STRATEGY ===
Since gates reset when you leave and re-enter the gym, you must solve it in one go:
1. Plan ahead - look at which way each gate's arms are pointing
2. Sometimes you need to go THROUGH a gate, then AROUND and THROUGH again to rotate it twice
3. Think about the RESULT you need: which direction do you need the gap to be open?

Layout: Entrance at bottom (15, 24). Winona at top (15, 2).

General path:
1. Pass through first gates going UP
2. Loop around (UP, LEFT, DOWN) to approach same gate from side
3. Push through going RIGHT to reorient it
4. Work through the middle section defeating trainers
5. Eventually reach the upper platforms
6. Navigate to Winona at the back

IMPORTANT: When you see a gate blocking your path:
- Check which directions the arms are pointing
- If arms block UP/DOWN, you can pass LEFT/RIGHT
- If arms block LEFT/RIGHT, you can pass UP/DOWN
- After passing, the arms will rotate 90° clockwise

Trainer locations to defeat:
- Humberto: (4, 23)
- Ashley: (5, 17)
- Jared: (4, 14)
- Darius: (1, 10)
- Flint: (10, 10)
- Edwardo: (9, 8)
- Winona: (15, 2)"""
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
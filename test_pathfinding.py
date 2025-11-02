#!/usr/bin/env python3
"""
Test pathfinding with actual map data from Littleroot Town.
"""

import sys
import logging
from utils.pathfinding import Pathfinder

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)

# Map data from out.txt - Littleroot Town ASCII map (20x20)
littleroot_grid = [
    "##########..########",
    "##########..########",
    "##...............P##",
    "##................##",
    "....................",
    "..#####......#####..",
    "..#####......#####..",
    "..#####......#####..",
    "..######....######..",
    "....................",
    "....................",
    "....................",
    "....................",
    "...#######.....#....",
    "...#######..........",
    "...#######..........",
    "...#######........##",
    "......#...........##",
    "##..............####",
    "##..............####",
]

# Warps from the log
warps = [
    (14, 8),   # MAP_LITTLEROOT_TOWN_MAYS_HOUSE_1F
    (5, 8),    # MAP_LITTLEROOT_TOWN_BRENDANS_HOUSE_1F
    (7, 16),   # MAP_LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB
]

# NPCs/Objects from the log (some are stationary, some wander)
npcs = [
    (16, 10),  # OBJ_EVENT_GFX_TWIN (WANDER_AROUND)
    (12, 13),  # OBJ_EVENT_GFX_FAT_MAN (WANDER_AROUND)
    (14, 17),  # OBJ_EVENT_GFX_BOY_2 (WANDER_AROUND)
    (5, 8),    # OBJ_EVENT_GFX_MOM (FACE_UP - stationary)
    (2, 10),   # OBJ_EVENT_GFX_TRUCK (stationary)
    (11, 10),  # OBJ_EVENT_GFX_TRUCK (stationary)
    (13, 10),  # OBJ_EVENT_GFX_VAR_0 (FACE_UP - stationary)
    (14, 10),  # OBJ_EVENT_GFX_PROF_BIRCH (FACE_UP - stationary)
]

def create_test_game_state(grid, warps, npcs, player_pos):
    """Create a game_state dict matching what the server provides"""
    # Convert grid to list of lists
    grid_list = [list(row) for row in grid]
    
    # Replace 'P' with '.' (player position is walkable)
    for row in grid_list:
        for i, cell in enumerate(row):
            if cell == 'P':
                row[i] = '.'
    
    # Create objects list matching porymap format
    objects = []
    for npc_x, npc_y in npcs:
        # Determine movement type based on NPC type
        if npc_y == 8 and npc_x == 5:
            movement_type = "MOVEMENT_TYPE_FACE_UP"  # MOM
        elif npc_x in [2, 11] and npc_y == 10:
            movement_type = "MOVEMENT_TYPE_FACE_RIGHT"  # TRUCK
        elif npc_x in [13, 14] and npc_y == 10:
            movement_type = "MOVEMENT_TYPE_FACE_UP"
        else:
            movement_type = "MOVEMENT_TYPE_WANDER_AROUND"
        
        objects.append({
            "x": npc_x,
            "y": npc_y,
            "elevation": 3,
            "movement_type": movement_type
        })
    
    # Create warp list
    warp_list = []
    for warp_x, warp_y in warps:
        warp_list.append({
            "x": warp_x,
            "y": warp_y,
            "elevation": 0
        })
    
    return {
        "player": {
            "location": "LITTLEROOT TOWN",
            "position": {
                "x": player_pos[0],
                "y": player_pos[1]
            }
        },
        "map": {
            "porymap": {
                "grid": grid_list,
                "objects": objects,
                "warps": warp_list,
                "dimensions": {
                    "width": len(grid_list[0]) if grid_list else 20,
                    "height": len(grid_list) if grid_list else 20
                }
            }
        }
    }

def test_pathfinding():
    """Test pathfinding with various scenarios"""
    pathfinder = Pathfinder()
    
    print("=" * 80)
    print("TESTING PATHFINDING WITH LITTLEROOT TOWN MAP DATA")
    print("=" * 80)
    
    # Test 1: From (17, 2) to (7, 16) - Professor Birch's Lab
    print("\n" + "=" * 80)
    print("TEST 1: (17, 2) -> (7, 16) [Professor Birch's Lab]")
    print("=" * 80)
    start = (17, 2)
    goal = (7, 16)
    game_state = create_test_game_state(littleroot_grid, warps, npcs, start)
    
    buttons = pathfinder.find_path(start, goal, game_state)
    if buttons:
        print(f"✅ SUCCESS: Found path with {len(buttons)} buttons: {buttons[:10]}...")
        print(f"   Expected distance: {abs(start[0]-goal[0]) + abs(start[1]-goal[1])} (Manhattan)")
    else:
        print(f"❌ FAILED: No path found from {start} to {goal}")
    
    # Test 2: From (17, 2) to (10, 0) - North to Route 101
    print("\n" + "=" * 80)
    print("TEST 2: (17, 2) -> (10, 0) [North towards Route 101]")
    print("=" * 80)
    start = (17, 2)
    goal = (10, 0)
    game_state = create_test_game_state(littleroot_grid, warps, npcs, start)
    
    buttons = pathfinder.find_path(start, goal, game_state)
    if buttons:
        print(f"✅ SUCCESS: Found path with {len(buttons)} buttons: {buttons[:10]}...")
        print(f"   Expected distance: {abs(start[0]-goal[0]) + abs(start[1]-goal[1])} (Manhattan)")
    else:
        print(f"❌ FAILED: No path found from {start} to {goal}")
    
    # Test 3: From (10, 17) to (10, 4) - Straight vertical path (the problematic case)
    print("\n" + "=" * 80)
    print("TEST 3: (10, 17) -> (10, 4) [Straight vertical path - problematic case]")
    print("=" * 80)
    start = (10, 17)
    goal = (10, 4)
    game_state = create_test_game_state(littleroot_grid, warps, npcs, start)
    
    buttons = pathfinder.find_path(start, goal, game_state)
    if buttons:
        print(f"✅ SUCCESS: Found path with {len(buttons)} buttons")
        print(f"   Expected distance: {abs(start[0]-goal[0]) + abs(start[1]-goal[1])} (Manhattan)")
        print(f"   First 10 buttons: {buttons[:10]}")
        print(f"   Last 10 buttons: {buttons[-10:]}")
        
        # Check if it's optimal
        if len(buttons) <= abs(start[0]-goal[0]) + abs(start[1]-goal[1]) + 2:
            print(f"   ✅ Path is optimal (within tolerance)")
        else:
            print(f"   ⚠️ Path is suboptimal ({len(buttons)} steps vs {abs(start[0]-goal[0]) + abs(start[1]-goal[1])} expected)")
    else:
        print(f"❌ FAILED: No path found from {start} to {goal}")
    
    # Test 4: From (17, 2) to (14, 8) - May's House warp
    print("\n" + "=" * 80)
    print("TEST 4: (17, 2) -> (14, 8) [May's House warp]")
    print("=" * 80)
    start = (17, 2)
    goal = (14, 8)  # This is a warp, should be walkable
    game_state = create_test_game_state(littleroot_grid, warps, npcs, start)
    
    buttons = pathfinder.find_path(start, goal, game_state)
    if buttons:
        print(f"✅ SUCCESS: Found path with {len(buttons)} buttons: {buttons[:10]}...")
        print(f"   Expected distance: {abs(start[0]-goal[0]) + abs(start[1]-goal[1])} (Manhattan)")
    else:
        print(f"❌ FAILED: No path found from {start} to {goal}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_pathfinding()



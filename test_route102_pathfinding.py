#!/usr/bin/env python3
"""
Test pathfinding for Route 102 - debug why navigate_to(0, 10) fails from (37, 10)
"""

import sys
import logging
from utils.pathfinding import Pathfinder

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)

# Route 102 ASCII map from out.txt (50x20)
route102_grid = [
    "##################################################",
    "##################################################",
    "########.........#......##..########........######",
    "########....................########........######",
    "##................................#...........####",
    "##................................#...........####",
    "..............####..######....##..#...........####",
    "..............##..........######..#...........####",
    "............####..........#########.............##",
    "............####..........########......#.......##",
    "########....####..........##########.P............",
    "########..######..........##########..............",
    "######....######..........####..####............##",
    "######..########.....##.........####............##",
    "######......######...####...................######",
    "######......######.....##...................######",
    "######....########......................##########",
    "######....########......................##########",
    "########################........##################",
    "########################........##################",
]

def create_test_game_state(grid, player_pos):
    """Create a game_state dict matching what the server provides"""
    # Convert grid to list of lists
    grid_list = [list(row) for row in grid]
    
    # Replace 'P' with '.' (player position is walkable)
    for row in grid_list:
        for i, cell in enumerate(row):
            if cell == 'P':
                row[i] = '.'
    
    return {
        "player": {
            "location": "ROUTE 102",
            "position": {
                "x": player_pos[0],
                "y": player_pos[1]
            }
        },
        "map": {
            "porymap": {
                "grid": grid_list,
                "objects": [],  # No NPCs for this test
                "warps": [],  # No warps for this test
                "dimensions": {
                    "width": len(grid_list[0]) if grid_list else 50,
                    "height": len(grid_list) if grid_list else 20
                }
            }
        }
    }

def test_pathfinding():
    """Test pathfinding from (37, 10) to (0, 10)"""
    pathfinder = Pathfinder()
    
    print("=" * 80)
    print("TESTING PATHFINDING: Route 102 - (37, 10) -> (0, 10)")
    print("=" * 80)
    
    start = (37, 10)
    goal = (0, 10)
    
    print(f"\nStart: {start}")
    print(f"Goal: {goal}")
    print(f"Expected Manhattan distance: {abs(start[0]-goal[0]) + abs(start[1]-goal[1])} steps")
    
    # Check if goal is walkable in the grid
    if 0 <= goal[1] < len(route102_grid):
        row = route102_grid[goal[1]]
        if 0 <= goal[0] < len(row):
            goal_cell = row[goal[0]]
            print(f"Goal cell at ({goal[0]}, {goal[1]}): '{goal_cell}'")
            if goal_cell == '#':
                print("⚠️  WARNING: Goal is on a blocked tile!")
            elif goal_cell == '.' or goal_cell == 'P':
                print("✅ Goal is on a walkable tile")
    
    # Check if start is walkable
    if 0 <= start[1] < len(route102_grid):
        row = route102_grid[start[1]]
        if 0 <= start[0] < len(row):
            start_cell = row[start[0]]
            print(f"Start cell at ({start[0]}, {start[1]}): '{start_cell}'")
    
    # Check the direct horizontal path
    print(f"\nChecking direct horizontal path from {start[0]} to {goal[0]} on row {start[1]}:")
    blocked_on_path = []
    for x in range(min(start[0], goal[0]), max(start[0], goal[0]) + 1):
        if 0 <= start[1] < len(route102_grid):
            row = route102_grid[start[1]]
            if 0 <= x < len(row):
                cell = row[x]
                if cell == '#':
                    blocked_on_path.append((x, start[1]))
                    print(f"  Blocked at ({x}, {start[1]})")
    
    if blocked_on_path:
        print(f"⚠️  Direct path has {len(blocked_on_path)} blocked tiles")
    else:
        print("✅ Direct path is clear")
    
    print("\n" + "=" * 80)
    print("Running pathfinder...")
    print("=" * 80)
    
    game_state = create_test_game_state(route102_grid, start)
    
    buttons = pathfinder.find_path(start, goal, game_state)
    
    if buttons:
        print(f"\n✅ SUCCESS: Found path with {len(buttons)} buttons")
        print(f"   Expected distance: {abs(start[0]-goal[0]) + abs(start[1]-goal[1])} (Manhattan)")
        print(f"   First 10 buttons: {buttons[:10]}")
        if len(buttons) > 10:
            print(f"   Last 10 buttons: {buttons[-10:]}")
        
        # Check if it's optimal
        if len(buttons) <= abs(start[0]-goal[0]) + abs(start[1]-goal[1]) + 2:
            print(f"   ✅ Path is optimal (within tolerance)")
        else:
            print(f"   ⚠️ Path is suboptimal ({len(buttons)} steps vs {abs(start[0]-goal[0]) + abs(start[1]-goal[1])} expected)")
    else:
        print(f"\n❌ FAILED: No path found from {start} to {goal}")
        print("\nDebugging info:")
        print(f"  - Map dimensions: {len(route102_grid[0])}x{len(route102_grid)}")
        print(f"  - Start in bounds: {0 <= start[0] < len(route102_grid[0]) and 0 <= start[1] < len(route102_grid)}")
        print(f"  - Goal in bounds: {0 <= goal[0] < len(route102_grid[0]) and 0 <= goal[1] < len(route102_grid)}")
        
        # Check if goal is actually reachable
        print(f"\n  Checking if goal ({goal[0]}, {goal[1]}) is walkable:")
        if 0 <= goal[1] < len(route102_grid):
            row = route102_grid[goal[1]]
            if 0 <= goal[0] < len(row):
                goal_cell = row[goal[0]]
                print(f"    Cell value: '{goal_cell}'")
                if goal_cell == '#':
                    print(f"    ⚠️  Goal is on a BLOCKED tile - pathfinder will fail!")
                elif goal_cell == '.' or goal_cell == 'P':
                    print(f"    ✅ Goal is walkable")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_pathfinding()




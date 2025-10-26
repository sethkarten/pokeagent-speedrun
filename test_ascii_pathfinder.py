#!/usr/bin/env python3
"""
Test cases for ASCII pathfinder to debug the "no path found" issue.
"""

import sys
sys.path.insert(0, '.')

from utils.pathfinding_ascii import ASCIIPathfinder

def test_simple_map():
    """Test with a simple map."""
    print("=" * 60)
    print("Test 1: Simple horizontal movement")
    print("=" * 60)
    
    ascii_pathfinder = ASCIIPathfinder()
    
    # Simple map: player at (3, 2), goal at (4, 2)
    ascii_map = """# # # # # # # # # # # # # # #
# # # # # . . . S # # # # # #
# # # # # . . P S # # # # # #
# # # # # # # . S # # # # # #
# # # # # # # # # # # # # # #"""
    
    start = (3, 2)
    goal = (4, 2)
    
    print(f"Map:\n{ascii_map}")
    print(f"\nStart: {start}, Goal: {goal}")
    
    result = ascii_pathfinder.find_path_from_ascii_map(ascii_map, start, goal)
    print(f"Result: {result}")
    
    if result == ["RIGHT"]:
        print("✅ PASS: Got expected path [RIGHT]")
    else:
        print(f"❌ FAIL: Expected [RIGHT], got {result}")
    
    return result == ["RIGHT"]

def test_map_with_coordinates():
    """Test with map that includes coordinate labels."""
    print("\n" + "=" * 60)
    print("Test 2: Map with coordinate labels")
    print("=" * 60)
    
    ascii_pathfinder = ASCIIPathfinder()
    
    # Map with coordinate labels (like from game state)
    ascii_map = """      0  1  2  3  4  5  6  7  8
  0    # # # # # # # # #
  1    # # # # # . . S #
  2    # # # # # . P . S
  3    # # # # # # . . S
  4    # # # # # # # # #"""
    
    start = (3, 2)
    goal = (4, 2)
    
    print(f"Map:\n{ascii_map}")
    print(f"\nStart: {start}, Goal: {goal}")
    
    result = ascii_pathfinder.find_path_from_ascii_map(ascii_map, start, goal)
    print(f"Result: {result}")
    
    if result == ["RIGHT"]:
        print("✅ PASS: Got expected path [RIGHT]")
    else:
        print(f"❌ FAIL: Expected [RIGHT], got {result}")
    
    return result == ["RIGHT"]

def test_vertical_movement():
    """Test vertical movement."""
    print("\n" + "=" * 60)
    print("Test 3: Vertical movement")
    print("=" * 60)
    
    ascii_pathfinder = ASCIIPathfinder()
    
    ascii_map = """# # # # # # # # # #
# # # # . . . S # #
# # # # . P . S # #
# # # # . . . S # #
# # # # # # # # # #"""
    
    start = (2, 2)
    goal = (2, 3)
    
    print(f"Map:\n{ascii_map}")
    print(f"\nStart: {start}, Goal: {goal}")
    
    result = ascii_pathfinder.find_path_from_ascii_map(ascii_map, start, goal)
    print(f"Result: {result}")
    
    if result == ["DOWN"]:
        print("✅ PASS: Got expected path [DOWN]")
    else:
        print(f"❌ FAIL: Expected [DOWN], got {result}")
    
    return result == ["DOWN"]

def test_actual_game_map():
    """Test with the actual map from the game state."""
    print("\n" + "=" * 60)
    print("Test 4: Actual game map format")
    print("=" * 60)
    
    ascii_pathfinder = ASCIIPathfinder()
    
    # This is the actual format from the game
    ascii_map = """# # # # # # # # # # # # # # #
# # # # # . . . S # # # # # #
# # # # # . . P S # # # # # #
# # # # # # # . S # # # # # #
# # # # # # # # # # # # # # #"""
    
    # Player is at (3, 2), goal (stairs) is at (4, 2)
    start = (3, 2)
    goal = (4, 2)
    
    print(f"Map:\n{ascii_map}")
    print(f"\nStart: {start}, Goal: {goal}")
    print("\nParsing map...")
    
    # Debug: parse the map
    grid = ascii_pathfinder._parse_ascii_map(ascii_map)
    print(f"Parsed grid: {len(grid)} rows x {len(grid[0]) if grid else 0} cols")
    if grid:
        for i, row in enumerate(grid[:5]):
            print(f"  Row {i}: {row}")
    
    # Check if coordinates match
    print(f"\nGrid dimensions: {len(grid)} x {len(grid[0]) if grid else 0}")
    print(f"Checking if start {start} is in bounds: {0 <= start[1] < len(grid) and 0 <= start[0] < len(grid[0]) if grid else False}")
    print(f"Checking if goal {goal} is in bounds: {0 <= goal[1] < len(grid) and 0 <= goal[0] < len(grid[0]) if grid else False}")
    
    # Debug: show the full grid with coordinates
    print("\nFull grid with coordinates:")
    for y, row in enumerate(grid):
        print(f"y={y}: {' '.join(row)}")
    
    if grid and 0 <= start[1] < len(grid) and 0 <= start[0] < len(grid[0]):
        start_char = grid[start[1]][start[0]]
        print(f"Start position {start}: char='{start_char}' at grid[{start[1]}][{start[0]}]")
        print(f"Start is blocked: {ascii_pathfinder._is_blocked(grid, start[0], start[1])}")
    else:
        print(f"Start position {start} is OUT OF BOUNDS for grid {len(grid)}x{len(grid[0]) if grid else 0}")
    
    if grid and 0 <= goal[1] < len(grid) and 0 <= goal[0] < len(grid[0]):
        goal_char = grid[goal[1]][goal[0]]
        print(f"Goal position {goal}: char='{goal_char}' at grid[{goal[1]}][{goal[0]}]")
        print(f"Goal is blocked: {ascii_pathfinder._is_blocked(grid, goal[0], goal[1])}")
    else:
        print(f"Goal position {goal} is OUT OF BOUNDS for grid {len(grid)}x{len(grid[0]) if grid else 0}")
    
    # Try to find 'P' in the grid
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char == 'P':
                print(f"Found 'P' at grid position (x={x}, y={y})")
    
    result = ascii_pathfinder.find_path_from_ascii_map(ascii_map, start, goal)
    print(f"\nResult: {result}")
    
    if result == ["RIGHT"]:
        print("✅ PASS: Got expected path [RIGHT]")
        return True
    else:
        print(f"❌ FAIL: Expected [RIGHT], got {result}")
        return False

if __name__ == "__main__":
    print("Testing ASCII Pathfinder\n")
    
    results = []
    results.append(test_simple_map())
    results.append(test_map_with_coordinates())
    results.append(test_vertical_movement())
    results.append(test_actual_game_map())
    
    print("\n" + "=" * 60)
    print(f"Summary: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)


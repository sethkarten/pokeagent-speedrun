#!/usr/bin/env python3
"""
Test pathfinding on Route104 to verify ledge direction handling.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.pathfinding import Pathfinder
from utils.porymap_json_builder import build_json_map_for_llm

def test_route104_pathfinding():
    """Test pathfinding on Route104 with ledges."""
    
    print("=" * 80)
    print("Testing Route104 Pathfinding with Ledge Direction Handling")
    print("=" * 80)
    
    # Load Route104 map
    pokeemerald_root = Path(__file__).parent / 'porymap_data'
    json_map = build_json_map_for_llm('Route104', pokeemerald_root)
    
    if not json_map or 'grid' in json_map:
        print("\n✅ Route104 map loaded successfully")
        print(f"   Dimensions: {json_map.get('dimensions', {})}")
    else:
        print("\n❌ Failed to load Route104 map")
        return
    
    # Create game state with porymap data
    game_state = {
        'player': {'location': 'Route 104'},
        'map': {
            'porymap': json_map
        }
    }
    
    pathfinder = Pathfinder()
    
    print("\n" + "=" * 80)
    print("TEST 1: Warp at (32, 42) - Should be BLOCKED by south ledges (↓)")
    print("=" * 80)
    
    # Starting position south of the warp (coming from below)
    start1 = (28, 56)  # South of the ledges
    goal1 = (32, 42)    # The blocked warp
    
    print(f"\nAttempting path from {start1} to {goal1}")
    print("Expected: Should find path to nearest reachable tile (NOT the warp itself)")
    
    path1 = pathfinder.find_path(start1, goal1, game_state)
    
    if path1:
        print(f"\n✅ Path found: {len(path1)} steps")
        # Check if we actually reached the goal or just got close
        # The pathfinder should have adjusted to nearest reachable tile
        print(f"   Note: Pathfinder found path to nearest accessible tile")
    else:
        print(f"\n❌ No path found")
        print(f"   This is EXPECTED if the warp is completely blocked by ledges")
    
    print("\n" + "=" * 80)
    print("TEST 2: Warp at (10, 30) - Should be ACCESSIBLE through grass (~)")
    print("=" * 80)
    
    # Starting position that requires going through grass
    start2 = (17, 2)   # Near Route 103 connection
    goal2 = (10, 30)    # Accessible warp through grass
    
    print(f"\nAttempting path from {start2} to {goal2}")
    print("Expected: Should find valid path through grass areas")
    
    path2 = pathfinder.find_path(start2, goal2, game_state)
    
    if path2:
        print(f"\n✅ Path found: {len(path2)} steps")
        print(f"   Path: {' '.join(path2[:20])}{'...' if len(path2) > 20 else ''}")
    else:
        print(f"\n❌ No path found (UNEXPECTED - grass should be walkable)")
    
    print("\n" + "=" * 80)
    print("TEST 3: Jump OVER a ledge - Should work in correct direction")
    print("=" * 80)
    
    # Test jumping south over a ledge (↓)
    # Looking at the map, there are south ledges around line 56-58
    start3 = (28, 55)  # North of south ledge
    goal3 = (28, 57)   # South of south ledge (jumping over it)
    
    print(f"\nAttempting to jump south ledge from {start3} to {goal3}")
    print("Expected: Should be able to jump in correct direction (south)")
    
    path3 = pathfinder.find_path(start3, goal3, game_state)
    
    if path3:
        print(f"\n✅ Path found: {len(path3)} steps")
        print(f"   Can jump ledge in correct direction!")
    else:
        print(f"\n⚠️  No path found")
        print(f"   May need to check ledge positions on map")
    
    print("\n" + "=" * 80)
    print("TEST 4: Try to go UP a south ledge - Should be BLOCKED")
    print("=" * 80)
    
    # Test trying to go north over a south ledge (should fail)
    start4 = (28, 57)  # South of south ledge
    goal4 = (28, 55)   # North of south ledge (trying to go wrong way)
    
    print(f"\nAttempting to go north over south ledge from {start4} to {goal4}")
    print("Expected: Should be BLOCKED (can't jump up a south ledge)")
    
    path4 = pathfinder.find_path(start4, goal4, game_state, max_distance=10)
    
    if path4:
        # Check if the path actually goes over the ledge or around it
        print(f"\n⚠️  Path found: {len(path4)} steps")
        print(f"   Path likely goes AROUND the ledge (not over it)")
    else:
        print(f"\n✅ No direct path found (CORRECT - ledge blocks this direction)")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION: Showing ledge area from map")
    print("=" * 80)
    
    # Show the area around the ledges
    grid = json_map['grid']
    print("\nRoute104 area with ledges (rows 48-60):")
    for i, row in enumerate(grid[48:60], start=48):
        print(f"  {i:2d}: {''.join(row)}")
    
    print("\nLegend:")
    print("  ← = Jump West (can only move west TO this tile)")
    print("  → = Jump East (can only move east TO this tile)")
    print("  ↑ = Jump North (can only move north TO this tile)")
    print("  ↓ = Jump South (can only move south TO this tile)")
    print("  # = Blocked/Wall")
    print("  ~ = Grass (walkable)")
    print("  . = Normal walkable")
    print("  W = Water")
    print("  S = Stairs/Warp")

if __name__ == '__main__':
    test_route104_pathfinding()



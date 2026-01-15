#!/usr/bin/env python3
"""Test pathfinding through bridge underpass on Route 110"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.pathfinding import Pathfinder
from utils.porymap_json_builder import build_json_map
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Build the map
print("Loading Route110 map data...")
porymap_data = build_json_map('Route110', Path('porymap_data'))

if not porymap_data:
    print("ERROR: Could not load map data")
    sys.exit(1)

# Player position: outside Trick House at (11, 67)
player_pos = (11, 67)

# Apply elevation filtering manually (same logic as state_formatter.py)
print("\n" + "="*80)
print("APPLYING ELEVATION FILTERING")
print("="*80)

px, py = player_pos
raw_tiles = porymap_data['raw_tiles']
grid = porymap_data['grid']

# Get player's elevation
if 0 <= py < len(raw_tiles) and 0 <= px < len(raw_tiles[py]):
    player_tile = raw_tiles[py][px]
    if len(player_tile) >= 4:
        player_elevation = player_tile[3]
        print(f"Player elevation at ({px}, {py}): {player_elevation}")
    else:
        print("ERROR: Player tile doesn't have elevation data")
        sys.exit(1)
else:
    print("ERROR: Player position out of bounds")
    sys.exit(1)

elevation_tolerance = 0  # Must be exact same elevation

# Find all warp positions
warp_positions = set()
for y in range(len(grid)):
    for x in range(len(grid[y])):
        if grid[y][x] in ['S', 'D']:
            warp_positions.add((x, y))

# Filter the grid based on elevation
filtered_grid = []

for y in range(len(grid)):
    filtered_row = []
    for x in range(len(grid[y])):
        original_char = grid[y][x]

        # Always preserve special markers
        if original_char in ['D', 'S', 'T', 'K', 'N', 'I', '←', '→', '↑', '↓']:
            filtered_row.append(original_char)
        elif y < len(raw_tiles) and x < len(raw_tiles[y]):
            tile = raw_tiles[y][x]
            if len(tile) >= 4:
                tile_elevation = tile[3]
                elevation_diff = abs(tile_elevation - player_elevation)

                # Check if adjacent to stairs
                is_adjacent_to_stairs = False
                for warp_x, warp_y in warp_positions:
                    if abs(x - warp_x) + abs(y - warp_y) == 1:
                        is_adjacent_to_stairs = True
                        break

                # Adjacent to stairs: keep walkable tiles accessible
                if is_adjacent_to_stairs and original_char in ['.', '~', '←', '→', '↑', '↓']:
                    filtered_row.append(original_char)
                # Handle bridge tiles (&)
                elif original_char == '&':
                    # Check for . & & & . pattern
                    # Need to search through consecutive & tiles to find ground at both ends
                    has_ground_path = False
                    if 0 <= y < len(grid):
                        # Search left through consecutive bridge tiles to find ground
                        left_walkable = False
                        search_x = x - 1
                        while search_x >= 0 and search_x < len(grid[y]):
                            search_char = grid[y][search_x]
                            if search_char == '&':
                                search_x -= 1
                            elif search_char in ['.', '~']:
                                if search_x < len(raw_tiles[y]):
                                    search_tile = raw_tiles[y][search_x]
                                    if len(search_tile) >= 4:
                                        search_elev = search_tile[3]
                                        if abs(search_elev - player_elevation) <= elevation_tolerance:
                                            left_walkable = True
                                break
                            else:
                                break

                        # Search right through consecutive bridge tiles to find ground
                        right_walkable = False
                        search_x = x + 1
                        while search_x < len(grid[y]):
                            search_char = grid[y][search_x]
                            if search_char == '&':
                                search_x += 1
                            elif search_char in ['.', '~']:
                                if search_x < len(raw_tiles[y]):
                                    search_tile = raw_tiles[y][search_x]
                                    if len(search_tile) >= 4:
                                        search_elev = search_tile[3]
                                        if abs(search_elev - player_elevation) <= elevation_tolerance:
                                            right_walkable = True
                                break
                            else:
                                break

                        has_ground_path = left_walkable and right_walkable

                    # If ground path underneath, show as walkable
                    if has_ground_path and tile_elevation > player_elevation:
                        filtered_row.append('.')
                    else:
                        filtered_row.append('&')
                # Block tiles beyond elevation tolerance
                elif elevation_diff > elevation_tolerance:
                    filtered_row.append('#')
                else:
                    filtered_row.append(original_char)
            else:
                filtered_row.append(original_char)
        else:
            filtered_row.append(original_char)
    filtered_grid.append(filtered_row)

# Print section of map around row 6-10 (where bridge underpass is)
print("\nFiltered grid rows 6-10 (bridge underpass area):")
for y in range(6, 11):
    if y < len(filtered_grid):
        row = filtered_grid[y]
        print(f"Row {y:2d}: {''.join(row[10:30])}")

# Now test pathfinding through the underpass
print("\n" + "="*80)
print("PATHFINDING TEST: From (11, 67) to (16, 16) (north cycling road entrance)")
print("="*80)

# Create pathfinding game state with filtered porymap
filtered_porymap = porymap_data.copy()
filtered_porymap['grid'] = filtered_grid

pathfinding_state = {
    'player': {
        'position': player_pos,
        'location': 'Route110'
    },
    'map': {
        'porymap': filtered_porymap  # Use filtered grid
    }
}

pathfinder = Pathfinder()
path = pathfinder.find_path(
    start=player_pos,
    goal=(16, 16),  # North cycling road entrance
    game_state=pathfinding_state,
    max_distance=150,
    variance=None,
    consider_npcs=False,
    allow_partial=True
)

if path:
    print(f"\n✓ Path found: {len(path)} actions")
    print(f"Path (first 20): {path[:20]}")
else:
    print("\n✗ No path found")

# Check connectivity at key positions along the expected path
print("\n" + "="*80)
print("CHECKING KEY POSITIONS:")
print("="*80)

# Check if (16, 10) and surrounding bridge tiles are walkable
for y in [6, 7, 8, 9, 10]:
    if y < len(filtered_grid):
        for x in [14, 15, 16, 17, 18]:
            if x < len(filtered_grid[y]):
                symbol = filtered_grid[y][x]
                print(f"  ({x}, {y}): '{symbol}'")

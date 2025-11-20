#!/usr/bin/env python3
"""
Porymap JSON Map Builder

Builds a complete JSON map structure from porymap data (map.json + map.bin)
that can be passed to the LLM for navigation.

The JSON structure includes:
- Full grid with walkability (from map.bin)
- Warp points (from map.json)
- Object events/NPCs (from map.json)
- Connections to adjacent maps (from map.json)
- Dimensions and metadata
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.map_formatter import format_tile_to_symbol
from utils.pokeemerald_parser import PokeemeraldMapLoader, PokeemeraldLayoutParser
from pokemon_env.enums import MetatileBehavior


def tile_to_symbol(tile_tuple: Tuple[int, Any, int, int], location_name: str = "") -> str:
    """Proxy to shared formatter so behaviour-based symbols stay in sync."""
    if not tile_tuple:
        return '?'
    return format_tile_to_symbol(tile_tuple, location_name=location_name)


def build_json_map(map_name: str, pokeemerald_root: Path, 
                   include_grid: bool = True,
                   include_ascii: bool = True) -> Optional[Dict[str, Any]]:
    """
    Build a complete JSON map structure from porymap data.
    
    Args:
        map_name: Map name (e.g., "OldaleTown", "LittlerootTown")
        pokeemerald_root: Root directory of pokeemerald project
        include_grid: Include full grid data as JSON (True) or just metadata (False)
        include_ascii: Include ASCII representation for visualization
        
    Returns:
        Dictionary with complete map data in JSON-serializable format
        {
            "name": "OldaleTown",
            "id": "MAP_OLDALE_TOWN",
            "dimensions": {"width": 20, "height": 20},
            "grid": [[".", "#", ...], ...],  # Optional: full grid
            "ascii": "...",  # Optional: ASCII visualization
            "warps": [...],
            "objects": [...],
            "connections": [...],
            "metadata": {...}
        }
    """
    pokeemerald_root = Path(pokeemerald_root)
    
    # Load map data
    map_loader = PokeemeraldMapLoader(pokeemerald_root)
    map_data = map_loader.load_map(map_name)
    
    if not map_data:
        return None
    
    # Get layout parser
    layout_parser = PokeemeraldLayoutParser(pokeemerald_root)
    
    # Get layout name
    layout_name = map_loader.get_layout_name_from_map(map_name)
    
    # Parse tile data if layout available
    grid = None
    ascii_map = None
    dimensions = {"width": 0, "height": 0}
    
    if layout_name:
        # Get metatiles with behavior
        metatiles = layout_parser.get_metatiles_with_behavior(layout_name)
        
        if metatiles:
            height = len(metatiles)
            width = len(metatiles[0]) if height > 0 else 0
            dimensions = {"width": width, "height": height}
            
            # Build grid
            if include_grid:
                grid = []
                for row in metatiles:
                    grid_row = []
                    for tile in row:
                        symbol = tile_to_symbol(tile, map_name)
                        grid_row.append(symbol)
                    grid.append(grid_row)
            
            # Build ASCII representation
            if include_ascii:
                ascii_lines = []
                for row in metatiles:
                    line = "".join(tile_to_symbol(tile, map_name) for tile in row)
                    ascii_lines.append(line)

                # Overlay special object markers (TV, Clock, etc.)
                # This must be done BEFORE joining ascii_lines into ascii_map
                # Convert lines to list of lists for easier modification
                ascii_grid = [list(line) for line in ascii_lines]

                # Define special object graphics IDs and their symbols
                special_object_markers = {
                    # TVs - common graphics_id for TV objects
                    "OBJ_EVENT_GFX_ITEM_BALL": "I",  # Item balls
                    "OBJ_EVENT_GFX_CUTTABLE_TREE": "T",  # Tree
                    # Add more as needed
                }

                # Mark background events (clocks, PCs, etc.) on the ASCII map
                for bg_event in map_data.get("bg_events", []):
                    bg_x = bg_event.get("x", 0)
                    bg_y = bg_event.get("y", 0)
                    script = bg_event.get("script", "")

                    # Check if position is valid
                    if 0 <= bg_y < height and 0 <= bg_x < width:
                        # Clock events
                        if "WallClock" in script or "Clock" in script:
                            ascii_grid[bg_y][bg_x] = 'K'  # Clock marker
                        # PC events
                        elif "PC" in script and "TurnOnPC" in script:
                            ascii_grid[bg_y][bg_x] = 'P'  # PC marker
                        # TV/GameCube events (optional - might already be visible)
                        elif "TV" in script or "GameCube" in script:
                            ascii_grid[bg_y][bg_x] = 'V'  # TV marker

                # Convert back to strings
                ascii_lines = [''.join(row) for row in ascii_grid]
                ascii_map = "\n".join(ascii_lines)
    
    # Extract warps
    warps = []
    warp_positions = set()  # Track warp positions for grid overlay
    needs_south_extension = False  # Track if we need to extend the grid south
    needs_east_extension = False  # Track if we need to extend the grid east
    needs_west_extension = False  # Track if we need to extend the grid west
    max_warp_y = 0  # Track the maximum warp Y position
    max_warp_x = 0  # Track the maximum warp X position
    min_warp_x = float('inf')  # Track the minimum warp X position

    for warp in map_data.get("warp_events", []):
        original_warp_x = warp.get("x", 0)
        original_warp_y = warp.get("y", 0)
        warp_x = original_warp_x
        warp_y = original_warp_y

        # Adjust warps that are AT map edges (detected by blocked tiles beyond them)
        # Map edge warps need offset since players transition from adjacent maps
        if layout_name and metatiles and include_grid and grid:
            height = len(grid)
            width = len(grid[0]) if height > 0 else 0

            # Check if south of warp is blocked (edge warp going south)
            if warp_y + 1 >= height or grid[warp_y + 1][warp_x] == '#':
                # Check if entire row south is blocked
                is_south_edge = True
                if warp_y + 1 < height:
                    # Check surrounding tiles to confirm it's a map edge
                    for check_x in range(max(0, warp_x - 2), min(width, warp_x + 3)):
                        if warp_y + 1 < height and grid[warp_y + 1][check_x] != '#':
                            is_south_edge = False
                            break

                if is_south_edge:
                    warp_y += 1
                    if warp_y >= height:
                        needs_south_extension = True

            # Check if east of warp is blocked (edge warp going east)
            elif warp_x + 1 >= width or grid[warp_y][warp_x + 1] == '#':
                # Check if entire column east is blocked
                is_east_edge = True
                if warp_x + 1 < width:
                    for check_y in range(max(0, warp_y - 2), min(height, warp_y + 3)):
                        if warp_x + 1 < width and grid[check_y][warp_x + 1] != '#':
                            is_east_edge = False
                            break

                if is_east_edge:
                    warp_x += 1
                    if warp_x >= width:
                        needs_east_extension = True

            # Check if west of warp is blocked (edge warp going west)
            elif warp_x - 1 < 0 or grid[warp_y][warp_x - 1] == '#':
                # Check if entire column west is blocked
                is_west_edge = True
                if warp_x - 1 >= 0:
                    for check_y in range(max(0, warp_y - 2), min(height, warp_y + 3)):
                        if warp_x - 1 >= 0 and grid[check_y][warp_x - 1] != '#':
                            is_west_edge = False
                            break

                if is_west_edge:
                    warp_x -= 1
                    if warp_x < 0:
                        needs_west_extension = True

            # Clear the original door marker if we adjusted the position
            if (warp_x != original_warp_x or warp_y != original_warp_y) and include_ascii and ascii_map:
                ascii_lines = ascii_map.split('\n')
                if 0 <= original_warp_y < len(ascii_lines) and 0 <= original_warp_x < len(ascii_lines[0]):
                    ascii_grid = [list(line) for line in ascii_lines]
                    if ascii_grid[original_warp_y][original_warp_x] in ['D', 'S']:
                        ascii_grid[original_warp_y][original_warp_x] = '.'
                    ascii_map = '\n'.join([''.join(row) for row in ascii_grid])

        max_warp_y = max(max_warp_y, warp_y)
        max_warp_x = max(max_warp_x, warp_x)
        min_warp_x = min(min_warp_x, warp_x)

        warps.append({
            "x": warp_x,
            "y": warp_y,
            "elevation": warp.get("elevation", 0),
            "dest_map": warp.get("dest_map", "?"),
            "dest_warp_id": warp.get("dest_warp_id", 0)
        })

        # Track warp position for grid overlay
        warp_positions.add((warp_x, warp_y))

    # Extend grid and ASCII if warps go out of bounds
    if layout_name and metatiles:
        # South extension
        if needs_south_extension:
            extra_rows_needed = max_warp_y - len(metatiles) + 1

            if include_grid and grid:
                for _ in range(extra_rows_needed):
                    grid.append(['#'] * len(grid[0]))

            if include_ascii and ascii_map:
                ascii_lines = ascii_map.split('\n')
                for _ in range(extra_rows_needed):
                    ascii_lines.append('#' * len(ascii_lines[0]))
                ascii_map = '\n'.join(ascii_lines)

            height = len(metatiles) + extra_rows_needed
            dimensions["height"] = height

        # East extension
        if needs_east_extension:
            extra_cols_needed = max_warp_x - len(metatiles[0]) + 1

            if include_grid and grid:
                for row in grid:
                    row.extend(['#'] * extra_cols_needed)

            if include_ascii and ascii_map:
                ascii_lines = ascii_map.split('\n')
                ascii_lines = [line + '#' * extra_cols_needed for line in ascii_lines]
                ascii_map = '\n'.join(ascii_lines)

            width = len(metatiles[0]) + extra_cols_needed
            dimensions["width"] = width

        # West extension (prepend columns)
        if needs_west_extension:
            extra_cols_needed = abs(min_warp_x)

            if include_grid and grid:
                for row in grid:
                    # Prepend blocked tiles to the left
                    for _ in range(extra_cols_needed):
                        row.insert(0, '#')

            if include_ascii and ascii_map:
                ascii_lines = ascii_map.split('\n')
                ascii_lines = ['#' * extra_cols_needed + line for line in ascii_lines]
                ascii_map = '\n'.join(ascii_lines)

            # Adjust all warp X positions to account for the prepended columns
            for w in warps:
                w["x"] += extra_cols_needed

            # Also update warp_positions set with adjusted coordinates
            warp_positions = {(x + extra_cols_needed, y) for x, y in warp_positions}

            width = len(metatiles[0]) + extra_cols_needed
            dimensions["width"] = width

    # CRITICAL: Overlay warp symbols on grid and ASCII after extracting warp positions
    # This ensures warps are always marked as walkable, even if the underlying tile
    # is NORMAL with collision (like Petalburg Gym doors)
    if layout_name and metatiles:
        # Update grid (JSON format)
        if include_grid and grid:
            for warp_x, warp_y in warp_positions:
                if 0 <= warp_y < len(grid) and 0 <= warp_x < len(grid[0]):
                    # Get current symbol
                    current_symbol = grid[warp_y][warp_x]
                    # Only override if it's not already a door/stairs
                    # Use 'D' for exit doors (to Petalburg City), 'S' for internal warps
                    warp_dest = None
                    for warp in warps:
                        if warp['x'] == warp_x and warp['y'] == warp_y:
                            warp_dest = warp.get('dest_map', '')
                            break

                    if current_symbol not in ['D', 'S']:
                        # External doors (exits) use 'D', internal warps use 'S'
                        if warp_dest and 'PETALBURG_CITY_GYM' not in warp_dest:
                            grid[warp_y][warp_x] = 'D'
                        else:
                            grid[warp_y][warp_x] = 'S'

        # Update ASCII representation
        if include_ascii and ascii_map:
            ascii_grid = [list(line) for line in ascii_map.split('\n')]
            for warp_x, warp_y in warp_positions:
                if 0 <= warp_y < len(ascii_grid) and 0 <= warp_x < len(ascii_grid[0]):
                    current_symbol = ascii_grid[warp_y][warp_x]

                    # Get warp destination
                    warp_dest = None
                    for warp in warps:
                        if warp['x'] == warp_x and warp['y'] == warp_y:
                            warp_dest = warp.get('dest_map', '')
                            break

                    if current_symbol not in ['D', 'S']:
                        # External doors use 'D', internal warps use 'S'
                        if warp_dest and 'PETALBURG_CITY_GYM' not in warp_dest:
                            ascii_grid[warp_y][warp_x] = 'D'
                        else:
                            ascii_grid[warp_y][warp_x] = 'S'

            # Rebuild ASCII map from grid
            ascii_map = '\n'.join(''.join(row) for row in ascii_grid)
    
    # Extract objects
    objects = []
    for obj in map_data.get("object_events", []):
        objects.append({
            "x": obj.get("x", 0),
            "y": obj.get("y", 0),
            "elevation": obj.get("elevation", 0),
            "graphics_id": obj.get("graphics_id", "?"),
            "movement_type": obj.get("movement_type", "?"),
            "movement_range_x": obj.get("movement_range_x", 0),
            "movement_range_y": obj.get("movement_range_y", 0),
            "trainer_type": obj.get("trainer_type", "?"),
            "trainer_sight_or_berry_tree_id": obj.get("trainer_sight_or_berry_tree_id", "?")
        })
    
    # Extract connections (handle null case)
    connections = []
    connections_data = map_data.get("connections")
    if connections_data is not None:  # Check for None explicitly (not just falsy)
        for conn in connections_data:
            connections.append({
                "direction": conn.get("direction", "?"),
                "offset": conn.get("offset", 0),
                "map": conn.get("map", "?")
            })
    
    # Build complete JSON structure
    json_map = {
        "name": map_data.get("name", map_name),
        "id": map_data.get("id", f"MAP_{map_name.upper()}"),
        "dimensions": dimensions,
        "warps": warps,
        "objects": objects,
        "connections": connections,
        "metadata": {
            "music": map_data.get("music", "?"),
            "map_type": map_data.get("map_type", "?"),
            "weather": map_data.get("weather", "?"),
            "show_map_name": map_data.get("show_map_name", False),
            "floor_number": map_data.get("floor_number", 0),
            "battle_scene": map_data.get("battle_scene", "?")
        }
    }
    
    # Add grid and ASCII if requested
    if include_grid and grid:
        json_map["grid"] = grid
    
    if include_ascii and ascii_map:
        json_map["ascii"] = ascii_map
    
    return json_map


def build_json_map_for_llm(map_name: str, pokeemerald_root: Path) -> Optional[Dict[str, Any]]:
    """
    Build a JSON map optimized for LLM consumption.
    
    This version includes:
    - Full grid as nested arrays (for coordinate-based queries)
    - ASCII visualization (for human readability)
    - All important navigation features (warps, objects, connections)
    
    Args:
        map_name: Map name (e.g., "OldaleTown")
        pokeemerald_root: Root directory of pokeemerald project
        
    Returns:
        JSON-serializable dictionary optimized for LLM
    """
    return build_json_map(
        map_name=map_name,
        pokeemerald_root=pokeemerald_root,
        include_grid=True,
        include_ascii=True
    )


def save_json_map(map_name: str, pokeemerald_root: Path, output_path: Path):
    """
    Build and save a JSON map to file.
    
    Args:
        map_name: Map name to build
        pokeemerald_root: Root directory of pokeemerald project
        output_path: Where to save the JSON file
    """
    json_map = build_json_map_for_llm(map_name, pokeemerald_root)
    
    if not json_map:
        raise ValueError(f"Could not build map for: {map_name}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(json_map, f, indent=2)
    
    print(f"✅ Saved JSON map to: {output_path}")
    print(f"   Map: {map_name}")
    print(f"   Dimensions: {json_map['dimensions']['width']}x{json_map['dimensions']['height']}")
    print(f"   Warps: {len(json_map['warps'])}")
    print(f"   Objects: {len(json_map['objects'])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build JSON map from porymap data")
    parser.add_argument("--map", type=str, required=True, help="Map name (e.g., OldaleTown)")
    parser.add_argument("--root", type=str, default="../pokeemerald", help="Pokeemerald root directory")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--print", action="store_true", help="Print JSON to stdout")
    
    args = parser.parse_args()
    
    pokeemerald_root = Path(args.root).resolve()
    
    json_map = build_json_map_for_llm(args.map, pokeemerald_root)
    
    if not json_map:
        print(f"Error: Could not build map for {args.map}")
        exit(1)
    
    if args.print:
        print(json.dumps(json_map, indent=2))
    elif args.output:
        save_json_map(args.map, pokeemerald_root, Path(args.output))
    else:
        # Default: save to maps_json/{map_name}.json
        output_dir = Path("maps_json")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{args.map}.json"
        save_json_map(args.map, pokeemerald_root, output_path)


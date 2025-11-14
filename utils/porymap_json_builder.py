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

                # For specific locations, add hardcoded object markers
                # This ensures important interactive objects are visible on the map
                if "BrendansHouse_2F" in map_name or "MaysHouse_2F" in map_name:
                    # Clock is at (5, 1) - to the right of TV - coordinates x=5, y=1
                    # In ASCII grid: ascii_grid[y][x] = ascii_grid[1][5]
                    if height > 1 and width > 5:
                        ascii_grid[1][5] = 'K'  # Clock marker at (5,1)

                    # Note: TV at (4, 1) should already be rendered from the base map data
                    # We don't need to override it here

                # Convert back to strings
                ascii_lines = [''.join(row) for row in ascii_grid]
                ascii_map = "\n".join(ascii_lines)
    
    # Extract warps
    warps = []
    for warp in map_data.get("warp_events", []):
        warp_x = warp.get("x", 0)
        warp_y = warp.get("y", 0)

        # Check if this warp is on a door tile in the south of the map
        # If metatiles exist, check if the warp is on a 'D' (door) tile
        if layout_name and metatiles:
            # Check if warp position is valid
            if 0 <= warp_y < len(metatiles) and 0 <= warp_x < len(metatiles[0]):
                tile = metatiles[warp_y][warp_x]
                symbol = tile_to_symbol(tile, map_name)

                # If it's a door tile (D) and in the southern half of the map
                if (symbol == 'D' or symbol == 'S') and warp_y >= len(metatiles) // 2:
                    # Move the warp 1 block down
                    warp_y += 1

        warps.append({
            "x": warp_x,
            "y": warp_y,
            "elevation": warp.get("elevation", 0),
            "dest_map": warp.get("dest_map", "?"),
            "dest_warp_id": warp.get("dest_warp_id", 0)
        })
    
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


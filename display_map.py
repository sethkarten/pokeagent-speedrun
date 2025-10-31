#!/usr/bin/env python3
"""
Display Pokeemerald Map from JSON/Binary Data

Usage:
    python display_map.py OldaleTown
    python display_map.py LittlerootTown --output oldale_map.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path so we can import from pokeagent-speedrun
sys.path.insert(0, str(Path(__file__).parent))

from utils.pokeemerald_parser import PokeemeraldLayoutParser, PokeemeraldMapLoader


def format_ascii_map(metatiles, title="Map"):
    """
    Format metatiles into a simple ASCII map display.
    
    Args:
        metatiles: 2D list of tuples (metatile_id, behavior, collision, elevation)
        title: Title for the map
    
    Returns:
        str: Formatted ASCII map
    """
    if not metatiles or len(metatiles) == 0:
        return f"{title}: No map data"
    
    lines = [f"{title} ({len(metatiles)}x{len(metatiles[0])}):", ""]
    
    # Create simple ASCII grid
    # Collision == 0 = walkable (.), collision > 0 = blocked (#)
    # metatile_id == 1023 = invalid/out of bounds (#)
    for row in metatiles:
        row_str = ""
        for tile in row:
            if len(tile) >= 3:
                metatile_id, behavior, collision = tile[:3]
            elif len(tile) >= 1:
                metatile_id = tile[0]
                collision = 0
            else:
                metatile_id = 0
                collision = 0
            
            if metatile_id == 1023:  # 0x3FF - invalid/out of bounds
                row_str += "#"
            elif collision == 0:
                row_str += "."
            else:
                row_str += "#"
        lines.append(row_str)
    
    return "\n".join(lines)


def get_map_data(map_name: str, pokeemerald_root: Path) -> tuple:
    """
    Load map data for a given location.
    
    Returns:
        Tuple of (map_json_data, metatiles_with_behavior, layout_info) or None if not found
    """
    map_loader = PokeemeraldMapLoader(pokeemerald_root)
    layout_parser = PokeemeraldLayoutParser(pokeemerald_root)
    
    # Load map JSON
    map_data = map_loader.load_map(map_name)
    if not map_data:
        print(f"Error: Map '{map_name}' not found in {pokeemerald_root / 'data' / 'maps'}")
        return None
    
    # Get layout name from map
    layout_id = map_data.get("layout")
    if not layout_id:
        print(f"Error: Map '{map_name}' has no layout specified")
        return None
    
    # Convert layout ID to name for parser
    layout_name = layout_id.replace("LAYOUT_", "").lower().replace("_", " ").title().replace(" ", "")
    
    # Parse layout binary data
    metatiles = layout_parser.get_metatiles_with_behavior(layout_name)
    if metatiles is None:
        print(f"Error: Could not parse layout '{layout_name}' for map '{map_name}'")
        return None
    
    layout_info = layout_parser.get_layout_info(layout_name)
    
    return (map_data, metatiles, layout_info)


def display_map(map_name: str, pokeemerald_root: Path, output_file: Optional[Path] = None):
    """
    Display an ASCII map for the given location.
    
    Args:
        map_name: Name of the map (e.g., "OldaleTown", "LittlerootTown")
        pokeemerald_root: Root directory of pokeemerald project
        output_file: Optional file path to save output
    """
    pokeemerald_root = Path(pokeemerald_root)
    
    # Load map data
    result = get_map_data(map_name, pokeemerald_root)
    if result is None:
        return
    
    map_data, metatiles, layout_info = result
    
    # Format map for display
    location_name = map_data.get("name", map_name)
    title = f"{location_name} Map ({layout_info['width']}x{layout_info['height']} metatiles)"
    
    # Format ASCII map
    formatted_map = format_ascii_map(metatiles, title=title)
    
    # Add map metadata
    output_lines = [formatted_map, ""]
    output_lines.append("=" * 80)
    output_lines.append("Map Metadata:")
    output_lines.append(f"  Map ID: {map_data.get('id', 'N/A')}")
    output_lines.append(f"  Layout: {map_data.get('layout', 'N/A')}")
    output_lines.append(f"  Music: {map_data.get('music', 'N/A')}")
    output_lines.append(f"  Map Type: {map_data.get('map_type', 'N/A')}")
    output_lines.append(f"  Dimensions: {layout_info['width']}x{layout_info['height']} metatiles")
    
    # Connections
    connections = map_data.get("connections", [])
    if connections:
        output_lines.append(f"  Connections: {len(connections)}")
        for conn in connections:
            output_lines.append(f"    - {conn.get('direction', '?')} to {conn.get('map', '?')}")
    
    # Warps
    warps = map_data.get("warp_events", [])
    if warps:
        output_lines.append(f"  Warps: {len(warps)}")
        for warp in warps[:5]:  # Show first 5
            dest_map = warp.get("dest_map", "?")
            output_lines.append(f"    - ({warp.get('x', '?')}, {warp.get('y', '?')}) -> {dest_map}")
        if len(warps) > 5:
            output_lines.append(f"    ... and {len(warps) - 5} more")
    
    # Object events (NPCs)
    objects = map_data.get("object_events", [])
    if objects:
        output_lines.append(f"  Object Events: {len(objects)}")
        for obj in objects[:5]:  # Show first 5
            graphics_id = obj.get("graphics_id", "?")
            output_lines.append(f"    - {graphics_id} at ({obj.get('x', '?')}, {obj.get('y', '?')})")
        if len(objects) > 5:
            output_lines.append(f"    ... and {len(objects) - 5} more")
    
    output_text = "\n".join(output_lines)
    
    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
        print(f"Map saved to {output_file}")
    else:
        print(output_text)


def main():
    parser = argparse.ArgumentParser(
        description="Display Pokeemerald map from JSON/binary data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python display_map.py OldaleTown
  python display_map.py LittlerootTown --output oldale_map.txt
  python display_map.py Route101 --root ../pokeemerald
        """
    )
    parser.add_argument(
        "map_name",
        type=str,
        help="Name of the map to display (e.g., 'OldaleTown', 'LittlerootTown', 'Route101')"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="../pokeemerald",
        help="Root directory of pokeemerald project (default: ../pokeemerald)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    pokeemerald_root = Path(args.root).resolve()
    if not pokeemerald_root.exists():
        print(f"Error: Pokeemerald root directory not found: {pokeemerald_root}")
        print(f"Please specify the correct path with --root")
        sys.exit(1)
    
    if not (pokeemerald_root / "data" / "maps").exists():
        print(f"Error: Invalid pokeemerald root directory: {pokeemerald_root}")
        print(f"Expected to find data/maps/ directory")
        sys.exit(1)
    
    output_file = Path(args.output) if args.output else None
    
    display_map(args.map_name, pokeemerald_root, output_file)


if __name__ == "__main__":
    main()


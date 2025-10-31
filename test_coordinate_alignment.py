#!/usr/bin/env python3
"""
Test Coordinate Alignment Between JSON and Memory Reader

This script helps determine if pokeemerald JSON coordinates match
memory reader coordinates, and what transformation is needed.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.pokeemerald_parser import PokeemeraldMapLoader


def analyze_coordinate_systems(map_name: str, pokeemerald_root: Path):
    """
    Analyze coordinate systems used in JSON vs what memory reader might return.
    
    Args:
        map_name: Name of map to analyze (e.g., "OldaleTown")
        pokeemerald_root: Root directory of pokeemerald project
    """
    print(f"Analyzing coordinate system for: {map_name}\n")
    print("=" * 80)
    
    map_loader = PokeemeraldMapLoader(pokeemerald_root)
    map_data = map_loader.load_map(map_name)
    
    if not map_data:
        print(f"Error: Map '{map_name}' not found")
        return
    
    layout_id = map_data.get("layout")
    layout_name = layout_id.replace("LAYOUT_", "").lower().replace("_", " ").title().replace(" ", "")
    
    print(f"Map: {map_data.get('name', map_name)}")
    print(f"Layout: {layout_name}")
    print(f"Map ID: {map_data.get('id')}")
    print()
    
    # Analyze warp events
    print("=" * 80)
    print("WARP EVENTS (JSON Coordinates):")
    print("=" * 80)
    warps = map_data.get("warp_events", [])
    if warps:
        print(f"\n{'X':>4} {'Y':>4} {'Elev':>5} {'Destination':40}")
        print("-" * 80)
        for warp in warps:
            x = warp.get("x", 0)
            y = warp.get("y", 0)
            elev = warp.get("elevation", 0)
            dest = warp.get("dest_map", "?")
            print(f"{x:4} {y:4} {elev:5} {dest}")
        
        print(f"\nTotal warps: {len(warps)}")
        print("\nNOTE: These are direct map coordinates from JSON.")
        print("      Compare with memory reader output when standing at these positions.")
    else:
        print("No warp events found")
    
    # Analyze object events (NPCs)
    print("\n" + "=" * 80)
    print("OBJECT EVENTS (JSON Coordinates):")
    print("=" * 80)
    objects = map_data.get("object_events", [])
    if objects:
        print(f"\n{'X':>4} {'Y':>4} {'Elev':>5} {'Graphics ID':30}")
        print("-" * 80)
        for obj in objects:
            x = obj.get("x", 0)
            y = obj.get("y", 0)
            elev = obj.get("elevation", 0)
            gfx = obj.get("graphics_id", "?")
            print(f"{x:4} {y:4} {elev:5} {gfx}")
        
        print(f"\nTotal objects: {len(objects)}")
    else:
        print("No object events found")
    
    # Analyze connections
    print("\n" + "=" * 80)
    print("MAP CONNECTIONS:")
    print("=" * 80)
    connections = map_data.get("connections", [])
    if connections:
        for conn in connections:
            direction = conn.get("direction", "?")
            target_map = conn.get("map", "?")
            offset = conn.get("offset", 0)
            print(f"  {direction:6} -> {target_map} (offset: {offset})")
    else:
        print("No connections (indoor map)")
    
    # Coordinate system analysis
    print("\n" + "=" * 80)
    print("COORDINATE SYSTEM ANALYSIS:")
    print("=" * 80)
    print("\nJSON Coordinate System:")
    print("  - Origin: Top-left corner (0, 0)")
    print("  - X-axis: Increases rightward (0 → width-1)")
    print("  - Y-axis: Increases downward (0 → height-1)")
    print("  - Units: Metatiles (16x16 pixels)")
    print("  - Type: Direct map coordinates")
    
    print("\nMemory Reader Coordinate System (from code analysis):")
    print("  - read_coordinates() returns (x, y)")
    print("  - Uses MAP_OFFSET=7 when accessing map buffer")
    print("  - Likely returns 'world' or 'map space' coordinates")
    print("  - UNKNOWN: Does it match JSON coordinates?")
    
    print("\n" + "=" * 80)
    print("TESTING RECOMMENDATIONS:")
    print("=" * 80)
    print("\nTo determine coordinate alignment:")
    print("1. Load save state at a known location")
    print("2. Stand at a warp point from JSON (e.g., 5, 7 for OldaleTown)")
    print("3. Call memory_reader.read_coordinates()")
    print("4. Compare returned value with JSON coordinate (5, 7)")
    print("5. If they match: no transformation needed")
    print("6. If they differ: determine transformation")
    
    if warps:
        test_warp = warps[0]
        test_x = test_warp.get("x")
        test_y = test_warp.get("y")
        print(f"\nSuggested test:")
        print(f"  Stand at warp: ({test_x}, {test_y})")
        print(f"  Read coordinates: memory_reader.read_coordinates()")
        print(f"  Compare: returned == ({test_x}, {test_y})?")
    
    print("\n" + "=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze coordinate systems in pokeemerald maps"
    )
    parser.add_argument(
        "map_name",
        type=str,
        help="Map name to analyze (e.g., 'OldaleTown')"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="../pokeemerald",
        help="Pokeemerald root directory (default: ../pokeemerald)"
    )
    
    args = parser.parse_args()
    
    pokeemerald_root = Path(args.root).resolve()
    if not pokeemerald_root.exists():
        print(f"Error: Directory not found: {pokeemerald_root}")
        sys.exit(1)
    
    analyze_coordinate_systems(args.map_name, pokeemerald_root)


if __name__ == "__main__":
    main()


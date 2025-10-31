#!/usr/bin/env python3
"""
Pokeemerald Map Data Parser

Parses pokeemerald JSON map files and binary layout files (map.bin, border.bin)
to extract map data for pathfinding and visualization.
"""

import struct
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Import MetatileBehavior only when needed to avoid circular dependencies
try:
    from pokemon_env.enums import MetatileBehavior
except ImportError:
    # Fallback if pokemon_env is not available
    from enum import IntEnum
    class MetatileBehavior(IntEnum):
        """Fallback MetatileBehavior enum"""
        NORMAL = 0


class PokeemeraldLayoutParser:
    """Parse pokeemerald layout binary files (map.bin and border.bin)"""
    
    def __init__(self, pokeemerald_root: Path):
        self.root = Path(pokeemerald_root)
        self.layouts_dir = self.root / "data" / "layouts"
        
        # Load layout index
        layouts_json = self.layouts_dir / "layouts.json"
        if not layouts_json.exists():
            raise FileNotFoundError(f"Layouts JSON not found: {layouts_json}")
        
        with open(layouts_json) as f:
            self.layouts_data = json.load(f)
            
        # Create lookup from layout name to layout info
        self.layout_lookup = {}
        for layout in self.layouts_data.get("layouts", []):
            # Store by both ID and name for easy lookup
            self.layout_lookup[layout["id"]] = layout
            layout_name = layout.get("name", "").replace("_Layout", "")
            self.layout_lookup[layout_name] = layout
    
    def get_layout_info(self, layout_name_or_id: str) -> Optional[Dict]:
        """Get layout information by name or ID"""
        # Try direct lookup
        if layout_name_or_id in self.layout_lookup:
            return self.layout_lookup[layout_name_or_id]
        
        # Try case-insensitive lookup
        for key, layout in self.layout_lookup.items():
            if key.upper() == layout_name_or_id.upper():
                return layout
        
        # Try partial match
        layout_name_lower = layout_name_or_id.lower().replace("_layout", "")
        for key, layout in self.layout_lookup.items():
            key_lower = key.lower().replace("layout_", "").replace("_", "")
            if layout_name_lower in key_lower or key_lower in layout_name_lower:
                return layout
        
        return None
    
    def parse_map_bin(self, layout_name_or_id: str) -> Optional[List[List[int]]]:
        """
        Parse map.bin file for a given layout.
        
        Returns:
            2D list of metatile values (16-bit packed: id + collision + elevation)
            or None if file not found
        """
        layout_info = self.get_layout_info(layout_name_or_id)
        if not layout_info:
            return None
            
        width = layout_info["width"]
        height = layout_info["height"]
        map_path = self.root / layout_info["blockdata_filepath"]
        
        if not map_path.exists():
            return None
            
        return self._parse_binary_file(map_path, width, height)
    
    def parse_border_bin(self, layout_name_or_id: str) -> Optional[List[List[int]]]:
        """
        Parse border.bin file for a given layout.
        
        Returns:
            2D list of border metatile values, or None if file not found
        """
        layout_info = self.get_layout_info(layout_name_or_id)
        if not layout_info:
            return None
            
        border_path = self.root / layout_info["border_filepath"]
        
        if not border_path.exists():
            return None
        
        # Border is typically 2x2 metatiles
        border_width = 2
        border_height = 2
        
        return self._parse_binary_file(border_path, border_width, border_height)
    
    def _parse_binary_file(self, file_path: Path, width: int, height: int) -> List[List[int]]:
        """
        Parse a binary file containing metatile data.
        
        Args:
            file_path: Path to .bin file
            width: Width in metatiles
            height: Height in metatiles
            
        Returns:
            2D list [y][x] of metatile values (16-bit packed: id + collision + elevation)
        """
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Expected size: width * height * 2 bytes (each metatile is u16)
        expected_size = width * height * 2
        if len(data) < expected_size:
            raise ValueError(f"File {file_path} too small: got {len(data)} bytes, expected {expected_size}")
        
        metatiles = []
        for y in range(height):
            row = []
            for x in range(width):
                # Calculate byte offset (row-major order)
                offset = (y * width + x) * 2
                # Read 16-bit little-endian unsigned integer
                metatile_value = struct.unpack('<H', data[offset:offset+2])[0]
                row.append(metatile_value)
            metatiles.append(row)
        
        return metatiles
    
    def unpack_metatile(self, metatile_value: int) -> Tuple[int, int, int]:
        """
        Unpack a metatile value into its components.
        
        Args:
            metatile_value: 16-bit packed metatile value
            
        Returns:
            Tuple of (metatile_id, collision, elevation)
        """
        metatile_id = metatile_value & 0x03FF      # Bits 0-9
        collision = (metatile_value & 0x0C00) >> 10  # Bits 10-11
        elevation = (metatile_value & 0xF000) >> 12  # Bits 12-15
        return (metatile_id, collision, elevation)
    
    def get_metatiles_with_behavior(self, layout_name_or_id: str) -> Optional[List[List[Tuple[int, MetatileBehavior, int, int]]]]:
        """
        Parse map.bin and return metatiles with behavior.
        
        Returns:
            2D list of tuples: (metatile_id, behavior, collision, elevation)
            Behavior is estimated from collision and basic heuristics.
        """
        metatiles = self.parse_map_bin(layout_name_or_id)
        if metatiles is None:
            return None
        
        result = []
        for row in metatiles:
            result_row = []
            for metatile_value in row:
                metatile_id, collision, elevation = self.unpack_metatile(metatile_value)
                
                # Estimate behavior from collision and metatile_id
                # For now, use NORMAL for most tiles
                # Collision == 0 means walkable, non-zero means blocked
                behavior = MetatileBehavior.NORMAL
                
                # Try to infer some behaviors from metatile_id
                # This is a simplified approach - proper behavior would come from metatile_attributes.bin
                if collision > 0:
                    # Blocked tiles - could be various types
                    if metatile_id > 512:  # Higher IDs often indicate special tiles
                        behavior = MetatileBehavior.SECRET_BASE_WALL
                elif metatile_id == 1023:  # 0x3FF - invalid/out of bounds
                    behavior = MetatileBehavior.NORMAL
                
                result_row.append((metatile_id, behavior, collision, elevation))
            result.append(result_row)
        
        return result


class PokeemeraldMapLoader:
    """Load and parse pokeemerald map data from JSON files"""
    
    def __init__(self, pokeemerald_root: Path):
        self.root = Path(pokeemerald_root)
        self.maps_dir = self.root / "data" / "maps"
        
        # Load map groups index
        map_groups_json = self.maps_dir / "map_groups.json"
        if not map_groups_json.exists():
            raise FileNotFoundError(f"Map groups JSON not found: {map_groups_json}")
        
        with open(map_groups_json) as f:
            self.map_groups = json.load(f)
    
    def load_map(self, map_name: str) -> Optional[Dict]:
        """Load a specific map's JSON data"""
        map_path = self.maps_dir / map_name / "map.json"
        if not map_path.exists():
            return None
        
        with open(map_path) as f:
            return json.load(f)
    
    def get_map_connections(self, map_name: str) -> List[Dict]:
        """Get all connections from a map"""
        map_data = self.load_map(map_name)
        if not map_data:
            return []
        return map_data.get("connections", []) or []
    
    def get_warp_events(self, map_name: str) -> List[Dict]:
        """Get all warp points for a map"""
        map_data = self.load_map(map_name)
        if not map_data:
            return []
        return map_data.get("warp_events", [])
    
    def get_layout_name_from_map(self, map_name: str) -> Optional[str]:
        """Get the layout name from a map JSON"""
        map_data = self.load_map(map_name)
        if not map_data:
            return None
        
        layout_id = map_data.get("layout")
        if layout_id:
            # Convert "LAYOUT_OLDALE_TOWN" -> "OldaleTown"
            return layout_id.replace("LAYOUT_", "").lower().replace("_", " ").title().replace(" ", "")
        return None


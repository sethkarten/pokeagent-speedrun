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
        """
        Get layout information by name or ID.
        
        Uses strict matching first, then falls back to more flexible matching.
        This ensures each location gets its own unique layout.
        """
        # Try direct lookup first (exact match)
        if layout_name_or_id in self.layout_lookup:
            return self.layout_lookup[layout_name_or_id]
        
        # Try case-insensitive exact match (but not partial)
        for key, layout in self.layout_lookup.items():
            if key.upper() == layout_name_or_id.upper():
                return layout
        
        # Normalize for partial matching (only if exact match failed)
        # Remove common prefixes/suffixes
        normalized_input = layout_name_or_id.upper().replace("LAYOUT_", "").replace("_", "").replace(" ", "")
        
        # Try to match normalized names, but prefer exact matches
        best_match = None
        best_match_score = 0
        
        for key, layout in self.layout_lookup.items():
            # Normalize the key similarly
            normalized_key = key.upper().replace("LAYOUT_", "").replace("_", "").replace(" ", "")
            
            # Check for exact normalized match first (highest priority)
            if normalized_input == normalized_key:
                return layout
            
            # Calculate partial match score (lower priority, only if no exact match)
            # Prefer matches that start with the input or where input starts with the key
            if normalized_input.startswith(normalized_key) or normalized_key.startswith(normalized_input):
                match_score = min(len(normalized_input), len(normalized_key))
                # Only update if this is a better match (longer match)
                if match_score > best_match_score:
                    best_match = layout
                    best_match_score = match_score
        
        # Only return partial match if it's a very good match (at least 80% of characters match)
        if best_match and best_match_score >= max(len(normalized_input) * 0.8, len(normalized_input) - 5):
            return best_match
        
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
    
    def _infer_behavior_from_metatile_id(self, metatile_id: int, collision: int) -> MetatileBehavior:
        """
        Infer metatile behavior from metatile ID using heuristics.
        
        This is a fallback when tileset metatile_attributes.bin is not available.
        Uses common metatile ID ranges from Pokemon Emerald.
        
        Args:
            metatile_id: Metatile ID (0-1023)
            collision: Collision value (0-3)
            
        Returns:
            Inferred MetatileBehavior
        """
        # Invalid/out of bounds
        if metatile_id == 1023:
            return MetatileBehavior.NORMAL
        
        # Blocked tiles (collision > 0)
        if collision > 0:
            # Check for directional impassable
            if 768 <= metatile_id < 832:
                # Common impassable ranges (varies by tileset)
                return MetatileBehavior.SECRET_BASE_WALL
            return MetatileBehavior.SECRET_BASE_WALL
        
        # Walkable tiles - use metatile ID ranges to infer behavior
        # These ranges are approximate and vary by tileset, but work for common cases
        
        # Water tiles (commonly in ranges 144-223, varies by tileset)
        if 144 <= metatile_id < 224:
            # Try to distinguish water types (heuristic)
            if 144 <= metatile_id < 160:
                return MetatileBehavior.POND_WATER
            elif 160 <= metatile_id < 176:
                return MetatileBehavior.DEEP_WATER
            elif 176 <= metatile_id < 192:
                return MetatileBehavior.OCEAN_WATER
            elif 192 <= metatile_id < 208:
                return MetatileBehavior.SHALLOW_WATER
            else:
                return MetatileBehavior.WATERFALL
        
        # Grass tiles (commonly in ranges 16-63)
        if 16 <= metatile_id < 64:
            if 16 <= metatile_id < 32:
                return MetatileBehavior.TALL_GRASS
            elif 32 <= metatile_id < 48:
                return MetatileBehavior.LONG_GRASS
            elif 48 <= metatile_id < 56:
                return MetatileBehavior.SHORT_GRASS
            else:
                return MetatileBehavior.LONG_GRASS_SOUTH_EDGE
        
        # Sand (commonly around 80-95)
        if 80 <= metatile_id < 96:
            if 86 <= metatile_id < 88:
                return MetatileBehavior.DEEP_SAND
            else:
                return MetatileBehavior.SAND
        
        # Ice (commonly around 224-255)
        if 224 <= metatile_id < 256:
            if 230 <= metatile_id < 232:
                return MetatileBehavior.THIN_ICE
            elif 232 <= metatile_id < 234:
                return MetatileBehavior.CRACKED_ICE
            else:
                return MetatileBehavior.ICE
        
        # Ledges (commonly in ranges 384-447)
        if 384 <= metatile_id < 448:
            # Jump directions
            if 400 <= metatile_id < 408:
                return MetatileBehavior.JUMP_EAST
            elif 408 <= metatile_id < 416:
                return MetatileBehavior.JUMP_WEST
            elif 416 <= metatile_id < 424:
                return MetatileBehavior.JUMP_NORTH
            elif 424 <= metatile_id < 432:
                return MetatileBehavior.JUMP_SOUTH
            elif 432 <= metatile_id < 440:
                return MetatileBehavior.JUMP_NORTHEAST
            elif 440 <= metatile_id < 448:
                return MetatileBehavior.JUMP_NORTHWEST
        
        # Water currents (commonly around 512-575)
        if 512 <= metatile_id < 576:
            if 528 <= metatile_id < 536:
                return MetatileBehavior.EASTWARD_CURRENT
            elif 536 <= metatile_id < 544:
                return MetatileBehavior.WESTWARD_CURRENT
            elif 544 <= metatile_id < 552:
                return MetatileBehavior.NORTHWARD_CURRENT
            elif 552 <= metatile_id < 560:
                return MetatileBehavior.SOUTHWARD_CURRENT
        
        # Warps/stairs/doors (commonly in ranges 640-767)
        if 640 <= metatile_id < 768:
            if 656 <= metatile_id < 664:
                return MetatileBehavior.NON_ANIMATED_DOOR
            elif 664 <= metatile_id < 672:
                return MetatileBehavior.LADDER
            elif 672 <= metatile_id < 680:
                return MetatileBehavior.SOUTH_ARROW_WARP
        
        # Special terrain
        if 96 <= metatile_id < 112:
            return MetatileBehavior.HOT_SPRINGS
        if 256 <= metatile_id < 320:
            return MetatileBehavior.CAVE
        
        # Default to NORMAL for walkable tiles
        return MetatileBehavior.NORMAL
    
    def get_metatiles_with_behavior(self, layout_name_or_id: str) -> Optional[List[List[Tuple[int, MetatileBehavior, int, int]]]]:
        """
        Parse map.bin and return metatiles with behavior.
        
        Returns:
            2D list of tuples: (metatile_id, behavior, collision, elevation)
            Behavior is inferred from metatile_id using heuristics since
            tileset metatile_attributes.bin files are not available.
        """
        metatiles = self.parse_map_bin(layout_name_or_id)
        if metatiles is None:
            return None
        
        result = []
        for row in metatiles:
            result_row = []
            for metatile_value in row:
                metatile_id, collision, elevation = self.unpack_metatile(metatile_value)
                
                # Infer behavior from metatile_id and collision
                behavior = self._infer_behavior_from_metatile_id(metatile_id, collision)
                
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


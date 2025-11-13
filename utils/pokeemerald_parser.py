#!/usr/bin/env python3
"""
Pokeemerald Map Data Parser

Parses pokeemerald JSON map files and binary layout files (map.bin)
to extract map data for pathfinding and visualization.

Uses ground truth metatile behavior data from tileset metatile_attributes.bin files.
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
    """Parse pokeemerald layout binary files (map.bin) and tileset attributes"""
    
    def __init__(self, pokeemerald_root: Path):
        self.root = Path(pokeemerald_root)
        self.layouts_dir = self.root / "data" / "layouts"
        self.tilesets_dir = self.root / "data" / "tilesets"
        
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
        
        # Tileset name to directory mapping (complete list of all Pokemon Emerald tilesets)
        self.tileset_to_dir = {
            # Primary tilesets
            "gTileset_General": "primary/general",
            "gTileset_Building": "primary/building",
            "gTileset_SecretBase": "primary/secret_base",
            # Secondary tilesets (all 67 of them)
            "gTileset_BattleArena": "secondary/battle_arena",
            "gTileset_BattleDome": "secondary/battle_dome",
            "gTileset_BattleFactory": "secondary/battle_factory",
            "gTileset_BattleFrontier": "secondary/battle_frontier",
            "gTileset_BattleFrontierOutsideEast": "secondary/battle_frontier_outside_east",
            "gTileset_BattleFrontierOutsideWest": "secondary/battle_frontier_outside_west",
            "gTileset_BattleFrontierRankingHall": "secondary/battle_frontier_ranking_hall",
            "gTileset_BattlePalace": "secondary/battle_palace",
            "gTileset_BattlePike": "secondary/battle_pike",
            "gTileset_BattlePyramid": "secondary/battle_pyramid",
            "gTileset_BattleTent": "secondary/battle_tent",
            "gTileset_BikeShop": "secondary/bike_shop",
            "gTileset_BrendansMaysHouse": "secondary/brendans_mays_house",
            "gTileset_CableClub": "secondary/cable_club",
            "gTileset_Cave": "secondary/cave",
            "gTileset_Contest": "secondary/contest",
            "gTileset_Dewford": "secondary/dewford",
            "gTileset_DewfordGym": "secondary/dewford_gym",
            "gTileset_EliteFour": "secondary/elite_four",
            "gTileset_EverGrande": "secondary/ever_grande",
            "gTileset_Facility": "secondary/facility",
            "gTileset_Fallarbor": "secondary/fallarbor",
            "gTileset_Fortree": "secondary/fortree",
            "gTileset_FortreeGym": "secondary/fortree_gym",
            "gTileset_GenericBuilding": "secondary/generic_building",
            "gTileset_InsideOfTruck": "secondary/inside_of_truck",
            "gTileset_InsideShip": "secondary/inside_ship",
            "gTileset_IslandHarbor": "secondary/island_harbor",
            "gTileset_Lab": "secondary/lab",
            "gTileset_Lavaridge": "secondary/lavaridge",
            "gTileset_LavaridgeGym": "secondary/lavaridge_gym",
            "gTileset_Lilycove": "secondary/lilycove",
            "gTileset_LilycoveMuseum": "secondary/lilycove_museum",
            "gTileset_Mauville": "secondary/mauville",
            "gTileset_MauvilleGameCorner": "secondary/mauville_game_corner",
            "gTileset_MauvilleGym": "secondary/mauville_gym",
            "gTileset_MeteorFalls": "secondary/meteor_falls",
            "gTileset_MirageTower": "secondary/mirage_tower",
            "gTileset_Mossdeep": "secondary/mossdeep",
            "gTileset_MossdeepGameCorner": "secondary/mossdeep_game_corner",
            "gTileset_MossdeepGym": "secondary/mossdeep_gym",
            "gTileset_MysteryEventsHouse": "secondary/mystery_events_house",
            "gTileset_NavelRock": "secondary/navel_rock",
            "gTileset_OceanicMuseum": "secondary/oceanic_museum",
            "gTileset_Pacifidlog": "secondary/pacifidlog",
            "gTileset_Petalburg": "secondary/petalburg",
            "gTileset_PetalburgGym": "secondary/petalburg_gym",
            "gTileset_PokemonCenter": "secondary/pokemon_center",
            "gTileset_PokemonDayCare": "secondary/pokemon_day_care",
            "gTileset_PokemonFanClub": "secondary/pokemon_fan_club",
            "gTileset_PokemonSchool": "secondary/pokemon_school",
            "gTileset_PrettyPetalFlowerShop": "secondary/pretty_petal_flower_shop",
            "gTileset_Rustboro": "secondary/rustboro",
            "gTileset_RustboroGym": "secondary/rustboro_gym",
            "gTileset_RusturfTunnel": "secondary/rusturf_tunnel",
            "gTileset_SeashoreHouse": "secondary/seashore_house",
            "gTileset_SecretBase": "secondary/secret_base",
            "gTileset_Shop": "secondary/shop",
            "gTileset_Slateport": "secondary/slateport",
            "gTileset_Sootopolis": "secondary/sootopolis",
            "gTileset_SootopolisGym": "secondary/sootopolis_gym",
            "gTileset_TrainerHill": "secondary/trainer_hill",
            "gTileset_TrickHousePuzzle": "secondary/trick_house_puzzle",
            "gTileset_Underwater": "secondary/underwater",
            "gTileset_UnionRoom": "secondary/union_room",
            "gTileset_Unused1": "secondary/unused_1",
            "gTileset_Unused2": "secondary/unused_2",
        }
        
        # Cache loaded tileset attributes to avoid re-reading files
        self._tileset_attributes_cache = {}
    
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
    
    def _load_tileset_attributes(self, tileset_name: str) -> List[int]:
        """
        Load metatile_attributes.bin for a tileset and extract behavior values.
        
        The metatile_attributes.bin file contains packed u16 values with:
        - Bits 0-7 (mask 0x00FF): Behavior ID (MB_NORMAL, MB_JUMP_EAST, etc.)
        - Bits 12-15 (mask 0xF000): Layer type
        
        This is the GROUND TRUTH source for metatile behaviors including ledges,
        grass, water, warps, etc. No heuristics or fallbacks are used.
        
        Args:
            tileset_name: Tileset name like "gTileset_General"
            
        Returns:
            List of behavior values (unpacked, bits 0-7 only)
            
        Raises:
            ValueError: If tileset name is unknown
            FileNotFoundError: If attributes file doesn't exist
        """
        # Check cache first
        if tileset_name in self._tileset_attributes_cache:
            return self._tileset_attributes_cache[tileset_name]
        
        # Get tileset directory
        tileset_dir = self.tileset_to_dir.get(tileset_name)
        if not tileset_dir:
            # Fail loudly for unknown tilesets
            raise ValueError(
                f"Unknown tileset: '{tileset_name}'. "
                f"Add mapping to tileset_to_dir. Known tilesets: {list(self.tileset_to_dir.keys())}"
            )
        
        attributes_path = self.tilesets_dir / tileset_dir / "metatile_attributes.bin"
        if not attributes_path.exists():
            raise FileNotFoundError(
                f"Tileset attributes file not found: {attributes_path}\n"
                f"Ensure pokeemerald root is correct and contains data/tilesets/")
        
        # Read attribute file (each entry is u16 = 2 bytes)
        with open(attributes_path, 'rb') as f:
            data = f.read()
        
        # Parse as array of u16 (little-endian)
        # Note: attributes are packed values with behavior in bits 0-7 (mask 0x00FF)
        # and layer type in bits 12-15 (mask 0xF000)
        num_metatiles = len(data) // 2
        attributes = []
        for i in range(num_metatiles):
            offset = i * 2
            packed_attr = struct.unpack('<H', data[offset:offset+2])[0]
            # Unpack behavior from bits 0-7
            behavior = packed_attr & 0x00FF
            attributes.append(behavior)
        
        # Cache for future use
        self._tileset_attributes_cache[tileset_name] = attributes
        return attributes
    
    def get_metatiles_with_behavior(self, layout_name_or_id: str) -> List[List[Tuple[int, MetatileBehavior, int, int]]]:
        """
        Parse map.bin and return metatiles with actual behavior from tileset attributes.
        
        Args:
            layout_name_or_id: Layout name or ID
        
        Returns:
            2D list of tuples: (metatile_id, behavior, collision, elevation)
            Behavior is loaded from tileset metatile_attributes.bin files.
            
        Raises:
            ValueError: If layout not found or missing tileset information
            FileNotFoundError: If tileset attributes files don't exist
            IndexError: If metatile ID is out of range for the tileset
        """
        # Get layout info
        layout_info = self.get_layout_info(layout_name_or_id)
        if not layout_info:
            raise ValueError(f"Layout not found: '{layout_name_or_id}'")
        
        # Get tileset names
        primary_tileset = layout_info.get("primary_tileset")
        secondary_tileset = layout_info.get("secondary_tileset")
        
        if not primary_tileset or not secondary_tileset:
            raise ValueError(
                f"Layout '{layout_name_or_id}' missing tileset information. "
                f"primary_tileset={primary_tileset}, secondary_tileset={secondary_tileset}"
            )
        
        # Load tileset attributes (will fail loudly if not found)
        primary_attrs = self._load_tileset_attributes(primary_tileset)
        secondary_attrs = self._load_tileset_attributes(secondary_tileset)
        
        # Parse map.bin
        metatiles = self.parse_map_bin(layout_name_or_id)
        if metatiles is None:
            raise ValueError(f"Failed to parse map.bin for layout '{layout_name_or_id}'")
        
        # Unpack each metatile and add actual behavior from tilesets
        result = []
        for row in metatiles:
            result_row = []
            for metatile_value in row:
                metatile_id, collision, elevation = self.unpack_metatile(metatile_value)
                
                # Look up actual behavior from tileset attributes
                # Primary tileset: metatiles 0-511
                # Secondary tileset: metatiles 512-1023
                if metatile_id < 512:
                    # Primary tileset
                    if metatile_id >= len(primary_attrs):
                        raise IndexError(
                            f"Metatile ID {metatile_id} out of range for primary tileset '{primary_tileset}' "
                            f"(has {len(primary_attrs)} metatiles)"
                        )
                    behavior_value = primary_attrs[metatile_id]
                else:
                    # Secondary tileset (offset by 512)
                    secondary_id = metatile_id - 512
                    if secondary_id >= len(secondary_attrs):
                        raise IndexError(
                            f"Metatile ID {metatile_id} (secondary {secondary_id}) out of range for "
                            f"secondary tileset '{secondary_tileset}' (has {len(secondary_attrs)} metatiles)"
                        )
                    behavior_value = secondary_attrs[secondary_id]
                
                # Convert to MetatileBehavior enum
                try:
                    behavior_enum = MetatileBehavior(behavior_value)
                except ValueError:
                    # Fail loudly on unknown behavior values
                    raise ValueError(
                        f"Unknown behavior value {behavior_value} (0x{behavior_value:04X}) for metatile {metatile_id} "
                        f"in layout '{layout_name_or_id}'. This may indicate a missing or corrupted tileset file."
                    )
                
                result_row.append((metatile_id, behavior_enum, collision, elevation))
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


#!/usr/bin/env python3
"""
Map Stitching System for Pokemon Emerald

Connects previously seen map areas with warps and transitions to create
a unified world map showing connections between routes, towns, and buildings.
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from pokemon_env.enums import MapLocation, MetatileBehavior

logger = logging.getLogger(__name__)

@dataclass
class WarpConnection:
    """Represents a connection between two map areas"""
    from_map_id: int  # (map_bank << 8) | map_number
    to_map_id: int
    from_position: Tuple[int, int]  # Player position when warp triggered
    to_position: Tuple[int, int]    # Player position after warp
    warp_type: str  # "door", "stairs", "exit", "route_transition"
    direction: str  # "north", "south", "east", "west", "up", "down"
    
    def get_reverse_connection(self) -> 'WarpConnection':
        """Get the reverse direction of this warp"""
        reverse_dirs = {
            "north": "south", "south": "north",
            "east": "west", "west": "east", 
            "up": "down", "down": "up"
        }
        return WarpConnection(
            from_map_id=self.to_map_id,
            to_map_id=self.from_map_id,
            from_position=self.to_position,
            to_position=self.from_position,
            warp_type=self.warp_type,
            direction=reverse_dirs.get(self.direction, "unknown")
        )

@dataclass
class MapArea:
    """Represents a single map area with its data"""
    map_id: int  # (map_bank << 8) | map_number
    location_name: str
    map_data: List[List[Tuple]]  # Raw tile data from memory
    player_last_position: Tuple[int, int]  # Last known player position
    warp_tiles: List[Tuple[int, int, str]]  # (x, y, warp_type) positions
    boundaries: Dict[str, int]  # north, south, east, west limits
    visited_count: int
    first_seen: float  # timestamp
    last_seen: float   # timestamp
    overworld_coords: Optional[Tuple[int, int]] = None  # (X, Y) in overworld coordinate system
    
    def get_map_bounds(self) -> Tuple[int, int, int, int]:
        """Return (min_x, min_y, max_x, max_y) for this map"""
        height = len(self.map_data)
        width = len(self.map_data[0]) if height > 0 else 0
        return (0, 0, width - 1, height - 1)
    
    def has_warp_at(self, x: int, y: int) -> Optional[str]:
        """Check if there's a warp at the given position"""
        for wx, wy, warp_type in self.warp_tiles:
            if wx == x and wy == y:
                return warp_type
        return None

class MapStitcher:
    """Main class for managing map stitching and connections"""
    
    def __init__(self, save_file: str = None):
        # Setup cache directory
        self.cache_dir = ".pokeagent_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Use cache folder for default save file
        if save_file is None:
            save_file = os.path.join(self.cache_dir, "map_stitcher_data.json")
        self.save_file = Path(save_file)
        self.map_areas: Dict[int, MapArea] = {}
        self.warp_connections: List[WarpConnection] = []
        self.pending_warps: List[Dict] = []  # Track potential warps
        self.last_map_id: Optional[int] = None
        self.last_position: Optional[Tuple[int, int]] = None
        
        # Load existing data
        self.load_from_file()
    
    def get_map_id(self, map_bank: int, map_number: int) -> int:
        """Convert map bank/number to unique ID"""
        return (map_bank << 8) | map_number
    
    def decode_map_id(self, map_id: int) -> Tuple[int, int]:
        """Convert map ID back to bank/number"""
        return (map_id >> 8, map_id & 0xFF)
    
    def update_save_file(self, new_save_file: str):
        """Update the save file path and reload data"""
        self.save_file = Path(new_save_file)
        # Clear current data and reload from new file
        self.map_areas = {}
        self.warp_connections = []
        self.pending_warps = []
        self.load_from_file()
    
    def update_map_area(self, map_bank: int, map_number: int, location_name: str,
                       map_data: List[List[Tuple]], player_pos: Tuple[int, int],
                       timestamp: float, overworld_coords: Optional[Tuple[int, int]] = None):
        """Update or create a map area with new data"""
        map_id = self.get_map_id(map_bank, map_number)
        
        # Skip map 0 (startup/initialization state) as it's not a real location
        if map_id == 0:
            logger.debug(f"Skipping map 0 (startup state)")
            return
        
        # Detect warp tiles in the map data
        warp_tiles = self._detect_warp_tiles(map_data)
        
        if map_id in self.map_areas:
            # Update existing area
            area = self.map_areas[map_id]
            # Update location name if we have a better one (not empty or "Unknown")
            if location_name and location_name.strip() and location_name != "Unknown":
                if area.location_name == "Unknown" or not area.location_name:
                    logger.info(f"Updating location name for map {map_id:04X}: '{area.location_name}' -> '{location_name}'")
                    area.location_name = location_name
                    # Try to resolve other unknown names since we got new location info
                    self.resolve_unknown_location_names()
                elif area.location_name != location_name:
                    # Handle case where we get a different name (e.g. more specific location)
                    logger.info(f"Found different location name for map {map_id:04X}: '{area.location_name}' vs '{location_name}', keeping current")
                else:
                    area.location_name = location_name
            area.map_data = map_data
            area.player_last_position = player_pos
            area.warp_tiles = warp_tiles
            area.visited_count += 1
            area.last_seen = timestamp
            # Keep overworld_coords as None to indicate separate maps
            area.overworld_coords = None
            logger.debug(f"Updated map area {area.location_name} (ID: {map_id:04X})")
        else:
            # Create new area without global coordinates
            boundaries = self._calculate_boundaries(map_data)
            # Try to resolve location name from map ID if empty
            if not location_name or not location_name.strip():
                # Import and use the location mapping
                from pokemon_env.enums import MapLocation
                try:
                    map_enum = MapLocation(map_id)
                    final_location_name = map_enum.name.replace('_', ' ').title()
                    logger.info(f"Resolved location name for map {map_id:04X}: {final_location_name}")
                except ValueError:
                    # Fallback for unknown map IDs
                    final_location_name = f"Map_{map_id:04X}"
                    logger.debug(f"Unknown map ID {map_id:04X}, using fallback name")
            else:
                final_location_name = location_name
            
            area = MapArea(
                map_id=map_id,
                location_name=final_location_name,
                map_data=map_data,
                player_last_position=player_pos,
                warp_tiles=warp_tiles,
                boundaries=boundaries,
                visited_count=1,
                first_seen=timestamp,
                last_seen=timestamp,
                overworld_coords=None  # Each location is separate
            )
            self.map_areas[map_id] = area
            logger.info(f"Added new map area: {final_location_name} (ID: {map_id:04X}) as separate location")
            
        # Check for area transitions and potential warp connections
        if self.last_map_id is not None and self.last_map_id != map_id:
            self._detect_warp_connection(self.last_map_id, map_id, 
                                       self.last_position, player_pos, timestamp)
            
            # Try to resolve any unknown location names after adding connections  
            # Note: resolve_unknown_location_names() can be called with memory_reader from calling code
            if self.resolve_unknown_location_names():
                logger.info("Resolved unknown location names after area transition")
                # Save will be handled by the calling code
        
        self.last_map_id = map_id
        self.last_position = player_pos
    
    def _detect_warp_tiles(self, map_data: List[List[Tuple]]) -> List[Tuple[int, int, str]]:
        """Detect tiles that can be warps (doors, stairs, exits)"""
        warp_tiles = []
        
        for y, row in enumerate(map_data):
            for x, tile in enumerate(row):
                if len(tile) >= 2:
                    tile_id, behavior = tile[:2]
                    
                    if hasattr(behavior, 'name'):
                        behavior_name = behavior.name
                    elif isinstance(behavior, int):
                        try:
                            behavior_enum = MetatileBehavior(behavior)
                            behavior_name = behavior_enum.name
                        except ValueError:
                            continue
                    else:
                        continue
                    
                    # Classify warp types
                    warp_type = None
                    if "DOOR" in behavior_name:
                        warp_type = "door"
                    elif "STAIRS" in behavior_name:
                        warp_type = "stairs"
                    elif "WARP" in behavior_name:
                        warp_type = "warp"
                    elif x == 0 or x == len(row) - 1 or y == 0 or y == len(map_data) - 1:
                        # Edge tiles might be exits to other routes/areas
                        if behavior_name == "NORMAL" and tile[2] == 0:  # collision == 0
                            warp_type = "exit"
                    
                    if warp_type:
                        warp_tiles.append((x, y, warp_type))
        
        return warp_tiles
    
    def _calculate_boundaries(self, map_data: List[List[Tuple]]) -> Dict[str, int]:
        """Calculate walkable boundaries of the map"""
        height = len(map_data)
        width = len(map_data[0]) if height > 0 else 0
        
        return {
            "north": 0,
            "south": height - 1,
            "west": 0, 
            "east": width - 1
        }
    
    def _detect_warp_connection(self, from_map_id: int, to_map_id: int,
                              from_pos: Optional[Tuple[int, int]], 
                              to_pos: Tuple[int, int], timestamp: float):
        """Detect and record warp connections between maps"""
        if from_pos is None:
            return
            
        from_area = self.map_areas.get(from_map_id)
        to_area = self.map_areas.get(to_map_id)
        
        if not from_area or not to_area:
            return
        
        # Determine warp type and direction
        warp_type = "route_transition"  # default
        direction = self._determine_warp_direction(from_area, to_area, from_pos, to_pos)
        
        # Check if we were near a warp tile
        near_warp = from_area.has_warp_at(from_pos[0], from_pos[1])
        if near_warp:
            warp_type = near_warp
        
        # Create the connection
        connection = WarpConnection(
            from_map_id=from_map_id,
            to_map_id=to_map_id,
            from_position=from_pos,
            to_position=to_pos,
            warp_type=warp_type,
            direction=direction
        )
        
        # Check if this connection already exists
        if not self._connection_exists(connection):
            self.warp_connections.append(connection)
            logger.info(f"Added warp connection: {from_area.location_name} -> {to_area.location_name} "
                       f"({warp_type}, {direction})")
            
            # Auto-add reverse connection for two-way warps
            if warp_type in ["door", "stairs", "route_transition"]:
                reverse = connection.get_reverse_connection()
                if not self._connection_exists(reverse):
                    self.warp_connections.append(reverse)
                    logger.debug(f"Added reverse connection: {to_area.location_name} -> {from_area.location_name}")
    
    def _determine_warp_direction(self, from_area: MapArea, to_area: MapArea,
                                from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Determine the direction of movement for a warp"""
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        
        # Check if this is a vertical building transition (indoors <-> outdoors)
        from_indoor = from_area.location_name and ("HOUSE" in from_area.location_name.upper() or "ROOM" in from_area.location_name.upper())
        to_indoor = to_area.location_name and ("HOUSE" in to_area.location_name.upper() or "ROOM" in to_area.location_name.upper())
        
        if from_indoor and not to_indoor:
            return "down"  # Exiting building
        elif not from_indoor and to_indoor:
            return "up"    # Entering building
        
        # For horizontal transitions, compare positions
        from_bounds = from_area.get_map_bounds()
        to_bounds = to_area.get_map_bounds()
        
        # Simple heuristic based on position relative to map center
        from_center_x = (from_bounds[2] - from_bounds[0]) // 2
        from_center_y = (from_bounds[3] - from_bounds[1]) // 2
        
        if from_x < from_center_x:
            return "west"
        elif from_x > from_center_x:
            return "east"
        elif from_y < from_center_y:
            return "north"
        else:
            return "south"
    
    def _connection_exists(self, connection: WarpConnection) -> bool:
        """Check if a similar connection already exists"""
        for existing in self.warp_connections:
            if (existing.from_map_id == connection.from_map_id and
                existing.to_map_id == connection.to_map_id and
                existing.warp_type == connection.warp_type):
                return True
        return False
    
    def _infer_overworld_coordinates(self, location_name: str, player_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Infer overworld coordinates - should return None to keep coordinates unknown until discovered"""
        # All coordinates start as unknown (?, ?) until actually discovered
        # This ensures authentic exploration without pre-existing knowledge
        return None
    
    def update_overworld_coordinates(self, map_id: int, coords: Tuple[int, int]):
        """Update overworld coordinates for a discovered area"""
        if map_id in self.map_areas:
            self.map_areas[map_id].overworld_coords = coords
            logger.info(f"Updated coordinates for {self.map_areas[map_id].location_name}: {coords}")
    
    def update_location_name(self, map_id: int, location_name: str):
        """Update location name for an existing area"""
        if map_id in self.map_areas and location_name and location_name.strip() and location_name != "Unknown":
            area = self.map_areas[map_id]
            if area.location_name == "Unknown" or not area.location_name:
                logger.info(f"Updating location name for map {map_id:04X}: '{area.location_name}' -> '{location_name}'")
                area.location_name = location_name
                # Try to resolve other unknown names since we got new location info
                self.resolve_unknown_location_names()
                return True
        return False
    
    def resolve_unknown_location_names(self, memory_reader=None):
        """Try to resolve 'Unknown' location names using the memory reader if available"""
        resolved_count = 0
        
        # If we have a memory reader, we can potentially resolve current location
        if memory_reader is not None:
            try:
                current_location = memory_reader.read_location()
                current_map_bank = memory_reader._read_u8(memory_reader.addresses.MAP_BANK)
                current_map_number = memory_reader._read_u8(memory_reader.addresses.MAP_NUMBER)
                current_map_id = (current_map_bank << 8) | current_map_number
                
                # Update current map if it's unknown
                if current_map_id in self.map_areas:
                    area = self.map_areas[current_map_id]
                    if area.location_name == "Unknown" and current_location and current_location.strip() and current_location != "Unknown":
                        old_name = area.location_name
                        area.location_name = current_location
                        logger.info(f"Resolved current location name for map {current_map_id:04X}: '{old_name}' -> '{area.location_name}'")
                        resolved_count += 1
            except Exception as e:
                logger.debug(f"Could not resolve current location: {e}")
        
        # For now, we keep a basic approach for some common areas if no memory reader
        # In the future, we could implement a system to revisit areas and get their names
        if memory_reader is None and resolved_count == 0:
            # Basic fallback for common map IDs - only the most essential ones
            # NOTE: Be careful with map ID mappings - check pokemon_env/enums.py for correct values
            basic_mappings = {
                0x0009: "LITTLEROOT TOWN",                 # Map 9 - actual outdoor town map
                0x0011: "ROUTE 102",                       # Map 17 - ROUTE_102 = 0x11
                0x0012: "ROUTE 103",                       # Map 18 - ROUTE_103 = 0x12
                0x0101: "LITTLEROOT TOWN BRENDANS HOUSE 2F", # Map 257 - LITTLEROOT_TOWN_BRENDANS_HOUSE_2F = 0x101
                0x0102: "LITTLEROOT TOWN MAYS HOUSE 1F",   # Map 258 - LITTLEROOT_TOWN_MAYS_HOUSE_1F = 0x102
                0x0103: "LITTLEROOT TOWN MAYS HOUSE 2F",   # Map 259 - LITTLEROOT_TOWN_MAYS_HOUSE_2F = 0x103  
            }
            
            for map_id, area in self.map_areas.items():
                if map_id in basic_mappings:
                    correct_name = basic_mappings[map_id]
                    if area.location_name == "Unknown":
                        # Resolve unknown names
                        old_name = area.location_name
                        area.location_name = correct_name
                        logger.info(f"Resolved location name for map {map_id:04X}: '{old_name}' -> '{area.location_name}'")
                        resolved_count += 1
                    elif area.location_name != correct_name:
                        # Fix incorrect existing names (e.g., house interiors mislabeled as routes)
                        old_name = area.location_name
                        area.location_name = correct_name
                        logger.info(f"Corrected location name for map {map_id:04X}: '{old_name}' -> '{area.location_name}'")
                        resolved_count += 1
        
        if resolved_count > 0:
            logger.info(f"Resolved {resolved_count} unknown location names")
            return True
        return False
    
    def get_connected_areas(self, map_id: int) -> List[Tuple[int, str, str]]:
        """Get all areas connected to the given map ID"""
        connections = []
        for conn in self.warp_connections:
            if conn.from_map_id == map_id:
                to_area = self.map_areas.get(conn.to_map_id)
                if to_area:
                    connections.append((conn.to_map_id, to_area.location_name, conn.direction))
        return connections
    
    def get_world_map_layout(self) -> Dict[str, Any]:
        """Generate a layout showing how all areas connect"""
        layout = {
            "areas": {},
            "connections": []
        }
        
        # Add all known areas
        for map_id, area in self.map_areas.items():
            layout["areas"][f"{map_id:04X}"] = {
                "name": area.location_name,
                "position": area.player_last_position,
                "bounds": area.boundaries,
                "warp_count": len(area.warp_tiles),
                "visited_count": area.visited_count
            }
        
        # Add all connections
        for conn in self.warp_connections:
            from_area = self.map_areas.get(conn.from_map_id)
            to_area = self.map_areas.get(conn.to_map_id)
            if from_area and to_area:
                layout["connections"].append({
                    "from": f"{conn.from_map_id:04X}",
                    "to": f"{conn.to_map_id:04X}",
                    "from_name": from_area.location_name,
                    "to_name": to_area.location_name,
                    "warp_type": conn.warp_type,
                    "direction": conn.direction,
                    "from_pos": conn.from_position,
                    "to_pos": conn.to_position
                })
        
        return layout
    
    def save_to_file(self):
        """Save stitching data to JSON file"""
        try:
            data = {
                "map_areas": {},
                "warp_connections": [],
                "location_connections": {}
            }
            
            # Convert map areas to serializable format
            for map_id, area in self.map_areas.items():
                # Save map_data for world map terrain display
                area_data = {
                    "map_id": area.map_id,
                    "location_name": area.location_name,
                    "map_data": area.map_data,
                    "player_last_position": area.player_last_position,
                    "warp_tiles": area.warp_tiles,
                    "boundaries": area.boundaries,
                    "visited_count": area.visited_count,
                    "first_seen": area.first_seen,
                    "last_seen": area.last_seen,
                    "overworld_coords": area.overworld_coords
                }
                data["map_areas"][str(map_id)] = area_data
            
            # Convert connections to serializable format
            for conn in self.warp_connections:
                data["warp_connections"].append(asdict(conn))
            
            # Save location connections from state_formatter
            try:
                # Don't reload - just import normally to avoid state loss
                from utils import state_formatter
                
                print(f"🗺️ DEBUG: MapStitcher saving - checking state_formatter.LOCATION_CONNECTIONS")
                if hasattr(state_formatter, 'LOCATION_CONNECTIONS'):
                    location_connections = state_formatter.LOCATION_CONNECTIONS
                    print(f"🗺️ DEBUG: Found LOCATION_CONNECTIONS: {location_connections}")
                    data["location_connections"] = location_connections
                    logger.debug(f"Saved {len(location_connections)} location connections")
                    print(f"🗺️ DEBUG: Successfully saved {len(location_connections)} location connections to MapStitcher")
                else:
                    print(f"🗺️ DEBUG: state_formatter has no LOCATION_CONNECTIONS attribute")
                    data["location_connections"] = {}
            except ImportError as e:
                print(f"🗺️ DEBUG: Could not import state_formatter for location connections: {e}")
                data["location_connections"] = {}
            except Exception as e:
                print(f"🗺️ DEBUG: Error saving location connections: {e}")
                data["location_connections"] = {}
            
            with open(self.save_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved map stitching data to {self.save_file}")
            
        except Exception as e:
            logger.error(f"Failed to save map stitching data: {e}")
    
    def load_from_file(self):
        """Load stitching data from JSON file"""
        if not self.save_file.exists():
            return
            
        try:
            with open(self.save_file, 'r') as f:
                data = json.load(f)
            
            # Add loaded data to existing map areas (accumulate knowledge)
            # Restore map areas (with map_data for world map display)
            for map_id_str, area_data in data.get("map_areas", {}).items():
                map_id = int(map_id_str)
                
                # Skip map 0 during loading as well (cleanup old data)
                if map_id == 0:
                    logger.debug(f"Skipping load of map 0 (startup state) during file load")
                    continue
                    
                # Try to resolve location name if it's Unknown or missing
                location_name = area_data.get("location_name")
                if not location_name or location_name == "Unknown":
                    # Import and use the location mapping
                    from pokemon_env.enums import MapLocation
                    try:
                        map_enum = MapLocation(map_id)
                        location_name = map_enum.name.replace('_', ' ').title()
                        logger.info(f"Resolved location name for map {map_id:04X} during load: {location_name}")
                    except ValueError:
                        # Fallback for unknown map IDs
                        location_name = f"Map_{map_id:04X}"
                        logger.debug(f"Unknown map ID {map_id:04X} during load, using fallback name")
                
                area = MapArea(
                    map_id=area_data["map_id"],
                    location_name=location_name,
                    map_data=area_data.get("map_data", []),  # Load saved map data
                    player_last_position=tuple(area_data["player_last_position"]),
                    warp_tiles=[tuple(wt) for wt in area_data["warp_tiles"]],
                    boundaries=area_data["boundaries"],
                    visited_count=area_data["visited_count"],
                    first_seen=area_data["first_seen"],
                    last_seen=area_data["last_seen"],
                    overworld_coords=area_data.get("overworld_coords")  # Handle missing field
                )
                self.map_areas[map_id] = area
            
            # Restore connections (skip any involving map 0)
            for conn_data in data.get("warp_connections", []):
                # Skip connections involving map 0 (startup state)
                if conn_data["from_map_id"] == 0 or conn_data["to_map_id"] == 0:
                    logger.debug(f"Skipping connection involving map 0: {conn_data['from_map_id']} -> {conn_data['to_map_id']}")
                    continue
                    
                conn = WarpConnection(
                    from_map_id=conn_data["from_map_id"],
                    to_map_id=conn_data["to_map_id"],
                    from_position=tuple(conn_data["from_position"]),
                    to_position=tuple(conn_data["to_position"]),
                    warp_type=conn_data["warp_type"],
                    direction=conn_data["direction"]
                )
                self.warp_connections.append(conn)
            
            # Restore location connections to state_formatter (always set, even if empty)
            location_connections = data.get("location_connections", {})
            try:
                from utils import state_formatter
                state_formatter.LOCATION_CONNECTIONS = location_connections
                logger.info(f"Loaded {len(location_connections)} location connections from file")
                print(f"🗺️ DEBUG: MapStitcher loaded LOCATION_CONNECTIONS: {location_connections}")
            except ImportError:
                logger.debug("Could not import state_formatter for location connections")
            
            logger.info(f"Loaded {len(self.map_areas)} areas and {len(self.warp_connections)} connections")
            
            # Try to resolve any "Unknown" location names
            if self.resolve_unknown_location_names():
                # Save the updated names
                self.save_to_file()
            
        except Exception as e:
            logger.error(f"Failed to load map stitching data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the stitched world map"""
        indoor_areas = sum(1 for area in self.map_areas.values() 
                          if area.location_name and ("HOUSE" in area.location_name.upper() or "ROOM" in area.location_name.upper()))
        outdoor_areas = len(self.map_areas) - indoor_areas
        
        warp_types = {}
        for conn in self.warp_connections:
            warp_types[conn.warp_type] = warp_types.get(conn.warp_type, 0) + 1
        
        return {
            "total_areas": len(self.map_areas),
            "indoor_areas": indoor_areas,
            "outdoor_areas": outdoor_areas,
            "total_connections": len(self.warp_connections),
            "warp_types": warp_types,
            "most_visited": max(self.map_areas.values(), key=lambda a: a.visited_count).location_name if self.map_areas else None
        }
    
    def generate_world_map_grid(self, current_map_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate a world map grid showing discovered areas and connections"""
        # Define world map bounds (rough Pokemon Emerald overworld size)
        map_width = 50
        map_height = 35
        
        # Initialize empty grid
        grid = [['.' for _ in range(map_width)] for _ in range(map_height)]
        area_labels = {}
        
        # Place discovered areas on the grid
        for map_id, area in self.map_areas.items():
            coords = area.overworld_coords
            if coords is None:
                continue  # Skip areas without known coordinates
                
            x, y = coords
            if 0 <= x < map_width and 0 <= y < map_height:
                # Determine symbol based on area type
                name = area.location_name.upper() if area.location_name else "UNKNOWN"
                if any(keyword in name for keyword in ["HOUSE", "CENTER", "MART", "GYM", "ROOM"]):
                    symbol = "H"  # Houses/buildings
                elif "ROUTE" in name:
                    symbol = "R"  # Routes
                elif any(keyword in name for keyword in ["TOWN", "CITY"]):
                    symbol = "T"  # Towns/cities
                else:
                    symbol = "?"  # Unknown/other
                
                # Mark current player location
                if map_id == current_map_id:
                    symbol = "P"  # Player
                
                grid[y][x] = symbol
                
                # Store area name for reference
                area_labels[f"{x},{y}"] = area.location_name
        
        # Add connection lines between areas
        for conn in self.warp_connections:
            from_area = self.map_areas.get(conn.from_map_id)
            to_area = self.map_areas.get(conn.to_map_id)
            
            if (from_area and to_area and 
                from_area.overworld_coords and to_area.overworld_coords):
                
                from_x, from_y = from_area.overworld_coords
                to_x, to_y = to_area.overworld_coords
                
                # Draw simple connection line (just mark endpoints for now)
                # In a more sophisticated version, we could draw actual paths
                if (0 <= from_x < map_width and 0 <= from_y < map_height and
                    0 <= to_x < map_width and 0 <= to_y < map_height):
                    
                    # Mark connection endpoints if they're empty
                    if grid[from_y][from_x] == '.':
                        grid[from_y][from_x] = "+"
                    if grid[to_y][to_x] == '.':
                        grid[to_y][to_x] = "+"
        
        return {
            "grid": grid,
            "width": map_width,
            "height": map_height,
            "area_labels": area_labels,
            "legend": {
                "P": "Current Player Location",
                "T": "Town/City", 
                "R": "Route",
                "H": "House/Building",
                "+": "Connection Point",
                ".": "Unexplored",
                "?": "Other Area"
            }
        }
    
    def format_world_map_display(self, current_map_id: Optional[int] = None, max_width: int = 50) -> str:
        """Format world map for display in agent context"""
        world_map = self.generate_world_map_grid(current_map_id)
        grid = world_map["grid"]
        labels = world_map["area_labels"]
        legend = world_map["legend"]
        
        lines = []
        lines.append("=== WORLD MAP ===")
        lines.append("")
        
        # Show grid with coordinates
        for y, row in enumerate(grid):
            row_str = ""
            for x, cell in enumerate(row):
                row_str += cell + " "
            lines.append(f"{y:2d}: {row_str}")
        
        # Add coordinate header at bottom
        header = "    "
        for x in range(0, len(grid[0]), 5):  # Show every 5th coordinate
            header += f"{x:2d}   "
        lines.append("")
        lines.append(header)
        
        # Add legend
        lines.append("")
        lines.append("Legend:")
        for symbol, meaning in legend.items():
            lines.append(f"  {symbol} = {meaning}")
        
        # Add discovered area list
        if labels:
            lines.append("")
            lines.append("Discovered Areas:")
            sorted_areas = sorted(labels.items(), key=lambda x: x[1])
            for coord, name in sorted_areas[:10]:  # Show first 10
                lines.append(f"  {coord}: {name}")
            if len(sorted_areas) > 10:
                lines.append(f"  ... and {len(sorted_areas) - 10} more")
        
        return "\n".join(lines)
    
    def save_to_checkpoint(self, checkpoint_data: dict):
        """Save map stitching data to checkpoint data structure"""
        try:
            map_stitcher_data = {
                "map_areas": {},
                "warp_connections": [],
                "location_connections": {}
            }
            
            # Convert map areas to serializable format (without map_data)
            for map_id, area in self.map_areas.items():
                area_data = {
                    "map_id": area.map_id,
                    "location_name": area.location_name,
                    "player_last_position": area.player_last_position,
                    "warp_tiles": area.warp_tiles,
                    "boundaries": area.boundaries,
                    "visited_count": area.visited_count,
                    "first_seen": area.first_seen,
                    "last_seen": area.last_seen,
                    "overworld_coords": area.overworld_coords
                }
                print(f"🗺️ DEBUG: Saving area {map_id} with overworld_coords = {area.overworld_coords}")
                map_stitcher_data["map_areas"][str(map_id)] = area_data
            
            # Convert connections to serializable format
            for conn in self.warp_connections:
                map_stitcher_data["warp_connections"].append(asdict(conn))
            
            # Save location connections from state_formatter
            try:
                from utils import state_formatter
                if hasattr(state_formatter, 'LOCATION_CONNECTIONS'):
                    map_stitcher_data["location_connections"] = state_formatter.LOCATION_CONNECTIONS
                    logger.debug(f"Saved {len(state_formatter.LOCATION_CONNECTIONS)} location connections to checkpoint")
            except ImportError:
                logger.debug("Could not import state_formatter for location connections in checkpoint")
            
            checkpoint_data["map_stitcher"] = map_stitcher_data
            logger.debug(f"Saved {len(self.map_areas)} areas and {len(self.warp_connections)} connections to checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to save map stitcher to checkpoint: {e}")
    
    def load_from_checkpoint(self, checkpoint_data: dict):
        """Load map stitching data from checkpoint data structure"""
        try:
            map_stitcher_data = checkpoint_data.get("map_stitcher")
            if not map_stitcher_data:
                return
            
            # Clear existing data
            self.map_areas.clear()
            self.warp_connections.clear()
            
            # Restore map areas (without map_data)
            for map_id_str, area_data in map_stitcher_data.get("map_areas", {}).items():
                map_id = int(map_id_str)
                area = MapArea(
                    map_id=area_data["map_id"],
                    location_name=area_data["location_name"],
                    map_data=[],  # Will be populated when area is revisited
                    player_last_position=tuple(area_data["player_last_position"]),
                    warp_tiles=[tuple(wt) for wt in area_data["warp_tiles"]],
                    boundaries=area_data["boundaries"],
                    visited_count=area_data["visited_count"],
                    first_seen=area_data["first_seen"],
                    last_seen=area_data["last_seen"],
                    overworld_coords=tuple(area_data["overworld_coords"]) if area_data.get("overworld_coords") else None
                )
                self.map_areas[map_id] = area
            
            # Restore connections
            for conn_data in map_stitcher_data.get("warp_connections", []):
                conn = WarpConnection(
                    from_map_id=conn_data["from_map_id"],
                    to_map_id=conn_data["to_map_id"],
                    from_position=tuple(conn_data["from_position"]),
                    to_position=tuple(conn_data["to_position"]),
                    warp_type=conn_data["warp_type"],
                    direction=conn_data["direction"]
                )
                self.warp_connections.append(conn)
            
            # Restore location connections to state_formatter
            location_connections = map_stitcher_data.get("location_connections", {})
            if location_connections:
                try:
                    from utils import state_formatter
                    state_formatter.LOCATION_CONNECTIONS = location_connections
                    logger.info(f"Loaded {len(location_connections)} location connections from checkpoint")
                except ImportError:
                    logger.debug("Could not import state_formatter for location connections from checkpoint")
            
            logger.info(f"Loaded {len(self.map_areas)} areas and {len(self.warp_connections)} connections from checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to load map stitcher from checkpoint: {e}")
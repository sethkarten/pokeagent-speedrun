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
from utils import state_formatter

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
    
    def _merge_map_tiles(self, area: MapArea, new_tiles: List[List[Tuple]], player_pos: Tuple[int, int]):
        """Merge new tiles into existing map data, building up complete map over time.
        
        This is the core stitching logic - it takes the new 15x15 view around
        the player and merges it into the accumulated map data for this area.
        """
        if not new_tiles:
            return
            
        # Get dimensions of new tile data (usually 15x15)
        new_height = len(new_tiles)
        new_width = len(new_tiles[0]) if new_tiles else 0
        
        if new_height == 0 or new_width == 0:
            return
        
        # Calculate the offset of the new tiles relative to player position
        # The player is at the center of the new tiles
        center_y = new_height // 2
        center_x = new_width // 2
        
        # If this is the first data for this area, initialize with a large empty grid
        if area.map_data is None or not area.map_data:
            # Create a 100x100 grid initially (can expand as needed)
            # Use None to indicate unexplored tiles
            area.map_data = [[None for _ in range(100)] for _ in range(100)]
            # Track the actual bounds of explored area
            area.explored_bounds = {
                'min_x': 50, 'max_x': 50,
                'min_y': 50, 'max_y': 50
            }
            # Place player at center of our coordinate system initially
            area.origin_offset = {'x': 50 - player_pos[0], 'y': 50 - player_pos[1]}
            
        # Ensure origin_offset exists
        if not hasattr(area, 'origin_offset'):
            area.origin_offset = {'x': 50 - player_pos[0], 'y': 50 - player_pos[1]}
        
        # Get the offset to map player coordinates to our stored grid
        offset_x = area.origin_offset.get('x', 0)
        offset_y = area.origin_offset.get('y', 0)
        
        # CRITICAL: Check for unreasonable coordinate jumps that indicate a map transition error
        # If the player position would require massive grid expansion, it's likely a different map
        grid_center_x = player_pos[0] + offset_x
        grid_center_y = player_pos[1] + offset_y
        
        MAX_REASONABLE_SIZE = 200  # Maximum reasonable size for a single map area
        
        # Check if this would cause unreasonable expansion
        if (grid_center_x < -50 or grid_center_x > MAX_REASONABLE_SIZE + 50 or
            grid_center_y < -50 or grid_center_y > MAX_REASONABLE_SIZE + 50):
            logger.warning(f"Detected unreasonable coordinate jump for map {area.map_id:04X}: "
                         f"player at {player_pos}, grid position would be ({grid_center_x}, {grid_center_y})")
            logger.warning(f"This likely indicates map areas are being incorrectly merged. "
                         f"Resetting origin offset for this area.")
            
            # Reset the map data for this area to prevent corruption
            area.map_data = [[None for _ in range(100)] for _ in range(100)]
            area.explored_bounds = {
                'min_x': 50, 'max_x': 50,
                'min_y': 50, 'max_y': 50
            }
            area.origin_offset = {'x': 50 - player_pos[0], 'y': 50 - player_pos[1]}
            offset_x = area.origin_offset['x']
            offset_y = area.origin_offset['y']
        
        # Merge the new tiles into the existing map
        for dy in range(new_height):
            for dx in range(new_width):
                # Calculate the world position of this tile
                world_x = player_pos[0] - center_x + dx
                world_y = player_pos[1] - center_y + dy
                
                # Calculate the position in our stored grid
                grid_x = world_x + offset_x
                grid_y = world_y + offset_y
                
                # Sanity check to prevent excessive memory usage
                if grid_x < 0 or grid_y < 0 or grid_x >= MAX_REASONABLE_SIZE or grid_y >= MAX_REASONABLE_SIZE:
                    logger.debug(f"Skipping tile at grid position ({grid_x}, {grid_y}) - out of reasonable bounds")
                    continue
                
                # Expand grid if necessary (but within reasonable limits)
                if grid_y >= len(area.map_data) and grid_y < MAX_REASONABLE_SIZE:
                    # Expand vertically
                    expansion_needed = min(grid_y - len(area.map_data) + 1, 
                                         MAX_REASONABLE_SIZE - len(area.map_data))
                    for _ in range(expansion_needed):
                        area.map_data.append([None] * len(area.map_data[0]))
                
                if grid_x >= len(area.map_data[0]) and grid_x < MAX_REASONABLE_SIZE:
                    # Expand horizontally
                    new_width_needed = min(grid_x + 1, MAX_REASONABLE_SIZE)
                    for row in area.map_data:
                        expansion = new_width_needed - len(row)
                        if expansion > 0:
                            row.extend([None] * expansion)
                
                # Store the tile (always update with latest data)
                if 0 <= grid_x < len(area.map_data[0]) and 0 <= grid_y < len(area.map_data):
                    tile = new_tiles[dy][dx]
                    # Store all tiles including 1023 (which represents walls/boundaries)
                    # The display logic will handle showing them correctly
                    if tile:
                        area.map_data[grid_y][grid_x] = tile
                        
                        # Update explored bounds for all tiles including boundaries
                        # tile_id 1023 represents trees/walls at map edges - we want to include these
                        tile_id = tile[0] if tile and len(tile) > 0 else None
                        if tile_id is not None:  # Include all tiles, even 1023
                            if not hasattr(area, 'explored_bounds'):
                                area.explored_bounds = {
                                    'min_x': grid_x, 'max_x': grid_x,
                                    'min_y': grid_y, 'max_y': grid_y
                                }
                            else:
                                area.explored_bounds['min_x'] = min(area.explored_bounds['min_x'], grid_x)
                                area.explored_bounds['max_x'] = max(area.explored_bounds['max_x'], grid_x)
                                area.explored_bounds['min_y'] = min(area.explored_bounds['min_y'], grid_y)
                                area.explored_bounds['max_y'] = max(area.explored_bounds['max_y'], grid_y)
    
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
        
        # Validate map ID is reasonable
        if map_id < 0 or map_id > 0xFFFF:
            logger.error(f"Invalid map ID {map_id} from bank={map_bank}, number={map_number}")
            return
        
        # Validate player position - check for invalid values
        if player_pos:
            px, py = player_pos
            # Check for invalid coordinates (65535 = 0xFFFF is a common error value)
            if px < 0 or px > 1000 or py < 0 or py > 1000 or px == 0xFFFF or py == 0xFFFF:
                logger.warning(f"Invalid player position {player_pos} for map {map_id:04X}, ignoring update")
                return
        
        if map_id in self.map_areas:
            # Update existing area - we're revisiting this location
            area = self.map_areas[map_id]
            logger.info(f"Revisiting existing map area {area.location_name} (ID: {map_id:04X})")
            area.visited_count = getattr(area, 'visited_count', 0) + 1
            # Update location name if we have a better one (not empty or "Unknown")
            if location_name and location_name.strip() and location_name != "Unknown":
                if area.location_name == "Unknown" or not area.location_name:
                    logger.info(f"Updating location name for map {map_id:04X}: '{area.location_name}' -> '{location_name}'")
                    area.location_name = location_name
                    # Try to resolve other unknown names since we got new location info
                    self.resolve_unknown_location_names()
                elif area.location_name != location_name:
                    # Check if this is a significant name difference that might indicate a problem
                    name1_words = set(area.location_name.lower().split())
                    name2_words = set(location_name.lower().split())
                    
                    # If the names share no common words, this might be a misidentified map
                    if not name1_words.intersection(name2_words):
                        logger.warning(f"Significant location name mismatch for map {map_id:04X}: "
                                     f"existing='{area.location_name}' vs new='{location_name}'. "
                                     f"This might indicate incorrect map identification.")
                    else:
                        logger.info(f"Found different location name for map {map_id:04X}: '{area.location_name}' vs '{location_name}', keeping current")
                else:
                    area.location_name = location_name
            
            # MERGE map data instead of replacing - this is the key to stitching!
            if map_data and player_pos:
                # When revisiting, check if player position makes sense with existing map
                if hasattr(area, 'origin_offset') and area.origin_offset:
                    expected_grid_x = player_pos[0] + area.origin_offset['x']
                    expected_grid_y = player_pos[1] + area.origin_offset['y']
                    
                    # Check if player position is reasonable for this map
                    if (0 <= expected_grid_x <= 200 and 0 <= expected_grid_y <= 200):
                        # Position is reasonable - merge tiles
                        self._merge_map_tiles(area, map_data, player_pos)
                        logger.debug(f"Merged {len(map_data) * len(map_data[0]) if map_data else 0} new tiles into area")
                    else:
                        logger.warning(f"Player position {player_pos} seems incorrect for map {map_id:04X} "
                                     f"(would be at grid {expected_grid_x},{expected_grid_y})")
                else:
                    # First visit to this area after loading - merge normally
                    self._merge_map_tiles(area, map_data, player_pos)
                    logger.debug(f"Merged {len(map_data) * len(map_data[0]) if map_data else 0} new tiles into area")
            
            area.player_last_position = player_pos
            area.last_seen = timestamp
            # Remove deprecated fields - keep it simple
            logger.debug(f"Updated map area {area.location_name} (ID: {map_id:04X})")
        else:
            # Create new area
            # Try to resolve location name from map ID if empty
            if not location_name or not location_name.strip():
                # Import and use the location mapping
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
                map_data=None,  # Start with empty data - will be populated by merge
                player_last_position=player_pos,
                warp_tiles=[],  # Deprecated - not needed
                boundaries={"north": 0, "south": 10, "west": 0, "east": 10},  # Simple default
                visited_count=1,
                first_seen=timestamp,
                last_seen=timestamp,
                overworld_coords=None  # Not needed
            )
            self.map_areas[map_id] = area
            
            # Now merge the initial tiles
            if map_data and player_pos:
                self._merge_map_tiles(area, map_data, player_pos)
                logger.debug(f"Initialized new area with {len(map_data) * len(map_data[0]) if map_data else 0} tiles")
            logger.info(f"Added new map area: {final_location_name} (ID: {map_id:04X}) as separate location")
            
        # Check for area transitions and potential warp connections
        # print(f"🔍 Transition check: last_map_id={self.last_map_id}, current_map_id={map_id}, last_pos={self.last_position}, current_pos={player_pos}")
        if self.last_map_id is not None and self.last_map_id != map_id:
            logger.info(f"🔄 Map transition detected! {self.last_map_id} -> {map_id}")
            
            # Use the last position stored in the previous map area for the from_pos
            # This is the actual exit point from the previous map
            from_area = self.map_areas.get(self.last_map_id)
            from_pos = from_area.player_last_position if from_area else self.last_position
            
            logger.info(f"🔄 Warp coordinates: from_pos={from_pos} (exit from map {self.last_map_id}), to_pos={player_pos} (entry to map {map_id})")
            self._detect_warp_connection(self.last_map_id, map_id, 
                                       from_pos, player_pos, timestamp)
            
            # Try to resolve any unknown location names after adding connections  
            # Note: resolve_unknown_location_names() can be called with memory_reader from calling code
            if self.resolve_unknown_location_names():
                logger.info("Resolved unknown location names after area transition")
                # Save will be handled by the calling code
        
        # Update tracking variables for next iteration
        if self.last_position != player_pos:
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
        print(f"🔄 Creating warp connection: {from_pos} -> {to_pos} (maps {from_map_id} -> {to_map_id})")
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
            print(f"Added warp connection: {from_area.location_name} -> {to_area.location_name} "
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
    
    def get_player_position_for_location(self, location_name: str) -> Optional[Tuple[int, int]]:
        """Get the last known player position for a specific location.
        
        Returns:
            Tuple of (x, y) coordinates or None if not found or invalid
        """
        # Find the map area with this location name
        for area in self.map_areas.values():
            if area.location_name and location_name and area.location_name.lower() == location_name.lower():
                if hasattr(area, 'player_last_position') and area.player_last_position:
                    px, py = area.player_last_position
                    # Validate the position
                    if px >= 0 and px < 1000 and py >= 0 and py < 1000 and px != 0xFFFF and py != 0xFFFF:
                        return (px, py)
                break
        return None
    
    def get_location_connections(self, location_name=None):
        """Get connections for a specific location or all locations.
        
        Args:
            location_name: Optional location name to get connections for.
                          If None, returns all location connections.
        
        Returns:
            If location_name provided: List of (to_location, from_coords, to_coords) tuples
            Otherwise: Dict mapping location names to connection lists
        """
        location_connections = {}
        
        # Process each warp connection
        for conn in self.warp_connections:
            from_area = self.map_areas.get(conn.from_map_id)
            to_area = self.map_areas.get(conn.to_map_id)
            
            if from_area and to_area:
                from_location = from_area.location_name
                to_location = to_area.location_name
                
                # Add forward connection
                if from_location not in location_connections:
                    location_connections[from_location] = []
                
                # Check if connection already exists
                exists = False
                for existing in location_connections[from_location]:
                    if existing[0] == to_location:
                        exists = True
                        break
                
                if not exists:
                    # Use the actual last positions from map areas, not the warp spawn point
                    # This gives more useful information about where transitions happen
                    from_pos = list(conn.from_position) if conn.from_position else [1, 1]
                    to_pos = list(to_area.player_last_position) if to_area.player_last_position else list(conn.to_position)
                    
                    location_connections[from_location].append([
                        to_location,
                        from_pos,
                        to_pos
                    ])
        
        # If specific location requested, return just its connections (case-insensitive)
        if location_name:
            # Try to find the location with case-insensitive matching
            for loc_name, connections in location_connections.items():
                if loc_name and loc_name.lower() == location_name.lower():
                    return connections
            return []
        
        return location_connections
    
    def get_location_grid(self, location_name: str, simplified: bool = True) -> Dict[Tuple[int, int], str]:
        """Get a simplified grid representation of a location for display.
        
        Args:
            location_name: Name of the location to get grid for
            simplified: If True, return simplified symbols (., #, D, etc.), otherwise raw tile data
            
        Returns:
            Dictionary mapping (x, y) coordinates to tile symbols
        """
        # Find the map area with this location name (case-insensitive)
        map_area = None
        for area in self.map_areas.values():
            if area.location_name and location_name and area.location_name.lower() == location_name.lower():
                map_area = area
                break
        
        if not map_area:
            # Debug: print available locations
            logger.debug(f"Could not find map area for '{location_name}'")
            logger.debug(f"Available locations: {[a.location_name for a in self.map_areas.values() if a.location_name][:5]}")
            return {}
        
        if not map_area.map_data:
            logger.debug(f"Map area found for '{location_name}' but has no map_data")
            return {}
        
        grid = {}
        
        # If we have explored bounds, use them to extract only the explored portion
        if hasattr(map_area, 'explored_bounds'):
            bounds = map_area.explored_bounds
            for y in range(bounds['min_y'], bounds['max_y'] + 1):
                for x in range(bounds['min_x'], bounds['max_x'] + 1):
                    if y < len(map_area.map_data) and x < len(map_area.map_data[0]):
                        tile = map_area.map_data[y][x]
                        if tile is not None:  # Only include explored tiles
                            # Adjust coordinates to be relative to the explored area
                            rel_x = x - bounds['min_x']
                            rel_y = y - bounds['min_y']
                            
                            if simplified:
                                # Convert to simplified symbol
                                symbol = self._tile_to_symbol(tile)
                                if symbol is not None:  # Only add if it's a valid tile
                                    # Debug specific problematic position
                                    if rel_x == 2 and rel_y == 1:
                                        logger.debug(f"Tile at rel(2,1) from grid[{y}][{x}]: {tile[:3] if len(tile) >= 3 else tile} -> symbol '{symbol}'")
                                    grid[(rel_x, rel_y)] = symbol
                            else:
                                grid[(rel_x, rel_y)] = tile
            
            # Add '?' for unexplored but adjacent tiles
            if simplified:
                # Find all positions adjacent to explored walkable tiles
                to_check = set()
                for (x, y), symbol in list(grid.items()):
                    # Only add ? next to truly walkable tiles, not walls
                    if symbol in ['.', 'D', 'S', '^', '~', 's', 'I',  # Walkable terrain
                                  '→', '←', '↑', '↓', '↗', '↖', '↘', '↙']:  # Ledges
                        # Check all 4 adjacent positions (not diagonal)
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            adj_pos = (x + dx, y + dy)
                            if adj_pos not in grid:
                                to_check.add(adj_pos)
                
                # Add '?' for these unexplored adjacent positions
                for pos in to_check:
                    grid[pos] = '?'
            
            return grid
        
        # Fallback: old logic for non-accumulated maps
        # Check if we should extract a focused area from the stored map
        extract_bounds = getattr(map_area, '_display_extract_bounds', None)
        if extract_bounds:
            extract_start_x, extract_start_y, display_size = extract_bounds
            # Extract only the specified area
            for y in range(display_size):
                for x in range(display_size):
                    stored_y = extract_start_y + y
                    stored_x = extract_start_x + x
                    if (stored_y < len(map_area.map_data) and 
                        stored_x < len(map_area.map_data[stored_y])):
                        tile = map_area.map_data[stored_y][stored_x]
                        if tile and len(tile) >= 3:
                            tile_id, behavior, collision = tile[:3]
                            
                            if simplified:
                                # Use the centralized tile_to_symbol function
                                symbol = self._tile_to_symbol(tile)
                                if symbol is not None:  # Only add if it's a valid tile
                                    grid[(x, y)] = symbol
                            else:
                                # Return raw tile data
                                grid[(x, y)] = tile
        else:
            # Use full stored map (fallback for old behavior)
            for y, row in enumerate(map_area.map_data):
                for x, tile in enumerate(row):
                    if tile and len(tile) >= 3:
                        tile_id, behavior, collision = tile[:3]
                        
                        if simplified:
                            # Use the centralized tile_to_symbol function
                            symbol = self._tile_to_symbol(tile)
                            if symbol is not None:  # Only add if it's a valid tile
                                grid[(x, y)] = symbol
                        else:
                            # Return raw tile data
                            grid[(x, y)] = tile
        
        return grid
    
    def get_all_location_grids(self, simplified: bool = True) -> Dict[str, Dict[Tuple[int, int], str]]:
        """Get grids for all known locations.
        
        Returns:
            Dictionary mapping location names to their grids
        """
        all_grids = {}
        for area in self.map_areas.values():
            if area.location_name and area.map_data:
                all_grids[area.location_name] = self.get_location_grid(area.location_name, simplified)
        return all_grids
    
    def save_to_file(self):
        """Save stitching data to JSON file"""
        try:
            data = {
                "map_areas": {},
                "location_connections": {}
            }
            
            # Convert map areas to serializable format
            for map_id, area in self.map_areas.items():
                # Trim null rows from map_data before saving
                if area.map_data:
                    trimmed_map_data, trim_offsets = self._trim_null_rows(area.map_data)
                else:
                    trimmed_map_data, trim_offsets = [], {}
                
                # Save only essential data
                area_data = {
                    "map_id": area.map_id,
                    "location_name": area.location_name,
                    "map_data": trimmed_map_data,
                    "player_last_position": area.player_last_position
                }
                
                # Save trim offsets if we trimmed the data
                if trim_offsets:
                    area_data["trim_offsets"] = trim_offsets
                
                # Save additional attributes for map stitching
                if hasattr(area, 'explored_bounds'):
                    area_data["explored_bounds"] = area.explored_bounds
                if hasattr(area, 'origin_offset'):
                    area_data["origin_offset"] = area.origin_offset
                data["map_areas"][str(map_id)] = area_data
            
            # Generate location_connections from warp_connections
            # MapStitcher is the single source of truth for connections
            data["location_connections"] = self.get_location_connections()
            logger.debug(f"Saved {len(data['location_connections'])} location connections from {len(self.warp_connections)} warp connections")
            
            with open(self.save_file, 'w') as f:
                # Save in minified format to reduce file size
                json.dump(data, f, separators=(',', ':'))
            
            logger.debug(f"Saved map stitching data to {self.save_file}")
            
        except Exception as e:
            logger.error(f"Failed to save map stitching data: {e}")
    
    def load_from_file(self):
        """Load stitching data from JSON file"""
        if not self.save_file.exists():
            return
        
        # Check if file is empty
        if self.save_file.stat().st_size == 0:
            logger.debug(f"Map stitcher file {self.save_file} is empty, starting fresh")
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
                    try:
                        map_enum = MapLocation(map_id)
                        location_name = map_enum.name.replace('_', ' ').title()
                        logger.info(f"Resolved location name for map {map_id:04X} during load: {location_name}")
                    except ValueError:
                        # Fallback for unknown map IDs
                        location_name = f"Map_{map_id:04X}"
                        logger.debug(f"Unknown map ID {map_id:04X} during load, using fallback name")
                
                # Reconstruct full map data from trimmed version
                trimmed_data = area_data.get("map_data", [])
                trim_offsets = area_data.get("trim_offsets", {})
                
                if trim_offsets and trim_offsets.get('compacted'):
                    # New compacted format - reconstruct from tile list
                    row_offset = trim_offsets.get('row_offset', 0)
                    col_offset = trim_offsets.get('col_offset', 0)
                    original_height = trim_offsets.get('original_height', 100)
                    original_width = trim_offsets.get('original_width', 100)
                    
                    # Create full-sized map data array
                    full_map_data = [[None for _ in range(original_width)] for _ in range(original_height)]
                    
                    # Restore tiles from compacted format
                    if isinstance(trimmed_data, list):
                        # New list format: [[rel_row, rel_col, tile], ...]
                        for item in trimmed_data:
                            if len(item) >= 3:
                                rel_row, rel_col, tile = item[0], item[1], item[2]
                                actual_row = row_offset + rel_row
                                actual_col = col_offset + rel_col
                                if actual_row < original_height and actual_col < original_width:
                                    full_map_data[actual_row][actual_col] = tile
                    elif isinstance(trimmed_data, dict) and 'tiles' in trimmed_data:
                        # Old dict format (backward compatibility)
                        for pos_key, tile in trimmed_data['tiles'].items():
                            rel_row, rel_col = map(int, pos_key.split(','))
                            actual_row = row_offset + rel_row
                            actual_col = col_offset + rel_col
                            if actual_row < original_height and actual_col < original_width:
                                full_map_data[actual_row][actual_col] = tile
                    
                    map_data = full_map_data
                elif trimmed_data and trim_offsets:
                    # Old trimmed format (backward compatibility)
                    row_offset = trim_offsets.get('row_offset', 0)
                    col_offset = trim_offsets.get('col_offset', 0)
                    original_height = trim_offsets.get('original_height', len(trimmed_data) + row_offset)
                    original_width = trim_offsets.get('original_width', 100)
                    
                    # Create full-sized map data array
                    full_map_data = [[None for _ in range(original_width)] for _ in range(original_height)]
                    
                    # Place trimmed data back at correct position
                    for i, row in enumerate(trimmed_data):
                        for j, tile in enumerate(row):
                            if tile is not None:
                                full_map_data[row_offset + i][col_offset + j] = tile
                    
                    map_data = full_map_data
                else:
                    # No trim offsets, use data as-is (backward compatibility)
                    map_data = trimmed_data
                
                # Validate and clean player position when loading
                player_pos_data = area_data.get("player_last_position", [0, 0])
                if player_pos_data:
                    px, py = player_pos_data[0], player_pos_data[1] if len(player_pos_data) > 1 else 0
                    # Clean up invalid positions (65535 = 0xFFFF is an error value)
                    if px < 0 or px > 1000 or py < 0 or py > 1000 or px == 0xFFFF or py == 0xFFFF:
                        logger.warning(f"Cleaning invalid player position {player_pos_data} for map {map_id:04X}")
                        player_pos_data = [0, 0]  # Reset to origin
                else:
                    player_pos_data = [0, 0]
                
                area = MapArea(
                    map_id=area_data["map_id"],
                    location_name=location_name,
                    map_data=map_data,
                    player_last_position=tuple(player_pos_data),
                    warp_tiles=[],  # Deprecated - not needed
                    boundaries={"north": 0, "south": 10, "west": 0, "east": 10},  # Default boundaries
                    visited_count=1,  # Default
                    first_seen=0,  # Default
                    last_seen=0,  # Default
                    overworld_coords=None  # Not needed
                )
                # Restore additional stitching attributes if present
                if "explored_bounds" in area_data:
                    area.explored_bounds = area_data["explored_bounds"]
                    # When loading trimmed data, adjust explored_bounds to match
                    # Since we trimmed null rows/columns, the bounds are now relative to the trimmed data
                    if area.map_data:
                        # The trimmed data starts at (0,0), so adjust bounds accordingly
                        actual_height = len(area.map_data)
                        actual_width = max(len(row) for row in area.map_data) if area.map_data else 0
                        # Keep the existing explored_bounds as they track the original coordinate space
                        # The map_data is now compact but explored_bounds maintains the relationship
                else:
                    # Initialize explored bounds from map data if not present
                    if area.map_data:
                        min_x, max_x = 100, 0
                        min_y, max_y = 100, 0
                        for y, row in enumerate(area.map_data):
                            for x, tile in enumerate(row):
                                if tile is not None:
                                    min_x = min(min_x, x)
                                    max_x = max(max_x, x)
                                    min_y = min(min_y, y)
                                    max_y = max(max_y, y)
                        if min_x <= max_x:
                            area.explored_bounds = {
                                'min_x': min_x, 'max_x': max_x,
                                'min_y': min_y, 'max_y': max_y
                            }
                
                if "origin_offset" in area_data:
                    area.origin_offset = area_data["origin_offset"]
                else:
                    # Initialize origin offset based on player position
                    if area.player_last_position:
                        # Assume player was at center of initial explored area
                        area.origin_offset = {'x': 50 - area.player_last_position[0], 
                                             'y': 50 - area.player_last_position[1]}
                self.map_areas[map_id] = area
                # Debug: log if map_data was loaded
                if area.map_data:
                    logger.debug(f"Loaded map_data for {location_name}: {len(area.map_data)}x{len(area.map_data[0]) if area.map_data else 0}")
            
            # Reconstruct warp_connections from location_connections
            location_connections = data.get("location_connections", {})
            
            # Clear existing warp connections to avoid duplicates
            self.warp_connections = []
            
            # Convert location_connections back to warp_connections
            for from_location, connections in location_connections.items():
                # Find the map_id for this location
                from_map_id = None
                for map_id, area in self.map_areas.items():
                    if area.location_name == from_location:
                        from_map_id = map_id
                        break
                
                if from_map_id is None:
                    continue
                
                for conn_data in connections:
                    to_location = conn_data[0]
                    from_pos = tuple(conn_data[1]) if len(conn_data) > 1 else (0, 0)
                    to_pos = tuple(conn_data[2]) if len(conn_data) > 2 else (0, 0)
                    
                    # Find the map_id for the destination
                    to_map_id = None
                    for map_id, area in self.map_areas.items():
                        if area.location_name == to_location:
                            to_map_id = map_id
                            break
                    
                    if to_map_id is None:
                        continue
                    
                    # Create warp connection
                    warp_conn = WarpConnection(
                        from_map_id=from_map_id,
                        to_map_id=to_map_id,
                        from_position=from_pos,
                        to_position=to_pos,
                        warp_type="stairs",  # Default type
                        direction=None
                    )
                    self.warp_connections.append(warp_conn)
            
            logger.info(f"Reconstructed {len(self.warp_connections)} warp connections from {len(location_connections)} location connections")
            
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
    
    def _should_trim_edge(self, tiles, is_row=True):
        """Check if an edge (row or column) should be trimmed.
        An edge should be trimmed if it's all walls (#) with no meaningful content."""
        # Count non-wall tiles
        non_wall_count = 0
        for tile in tiles:
            if tile and tile not in ['#', ' ', None]:
                non_wall_count += 1
        # Trim if it's all walls or mostly walls with no content
        return non_wall_count == 0
    
    def _trim_null_rows(self, map_data: List[List]) -> Tuple[List[List], Dict[str, int]]:
        """Trim rows that are entirely null/None from map data to reduce file size.
        
        Returns a tuple of (trimmed_data, trim_offsets) where trim_offsets contains
        the offsets needed to reconstruct original positions.
        """
        if not map_data:
            return [], {}
        
        # Find bounds of actual data
        start_row = None
        end_row = None
        start_col = None
        end_col = None
        
        # Find row bounds
        for i, row in enumerate(map_data):
            if row and any(tile is not None for tile in row):
                if start_row is None:
                    start_row = i
                end_row = i
        
        if start_row is None:
            # All data is null
            return [], {}
        
        # Find column bounds across all rows
        for row in map_data[start_row:end_row + 1]:
            if row:
                for j, tile in enumerate(row):
                    if tile is not None:
                        if start_col is None or j < start_col:
                            start_col = j
                        if end_col is None or j > end_col:
                            end_col = j
        
        if start_col is None:
            return [], {}
        
        # Create compacted data - use a list of [row, col, tile] to save space
        # This eliminates ALL null-only rows while preserving position information
        tiles_list = []
        
        # Store only non-null tiles with their positions
        for i in range(start_row, end_row + 1):
            if map_data[i]:
                for j in range(start_col, end_col + 1):
                    if j < len(map_data[i]) and map_data[i][j] is not None:
                        # Store as [relative_row, relative_col, tile_data]
                        # This is more compact than dict with string keys
                        rel_row = i - start_row
                        rel_col = j - start_col
                        tiles_list.append([rel_row, rel_col, map_data[i][j]])
        
        trim_offsets = {
            'row_offset': start_row,
            'col_offset': start_col,
            'original_height': len(map_data),
            'original_width': max(len(row) for row in map_data) if map_data else 0,
            'compacted': True  # Flag to indicate new format
        }
        
        return tiles_list, trim_offsets
    
    def generate_location_map_display(self, location_name: str, player_pos: Tuple[int, int] = None, 
                                      npcs: List[Dict] = None, connections: List[Dict] = None) -> List[str]:
        """Generate a detailed map display for a specific location.
        
        Args:
            location_name: Name of the location to display
            player_pos: Current player position (x, y)
            npcs: List of NPC positions and data
            connections: List of location connections
            
        Returns:
            List of display lines ready for formatting
        """
        lines = []
        
        # Get stored map data for this location
        location_grid = self.get_location_grid(location_name, simplified=True)
        
        if not location_grid:
            # No map data available - return empty to trigger memory fallback
            return []
        
        # For accumulated maps, show the full explored area
        # Get the dimensions of the explored area
        max_x = max(x for x, y in location_grid.keys()) if location_grid else 0
        max_y = max(y for x, y in location_grid.keys()) if location_grid else 0
        min_x = min(x for x, y in location_grid.keys()) if location_grid else 0
        min_y = min(y for x, y in location_grid.keys()) if location_grid else 0
        
        explored_width = max_x - min_x + 1
        explored_height = max_y - min_y + 1
        
        # Show the full accumulated map (up to reasonable size)
        # Don't try to focus on player for accumulated maps
        if explored_width <= 30 and explored_height <= 30:
            # Show the entire accumulated map
            display_radius = max(explored_width, explored_height) // 2
            display_size = max(explored_width, explored_height)
        else:
            # Very large area, limit to 30x30 for readability
            display_radius = 15
            display_size = 30
        
        display_center = display_radius  # Player at center
        
        # For accumulated maps, just use the entire grid without focusing
        # This shows the full explored area
        all_positions = list(location_grid.keys())
        
        # Find player position in the grid if available
        local_player_pos = None
        if player_pos:
            # Validate player position first
            px, py = player_pos
            if px >= 0 and px < 1000 and py >= 0 and py < 1000 and px != 0xFFFF and py != 0xFFFF:
                # Find the stored map area to get coordinate conversion info
                map_area = None
                for area in self.map_areas.values():
                    if area.location_name and location_name and area.location_name.lower() == location_name.lower():
                        map_area = area
                        break
                
                if map_area:
                    # Use the stored player position from the map area if available
                    if hasattr(map_area, 'player_last_position') and map_area.player_last_position:
                        last_px, last_py = map_area.player_last_position
                        # Validate the stored position
                        if last_px >= 0 and last_px < 1000 and last_py >= 0 and last_py < 1000 and last_px != 0xFFFF and last_py != 0xFFFF:
                            player_pos = map_area.player_last_position
                            px, py = player_pos
                    
                    if hasattr(map_area, 'origin_offset') and map_area.origin_offset:
                        # Convert player world coordinates to grid-relative coordinates
                        offset_x = map_area.origin_offset.get('x', 0)
                        offset_y = map_area.origin_offset.get('y', 0)
                        
                        # Calculate player's position relative to the explored bounds
                        grid_player_x = px + offset_x
                        grid_player_y = py + offset_y
                        
                        # Convert to relative coordinates in the location_grid
                        if hasattr(map_area, 'explored_bounds'):
                            bounds = map_area.explored_bounds
                            rel_x = grid_player_x - bounds['min_x']
                            rel_y = grid_player_y - bounds['min_y']
                            
                            # Check if player is within the displayed area
                            if 0 <= rel_x <= (max_x - min_x) and 0 <= rel_y <= (max_y - min_y):
                                local_player_pos = (rel_x, rel_y)
                                logger.debug(f"Player at relative position {local_player_pos} in {location_name}")
                            else:
                                logger.debug(f"Player at {player_pos} is outside displayed area of {location_name}")
        
        if not all_positions:
            return []
        
        min_x = min(pos[0] for pos in all_positions)
        max_x = max(pos[0] for pos in all_positions)
        min_y = min(pos[1] for pos in all_positions)
        max_y = max(pos[1] for pos in all_positions)
        
        # Minimal trimming - only remove completely empty space
        # Don't trim '?' as those are unexplored areas we want to show
        # Don't aggressively trim walls as they show room boundaries
        
        # Only trim rows that are completely empty (all spaces/None)
        while min_y < max_y:
            row_tiles = [location_grid.get((x, min_y), ' ') for x in range(min_x, max_x + 1)]
            # Keep the row if it has ANY content (including ? and #)
            if any(t not in [' ', None] for t in row_tiles):
                break
            min_y += 1
        
        # Check bottom rows - only trim completely empty
        while max_y > min_y:
            row_tiles = [location_grid.get((x, max_y), ' ') for x in range(min_x, max_x + 1)]
            if any(t not in [' ', None] for t in row_tiles):
                break
            max_y -= 1
        
        # Check left columns - only trim completely empty
        while min_x < max_x:
            col_tiles = [location_grid.get((min_x, y), ' ') for y in range(min_y, max_y + 1)]
            if any(t not in [' ', None] for t in col_tiles):
                break
            min_x += 1
        
        # Check right columns - only trim completely empty
        while max_x > min_x:
            col_tiles = [location_grid.get((max_x, y), ' ') for y in range(min_y, max_y + 1)]
            if any(t not in [' ', None] for t in col_tiles):
                break
            max_x -= 1
        
        # Build portal positions from connections
        portal_positions = {}
        
        lines.append(f"\n--- MAP: {location_name.upper()} ---")
        
        # Create the map display
        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                # Check if this is an edge position
                is_edge = (x == min_x or x == max_x or y == min_y or y == max_y)
                
                # Check for NPCs at this position
                npc_at_pos = None
                if npcs:
                    for npc in npcs:
                        npc_x = npc.get('current_x', npc.get('x'))
                        npc_y = npc.get('current_y', npc.get('y'))
                        if npc_x == x and npc_y == y:
                            npc_at_pos = npc
                            break
                
                if local_player_pos and (x, y) == local_player_pos:
                    row += "P"
                elif npc_at_pos:
                    # Use simple N symbol for all NPCs
                    row += "N"
                elif (x, y) in location_grid:
                    tile = location_grid[(x, y)]
                    # Check for portal markers at edges
                    if is_edge and tile == '.' and connections:
                        portal_added = False
                        for conn in connections:
                            direction = conn.get('direction', '').lower()
                            conn_name = conn.get('name', '')
                            if direction and conn_name and conn_name not in ['Unknown', 'None', '']:
                                if direction == 'east' and x == max_x:
                                    row += "→"
                                    portal_positions[(x, y)] = conn_name
                                    portal_added = True
                                    break
                                elif direction == 'west' and x == min_x:
                                    row += "←"
                                    portal_positions[(x, y)] = conn_name
                                    portal_added = True
                                    break
                                elif direction == 'north' and y == min_y:
                                    row += "↑"
                                    portal_positions[(x, y)] = conn_name
                                    portal_added = True
                                    break
                                elif direction == 'south' and y == max_y:
                                    row += "↓"
                                    portal_positions[(x, y)] = conn_name
                                    portal_added = True
                                    break
                        
                        if not portal_added:
                            row += tile
                    else:
                        row += tile
                else:
                    # Position not in grid - just show as space
                    # The grid already has '?' symbols where needed from get_location_grid
                    row += " "
            
            # Add spacing between characters for square aspect ratio
            # Most terminals have characters ~2x taller than wide, so spacing helps
            spaced_row = " ".join(row)
            lines.append(spaced_row)
        
        # Add legend
        legend_lines = ["", "Legend:"]
        legend_lines.append("  Movement: P=Player")
        if npcs:
            legend_lines.append("            N=NPC/Trainer")
        
        # Check what terrain symbols are visible
        visible_symbols = set(location_grid.values())
        
        terrain_items = []
        symbol_meanings = {
            ".": ".=Walkable path",
            "#": "#=Wall/Blocked",
            "~": "~=Tall grass",
            "^": "^=Grass",
            "W": "W=Water",
            "I": "I=Ice",
            "s": "s=Sand",
            "D": "D=Door",
            "S": "S=Stairs/Ladder",
            "C": "C=Computer/PC",
            "→": "→=Ledge (jump east)",
            "←": "←=Ledge (jump west)",
            "↑": "↑=Ledge (jump north)",
            "↓": "↓=Ledge (jump south)",
            "↗": "↗=Ledge (jump NE)",
            "↖": "↖=Ledge (jump NW)",
            "↘": "↘=Ledge (jump SE)",
            "↙": "↙=Ledge (jump SW)",
            "L": "L=Ledge",
            "T": "T=TV",
            "?": "?=Unknown"
        }
        
        for symbol, meaning in symbol_meanings.items():
            if symbol in visible_symbols:
                terrain_items.append(meaning)
        
        if terrain_items:
            legend_lines.append(f"  Terrain: {', '.join(terrain_items)}")
        
        # Add portal markers to legend if any
        if portal_positions:
            unique_portals = {}
            for pos, dest in portal_positions.items():
                x, y = pos
                if x == min_x:
                    unique_portals["←"] = dest
                elif x == max_x:
                    unique_portals["→"] = dest
                elif y == min_y:
                    unique_portals["↑"] = dest
                elif y == max_y:
                    unique_portals["↓"] = dest
            
            if unique_portals:
                portal_items = []
                for symbol, dest in unique_portals.items():
                    portal_items.append(f"{symbol}={dest}")
                legend_lines.append(f"  Exits: {', '.join(portal_items)}")
        
        lines.extend(legend_lines)
        
        # Add explicit portal connections with coordinates
        if connections:
            lines.append("")
            lines.append("Portal Connections:")
            for conn in connections:
                to_location = conn.get('to', 'Unknown')
                from_pos = conn.get('from_pos', [])
                to_pos = conn.get('to_pos', [])
                
                if from_pos and to_pos and len(from_pos) >= 2 and len(to_pos) >= 2:
                    lines.append(f"  {location_name} ({from_pos[0]},{from_pos[1]}) → {to_location} ({to_pos[0]},{to_pos[1]})")
                elif from_pos and len(from_pos) >= 2:
                    lines.append(f"  {location_name} ({from_pos[0]},{from_pos[1]}) → {to_location}")
                else:
                    lines.append(f"  → {to_location}")
        
        return lines
    
    def _tile_to_symbol(self, tile) -> str:
        """Convert a tile tuple to a simplified symbol for display."""
        if tile is None:
            # This will be handled specially - unexplored areas next to walkable will show ?
            return None  # Mark as unexplored for special handling
        
        if len(tile) < 3:
            return None  # Invalid tile - unexplored
        
        tile_id, behavior, collision = tile[:3]
        
        # tile_id 1023 (0x3FF) means out-of-bounds/unloaded area
        # These are trees/boundaries at the edge of maps - show as walls
        if tile_id == 1023:
            return '#'  # Display as wall/blocked
        
        # Get behavior value
        if hasattr(behavior, 'value'):
            behavior_val = behavior.value
        else:
            behavior_val = behavior
        
        # Check behavior first for special terrain (even if impassable)
        # Grass types (from MetatileBehavior enum)
        if behavior_val == 2:  # TALL_GRASS
            return '~'  # Tall grass (encounters)
        elif behavior_val == 3:  # LONG_GRASS
            return '^'  # Long grass
        elif behavior_val == 7:  # SHORT_GRASS
            return '^'  # Short grass
        elif behavior_val == 36:  # ASHGRASS
            return '^'  # Ash grass
        
        # Water types
        elif behavior_val in [16, 17, 18, 19, 20, 21, 22, 23, 24, 26]:  # Various water types
            return 'W'  # Water
        
        # Ice
        elif behavior_val in [32, 38, 39]:  # ICE, THIN_ICE, CRACKED_ICE
            return 'I'  # Ice
        
        # Sand
        elif behavior_val in [6, 33]:  # DEEP_SAND, SAND
            return 's'  # Sand
        
        # Doors and warps
        elif behavior_val == 96:  # NON_ANIMATED_DOOR
            return 'D'  # Door
        elif behavior_val == 105:  # ANIMATED_DOOR
            return 'D'  # Door
        elif behavior_val in [98, 99, 100, 101]:  # Arrow warps
            return 'D'  # Warp/Door
        elif behavior_val == 97:  # LADDER
            return 'S'  # Stairs/Ladder
        elif behavior_val in [106, 107]:  # Escalators
            return 'S'  # Stairs
        
        # PC and other interactables
        elif behavior_val in [131, 197]:  # PC, PLAYER_ROOM_PC_ON
            return 'C'  # Computer/PC (changed from 'P' to avoid conflict with Player)
        elif behavior_val == 134:  # TELEVISION
            return 'T'  # TV
        
        # Ledges/Jumps with directional arrows
        elif behavior_val == 56:  # JUMP_EAST
            return '→'  # Ledge east
        elif behavior_val == 57:  # JUMP_WEST
            return '←'  # Ledge west
        elif behavior_val == 58:  # JUMP_NORTH
            return '↑'  # Ledge north
        elif behavior_val == 59:  # JUMP_SOUTH
            return '↓'  # Ledge south
        elif behavior_val == 60:  # JUMP_NORTHEAST
            return '↗'  # Ledge northeast
        elif behavior_val == 61:  # JUMP_NORTHWEST
            return '↖'  # Ledge northwest
        elif behavior_val == 62:  # JUMP_SOUTHEAST
            return '↘'  # Ledge southeast
        elif behavior_val == 63:  # JUMP_SOUTHWEST
            return '↙'  # Ledge southwest
        
        # Now check collision for basic terrain
        elif collision == 1:  # Impassable
            return '#'  # Wall
        elif collision == 0:  # Walkable
            return '.'  # Floor
        elif collision == 3:  # Ledge/special
            return 'L'  # Ledge
        elif collision == 4:  # Water/surf
            return 'W'  # Water
        else:
            return '?'  # Unknown
    
    def _is_explorable_edge(self, x: int, y: int, location_grid: Dict[Tuple[int, int], str]) -> bool:
        """Check if an unexplored coordinate is worth exploring (adjacent to walkable tiles)."""
        # Check all 4 adjacent tiles
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_x, adj_y = x + dx, y + dy
            if (adj_x, adj_y) in location_grid:
                tile = location_grid[(adj_x, adj_y)]
                # If adjacent to walkable tile, this is explorable
                # Include all walkable terrain types and ledges
                if tile in ['.', 'D', 'S', '^', '~', 's', 'I',  # Floor, doors, stairs, grass, sand, ice
                           '→', '←', '↑', '↓', '↗', '↖', '↘', '↙']:  # Ledges in all directions
                    return True
        return False
    
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
            # print( Saving area {map_id} with overworld_coords = {area.overworld_coords}")
                map_stitcher_data["map_areas"][str(map_id)] = area_data
            
            # Convert connections to serializable format
            for conn in self.warp_connections:
                map_stitcher_data["warp_connections"].append(asdict(conn))
            
            # Save location connections from state_formatter
            try:
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
                    state_formatter.LOCATION_CONNECTIONS = location_connections
                    logger.info(f"Loaded {len(location_connections)} location connections from checkpoint")
                except ImportError:
                    logger.debug("Could not import state_formatter for location connections from checkpoint")
            
            logger.info(f"Loaded {len(self.map_areas)} areas and {len(self.warp_connections)} connections from checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to load map stitcher from checkpoint: {e}")
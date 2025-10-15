"""
Agent utilities and classes for Pokemon Emerald AI

This module contains utility classes and functions for agent operations,
including NPC detection, visual analysis, and pathfinding capabilities.
"""

import logging
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from utils.vlm import VLM
import heapq

logger = logging.getLogger(__name__)

@dataclass
class DetectedNPC:
    """Represents a detected NPC from visual analysis"""
    x: int
    y: int
    description: str
    confidence: float

class NPCDetector:
    """
    Visual NPC detection system using VLM to analyze game frames
    and identify NPCs.
    """
    
    def __init__(self, vlm: VLM):
        """
        Initialize the NPC detector with a VLM instance.
        
        Args:
            vlm: VLM instance for visual analysis
        """
        self.vlm = vlm
        self.detection_prompt = self._create_detection_prompt()
        
    def _create_detection_prompt(self) -> str:
        """Create the prompt for NPC detection"""
        return """You are analyzing a Pokemon Emerald game frame to detect NPCs, trainers, and interactive objects.

Your task is to identify all visible NPCs (only) in the frame and provide their positions relative to the player (who is at the center of the 15x15 grid).

For each detected entity, provide:
1. Relative position from player (x, y offset where player is at 0,0)
2. Brief description of what you see
3. Confidence level (0.0 to 1.0)

Respond in JSON format:
{
    "npcs": [
        {
            "x": 2,
            "y": -1,
            "description": "Youngster with red hat",
            "confidence": 0.9
        }
    ]
}

Important guidelines:
- Player is at the center position (0, 0) in the center, don't include the player in the response
- Use relative coordinates (x, y) where positive x is right, positive y is down
- Only include entities that are clearly visible and identifiable
- If you're unsure about an entity, set confidence low or exclude it

Focus on the main game area, not UI elements or text boxes."""

    def detect_npcs(self, frame: Union[Image.Image, np.ndarray], player_coords: Optional[Tuple[int, int]] = None) -> List[DetectedNPC]:
        """
        Detect NPCs in a game frame using visual analysis.
        
        Args:
            frame: Game frame as PIL Image or numpy array
            player_coords: Player's absolute coordinates (optional, for logging)
            
        Returns:
            List of DetectedNPC objects
        """
        try:
            print(f"🔍 NPCDetector.detect_npcs called with player at {player_coords}")
            logger.info(f"Detecting NPCs in frame (player at {player_coords})")
            
            # Query the VLM for NPC detection
            response = self.vlm.get_query(
                img=frame,
                text=self.detection_prompt,
                module_name="NPCDetector"
            )
            print(f"🔍 VLM response received: {response[:100]}...")
            
            # Parse the JSON response
            try:
                # Extract JSON from response (handle cases where VLM adds extra text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    logger.warning("No valid JSON found in VLM response")
                    return []
                
                # Convert to DetectedNPC objects
                npcs = []
                for npc_data in data.get('npcs', []):
                    try:
                        npc = DetectedNPC(
                            x=npc_data.get('x', 0),
                            y=npc_data.get('y', 0),
                            description=npc_data.get('description', 'Unknown entity'),
                            confidence=npc_data.get('confidence', 0.0)
                        )
                        npcs.append(npc)
                    except Exception as e:
                        logger.warning(f"Failed to parse NPC data: {e}")
                        continue
                
                print(f"🔍 Successfully parsed {len(npcs)} NPCs from VLM response")
                logger.info(f"Detected {len(npcs)} NPCs")
                return npcs
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from VLM: {e}")
                logger.error(f"Response was: {response}")
                return []
                
        except Exception as e:
            logger.error(f"Error in NPC detection: {e}")
            return []
    
    def convert_to_absolute_coords(self, detected_npcs: List[DetectedNPC], player_coords: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Convert relative NPC positions to absolute world coordinates.
        
        Args:
            detected_npcs: List of detected NPCs with relative coordinates
            player_coords: Player's absolute coordinates (x, y)
            
        Returns:
            List of NPC dictionaries with absolute coordinates
        """
        player_x, player_y = player_coords
        npcs_with_abs_coords = []
        
        for npc in detected_npcs:
            abs_x = player_x + npc.x
            abs_y = player_y + npc.y
            
            npc_dict = {
                'current_x': abs_x,
                'current_y': abs_y,
                'description': npc.description,
                'confidence': npc.confidence
            }
            npcs_with_abs_coords.append(npc_dict)
        
        return npcs_with_abs_coords

class VisualMapGenerator:
    """
    Generates formatted map data using visual analysis of game frames.
    This is a stepping stone toward full visual map generation.
    """
    
    def __init__(self, vlm: VLM):
        """
        Initialize the visual map generator.
        
        Args:
            vlm: VLM instance for visual analysis
        """
        self.vlm = vlm
        self.npc_detector = NPCDetector(vlm)
        
    def enhance_map_with_visual_npcs(self, map_info: Dict[str, Any], frame: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Enhance map information with visually detected NPCs.
        
        Args:
            map_info: Current map information from memory
            frame: Current game frame
            
        Returns:
            Enhanced map information with detected NPCs
        """
        try:
            # Get player coordinates
            player_coords = map_info.get('player_coords')
            if not player_coords:
                logger.warning("No player coordinates available for NPC detection")
                return map_info
            
            # Detect NPCs in the frame
            detected_npcs = self.npc_detector.detect_npcs(frame, player_coords)
            
            if not detected_npcs:
                logger.info("No NPCs detected in current frame")
                return map_info
            
            # Convert to absolute coordinates
            npcs_with_abs_coords = self.npc_detector.convert_to_absolute_coords(
                detected_npcs, player_coords
            )
            
            # Merge with existing object events
            existing_npcs = map_info.get('object_events', [])
            
            # Debug logging
            logger.info(f"Existing NPCs: {len(existing_npcs)}")
            logger.info(f"Detected NPCs (converted): {len(npcs_with_abs_coords)}")
            for i, npc in enumerate(npcs_with_abs_coords):
                logger.info(f"  NPC {i+1}: {npc}")
            
            # Create a combined list, prioritizing detected NPCs for positions near player
            # and keeping existing NPCs for positions far from player
            combined_npcs = self._merge_npc_lists(existing_npcs, npcs_with_abs_coords, player_coords)
            
            # Update map info
            enhanced_map_info = map_info.copy()
            enhanced_map_info['object_events'] = combined_npcs
            enhanced_map_info['visual_npcs_detected'] = len(detected_npcs)
            
            # Override map tiles at NPC coordinates
            enhanced_map_info = self._override_map_tiles_with_npcs(enhanced_map_info, detected_npcs, player_coords)
            
            logger.info(f"Final combined NPCs: {len(combined_npcs)}")
            for i, npc in enumerate(combined_npcs):
                logger.info(f"  Final NPC {i+1}: {npc}")
            
            logger.info(f"Enhanced map with {len(combined_npcs)} total NPCs ({len(detected_npcs)} newly detected)")
            return enhanced_map_info
            
        except Exception as e:
            logger.error(f"Error enhancing map with visual NPCs: {e}")
            return map_info
    
    def _merge_npc_lists(self, existing_npcs: List[Dict], detected_npcs: List[Dict], player_coords: Tuple[int, int]) -> List[Dict]:
        """
        Merge existing NPCs with newly detected ones, avoiding duplicates.
        
        Args:
            existing_npcs: NPCs from memory/previous detection
            detected_npcs: Newly detected NPCs (already converted to dicts with absolute coords)
            player_coords: Player's current position
            
        Returns:
            Merged list of NPCs
        """
        player_x, player_y = player_coords
        merged = []
        
        # Add detected NPCs (these are near the player and most relevant)
        for npc in detected_npcs:
            merged.append(npc)
        
        # Add existing NPCs that are far from player (to maintain map continuity)
        for npc in existing_npcs:
            npc_x = npc.get('current_x', 0)
            npc_y = npc.get('current_y', 0)
            
            # Calculate distance from player
            distance = ((npc_x - player_x) ** 2 + (npc_y - player_y) ** 2) ** 0.5
            
            # Only keep NPCs that are far from player (outside the 15x15 grid)
            if distance > 8:  # 8 tiles is roughly half the 15x15 grid
                merged.append(npc)
        
        return merged
    
    def _override_map_tiles_with_npcs(self, map_info: Dict[str, Any], detected_npcs: List[DetectedNPC], player_coords: Tuple[int, int]) -> Dict[str, Any]:
        """
        Override map tiles at NPC coordinates to mark them as occupied.
        
        Args:
            map_info: Map information containing tiles
            detected_npcs: List of detected NPCs with relative coordinates
            player_coords: Player's absolute coordinates (x, y)
            
        Returns:
            Updated map information with overridden tiles
        """
        try:
            # Get the tiles from map info
            tiles = map_info.get('tiles')
            if not tiles:
                logger.warning("No tiles found in map info for NPC override")
                print(f"🔍 ❌ No tiles found in map info for NPC override")
                return map_info
            
            print(f"🔍 Tile override: Found tiles array with {len(tiles)} rows")
            player_x, player_y = player_coords
            overridden_count = 0
            
            # Process each detected NPC
            for npc in detected_npcs:
                # Calculate absolute coordinates
                abs_x = player_x + npc.x
                abs_y = player_y + npc.y
                
                # Calculate position on the 15x15 grid (player is at center 7,7)
                grid_x = npc.x + 7  # relative_x + center_offset
                grid_y = npc.y + 7  # relative_y + center_offset
                
                print(f"🔍 Processing NPC: relative=({npc.x}, {npc.y}), absolute=({abs_x}, {abs_y}), grid=({grid_x}, {grid_y})")
                
                # Check if position is within the 15x15 grid bounds
                if 0 <= grid_y < len(tiles) and 0 <= grid_x < len(tiles[grid_y]):
                    # Override the tile to mark it as occupied by an NPC
                    # We'll modify the tile to indicate an NPC is present
                    # The tile structure is: (metatile_id, behavior, ...)
                    original_tile = tiles[grid_y][grid_x]
                    print(f"🔍 ✓ NPC is within bounds. Original tile: {original_tile}")
                    
                    if len(original_tile) >= 2:
                        # Create a new tile tuple with NPC marker
                        # We'll use a special behavior value to indicate NPC presence
                        # Behavior 999 is used as a marker for visually detected NPCs
                        new_tile = (original_tile[0], 999, *original_tile[2:])
                        tiles[grid_y][grid_x] = new_tile
                        overridden_count += 1
                        print(f"🔍 ✓ Overrode tile at grid ({grid_x}, {grid_y}) - new tile: {new_tile}")
                        
                        logger.debug(f"Overrode tile at grid ({grid_x}, {grid_y}) for NPC at ({abs_x}, {abs_y})")
                    else:
                        print(f"🔍 ❌ Invalid tile structure at ({grid_x}, {grid_y}): {original_tile}")
                        logger.warning(f"Invalid tile structure at ({grid_x}, {grid_y}): {original_tile}")
                else:
                    print(f"🔍 ❌ NPC at ({abs_x}, {abs_y}) is outside 15x15 grid bounds (grid_y={grid_y}, grid_x={grid_x}, tiles.len={len(tiles)})")
                    logger.debug(f"NPC at ({abs_x}, {abs_y}) is outside 15x15 grid bounds")
            
            if overridden_count > 0:
                print(f"🔍 ✓✓✓ Successfully overrode {overridden_count} map tiles with NPC markers")
                logger.info(f"Overrode {overridden_count} map tiles with NPC markers")
                # Update the tiles in the map info
                map_info['tiles'] = tiles
                map_info['npc_tiles_overridden'] = overridden_count
            else:
                print(f"🔍 ⚠️ No tiles were overridden (overridden_count=0)")
            
            return map_info
            
        except Exception as e:
            logger.error(f"Error overriding map tiles with NPCs: {e}")
            return map_info


class PathfinderUtility:
    """
    A* pathfinding utility for navigating the 15x15 map grid.
    Helps the agent find optimal paths around obstacles and NPCs.
    """
    
    def __init__(self, max_path_length: int = 5):
        """
        Initialize the pathfinder.
        
        Args:
            max_path_length: Maximum number of moves to return (default 5)
        """
        self.grid_size = 15
        self.player_center = (7, 7)  # Player is always at center of 15x15 grid
        self.max_path_length = max_path_length
        
    def find_path(self, game_state: Dict[str, Any], target_grid_coords: Tuple[int, int]) -> Optional[List[str]]:
        """
        Find path from player to target using A* search.
        
        Args:
            game_state: Full game state with map tiles and NPCs
            target_grid_coords: Target position on 15x15 grid (0-14, 0-14)
            
        Returns:
            List of movement actions (UP/DOWN/LEFT/RIGHT) or None if no path found
        """
        try:
            # Validate target coordinates
            target_x, target_y = target_grid_coords
            logger.info(f"🗺️ Pathfinder: Searching for path from {self.player_center} to {target_grid_coords}")
            
            if not (0 <= target_x < self.grid_size and 0 <= target_y < self.grid_size):
                logger.warning(f"Target coordinates {target_grid_coords} out of bounds (0-{self.grid_size-1})")
                return None
            
            # Build passability grid from game state
            passability_grid = self._build_passability_grid(game_state)
            if passability_grid is None:
                logger.warning("Failed to build passability grid")
                return None
            
            # Check if target is passable
            if passability_grid[target_y, target_x] == 1:
                logger.info(f"Target {target_grid_coords} is blocked/impassable (behavior check)")
                # Debug: Show grid around target
                logger.info(f"🗺️ Pathfinder: Grid around target {target_grid_coords}:")
                for dy in range(-2, 3):
                    row_str = ""
                    for dx in range(-2, 3):
                        gx, gy = target_x + dx, target_y + dy
                        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                            if gx == target_x and gy == target_y:
                                row_str += "T"  # Target
                            elif gx == self.player_center[0] and gy == self.player_center[1]:
                                row_str += "P"  # Player
                            elif passability_grid[gy, gx] == 0:
                                row_str += "."
                            else:
                                row_str += "#"
                        else:
                            row_str += " "
                    logger.info(f"🗺️ Pathfinder: {row_str}")
                return None
            
            # Run A* from player position to target
            path = self._astar(passability_grid, self.player_center, target_grid_coords)
            
            if path is None or len(path) <= 1:
                logger.info(f"No path found from {self.player_center} to {target_grid_coords}")
                return None
            
            # Convert coordinate path to action sequence
            actions = self._path_to_actions(path)
            
            # Limit to max_path_length
            if len(actions) > self.max_path_length:
                logger.info(f"Path length {len(actions)} exceeds max {self.max_path_length}, truncating")
                actions = actions[:self.max_path_length]
            
            logger.info(f"Found path to {target_grid_coords}: {actions}")
            return actions
            
        except Exception as e:
            logger.error(f"Error in pathfinding: {e}")
            return None
    
    def _build_passability_grid(self, game_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Convert map tiles + NPCs into binary passability grid.
        
        Args:
            game_state: Full game state
            
        Returns:
            numpy array where 0=passable, 1=blocked, or None if failed
        """
        try:
            # Get map tiles from game state
            map_info = game_state.get('map', {})
            raw_tiles = map_info.get('tiles', [])
            
            if not raw_tiles or len(raw_tiles) == 0:
                logger.warning("No tiles in map info")
                return None
            
            # Initialize passability grid (0 = passable, 1 = blocked)
            grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
            
            # Debug: Track tile behaviors
            blocked_behaviors = []
            walkable_behaviors = []
            
            # Process tiles
            for y in range(len(raw_tiles)):
                for x in range(len(raw_tiles[y])):
                    if y >= self.grid_size or x >= self.grid_size:
                        continue
                    
                    tile = raw_tiles[y][x]
                    if not tile or len(tile) < 2:
                        grid[y, x] = 1  # Mark unknown tiles as blocked
                        continue
                    
                    behavior = tile[1]
                    
                    # Mark tile as blocked based on behavior
                    # Using same logic as format_tile_to_symbol from map_formatter
                    if behavior == 999:
                        # Visual NPC detection marker - BLOCKED
                        grid[y, x] = 1
                        blocked_behaviors.append(f"NPC_marker_{behavior}")
                    elif behavior in [0, 2, 3, 5, 7, 11, 12, 13, 17, 18, 28, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 52, 60, 66, 67, 68]:
                        # Walkable behaviors (including grass types that trigger encounters)
                        # 2=TALL_GRASS, 3=LONG_GRASS, 7=SHORT_GRASS - these are passable but trigger encounters
                        grid[y, x] = 0
                        walkable_behaviors.append(behavior)
                    else:
                        # Everything else is blocked by default
                        grid[y, x] = 1
                        blocked_behaviors.append(f"behavior_{behavior}")
            
            # Debug: Log tile behavior analysis
            logger.info(f"🗺️ Pathfinder: Tile analysis - Walkable: {len(set(walkable_behaviors))}, Blocked: {len(set(blocked_behaviors))}")
            if blocked_behaviors:
                logger.info(f"🗺️ Pathfinder: Sample blocked behaviors: {list(set(blocked_behaviors))[:5]}")
            
            # Also mark NPC positions from object_events
            npcs = map_info.get('object_events', [])
            player_coords = map_info.get('player_coords')
            
            if npcs and player_coords:
                # player_coords might be tuple or dict
                if isinstance(player_coords, dict):
                    player_x = player_coords.get('x', 0)
                    player_y = player_coords.get('y', 0)
                else:
                    player_x, player_y = player_coords
                
                logger.info(f"🗺️ Pathfinder: Player at ({player_x}, {player_y}), processing {len(npcs)} NPCs")
                
                for npc in npcs:
                    # Get NPC coordinates
                    npc_x = npc.get('current_x', npc.get('x'))
                    npc_y = npc.get('current_y', npc.get('y'))
                    
                    if npc_x is not None and npc_y is not None:
                        # Convert absolute NPC coords to grid coords (relative to player)
                        rel_x = npc_x - player_x + self.player_center[0]
                        rel_y = npc_y - player_y + self.player_center[1]
                        
                        logger.info(f"🗺️ Pathfinder: NPC {npc.get('name', 'Unknown')} at ({npc_x}, {npc_y}) -> grid ({rel_x}, {rel_y})")
                        
                        # Mark NPC position as blocked if within grid
                        if 0 <= rel_x < self.grid_size and 0 <= rel_y < self.grid_size:
                            grid[rel_y, rel_x] = 1
                            logger.info(f"🗺️ Pathfinder: Marked grid ({rel_x}, {rel_y}) as blocked by NPC")
            
            return grid
            
        except Exception as e:
            logger.error(f"Error building passability grid: {e}")
            return None
    
    def _astar(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm implementation.
        
        Args:
            grid: Binary passability grid (0=passable, 1=blocked)
            start: Start coordinates (x, y)
            goal: Goal coordinates (x, y)
            
        Returns:
            List of coordinates forming the path, or None if no path exists
        """
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            """Manhattan distance heuristic"""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            """Get valid neighboring positions"""
            x, y = pos
            neighbors = []
            
            # Four cardinal directions
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # UP, DOWN, LEFT, RIGHT
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    # Check if passable
                    if grid[ny, nx] == 0:
                        neighbors.append((nx, ny))
            
            return neighbors
        
        # A* algorithm
        start_x, start_y = start
        goal_x, goal_y = goal
        
        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(heuristic(start, goal), counter, start)]
        counter += 1
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        visited = set()
        max_iterations = 1000  # Safety limit to prevent hanging
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check if we reached the goal
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Explore neighbors
            for neighbor in get_neighbors(current):
                if neighbor in visited:
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
        
        # Check if we hit iteration limit
        if iterations >= max_iterations:
            logger.warning(f"A* search hit iteration limit ({max_iterations}) - path may not exist or is too complex")
        
        # No path found
        return None
    
    def _path_to_actions(self, path: List[Tuple[int, int]]) -> List[str]:
        """
        Convert coordinate path to action sequence.
        
        Args:
            path: List of (x, y) coordinates
            
        Returns:
            List of action strings (UP/DOWN/LEFT/RIGHT)
        """
        actions = []
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dy == -1:
                actions.append('UP')
            elif dy == 1:
                actions.append('DOWN')
            elif dx == -1:
                actions.append('LEFT')
            elif dx == 1:
                actions.append('RIGHT')
        
        return actions

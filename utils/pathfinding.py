"""
Pathfinding utilities for Pokemon Emerald navigation.

Provides A* pathfinding algorithm with collision detection to enable
intelligent navigation around obstacles.
"""

import heapq
import logging
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

# Try to import MetatileBehavior, but don't fail if not available
try:
    from pokemon_env.enums import MetatileBehavior
except ImportError:
    MetatileBehavior = None

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Represents a position in the pathfinding grid."""
    x: int
    y: int
    g_cost: float = 0  # Cost from start
    h_cost: float = 0  # Heuristic cost to goal
    f_cost: float = 0  # Total cost (g + h)
    parent: Optional['Node'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class Pathfinder:
    """
    A* pathfinding implementation for Pokemon Emerald navigation.
    
    Handles collision detection, NPC avoidance, and terrain considerations.
    """
    
    def __init__(self, collision_map: Optional[Dict] = None, allow_diagonal: bool = False):
        """
        Initialize the pathfinder.
        
        Args:
            collision_map: Dictionary containing collision data for the current map
            allow_diagonal: Whether to allow diagonal movement (default False for Pokemon)
        """
        self.collision_map = collision_map or {}
        self.allow_diagonal = allow_diagonal
        
    def find_path(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int],
        game_state: Dict,
        max_distance: int = 50
    ) -> Optional[List[str]]:
        """
        Find a path from start to goal using A* algorithm.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            game_state: Current game state with map data, NPCs, etc.
            max_distance: Maximum distance to search (prevents infinite loops)
            
        Returns:
            List of button commands to reach the goal, or None if no path found
        """
        # Extract map data from game state
        map_data = self._extract_map_data(game_state)
        if not map_data:
            logger.warning("No map data available for pathfinding")
            return self._simple_path(start, goal)
        
        # Get blocked positions (walls, NPCs, water, etc.)
        blocked = self._get_blocked_positions(game_state, map_data)
        
        # Run A* algorithm
        path = self._astar(start, goal, blocked, map_data, max_distance)
        
        if not path:
            logger.info(f"No path found from {start} to {goal}, trying nearest reachable")
            # Try to find nearest reachable position
            nearest = self._find_nearest_reachable(start, goal, blocked, map_data)
            if nearest and nearest != start:
                path = self._astar(start, nearest, blocked, map_data, max_distance)
        
        if not path:
            logger.warning(f"No path found from {start} to {goal}")
            return None
        
        # Convert path to button commands
        return self._path_to_buttons(path)
    
    def _extract_map_data(self, game_state: Dict) -> Optional[Dict]:
        """Extract map data from game state."""
        # Try to get map data from various possible locations in game state
        if 'map_data' in game_state:
            return game_state['map_data']
        elif 'game_state' in game_state and 'map_data' in game_state['game_state']:
            return game_state['game_state']['map_data']
        elif 'visual' in game_state and 'map' in game_state['visual']:
            return game_state['visual']['map']
        elif 'map' in game_state:
            # This is the actual structure from get_comprehensive_state()
            return game_state['map']
        return None
    
    def _get_blocked_positions(self, game_state: Dict, map_data: Dict) -> Set[Tuple[int, int]]:
        """
        Get all blocked positions on the current map.

        Returns set of (x, y) tuples that are not walkable.
        Note: Ledges are handled separately via directional checks.
        """
        blocked = set()

        # Get map dimensions
        width = map_data.get('width', 50)
        height = map_data.get('height', 50)

        # Check collision data from map tiles
        if 'tiles' in map_data:
            tiles = map_data['tiles']
            for y, row in enumerate(tiles):
                if isinstance(row, list):
                    for x, tile in enumerate(row):
                        if self._is_tile_blocked(tile):
                            blocked.add((x, y))

        # Add NPC positions as blocked
        npcs = self._get_npc_positions(game_state)
        blocked.update(npcs)

        # Add out-of-bounds positions
        for x in range(-1, width + 1):
            blocked.add((x, -1))
            blocked.add((x, height))
        for y in range(-1, height + 1):
            blocked.add((-1, y))
            blocked.add((width, y))

        return blocked
    
    def _is_tile_blocked(self, tile) -> bool:
        """
        Check if a tile is blocked based on its properties.
        Uses the same logic as MapStitcher and map_formatter for consistency.

        Note: Ledges (JUMP_*) are NOT blocked here - they're handled
        via directional validation in _can_move_to().
        """
        if isinstance(tile, tuple) and len(tile) >= 3:
            # Format: (tile_id, behavior, collision, elevation)
            tile_id = tile[0]
            behavior = tile[1] if len(tile) > 1 else 0
            collision = tile[2] if len(tile) > 2 else 0

            # Handle both enum and integer behavior values
            behavior_value = behavior
            if hasattr(behavior, 'value'):
                behavior_value = behavior.value
            elif hasattr(behavior, 'name'):
                behavior_name = behavior.name
            else:
                behavior_name = str(behavior)

            # Use the same logic as map_formatter.py for consistency
            # This ensures pathfinding sees the same tiles as walkable that the map display shows
            
            # Always block invalid tiles
            if tile_id == 1023:  # Invalid/out-of-bounds tile
                return True
            
            # Handle behavior-based blocking using same logic as map_formatter
            if hasattr(behavior, 'name'):
                behavior_name = behavior.name
            elif isinstance(behavior, int) and MetatileBehavior is not None:
                try:
                    behavior_enum = MetatileBehavior(behavior)
                    behavior_name = behavior_enum.name
                except ValueError:
                    behavior_name = "UNKNOWN"
            else:
                behavior_name = "UNKNOWN"

            # Special case for Brendan's House - stairs and doors are reversed
            # NON_ANIMATED_DOOR (96) appears at top and should be stairs (walkable)
            # SOUTH_ARROW_WARP (101) appears at bottom and should be door (walkable)
            if behavior == 96 or "NON_ANIMATED_DOOR" in behavior_name:
                return False  # Stairs are walkable
            elif behavior == 101 or "SOUTH_ARROW_WARP" in behavior_name:
                return False  # Door is walkable
            
            # Other doors and stairs are walkable
            elif "DOOR" in behavior_name or "STAIRS" in behavior_name or "WARP" in behavior_name:
                return False
            
            # Normal tiles - check collision
            elif behavior_name == "NORMAL":
                return collision > 0
            
            # Walkable behaviors (same as map_formatter)
            elif behavior_name in ["INDOOR", "DECORATION", "HOLDS"]:
                return False
            
            # Blocked behaviors
            elif "IMPASSABLE" in behavior_name or "SEALED" in behavior_name:
                return True
            
            # Water (need surf) - blocked for now
            elif "WATER" in behavior_name and "SHALLOW" not in behavior_name:
                return True
            
            # Waterfall (need waterfall) - blocked for now
            elif "WATERFALL" in behavior_name:
                return True
            
            # NPC markers are blocked
            elif behavior == 999:
                return True
            
            # Default: use collision data
            else:
                return collision > 0

        elif isinstance(tile, str):
            # String representation - check for wall symbols
            return tile in ['#', 'X', '█', '▓']

        return False
    
    def _get_npc_positions(self, game_state: Dict) -> Set[Tuple[int, int]]:
        """Get positions of all NPCs on the current map."""
        npcs = set()

        # Check various possible NPC data locations
        npc_data = None
        if 'npcs' in game_state:
            npc_data = game_state['npcs']
        elif 'game_state' in game_state and 'npcs' in game_state['game_state']:
            npc_data = game_state['game_state']['npcs']

        if npc_data:
            for npc in npc_data:
                if isinstance(npc, dict) and 'x' in npc and 'y' in npc:
                    npcs.add((npc['x'], npc['y']))

        return npcs

    def _can_move_to(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                     map_data: Dict) -> bool:
        """
        Check if movement from from_pos to to_pos is valid.

        Handles one-way ledges: can only move in the direction of the ledge.
        - JUMP_EAST (56): can only move TO this tile from the WEST (x-1)
        - JUMP_WEST (57): can only move TO this tile from the EAST (x+1)
        - JUMP_NORTH (58): can only move TO this tile from the SOUTH (y+1)
        - JUMP_SOUTH (59): can only move TO this tile from the NORTH (y-1)
        """
        if 'tiles' not in map_data:
            return True

        tiles = map_data['tiles']

        # Get tile at destination
        # Assuming tiles is centered around player
        # Need to convert world coordinates to tile array indices
        # This is tricky - for now, just allow all movements
        # The blocking is handled by _is_tile_blocked

        # TODO: Implement proper ledge direction checking when we have
        # reliable world-to-tile coordinate mapping

        return True
    
    def _astar(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
        max_distance: int
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm implementation.
        
        Returns:
            List of (x, y) positions from start to goal, or None if no path found
        """
        start_node = Node(start[0], start[1], 0, self._heuristic(start, goal))
        goal_node = Node(goal[0], goal[1])
        
        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()
        node_map = {start: start_node}
        
        while open_list:
            current = heapq.heappop(open_list)
            
            # Check if we reached the goal
            if (current.x, current.y) == goal:
                return self._reconstruct_path(current)
            
            # Check if we've searched too far
            if current.g_cost > max_distance:
                continue
            
            closed_set.add((current.x, current.y))
            
            # Check all neighbors
            for neighbor_pos in self._get_neighbors(current, blocked):
                if neighbor_pos in closed_set:
                    continue
                
                g_cost = current.g_cost + 1  # All moves cost 1 in Pokemon
                h_cost = self._heuristic(neighbor_pos, goal)
                
                if neighbor_pos not in node_map:
                    neighbor = Node(neighbor_pos[0], neighbor_pos[1], g_cost, h_cost)
                    neighbor.parent = current
                    node_map[neighbor_pos] = neighbor
                    heapq.heappush(open_list, neighbor)
                else:
                    neighbor = node_map[neighbor_pos]
                    if g_cost < neighbor.g_cost:
                        neighbor.g_cost = g_cost
                        neighbor.f_cost = g_cost + neighbor.h_cost
                        neighbor.parent = current
        
        return None
    
    def _get_neighbors(self, node: Node, blocked: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get valid neighbor positions for a node."""
        neighbors = []
        
        # Cardinal directions (up, down, left, right)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        if self.allow_diagonal:
            # Add diagonal directions
            directions.extend([(-1, -1), (1, -1), (-1, 1), (1, 1)])
        
        for dx, dy in directions:
            new_x, new_y = node.x + dx, node.y + dy
            if (new_x, new_y) not in blocked:
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Manhattan distance for grid movement)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to goal."""
        path = []
        current = node
        while current:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()
        return path
    
    def _find_nearest_reachable(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        blocked: Set[Tuple[int, int]],
        map_data: Dict
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest reachable position to the goal.
        
        Uses BFS to explore positions near the goal and find the closest
        reachable one.
        """
        if goal not in blocked:
            return goal
        
        visited = set()
        queue = [(goal, 0)]
        visited.add(goal)
        max_search_distance = 10
        
        while queue:
            pos, dist = queue.pop(0)
            
            if dist > max_search_distance:
                break
            
            # Check all neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    
                    if new_pos not in visited:
                        visited.add(new_pos)
                        
                        if new_pos not in blocked:
                            # Found a reachable position
                            # Check if we can path from start to here
                            test_path = self._astar(start, new_pos, blocked, map_data, 50)
                            if test_path:
                                return new_pos
                        
                        queue.append((new_pos, dist + 1))
        
        return None
    
    def _path_to_buttons(self, path: List[Tuple[int, int]]) -> List[str]:
        """Convert a path of positions to button commands."""
        if len(path) < 2:
            return []
        
        buttons = []
        for i in range(1, len(path)):
            prev = path[i - 1]
            curr = path[i]
            
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            
            if dx > 0:
                buttons.append("RIGHT")
            elif dx < 0:
                buttons.append("LEFT")
            elif dy > 0:
                buttons.append("DOWN")
            elif dy < 0:
                buttons.append("UP")
        
        return buttons
    
    def _simple_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
        """Simple straight-line path when no map data is available."""
        buttons = []
        
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        # Move horizontally first
        if dx > 0:
            buttons.extend(["RIGHT"] * min(abs(dx), 10))
        elif dx < 0:
            buttons.extend(["LEFT"] * min(abs(dx), 10))
        
        # Then vertically
        if dy > 0:
            buttons.extend(["DOWN"] * min(abs(dy), 10))
        elif dy < 0:
            buttons.extend(["UP"] * min(abs(dy), 10))
        
        return buttons if buttons else None


# Convenience function
def find_path(start: Tuple[int, int], goal: Tuple[int, int], game_state: Dict) -> Optional[List[str]]:
    """
    Find a path from start to goal position.
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        game_state: Current game state with map data
        
    Returns:
        List of button commands to reach the goal, or None if no path found
    """
    pathfinder = Pathfinder()
    return pathfinder.find_path(start, goal, game_state)
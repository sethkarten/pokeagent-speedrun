"""
Pathfinding utilities for Pokemon Emerald navigation.

Provides A* pathfinding algorithm with collision detection to enable
intelligent navigation around obstacles.
"""

import heapq
import logging
import random
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field

# Try to import MetatileBehavior, but don't fail if not available
try:
    from pokemon_env.enums import MetatileBehavior
except ImportError:
    MetatileBehavior = None

logger = logging.getLogger(__name__)

# Mapping from variance level to the number of initial moves that must differ
VARIANCE_LEVEL_TO_STEPS = {
    "low": 1,
    "medium": 3,
    "high": 5,
    "extreme": 8,
}


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
        """
        Comparison for heapq tie-breaking.
        When f_cost is equal, prefer nodes with higher g_cost (closer to goal),
        or if still equal, prefer nodes closer to goal by h_cost (lower h_cost = closer).
        This ensures deterministic, optimal path selection.
        """
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        # Tie-breaking: prefer nodes with higher g_cost (explored more = closer to goal)
        if self.g_cost != other.g_cost:
            return self.g_cost > other.g_cost
        # Secondary tie-breaking: prefer nodes closer to goal (lower h_cost)
        return self.h_cost < other.h_cost
    
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
        max_distance: int = 150,
        variance: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Find a path from start to goal using A* algorithm.
        
        Args:
            start: Starting position (x, y) - ROM coordinates (0,0 = top-left of current map)
            goal: Goal position (x, y) - ROM coordinates
            game_state: Current game state with map data, NPCs, etc.
            max_distance: Maximum distance to search (default 150, handles large Pokemon maps)
            variance: Optional variance level ('low', 'medium', 'high') controlling
                how many initial moves must differ when sampling alternative paths.
                When None or invalid, pathfinding remains deterministic.
            
        Returns:
            List of button commands to reach the goal, or None if no path found
        """
        # Extract map data from game state
        map_data = self._extract_map_data(game_state)
        
        # CRITICAL: Require porymap data - no fallback allowed
        if not map_data:
            logger.error(f"Pathfinding: No map data available from {start} to {goal}")
            return None
        
        # Only use porymap data - reject other map types
        if map_data.get('type') != 'porymap':
            logger.error(f"Pathfinding: Map data type is '{map_data.get('type')}', but only 'porymap' is allowed. Rejecting pathfinding.")
            return None
        
        if 'grid' not in map_data or not map_data.get('grid'):
            logger.error(f"Pathfinding: Porymap data missing 'grid' field. Cannot pathfind.")
            return None
        
        # Get current location for debugging
        location_name = game_state.get('player', {}).get('location', 'Unknown')
        logger.info(f"Pathfinding: {start} -> {goal} in {location_name}")
        logger.debug(f"Map type: {map_data.get('type')}, dimensions: {map_data.get('width', '?')}x{map_data.get('height', '?')}")
        
        # Normalize variance level (None when disabled)
        variance_level = None
        if variance:
            variance_lower = str(variance).strip().lower()
            if variance_lower in VARIANCE_LEVEL_TO_STEPS:
                variance_level = variance_lower
            elif variance_lower in ("", "none"):
                variance_level = None
            else:
                logger.debug(f"Ignoring unrecognized variance level '{variance}'")

        # Get blocked positions (walls, NPCs, water, etc.)
        # CRITICAL: Exclude start position - player is there so it must be walkable
        blocked = self._get_blocked_positions(game_state, map_data, start_pos=start, goal_pos=goal)
        
        # Get warp positions and ensure they're walkable (doors/stairs)
        warps = self._get_warp_positions(game_state, map_data)
        logger.debug(f"Found {len(warps)} warp positions: {list(warps)[:10]}")  # Show first 10
        for warp_pos in warps:
            blocked.discard(warp_pos)  # Warps are always walkable
            # IMPORTANT: Also unblock the tile ABOVE the warp (in case warp was moved down from a D/S tile)
            # This handles the case where porymap_json_builder adjusted the warp position down by 1
            above_pos = (warp_pos[0], warp_pos[1] - 1)
            if above_pos[1] >= 0:  # Check it's not out of bounds
                blocked.discard(above_pos)
                logger.debug(f"Also unblocking position above warp: {above_pos} (warp at {warp_pos})")

        # SAFEGUARD: Explicitly unblock all 'D' (door) and 'S' (stairs) tiles in the grid
        if 'grid' in map_data and map_data.get('type') == 'porymap':
            grid = map_data['grid']
            doors_and_stairs = []
            for y, row in enumerate(grid):
                for x, cell in enumerate((row if isinstance(row, (list, str)) else [])):
                    if cell in ['D', 'S']:
                        pos = (x, y)
                        if pos in blocked:
                            logger.warning(f"Door/stairs at {pos} ('{cell}') was in blocked set - removing!")
                        blocked.discard(pos)
                        doors_and_stairs.append(pos)
            if doors_and_stairs:
                logger.debug(f"Explicitly unblocked {len(doors_and_stairs)} door/stairs tiles: {doors_and_stairs[:5]}")

        # Ensure start is never blocked
        blocked.discard(start)

        # Ensure door ('D') and stairs ('S') tiles at goal position are always walkable
        if 'grid' in map_data and map_data.get('type') == 'porymap':
            grid = map_data['grid']
            goal_x, goal_y = goal
            if 0 <= goal_y < len(grid) and isinstance(grid[goal_y], (list, str)):
                row = grid[goal_y]
                goal_cell = row[goal_x] if 0 <= goal_x < len(row) else '?'
                if goal_cell in ['D', 'S']:
                    blocked.discard(goal)
                    logger.debug(f"Goal {goal} is on a door/stairs tile '{goal_cell}' - ensuring it's walkable")

        # CRITICAL: Always unblock the goal if it's a warp position, regardless of tile type
        # This handles cases where warps were adjusted (e.g., moved down 1 tile)
        if goal in warps:
            blocked.discard(goal)
            logger.debug(f"Goal {goal} is a warp position - ensuring it's walkable")
        else:
            logger.debug(f"Goal {goal} is NOT in warp positions list")

        # Check if goal is blocked - if so, find nearest reachable position first
        goal_was_blocked = goal in blocked
        if goal_was_blocked:
            logger.warning(f"Goal {goal} is STILL BLOCKED after unblocking attempts!")
        if goal_was_blocked:
            logger.info(f"Goal {goal} is on a blocked tile, finding nearest reachable position")
            # Temporarily add goal back to blocked for nearest search
            blocked.add(goal)
            nearest = self._find_nearest_reachable(start, goal, blocked, map_data)
            if nearest and nearest != start:
                logger.info(f"Using nearest reachable position {nearest} instead of blocked goal {goal}")
                goal = nearest
            else:
                logger.warning(f"Could not find reachable position near blocked goal {goal}")
            # Remove goal from blocked now that we have a new goal
            blocked.discard(goal)
        else:
            # Goal is not blocked, but ensure it's not in blocked set
            blocked.discard(goal)
        
        # Log grid cell status for debugging
        if map_data.get('type') == 'porymap' and 'grid' in map_data:
            grid = map_data['grid']
            start_x, start_y = start
            goal_x, goal_y = goal
            if 0 <= start_y < len(grid) and isinstance(grid[start_y], (list, str)):
                row = grid[start_y]
                if isinstance(row, str):
                    start_cell = row[start_x] if 0 <= start_x < len(row) else '?'
                else:
                    start_cell = row[start_x] if 0 <= start_x < len(row) else '?'
                logger.debug(f"Start ({start_x}, {start_y}) in grid: '{start_cell}' {'(was blocked, now unblocked)' if start_cell == '#' else ''}")
            if 0 <= goal_y < len(grid) and isinstance(grid[goal_y], (list, str)):
                row = grid[goal_y]
                if isinstance(row, str):
                    goal_cell = row[goal_x] if 0 <= goal_x < len(row) else '?'
                else:
                    goal_cell = row[goal_x] if 0 <= goal_x < len(row) else '?'
                goal_pos = (goal_x, goal_y)
                logger.debug(f"Goal ({goal_x}, {goal_y}) in grid: '{goal_cell}' {'(warp - unblocked)' if goal_pos in warps else ''}")
        
        logger.debug(f"Total blocked positions: {len(blocked)}, warps: {len(warps)}")
        
        # Run A* algorithm (returns both path and best_node if path fails)
        result = self._astar(start, goal, blocked, map_data, max_distance)
        
        if isinstance(result, tuple):
            # A* returned (None, best_node) - no complete path but found closest point
            path, best_node = result
            if best_node and (best_node.x, best_node.y) != start:
                logger.warning(f"Pathfinding failed: {start} -> {goal} in {location_name}")
                logger.warning(f"  Blocked count: {len(blocked)}, Map: {map_data.get('type', 'unknown')}")
                logger.info(f"Using partial path to closest point: {(best_node.x, best_node.y)} (distance {best_node.h_cost:.1f} from goal)")
                
                # Return partial path (up to 25 steps) toward the goal
                partial_path = self._reconstruct_path(best_node)
                if len(partial_path) > 25:
                    partial_path = partial_path[:15]
                    logger.info(f"Limiting partial path to first 25 steps")
                
                logger.info(f"Partial path: {len(partial_path)} steps toward goal")
                return self._path_to_buttons(partial_path)
            else:
                logger.warning(f"No progress possible toward {goal}")
                return None
        else:
            path = result
        
        logger.info(f"Path found: {len(path)} steps from {start} to {goal}")
        
        # Warn if path is significantly longer than Manhattan distance (suggests suboptimal pathfinding)
        manhattan_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if len(path) > manhattan_dist * 1.5:
            logger.warning(f"⚠️ Suboptimal path detected: {len(path)} steps (expected ~{manhattan_dist} for straight line)")
            logger.debug(f"Path positions (first 10): {path[:10]}")
            # Log what's blocking the direct path
            direct_path_blocked = []
            if start[0] == goal[0]:  # Vertical path
                for y in range(min(start[1], goal[1]), max(start[1], goal[1]) + 1):
                    pos = (start[0], y)
                    if pos in blocked:
                        direct_path_blocked.append(pos)
                if direct_path_blocked:
                    logger.warning(f"Direct vertical path blocked at: {direct_path_blocked[:5]}")
            elif start[1] == goal[1]:  # Horizontal path
                for x in range(min(start[0], goal[0]), max(start[0], goal[0]) + 1):
                    pos = (x, start[1])
                    if pos in blocked:
                        direct_path_blocked.append(pos)
                if direct_path_blocked:
                    logger.warning(f"Direct horizontal path blocked at: {direct_path_blocked[:5]}")
        
        base_buttons = self._path_to_buttons(path)

        if variance_level:
            variance_steps = VARIANCE_LEVEL_TO_STEPS.get(variance_level)
            if variance_steps:
                variant_buttons = self._generate_variance_candidates(
                    start=start,
                    goal=goal,
                    blocked=blocked,
                    map_data=map_data,
                    max_distance=max_distance,
                    variance_steps=variance_steps,
                    base_path_buttons=base_buttons
                )
                if variant_buttons:
                    chosen_buttons = random.choice(variant_buttons)
                    logger.info(
                        f"Variance '{variance_level}' selected alternative path (prefix {chosen_buttons[:variance_steps]})"
                    )
                    return chosen_buttons
                else:
                    logger.debug(f"No alternative paths found for variance level '{variance_level}', using base path")

        return base_buttons
    
    def _extract_map_data(self, game_state: Dict) -> Optional[Dict]:
        """Extract map data from game state. ONLY returns porymap data - no fallbacks."""
        # ONLY use porymap data (ground truth from JSON) - no fallbacks
        map_data = game_state.get('map', {})
        if map_data and 'porymap' in map_data:
            porymap = map_data['porymap']
            if porymap.get('grid'):
                # Return porymap data structure for pathfinding
                return {
                    'type': 'porymap',
                    'grid': porymap['grid'],  # ASCII grid: [['.', '#', ...], ...]
                    'objects': porymap.get('objects', []),
                    'width': porymap.get('dimensions', {}).get('width', 0),
                    'height': porymap.get('dimensions', {}).get('height', 0),
                    'warps': porymap.get('warps', [])  # Include warps for pathfinding
                }
        
        # No porymap data available - return None (no fallback)
        return None
    
    def _get_blocked_positions(
        self,
        game_state: Dict,
        map_data: Dict,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None
    ) -> Set[Tuple[int, int]]:
        """
        Get all blocked positions on the current map.
        ONLY uses porymap ASCII grid data - no fallbacks.
        
        Args:
            start_pos: Starting position - will NEVER be marked as blocked (player is there, so it is walkable)
            goal_pos: Goal position - excluded from stationary NPC blocking so we can reach the target NPC.

        Returns set of (x, y) tuples that are not walkable.
        Note: Ledges are handled separately via directional checks.
        """
        blocked = set()

        # Get map dimensions
        width = map_data.get('width', 50)
        height = map_data.get('height', 50)

        # REQUIRED: Only use porymap ASCII grid (ground truth data)
        if map_data.get('type') != 'porymap' or 'grid' not in map_data:
            logger.error(f"_get_blocked_positions: Expected porymap data with grid, got type='{map_data.get('type')}', has_grid={'grid' in map_data}")
            return blocked
        
        grid = map_data['grid']
        for y, row in enumerate(grid):
            if isinstance(row, list):
                for x, cell in enumerate(row):
                    # In porymap ASCII:
                    # Walkable: '.' (normal), '~' (grass), 'S' (stairs/warps), 'D' (doors), '←↓↑→' (ledges - directionally walkable)
                    # Blocked: '#' (walls), 'X' (out of bounds), 'W' (water - requires Surf)
                    # CRITICAL: 'D' (door) and 'S' (stairs) tiles are ALWAYS walkable
                    if cell in ['#', 'X', 'W']:
                        pos = (x, y)
                        # CRITICAL: Never block the starting position - player is there so it must be walkable
                        if start_pos and pos == start_pos:
                            logger.debug(f"Excluding start position {start_pos} from blocked set (player is there)")
                            continue
                        blocked.add(pos)
                    # Never block doors or stairs
                    elif cell in ['D', 'S']:
                        continue
            elif isinstance(row, str):
                # Handle string rows (each character is a cell)
                for x, cell in enumerate(row):
                    # CRITICAL: 'D' (door) and 'S' (stairs) tiles are ALWAYS walkable
                    if cell in ['#', 'X', 'W']:
                        pos = (x, y)
                        if start_pos and pos == start_pos:
                            logger.debug(f"Excluding start position {start_pos} from blocked set (player is there)")
                            continue
                        blocked.add(pos)
                    # Never block doors or stairs
                    elif cell in ['D', 'S']:
                        continue

        # Add NPC/object positions as blocked (from porymap objects)
        if 'objects' in map_data:
            objects = map_data['objects']
            for obj in objects:
                obj_x = obj.get('x', 0)
                obj_y = obj.get('y', 0)
                # Block objects that are stationary or have limited movement
                # Block: NONE, FACE_*, WANDER_AROUND (they occupy their position)
                # Allow: WALK_*, RUN_*, JUMP_* (they actively move around)
                movement_type = obj.get('movement_type', '')
                movement_type_upper = movement_type.upper()
                
                if ('NONE' in movement_type_upper or 
                    'STATIC' in movement_type_upper or 
                    'FACE_' in movement_type_upper or  # FACE_DOWN, FACE_UP, etc.
                    'WANDER' in movement_type_upper or  # WANDER_AROUND
                    'LOOK' in movement_type_upper or  # LOOK_AROUND, LOOK_AROUND_EX
                    'BERRY' in movement_type_upper or  # BERRY_TREE_GROWTH
                    not movement_type):
                    obj_pos = (obj_x, obj_y)
                    if goal_pos and obj_pos == goal_pos:
                        continue
                    blocked.add(obj_pos)

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

        Note: Ledges (JUMP_*) are NOT blocked here - they are handled
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
    
    def _get_warp_positions(self, game_state: Dict, map_data: Dict) -> Set[Tuple[int, int]]:
        """
        Get positions of warps (doors, stairs, transitions) which are always walkable.
        
        Args:
            game_state: Current game state
            map_data: Map data structure
            
        Returns:
            Set of (x, y) positions that are warps (always walkable even if marked as blocked)
        """
        warps = set()
        
        # Get warps from porymap data
        if map_data.get('type') == 'porymap':
            # Check if warps are in a separate field
            # Porymap JSON structure should have warps array
            if 'warps' in map_data:
                for warp in map_data['warps']:
                    if isinstance(warp, dict) and 'x' in warp and 'y' in warp:
                        warps.add((warp['x'], warp['y']))
        
        # Also check game_state for warp information
        map_info = game_state.get('map', {})
        if 'warps' in map_info:
            for warp in map_info['warps']:
                if isinstance(warp, dict) and 'x' in warp and 'y' in warp:
                    warps.add((warp['x'], warp['y']))
        
        # Check porymap JSON data if available
        porymap_data = map_info.get('porymap', {})
        if 'warps' in porymap_data:
            for warp in porymap_data['warps']:
                if isinstance(warp, dict) and 'x' in warp and 'y' in warp:
                    warps.add((warp['x'], warp['y']))
        
        return warps

    def _can_move_to(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int],
                     map_data: Dict) -> bool:
        """
        Check if movement from from_pos to to_pos is valid.

        Handles one-way ledges: can only move in the direction of the ledge.
        - JUMP_EAST: can only jump TO this tile by moving EAST (from x-1)
        - JUMP_WEST: can only jump TO this tile by moving WEST (from x+1)
        - JUMP_NORTH: can only jump TO this tile by moving NORTH (from y+1)
        - JUMP_SOUTH: can only jump TO this tile by moving SOUTH (from y-1)
        """
        if 'grid' not in map_data:
            return True

        grid = map_data['grid']
        
        # Check bounds
        to_y, to_x = to_pos[1], to_pos[0]  # grid is [y][x]
        if to_y < 0 or to_y >= len(grid) or to_x < 0 or to_x >= len(grid[0]):
            return False
        
        # Get symbol at destination
        dest_symbol = grid[to_y][to_x]
        
        # Check if destination is a ledge - if so, validate direction
        # Calculate movement direction
        dx = to_pos[0] - from_pos[0]  # positive = moving east, negative = moving west
        dy = to_pos[1] - from_pos[1]  # positive = moving south, negative = moving north
        
        # Ledge direction rules:
        # - Can ONLY move TO a ledge from the correct direction
        # - Moving away from a ledge is always OK
        if dest_symbol == '→':  # JUMP_EAST ledge
            # Can only jump east (dx > 0)
            if dx <= 0:  # Trying to move west, north, south, or diagonal
                return False
        elif dest_symbol == '←':  # JUMP_WEST ledge
            # Can only jump west (dx < 0)
            if dx >= 0:  # Trying to move east, north, south, or diagonal
                return False
        elif dest_symbol == '↑':  # JUMP_NORTH ledge
            # Can only jump north (dy < 0)
            if dy >= 0:  # Trying to move south, east, west, or diagonal
                return False
        elif dest_symbol == '↓':  # JUMP_SOUTH ledge
            # Can only jump south (dy > 0)
            if dy <= 0:  # Trying to move north, east, west, or diagonal
                return False
        elif dest_symbol in ['↗', '↖', '↘', '↙']:  # Diagonal ledges
            # Diagonal ledges require both components to match
            if dest_symbol == '↗':  # JUMP_NORTHEAST
                if dx <= 0 or dy >= 0:  # Must move east AND north
                    return False
            elif dest_symbol == '↖':  # JUMP_NORTHWEST
                if dx >= 0 or dy >= 0:  # Must move west AND north
                    return False
            elif dest_symbol == '↘':  # JUMP_SOUTHEAST
                if dx <= 0 or dy <= 0:  # Must move east AND south
                    return False
            elif dest_symbol == '↙':  # JUMP_SOUTHWEST
                if dx >= 0 or dy <= 0:  # Must move west AND south
                    return False
        
        return True
    
    def _astar(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
        max_distance: int
    ):
        """
        A* pathfinding algorithm implementation.
        
        Returns:
            - List of (x, y) positions from start to goal if path found
            - Tuple of (None, best_node) if no complete path found (for partial paths)
        """
        start_node = Node(start[0], start[1], 0, self._heuristic(start, goal))
        goal_node = Node(goal[0], goal[1])
        
        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()
        node_map = {start: start_node}
        
        # Track the best node (closest to goal) in case we can't reach it
        best_node = start_node
        
        while open_list:
            current = heapq.heappop(open_list)
            
            # Check if we reached the goal
            if (current.x, current.y) == goal:
                return self._reconstruct_path(current)
            
            # Check if we've searched too far
            if current.g_cost > max_distance:
                continue
            
            closed_set.add((current.x, current.y))
            
            # Update best node if this one is closer to goal
            if current.h_cost < best_node.h_cost:
                best_node = current
            
            # Check all neighbors
            neighbors = self._get_neighbors(current, blocked)
            # Sort neighbors by direction preference for deterministic tie-breaking:
            # Prefer moving toward goal (lower h_cost) when f_cost is equal
            # This helps find optimal paths faster
            neighbors_with_cost = []
            for neighbor_pos in neighbors:
                if neighbor_pos in closed_set:
                    continue
                
                # Check if this move is valid (handles ledge directions)
                if not self._can_move_to((current.x, current.y), neighbor_pos, map_data):
                    continue
                
                g_cost = current.g_cost + 1  # All moves cost 1 in Pokemon
                h_cost = self._heuristic(neighbor_pos, goal)
                neighbors_with_cost.append((neighbor_pos, g_cost, h_cost))
            
            # Sort by h_cost (distance to goal) so we explore promising directions first
            # This helps find optimal paths when multiple neighbors have same f_cost
            neighbors_with_cost.sort(key=lambda x: x[2])  # Sort by h_cost
            
            for neighbor_pos, g_cost, h_cost in neighbors_with_cost:
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
        
        # No path found - return the best node we explored (for partial paths)
        return (None, best_node)
    
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
        map_data: Dict,
        max_distance: int = 10
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest reachable position to the goal.
        
        Uses BFS to explore positions near the goal and find the closest
        reachable one.
        
        Args:
            start: Starting position
            goal: Goal position (may be blocked)
            blocked: Set of blocked positions
            map_data: Map data for pathfinding
            max_distance: Maximum distance to search from goal (default 10)
        
        Returns:
            Nearest reachable position, or None if none found
        """
        if goal not in blocked:
            return goal
        
        visited = set()
        queue = [(goal, 0)]
        visited.add(goal)
        max_search_distance = max_distance
        
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
                            test_result = self._astar(start, new_pos, blocked, map_data, 50)
                            # Handle both tuple (None, best_node) and list returns
                            if isinstance(test_result, list):
                                # Found a complete path
                                return new_pos
                        
                        queue.append((new_pos, dist + 1))
        
        return None
    
    def _generate_variance_candidates(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
        max_distance: int,
        variance_steps: int,
        base_path_buttons: List[str],
        max_variants: int = 8
    ) -> List[List[str]]:
        """
        Generate alternative button sequences that reach the goal but differ in their initial moves.
        Each path is guaranteed to have no cycles (never revisits a tile).
        """
        if variance_steps <= 0 or not base_path_buttons or len(base_path_buttons) < variance_steps:
            return []

        # Track seen prefixes to ensure we generate diverse candidates
        base_prefix_key = tuple(base_path_buttons[:variance_steps])
        seen_prefixes = {base_prefix_key}
        candidates: List[List[str]] = []

        prefix_paths = self._enumerate_prefix_paths(
            start=start,
            blocked=blocked,
            map_data=map_data,
            steps_required=variance_steps,
            max_prefixes=max_variants * 4  # oversample to increase odds of unique prefixes
        )

        for prefix_positions in prefix_paths:
            prefix_buttons = self._path_to_buttons(prefix_positions)
            if len(prefix_buttons) < variance_steps:
                continue

            prefix_key = tuple(prefix_buttons[:variance_steps])
            if prefix_key in seen_prefixes:
                continue

            # Block all tiles in the prefix path (except the last one, which is the start of remainder)
            # This prevents the remainder path from looping back through already-visited tiles
            blocked_with_prefix = blocked.copy()
            for pos in prefix_positions[:-1]:  # Exclude last position (start of remainder)
                blocked_with_prefix.add(pos)
            
            remainder = self._astar(prefix_positions[-1], goal, blocked_with_prefix, map_data, max_distance)
            if not isinstance(remainder, list) or len(remainder) < 1:
                continue

            full_positions = prefix_positions + remainder[1:]
            
            # Validate that the full path has no cycles (no position appears twice)
            if len(set(full_positions)) != len(full_positions):
                logger.debug(f"Rejecting path with cycle: {len(full_positions)} steps but only {len(set(full_positions))} unique positions")
                continue
            
            buttons = self._path_to_buttons(full_positions)
            if len(buttons) < variance_steps:
                continue

            initial_key = tuple(buttons[:variance_steps])
            if initial_key in seen_prefixes:
                continue

            seen_prefixes.add(initial_key)
            candidates.append(buttons)

            if len(candidates) >= max_variants:
                break

        return candidates

    def _enumerate_prefix_paths(
        self,
        start: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        map_data: Dict,
        steps_required: int,
        max_prefixes: int = 256
    ) -> List[List[Tuple[int, int]]]:
        """
        Enumerate walkable position sequences originating from start with a fixed number of steps.
        """
        if steps_required <= 0:
            return []

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        results: List[List[Tuple[int, int]]] = []
        seen_paths: Set[Tuple[Tuple[int, int], ...]] = set()
        stack = [(start, [start])]

        while stack and len(results) < max_prefixes:
            current_pos, positions = stack.pop()
            steps_taken = len(positions) - 1

            if steps_taken == steps_required:
                path_tuple = tuple(positions)
                if path_tuple not in seen_paths:
                    seen_paths.add(path_tuple)
                    results.append(positions)
                continue

            for dx, dy in directions:
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if next_pos in blocked:
                    continue

                if next_pos in positions:
                    continue

                if not self._can_move_to(current_pos, next_pos, map_data):
                    continue

                stack.append((next_pos, positions + [next_pos]))

        return results
    
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
def find_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    game_state: Dict,
    max_distance: int = 150,
    variance: Optional[str] = None
) -> Optional[List[str]]:
    """
    Find a path from start to goal position.
    
    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        game_state: Current game state with map data
        max_distance: Maximum distance to search (default 150)
        variance: Optional variance level ('low', 'medium', 'high')
        
    Returns:
        List of button commands to reach the goal, or None if no path found
    """
    pathfinder = Pathfinder()
    return pathfinder.find_path(start, goal, game_state, max_distance=max_distance, variance=variance)
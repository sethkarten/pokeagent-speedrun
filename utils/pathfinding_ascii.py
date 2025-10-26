"""
ASCII Map-based Pathfinding for Pokemon Emerald

This module provides pathfinding that works directly on the ASCII map representation
that the agent sees, avoiding reliance on potentially corrupted map buffer data.
"""

import logging
import re
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Represents a position in the pathfinding grid."""
    x: int
    y: int
    g_cost: float = 0
    h_cost: float = 0
    f_cost: float = 0
    parent: Optional['Node'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class ASCIIPathfinder:
    """
    A* pathfinding that works on ASCII map representations.
    
    This avoids reliance on map buffer data which may be corrupted,
    and works directly on the textual map display the agent sees.
    """
    
    # Characters that represent blocked/wall tiles
    BLOCKED_CHARS = {'#', 'X', '█', '▓'}
    
    # Characters that represent walkable tiles
    WALKABLE_CHARS = {'.', ' ', 'P'}  # P is player, treated as walkable for pathfinding
    
    def __init__(self):
        pass
    
    def find_path_from_ascii_map(
        self, 
        ascii_map: str,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[str]]:
        """
        Find path using ASCII map representation.
        
        Args:
            ascii_map: Multi-line string representing the map (e.g., from game state)
            start: Starting position (x, y) in game coordinates
            goal: Goal position (x, y) in game coordinates
            
        Returns:
            List of button commands (UP/DOWN/LEFT/RIGHT) or None if no path found
        """
        try:
            # Parse ASCII map into grid
            grid = self._parse_ascii_map(ascii_map)
            if not grid:
                logger.warning(f"Failed to parse ASCII map. Input length: {len(ascii_map) if ascii_map else 0}")
                logger.debug(f"Input: {ascii_map[:200] if ascii_map else 'None'}")
                return None
            
            # Find 'P' (player) in the grid to understand coordinate system
            player_grid_pos = None
            for y, row in enumerate(grid):
                for x, char in enumerate(row):
                    if char == 'P':
                        player_grid_pos = (x, y)
                        break
                if player_grid_pos:
                    break
            
            if not player_grid_pos:
                logger.warning("Could not find player (P) in ASCII map")
                # Fallback: assume coordinates are already in grid space
                grid_start = start
                grid_goal = goal
            else:
                # Convert game coordinates to grid coordinates
                # If player is at game (3, 2) but grid (7, 2), the offset is +4
                offset_x = player_grid_pos[0] - start[0]
                grid_start = (start[0] + offset_x, start[1])
                grid_goal = (goal[0] + offset_x, goal[1])
                logger.info(f"Player game pos: {start}, grid pos: {player_grid_pos}, offset: {offset_x}")
                logger.info(f"Goal converted from {goal} to {grid_goal}")
            
            # Run A* pathfinding
            path = self._astar(grid, grid_start, grid_goal)
            
            if not path or len(path) <= 1:
                logger.warning(f"No path found from {grid_start} to {grid_goal} on ASCII map")
                return None
            
            # Convert path to button commands
            return self._path_to_buttons(path)
            
        except Exception as e:
            logger.error(f"Error in ASCII pathfinding: {e}")
            return None
    
    def _parse_ascii_map(self, ascii_map: str) -> List[List[str]]:
        """
        Parse ASCII map string into 2D grid.
        
        Handles both formats:
        1. Compact format: "# # #" (spaces as separators)
        2. Coordinate format with labels
        
        The map is assumed to use spaces to separate cells (like "# # .").
        We split by spaces but keep track of the actual symbols.
        """
        lines = ascii_map.strip().split('\n')
        grid = []
        
        for line in lines:
            # Skip header rows (coordinate labels) and separator lines
            if not line.strip() or line.strip().startswith('---'):
                continue
                
            # Skip coordinate labels (lines starting with numbers like "  0  1  2")
            if re.match(r'^\s*\d+', line):
                continue
            
            # Split the line into tokens by spaces
            # For example: "# # . . S" -> ['#', '#', '.', '.', 'S']
            tokens = line.split()
            
            # Only add if we have actual content (not just whitespace)
            if tokens:
                grid.append(tokens)
        
        return grid
    
    def _is_blocked(self, grid: List[List[str]], x: int, y: int) -> bool:
        """Check if position (x, y) is blocked."""
        if not (0 <= y < len(grid) and 0 <= x < len(grid[y])):
            return True  # Out of bounds
        
        char = grid[y][x]
        return char in self.BLOCKED_CHARS
    
    def _astar(
        self,
        grid: List[List[str]],
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm."""
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
            
            closed_set.add((current.x, current.y))
            
            # Check all neighbors
            for neighbor_pos in self._get_neighbors(grid, current.x, current.y):
                if neighbor_pos in closed_set:
                    continue
                
                g_cost = current.g_cost + 1
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
    
    def _get_neighbors(self, grid: List[List[str]], x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighbor positions."""
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if not self._is_blocked(grid, new_x, new_y):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start."""
        path = []
        current = node
        while current:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()
        return path
    
    def _path_to_buttons(self, path: List[Tuple[int, int]]) -> List[str]:
        """Convert path to button commands."""
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


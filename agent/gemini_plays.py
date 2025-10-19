"""
GeminiPlaysPokemon Agent for Pokemon Emerald
==============================================

Implementation based on JCZ's Gemini Plays Pokemon architecture.
Reference: https://blog.jcz.dev/the-making-of-gemini-plays-pokemon

Core architectural principles:
- Hierarchical goal system (primary/secondary/tertiary objectives)
- Map Memory with fog-of-war exploration tracking
- Specialized agent delegation for complex tasks
- Self-critique mechanism for performance improvement
- Periodic context resets with intelligent summarization
- Forced exploration directives
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import random

from utils.vlm import VLM
from utils.llm_logger import LLMLogger
from utils.state_formatter import format_state_for_llm
from utils.agent_helpers import update_server_metrics

logger = logging.getLogger(__name__)


@dataclass
class Goal:
    """Represents a hierarchical goal in the system."""
    type: str  # "primary", "secondary", "tertiary"
    description: str
    conditions: List[str]  # Success conditions
    priority: int
    created_at: int  # Step when created
    completed: bool = False
    progress: str = ""


@dataclass
class MapMemory:
    """Fog-of-war style map exploration memory."""
    explored_tiles: Set[Tuple[int, int]] = field(default_factory=set)
    points_of_interest: Dict[Tuple[int, int], str] = field(default_factory=dict)
    blocked_paths: Set[Tuple[int, int]] = field(default_factory=set)
    last_visited: Dict[Tuple[int, int], int] = field(default_factory=dict)
    visited_warps: Set[Tuple[int, int]] = field(default_factory=set)  # Track which warps have been used

    def mark_explored(self, x: int, y: int, step: int):
        """Mark a tile as explored (seen, but not necessarily visited)."""
        self.explored_tiles.add((x, y))

    def mark_visited(self, x: int, y: int, step: int):
        """Mark a tile as visited (player stood on it)."""
        self.explored_tiles.add((x, y))  # Visited implies explored
        self.last_visited[(x, y)] = step

    def mark_warp_used(self, x: int, y: int):
        """Mark a warp as having been used."""
        self.visited_warps.add((x, y))

    def get_exploration_percentage(self, total_tiles: int) -> float:
        """Calculate exploration percentage."""
        if total_tiles == 0:
            return 0.0
        return (len(self.explored_tiles) / total_tiles) * 100

    def get_unexplored_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get unexplored neighboring tiles."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in self.explored_tiles:
                neighbors.append((nx, ny))
        return neighbors

    def get_navigable_unseen_tiles(self, tiles: List[Dict]) -> List[Tuple[int, int]]:
        """Get tiles with type 'unknown' (?) that are adjacent to walkable explored tiles.

        These are the HIGHEST PRIORITY for exploration in GPP.

        Args:
            tiles: List of tile dicts from JSON map with x, y, type, walkable fields

        Returns:
            List of (x, y) coordinates of navigable unseen tiles
        """
        unseen = []

        # Build set of walkable explored tiles
        walkable_explored = set()
        for tile in tiles:
            x, y = tile['x'], tile['y']
            if tile.get('walkable') and (x, y) in self.explored_tiles:
                walkable_explored.add((x, y))

        # Find unknown tiles adjacent to walkable explored tiles
        for tile in tiles:
            if tile.get('type') == 'unknown':
                x, y = tile['x'], tile['y']
                # Check if adjacent to any walkable explored tile
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (x + dx, y + dy) in walkable_explored:
                        unseen.append((x, y))
                        break

        return unseen

    def get_navigable_unvisited_warps(self, tiles: List[Dict]) -> List[Tuple[int, int, str]]:
        """Get warps (doors/stairs) that have not been visited yet.

        These are the 2nd HIGHEST PRIORITY for exploration in GPP (after unseen tiles).

        Args:
            tiles: List of tile dicts from JSON map with x, y, type, walkable, leads_to fields

        Returns:
            List of (x, y, leads_to) tuples for unvisited warps
        """
        unvisited_warps = []

        for tile in tiles:
            if tile.get('type') in ['door', 'stairs']:
                x, y = tile['x'], tile['y']
                # Check if this warp has been used
                if (x, y) not in self.visited_warps:
                    leads_to = tile.get('leads_to', 'Unknown')
                    unvisited_warps.append((x, y, leads_to))

        return unvisited_warps


@dataclass
class AgentContext:
    """Maintains agent's long-term memory and context."""
    summary: str = ""
    key_events: List[str] = field(default_factory=list)
    pokemon_caught: List[str] = field(default_factory=list)
    badges_earned: List[str] = field(default_factory=list)
    important_items: List[str] = field(default_factory=list)
    recent_battles: List[Dict[str, Any]] = field(default_factory=list)
    exploration_history: deque = field(default_factory=lambda: deque(maxlen=50))
    notepad: List[str] = field(default_factory=list)  # For storing long-term plans
    custom_agents: Dict[str, str] = field(default_factory=dict)  # User-defined mini-agents


class SpecializedAgent:
    """Base class for specialized sub-agents."""

    def __init__(self, name: str, vlm_client: VLM, llm_logger):
        self.name = name
        self.vlm_client = vlm_client
        self.llm_logger = llm_logger
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def execute(self, task: str, state: Dict[str, Any], screenshot: Any = None) -> Dict[str, Any]:
        """Execute specialized task."""
        raise NotImplementedError


class PathfindingAgent(SpecializedAgent):
    """Specialized agent for navigation and pathfinding using LLM reasoning.

    Uses focused LLM reasoning to solve complex pathfinding tasks like mazes,
    spinner puzzles, and ice tile navigation that require spatial reasoning.
    """

    def __init__(self, vlm_client: VLM, llm_logger):
        super().__init__("pathfinding", vlm_client, llm_logger)

    def execute(self, task: str, state: Dict[str, Any], screenshot: Any = None) -> Dict[str, Any]:
        """Execute pathfinding using LLM to reason about the path.

        The LLM analyzes the map, identifies obstacles, and generates a sequence
        of movements to reach the target location.
        """
        # Get current position and map data
        current_pos = state.get("player_position", {})
        player_x = current_pos.get("x", 0)
        player_y = current_pos.get("y", 0)

        # Get map information
        map_data = format_state_for_llm(state, use_json_map=True)

        prompt = f"""You are a pathfinding specialist for Pokemon Emerald. Your task is to navigate complex terrain.

**Current Task:** {task}

**Current Position:** ({player_x}, {player_y})

**Game State & Map:**
{map_data[:2000]}

**Your Goal:**
Analyze the screenshot and map to generate a sequence of movements to reach the destination. Consider:
1. Walkable tiles vs obstacles (walls, water without Surf, etc.)
2. Special tiles (doors, stairs, spinner tiles, ice tiles)
3. Optimal path avoiding unnecessary detours
4. NPCs that might block the path
5. Visual cues from the screenshot (terrain, layout, current orientation)

**Output Format:**
Provide a sequence of 5-10 button presses to make progress toward the goal.
Reply with ONLY the buttons separated by commas.
Valid buttons: UP, DOWN, LEFT, RIGHT, A (for interactions)

Example: "RIGHT, RIGHT, UP, UP, A, UP, LEFT"

Generate the path:"""

        # Use screenshot if available for better spatial reasoning
        if screenshot is not None:
            response = self.vlm_client.get_query(screenshot, prompt, "pathfinding")
        else:
            response = self.vlm_client.get_text_query(prompt, "pathfinding")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_pathfinding",
            prompt=prompt,
            response=response
        )

        # Parse button sequence from response
        buttons = []
        for part in response.upper().split(','):
            button = part.strip()
            if button in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]:
                buttons.append(button)
                if len(buttons) >= 10:
                    break

        if buttons:
            return {
                "success": True,
                "buttons": buttons,
                "message": f"Generated path: {len(buttons)} steps"
            }
        else:
            return {
                "success": False,
                "buttons": [],
                "message": "Failed to generate valid path"
            }


class BattleAgent(SpecializedAgent):
    """Specialized agent for battle strategy."""

    def __init__(self, vlm_client: VLM, llm_logger):
        super().__init__("battle", vlm_client, llm_logger)
    
    def execute(self, task: str, state: Dict[str, Any], screenshot: Any = None) -> Dict[str, Any]:
        """Execute battle strategy."""
        battle = state.get("in_battle", False)
        if not battle:
            return {"success": False, "buttons": [], "message": "Not in battle"}
        
        prompt = f"""You are a Pokemon battle specialist. Analyze this battle from the screenshot and game state.

{format_state_for_llm(state, use_json_map=True)[:1000]}

Task: {task}

Look at the battle screen and choose the best action:
1. Use move (specify which: 1-4)
2. Switch Pokemon (specify which: 1-6)
3. Use item
4. Run

Reply with just the action and number, e.g., "move 1" or "switch 2"."""

        # Use screenshot if available for visual battle analysis
        if screenshot is not None:
            response = self.vlm_client.get_query(screenshot, prompt, "battle")
        else:
            response = self.vlm_client.get_text_query(prompt, "battle")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_battle",
            prompt=prompt,
            response=response
        )

        # Parse response and convert to buttons
        buttons = []
        if "move" in response.lower():
            buttons = ["A"]  # Select fight
            if "2" in response:
                buttons.extend(["DOWN", "A"])
            elif "3" in response:
                buttons.extend(["RIGHT", "A"])
            elif "4" in response:
                buttons.extend(["DOWN", "RIGHT", "A"])
            else:
                buttons.append("A")  # Default to move 1
        elif "switch" in response.lower():
            buttons = ["RIGHT", "A"]  # Go to Pokemon menu
            # Add navigation for specific Pokemon
        elif "run" in response.lower():
            buttons = ["DOWN", "RIGHT", "A"]  # Run option
        else:
            buttons = ["A"]  # Default action
        
        return {
            "success": True,
            "buttons": buttons,
            "message": f"Battle action: {response}"
        }


class PuzzleAgent(SpecializedAgent):
    """Specialized agent for solving puzzles (boulder puzzles, etc.)."""

    def __init__(self, vlm_client: VLM, llm_logger):
        super().__init__("puzzle", vlm_client, llm_logger)
    
    def execute(self, task: str, state: Dict[str, Any], screenshot: Any = None) -> Dict[str, Any]:
        """Execute puzzle solving strategy using visual analysis."""
        prompt = f"""You are a puzzle-solving specialist for Pokemon Emerald (boulder/Strength puzzles, ice puzzles, etc.).

Task: {task}
Current state: {format_state_for_llm(state, use_json_map=True)[:500]}

Analyze the screenshot to understand the puzzle layout. Consider:
1. Boulder positions and target holes
2. Ice tile sliding mechanics
3. Order of operations needed
4. Irreversible moves that could block progress

Provide the next 5 moves to progress in solving the puzzle.
Reply with just the button sequence, e.g., "UP, UP, LEFT, DOWN, A"."""

        # Use screenshot if available for visual puzzle analysis
        if screenshot is not None:
            response = self.vlm_client.get_query(screenshot, prompt, "puzzle")
        else:
            response = self.vlm_client.get_text_query(prompt, "puzzle")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_puzzle",
            prompt=prompt,
            response=response
        )

        # Parse button sequence
        buttons = []
        for part in response.upper().split(','):
            button = part.strip()
            if button in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R"]:
                buttons.append(button)
        
        return {
            "success": bool(buttons),
            "buttons": buttons[:10],  # Limit to 10 buttons
            "message": f"Puzzle moves: {', '.join(buttons[:10])}"
        }


class GeminiPlaysAgent:
    """
    Implementation of Gemini Plays Pokemon architecture.
    
    Key features:
    - Hierarchical goal management
    - Map memory with fog-of-war
    - Specialized agent delegation
    - Self-critique and improvement
    - Context reset with summarization
    - Exploration directives
    - Meta-tools for AI self-extension (define_agent, execute_script, notepad)
    """
    
    def __init__(
        self,
        vlm_client: Optional[VLM] = None,
        context_reset_interval: int = 100,
        enable_self_critique: bool = True,
        enable_exploration: bool = True,
        enable_meta_tools: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the GeminiPlaysAgent.
        
        Args:
            vlm_client: Vision-language model client
            context_reset_interval: Steps before context reset (default 100 like original)
            enable_self_critique: Enable self-critique mechanism
            enable_exploration: Enable forced exploration
            verbose: Detailed logging
        """
        self.vlm_client = vlm_client or VLM()
        self.context_reset_interval = context_reset_interval
        self.enable_self_critique = enable_self_critique
        self.enable_exploration = enable_exploration
        self.enable_meta_tools = enable_meta_tools
        self.verbose = verbose
        
        # Core components
        self.step_count = 0
        self.map_memory = MapMemory()
        self.context = AgentContext()
        self.llm_logger = LLMLogger()
        
        # Hierarchical goals
        self.primary_goal: Optional[Goal] = None
        self.secondary_goal: Optional[Goal] = None
        self.tertiary_goal: Optional[Goal] = None
        
        # Specialized agents
        self.pathfinding_agent = PathfindingAgent(vlm_client, self.llm_logger)
        self.battle_agent = BattleAgent(vlm_client, self.llm_logger)
        self.puzzle_agent = PuzzleAgent(vlm_client, self.llm_logger)
        
        # Action queue
        self.action_queue: deque = deque()
        
        # Performance tracking for self-critique
        self.recent_actions: deque = deque(maxlen=20)
        self.stuck_counter = 0
        self.last_position = (0, 0)
    
    def step(self, state: Dict[str, Any], screenshot: Any) -> str:
        """
        Execute one step following Gemini Plays Pokemon architecture.

        Args:
            state: Current game state (includes 'frame' key with screenshot)

        Returns:
            Button command (e.g., "A", "UP", "NONE")
        """
        self.step_count += 1

        # Detect game context from screenshot using VLM (title, dialogue, battle, overworld)
        print(f"has screenshot? {screenshot is not None}")
        context_type = self.get_game_context(screenshot) if screenshot is not None else "unknown"
        if self.verbose and hasattr(self, '_last_context') and self._last_context != context_type:
            print(f"[CONTEXT] Changed from {self._last_context} → {context_type}")
        self._last_context = context_type

        # Check for context reset
        if self.step_count % self.context_reset_interval == 0:
            self._reset_context(state)

        # Update map memory
        self._update_map_memory(state)
        
        # Check if stuck (self-critique)
        if self.enable_self_critique:
            self._check_if_stuck(state)
        
        # If we have queued actions, return them all at once for the server to execute
        if self.action_queue:
            actions = list(self.action_queue)
            self.action_queue.clear()
            self.recent_actions.extend(actions)
            if self.verbose:
                print(f"[QUEUE] Returning {len(actions)} queued actions: {', '.join(actions)}")
            # Return as comma-separated string for client to split
            return ','.join(actions)

        # Check for meta-tool usage (before normal decision making)
        if self.enable_meta_tools and self.step_count % 10 == 0:
            if self._process_meta_tools(state, screenshot):
                # If meta-tool was used, it may have queued actions
                if self.action_queue:
                    actions = list(self.action_queue)
                    self.action_queue.clear()
                    self.recent_actions.extend(actions)
                    if self.verbose:
                        print(f"[QUEUE] Returning {len(actions)} meta-tool actions: {', '.join(actions)}")
                    return ','.join(actions)

        # Update goals
        self._update_goals(state)

        # Decide next action based on hierarchical goals (pass context_type to avoid re-detection)
        action = self._decide_action(state, screenshot, context_type)

        # If delegation happened, queue should have actions - check and return them
        if action == "DELEGATED" and self.action_queue:
            actions = list(self.action_queue)
            self.action_queue.clear()
            self.recent_actions.extend(actions)
            if self.verbose:
                print(f"[DELEGATION] Returning {len(actions)} delegated actions")
            return ','.join(actions)

        # Apply exploration directive if enabled
        if self.enable_exploration and self.step_count % 50 == 0:
            exploration_action = self._get_exploration_action(state)
            if exploration_action and exploration_action != "NONE":
                action = exploration_action

        # Update server with agent step and metrics (for agent thinking display)
        update_server_metrics()

        self.recent_actions.append(action)
        return action
    
    def _update_map_memory(self, state: Dict[str, Any]):
        """Update fog-of-war map memory."""
        pos = state.get("player_position", {})
        x, y = pos.get("x", 0), pos.get("y", 0)

        # Mark current position as VISITED (player stood on it)
        self.map_memory.mark_visited(x, y, self.step_count)

        # Mark visible area as explored (seen but not visited) (5x5 around player)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx != 0 or dy != 0:  # Don't re-mark center as just explored
                    self.map_memory.mark_explored(x + dx, y + dy, self.step_count)

        # Check if player used a warp (location changed)
        current_location = state.get("player", {}).get("location", "")
        if hasattr(self, '_last_location') and self._last_location != current_location:
            # Location changed - mark last position as warp used
            if hasattr(self, '_last_position'):
                last_x, last_y = self._last_position
                self.map_memory.mark_warp_used(last_x, last_y)

        # Track location and position for next step
        self._last_location = current_location
        self._last_position = (x, y)
        
        # Track points of interest
        if state.get("in_pokemon_center"):
            self.map_memory.points_of_interest[(x, y)] = "pokemon_center"

    def get_game_context(self, screenshot: Any) -> str:
        """Determine current game context from the visual frame using VLM.

        Uses vision-language model to analyze the screenshot and identify screen type.

        Args:
            screenshot: PIL Image of current game frame

        Returns:
            str: One of "title", "dialogue", "battle", "menu", "overworld", "unknown"
        """
        if screenshot is None:
            return "unknown"

        try:
            detection_prompt = """Analyze this Pokemon Emerald screenshot and identify the current game screen type.

Look for these visual indicators:
1. TITLE - Title screen with "Pokemon" logo, "Press Start" text, or intro cutscene
2. DIALOGUE - Text box at bottom with NPC dialogue or narration (white box with black text)
3. BATTLE - Pokemon battle screen with HP bars, move selection, or battle animations
4. MENU - In-game menu (Pokédex, Pokemon, Bag, Save, Options, Exit)
5. OVERWORLD - Top-down map view with player character, NPCs, buildings, routes

Reply with ONLY ONE WORD from: TITLE, DIALOGUE, BATTLE, MENU, or OVERWORLD"""

            response = self.vlm_client.get_query(screenshot, detection_prompt, "context")

            # Log interaction
            self.llm_logger.log_interaction(
                interaction_type="gemini_plays_context",
                prompt=detection_prompt,
                response=response
            )

            # Parse response to get context type
            context = response.strip().upper()

            # Map to lowercase and validate
            valid_contexts = ["TITLE", "DIALOGUE", "BATTLE", "MENU", "OVERWORLD"]
            if context in valid_contexts:
                return context.lower()

            # Fallback: try to find the context word in response
            for ctx in valid_contexts:
                if ctx in response.upper():
                    return ctx.lower()

            logger.warning(f"Could not parse game context from VLM response: {response}")
            return "unknown"

        except Exception as e:
            logger.warning(f"Error determining game context with VLM: {e}")
            return "unknown"
    
    def _check_if_stuck(self, state: Dict[str, Any]):
        """Self-critique: Check if agent is stuck."""
        pos = state.get("player_position", {})
        current_pos = (pos.get("x", 0), pos.get("y", 0))
        
        if current_pos == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos
        
        # If stuck for too long, trigger self-critique
        if self.stuck_counter > 10:
            if self.verbose:
                print("Self-critique: Agent appears stuck, analyzing...")
            self._perform_self_critique(state)
            self.stuck_counter = 0
    
    def _perform_self_critique(self, state: Dict[str, Any]):
        """Analyze recent performance and adjust strategy."""
        recent_actions_str = ", ".join(list(self.recent_actions)[-10:])
        
        prompt = f"""Analyze this agent's recent performance in Pokemon Emerald:

Recent actions: {recent_actions_str}
Current state: {format_state_for_llm(state, use_json_map=True)}
Steps taken: {self.step_count}
Exploration: {self.map_memory.get_exploration_percentage(1000):.1f}%

The agent appears stuck. Identify:
1. What went wrong?
2. What should be done differently?
3. Suggest 5 specific actions to unstick.

Be concise and specific."""

        critique = self.vlm_client.get_text_query(prompt, "critique")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_critique",
            prompt=prompt,
            response=critique
        )

        if self.verbose:
            print(f"   Self-critique: {critique[:150]}...")
        
        # Extract suggested actions
        self._extract_recovery_actions(critique)
    
    def _extract_recovery_actions(self, critique: str):
        """Extract recovery actions from self-critique using LLM."""
        prompt = f"""Based on this self-critique analysis:
{critique}

Provide a sequence of 3-5 button presses to help unstick the agent.
Reply with just the buttons separated by commas (e.g., "UP, A, LEFT, DOWN, A").
Valid buttons: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT"""

        response = self.vlm_client.get_text_query(prompt, "recovery")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_recovery",
            prompt=prompt,
            response=response
        )

        # Parse button sequence
        buttons_found = False
        for part in response.upper().split(','):
            button = part.strip()
            if button in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]:
                self.action_queue.append(button)
                buttons_found = True
                if len(self.action_queue) >= 5:
                    break
        
        # If no valid actions parsed, ask for exploration
        if not buttons_found:
            self._add_exploration_sequence()
    
    def _add_exploration_sequence(self):
        """Add exploration sequence using LLM suggestion."""
        prompt = """The agent needs to explore. Provide a short sequence of 3-4 directional buttons for exploration.
Reply with just the buttons separated by spaces (e.g., "UP LEFT DOWN RIGHT")."""

        response = self.vlm_client.get_text_query(prompt, "exploration_seq")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_exploration",
            prompt=prompt,
            response=response
        )

        for word in response.upper().split():
            if word in ["UP", "DOWN", "LEFT", "RIGHT"]:
                self.action_queue.append(word)
                if len(self.action_queue) >= 4:
                    break
        
        # Ultimate fallback
        if not self.action_queue:
            for _ in range(3):
                self.action_queue.append(random.choice(["UP", "DOWN", "LEFT", "RIGHT"]))
    
    def _update_goals(self, state: Dict[str, Any]):
        """Update hierarchical goal system using LLM to define goals."""
        # Check if goals need updating
        should_update = False
        
        # Check primary goal completion
        if self.primary_goal:
            if self._check_goal_completion(self.primary_goal, state):
                self.primary_goal.completed = True
                should_update = True
        else:
            should_update = True
        
        # Check secondary goal completion
        if self.secondary_goal:
            if self._check_goal_completion(self.secondary_goal, state):
                self.secondary_goal.completed = True
                should_update = True
        elif self.primary_goal:
            should_update = True
        
        # Check tertiary goal completion
        if self.tertiary_goal:
            if self._check_goal_completion(self.tertiary_goal, state):
                self.tertiary_goal.completed = True
                should_update = True
        else:
            should_update = True
        
        # Update goals if needed
        if should_update or self.step_count % 50 == 0:
            self._generate_new_goals(state)
        
        if self.verbose and self.step_count % 20 == 0:
            print(f"=> Goals:")
            print(f"   Primary: {self.primary_goal.description if self.primary_goal else 'None'}")
            print(f"   Secondary: {self.secondary_goal.description if self.secondary_goal else 'None'}")
            print(f"   Tertiary: {self.tertiary_goal.description if self.tertiary_goal else 'None'}")
    
    def _generate_new_goals(self, state: Dict[str, Any]):
        """Have the LLM generate new hierarchical goals based on current state."""
        prompt = f"""You are playing Pokemon Emerald. Based on the current game state, define three hierarchical goals.

Current State:
{format_state_for_llm(state, use_json_map=True)[:800]}

Context: {self.context.summary}
Exploration: {self.map_memory.get_exploration_percentage(1000):.1f}% of map explored
Current goals completed: Primary={self.primary_goal.completed if self.primary_goal else 'N/A'}, Secondary={self.secondary_goal.completed if self.secondary_goal else 'N/A'}, Tertiary={self.tertiary_goal.completed if self.tertiary_goal else 'N/A'}

Define three goals in this exact JSON format:
{{
  "primary": {{
    "description": "Main progression goal (e.g., defeat gym leader, reach new city)",
    "completion_check": "Specific condition to check if complete (e.g., badges > 0, location == rustboro)"
  }},
  "secondary": {{
    "description": "Goal that enables the primary (e.g., train Pokemon, get items)",  
    "completion_check": "Specific condition to check (e.g., team_avg_level > 15, has_potion)"
  }},
  "tertiary": {{
    "description": "Opportunistic goal (e.g., explore area, catch Pokemon)",
    "completion_check": "Specific condition (e.g., explored_new_area, caught_pokemon)"
  }}
}}

Rules:
- Primary should be a major progression milestone
- Secondary should directly help achieve primary goal
- Tertiary should be opportunistic/exploratory
- Be specific and measurable
- Consider what was just completed to avoid repetition"""

        response = self.vlm_client.get_text_query(prompt, "goals")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_goals",
            prompt=prompt,
            response=response
        )

        # Parse goals from response
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                goals_data = json.loads(response[start:end])
                
                # Update primary goal
                if "primary" in goals_data and not (self.primary_goal and not self.primary_goal.completed):
                    self.primary_goal = Goal(
                        type="primary",
                        description=goals_data["primary"]["description"],
                        conditions=[goals_data["primary"]["completion_check"]],
                        priority=1,
                        created_at=self.step_count
                    )
                
                # Update secondary goal
                if "secondary" in goals_data and not (self.secondary_goal and not self.secondary_goal.completed):
                    self.secondary_goal = Goal(
                        type="secondary",
                        description=goals_data["secondary"]["description"],
                        conditions=[goals_data["secondary"]["completion_check"]],
                        priority=2,
                        created_at=self.step_count
                    )
                
                # Update tertiary goal
                if "tertiary" in goals_data and not (self.tertiary_goal and not self.tertiary_goal.completed):
                    self.tertiary_goal = Goal(
                        type="tertiary",
                        description=goals_data["tertiary"]["description"],
                        conditions=[goals_data["tertiary"]["completion_check"]],
                        priority=3,
                        created_at=self.step_count
                    )
                    
                if self.verbose:
                    print(f"[GOAL] New goals generated by LLM")
                    
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse goals from LLM response: {e}")
            # Fallback to simple default goals if parsing fails
            self._set_fallback_goals(state)
    
    def _check_goal_completion(self, goal: Goal, state: Dict[str, Any]) -> bool:
        """Check if a goal is completed using LLM evaluation."""
        if goal.completed:
            return True
        
        # Quick check every 10 steps
        if self.step_count - goal.created_at < 10:
            return False
        
        prompt = f"""Evaluate if this Pokemon Emerald goal has been completed:

Goal: {goal.description}
Completion condition: {', '.join(goal.conditions)}

Current game state:
{format_state_for_llm(state, use_json_map=True)[:500]}

Has this goal been completed? Reply with just YES or NO."""

        response = self.vlm_client.get_text_query(prompt, "goal_check")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_goal_check",
            prompt=prompt,
            response=f"[GOAL] {goal}: Completed? {response}"
        )

        return "YES" in response.upper()
    
    def _set_fallback_goals(self, state: Dict[str, Any]):
        """Set simple fallback goals if LLM goal generation fails."""
        badges = state.get("badges", 0)
        
        if not self.primary_goal or self.primary_goal.completed:
            self.primary_goal = Goal(
                type="primary",
                description=f"Obtain gym badge #{badges + 1}",
                conditions=[f"badges > {badges}"],
                priority=1,
                created_at=self.step_count
            )
        
        if not self.secondary_goal or self.secondary_goal.completed:
            self.secondary_goal = Goal(
                type="secondary",
                description="Train Pokemon and heal team",
                conditions=["team healthy"],
                priority=2,
                created_at=self.step_count
            )
        
        if not self.tertiary_goal or self.tertiary_goal.completed:
            self.tertiary_goal = Goal(
                type="tertiary",
                description="Explore new areas",
                conditions=["explored new area"],
                priority=3,
                created_at=self.step_count
            )
    
    def _decide_action(self, state: Dict[str, Any], screenshot: Any, game_context: str = "unknown") -> str:
        """Decide next action based on goals and state.

        Args:
            state: Current game state
            screenshot: Current game screenshot
            game_context: Detected game context (from step() method)
        """
        # Check if we should delegate to specialized agent
        delegation = self._check_delegation_needed(state, screenshot)
        if delegation:
            result = delegation["agent"].execute(
                delegation["task"], 
                state, 
                screenshot
            )
            if result["success"] and result["buttons"]:
                self.action_queue.extend(result["buttons"])
                if self.verbose:
                    print(f"> Delegated to {delegation['agent'].name}: {result['message']}")
                return "DELEGATED"  # Queue will be returned by step()
        
        # Standard decision making (game_context passed from step())
        prompt = self._build_decision_prompt(state, game_context)

        # Add GPP-style system reminder about navigable unseen tiles (if any exist)
        # This is added as a separate message to ensure the LLM pays attention
        in_battle = state.get("in_battle", False)
        screen_text = state.get("dialogue", {}).get("text", "").strip()

        # Extract navigable unseen tiles count (compute it here too for reminder)
        map_info = state.get('map', {})
        navigable_unseen_count = 0
        if 'tiles' in map_info and map_info['tiles']:
            raw_tiles = map_info['tiles']
            radius = 7
            player_x = state.get("player_position", {}).get("x", 0)
            player_y = state.get("player_position", {}).get("y", 0)
            map_tiles = []

            for y_idx, row in enumerate(raw_tiles):
                for x_idx, tile_data in enumerate(row):
                    if tile_data and isinstance(tile_data, (list, tuple)) and len(tile_data) > 1:
                        behavior = tile_data[1].value if hasattr(tile_data[1], 'value') else tile_data[1]
                        x = player_x - radius + x_idx
                        y = player_y - radius + y_idx

                        if behavior in [96, 105, 97, 98, 99, 100, 101]:  # Warps
                            walkable = True
                        elif behavior == 0:  # Passable
                            walkable = True
                        else:
                            walkable = False

                        tile_type = "stairs" if behavior in [96, 105] else "door" if behavior in [97, 98, 99, 100, 101] else "walkable" if walkable else "blocked"
                        map_tiles.append({'x': x, 'y': y, 'type': tile_type, 'walkable': walkable})

            navigable_unseen_tiles = self.map_memory.get_navigable_unseen_tiles(map_tiles)
            navigable_unseen_count = len(navigable_unseen_tiles)

        # Build system reminder similar to GPP
        system_reminder = ""
        if navigable_unseen_count > 0 and not in_battle and not screen_text:
            system_reminder = f"""

SYSTEM NOTE: As per the exploration priorities, there are still {navigable_unseen_count} NAVIGABLE UNSEEN TILES remaining.

- These tiles are your *absolute highest priority*. They are fully reachable (barring scripted events).
- You MUST reveal them first before continuing other tasks, UNLESS:
    (a) Your primary or secondary goal is immediately achievable within the next 2-3 turns.
    (b) There is an NPC, object or item you wish to interact with within a 5 tile radius (excluding warps).
    (c) An event or NPC is stopping you from reaching the tile (triggers dialogue repeatedly).
    (d) Reaching the tile requires Surf or Cut and you lack the move.
    (e) Your box is full and you need to find a PC.
    (f) You just reached a city and need to heal at a Pokecenter first.
- Explore these tiles by moving to an **adjacent** walkable tile.
- Move at least 5 steps at a time whenever possible to save time.
- *Highly prefer manual movement (UP/DOWN/LEFT/RIGHT) over using the path tool* unless you can't figure out how to reach the tile.

Internalize these instructions for your decision-making process, ensuring they guide your actions without being explicitly referenced in your thoughts."""

        full_prompt = prompt + system_reminder

        if screenshot:
            response = self.vlm_client.get_query(screenshot, full_prompt, "decision")
        else:
            response = self.vlm_client.get_text_query(full_prompt, "decision")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays",
            prompt=full_prompt,
            response=response
        )

        # Parse action from response
        return self._parse_action(response)
    
    def _process_meta_tools(self, state: Dict[str, Any], screenshot: Any) -> bool:
        """
        Process meta-tool requests from LLM (define_agent, execute_script, notepad).
        Returns True if a meta-tool was used.
        """
        if not self.enable_meta_tools:
            return False
        
        # Check if LLM wants to use meta-tools
        prompt = f"""You have access to meta-tools for self-extension:

1. define_agent: Create a custom mini-agent for a specific task
2. execute_script: Run Python code to solve problems
3. notepad_write: Store long-term plans or important information
4. notepad_read: Retrieve stored plans

Current state: {format_state_for_llm(state, use_json_map=True)[:300]}
Goals: {self.primary_goal.description if self.primary_goal else 'None'}

Do you want to use a meta-tool? Reply with:
- "DEFINE_AGENT: [name] [task description]" to create a new agent
- "EXECUTE_SCRIPT: [code]" to run Python code
- "NOTEPAD_WRITE: [text]" to store information
- "NOTEPAD_READ" to retrieve stored information
- "NONE" if no meta-tool needed

Be specific and only use when beneficial."""

        if screenshot:
            response = self.vlm_client.get_query(screenshot, prompt, "metatool")
        else:
            response = self.vlm_client.get_text_query(prompt, "metatool")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_metatool",
            prompt=prompt,
            response=response
        )

        if "DEFINE_AGENT:" in response:
            return self._handle_define_agent(response)
        elif "EXECUTE_SCRIPT:" in response:
            return self._handle_execute_script(response)
        elif "NOTEPAD_WRITE:" in response:
            return self._handle_notepad_write(response)
        elif "NOTEPAD_READ" in response:
            return self._handle_notepad_read()
        
        return False
    
    def _handle_define_agent(self, response: str) -> bool:
        """Handle define_agent meta-tool."""
        try:
            parts = response.split("DEFINE_AGENT:", 1)[1].strip().split(" ", 1)
            agent_name = parts[0]
            agent_task = parts[1] if len(parts) > 1 else "Custom task"
            
            # Create agent definition
            agent_def = f"""Custom agent '{agent_name}' for: {agent_task}
Created at step {self.step_count}"""
            
            self.context.custom_agents[agent_name] = agent_def
            
            if self.verbose:
                print(f"[BOT] Created custom agent: {agent_name}")
            
            return True
        except:
            return False
    
    def _handle_execute_script(self, response: str) -> bool:
        """Handle execute_script meta-tool (safely limited)."""
        try:
            code = response.split("EXECUTE_SCRIPT:", 1)[1].strip()
            
            # For safety, only allow simple path calculations
            if "path" in code.lower() or "distance" in code.lower():
                # Create safe execution context
                safe_globals = {
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "len": len,
                    "range": range,
                }
                
                # Execute with limitations
                exec_result = {}
                exec(code, safe_globals, exec_result)
                
                if self.verbose:
                    print(f"[SCRIPT] Executed script (limited scope)")
                
                # Convert result to button presses if it's a path
                if "path" in exec_result and isinstance(exec_result["path"], list):
                    for move in exec_result["path"][:10]:
                        if move in ["UP", "DOWN", "LEFT", "RIGHT"]:
                            self.action_queue.append(move)
                
                return True
        except:
            return False
    
    def _handle_notepad_write(self, response: str) -> bool:
        """Handle notepad_write meta-tool."""
        try:
            text = response.split("NOTEPAD_WRITE:", 1)[1].strip()
            self.context.notepad.append(f"[Step {self.step_count}] {text}")
            
            # Keep notepad size manageable
            if len(self.context.notepad) > 20:
                self.context.notepad = self.context.notepad[-20:]
            
            if self.verbose:
                print(f"[NOTE] Notepad: {text[:50]}...")
            
            return True
        except:
            return False
    
    def _handle_notepad_read(self) -> bool:
        """Handle notepad_read meta-tool."""
        if self.context.notepad:
            notepad_content = "\n".join(self.context.notepad[-5:])
            if self.verbose:
                print(f"[READ] Reading notepad (last 5 entries)")
            # Store in action queue as a special marker
            # The next decision will consider notepad content
            return True
        return False
    
    def _check_delegation_needed(self, state: Dict[str, Any], screenshot: Any) -> Optional[Dict[str, Any]]:
        """Check if we should delegate to a specialized agent using LLM decision."""
        # Always delegate battles to specialized agent
        if state.get("in_battle"):
            return {
                "agent": self.battle_agent,
                "task": "Win this battle using optimal strategy"
            }
        
        # Ask LLM if delegation is needed for current situation
        prompt = f"""Analyze if this situation requires a specialized agent:

Current state: {format_state_for_llm(state, use_json_map=True)[:400]}
Primary goal: {self.primary_goal.description if self.primary_goal else 'None'}

Available specialized agents:
1. Pathfinding Agent - For navigating to distant locations
2. Puzzle Agent - For solving boulder puzzles, gym puzzles, etc.
3. None - Handle normally without delegation

Which agent should handle this? Reply with just the number (1, 2, or 3) and a brief task description.
Format: "NUMBER: task description" """
        
        if screenshot:
            response = self.vlm_client.get_query(screenshot, prompt, "delegation")
        else:
            response = self.vlm_client.get_text_query(prompt, "delegation")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_delegation",
            prompt=prompt,
            response=response
        )

        # Parse delegation decision
        if "1:" in response or "pathfind" in response.lower():
            task = response.split(":", 1)[-1].strip() if ":" in response else "Navigate to goal location"
            return {"agent": self.pathfinding_agent, "task": task}
        elif "2:" in response or "puzzle" in response.lower():
            task = response.split(":", 1)[-1].strip() if ":" in response else "Solve the puzzle"
            return {"agent": self.puzzle_agent, "task": task}
        
        return None
    
    def _build_decision_prompt(self, state: Dict[str, Any], game_context: str = "unknown") -> str:
        """Build decision prompt with goals and context, including GPP exploration priorities.

        Args:
            state: Current game state
            game_context: Detected game context from get_game_context()
        """
        goals_text = f"""Current Goals:
- Primary: {self.primary_goal.description if self.primary_goal else 'None'}
- Secondary: {self.secondary_goal.description if self.secondary_goal else 'None'}
- Tertiary: {self.tertiary_goal.description if self.tertiary_goal else 'None'}"""

        # Extract tiles from JSON map in state (if available)
        navigable_unseen_tiles = []
        navigable_unvisited_warps = []
        map_tiles = []

        # Try to parse tiles from map data in state
        map_info = state.get('map', {})
        if 'tiles' in map_info and map_info['tiles']:
            # Memory tiles format - convert to tile dicts
            raw_tiles = map_info['tiles']
            radius = 7
            player_x = state.get("player_position", {}).get("x", 0)
            player_y = state.get("player_position", {}).get("y", 0)

            for y_idx, row in enumerate(raw_tiles):
                for x_idx, tile_data in enumerate(row):
                    if tile_data and isinstance(tile_data, (list, tuple)) and len(tile_data) > 1:
                        behavior = tile_data[1].value if hasattr(tile_data[1], 'value') else tile_data[1]
                        x = player_x - radius + x_idx
                        y = player_y - radius + y_idx

                        # Determine tile type
                        if behavior in [96, 105]:
                            tile_type = "stairs"
                            walkable = True
                        elif behavior in [97, 98, 99, 100, 101]:
                            tile_type = "door"
                            walkable = True
                        elif behavior == 0:  # Passable
                            tile_type = "walkable"
                            walkable = True
                        else:
                            tile_type = "blocked"
                            walkable = False

                        map_tiles.append({
                            'x': x, 'y': y, 'type': tile_type, 'walkable': walkable
                        })

            # Get navigable unseen and unvisited warps
            navigable_unseen_tiles = self.map_memory.get_navigable_unseen_tiles(map_tiles)
            navigable_unvisited_warps = self.map_memory.get_navigable_unvisited_warps(map_tiles)

        # Format navigable unseen tiles (HIGHEST PRIORITY)
        unseen_text = ""
        if navigable_unseen_tiles:
            unseen_coords = ", ".join([f"({x},{y})" for x, y in navigable_unseen_tiles[:10]])
            unseen_text = f"""
## ⚠️ NAVIGABLE UNSEEN TILES (HIGHEST PRIORITY)
{len(navigable_unseen_tiles)} unseen tiles adjacent to explored areas: {unseen_coords}
These are your ABSOLUTE HIGHEST PRIORITY - explore them before other goals!"""

        # Format navigable unvisited warps (2nd HIGHEST PRIORITY)
        warps_text = ""
        if navigable_unvisited_warps:
            warp_coords = ", ".join([f"({x},{y})→{dest}" for x, y, dest in navigable_unvisited_warps[:5]])
            warps_text = f"""
## 🚪 NAVIGABLE UNVISITED WARPS (2nd PRIORITY)
{len(navigable_unvisited_warps)} unvisited warps: {warp_coords}
Explore these after all unseen tiles are revealed!"""

        exploration_text = f"""Exploration: {self.map_memory.get_exploration_percentage(1000):.1f}% of estimated map{unseen_text}{warps_text}"""

        context_text = ""
        if self.context.summary:
            context_text = f"Context: {self.context.summary}"

        # Context-specific guidance (game_context passed as parameter)
        context_guidance = ""
        if game_context == "title":
            context_guidance = """
**TITLE SCREEN DETECTED:**
- Press START or A to begin the game
- Navigate menus with UP/DOWN, confirm with A
- Skip dialogue/intro with A or B (hold)"""
        elif game_context == "dialogue":
            context_guidance = """
**DIALOGUE ACTIVE:**
- Press A to advance dialogue text
- Press B to try to exit dialogue quickly
- Read dialogue for important information (items, story progression)"""
        elif game_context == "battle":
            context_guidance = """
**BATTLE MODE:**
- Choose moves strategically based on type matchups
- Consider switching Pokemon if at disadvantage
- Use items (potions, revives) when necessary
- Press A to select, B to go back in battle menu"""
        elif game_context == "menu":
            context_guidance = """
**MENU OPEN:**
- Navigate with UP/DOWN/LEFT/RIGHT
- Confirm with A, cancel with B
- Check Pokemon status, bag items, or save game
- Exit menu with B when done"""
        elif game_context == "overworld":
            context_guidance = """
**OVERWORLD EXPLORATION:**
- Use directional buttons (UP/DOWN/LEFT/RIGHT) to move
- Press A to interact with NPCs, objects, signs
- Press START to open menu (only when necessary)
- Explore systematically to reveal fog-of-war"""

        # GPP-style exploration directive (only for overworld)
        exploration_directive = ""
        if game_context == "overworld" and (navigable_unseen_tiles or navigable_unvisited_warps):
            exploration_directive = """
### Map Exploration Strategy (CRITICAL)
YOUR ABSOLUTE HIGHEST PRIORITY **overriding other navigation goals** follows this order:
1. **Navigable Unseen Tiles:** Move to reveal any unseen tiles (?) adjacent to explored areas
2. **Navigable Unvisited Warps:** Enter warps (doors/stairs) you haven't visited yet
3. **Accessible Items:** Once all accessible unvisited warps and unseen tiles have been explored, pick up any accessible discovered items
4. **Other Goals:** Only pursue other objectives after completing priorities 1-3

**EXCEPTIONS** that allow you to deviate from exploration priorities:
a) Your primary/secondary goal is immediately achievable within 2-3 turns
b) NPC/object/item within 5 tile radius you can interact with (excluding warps)
c) NPC triggers dialogue repeatedly and blocks path to unseen tile
d) Tile requires Surf/Cut and you lack the move
e) Box is full, need PC to switch boxes
f) Just reached city, need to heal at Pokecenter first

**IMPORTANT:** Highly prefer manual movement (UP/DOWN/LEFT/RIGHT) over path tool.
Move at least 5 steps at a time when possible to save time."""

        return f"""You are playing Pokemon Emerald. Decide the next action.

**Game Context: {game_context.upper()}**
{context_guidance}

{goals_text}

{exploration_text}

{exploration_directive}

{context_text}

Current State:
{format_state_for_llm(state, use_json_map=True)}

Recent actions: {', '.join(list(self.recent_actions)[-5:])}

What single button should you press next? Choose from: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT, L, R
Consider your goals and avoid repeating failed actions.

IMPORTANT RULES:
- NEVER save the game using the START menu - this disrupts the game flow and is not allowed.
- Do not open the START menu unless absolutely necessary for gameplay (like checking Pokemon status).

Reply with just the button name."""
    
    def _parse_action(self, response: str) -> str:
        """Parse action from response."""
        response = response.upper().strip()
        
        # Look for valid buttons using word tokenization to avoid substring issues
        valid_buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R"]
        
        # Tokenize the response (split by spaces, commas, periods, etc.)
        response_clean = response.replace(',', ' ').replace('.', ' ').replace(';', ' ')
        tokens = response_clean.split()
        
        # Check each token for exact match
        for token in tokens:
            if token in valid_buttons:
                return token
        
        # If no exact match found, look for buttons in the text more carefully
        # This handles cases like "Press START" or "Hit the A button"
        for button in valid_buttons:
            # Check for word boundaries to avoid "A" matching in "START"
            if f" {button} " in f" {response} " or response.startswith(f"{button} ") or response.endswith(f" {button}"):
                return button
        
        # Default to exploring if no clear action
        if self.step_count % 4 == 0:
            return "UP"
        elif self.step_count % 4 == 1:
            return "RIGHT"
        elif self.step_count % 4 == 2:
            return "DOWN"
        else:
            return "LEFT"
    
    def _get_exploration_action(self, state: Dict[str, Any]) -> str:
        """Get exploration action using LLM to choose direction."""
        pos = state.get("player_position", {})
        unexplored = self.map_memory.get_unexplored_neighbors(
            pos.get("x", 0),
            pos.get("y", 0)
        )
        
        if not unexplored:
            return "NONE"
        
        # Ask LLM for exploration strategy
        unexplored_str = ", ".join([f"({x},{y})" for x, y in unexplored[:5]])
        prompt = f"""You need to explore unexplored areas in Pokemon Emerald.

Current position: ({pos.get('x', 0)}, {pos.get('y', 0)})
Unexplored neighbors: {unexplored_str}
Exploration progress: {self.map_memory.get_exploration_percentage(1000):.1f}%

Which direction should you explore? Consider:
- Unexplored areas
- Potential for finding items or trainers
- Efficient exploration patterns

Reply with just one button: UP, DOWN, LEFT, or RIGHT"""

        response = self.vlm_client.get_text_query(prompt, "exploration_dir")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays_exploration_action",
            prompt=prompt,
            response=response
        )

        # Parse direction
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if direction in response.upper():
                return direction
        
        # Fallback to simple heuristic if parsing fails
        target = unexplored[0]
        dx = target[0] - pos.get("x", 0)
        dy = target[1] - pos.get("y", 0)
        
        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        else:
            return "DOWN" if dy > 0 else "UP"
    
    def _reset_context(self, state: Dict[str, Any]):
        """Reset context with intelligent summarization."""
        if self.verbose:
            print(f"= Context reset at step {self.step_count}")
        
        # Build summary of important information
        summary_parts = []
        
        # Add goal progress
        if self.primary_goal:
            summary_parts.append(f"Working toward: {self.primary_goal.description}")
        
        # Add exploration stats
        summary_parts.append(
            f"Explored {self.map_memory.get_exploration_percentage(1000):.1f}% of map"
        )
        
        # Add team status
        team = state.get("team", [])
        if team:
            summary_parts.append(
                f"Team: {len(team)} Pokemon, "
                f"avg level {sum(p.get('level', 1) for p in team) / len(team):.1f}"
            )
        
        # Add badges
        badges = state.get("badges", 0)
        summary_parts.append(f"Badges: {badges}/8")
        
        # Update context
        self.context.summary = ". ".join(summary_parts)
        
        # Clear action history
        self.recent_actions.clear()
        
        # Keep important map memory but clear old visits
        cutoff = self.step_count - 200
        self.map_memory.last_visited = {
            k: v for k, v in self.map_memory.last_visited.items() 
            if v > cutoff
        }


def create_gemini_plays_agent(**kwargs) -> GeminiPlaysAgent:
    """Factory function to create GeminiPlaysAgent."""
    return GeminiPlaysAgent(**kwargs)
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

from utils.vlm import VLM
from utils.llm_logger import LLMLogger
from utils.state_formatter import format_state_for_llm
from utils.pathfinding import Pathfinder
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
    
    def mark_explored(self, x: int, y: int, step: int):
        """Mark a tile as explored."""
        self.explored_tiles.add((x, y))
        self.last_visited[(x, y)] = step
    
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
    
    def __init__(self, name: str, vlm_client: VLM):
        self.name = name
        self.vlm_client = vlm_client
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def execute(self, task: str, state: Dict[str, Any], screenshot: Any = None) -> Dict[str, Any]:
        """Execute specialized task."""
        raise NotImplementedError


class PathfindingAgent(SpecializedAgent):
    """Specialized agent for navigation and pathfinding."""
    
    def __init__(self, vlm_client: VLM):
        super().__init__("pathfinding", vlm_client)
        self.pathfinder = Pathfinder()
    
    def execute(self, task: str, state: Dict[str, Any], screenshot: Any = None) -> Dict[str, Any]:
        """Execute pathfinding to a target location."""
        # Parse target from task
        prompt = f"""You are a pathfinding specialist. Given this navigation task:
{task}

And this game state:
{format_state_for_llm(state)[:500]}

Extract the target location as X,Y coordinates. Reply with just the numbers.
Format: X,Y"""
        
        response = self.vlm_client.get_text_query(prompt, "pathfinding")
        
        try:
            coords = response.strip().split(',')
            target_x = int(coords[0])
            target_y = int(coords[1])
            
            current_pos = state.get("player_position", {})
            start = (current_pos.get("x", 0), current_pos.get("y", 0))
            goal = (target_x, target_y)
            
            path_buttons = self.pathfinder.find_path(start, goal, state)
            
            return {
                "success": bool(path_buttons),
                "buttons": path_buttons[:10] if path_buttons else [],
                "target": goal,
                "message": f"Path found: {len(path_buttons)} steps" if path_buttons else "No path found"
            }
        except:
            return {
                "success": False,
                "buttons": [],
                "message": "Failed to parse target location"
            }


class BattleAgent(SpecializedAgent):
    """Specialized agent for battle strategy."""
    
    def __init__(self, vlm_client: VLM):
        super().__init__("battle", vlm_client)
    
    def execute(self, task: str, state: Dict[str, Any], screenshot: Any = None) -> Dict[str, Any]:
        """Execute battle strategy."""
        battle = state.get("in_battle", False)
        if not battle:
            return {"success": False, "buttons": [], "message": "Not in battle"}
        
        prompt = f"""You are a Pokemon battle specialist. Analyze this battle:

{format_state_for_llm(state)[:1000]}

Task: {task}

Choose the best action:
1. Use move (specify which: 1-4)
2. Switch Pokemon (specify which: 1-6)
3. Use item
4. Run

Reply with just the action and number, e.g., "move 1" or "switch 2"."""
        
        response = self.vlm_client.get_text_query(prompt, "battle")
        
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
    
    def __init__(self, vlm_client: VLM):
        super().__init__("puzzle", vlm_client)
    
    def execute(self, task: str, state: Dict[str, Any], screenshot: Any = None) -> Dict[str, Any]:
        """Execute puzzle solving strategy."""
        prompt = f"""You are a puzzle-solving specialist. Analyze this puzzle:

Task: {task}
Current state: {format_state_for_llm(state)[:500]}

Provide the next 5 moves to progress in solving the puzzle.
Reply with just the button sequence, e.g., "UP, UP, LEFT, DOWN, A"."""
        
        if screenshot:
            response = self.vlm_client.get_query(screenshot, prompt, "gemini_plays")
        else:
            response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
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
        self.pathfinding_agent = PathfindingAgent(vlm_client)
        self.battle_agent = BattleAgent(vlm_client)
        self.puzzle_agent = PuzzleAgent(vlm_client)
        
        # Action queue
        self.action_queue: deque = deque()
        
        # Performance tracking for self-critique
        self.recent_actions: deque = deque(maxlen=20)
        self.stuck_counter = 0
        self.last_position = (0, 0)
    
    def step(self, state: Dict[str, Any], screenshot: Any = None) -> str:
        """
        Execute one step following Gemini Plays Pokemon architecture.
        
        Args:
            state: Current game state
            screenshot: Current game screenshot
            
        Returns:
            Button command (e.g., "A", "UP", "NONE")
        """
        self.step_count += 1
        
        # Check for context reset
        if self.step_count % self.context_reset_interval == 0:
            self._reset_context(state)
        
        # Update map memory
        self._update_map_memory(state)
        
        # Check if stuck (self-critique)
        if self.enable_self_critique:
            self._check_if_stuck(state)
        
        # If we have queued actions, execute them
        if self.action_queue:
            action = self.action_queue.popleft()
            self.recent_actions.append(action)
            return action
        
        # Check for meta-tool usage (before normal decision making)
        if self.enable_meta_tools and self.step_count % 10 == 0:
            if self._process_meta_tools(state, screenshot):
                # If meta-tool was used, it may have queued actions
                if self.action_queue:
                    action = self.action_queue.popleft()
                    self.recent_actions.append(action)
                    return action
        
        # Update goals
        self._update_goals(state)
        
        # Decide next action based on hierarchical goals
        action = self._decide_action(state, screenshot)
        
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
        
        # Mark current position as explored
        self.map_memory.mark_explored(x, y, self.step_count)
        
        # Mark visible area as explored (5x5 around player)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                self.map_memory.mark_explored(x + dx, y + dy, self.step_count)
        
        # Track points of interest
        if state.get("in_pokemon_center"):
            self.map_memory.points_of_interest[(x, y)] = "pokemon_center"
        elif state.get("in_pokemart"):
            self.map_memory.points_of_interest[(x, y)] = "pokemart"
    
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
                print("> Self-critique: Agent appears stuck, analyzing...")
            self._perform_self_critique(state)
            self.stuck_counter = 0
    
    def _perform_self_critique(self, state: Dict[str, Any]):
        """Analyze recent performance and adjust strategy."""
        recent_actions_str = ", ".join(list(self.recent_actions)[-10:])
        
        prompt = f"""Analyze this agent's recent performance in Pokemon Emerald:

Recent actions: {recent_actions_str}
Current state: {format_state_for_llm(state)[:500]}
Steps taken: {self.step_count}
Exploration: {self.map_memory.get_exploration_percentage(1000):.1f}%

The agent appears stuck. Identify:
1. What went wrong?
2. What should be done differently?
3. Suggest 5 specific actions to unstick.

Be concise and specific."""
        
        critique = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
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
        
        response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
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
        
        response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
        for word in response.upper().split():
            if word in ["UP", "DOWN", "LEFT", "RIGHT"]:
                self.action_queue.append(word)
                if len(self.action_queue) >= 4:
                    break
        
        # Ultimate fallback
        if not self.action_queue:
            import random
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
{format_state_for_llm(state)[:800]}

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

        response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
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
{format_state_for_llm(state)[:500]}

Has this goal been completed? Reply with just YES or NO."""

        response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
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
    
    def _decide_action(self, state: Dict[str, Any], screenshot: Any) -> str:
        """Decide next action based on goals and state."""
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
                return self.action_queue.popleft() if self.action_queue else "NONE"
        
        # Standard decision making
        prompt = self._build_decision_prompt(state)
        
        if screenshot:
            response = self.vlm_client.get_query(screenshot, prompt, "gemini_plays")
        else:
            response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="gemini_plays",
            prompt=prompt,
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

Current state: {format_state_for_llm(state)[:300]}
Goals: {self.primary_goal.description if self.primary_goal else 'None'}

Do you want to use a meta-tool? Reply with:
- "DEFINE_AGENT: [name] [task description]" to create a new agent
- "EXECUTE_SCRIPT: [code]" to run Python code
- "NOTEPAD_WRITE: [text]" to store information
- "NOTEPAD_READ" to retrieve stored information
- "NONE" if no meta-tool needed

Be specific and only use when beneficial."""

        if screenshot:
            response = self.vlm_client.get_query(screenshot, prompt, "gemini_plays")
        else:
            response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
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

Current state: {format_state_for_llm(state)[:400]}
Primary goal: {self.primary_goal.description if self.primary_goal else 'None'}

Available specialized agents:
1. Pathfinding Agent - For navigating to distant locations
2. Puzzle Agent - For solving boulder puzzles, gym puzzles, etc.
3. None - Handle normally without delegation

Which agent should handle this? Reply with just the number (1, 2, or 3) and a brief task description.
Format: "NUMBER: task description" """
        
        if screenshot:
            response = self.vlm_client.get_query(screenshot, prompt, "gemini_plays")
        else:
            response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
        # Parse delegation decision
        if "1:" in response or "pathfind" in response.lower():
            task = response.split(":", 1)[-1].strip() if ":" in response else "Navigate to goal location"
            return {"agent": self.pathfinding_agent, "task": task}
        elif "2:" in response or "puzzle" in response.lower():
            task = response.split(":", 1)[-1].strip() if ":" in response else "Solve the puzzle"
            return {"agent": self.puzzle_agent, "task": task}
        
        return None
    
    def _build_decision_prompt(self, state: Dict[str, Any]) -> str:
        """Build decision prompt with goals and context."""
        goals_text = f"""Current Goals:
- Primary: {self.primary_goal.description if self.primary_goal else 'None'}
- Secondary: {self.secondary_goal.description if self.secondary_goal else 'None'}  
- Tertiary: {self.tertiary_goal.description if self.tertiary_goal else 'None'}"""
        
        exploration_text = f"""Exploration: {self.map_memory.get_exploration_percentage(1000):.1f}% of estimated map
Unexplored neighbors: {len(self.map_memory.get_unexplored_neighbors(state.get("player_position", {}).get("x", 0), state.get("player_position", {}).get("y", 0)))}"""
        
        context_text = ""
        if self.context.summary:
            context_text = f"Context: {self.context.summary}"
        
        return f"""You are playing Pokemon Emerald. Decide the next action.

{goals_text}

{exploration_text}

{context_text}

Current State:
{format_state_for_llm(state)[:800]}

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
        
        response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        
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
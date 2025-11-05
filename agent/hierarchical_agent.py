#!/usr/bin/env python3
"""
Hierarchical Agent for Pokemon Emerald

This agent uses a multi-layered architecture, separating high-level strategy
from tactical execution. It features robust state management and history tracking.

Layers within the Agent:
1. Strategic Layer: Analyzes the game state to determine a high-level objective
   (e.g., "navigate_to:10,12").
2. Tactical Layer: Translates the high-level objective into specific MCP tool calls
   (e.g., calling the /mcp/navigate_to endpoint).

The Execution Layer (sending button presses to the emulator) is handled by the
client-server infrastructure, not directly by this agent.
"""

import logging
import os
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import requests
import json
import numpy as np
from PIL import Image

from utils.state_formatter import format_state_for_llm

logger = logging.getLogger(__name__)

@dataclass
class Objective:
    """Single objective/goal for the agent"""
    id: str
    description: str
    objective_type: str  # "location", "battle", "item", "dialogue", "custom"
    target_value: Optional[Any] = None  # Specific target (coords, trainer name, item name, etc.)
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    progress_notes: str = ""
    storyline: bool = False  # True for main storyline objectives (auto-verified), False for agent sub-objectives
    milestone_id: Optional[str] = None  # Emulator milestone ID for storyline objectives

@dataclass
class HistoryEntry:
    """Single entry in the agent's history"""
    timestamp: datetime
    player_coords: Optional[Tuple[int, int]]
    map_id: Optional[int]
    context: str  # "overworld", "battle", "menu", "title", "moving_van"
    action_taken: str
    game_state_summary: str

@dataclass
class HierarchicalAgentState:
    """Maintains state for the hierarchical agent"""
    # Core strategic state
    current_mode: str = "strategic"
    current_objective: Optional[str] = None
    high_level_plan: List[str] = field(default_factory=list)
    
    # History tracking
    history: deque = None
    recent_actions: deque = None
    
    # Objectives and knowledge
    objectives: List[Objective] = field(default_factory=list)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    
    # Movement memory
    failed_movements: Dict[str, List[str]] = field(default_factory=dict)
    npc_interactions: Dict[str, str] = field(default_factory=dict)
    stuck_detection: Dict[str, int] = field(default_factory=dict)
    
    # Step counter
    step_counter: int = 0
    
    def __post_init__(self):
        """Initialize deques if not provided"""
        if self.history is None:
            self.history = deque(maxlen=100)
        if self.recent_actions is None:
            self.recent_actions = deque(maxlen=50)

class HierarchicalAgent:
    """
    Hierarchical agent with three layers and robust state management.
    """

    def __init__(self, vlm, mcp_server_url: str, max_history_entries: int = 100,
                 max_recent_actions: int = 50, history_display_count: int = 15,
                 actions_display_count: int = 20):
        """
        Initialize the Hierarchical Agent.

        Args:
            vlm: The Vision Language Model instance
            mcp_server_url: The URL of the MCP server
            max_history_entries: Maximum history entries to keep
            max_recent_actions: Maximum recent actions to track
            history_display_count: Number of history entries to show in prompts
            actions_display_count: Number of recent actions to show in prompts
        """
        self.vlm = vlm
        self.mcp_server_url = mcp_server_url
        
        # Initialize consolidated state
        self.state = HierarchicalAgentState()
        self.state.history = deque(maxlen=max_history_entries)
        self.state.recent_actions = deque(maxlen=max_recent_actions)
        
        # Display parameters
        self.history_display_count = history_display_count
        self.actions_display_count = actions_display_count

        self._valid_actions = {'A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'}

        # Define context handlers
        self.context_handlers = {
            "overworld": self._handle_overworld,
            "battle": self._handle_battle,
            "title": self._handle_title,
            "menu": self._handle_menu,
        }
        
        # Initialize storyline objectives
        self._initialize_storyline_objectives()
        
        logger.info("Hierarchical Agent initialized with history tracking and checkpoint support")

    def _initialize_storyline_objectives(self):
        """Initialize the main storyline objectives for Pokémon Emerald progression"""
        storyline_objectives = [
            {
                "id": "story_game_start",
                "description": "Complete title sequence and begin the game",
                "objective_type": "system",
                "target_value": "Game Running",
                "milestone_id": "GAME_RUNNING"
            },
            {
                "id": "story_player_name_set",
                "description": "Set the player's name",
                "objective_type": "system",
                "target_value": "Player Name Set",
                "milestone_id": "PLAYER_NAME_SET"
            },
            {
                "id": "story_intro_complete",
                "description": "Complete intro cutscene with moving van",
                "objective_type": "cutscene",
                "target_value": "Intro Complete",
                "milestone_id": "INTRO_CUTSCENE_COMPLETE"
            },
            {
                "id": "story_littleroot_town",
                "description": "Arrive in Littleroot Town",
                "objective_type": "location",
                "target_value": "Littleroot Town",
                "milestone_id": "LITTLEROOT_TOWN"
            },
            {
                "id": "story_player_house",
                "description": "Enter player's house for the first time",
                "objective_type": "location",
                "target_value": "Player's House",
                "milestone_id": "PLAYER_HOUSE_ENTERED"
            },
            {
                "id": "story_player_bedroom",
                "description": "Go upstairs to player's bedroom",
                "objective_type": "location",
                "target_value": "Player's Bedroom",
                "milestone_id": "PLAYER_BEDROOM"
            },
            {
                "id": "story_clock_set",
                "description": "Set the clock on the wall in the player's bedroom. Interact with the clock (5,1) by pressing A while facing it. Then, leave the house.",
                "objective_type": "location",
                "target_value": "Clock Set",
                "milestone_id": "CLOCK_SET"
            },
            {
                "id": "story_rival_house",
                "description": "Visit May's house next door",
                "objective_type": "location",
                "target_value": "Rival's House",
                "milestone_id": "RIVAL_HOUSE"
            },
            {
                "id": "story_rival_bedroom",
                "description": "Visit May's bedroom on the second floor",
                "objective_type": "location",
                "target_value": "Rival's Bedroom",
                "milestone_id": "RIVAL_BEDROOM"
            },
            {
                "id": "story_route_101",
                "description": "Travel north to Route 101 and encounter Prof. Birch",
                "objective_type": "location",
                "target_value": "Route 101",
                "milestone_id": "ROUTE_101"
            },
            {
                "id": "story_starter_chosen",
                "description": "Choose starter Pokémon and receive first party member",
                "objective_type": "pokemon",
                "target_value": "Starter Pokémon",
                "milestone_id": "STARTER_CHOSEN"
            },
            {
                "id": "story_birch_lab",
                "description": "Visit Professor Birch's lab in Littleroot Town and receive the Pokedex",
                "objective_type": "location",
                "target_value": "Birch's Lab",
                "milestone_id": "BIRCH_LAB_VISITED"
            },
            {
                "id": "story_oldale_town",
                "description": "Leave lab and continue journey north to Oldale Town",
                "objective_type": "location",
                "target_value": "Oldale Town",
                "milestone_id": "OLDALE_TOWN"
            },
            {
                "id": "story_route_103",
                "description": "Travel to Route 103 to meet rival",
                "objective_type": "location",
                "target_value": "Route 103",
                "milestone_id": "ROUTE_103"
            },
            {
                "id": "story_rival_battle_1",
                "description": "Battle rival for the first time",
                "objective_type": "battle",
                "target_value": "Rival Battle 1",
                "milestone_id": "RIVAL_BATTLE_1"
            },
            {
                "id": "story_received_pokedex",
                "description": "Return to Birch's lab and receive the Pokédex",
                "objective_type": "item",
                "target_value": "Pokédex",
                "milestone_id": "RECEIVED_POKEDEX"
            },
            {
                "id": "story_route_102",
                "description": "Return through Route 102 toward Petalburg City",
                "objective_type": "location",
                "target_value": "Route 102",
                "milestone_id": "ROUTE_102"
            },
            {
                "id": "story_petalburg_city",
                "description": "Navigate to Petalburg City and visit Dad's gym",
                "objective_type": "location",
                "target_value": "Petalburg City",
                "milestone_id": "PETALBURG_CITY"
            },
            {
                "id": "story_dad_meeting",
                "description": "Meet Dad at Petalburg City Gym",
                "objective_type": "dialogue",
                "target_value": "Dad Meeting",
                "milestone_id": "DAD_FIRST_MEETING"
            },
            {
                "id": "story_gym_explanation",
                "description": "Receive explanation about Gym challenges",
                "objective_type": "dialogue",
                "target_value": "Gym Tutorial",
                "milestone_id": "GYM_EXPLANATION"
            },
            {
                "id": "story_route_104_south",
                "description": "Travel through southern section of Route 104",
                "objective_type": "location",
                "target_value": "Route 104 South",
                "milestone_id": "ROUTE_104_SOUTH"
            },
            {
                "id": "story_petalburg_woods",
                "description": "Navigate through Petalburg Woods to help Devon researcher",
                "objective_type": "location",
                "target_value": "Petalburg Woods",
                "milestone_id": "PETALBURG_WOODS"
            },
            {
                "id": "story_aqua_grunt",
                "description": "Defeat Team Aqua Grunt in Petalburg Woods",
                "objective_type": "battle",
                "target_value": "Aqua Grunt Defeated",
                "milestone_id": "TEAM_AQUA_GRUNT_DEFEATED"
            },
            {
                "id": "story_route_104_north",
                "description": "Travel through northern section of Route 104 to Rustboro",
                "objective_type": "location",
                "target_value": "Route 104 North",
                "milestone_id": "ROUTE_104_NORTH"
            },
            {
                "id": "story_rustboro_city",
                "description": "Arrive in Rustboro City and deliver Devon Goods",
                "objective_type": "location",
                "target_value": "Rustboro City",
                "milestone_id": "RUSTBORO_CITY"
            },
            {
                "id": "story_rustboro_gym",
                "description": "Enter the Rustboro Gym and challenge Roxanne",
                "objective_type": "location",
                "target_value": "Rustboro Gym",
                "milestone_id": "RUSTBORO_GYM_ENTERED"
            },
            {
                "id": "story_roxanne_defeated",
                "description": "Defeat Gym Leader Roxanne",
                "objective_type": "battle",
                "target_value": "Roxanne Defeated",
                "milestone_id": "ROXANNE_DEFEATED"
            },
            {
                "id": "story_stone_badge",
                "description": "Receive the Stone Badge and complete first gym",
                "objective_type": "badge",
                "target_value": "Stone Badge",
                "milestone_id": "FIRST_GYM_COMPLETE"
            }
        ]
        
        for obj_data in storyline_objectives:
            objective = Objective(
                id=obj_data["id"],
                description=obj_data["description"],
                objective_type=obj_data["objective_type"],
                target_value=obj_data["target_value"],
                completed=False,
                progress_notes="Storyline objective - verified by emulator milestones",
                storyline=True,
                milestone_id=obj_data["milestone_id"]
            )
            self.state.objectives.append(objective)

    def step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for the agent's decision-making process."""
        frame = game_state.get('frame')
        if frame is None:
            logger.error("🚫 No frame in game_state")
            return {"action": ["WAIT"]}
        
        # Check for black frame (transition)
        if self.is_black_frame(frame):
            logger.info("⏳ Black frame detected (transition), waiting...")
            return {"action": ["WAIT"]}
        
        # Increment step counter
        self.state.step_counter += 1
        
        # Check storyline milestones and auto-complete objectives
        self.check_storyline_milestones(game_state)
        
        # Handle Moving Van special case
        player_location = game_state.get("player", {}).get("location", "")
        
        # Check if the next milestone (Littleroot Town) is completed.
        # If not, we might still be in the moving van sequence.
        littleroot_objective = next((obj for obj in self.state.objectives if obj.id == "story_littleroot_town"), None)
        littleroot_milestone_completed = littleroot_objective.completed if littleroot_objective else False

        actions = None
        game_context = "moving_van"  # for history
        if player_location == "MOVING_VAN" and not littleroot_milestone_completed:
            logger.info("🚚 In moving van, executing special action sequence.")
            actions = self._handle_moving_van(game_state)
        else:
            # Determine game context
            game_context = self._determine_game_context(game_state)

            # Route to appropriate handler
            handler = self.context_handlers.get(game_context, self._handle_default)
            actions = handler(game_state)

        # Record in history
        coords = self.get_player_coords(game_state)
        map_id = self.get_map_id(game_state)
        game_state_summary = self.create_game_state_summary(game_state)
        
        history_entry = HistoryEntry(
            timestamp=datetime.now(),
            player_coords=coords,
            map_id=map_id,
            context=game_context,
            action_taken=str(actions),
            game_state_summary=game_state_summary
        )
        self.state.history.append(history_entry)
        
        # Update recent actions
        if isinstance(actions, list):
            self.state.recent_actions.extend(actions)
        else:
            self.state.recent_actions.append(actions)
        
        return {"action": actions}

    def _determine_game_context(self, game_state: Dict[str, Any]) -> str:
        """Determine current game context with comprehensive detection."""
        try:
            # Check for title sequence
            player_location = game_state.get("player", {}).get("location", "")
            if player_location == "TITLE_SEQUENCE":
                return "title"
            
            game_state_value = game_state.get("game", {}).get("game_state", "").lower()
            if "title" in game_state_value or "intro" in game_state_value:
                return "title"
            
            player_name = game_state.get("player", {}).get("name", "").strip()
            if not player_name or player_name == "????????":
                return "title"
            
            # Check for battle
            is_in_battle = game_state.get("game", {}).get("is_in_battle", False)
            if is_in_battle:
                return "battle"
            
            # Check for menu
            player_state = game_state.get("player", {})
            if player_state.get("in_menu", False):
                return "menu"
            
            return "overworld"
            
        except Exception as e:
            logger.warning(f"Error determining game context: {e}")
            return "unknown"

    def is_black_frame(self, frame) -> bool:
        """Check if the frame is mostly black (transition/loading screen)."""
        try:
            if hasattr(frame, 'convert'):
                img = frame
            elif hasattr(frame, 'shape'):
                img = Image.fromarray(frame)
            else:
                return False
            
            img_array = np.array(img)
            mean_brightness = np.mean(img_array)
            std_dev = np.std(img_array)
            
            is_black = mean_brightness < 10 or (mean_brightness < 30 and std_dev < 5)
            if is_black:
                logger.debug(f"Black frame: mean={mean_brightness:.2f}, std={std_dev:.2f}")
            
            return is_black
            
        except Exception as e:
            logger.warning(f"Error checking for black frame: {e}")
            return False

    def get_player_coords(self, game_state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract player coordinates from game state with fallback logic."""
        try:
            player = game_state.get("player", {})
            position = player.get("position", {})
            if position:
                x, y = position.get("x"), position.get("y")
                if x is not None and y is not None:
                    return (x, y)
            
            x, y = player.get("x"), player.get("y")
            if x is not None and y is not None:
                return (x, y)
        except Exception as e:
            logger.warning(f"Error getting player coords: {e}")
        return None

    def get_map_id(self, game_state: Dict[str, Any]) -> Optional[int]:
        """Extract map ID from game state."""
        try:
            return game_state.get("map", {}).get("id")
        except Exception as e:
            logger.warning(f"Error getting map ID: {e}")
        return None

    def create_game_state_summary(self, game_state: Dict[str, Any]) -> str:
        """Create a concise summary of the current game state."""
        try:
            summary_parts = []
            coords = self.get_player_coords(game_state)
            if coords:
                summary_parts.append(f"Player at ({coords[0]}, {coords[1]})")
            
            map_id = self.get_map_id(game_state)
            if map_id:
                summary_parts.append(f"Map {map_id}")
            
            context = self._determine_game_context(game_state)
            if context == "battle":
                summary_parts.append("In battle")
            elif context == "dialogue":
                dialogue_text = game_state.get("game", {}).get("dialogue", {}).get("text", "")
                if dialogue_text:
                    summary_parts.append(f"Dialogue: {dialogue_text[:30]}...")
            
            return " | ".join(summary_parts) if summary_parts else "Unknown state"
        except Exception as e:
            logger.warning(f"Error creating game state summary: {e}")
            return "Error reading state"

    def check_storyline_milestones(self, game_state: Dict[str, Any]) -> List[str]:
        """Check emulator milestones and auto-complete corresponding storyline objectives."""
        completed_ids = []
        milestones = game_state.get("milestones", {})
        if not milestones:
            return completed_ids

        for obj in self.state.objectives:
            if obj.storyline and obj.milestone_id and not obj.completed:
                milestone_completed = milestones.get(obj.milestone_id, {}).get("completed", False)
                if milestone_completed:
                    obj.completed = True
                    obj.completed_at = datetime.now()
                    obj.progress_notes = f"Auto-completed by milestone: {obj.milestone_id}"
                    completed_ids.append(obj.id)
                    logger.info(f"✅ Auto-completed storyline objective: {obj.description}")

        return completed_ids

    def _handle_overworld(self, game_state: Dict[str, Any]) -> List[str]:
        """Handles decision-making when in the overworld."""
        action, objective = self._run_strategic_layer(game_state)
        if action:
            return [action]
        
        if objective:
            actions = self._run_tactical_layer(objective, game_state)
            return actions
        
        return ["WAIT"]

    def _run_strategic_layer(self, game_state: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """The strategic layer decides on the high-level goal or immediate action."""
        prompt = self._format_strategic_prompt(game_state)
        frame = game_state.get("frame")
        
        try:
            response_text = self.vlm.get_query(frame, prompt, "strategic_layer")
        except Exception as e:
            logger.error(f"VLM call failed in strategic layer: {e}")
            return None, "wait"

        action, objective = self._parse_strategic_response(response_text)
        
        if action:
            logger.info(f"Strategic Layer decided action: {action}")
            return action, None

        if objective:
            self.state.current_objective = objective
            logger.info(f"Strategic Layer set objective: {objective}")
            return None, objective
            
        return None, "wait"

    def _format_strategic_prompt(self, game_state: Dict[str, Any]) -> str:
        """Formats the prompt for the strategic layer with rich context."""
        player_location = game_state.get("player", {}).get("location", "Unknown")
        badges = game_state.get("game", {}).get("badges", [])
        party = game_state.get("player", {}).get("party", [])
        coords = self.get_player_coords(game_state)
        
        # Get exploration status and map
        unseen_tiles_info = "Not available."
        visual_map = game_state.get("map", {}).get("visual_map", "")
        clean_map = ""
        legend_text = ""
        if visual_map:
            clean_map, legend_text, _ = self.preprocess_visual_map(visual_map)
            if '?' in visual_map:
                unseen_tiles_info = "There are unseen reachable tiles. Your top priority is to explore them."
            else:
                unseen_tiles_info = "All reachable tiles in the current area have been explored."

        # Get objectives
        active_objectives = [obj for obj in self.state.objectives if not obj.completed]
        objectives_str = json.dumps([{
            "id": obj.id, "description": obj.description,
            "type": obj.objective_type, "target": str(obj.target_value)
        } for obj in active_objectives[:5]], indent=2)

        # Get history and stuck warning
        history_summary = self.get_relevant_history_summary()
        stuck_warning = self.get_stuck_warning(coords, "overworld", game_state)
        recent_actions_str = ', '.join(list(self.state.recent_actions)[-self.actions_display_count:])
        movement_memory = self.get_area_movement_memory(coords) if coords else ""

        prompt = f"""You are the strategic planner for a Pokémon Emerald agent.
Your primary directive is to explore every reachable tile and progress through the game.

First, check the screen for any dialogue boxes.
- If a dialogue box is visible, your only goal is to advance it. The best action is usually 'A' or 'B'.
- If there is no dialogue, proceed with strategic planning.

RECENT ACTION HISTORY (last {self.actions_display_count} actions):
{recent_actions_str}

LOCATION/CONTEXT HISTORY (last {self.history_display_count} steps):
{history_summary}

CURRENT STATUS:
- Location: {player_location}
- Coordinates: {coords}
- Badges: {len(badges)}
- Party Size: {len(party)}
- Exploration Status: {unseen_tiles_info}

MAP:
{clean_map}

LEGEND:
{legend_text}

ACTIVE OBJECTIVES:
{objectives_str}

{movement_memory}

{stuck_warning}

AVAILABLE ACTIONS (if dialogue or interactive menu is present):
- A, B, UP, DOWN, LEFT, RIGHT
- For simple text boxes, 'A' or 'B' is usually sufficient.
- For interactive menus (like setting a clock), use directional buttons as needed based on the screen.

AVAILABLE COMMANDS (if no dialogue):
- navigate_to:<X,Y>
- get_world_map
- get_navigation_hints:<target_area_name>
- add_objective:<type>:<description>:<target>
- complete_objective:<id>

Based on the screen and current status, what is your next action or objective?
The '?' symbol on the map marks tiles that are adjacent to explored areas but have not been visited yet. To explore the map fully, you should navigate to these tiles.
If there is dialogue, provide an ACTION.
If there are unseen tiles ('?') and no dialogue, you MUST provide a 'navigate_to:<X,Y>' OBJECTIVE to explore one of them.
For long-distance travel, first use `get_world_map` to understand your surroundings, then `get_navigation_hints` to plan your route.
Provide your response in ONE of the following formats:
ACTION: <your_action_here>
OBJECTIVE: <your_objective_here>
"""
        return prompt

    def _parse_strategic_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """Parses action or objective from the strategic VLM's response."""
        action = None
        objective = None
        for line in response.split('\n'):
            line_upper = line.upper()
            if line_upper.startswith("ACTION:"):
                action = line.split(":", 1)[1].strip().upper()
                break  # Action takes precedence
            elif line_upper.startswith("OBJECTIVE:"):
                objective = line.split(":", 1)[1].strip()
        
        return action, objective

    def get_relevant_history_summary(self) -> str:
        """Get a concise summary of relevant recent history."""
        if not self.state.history:
            return "No previous history."
        
        recent_entries = list(self.state.history)[-self.history_display_count:]
        summary_lines = []
        for i, entry in enumerate(recent_entries, 1):
            coord_str = f"({entry.player_coords[0]},{entry.player_coords[1]})" if entry.player_coords else "(?)"
            summary_lines.append(f"{i}. {entry.context} at {coord_str}: {entry.action_taken}")
        
        return "\n".join(summary_lines)

    def get_stuck_warning(self, coords: Optional[Tuple[int, int]], context: str, game_state: Dict[str, Any]) -> str:
        """Generate warning text if stuck pattern detected."""
        if context == "title":
            return ""
        
        if not coords:
            return ""
        
        key = f"{coords[0]}_{coords[1]}_{context}"
        self.state.stuck_detection[key] = self.state.stuck_detection.get(key, 0) + 1
        
        if self.state.stuck_detection[key] >= 8:
            return "\n⚠️ WARNING: You appear to be stuck at this location. Try a different approach!"
        return ""

    def get_area_movement_memory(self, center_coords: Tuple[int, int], radius: int = 7) -> str:
        """Get movement memory for the area around the player."""
        if not center_coords:
            return ""
        
        cx, cy = center_coords
        memory_lines = []
        nearby_memories = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                check_coords = (cx + dx, cy + dy)
                coord_key = f"{check_coords[0]},{check_coords[1]}"
                
                if coord_key in self.state.failed_movements:
                    failed_list = self.state.failed_movements[coord_key]
                    nearby_memories.append(f"({check_coords[0]},{check_coords[1]}): Failed moves: {', '.join(failed_list)}")
                
                if coord_key in self.state.npc_interactions:
                    interaction = self.state.npc_interactions[coord_key]
                    nearby_memories.append(f"({check_coords[0]},{check_coords[1]}): NPC: {interaction}")
        
        if nearby_memories:
            memory_lines.append("🧠 MOVEMENT MEMORY (nearby area):")
            for memory in nearby_memories[:5]:
                memory_lines.append(f"  {memory}")
        
        return "\n".join(memory_lines)

    @staticmethod
    def preprocess_visual_map(visual_map: str):
        lines = visual_map.splitlines()
        grid_lines = []
        inside_map = False

        for line in lines:
            # start collecting after the "--- MAP:" header
            if line.strip().startswith("--- MAP:"):
                inside_map = True
                continue
            # stop before "Player at" or "Legend"
            if line.strip().startswith("Player at") or line.strip().startswith("Legend:"):
                break
            # keep coordinate and map lines
            if inside_map and line.strip():
                grid_lines.append(line)

        # extract legend
        legend_match = re.search(r"Legend:\s*(.*?)\n\n", visual_map, re.DOTALL)
        legend_text = legend_match.group(1).strip() if legend_match else ""
        
        # extract portals
        portal_match = re.search(r"Portal Connections:\s*(.*)", visual_map, re.DOTALL)
        portal_text = portal_match.group(1).strip() if portal_match else "None"

        # rebuild clean map (preserve axis coordinates)
        clean_grid = "\n".join(grid_lines)

        return clean_grid, legend_text, portal_text

    def _run_tactical_layer(self, objective: str, game_state: Dict[str, Any]) -> List[str]:
        """Breaks down the high-level objective into executable actions."""
        if objective.startswith("navigate_to:"):
            destination = objective.split(":", 1)[1]
            return self._execute_navigation(destination, game_state)
        elif objective == "get_world_map":
            return self._get_world_map()
        elif objective.startswith("get_navigation_hints:"):
            target_area_name = objective.split(":", 1)[1]
            return self._get_navigation_hints(target_area_name)
        elif objective.startswith("add_objective:"):
            parts = objective.split(":", 3)
            if len(parts) >= 3:
                obj_type, description = parts[1], parts[2]
                target = parts[3] if len(parts) > 3 else None
                self.add_objective(description, obj_type, target)
            return ["WAIT"]
        elif objective.startswith("complete_objective:"):
            obj_id = objective.split(":", 1)[1]
            self.complete_objective(obj_id)
            return ["WAIT"]
        return ["WAIT"]

    def add_objective(self, description: str, objective_type: str, target_value: Any = None):
        """Add a new objective."""
        obj_id = f"obj_{len(self.state.objectives)}_{int(datetime.now().timestamp())}"
        objective = Objective(
            id=obj_id,
            description=description,
            objective_type=objective_type,
            target_value=target_value,
            storyline=False
        )
        self.state.objectives.append(objective)
        logger.info(f"Added objective: {description}")

    def complete_objective(self, obj_id: str):
        """Mark an objective as completed (storyline objectives cannot be manually completed)."""
        for obj in self.state.objectives:
            if obj.id == obj_id and not obj.completed:
                if obj.storyline:
                    logger.warning(f"Cannot manually complete storyline objective: {obj.description}")
                    return
                obj.completed = True
                obj.completed_at = datetime.now()
                logger.info(f"Completed objective: {obj.description}")
                return

    def _execute_navigation(self, destination: str, game_state: Dict[str, Any]) -> List[str]:
        """Executes a navigation task using the pathfinder."""
        target_coords = self._get_coords_for_location(destination)
        if not target_coords:
            logger.error(f"Could not parse coordinates for: {destination}")
            return ["WAIT"]

        try:
            response = requests.post(f"{self.mcp_server_url}/mcp/navigate_to", json={
                "x": target_coords["x"],
                "y": target_coords["y"],
                "reason": f"Navigating to {destination}"
            })
            response.raise_for_status()
            result = response.json()
            if result.get("success"):
                logger.info(f"Pathfinding successful")
                return ["WAIT"]
            else:
                logger.error(f"Pathfinding failed: {result.get('error')}")
                return ["WAIT"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call navigate_to: {e}")
            return ["WAIT"]

    def _get_world_map(self) -> List[str]:
        """Calls the MCP server to get the world map."""
        try:
            response = requests.post(f"{self.mcp_server_url}/mcp/get_world_map", json={})
            response.raise_for_status()
            result = response.json()
            if result.get("success"):
                logger.info("Successfully retrieved world map.")
                # The result is informational for the LLM, so we just wait.
                return ["WAIT"]
            else:
                logger.error(f"Failed to get world map: {result.get('error')}")
                return ["WAIT"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call get_world_map: {e}")
            return ["WAIT"]

    def _get_navigation_hints(self, target_area_name: str) -> List[str]:
        """Calls the MCP server to get navigation hints."""
        try:
            response = requests.post(f"{self.mcp_server_url}/mcp/get_navigation_hints", json={
                "target_area_name": target_area_name
            })
            response.raise_for_status()
            result = response.json()
            if result.get("success"):
                logger.info(f"Successfully retrieved navigation hints for {target_area_name}.")
                # The result is informational for the LLM, so we just wait.
                return ["WAIT"]
            else:
                logger.error(f"Failed to get navigation hints: {result.get('error')}")
                return ["WAIT"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call get_navigation_hints: {e}")
            return ["WAIT"]

    def _get_coords_for_location(self, location_name: str) -> Dict[str, int]:
        """Retrieve coordinates from a string, handling various formats."""
        if not location_name or "," not in location_name:
            return None

        try:
            # Clean the string to remove anything that's not a digit, comma, or minus sign
            cleaned_name = re.sub(r"[^0-9,\-]", "", location_name)
            
            x_str, y_str = cleaned_name.split(',')
            
            # Final strip to remove any leftover whitespace if any
            x = int(x_str.strip())
            y = int(y_str.strip())
            
            return {"x": x, "y": y}
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse coordinates from '{location_name}': {e}")
            return None

    def _handle_battle(self, game_state: Dict[str, Any]) -> List[str]:
        """Handles decision-making during a battle."""
        prompt = self._format_battle_prompt(game_state)
        frame = game_state.get("frame")
        try:
            response_text = self.vlm.get_query(frame, prompt, "battle_handler")
        except Exception as e:
            logger.error(f"VLM call failed in battle: {e}")
            return ["A"]

        action = self._parse_action_from_response(response_text)
        return [action]

    def _format_battle_prompt(self, game_state: Dict[str, Any]) -> str:
        """Formats the prompt for the battle layer."""
        player_pokemon = game_state.get("player", {}).get("party", [{}])[0]
        prompt = f"""
You are in a Pokémon battle. Your goal is to win.

YOUR POKEMON:
- Species: {player_pokemon.get('species_name', 'Unknown')}
- Level: {player_pokemon.get('level', '??')}
- HP: {player_pokemon.get('current_hp', '??')}/{player_pokemon.get('max_hp', '??')}
- Status: {player_pokemon.get('status', 'OK')}
- Moves: {player_pokemon.get('moves', [])}

AVAILABLE ACTIONS: A, B, UP, DOWN, LEFT, RIGHT
Based on the battle situation, what is the best action?
ACTION: <your_action_here>
"""
        return prompt

    def _handle_default(self, game_state: Dict[str, Any]) -> List[str]:
        """Handles default game states like menus or dialogs."""
        return ["B"]

    def _handle_title(self, game_state: Dict[str, Any]) -> List[str]:
        """Handles the title screen and introductory sequence."""
        prompt = self._format_title_prompt(game_state)
        frame = game_state.get("frame")
        try:
            response_text = self.vlm.get_query(frame, prompt, "title_handler")
        except Exception as e:
            logger.error(f"VLM call failed in title handler: {e}")
            return ["START"]  # Default action on error

        action = self._parse_action_from_response(response_text)
        return [action]

    def _format_title_prompt(self, game_state: Dict[str, Any]) -> str:
        """Formats the prompt for the title screen handler."""
        return """
You are at the beginning of the game, on the title screen or in the introductory sequence (e.g., character selection, naming).
Your goal is to start the game. Look at the screen and determine the single best button press to advance.

AVAILABLE ACTIONS: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT
ACTION: <your_action_here>
"""

    def _handle_menu(self, game_state: Dict[str, Any]) -> List[str]:
        """Handles in-game menus with VLM assistance."""
        prompt = self._format_menu_prompt(game_state)
        frame = game_state.get("frame")
        try:
            response_text = self.vlm.get_query(frame, prompt, "menu_handler")
        except Exception as e:
            logger.error(f"VLM call failed in menu handler: {e}")
            return ["B"]  # Default action to exit menu on error

        action = self._parse_action_from_response(response_text)
        return [action]

    def _format_menu_prompt(self, game_state: Dict[str, Any]) -> str:
        """Formats the prompt for the menu handler."""
        return """
You are in a menu. Your goal is to navigate it correctly based on your current high-level objective.
Look at the screen and determine the single best button press to advance.

AVAILABLE ACTIONS: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT
ACTION: <your_action_here>
"""

    def _parse_action_from_response(self, response: str) -> str:
        """Parses a single action from the VLM's response."""
        for line in response.split('\n'):
            if line.upper().startswith("ACTION:"):
                return line.split(":", 1)[1].strip().upper()
        return "A"

    def _handle_moving_van(self, game_state: Dict[str, Any]) -> List[str]:
        """Handles the moving van sequence with a text-only VLM query."""
        prompt = (
            f"Pokemon Emerald - Moving Van Navigation\n"
            f"You are inside the moving van. To exit, you need to move right three times. Return a list of actions.\n\n"
            f"Return exactly one line: ACTION_LIST: [BUTTON, BUTTON, ...]\n"
            f"Example: ACTION_LIST: [RIGHT, RIGHT, RIGHT]\n"
            f"Valid buttons: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, WAIT\n"
            f"Do not add any other text."
        )
        try:
            frame = game_state.get("frame")
            response_text = self.vlm.get_query(frame, prompt, "moving_van_handler")
            return self._parse_action_list(response_text)
        except Exception as e:
            logger.error(f"VLM call failed in moving van handler: {e}")
            return ["RIGHT", "RIGHT", "RIGHT"]  # Fallback

    def _parse_action_list(self, response: str) -> List[str]:
        """Parse ACTION_LIST: [BUTTON, ...] from VLM response. Returns a list of valid buttons."""
        if response is None:
            return ['WAIT']
        try:
            match = re.search(r"ACTION_LIST\s*:\s*\[(.*?)\]", response, re.IGNORECASE)
            if match:
                actions_str = match.group(1)
                # Split by comma, strip whitespace, remove quotes, convert to uppercase
                raw_actions = [a.strip().strip("'\"") for a in actions_str.split(',')]
                valid_actions = [a.upper() for a in raw_actions if a.upper() in self._valid_actions]
                return valid_actions if valid_actions else ['WAIT']
        except Exception:
            logger.warning(f"Could not parse action list from response: {response}")
        return ['WAIT']

    def save_checkpoint(self, checkpoint_file: str = None):
        """Save agent state to checkpoint file."""
        try:
            from utils.llm_logger import get_llm_logger
            llm_logger = get_llm_logger()
            if llm_logger:
                llm_logger.save_checkpoint(checkpoint_file, agent_step_count=self.state.step_counter)
                logger.info(f"💾 Saved checkpoint: step {self.state.step_counter}")
                return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
        return False

    def load_checkpoint(self, checkpoint_file: str):
        """Load agent state from checkpoint file."""
        try:
            from utils.llm_logger import get_llm_logger
            llm_logger = get_llm_logger()
            if llm_logger:
                restored_step_count = llm_logger.load_checkpoint(checkpoint_file)
                if restored_step_count is not None:
                    self.state.step_counter = restored_step_count
                    logger.info(f"✅ Loaded checkpoint: step {restored_step_count}")
                    return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
        return False

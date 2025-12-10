"""
Simple Agent Module

Provides a streamlined approach for direct frame + state -> action processing,
with enhanced history tracking to prevent getting stuck in loops.

Key improvements over the original simple mode:
- Location-based stuck detection (tracks repeated actions at same coordinates)
- Context-aware history (overworld/battle/menu/dialogue awareness)  
- Memory management to fit within LLM context limits
- Detailed history tracking with timestamps and game state summaries
- Smart context switching that helps agent avoid infinite loops
- Configurable history window sizes for different use cases
- Chain of thought reasoning with structured LLM responses
- Objectives system with automatic and manual completion tracking
- Dynamic goal setting and progress monitoring

The agent maintains objectives (go to location, battle trainer, etc.) that are
automatically tracked and marked complete when achieved. The LLM can also
manually complete objectives and create new ones dynamically through structured
commands. It uses chain of thought reasoning to make better decisions while
considering current objectives. All state including objectives is forwarded
to support external monitoring and debugging.

Configuration defaults (can be customized):
- 100 previous state/location entries (with context and reasoning)
- 50 recent button presses tracked  
- 15 history entries shown to LLM in prompts
- 20 recent actions shown to LLM in prompts
- Automatic memory management to stay within LLM context limits
"""

import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from utils.state_formatter import format_state_for_llm
from agent.agent_utils import VisualMapGenerator, PathfinderUtility
from agent.direct_objectives import DirectObjectiveManager
from utils.vlm import VLM

logger = logging.getLogger(__name__)

# Configurable parameters for history tracking
DEFAULT_MAX_HISTORY_ENTRIES = 100  # Previous states/locations with context
DEFAULT_MAX_RECENT_ACTIONS = 50    # Recent button presses
DEFAULT_HISTORY_DISPLAY_COUNT = 99 # Number of history entries shown to LLM
DEFAULT_ACTIONS_DISPLAY_COUNT = 20 # Number of recent actions shown to LLM


def configure_simple_agent_defaults(max_history_entries: int = None, max_recent_actions: int = None, 
                                  history_display_count: int = None, actions_display_count: int = None):
    """Configure default parameters for all new SimpleAgent instances"""
    global DEFAULT_MAX_HISTORY_ENTRIES, DEFAULT_MAX_RECENT_ACTIONS
    global DEFAULT_HISTORY_DISPLAY_COUNT, DEFAULT_ACTIONS_DISPLAY_COUNT
    
    if max_history_entries is not None:
        DEFAULT_MAX_HISTORY_ENTRIES = max_history_entries
    if max_recent_actions is not None:
        DEFAULT_MAX_RECENT_ACTIONS = max_recent_actions
    if history_display_count is not None:
        DEFAULT_HISTORY_DISPLAY_COUNT = history_display_count
    if actions_display_count is not None:
        DEFAULT_ACTIONS_DISPLAY_COUNT = actions_display_count
        
    logger.info(f"Updated SimpleAgent defaults: {DEFAULT_MAX_HISTORY_ENTRIES} history, {DEFAULT_MAX_RECENT_ACTIONS} actions, "
               f"display {DEFAULT_HISTORY_DISPLAY_COUNT}/{DEFAULT_ACTIONS_DISPLAY_COUNT}")

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
    context: str  # "overworld", "battle", "menu", "dialogue"
    action_taken: str
    game_state_summary: str

@dataclass
class SimpleAgentState:
    """Maintains history and state for the simple agent"""
    # Note: We don't use defaults here because they're captured at class definition time
    history: deque = None
    recent_actions: deque = None
    stuck_detection: Dict[str, int] = field(default_factory=dict)
    step_counter: int = 0
    objectives: List[Objective] = field(default_factory=list)
    objectives_updated: bool = False
    failed_movements: Dict[str, List[str]] = field(default_factory=dict)  # coord_key -> [failed_directions]
    npc_interactions: Dict[str, str] = field(default_factory=dict)  # coord_key -> interaction_notes
    
    def __post_init__(self):
        """Initialize deques with current default values"""
        if self.history is None:
            self.history = deque(maxlen=DEFAULT_MAX_HISTORY_ENTRIES)
        if self.recent_actions is None:
            self.recent_actions = deque(maxlen=DEFAULT_MAX_RECENT_ACTIONS)

class MySimpleAgent:
    """
    Simple agent that processes frame + state -> action directly with history tracking
    """
    
    def __init__(self, vlm_or_args, max_history_entries: int = None, max_recent_actions: int = None, 
                 history_display_count: int = None, actions_display_count: int = None, 
                 use_direct_objectives: bool = False, direct_sequence: str = None):
        # Handle both VLM and args - if args provided, create VLM
        if hasattr(vlm_or_args, 'backend'):  # It's args
            backend = getattr(vlm_or_args, "backend", "gemini")
            model_name = getattr(vlm_or_args, "model_name", "gemini-2.5-flash")
            self.vlm = VLM(model_name=model_name, backend=backend)
            print(f"   VLM: {backend}/{model_name}")
            print(f"   Mode: MySimple (forked simple agent for custom modifications)")
        else:  # It's a VLM
            self.vlm = vlm_or_args
        # Toggle for NPC detection - set to False to disable VLM calls for NPC detection
        self.ENABLE_NPC_DETECTION = False 
        
        # Initialize visual map generator for NPC detection
        print(f"🔍 Initializing VisualMapGenerator with VLM: {self.vlm}")
        self.visual_map_generator = VisualMapGenerator(self.vlm)
        print(f"🔍 VisualMapGenerator initialized successfully")
        
        # Initialize simplified pathfinder for A* navigation
        self.pathfinder = PathfinderUtility(max_path_length=7)
        print(f"🗺️ PathfinderUtility initialized (max path length: 7)")
        
        # Use current global defaults if not specified
        max_history_entries = max_history_entries or DEFAULT_MAX_HISTORY_ENTRIES
        max_recent_actions = max_recent_actions or DEFAULT_MAX_RECENT_ACTIONS
        history_display_count = history_display_count or DEFAULT_HISTORY_DISPLAY_COUNT
        actions_display_count = actions_display_count or DEFAULT_ACTIONS_DISPLAY_COUNT
        
        self.state = SimpleAgentState()
        self.state.history = deque(maxlen=max_history_entries)
        self.state.recent_actions = deque(maxlen=max_recent_actions)
        
        # Display parameters for LLM prompts
        self.history_display_count = history_display_count
        self.actions_display_count = actions_display_count
        
        # Initialize storyline objectives for Emerald progression
        self._initialize_storyline_objectives()
        
        # Initialize direct objectives manager
        self.use_direct_objectives = use_direct_objectives
        self.direct_objective_manager = DirectObjectiveManager()
        self.current_direct_guidance = None
        
        if self.use_direct_objectives and direct_sequence:
            self._load_direct_sequence(direct_sequence)
            print(f"🎯 Direct objectives enabled: {direct_sequence}")
        else:
            print(f"🧠 Using LLM-based reasoning")
        
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
                "id": "story_littleroot_town",
                "description": "Arrive in Littleroot Town and explore the area",
                "objective_type": "location", 
                "target_value": "Littleroot Town",
                "milestone_id": "LITTLEROOT_TOWN"
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
                "id": "story_oldale_town",
                "description": "Continue journey to Oldale Town",
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
                "id": "story_route_104",
                "description": "Travel north through Route 104 toward Petalburg Woods",
                "objective_type": "location",
                "target_value": "Route 104",
                "milestone_id": "ROUTE_104"
            },
            {
                "id": "story_petalburg_woods",
                "description": "Navigate through Petalburg Woods to help Devon researcher",
                "objective_type": "location",
                "target_value": "Petalburg Woods",
                "milestone_id": "PETALBURG_WOODS"
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
                "description": "Enter the Rustboro Gym and prepare for Roxanne battle",
                "objective_type": "location",
                "target_value": "Rustboro Gym",
                "milestone_id": None  # Gym entry doesn't have separate milestone
            },
            {
                "id": "story_stone_badge",
                "description": "Defeat Roxanne and earn the Stone Badge",
                "objective_type": "battle",
                "target_value": "Stone Badge",
                "milestone_id": "STONE_BADGE"
            }
        ]
        
        # Add storyline objectives to the state
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
            
        logger.info(f"Initialized {len(storyline_objectives)} storyline objectives for Emerald progression")
        
    def _check_storyline_direct_sequence_trigger(self, game_state: Dict[str, Any]) -> Optional[str]:
        """Check if any active storyline objective should trigger a direct sequence"""
        active_objectives = self.get_active_objectives()
        
        for obj in active_objectives:
            if obj.storyline and not obj.completed:
                # Check for specific storyline objectives that should trigger direct sequences
                if obj.id == "story_route_102":
                    # Check if we're in hackathon mode (has party but not at Petalburg yet)
                    milestones = game_state.get("milestones", {})
                    has_party = milestones.get("HAS_PARTY", {}).get("completed", False)
                    at_petalburg = milestones.get("PETALBURG_CITY", {}).get("completed", False)
                    
                    if has_party and not at_petalburg:
                        logger.info("🎯 Triggering hackathon Route 102 to Petalburg direct sequence")
                        return "hackathon_route102_to_petalburg"
                        
        return None
        
    def _load_direct_sequence(self, sequence_name: str):
        """Load a specific direct objective sequence"""
        if sequence_name == "birch_to_rival":
            self.direct_objective_manager.load_birch_to_rival_sequence()
        elif sequence_name == "hackathon_route102_to_petalburg":
            self.direct_objective_manager.load_hackathon_route102_to_petalburg_sequence()
        else:
            logger.warning(f"Unknown direct sequence: {sequence_name}")
            self.use_direct_objectives = False
        
    def get_game_context(self, game_state: Dict[str, Any]) -> str:
        """Determine current game context (overworld, battle, menu, dialogue)"""
        try:
            # Check if in title sequence first
            player_location = game_state.get("player", {}).get("location", "")
            if player_location == "TITLE_SEQUENCE":
                return "title"
            
            # Check game state for title/intro
            game_state_value = game_state.get("game", {}).get("game_state", "").lower()
            if "title" in game_state_value or "intro" in game_state_value:
                return "title"
            
            # Check if player name is not set (indicates title sequence)
            player_name = game_state.get("player", {}).get("name", "").strip()
            if not player_name or player_name == "????????":
                return "title"
            
            # Check if in battle
            is_in_battle = game_state.get("game", {}).get("is_in_battle", False)
            if is_in_battle:
                logger.debug(f"Detected battle context")
                return "battle"
            
            # Check if dialogue is active
            dialogue_state = game_state.get("game", {}).get("dialogue", {})
            if dialogue_state.get("active", False) or dialogue_state.get("text", "").strip():
                return "dialogue"
            
            # Check if in menu (simplified detection)
            # Could be enhanced with more sophisticated menu detection
            player_state = game_state.get("player", {})
            if player_state.get("in_menu", False):
                return "menu"
            
            # Default to overworld
            return "overworld"
            
        except Exception as e:
            logger.warning(f"Error determining game context: {e}")
            return "unknown"
    
    def get_player_coords(self, game_state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract player coordinates from game state"""
        try:
            player = game_state.get("player", {})
            # Try position.x/y first (standard format)
            position = player.get("position", {})
            if position:
                x = position.get("x")
                y = position.get("y")
                if x is not None and y is not None:
                    return (x, y)
            
            # Fallback: try direct x/y on player
            x = player.get("x")
            y = player.get("y")
            if x is not None and y is not None:
                return (x, y)
        except Exception as e:
            logger.warning(f"Error getting player coords: {e}")
        return None
    
    def get_map_id(self, game_state: Dict[str, Any]) -> Optional[int]:
        """Extract map ID from game state"""
        try:
            return game_state.get("map", {}).get("id")
        except Exception as e:
            logger.warning(f"Error getting map ID: {e}")
        return None
    
    def add_objective(self, description: str, objective_type: str, target_value: Any = None) -> str:
        """Add a new objective and return its ID"""
        obj_id = f"obj_{len(self.state.objectives)}_{int(datetime.now().timestamp())}"
        objective = Objective(
            id=obj_id,
            description=description,
            objective_type=objective_type,
            target_value=target_value
        )
        self.state.objectives.append(objective)
        self.state.objectives_updated = True
        logger.info(f"Added objective: {description}")
        return obj_id
    
    def complete_objective(self, obj_id: str, progress_notes: str = ""):
        """Mark an objective as completed (storyline objectives cannot be manually completed)"""
        for obj in self.state.objectives:
            if obj.id == obj_id and not obj.completed:
                # Prevent manual completion of storyline objectives
                if obj.storyline:
                    logger.warning(f"Cannot manually complete storyline objective: {obj.description}. These are verified by emulator milestones.")
                    return False
                
                obj.completed = True
                obj.completed_at = datetime.now()
                obj.progress_notes = progress_notes
                self.state.objectives_updated = True
                logger.info(f"Completed objective: {obj.description}")
                return True
        return False
    
    def get_active_objectives(self) -> List[Objective]:
        """Get list of uncompleted objectives"""
        return [obj for obj in self.state.objectives if not obj.completed]
    
    def get_completed_objectives(self) -> List[Objective]:
        """Get list of completed objectives"""
        return [obj for obj in self.state.objectives if obj.completed]
    
    def check_objective_completion(self, game_state: Dict[str, Any]) -> List[str]:
        """Check if any objectives should be marked as completed based on game state"""
        completed_ids = []
        coords = self.get_player_coords(game_state)
        context = self.get_game_context(game_state)
        map_id = self.get_map_id(game_state)
        
        for obj in self.get_active_objectives():
            should_complete = False
            notes = ""
            
            if obj.objective_type == "location" and coords and obj.target_value:
                # Check if player reached target location
                # Note: target_value is a string (location name) for storyline objectives
                # Location objectives are completed via milestone verification, not coordinate checking
                # This section is for dynamically added coordinate-based objectives
                if isinstance(obj.target_value, (tuple, list)) and len(obj.target_value) == 2:
                    target_x, target_y = obj.target_value
                    if abs(coords[0] - target_x) <= 2 and abs(coords[1] - target_y) <= 2:
                        should_complete = True
                        notes = f"Reached location ({coords[0]}, {coords[1]})"
            
            elif obj.objective_type == "battle" and context == "battle":
                # Objective completed when battle starts
                should_complete = True
                notes = "Entered battle"
            
            elif obj.objective_type == "dialogue" and context == "dialogue":
                # Objective completed when dialogue starts
                should_complete = True
                notes = "Started dialogue"
            
            elif obj.objective_type == "map" and map_id and obj.target_value:
                # Check if player reached target map
                if map_id == obj.target_value:
                    should_complete = True
                    notes = f"Reached map {map_id}"
            
            if should_complete:
                self.complete_objective(obj.id, notes)
                completed_ids.append(obj.id)
        
        return completed_ids
    
    def check_storyline_milestones(self, game_state: Dict[str, Any]) -> List[str]:
        """Check emulator milestones and auto-complete corresponding storyline objectives"""
        completed_ids = []
        
        # Get milestones from the game state (if available)
        milestones = game_state.get("milestones", {})
        if not milestones:
            # No milestone data available, skip checking
            return completed_ids
            
        for obj in self.get_active_objectives():
            # Only check storyline objectives with milestone IDs
            if obj.storyline and obj.milestone_id and not obj.completed:
                # Check if the corresponding emulator milestone is completed
                milestone_completed = milestones.get(obj.milestone_id, {}).get("completed", False)
                
                if milestone_completed:
                    # Auto-complete the storyline objective
                    obj.completed = True
                    obj.completed_at = datetime.now()
                    obj.progress_notes = f"Auto-completed by emulator milestone: {obj.milestone_id}"
                    self.state.objectives_updated = True
                    completed_ids.append(obj.id)
                    logger.info(f"Auto-completed storyline objective via milestone {obj.milestone_id}: {obj.description}")
        
        return completed_ids
    
    def detect_stuck_pattern(self, coords: Optional[Tuple[int, int]], context: str, game_state: Dict[str, Any] = None) -> bool:
        """Detect if the agent appears to be stuck in a location/context"""
        # Don't trigger stuck detection during contexts where staying in place is expected
        if context in ["battle", "dialogue", "menu", "title"]:
            logger.debug(f"Skipping stuck detection - context: {context}")
            return False
        
        # Need valid coordinates for stuck detection
        if not coords or coords[0] is None or coords[1] is None:
            return False
        
        # Check for title sequence if game state is available
        if game_state:
            # Check if in title sequence (no player name or invalid coordinates)
            player_name = game_state.get("player", {}).get("name", "").strip()
            if not player_name or player_name == "????????":
                return False
                
            # Check if game state indicates title/intro
            game_state_value = game_state.get("game", {}).get("game_state", "").lower()
            if "title" in game_state_value or "intro" in game_state_value:
                return False
            
            # Check location for title sequence
            player_location = game_state.get("player", {}).get("location", "")
            if player_location == "TITLE_SEQUENCE":
                return False
            
        key = f"{coords[0]}_{coords[1]}_{context}"
        self.state.stuck_detection[key] = self.state.stuck_detection.get(key, 0) + 1
        
        # Consider stuck if we've been in the same location/context for 8+ consecutive steps
        return self.state.stuck_detection[key] >= 8
    
    def is_black_frame(self, frame) -> bool:
        """
        Check if the frame is mostly black (transition/loading screen).
        
        Args:
            frame: PIL Image or numpy array
            
        Returns:
            bool: True if frame is mostly black, False otherwise
        """
        try:
            
            # Convert to PIL Image if needed
            if hasattr(frame, 'convert'):  # It's already a PIL Image
                img = frame
            elif hasattr(frame, 'shape'):  # It's a numpy array
                img = Image.fromarray(frame)
            else:
                return False  # Unknown type, assume not black
            
            # Convert to numpy array for analysis
            img_array = np.array(img)
            
            # Calculate the mean brightness
            # For RGB images, average across all channels
            if len(img_array.shape) == 3:
                mean_brightness = np.mean(img_array)
            else:
                mean_brightness = np.mean(img_array)
            
            # Also check the standard deviation to catch completely uniform frames
            std_dev = np.std(img_array)
            
            # A frame is considered "black" if:
            # 1. Mean brightness is very low (< 10 out of 255)
            # 2. OR standard deviation is very low (< 5) indicating uniform color
            is_black = mean_brightness < 10 or (mean_brightness < 30 and std_dev < 5)
            
            if is_black:
                logger.debug(f"Black frame detected: mean_brightness={mean_brightness:.2f}, std_dev={std_dev:.2f}")
            
            return is_black
            
        except Exception as e:
            logger.warning(f"Error checking for black frame: {e}")
            return False  # On error, assume not black to continue processing
    
    def get_relevant_history_summary(self, current_context: str, coords: Optional[Tuple[int, int]]) -> str:
        """Get a concise summary of relevant recent history"""
        # current_context and coords could be used for more sophisticated filtering in the future
        _ = current_context, coords  # Acknowledge unused parameters for now
        if not self.state.history:
            return "No previous history."
        
        # Get last N entries based on display count
        recent_entries = list(self.state.history)[-self.history_display_count:]
        
        # Format for LLM consumption
        summary_lines = []
        for i, entry in enumerate(recent_entries, 1):
            coord_str = f"({entry.player_coords[0]},{entry.player_coords[1]})" if entry.player_coords else "(?)"
            summary_lines.append(f"{i}. {entry.context} at {coord_str}: {entry.action_taken}")
        
        return "\n".join(summary_lines)
    
    def get_stuck_warning(self, coords: Optional[Tuple[int, int]], context: str, game_state: Dict[str, Any] = None) -> str:
        """Generate warning text if stuck pattern detected"""
        # Never show stuck warning in title sequence
        if context == "title":
            return ""
            
        if self.detect_stuck_pattern(coords, context, game_state):
            return "\n⚠️ WARNING: You appear to be stuck at this location/context. Try a different approach!\n" \
                   "💡 TIP: If you try an action like RIGHT but coordinates don't change from (X,Y) to (X+1,Y), there's likely an obstacle. Check the map around player P for walls (#) or other barriers blocking your path."
        return ""
    
    def create_game_state_summary(self, game_state: Dict[str, Any]) -> str:
        """Create a concise summary of the current game state"""
        try:
            game_info = game_state.get("game", {})
            
            summary_parts = []
            # Player location
            coords = self.get_player_coords(game_state)
            if coords:
                summary_parts.append(f"Player at ({coords[0]}, {coords[1]})")
            
            # Map info
            map_id = self.get_map_id(game_state)
            if map_id:
                summary_parts.append(f"Map {map_id}")
            
            # Context-specific info
            context = self.get_game_context(game_state)
            if context == "battle":
                summary_parts.append("In battle")
            elif context == "dialogue":
                dialogue_text = game_info.get("dialogue", {}).get("text", "")
                if dialogue_text:
                    summary_parts.append(f"Dialogue: {dialogue_text}")
            
            return " | ".join(summary_parts) if summary_parts else "Unknown state"
            
        except Exception as e:
            logger.warning(f"Error creating game state summary: {e}")
            return "Error reading state"
    
    def step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compatibility method for client that expects agent.step(game_state)
        
        Args:
            game_state: Complete game state dictionary (should include 'frame')
            
        Returns:
            Dictionary with 'action' and optional 'reasoning'
        """
        frame = game_state.get('frame')
        if frame is None:
            logger.error("🚫 No frame in game_state for SimpleAgent.step")
            return {"action": "WAIT", "reasoning": "No frame available"}
        
        action = self.process_step(frame, game_state)
        return {"action": action, "reasoning": "Simple agent decision"}
    
    def process_step(self, frame, game_state: Dict[str, Any]) -> str:
        """
        Main processing step for simple mode with history tracking
        
        Args:
            frame: Current game frame (PIL Image or similar)
            game_state: Complete game state dictionary
            
        Returns:
            Action string or list of actions
        """
        # CRITICAL: Validate frame before any VLM processing
        if frame is None:
            logger.error("🚫 CRITICAL: SimpleAgent.process_step called with None frame - cannot proceed")
            return "WAIT"
        
        # Validate frame is a proper image
        if not (hasattr(frame, 'save') or hasattr(frame, 'shape')):
            logger.error(f"🚫 CRITICAL: SimpleAgent.process_step called with invalid frame type {type(frame)} - cannot proceed")
            return "WAIT"
        
        # Additional PIL Image validation
        if hasattr(frame, 'size'):
            width, height = frame.size
            if width <= 0 or height <= 0:
                logger.error(f"🚫 CRITICAL: SimpleAgent.process_step called with invalid frame size {width}x{height} - cannot proceed")
                return "WAIT"
        
        # Check for black frame (transition screen)
        if self.is_black_frame(frame):
            logger.info("⏳ Black frame detected (likely a transition), waiting for next frame...")
            return "WAIT"  # Return WAIT to skip this frame and wait for the next one
        
        try:
            # Check if we should trigger a direct sequence based on storyline objectives
            if not self.direct_objective_manager.is_sequence_active():
                sequence_name = self._check_storyline_direct_sequence_trigger(game_state)
                if sequence_name:
                    self._load_direct_sequence(sequence_name)
                    self.use_direct_objectives = True
                    print(f"🎯 Auto-triggered direct sequence: {sequence_name}")
            
            # Check if we should use direct objectives
            if self.use_direct_objectives and self.direct_objective_manager.is_sequence_active():
                # Get guidance instead of specific actions
                guidance = self.direct_objective_manager.get_current_objective_guidance(game_state)
                if guidance:
                    # Log direct objective status
                    status = self.direct_objective_manager.get_sequence_status()
                    print(f"🎯 Direct objective: {status['current_objective']} ({status['completed_count']}/{status['total_objectives']})")
                    print(f"🎯 Navigation hint: {guidance.get('navigation_hint', 'No hint available')}")
                    
                    # Store guidance for LLM to use
                    self.current_direct_guidance = guidance
                else:
                    print("🎯 Direct objectives sequence completed, switching to LLM reasoning")
                    self.use_direct_objectives = False
                    self.current_direct_guidance = None
            
            # Increment step counter
            self.state.step_counter += 1
            
            # Get current state info
            coords = self.get_player_coords(game_state)
            context = self.get_game_context(game_state)
            map_id = self.get_map_id(game_state)
            
            # Enhance game state with visual NPC detection (if enabled and not in battle/menu)
            # Skip NPC detection in battles and menus to reduce inference costs
            if self.ENABLE_NPC_DETECTION and context not in ["battle", "menu", "dialogue", "title"]:
                try:
                    print(f"🔍 Attempting NPC detection - map in game_state: {'map' in game_state}, coords: {coords}")
                    if 'map' in game_state and coords:
                        # Add player coordinates to map info for NPC detection
                        map_info_with_coords = game_state['map'].copy()
                        map_info_with_coords['player_coords'] = coords
                        
                        print(f"🔍 Calling enhance_map_with_visual_npcs with coords: {coords}")
                        enhanced_map_info = self.visual_map_generator.enhance_map_with_visual_npcs(
                            map_info_with_coords, frame
                        )
                        game_state['map'] = enhanced_map_info
                        
                        # Log the detected NPCs for debugging
                        detected_npcs = enhanced_map_info.get('object_events', [])
                        print(f"🔍 Enhanced map with {len(detected_npcs)} total NPCs")
                        for i, npc in enumerate(detected_npcs[:3]):  # Show first 3 NPCs
                            npc_x = npc.get('current_x', npc.get('x', '?'))
                            npc_y = npc.get('current_y', npc.get('y', '?'))
                            npc_name = npc.get('name', 'Unknown')
                            npc_type = npc.get('npc_type', 'npc')
                            print(f"🔍   NPC {i+1}: {npc_name} ({npc_type}) at ({npc_x}, {npc_y})")
                        
                        logger.info(f"Enhanced map with visual NPC detection - {len(detected_npcs)} NPCs total")
                    else:
                        print(f"🔍 Skipping NPC detection - map: {'map' in game_state}, coords: {coords}")
                except Exception as e:
                    print(f"🔍 Error in NPC detection: {e}")
                    logger.warning(f"Failed to enhance map with visual NPCs: {e}")
            else:
                if context in ["battle", "menu", "dialogue", "title"]:
                    print(f"🔍 NPC detection skipped - not needed in {context} context (saves inference cost)")
                else:
                    print("🔍 NPC detection disabled - skipping VLM calls")
            
            # Format the current state for LLM (movement preview disabled - using pathfinding utility instead)
            formatted_state = format_state_for_llm(game_state, include_movement_preview=True)
            
            # Get movement memory for the current area
            movement_memory = ""
            if coords:
                movement_memory = self.get_area_movement_memory(coords)
        
            
            # Check for objective completion first
            self.check_objective_completion(game_state)
            
            # Check storyline milestones and auto-complete objectives
            self.check_storyline_milestones(game_state)
            
            # Get relevant history and stuck detection
            history_summary = self.get_relevant_history_summary(context, coords)
            stuck_warning = self.get_stuck_warning(coords, context, game_state)
            recent_actions_str = ', '.join(list(self.state.recent_actions)[-self.actions_display_count:]) if self.state.recent_actions else 'None'
            
            # Format objectives for LLM
            active_objectives = self.get_active_objectives()
            completed_objectives_list = self.get_completed_objectives()
            objectives_summary = self._format_objectives_for_llm(active_objectives, completed_objectives_list)
            
            # Add direct guidance if available
            if self.current_direct_guidance:
                direct_guidance_text = f"""
🎯 **DIRECT OBJECTIVE GUIDANCE**:
Current Direct Objective: {self.current_direct_guidance['description']}
Action Type: {self.current_direct_guidance['action_type']}
Target Location: {self.current_direct_guidance['target_location']}
Navigation Hint: {self.current_direct_guidance['navigation_hint']}

**IMPORTANT**: Follow this direct objective guidance. Use the navigation hint to determine your pathfinding target and approach.

**OBJECTIVE COMPLETION**: When you have successfully completed this objective (e.g., passed the trainer, navigated through the area), use this command:
COMPLETE_OBJECTIVE: {self.current_direct_guidance.get('id', 'current_objective')}:[brief description of what you accomplished]

Example: COMPLETE_OBJECTIVE: hackathon_01_west_blue_hat_trainer:Successfully moved west past the blue hat trainer

"""
            objectives_summary = objectives_summary + direct_guidance_text
            
            # Build pathfinding rules section (only if not in title sequence)
            pathfinding_rules = ""
            if context != "title":
                pathfinding_rules = """
🗺️ **NAVIGATION: USE A* PATHFINDING UTILITY**:
   For ALL movement, use the pathfinding utility to navigate efficiently:
   
   **HOW TO NAVIGATE**:
   In your ACTION section, request: REQUEST_PATH: (grid_x, grid_y)
   - You are ALWAYS at the center (7,7) of a 15x15 grid (coordinates 0-14)
   - Look at the map to find where you want to go
   - The pathfinder will ATTEMPT to find the shortest path using A* avoiding walls/NPCs (max 7 moves at a time)
   - ⚠️ **IMPORTANT**: Pathfinding can FAIL if the target is unreachable or blocked - always pick realistic, walkable targets in the current visual frame
   
   ⚡ **15x15 GRID COORDINATE SYSTEM**:
   - You are ALWAYS at position (7,7) - the center of a 15x15 grid
   - Grid coordinates range from (0,0) to (14,14)
   - To go NORTH: Pick y < 7 (e.g., 7,4 or 7,3 or 7,2)
   - To go SOUTH: Pick y > 7 (e.g., 7,10 or 7,11 or 7,12)
   - To go WEST: Pick x < 7 (e.g., 4,7 or 3,7 or 2,7)
   - To go EAST: Pick x > 7 (e.g., 10,7 or 11,7 or 12,7)
   - Look at the map, find a walkable tile (.) in your target direction, note its grid position, REQUEST_PATH to it!
   - NOTE: Trust the pathfinder to navigate around obstacles, you're only role is to identify the final location you want to arrive at in accordance with your goal.
   - ⚠️ **CRITICAL**: Feel free to request paths up to 7 actions away, beyond this distance is unreasonable
   - SPECIAL CASE - LEDGE NAVIGATION: When you see ledges (↓ symbols) creating a passage, look for walkable tiles (.) above the gap and pathfind to those coordinates
   
   ⚡ **EXAMPLES**:
   - "I need to go north-east, around this house. I see walkable tile at (9,4). REQUEST_PATH: (9, 4)"
   - "I need to go west. I see walkable tile at (4,7). REQUEST_PATH: (4, 7)"
   - "I need to walk through that door at (9,5). REQUEST_PATH: (9, 5)"
   - "I need to move one step left and 3 steps up. REQUEST_PATH: (6, 4)"
   
   **LEDGE NAVIGATION EXAMPLE:**
   You see a map like this (15x15 grid):
   ```
   . . . . . . . . . . . . . . ?                                
   ? . . . . . # # # # ~ ~ ~ ~ ~ ~ ~ ~ . . . . . . . . . . . . . . . . ?                                
   ? . . . . . . . . . . ~ ~ ~ ~ ~ ~ ~ . . . . # # ↓ ↓ . . ↓ ↓ ↓ ↓ ↓ ↓ ?                                
   ? . . . . . . . . . . . ~ ~ ~ ~ . . . . . . # # . . . . ~ ~ ~ ~ ~ . ?                                
     ? ? ? . . . . . . . . . . . . . . . . # # # # . . ~ ~ ~ ~ ~ ~ ~ . ?             
         ? . . . . . . . . . . . . . . . . # # # # P ~ ~ ~ ~ ~ ~ ~ . . # 
   ```
   The player P (at center 7,7) needs to make it through the passage between the two ledges (↓ symbols). 
   To do so, the agent needs to pick any walkable path above the gap. 
   I can see walkable tiles (.) at coordinates (7,3) and (8,3) above the passage.
   REQUEST_PATH: (7, 3)

💡 **NPC & OBSTACLE HANDLING**:
    - NPCs are detected visually in the game frame - check the image for trainers and obstacles
    - The pathfinder attempts to avoid walls, obstacles, and NPCs (but may fail if target is unreachable)
    - If a movement fails (coordinates don't change), use MOVEMENT MEMORY to remember and try different route (REQUEST_PATH to a different set of coordinates in line with your objective)
    - Trainers typically block movement and must be battled - the pathfinder will route around or you can approach them
    - IMPORTANT: You need to be directly adjacent to AND facing an NPC to interact with them before using the A button

🛑 **INPUT HANDLING RULES**:
    - If DIALOGUE TEXT is visible or dialogue is active → press A up to 1 time to advance, then reassess
    - If NO dialogue text is visible AND your coordinates are NOT changing in OVERWORLD/menu → DO NOT press A again
    - Never output more than 1 consecutive 'A' at the same coordinates in OVERWORLD, especially if your action history shows you've consistently pressed A at the same coordinates without changing coordinates
    - In battle, think carefully about move selection - don't spam A
"""

            # Create enhanced prompt with objectives, history context and chain of thought request
            prompt = f"""SYSTEM PROMPT: 
            <<< You are speedrunning Pokemon Emerald and must progress quickly to the milestones.  Based on the current game frame and state information, think through your next move and choose the best action. 
                Some pointers to keep in mind (guard rails) as you problem solve:
                1) You must think step-by-step when solving problems and making decisions. 
                2) Always provide detailed, context-aware responses that bias for ground-truth.
                3) Consider the current situation in the game as well as what you've learned over time.
                4) Do not fixate on the correctness of a particular solution, be flexible and adapt your strategy as needed (i.e moving in a different direction if you see to be stuck).
            >>>

RECENT ACTION HISTORY (last {self.actions_display_count} actions):
{recent_actions_str}

LOCATION/CONTEXT HISTORY (last {self.history_display_count} steps):
{history_summary}

CURRENT OBJECTIVES:
{objectives_summary}

CURRENT GAME STATE:

Detailed Map Display:
{formatted_state}

{movement_memory}

{stuck_warning}

Available actions: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT

IMPORTANT NOTE: Please think via the (analysis, objectives, plan, reasoning) loop before choosing your action. Structure your response like this:

ANALYSIS:
[Analyze what you see in the frame and current game state - what's happening? where are you? what should you be doing? 
IMPORTANT: Look carefully at the game image for NPCs (people, trainers) that might not be shown on the map. NPCs appear as sprite characters and can block movement or trigger battles/dialogue.
The pathfinder will attempt to find the best route to your target, avoiding obstacles and NPCs (but may fail if target is unreachable).]

OBJECTIVES:
[Review your current objectives. You have main storyline objectives (story_*) that track overall Emerald progression - these are automatically verified and you CANNOT manually complete them. You can create your own sub-objectives to help achieve the main goals. Do any need to be updated, added, or marked as complete?
- Add sub-objectives: ADD_OBJECTIVE: type:description:target_value (e.g., "ADD_OBJECTIVE: location:Find Pokemon Center in town:(15,20)" or "ADD_OBJECTIVE: item:Buy Pokeballs:5")
- Complete sub-objectives only: COMPLETE_OBJECTIVE: objective_id:notes (e.g., "COMPLETE_OBJECTIVE: my_sub_obj_123:Successfully bought Pokeballs")
- NOTE: Do NOT try to complete storyline objectives (story_*) - they auto-complete when milestones are reached]

PLAN:
[Think about your immediate goal - what do you want to accomplish in the next few actions? Consider your current objectives and recent history. 

🗺️ **FOR NAVIGATION**:
Look at the map and identify where you want to go. You'll request pathfinding in the ACTION section.

You are at grid center (7,7). Pick a walkable tile in your target direction:
- NORTH: y < 7 (e.g., 7,4)
- SOUTH: y > 7 (e.g., 7,11)
- EAST: x > 7 (e.g., 10,7)
- WEST: x < 7 (e.g., 4,7)

Example plan: "Need to go south to reach Petalburg. I see tile (7,11) is walkable. Will request path to it."]

REASONING:
[Explain why you're choosing this specific action. Reference the MOVEMENT MEMORY sections. Check the visual frame for NPCs before moving. If you see NPCs in the image, avoid walking into them. Consider any failed movements or known obstacles from your memory.
IMPORTANT: Check the visual frame for NPCs and obstacles that might block your path.

⚔️ **BATTLE STRATEGY**:
- When your HP is below 75%, prioritize healing moves that also deal damage (e.g., Absorb, Mega Drain, Giga Drain)
- These moves let you heal while still progressing the battle - very efficient for speedrunning
- Against trainers, you cannot run - you must fight strategically
- Example: "My HP is at 63%. I'll use Absorb to heal while damaging the opponent."]

ACTION:
[Your final action choice.
- For NAVIGATION in overworld: Use REQUEST_PATH: (grid_x, grid_y) - the pathfinder will attempt to calculate optimal route using A*
  Example: REQUEST_PATH: (7, 11)
- For DIALOGUE: Press A to advance (up to 2 times)
- For BATTLES: Select appropriate move (e.g., Absorb, Tackle, etc.)
- For MENUS: Select appropriate option
- Never output more than 2 consecutive 'A' at the same coordinates in OVERWORLD]

{pathfinding_rules}

Context: {context} | Coords: {coords} """
            
            # Print complete prompt to terminal for debugging
            print("\n" + "="*120)
            print("🤖 SIMPLE AGENT PROMPT SENT TO VLM:")
            print("="*120)
            
            # Print prompt in chunks to avoid terminal truncation
            sys.stdout.write(prompt)
            sys.stdout.write("\n")
            sys.stdout.flush()
            
            print("="*120)
            print("🤖 END OF SIMPLE AGENT PROMPT")
            print("="*120 + "\n")
            sys.stdout.flush()
            
            # Make VLM call - double-check frame validation before VLM
            if frame and (hasattr(frame, 'save') or hasattr(frame, 'shape')):
                print("🔍 Making VLM call...")
                try:
                    response = self.vlm.get_query(frame, prompt, "simple_mode")
                    print(f"🔍 VLM response received: {response[:100]}..." if len(response) > 100 else f"🔍 VLM response: {response}")
                except Exception as e:
                    print(f"❌ VLM call failed: {e}")
                    return "WAIT"
            else:
                logger.error("🚫 CRITICAL: About to call VLM but frame validation failed - this should never happen!")
                return "WAIT"
            
            # Extract action(s) from structured response
            actions, reasoning = self._parse_structured_response(response, game_state)
            
            # Check for failed movement by comparing previous coordinates
            if len(self.state.history) > 0:
                prev_coords = self.state.history[-1].player_coords
                if prev_coords and coords:
                    # If coordinates didn't change and we attempted a movement, record it as failed
                    if (prev_coords == coords and 
                        isinstance(actions, list) and len(actions) > 0 and 
                        actions[0] in ['UP', 'DOWN', 'LEFT', 'RIGHT']):
                        self.record_failed_movement(coords, actions[0], "movement_blocked")
                    elif (prev_coords == coords and 
                          isinstance(actions, str) and 
                          actions in ['UP', 'DOWN', 'LEFT', 'RIGHT']):
                        self.record_failed_movement(coords, actions, "movement_blocked")

            # Record this step in history with reasoning
            game_state_summary = self.create_game_state_summary(game_state)
            action_with_reasoning = f"{actions} | Reasoning: {reasoning}" if reasoning else str(actions)
            history_entry = HistoryEntry(
                timestamp=datetime.now(),
                player_coords=coords,
                map_id=map_id,
                context=context,
                action_taken=action_with_reasoning,
                game_state_summary=game_state_summary
            )
            self.state.history.append(history_entry)
            
            # Update recent actions
            if isinstance(actions, list):
                self.state.recent_actions.extend(actions)
            else:
                self.state.recent_actions.append(actions)
            
            # Reset stuck detection for other locations when we move
            if coords:
                keys_to_reset = [k for k in self.state.stuck_detection.keys() 
                               if not k.startswith(f"{coords[0]}_{coords[1]}")]
                for key in keys_to_reset:
                    if self.state.stuck_detection[key] > 0:
                        self.state.stuck_detection[key] = max(0, self.state.stuck_detection[key] - 1)
            
            # Update server with agent step and metrics (for agent thinking display)
            self._update_server_metrics()
            
            return actions
            
        except Exception as e:
            logger.error(f"Error in simple agent processing: {e}")
            return ["A"]  # Default safe action as list
    
    def _update_server_metrics(self):
        """Update server with current agent step count and LLM metrics"""
        try:
            import requests
            from utils.llm_logger import get_llm_logger
            
            # Get current LLM metrics
            llm_logger = get_llm_logger()
            metrics = llm_logger.get_cumulative_metrics()
            
            # Send metrics to server
            try:
                response = requests.post(
                    "http://localhost:8000/agent_step",
                    json={"metrics": metrics},
                    timeout=1
                )
                if response.status_code != 200:
                    logger.warning(f"Failed to update server metrics: {response.status_code}")
            except requests.exceptions.RequestException:
                # Silent fail - server might not be running or in different mode
                pass
                
        except Exception as e:
            logger.warning(f"Error updating server metrics: {e}")
    
    def _parse_actions(self, response: str, game_state: Dict[str, Any] = None) -> List[str]:
        """Parse action response from LLM into list of valid actions"""
        response_upper = response.upper().strip()
        valid_actions = ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
        
        # Skip if this is a pathfinding request (should be handled separately)
        if 'REQUEST_PATH' in response_upper:
            logger.debug("Skipping action parsing - REQUEST_PATH detected")
            return []
        
        # Parse multiple actions (could be comma or space separated)
        actions_found = []
        # Replace commas with spaces for consistent parsing
        response_clean = response_upper.replace(',', ' ').replace('.', ' ')
        tokens = response_clean.split()
        
        for token in tokens:
            if token in valid_actions:
                actions_found.append(token)
                if len(actions_found) >= 7:  # Max 10 actions
                    break
        
        # Validate movement sequences if we have game state
        if game_state and len(actions_found) > 1:
            # Check if this is a movement sequence
            movement_actions = [a for a in actions_found if a in ['UP', 'DOWN', 'LEFT', 'RIGHT']]
            if movement_actions:
                # Validate the movement sequence
                is_valid, reason = self.validate_movement_sequence(movement_actions, game_state)
                if not is_valid:
                    logger.warning(f"Movement sequence validation failed: {reason}")
                    # Only take the first movement if sequence is invalid
                    if movement_actions:
                        actions_found = [movement_actions[0]]
                        logger.info(f"Reduced to single movement: {actions_found[0]}")
        
        # If no valid actions found, use default
        if not actions_found:
            actions_found = ['A']
        
        return actions_found
    
    def _format_objectives_for_llm(self, active_objectives: List[Objective], completed_objectives: List[Objective]) -> str:
        """Format objectives for LLM consumption"""
        lines = []
        
        if active_objectives:
            lines.append("🎯 ACTIVE OBJECTIVES:")
            for i, obj in enumerate(active_objectives[:5], 1):  # Show top 5 active
                target_str = f" (Target: {obj.target_value})" if obj.target_value else ""
                lines.append(f"  {i}. [{obj.objective_type}] {obj.description}{target_str} [ID: {obj.id}]")
        else:
            lines.append("🎯 ACTIVE OBJECTIVES: None - Consider setting some goals!")
        
        if completed_objectives:
            recent_completed = completed_objectives[-3:]  # Show last 3 completed
            lines.append("✅ RECENTLY COMPLETED:")
            for obj in recent_completed:
                lines.append(f"  ✓ [{obj.objective_type}] {obj.description}")
        
        return "\n".join(lines)
    
    def _parse_structured_response(self, response: str, game_state: Dict[str, Any] = None) -> Tuple[List[str], str]:
        """Parse structured chain-of-thought response and extract actions and reasoning"""
        try:
            # Extract sections from structured response
            analysis = ""
            objectives_section = ""
            plan = ""
            reasoning = ""
            actions = []
            pathfinding_requested = False
            pathfinding_result = ""
            pathfinding_request = None  # Initialize pathfinding_request
            
            # Split response into lines for processing
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # Identify section headers
                if line.upper().startswith('ANALYSIS:'):
                    current_section = 'analysis'
                    analysis = line[9:].strip()  # Remove "ANALYSIS:" prefix
                elif line.upper().startswith('OBJECTIVES:'):
                    current_section = 'objectives'
                    objectives_section = line[11:].strip()  # Remove "OBJECTIVES:" prefix
                elif line.upper().startswith('PLAN:'):
                    current_section = 'plan'
                    plan = line[5:].strip()  # Remove "PLAN:" prefix
                elif line.upper().startswith('REASONING:'):
                    current_section = 'reasoning'
                    reasoning = line[10:].strip()  # Remove "REASONING:" prefix
                elif line.upper().startswith('ACTION:'):
                    current_section = 'action'
                    # Extract actions from this line
                    action_text = line[7:].strip()  # Remove "ACTION:" prefix
                    
                    # Check for REQUEST_PATH first before parsing as normal action
                    pathfinding_request = self._extract_pathfinding_request(action_text)
                    if not pathfinding_request and action_text:
                        # Only parse if there's content and it's not a pathfinding request
                        actions = self._parse_actions(action_text, game_state)
                elif line and current_section:
                    # Continue content of current section
                    if current_section == 'analysis':
                        analysis += " " + line
                    elif current_section == 'objectives':
                        objectives_section += " " + line
                    elif current_section == 'plan':
                        plan += " " + line
                    elif current_section == 'reasoning':
                        reasoning += " " + line
                    elif current_section == 'action':
                        # Additional action parsing from action section content
                        if line.strip():  # Only process non-empty lines
                            # Check for pathfinding request first
                            pathfinding_check = self._extract_pathfinding_request(line)
                            if pathfinding_check:
                                # Found pathfinding request in continuation line
                                pathfinding_request = pathfinding_check
                            elif not pathfinding_request:
                                # Only parse as action if no pathfinding request found yet
                                additional_actions = self._parse_actions(line, game_state)
                                actions.extend(additional_actions)
                                if len(actions) >= 7:  # Max 10 actions
                                    actions = actions[:7]
                                    break
            
            # pathfinding_request was extracted in the ACTION parsing above
            # Now check if we found one and use it
            if pathfinding_request and game_state:
                target_x, target_y = pathfinding_request
                logger.info(f"🗺️ Pathfinding requested to grid ({target_x}, {target_y})")
                
                # Call pathfinder
                path_actions = self.pathfinder.find_path(game_state, (target_x, target_y))
                
                if path_actions:
                    logger.info(f"🗺️ Pathfinder found path: {path_actions}")
                    pathfinding_requested = True
                    pathfinding_result = f"Path found: {', '.join(path_actions)}"
                    
                    # ALWAYS use pathfinder actions when pathfinding was requested
                    # This overrides any manual actions the agent might have specified
                    actions = path_actions
                    logger.info(f"🗺️ Using pathfinder actions (overriding manual actions): {actions}")
                else:
                    logger.info(f"🗺️ No path found to ({target_x}, {target_y})")
                    pathfinding_result = f"No path to ({target_x}, {target_y}) - obstacle blocking or target unreachable"
                    # ALWAYS try to move towards target when pathfinding fails - this helps agent learn
                    # Calculate direction to target and try a single step
                    player_x, player_y = 7, 7  # Player is always at center of 15x15 grid
                    dx = target_x - player_x
                    dy = target_y - player_y
                    
                    # Try to move one step towards target (override any existing actions)
                    if abs(dx) > abs(dy):
                        # Move horizontally first
                        if dx > 0:
                            actions = ['RIGHT']
                        else:
                            actions = ['LEFT']
                    else:
                        # Move vertically first
                        if dy > 0:
                            actions = ['DOWN']
                        else:
                            actions = ['UP']
                    
                    logger.info(f"🗺️ Pathfinding failed, attempting single step towards target: {actions}")
                    logger.info(f"🗺️ This will help agent learn if the target is reachable or if a different approach is needed")
            
            # Process objectives if mentioned
            if objectives_section:
                self._process_objectives_from_response(objectives_section)
            
            # Also process COMPLETE_OBJECTIVE commands from the entire response (fallback)
            # This ensures objectives are completed even if they appear in other sections
            self._process_objectives_from_response(response)
            
            # If no actions found in structured format, fall back to parsing entire response
            if not actions:
                actions = self._parse_actions(response, game_state)
            
            # Create concise reasoning summary
            reasoning_parts = []
            if analysis:
                reasoning_parts.append(f"Analysis: {analysis}")
            if objectives_section:
                reasoning_parts.append(f"Objectives: {objectives_section}")
            if plan:
                reasoning_parts.append(f"Plan: {plan}")
            if reasoning:
                reasoning_parts.append(f"Reasoning: {reasoning}")
            if pathfinding_result:
                reasoning_parts.append(f"Pathfinding: {pathfinding_result}")
            
            full_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No reasoning provided"
            
            return actions, full_reasoning
            
        except Exception as e:
            logger.warning(f"Error parsing structured response: {e}")
            # Fall back to basic action parsing
            return self._parse_actions(response, game_state), "Error parsing reasoning"
    
    def _extract_pathfinding_request(self, text: str) -> Optional[Tuple[int, int]]:
        """
        Extract pathfinding request from text.
        Looks for patterns like "REQUEST_PATH: (x, y)" or "REQUEST_PATH: x,y"
        
        Args:
            text: Text to search for pathfinding request
            
        Returns:
            Tuple of (x, y) coordinates if found, None otherwise
        """
        import re
        
        try:
            # Look for REQUEST_PATH: (x, y) or REQUEST_PATH: x,y
            # Case insensitive matching
            pattern = r'REQUEST_PATH\s*:\s*\(?(\d+)\s*,\s*(\d+)\)?'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                return (x, y)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting pathfinding request: {e}")
            return None
    
    def _process_objectives_from_response(self, objectives_text: str):
        """Process objective management commands from LLM response"""
        try:
            # Look for ADD_OBJECTIVE and COMPLETE_OBJECTIVE commands
            for line in objectives_text.split('\n'):
                line = line.strip()
                if line.upper().startswith('ADD_OBJECTIVE:'):
                    # Parse format: ADD_OBJECTIVE: type:description:target_value
                    content = line[14:].strip()  # Remove "ADD_OBJECTIVE:" prefix
                    parts = content.split(':', 2)  # Split into max 3 parts
                    
                    if len(parts) >= 2:
                        obj_type = parts[0].strip()
                        description = parts[1].strip()
                        target_value = parts[2].strip() if len(parts) > 2 else None
                        
                        # Parse target_value based on type
                        parsed_target = self._parse_target_value(obj_type, target_value)
                        
                        # Add the objective
                        self.add_objective(description, obj_type, parsed_target)
                
                elif line.upper().startswith('COMPLETE_OBJECTIVE:'):
                    # Parse format: COMPLETE_OBJECTIVE: objective_id:notes
                    content = line[19:].strip()  # Remove "COMPLETE_OBJECTIVE:" prefix
                    parts = content.split(':', 1)  # Split into max 2 parts
                    
                    if len(parts) >= 1:
                        obj_id = parts[0].strip()
                        notes = parts[1].strip() if len(parts) > 1 else "Manually completed by LLM"
                        
                        # Check if this is a direct objective completion
                        if self.use_direct_objectives and self.direct_objective_manager.is_sequence_active():
                            current_obj = self.direct_objective_manager.get_current_objective()
                            if current_obj and current_obj.id == obj_id:
                                # Complete the direct objective
                                self.direct_objective_manager._mark_objective_completed(current_obj)
                                self.direct_objective_manager.current_index += 1
                                logger.info(f"LLM completed direct objective: {obj_id} - {notes}")
                                print(f"🎯 Direct objective completed: {current_obj.description}")
                                
                                # Regenerate guidance for next objective
                                if self.direct_objective_manager.is_sequence_active():
                                    # Use the game_state parameter passed to this method
                                    self.current_direct_guidance = self.direct_objective_manager.get_current_objective_guidance(game_state)
                                    if self.current_direct_guidance:
                                        print(f"🎯 Next objective: {self.current_direct_guidance['description']}")
                                    else:
                                        print("🎯 All direct objectives completed!")
                                        self.current_direct_guidance = None
                                
                                continue
                        
                        # Complete regular objective
                        success = self.complete_objective(obj_id, notes)
                        if success:
                            logger.info(f"LLM manually completed objective: {obj_id}")
                        else:
                            logger.warning(f"LLM tried to complete non-existent or already completed objective: {obj_id}")
                        
        except Exception as e:
            logger.warning(f"Error processing objectives from response: {e}")
    
    def _parse_target_value(self, obj_type: str, target_str: Optional[str]) -> Any:
        """Parse target value based on objective type"""
        if not target_str:
            return None
            
        try:
            if obj_type == "location":
                # Try to parse coordinates like "(15,20)" or "15,20"
                target_str = target_str.strip('()')
                if ',' in target_str:
                    x, y = map(int, target_str.split(','))
                    return (x, y)
            elif obj_type == "map":
                # Try to parse map ID as integer
                return int(target_str)
            else:
                # For other types, return as string
                return target_str
        except (ValueError, TypeError):
            # If parsing fails, return as string
            return target_str
    
    def get_memory_usage_estimate(self) -> Dict[str, int]:
        """Estimate current memory usage for context management"""
        history_chars = sum(len(str(entry)) for entry in self.state.history)
        recent_actions_chars = sum(len(action) for action in self.state.recent_actions)
        objectives_chars = sum(len(f"{obj.description} {obj.target_value}") for obj in self.state.objectives)
        
        return {
            "history_entries": len(self.state.history),
            "history_chars": history_chars, 
            "recent_actions": len(self.state.recent_actions),
            "recent_actions_chars": recent_actions_chars,
            "objectives_count": len(self.state.objectives),
            "objectives_chars": objectives_chars,
            "estimated_total_chars": history_chars + recent_actions_chars + objectives_chars
        }
    
    def get_objectives_state(self) -> Dict[str, Any]:
        """Get objectives formatted for forwarding in game state"""
        return {
            "active": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "type": obj.objective_type,
                    "target": obj.target_value,
                    "created_at": obj.created_at.isoformat()
                }
                for obj in self.get_active_objectives()
            ],
            "completed": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "type": obj.objective_type,
                    "target": obj.target_value,
                    "completed_at": obj.completed_at.isoformat() if obj.completed_at else None,
                    "notes": obj.progress_notes
                }
                for obj in self.get_completed_objectives()[-5:]  # Last 5 completed
            ],
            "updated": self.state.objectives_updated
        }
    
    def trim_history_for_context(self, max_chars: int = 4000):
        """Trim history to fit within context limits"""
        # Preserve minimum history for context
        min_history = max(5, self.history_display_count // 2)
        min_actions = max(10, self.actions_display_count // 2)
        
        while self.get_memory_usage_estimate()["estimated_total_chars"] > max_chars and len(self.state.history) > min_history:
            self.state.history.popleft()
            
        while len(self.state.recent_actions) > min_actions and self.get_memory_usage_estimate()["estimated_total_chars"] > max_chars:
            self.state.recent_actions.popleft()
    
    def reset_objectives_updated_flag(self):
        """Reset the objectives updated flag (call after forwarding state)"""
        self.state.objectives_updated = False
    
    def configure_history_limits(self, max_history_entries: int = None, max_recent_actions: int = None, 
                                history_display_count: int = None, actions_display_count: int = None):
        """Configure history tracking parameters at runtime"""
        if max_history_entries is not None:
            # Create new deque with updated max length, preserving existing data
            existing_history = list(self.state.history)
            self.state.history = deque(existing_history, maxlen=max_history_entries)
            
        if max_recent_actions is not None:
            # Create new deque with updated max length, preserving existing data
            existing_actions = list(self.state.recent_actions)
            self.state.recent_actions = deque(existing_actions, maxlen=max_recent_actions)
            
        if history_display_count is not None:
            self.history_display_count = history_display_count
            
        if actions_display_count is not None:
            self.actions_display_count = actions_display_count
        
        logger.info(f"Updated history configuration: {len(self.state.history)}/{self.state.history.maxlen} history, "
                   f"{len(self.state.recent_actions)}/{self.state.recent_actions.maxlen} actions, "
                   f"display {self.history_display_count}/{self.actions_display_count}")
    
    def load_history_from_llm_checkpoint(self, checkpoint_file: str):
        """Load SimpleAgent history from LLM checkpoint file"""
        try:
            from utils.llm_logger import get_llm_logger
            import json
            import re
            from datetime import datetime
            
            if not os.path.exists(checkpoint_file):
                logger.info(f"No checkpoint file found: {checkpoint_file}")
                return False
            
            # Use LLM logger to restore cumulative metrics first
            llm_logger = get_llm_logger()
            if llm_logger:
                restored_step_count = llm_logger.load_checkpoint(checkpoint_file)
                if restored_step_count is not None:
                    logger.info(f"✅ LLM logger restored checkpoint with {restored_step_count} steps")
                    # Update SimpleAgent step counter to match LLM logger
                    self.state.step_counter = restored_step_count
            
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            log_entries = checkpoint_data.get("log_entries", [])
            restored_count = 0
            
            for entry in log_entries:
                if entry.get("type") == "interaction" and "simple_mode" in entry.get("interaction_type", ""):
                    try:
                        # Extract state info from prompt
                        prompt = entry.get("prompt", "")
                        response = entry.get("response", "")
                        timestamp_str = entry.get("timestamp", "")
                        
                        # Parse coordinates from prompt
                        coords_match = re.search(r"Position: X=(\d+), Y=(\d+)", prompt)
                        coords = None
                        if coords_match:
                            coords = (int(coords_match.group(1)), int(coords_match.group(2)))
                        
                        # Parse context from prompt  
                        context = "overworld"  # default
                        if "Game State: battle" in prompt:
                            context = "battle"
                        elif "DIALOGUE:" in prompt or "dialogue" in prompt.lower():
                            context = "dialogue"
                        elif "menu" in prompt.lower():
                            context = "menu"
                        
                        # Extract action from response
                        action_taken = "UNKNOWN"
                        if "ACTION:" in response:
                            action_section = response.split("ACTION:")[-1].strip()
                            action_line = action_section.split('\n')[0].strip()
                            action_taken = action_line
                        
                        # Parse timestamp
                        timestamp = datetime.now()
                        if timestamp_str:
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str)
                            except:
                                pass
                        
                        # Create simplified game state summary
                        game_state_summary = f"Position: {coords}" if coords else "Position unknown"
                        if coords:
                            game_state_summary += f" | Context: {context}"
                        
                        # Add reasoning summary
                        reasoning = ""
                        if "REASONING:" in response:
                            reasoning_section = response.split("REASONING:")[-1].split("ACTION:")[0].strip()
                            reasoning = reasoning_section
                        
                        action_with_reasoning = f"{action_taken} | Reasoning: {reasoning}" if reasoning else action_taken
                        
                        # Create history entry
                        history_entry = HistoryEntry(
                            timestamp=timestamp,
                            player_coords=coords,
                            map_id=None,  # Not available in checkpoint
                            context=context,
                            action_taken=action_with_reasoning,
                            game_state_summary=game_state_summary
                        )
                        
                        self.state.history.append(history_entry)
                        
                        # Also add to recent actions if it's a valid action
                        if action_taken and action_taken not in ["UNKNOWN", "WAIT"]:
                            # Parse multiple actions if comma-separated
                            actions = [a.strip() for a in action_taken.replace(',', ' ').split()]
                            for action in actions:
                                if action in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT']:
                                    self.state.recent_actions.append(action)
                        
                        restored_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error parsing checkpoint entry: {e}")
                        continue
            
            # Update step counter to match checkpoint
            self.state.step_counter = restored_count
            
            logger.info(f"✅ Restored {restored_count} history entries from {checkpoint_file}")
            logger.info(f"   History: {len(self.state.history)} entries")
            logger.info(f"   Recent actions: {len(self.state.recent_actions)} actions")
            logger.info(f"   Step counter: {self.state.step_counter}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load history from checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_history_to_llm_checkpoint(self, checkpoint_file: str = None):
        """Save SimpleAgent history using LLM logger checkpoint system"""
        try:
            from utils.llm_logger import get_llm_logger
            
            # Get the global LLM logger instance
            llm_logger = get_llm_logger()
            if llm_logger is None:
                logger.warning("No LLM logger available for checkpoint saving")
                return False
            
            # Save checkpoint using LLM logger which includes cumulative metrics
            # The LLM logger will handle saving log_entries AND cumulative_metrics
            # If checkpoint_file is None, it will use the cache folder
            llm_logger.save_checkpoint(checkpoint_file, agent_step_count=self.state.step_counter)
            
            logger.info(f"💾 Saved LLM checkpoint to {checkpoint_file}")
            logger.info(f"   Step counter: {self.state.step_counter}")
            logger.info(f"   History: {len(self.state.history)} entries")
            logger.info(f"   Recent actions: {len(self.state.recent_actions)} actions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save LLM checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False

    def record_failed_movement(self, coords: Tuple[int, int], direction: str, reason: str = "blocked"):
        """Record a failed movement attempt for future reference"""
        coord_key = f"{coords[0]},{coords[1]}"
        if coord_key not in self.state.failed_movements:
            self.state.failed_movements[coord_key] = []
        
        failed_entry = f"{direction}:{reason}"
        if failed_entry not in self.state.failed_movements[coord_key]:
            self.state.failed_movements[coord_key].append(failed_entry)
            logger.info(f"Recorded failed movement: {coord_key} -> {direction} ({reason})")
    
    def record_npc_interaction(self, coords: Tuple[int, int], interaction_type: str, notes: str = ""):
        """Record an NPC interaction for future reference"""
        coord_key = f"{coords[0]},{coords[1]}"
        interaction_info = f"{interaction_type}: {notes}" if notes else interaction_type
        self.state.npc_interactions[coord_key] = interaction_info
        logger.info(f"Recorded NPC interaction: {coord_key} -> {interaction_info}")
    
    def get_movement_memory(self, coords: Tuple[int, int]) -> str:
        """Get memory about failed movements and interactions at specific coordinates"""
        coord_key = f"{coords[0]},{coords[1]}"
        memory_parts = []
        
        # Check for failed movements
        if coord_key in self.state.failed_movements:
            failed_list = self.state.failed_movements[coord_key]
            memory_parts.append(f"Failed moves: {', '.join(failed_list)}")
        
        # Check for NPC interactions
        if coord_key in self.state.npc_interactions:
            interaction = self.state.npc_interactions[coord_key]
            memory_parts.append(f"NPC: {interaction}")
        
        return " | ".join(memory_parts) if memory_parts else ""
    
    def get_area_movement_memory(self, center_coords: Tuple[int, int], radius: int = 7) -> str:
        """Get movement memory for the area around the player"""
        cx, cy = center_coords
        memory_lines = []
        
        # Check nearby coordinates for failed movements or NPC interactions
        nearby_memories = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip current position
                
                check_coords = (cx + dx, cy + dy)
                memory = self.get_movement_memory(check_coords)
                if memory:
                    nearby_memories.append(f"({check_coords[0]},{check_coords[1]}): {memory}")
        
        if nearby_memories:
            memory_lines.append("🧠 MOVEMENT MEMORY (nearby area):")
            for memory in nearby_memories[:5]:  # Limit to 5 most relevant
                memory_lines.append(f"  {memory}")
        
        return "\n".join(memory_lines)
    
    def analyze_movement_preview(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the movement preview data from game state to find valid moves.
        
        Returns:
            Dict with 'walkable_directions', 'blocked_directions', and 'special_tiles'
        """
        walkable_directions = []
        blocked_directions = []
        special_tiles = {}
        
        # Look for movement preview in the formatted state
        formatted_state = format_state_for_llm(game_state, include_movement_preview=False)
        lines = formatted_state.split('\n')
        
        in_movement_preview = False
        for line in lines:
            if 'MOVEMENT PREVIEW:' in line:
                in_movement_preview = True
                continue
            
            if in_movement_preview:
                # Parse movement preview lines
                # Format: "  UP   : ( 15, 10) [.] WALKABLE - Optional description"
                if line.strip() and ':' in line:
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        direction = parts[0].strip()
                        rest = parts[1].strip()
                        
                        if direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                            if 'WALKABLE' in rest:
                                walkable_directions.append(direction)
                                # Check for special tiles
                                if 'Door/Entrance' in rest:
                                    special_tiles[direction] = 'door'
                                elif 'Stairs/Warp' in rest:
                                    special_tiles[direction] = 'stairs'
                                elif 'Tall grass' in rest:
                                    special_tiles[direction] = 'grass'
                                elif 'Jump ledge' in rest and 'can jump' in rest:
                                    special_tiles[direction] = 'ledge'
                            elif 'BLOCKED' in rest:
                                blocked_directions.append(direction)
                elif not line.strip():
                    # Empty line typically ends the movement preview section
                    in_movement_preview = False
        
        return {
            'walkable_directions': walkable_directions,
            'blocked_directions': blocked_directions,
            'special_tiles': special_tiles
        }
    
    def validate_movement_sequence(self, movements: List[str], game_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if a sequence of movements is valid based on current state.
        
        Args:
            movements: List of movement directions
            game_state: Current game state
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not movements:
            return True, "No movements to validate"
        
        # Analyze current movement options
        movement_info = self.analyze_movement_preview(game_state)
        walkable = movement_info['walkable_directions']
        blocked = movement_info['blocked_directions']
        
        # Check first movement
        first_move = movements[0].upper()
        if first_move in blocked:
            return False, f"First movement {first_move} is BLOCKED"
        
        if first_move not in walkable and first_move in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return False, f"First movement {first_move} is not confirmed WALKABLE"
        
        # For multiple movements, only allow if we're very confident
        if len(movements) > 1:
            # We can't predict beyond the first move accurately
            # So we should discourage chaining unless explicitly safe
            return False, "Cannot validate multi-step movements - use single steps instead"
        
        return True, "Movement validated"

    def get_history_stats(self) -> Dict[str, int]:
        """Get current history tracking statistics"""
        return {
            "history_entries": len(self.state.history),
            "max_history_entries": self.state.history.maxlen,
            "recent_actions": len(self.state.recent_actions),
            "max_recent_actions": self.state.recent_actions.maxlen,
            "history_display_count": self.history_display_count,
            "actions_display_count": self.actions_display_count,
            "objectives_count": len(self.state.objectives),
            "step_counter": self.state.step_counter,
            "failed_movements": len(self.state.failed_movements),
            "npc_interactions": len(self.state.npc_interactions)
        }
    
    def set_npc_detection(self, enabled: bool):
        """Enable or disable NPC detection to improve performance"""
        self.ENABLE_NPC_DETECTION = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🔍 NPC detection {status}")
        logger.info(f"NPC detection {status}")
    
    def toggle_npc_detection(self):
        """Toggle NPC detection on/off"""
        self.ENABLE_NPC_DETECTION = not self.ENABLE_NPC_DETECTION
        status = "enabled" if self.ENABLE_NPC_DETECTION else "disabled"
        print(f"🔍 NPC detection toggled to {status}")
        logger.info(f"NPC detection toggled to {status}")
        return self.ENABLE_NPC_DETECTION

# Global simple agent instance for backward compatibility with existing multiprocess code
_global_simple_agent = None

def get_simple_agent(vlm) -> MySimpleAgent:
    """Get or create the global simple agent instance"""
    global _global_simple_agent
    if _global_simple_agent is None:
        _global_simple_agent = MySimpleAgent(vlm)
        
        # Check if we should load from checkpoint
        import os
        if os.environ.get("LOAD_CHECKPOINT_MODE") == "true":
            # Check cache folder first, then fall back to old location
            cache_dir = ".pokeagent_cache"
            checkpoint_file = os.path.join(cache_dir, "checkpoint_llm.txt") if os.path.exists(cache_dir) else "checkpoint_llm.txt"
            if not os.path.exists(checkpoint_file) and os.path.exists("checkpoint_llm.txt"):
                checkpoint_file = "checkpoint_llm.txt"
            if os.path.exists(checkpoint_file):
                logger.info(f"🔄 Loading SimpleAgent history from {checkpoint_file}")
                _global_simple_agent.load_history_from_llm_checkpoint(checkpoint_file)
            else:
                logger.info(f"⚠️ No checkpoint file found: {checkpoint_file}")
                
    elif _global_simple_agent.vlm != vlm:
        # VLM changed, create new instance
        _global_simple_agent = MySimpleAgent(vlm)
        
        # Load checkpoint for new instance too if mode is set
        import os
        if os.environ.get("LOAD_CHECKPOINT_MODE") == "true":
            # Check cache folder first, then fall back to old location
            cache_dir = ".pokeagent_cache"
            checkpoint_file = os.path.join(cache_dir, "checkpoint_llm.txt") if os.path.exists(cache_dir) else "checkpoint_llm.txt"
            if not os.path.exists(checkpoint_file) and os.path.exists("checkpoint_llm.txt"):
                checkpoint_file = "checkpoint_llm.txt"
            if os.path.exists(checkpoint_file):
                logger.info(f"🔄 Loading SimpleAgent history from {checkpoint_file}")
                _global_simple_agent.load_history_from_llm_checkpoint(checkpoint_file)
                
    return _global_simple_agent

def simple_mode_processing_multiprocess(vlm, game_state, args=None):
    """Simple mode processing function for multiprocess mode (backward compatibility)"""
    # args parameter kept for backward compatibility but not used
    _ = args  # Acknowledge unused parameter
    agent = get_simple_agent(vlm)
    frame = game_state["visual"]["screenshot"]
    
    # CRITICAL: Validate frame before processing
    if frame is None:
        logger.error("🚫 CRITICAL: simple_step called with None frame")
        return "WAIT"
    
    return agent.process_step(frame, game_state)


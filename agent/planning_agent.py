"""
High-level Planning Agent for Pokemon Emerald.

This agent manages the overall game progression by planning goals,
delegating to sub-agents, and updating plans based on outcomes.

Architecture:
    PlanningAgent (High Level)
    ├─ Action Type 1: plan - Decide goal, use MCP tools, generate subgoals
    ├─ Action Type 2: execute_subgoal - Delegate to sub-agents
    └─ Action Type 3: update_plan - When subgoals finish/interrupt

    Sub-Agents:
    1. ExploreAgent - Navigation, exploration, walking
    2. BattleAgent - Battles vs NPC/Pokemon, catching Pokemon
    3. UtilsAgent - Dialogue, shopping, naming, everything else

Integration with Hierarchical Agent:
    - Uses the same milestone tracking system (80+ storyline objectives)
    - Leverages MCP tools for planning: get_world_map, get_navigation_hints,
      get_walkthrough, search_knowledge, etc.
    - Planning agent focuses on high-level strategy, sub-agents handle execution
"""

import logging
import time
import requests
import json
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

# Import sub-agents
from agent.explore_agent import ExploreAgent
from agent.battle_agent import BattleAgent
from agent.utils_agent import UtilsAgent

logger = logging.getLogger(__name__)


# Pydantic models for structured output
class SubgoalSchema(BaseModel):
    """Schema for a single subgoal."""
    id: int = Field(description="Unique ID for this subgoal")
    description: str = Field(description="Clear, actionable description of what needs to be done")
    agent_type: Literal["EXPLORE", "BATTLE", "UTILS"] = Field(
        description="Which agent should handle this: EXPLORE for navigation, BATTLE for fights, UTILS for dialogue/menus/other"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the subgoal execution"
    )


class SubgoalsResponse(BaseModel):
    """Schema for the list of subgoals to accomplish a milestone."""
    reasoning: str = Field(description="Reasoning about the plan and why these subgoals were chosen")
    subgoals: List[SubgoalSchema] = Field(
        description="Ordered list of 2-5 subgoals needed to accomplish the milestone"
    )
    

class AgentMode(Enum):
    """Current mode of the planning agent."""
    PLANNING = "planning"  # Making/updating plan
    EXECUTING = "executing"  # Sub-agent is executing


class SubAgentType(Enum):
    """Types of sub-agents."""
    EXPLORE = "explore"  # Navigation, exploration, walking
    BATTLE = "battle"    # Battles, catching Pokemon
    UTILS = "utils"      # Dialogue, shopping, naming, etc.


@dataclass
class Subgoal:
    """Represents a single subgoal in the plan."""
    id: int
    description: str
    agent_type: SubAgentType
    status: str = "pending"  # pending, in_progress, completed, failed
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Milestone:
    """Represents a storyline milestone from the game."""
    id: str
    description: str
    objective_type: str  # "location", "battle", "item", "dialogue", "system"
    completed: bool = False
    target_value: Optional[str] = None


@dataclass
class PlanningState:
    """State of the planning agent."""
    mode: AgentMode = AgentMode.PLANNING
    current_goal: Optional[str] = None
    subgoals: List[Subgoal] = field(default_factory=list)
    current_subgoal_index: int = -1
    active_subagent: Optional[SubAgentType] = None

    # Milestone tracking (from hierarchical agent)
    milestones: List[Milestone] = field(default_factory=list)
    next_milestone: Optional[Milestone] = None

    # Frame accumulation (store all frames, sample 128 evenly for planning)
    frame_history: List[Dict[str, Any]] = field(default_factory=list)
    last_frame_time: float = 0.0
    frame_interval: float = 5.0  # Store frame every 5 seconds
    max_frames_for_planning: int = 128  # Sample this many frames for planning

    # Context from sub-agent returns
    subagent_context: List[str] = field(default_factory=list)

    # Planning context (passed to sub-agents)
    planning_context: Dict[str, Any] = field(default_factory=dict)

    # Planning metadata
    last_planning_time: float = 0.0
    planning_interval: float = 10.0  # Replan every 10 seconds if needed


class PlanningAgent:
    """
    High-level planning agent that orchestrates sub-agents.

    The planning agent has three types of actions:
    1. plan - Decide goal, use MCP tools, generate ordered subgoals
    2. execute_subgoal - Delegate to appropriate sub-agent
    3. update_plan - When subgoals finish or get interrupted
    """

    def __init__(self, vlm=None, mcp_server_url: str = "http://localhost:8000"):
        self.vlm = vlm
        self.mcp_server_url = mcp_server_url
        self.state = PlanningState()

        # Initialize sub-agents (each creates its own VLM instance)
        self.explore_agent = ExploreAgent(mcp_server_url)
        self.battle_agent = BattleAgent(mcp_server_url)
        self.utils_agent = UtilsAgent(mcp_server_url)

        # Initialize milestone tracking
        self._initialize_milestones()

        logger.info("🧠 PlanningAgent initialized")

    def _initialize_milestones(self):
        """
        Initialize storyline milestones from hierarchical agent.
        These are the 80+ objectives that track game progression.
        """
        storyline_milestones = [
            {"id": "GAME_RUNNING", "description": "Complete title sequence and begin the game", "type": "system", "target": "Game Running"},
            {"id": "PLAYER_NAME_SET", "description": "Set the player's name", "type": "system", "target": "Player Name Set"},
            {"id": "INTRO_CUTSCENE_COMPLETE", "description": "Complete intro cutscene with moving van", "type": "cutscene", "target": "Intro Complete"},
            {"id": "LITTLEROOT_TOWN", "description": "Arrive in Littleroot Town", "type": "location", "target": "Littleroot Town"},
            {"id": "PLAYER_HOUSE_ENTERED", "description": "Enter player's house for the first time", "type": "location", "target": "Player's House"},
            {"id": "PLAYER_BEDROOM", "description": "Go upstairs to player's bedroom", "type": "location", "target": "Player's Bedroom"},
            {"id": "CLOCK_SET", "description": "Set the clock on the wall", "type": "interaction", "target": "Clock Set"},
            {"id": "RIVAL_HOUSE", "description": "Visit May's house next door", "type": "location", "target": "Rival's House"},
            {"id": "RIVAL_BEDROOM", "description": "Visit May's bedroom on the second floor", "type": "location", "target": "Rival's Bedroom"},
            {"id": "ROUTE_101", "description": "Travel north to Route 101 and encounter Prof. Birch", "type": "location", "target": "Route 101"},
            {"id": "STARTER_CHOSEN", "description": "Choose starter Pokémon", "type": "pokemon", "target": "Starter Pokémon"},
            {"id": "BIRCH_LAB_VISITED", "description": "Visit Professor Birch's lab and receive the Pokedex", "type": "location", "target": "Birch's Lab"},
            {"id": "OLDALE_TOWN", "description": "Continue journey north to Oldale Town", "type": "location", "target": "Oldale Town"},
            {"id": "ROUTE_103", "description": "Travel to Route 103 to meet rival", "type": "location", "target": "Route 103"},
            {"id": "RIVAL_BATTLE_1", "description": "Battle rival for the first time", "type": "battle", "target": "Rival Battle 1"},
            {"id": "RECEIVED_POKEDEX", "description": "Receive the Pokédex", "type": "item", "target": "Pokédex"},
            {"id": "ROUTE_102", "description": "Travel through Route 102 toward Petalburg City", "type": "location", "target": "Route 102"},
            {"id": "PETALBURG_CITY", "description": "Arrive in Petalburg City", "type": "location", "target": "Petalburg City"},
            {"id": "DAD_FIRST_MEETING", "description": "Meet Dad at Petalburg City Gym", "type": "dialogue", "target": "Dad Meeting"},
            {"id": "GYM_EXPLANATION", "description": "Receive explanation about Gym challenges", "type": "dialogue", "target": "Gym Tutorial"},
            {"id": "ROUTE_104_SOUTH", "description": "Travel through southern Route 104", "type": "location", "target": "Route 104 South"},
            {"id": "PETALBURG_WOODS", "description": "Navigate through Petalburg Woods", "type": "location", "target": "Petalburg Woods"},
            {"id": "TEAM_AQUA_GRUNT_DEFEATED", "description": "Defeat Team Aqua Grunt in Petalburg Woods", "type": "battle", "target": "Aqua Grunt"},
            {"id": "ROUTE_104_NORTH", "description": "Travel through northern Route 104", "type": "location", "target": "Route 104 North"},
            {"id": "RUSTBORO_CITY", "description": "Arrive in Rustboro City", "type": "location", "target": "Rustboro City"},
            {"id": "RUSTBORO_GYM_ENTERED", "description": "Enter the Rustboro Gym", "type": "location", "target": "Rustboro Gym"},
            {"id": "ROXANNE_DEFEATED", "description": "Defeat Gym Leader Roxanne", "type": "battle", "target": "Roxanne"},
            {"id": "FIRST_GYM_COMPLETE", "description": "Receive the Stone Badge", "type": "badge", "target": "Stone Badge"},
        ]

        for ms_data in storyline_milestones:
            milestone = Milestone(
                id=ms_data["id"],
                description=ms_data["description"],
                objective_type=ms_data["type"],
                target_value=ms_data["target"]
            )
            self.state.milestones.append(milestone)

        logger.info(f"📋 Initialized {len(self.state.milestones)} storyline milestones")

    def step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main step function - routes to appropriate action based on mode.

        Args:
            game_state: Current game state from /state endpoint

        Returns:
            {"action": List[str]} - Button commands to execute
        """
        # Update milestone completion status
        self._update_milestones(game_state)

        # Accumulate frame history (one frame every 5 seconds)
        self._accumulate_frame(game_state)

        if self.state.mode == AgentMode.PLANNING:
            # Planning mode: generate/update plan (no buttons returned)
            return self._planning_step(game_state)

        elif self.state.mode == AgentMode.EXECUTING:
            # Executing mode: delegate to sub-agent
            return self._executing_step(game_state)

        else:
            logger.error(f"Unknown agent mode: {self.state.mode}")
            return {"action": ["WAIT"]}

    def _update_milestones(self, game_state: Dict[str, Any]) -> None:
        """Update milestone completion status from game state."""
        milestones_data = game_state.get("milestones", {})
        if not milestones_data:
            return

        for milestone in self.state.milestones:
            if not milestone.completed:
                milestone_info = milestones_data.get(milestone.id, {})
                if milestone_info.get("completed", False):
                    milestone.completed = True
                    logger.info(f"✅ Milestone completed: {milestone.description}")

                    # Find next incomplete milestone
                    self._update_next_milestone()

    def _update_next_milestone(self) -> None:
        """Determine the next incomplete milestone."""
        for milestone in self.state.milestones:
            if not milestone.completed:
                self.state.next_milestone = milestone
                logger.info(f"🎯 Next milestone: {milestone.description}")
                return

        self.state.next_milestone = None
        logger.info("🎉 All milestones completed!")

    def _accumulate_frame(self, game_state: Dict[str, Any]) -> None:
        """Store one frame every 5 seconds for planning context."""
        current_time = time.time()

        if (current_time - self.state.last_frame_time) >= self.state.frame_interval:
            # Store frame with timestamp
            frame_snapshot = {
                "timestamp": current_time,
                "player": game_state.get("player", {}),
                "game": game_state.get("game", {}),
                "map": game_state.get("map", {}),
                "frame": game_state.get("frame", {})  # Screenshot
            }

            self.state.frame_history.append(frame_snapshot)
            self.state.last_frame_time = current_time

            logger.debug(f"📸 Frame accumulated ({len(self.state.frame_history)} total)")

    def _get_sampled_frames(self) -> List[Dict[str, Any]]:
        """
        Get frames sampled evenly for planning.

        If we have more than 128 frames, sample 128 evenly spaced.
        Otherwise return all frames.
        """
        num_frames = len(self.state.frame_history)
        max_frames = self.state.max_frames_for_planning

        if num_frames == 0:
            return []

        if num_frames <= max_frames:
            # Use all frames
            return self.state.frame_history

        # Sample evenly
        indices = [int(i * num_frames / max_frames) for i in range(max_frames)]
        sampled = [self.state.frame_history[i] for i in indices]

        logger.debug(f"📊 Sampled {len(sampled)} frames from {num_frames} total")
        return sampled

    def _planning_step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planning mode: Decide goal and generate subgoals.

        Uses MCP tools to:
        - Get world map information
        - Look up walkthrough hints
        - Search knowledge base
        - Analyze current progress vs next milestone

        This does NOT return button commands, only updates internal state.
        """
        current_time = time.time()

        logger.info("🧠 Planning step")

        # Check if we need to create a new plan
        if not self.state.current_goal or not self.state.subgoals:
            self._create_plan(game_state)
            self.state.last_planning_time = current_time

        # Check if we need to update the plan
        elif self._should_update_plan(game_state):
            self._update_plan(game_state)
            self.state.last_planning_time = current_time

        # If we have subgoals, transition to executing mode
        if self.state.subgoals:
            self.state.mode = AgentMode.EXECUTING
            self.state.current_subgoal_index = 0

            # Set active subagent
            current_subgoal = self.state.subgoals[0]
            current_subgoal.status = "in_progress"
            self.state.active_subagent = current_subgoal.agent_type

            logger.info(f"🎯 Starting execution: {current_subgoal.description}")

        # Planning steps don't return buttons, just update state
        return {"action": ["WAIT"]}

    def _executing_step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executing mode: Delegate to appropriate sub-agent.

        Returns button commands from sub-agent.
        """
        if self.state.current_subgoal_index < 0:
            logger.error("No active subgoal!")
            self.state.mode = AgentMode.PLANNING
            return {"action": ["WAIT"]}

        # Get current subgoal
        current_subgoal = self.state.subgoals[self.state.current_subgoal_index]

        # Delegate to appropriate sub-agent
        result = self._delegate_to_subagent(game_state, current_subgoal)

        #TODO: parse the subagent result to return the correct action

        return {"action": ["WAIT"]}

    def _delegate_to_subagent(
        self,
        game_state: Dict[str, Any],
        subgoal: Subgoal
    ) -> Dict[str, Any]:
        """Delegate to the appropriate sub-agent based on subgoal type."""
        # Pass planning context to sub-agent
        planning_context = self.state.planning_context

        if subgoal.agent_type == SubAgentType.EXPLORE:
            return self.explore_agent.step(game_state, subgoal, planning_context)

        elif subgoal.agent_type == SubAgentType.BATTLE:
            return self.battle_agent.step(game_state, subgoal, planning_context)

        elif subgoal.agent_type == SubAgentType.UTILS:
            return self.utils_agent.step(game_state, subgoal, planning_context)

        else:
            logger.error(f"Unknown agent type: {subgoal.agent_type}")

    def _create_plan(self, game_state: Dict[str, Any]) -> None:
        """
        Create a new plan with goal and subgoals.

        Uses MCP tools to gather information, then VLM to generate plan.
        Stores planning context for sub-agents.
        """
        logger.info("📝 Creating new plan")

        # Determine next milestone
        if not self.state.next_milestone:
            self._update_next_milestone()

        if not self.state.next_milestone:
            logger.warning("No next milestone found - all objectives complete?")
            return

        next_milestone = self.state.next_milestone
        logger.info(f"📋 Planning for milestone: {next_milestone.description}")

        # Gather information using MCP tools
        world_map_info = ""
        nav_hints_info = ""
        walkthrough_info = ""

        try:
            # Get world map
            world_map_response = requests.post(
                f"{self.mcp_server_url}/mcp/get_world_map",
                json={},
                timeout=5
            )
            if world_map_response.ok:
                world_map = world_map_response.json()
                if world_map.get("success"):
                    world_map_info = world_map.get("map_description", "")
                    logger.debug(f"✓ World map retrieved")

            # Get navigation hints for target
            nav_hints_response = requests.post(
                f"{self.mcp_server_url}/mcp/get_navigation_hints",
                json={"target_area": next_milestone.target_value},
                timeout=5
            )
            if nav_hints_response.ok:
                nav_hints = nav_hints_response.json()
                if nav_hints.get("success"):
                    nav_hints_info = nav_hints.get("hints", "")
                    logger.debug(f"✓ Navigation hints retrieved")

            # Get walkthrough for this milestone
            walkthrough_response = requests.post(
                f"{self.mcp_server_url}/mcp/get_walkthrough",
                json={"query": next_milestone.description},
                timeout=5
            )
            if walkthrough_response.ok:
                walkthrough = walkthrough_response.json()
                if walkthrough.get("success"):
                    walkthrough_info = walkthrough.get("content", "")
                    logger.debug(f"✓ Walkthrough retrieved")

        except Exception as e:
            logger.warning(f"Error calling MCP tools: {e}")

        # Generate subgoals using VLM
        subgoals = self._generate_subgoals_for_milestone(
            next_milestone,
            game_state,
            world_map_info,
            nav_hints_info,
            walkthrough_info
        )

        # Create goal and store planning context
        self.state.current_goal = f"Complete milestone: {next_milestone.description}"
        self.state.subgoals = subgoals

        self.state.planning_context = {
            "goal": self.state.current_goal,
            "next_milestone": {
                "id": next_milestone.id,
                "description": next_milestone.description,
                "type": next_milestone.objective_type,
                "target": next_milestone.target_value
            },
            "world_map_info": world_map_info,
            "nav_hints": nav_hints_info,
            "walkthrough": walkthrough_info,
            "completed_subgoals": [],
            "failed_subgoals": [],
            "subgoal_count": len(self.state.subgoals),
            "frame_history_count": len(self.state.frame_history)
        }

        logger.info(f"📋 Plan created with {len(self.state.subgoals)} subgoals")
        for i, sg in enumerate(subgoals):
            logger.info(f"  {i+1}. [{sg.agent_type.value}] {sg.description}")

    def _generate_subgoals_for_milestone(
        self,
        milestone: Milestone,
        game_state: Dict[str, Any],
        world_map_info: str,
        nav_hints: str,
        walkthrough: str
    ) -> List[Subgoal]:
        """
        Generate subgoals using VLM structured output.

        Uses up to 128 sampled frames and MCP tool information to generate
        2-5 actionable subgoals for accomplishing the milestone.
        """
        import time

        # Get sampled frames for context
        sampled_frames = [frame_data.get('frame') for frame_data in self._get_sampled_frames()]
        current_frame = game_state.get('frame')
        
        sampled_frames.append(current_frame)

        # Get player state info
        player_state = game_state.get('game_state', {})
        player_location = player_state.get('player', {})

        # Build detailed prompt
        prompt = f"""You are a high-level planning agent for Pokemon Emerald speedrunning.

The images provided show {len(sampled_frames)} frames sampled from {len(self.state.frame_history)} total frames, showing the progression over time.

CURRENT MILESTONE:
- ID: {milestone.id}
- Description: {milestone.description}
- Type: {milestone.objective_type}
- Target: {milestone.target_value}

PLAYER STATE:
- Location: Map {player_location.get('map_id', 'unknown')}, Coords ({player_location.get('x', '?')}, {player_location.get('y', '?')})
- Frames provided: {len(sampled_frames)} frames showing recent progression

WORLD MAP INFORMATION:
{world_map_info}

NAVIGATION HINTS:
{nav_hints}

WALKTHROUGH GUIDANCE:
{walkthrough}

RECENT SUBAGENT CONTEXT:
{self.state.subagent_context[-3:] if self.state.subagent_context else "None"}

Your task is to break down this milestone into 2-5 concrete, actionable subgoals.

AGENT TYPES:
- EXPLORE: Use for navigation, moving between locations, finding NPCs/items
- BATTLE: Use for trainer battles, wild Pokemon encounters, catching Pokemon
- UTILS: Use for dialogue, menus, item usage, system interactions

REQUIREMENTS:
1. Each subgoal must be clear and actionable
2. Order subgoals logically (navigation before interaction, etc.)
3. Include relevant context for each subgoal (target locations, battle info, etc.)
4. Keep subgoals focused on the current milestone only
5. Use information from world map, navigation hints, and walkthrough

Generate the subgoals now."""

        # Retry logic: 3 attempts with 1-second delays
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Generating subgoals with VLM using {len(sampled_frames)} frames (attempt {attempt + 1}/{max_retries})")

                # Call VLM structured output with all sampled frames
                # (VLM backend automatically manages conversation history)
                response = self.vlm.get_structured_query(
                    img=sampled_frames,
                    text=prompt,
                    response_schema=SubgoalsResponse,
                    module_name="planning_subgoals"
                )

                # Convert SubgoalSchema objects to Subgoal dataclass instances
                subgoals = []
                for sg_schema in response.subgoals:
                    # Map agent_type string to enum
                    agent_type_map = {
                        "EXPLORE": SubAgentType.EXPLORE,
                        "BATTLE": SubAgentType.BATTLE,
                        "UTILS": SubAgentType.UTILS
                    }
                    agent_type = agent_type_map.get(sg_schema.agent_type, SubAgentType.EXPLORE)

                    subgoal = Subgoal(
                        id=sg_schema.id,
                        description=sg_schema.description,
                        agent_type=agent_type,
                        context=sg_schema.context
                    )
                    subgoals.append(subgoal)

                logger.info(f"✅ Generated {len(subgoals)} subgoals successfully")
                logger.info(f"💭 Reasoning: {response.reasoning}")
                for i, sg in enumerate(subgoals):
                    logger.info(f"   {i+1}. [{sg.agent_type.value}] {sg.description}")

                return subgoals

            except Exception as e:
                last_error = e
                logger.warning(f"⚠️  VLM subgoal generation failed (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    # Sleep before retry
                    time.sleep(1.0)
                    logger.info(f"🔄 Retrying...")

        # All retries failed
        error_msg = f"Failed to generate subgoals after {max_retries} attempts. Last error: {last_error}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    def _update_plan(self, game_state: Dict[str, Any]) -> None:
        """
        Update the current plan based on failures/successes.

        Appends to planning conversation history to maintain context.
        """
        logger.info("🔄 Updating plan based on failures/successes")

        # Get current milestone
        next_milestone = self.state.next_milestone
        if not next_milestone:
            logger.warning("⚠️  No milestone to replan for")
            return

        # Analyze current execution results
        failed_subgoals = [sg for sg in self.state.subgoals if sg.status == "failed"]
        completed_subgoals = [sg for sg in self.state.subgoals if sg.status == "completed"]

        logger.info(f"📊 Execution results: {len(completed_subgoals)} completed, {len(failed_subgoals)} failed")

        # Build update message describing what happened
        update_message = f"""UPDATE ON EXECUTION:

Completed subgoals ({len(completed_subgoals)}):"""

        if completed_subgoals:
            for sg in completed_subgoals:
                update_message += f"\n- ✓ Subgoal {sg.id}: {sg.description}"
                if sg.result:
                    update_message += f"\n  Result: {sg.result}"
        else:
            update_message += "\n- None"

        update_message += f"\n\nFailed/Interrupted subgoals ({len(failed_subgoals)}):"

        if failed_subgoals:
            for sg in failed_subgoals:
                update_message += f"\n- ✗ Subgoal {sg.id}: {sg.description}"
                update_message += f"\n  Agent: {sg.agent_type.value}"
                update_message += f"\n  Result: {sg.result or 'No result recorded'}"
        else:
            update_message += "\n- None"

        # Add recent subagent context
        if self.state.subagent_context:
            update_message += "\n\nRecent execution context:"
            for ctx in self.state.subagent_context[-3:]:
                update_message += f"\n- {ctx}"

        update_message += f"""

Please generate new subgoals that:
1. DO NOT repeat completed subgoals
2. Account for what failed and try a different approach
3. Build on what was accomplished so far
4. Keep focusing on the milestone: {next_milestone.description}"""

        # Get sampled frames for current state
        sampled_frames = [frame_data.get('frame') for frame_data in self._get_sampled_frames()]
        current_frame = game_state.get('frame')
        if current_frame:
            sampled_frames.append(current_frame)

        # Generate new subgoals using conversation history
        subgoals = self._generate_subgoals_with_conversation(
            milestone=next_milestone,
            game_state=game_state,
            update_message=update_message,
            sampled_frames=sampled_frames
        )

        # Update state
        self.state.subgoals = subgoals
        self.state.current_subgoal_index = -1

        # Update planning context with execution results
        self.state.planning_context["completed_subgoals"] = [
            {"id": sg.id, "description": sg.description, "result": sg.result}
            for sg in completed_subgoals
        ]
        self.state.planning_context["failed_subgoals"] = [
            {"id": sg.id, "description": sg.description, "result": sg.result}
            for sg in failed_subgoals
        ]
        self.state.planning_context["subgoal_count"] = len(self.state.subgoals)
        self.state.planning_context["frame_history_count"] = len(self.state.frame_history)

        logger.info(f"📋 Plan updated with {len(subgoals)} new subgoals")
        for i, sg in enumerate(subgoals):
            logger.info(f"  {i+1}. [{sg.agent_type.value}] {sg.description}")

    def _generate_subgoals_with_conversation(
        self,
        milestone: Milestone,
        game_state: Dict[str, Any],
        update_message: str,
        sampled_frames: List[Any]
    ) -> List[Subgoal]:
        """
        Generate updated subgoals using conversation history.

        Appends the update message to conversation history and gets new subgoals.
        """
        import time

        # Get player state
        player_state = game_state.get('game_state', {})
        player_location = player_state.get('player', {})

        # Build prompt with current state
        prompt = f"""{update_message}

CURRENT GAME STATE:
- Location: Map {player_location.get('map_id', 'unknown')}, Coords ({player_location.get('x', '?')}, {player_location.get('y', '?')})
- Frames provided: {len(sampled_frames)} frames showing current progression

Generate the updated subgoals now."""

        # Retry logic
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Updating subgoals with conversation history using {len(sampled_frames)} frames (attempt {attempt + 1}/{max_retries})")

                # Call VLM with conversation history
                # (VLM backend automatically manages conversation history)
                response = self.vlm.get_structured_query(
                    img=sampled_frames,
                    text=prompt,
                    response_schema=SubgoalsResponse,
                    module_name="planning_update"
                )

                # Convert to Subgoal dataclass instances
                subgoals = []
                for sg_schema in response.subgoals:
                    agent_type_map = {
                        "EXPLORE": SubAgentType.EXPLORE,
                        "BATTLE": SubAgentType.BATTLE,
                        "UTILS": SubAgentType.UTILS
                    }
                    agent_type = agent_type_map.get(sg_schema.agent_type, SubAgentType.EXPLORE)

                    subgoal = Subgoal(
                        id=sg_schema.id,
                        description=sg_schema.description,
                        agent_type=agent_type,
                        context=sg_schema.context
                    )
                    subgoals.append(subgoal)

                logger.info(f"✅ Updated {len(subgoals)} subgoals successfully")
                logger.info(f"💭 Reasoning: {response.reasoning}")
                for i, sg in enumerate(subgoals):
                    logger.info(f"   {i+1}. [{sg.agent_type.value}] {sg.description}")

                return subgoals

            except Exception as e:
                last_error = e
                logger.warning(f"⚠️  VLM subgoal update failed (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    time.sleep(1.0)
                    logger.info(f"🔄 Retrying...")

        # All retries failed
        error_msg = f"Failed to update subgoals after {max_retries} attempts. Last error: {last_error}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    def _should_update_plan(self, game_state: Dict[str, Any]) -> bool:
        """
        Check if the plan needs to be updated.

        Returns True if:
        - A subgoal failed
        - Milestone was completed (need new milestone)
        - Too much time passed since last plan
        """
        # Check if any subgoal failed
        for subgoal in self.state.subgoals:
            if subgoal.status == "failed":
                return True

        # Check if milestone changed
        if self.state.next_milestone:
            if self.state.next_milestone.completed:
                logger.info("Milestone completed, need new plan")
                return True

        # Check if too much time passed
        current_time = time.time()
        if (current_time - self.state.last_planning_time) > self.state.planning_interval:
            logger.info("Planning interval exceeded, considering replan")
            # Only replan if we're not making progress
            # TODO: Add progress detection
            return False

        return False

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current planning state."""
        completed_milestones = [m for m in self.state.milestones if m.completed]

        return {
            "mode": self.state.mode.value,
            "goal": self.state.current_goal,
            "active_subagent": self.state.active_subagent.value if self.state.active_subagent else None,
            "next_milestone": {
                "id": self.state.next_milestone.id,
                "description": self.state.next_milestone.description
            } if self.state.next_milestone else None,
            "milestones_completed": len(completed_milestones),
            "milestones_total": len(self.state.milestones),
            "subgoals": [
                {
                    "id": sg.id,
                    "description": sg.description,
                    "agent": sg.agent_type.value,
                    "status": sg.status
                }
                for sg in self.state.subgoals
            ],
            "frames_collected": len(self.state.frame_history),
            "subagent_context_items": len(self.state.subagent_context)
        }

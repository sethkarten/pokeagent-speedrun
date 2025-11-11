"""
Explore Agent - Handles navigation and exploration in Pokemon Emerald.

Responsibilities:
- Navigate to coordinates
- Explore areas
- Find items
- Move around the world
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, Any, List
from urllib import request
from pydantic import BaseModel, Field
from typing import Literal
import time
import requests

logger = logging.getLogger(__name__)


class SubAgentActionResponse(BaseModel):
    """Schema for sub-agent action response.

    The sub-agent can choose one of three action types:
    - High-level action: Use predefined tools (e.g., navigate_to, find_npc)
    - press_buttons: Direct button inputs
    - complete_subgoal: Mark subgoal as done with status
    """
    reasoning: str = Field(description="Reasoning about what to do next")
    action: Literal["high_level_action", "press_buttons", "complete_subgoal"] = Field(
        description="Type of action to take"
    )
    action_detail: Dict[str, Any] = Field(
        description=(
            "Details for the action. "
            "For 'high_level_action': {tool_name: str, tool_input: dict}. "
            "For 'press_buttons': {buttons: [list of button strings]}. "
            "For 'complete_subgoal': {status: str, context: str}"
        )
    )

class JudgementResponse(BaseModel):
    """Schema for judgement response."""
    subgoal_completed: bool = Field(description="Whether the subgoal is completed")
    back_to_planning: bool = Field(description="Whether to return to high-level planner")

@dataclass
class ExplorationTypes(Enum):
    NAVIGATE = "navigate"
    FIND = "find"
    EXPLORE = "explore"
    MOVE_AREA = "move_area"

@dataclass
class Coordinate:
    x: int
    y: int
    is_blocked: bool

class ExplorationTypeResponse(BaseModel):
    """Schema for exploration type response.

    The sub-agent can choose one of the 4 exploration types:
    - a. navigate to an (unblocked) coordinate in current area
    - b. finding item/npc in current area
    - c. explore the current area
    - d. move to another area (cross-area navigation)
    """
    exploration_type: Literal[
        "navigate",
        "find",
        "explore",
        "move_area"
    ] = Field(description="Type of exploration to perform")
    details: Dict[str, Any] = Field(
        description="More specific details on where the player should go or what to find"
    )

class CoordinateResponse(BaseModel):
    """Schema for coordinate response."""
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")
    is_blocked: bool = Field(description="Whether the coordinate is blocked by structure, npc or item or not")

class FindItemState(BaseModel):
    """Schema for finding item state response."""
    target_name: str = Field(description="Name of the item/NPC to find")
    target_coordinate: Coordinate = Field(description="Coordinate of the item/NPC to find")
    found: bool = Field(description="Whether the item/NPC is found in the current area")
    visited_locations: List[Coordinate] = Field(
        description="List of coordinates already visited while searching for the item/NPC"
    )
    planned_locations: List[Coordinate] = Field(
        description="List of coordinates planned to visit next while searching for the item/NPC"
    )
    details: Dict[str, Any] = Field(
        description="More specific details on how to find the item/NPC or why it cannot be found"
    )

class ExploreAgent:
    """
    Sub-agent for navigation, exploration, and walking.

    Responsibilities:
    - Navigate to coordinates
    - Explore areas
    - Find items
    - Move around the world
    """

    def __init__(self, mcp_server_url: str):
        from utils.vlm import VLM

        self.mcp_server_url = mcp_server_url
        self.vlm = VLM()  # Create own VLM instance with own conversation history
        self.find_state = FindItemState(
            target_name="",
            target_coordinate=Coordinate(x=-1, y=-1, is_blocked=False),
            found=False,
            visited_locations=[],
            planned_locations=[],
            details={}
        )
        self._initiate_tools()
    
    def _initiate_tools(self):
        """
        Initialize tools for the sub-agent.
        """
        self.tools = [
            {
                "name": "navigate_to",
                "description": "Move to specified coordinates in the current area.",
                "input_schema": {
                    "x": "int - X coordinate to navigate to",
                    "y": "int - Y coordinate to navigate to",
                    "reason": "str - Reason for navigation"
                },
                "output_schema": {
                    "success": "bool - Whether the navigation is successful"
                }
            },
            {
                "name": "navigate_interact",
                "description": "Move to specified coordinates and interact with the target item or npc.", 
                "input_schema": {
                    "x": "int - X coordinate to navigate to",
                    "y": "int - Y coordinate to navigate to",
                }, 
                "output_schema": {
                    "success": "bool - Whether the navigation is successful"
                }
            }, 
            {
                "name": "get_world_map",
                "description": "Get world map information to assist in cross-area navigation.",
                "input_schema": {},
                "output_schema": {
                    "success": "bool - Whether the world map retrieval is successful",
                    "world_map_overview": "str - Overview of the world map",
                }
            }, 
            {
                "name": "get_navigation_hints",
                "description": "Get navigation hints to assist in navigation and exploration.",
                "input_schema": {
                    "target_area_name": "str - Target area to navigate to"
                },
                "output_schema": {
                    "success": "bool - Whether the navigation hints retrieval is successful",
                    "navigation_hints": "str - Useful navigation hints",
                }
            },
            {
                "name": "search_knowledge",
                "description": "Search external knowledge base for information to assist in navigation and exploration.",
                "input_schema": {
                    "query": "str - Query string to search in the knowledge base"
                },
                "output_schema": {
                    "success": "bool - Whether the knowledge search is successful",
                    "query": "str - The original query string",
                    "search_results": "Dict[str, str] - Search results with titles and snippets",
                }
            }, 
            {
                "name": "add_knowledge",
                "description": "Add new knowledge to the knowledge base.",
                "input_schema": {
                    "key": "str - Knowledge key or title",
                    "value": "str - Knowledge content or description"
                },
                "output_schema": {
                    "success": "bool - Whether the knowledge addition was successful",
                    "key": "str - Knowledge key or title",
                    "value": "str - Knowledge content or description"
                }
            }
        ]

    def step(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,  # Subgoal dataclass
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step for exploration subgoal.
        """
        # 1. determine the nature of the subgoal by calling VLM
        exploration_type = self._determine_exploration_type(
            game_state,
            subgoal,
            planning_context
        )
        # 2. User different prompts based on the nature determined above
        if exploration_type == ExplorationTypes.NAVIGATE:
            logger.info("Exploration type determined: NAVIGATE")
            prompt = self._get_navi_prompt(subgoal, game_state, planning_context)
        elif exploration_type == ExplorationTypes.FIND:
            logger.info("Exploration type determined: FIND")
            prompt = self._get_find_prompt(subgoal, game_state, planning_context)
        elif exploration_type == ExplorationTypes.EXPLORE:  
            prompt = self._get_explore_prompt(subgoal, game_state, planning_context)
        elif exploration_type == ExplorationTypes.MOVE_AREA:
            prompt = self._get_move_area_prompt(subgoal, game_state, planning_context)  
        else:
            # In case where it cannot be classified, return to high-level planner
            logger.warning("Exploration type could not be determined, returning to high-level planner.")
            return {"actions": "WAIT", 
                    "reasoning": "Could not determine exploration type, please replan or refine the subgoal.",
                    "back_to_planning": True}

        # Call the VLM with the constructed prompt
        # and parse the response to get the next action
        response = self.vlm.get_structured_query(
            text=prompt,
            response_schema=SubAgentActionResponse,
            module_name="explore_agent",
        )
        output = []
        # extract the response actions and execute them
        if response.action == "high_level_action":
            tool_name = response.action_detail.get("tool_name", "")
            tool_input = response.action_detail.get("tool_input", {})
            try:
                result = self._execute_tool(tool_name, tool_input)   
                if result.get("success"):
                    logger.info(f"Successfully executed tool: {tool_name} with input: {tool_input}")
                    output.append(f"Executed tool: {tool_name} successfully.")
                    output.append(result)
                else:
                    logger.error(f"Failed to execute tool: {tool_name} with input: {tool_input}, error: {result.get('error')}")
                    return {"actions": ["WAIT"],
                        "reasoning": f"Failed to execute tool: {tool_name} due to error: {result.get('error')}",
                        "subgoal_completed": False,
                        "back_to_planning": True}
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return {"actions": ["WAIT"],
                        "reasoning": f"Failed to execute tool: {tool_name} due to error: {result.get('error')}",
                        "subgoal_completed": False,
                        "back_to_planning": True}
        elif response.action == "press_buttons":
            buttons = response.action_detail.get("buttons", [])
            try:
                action_response = requests.post(f"{self.mcp_server_url}/mcp/press_buttons", json={
                    "buttons": buttons,
                })
                action_response.raise_for_status()
                result = action_response.json()
                if result.get("success"):
                    logger.info(f"Successfully pressed buttons: {buttons}")
                else:
                    logger.error(f"Failed to press buttons: {result.get('error')}")
                    return {"actions": ["WAIT"],
                        "reasoning": f"Failed to press buttons due to error: {e}",
                        "subgoal_completed": False,
                        "back_to_planning": True}
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to call press_buttons: {e}")
                return {"actions": ["WAIT"],
                        "reasoning": f"Failed to press buttons due to error: {e}",
                        "subgoal_completed": False,
                        "back_to_planning": True}
        elif response.action == "complete_subgoal":
            status = response.action_detail.get("status", "unknown")
            context = response.action_detail.get("context", "")
            logger.info(f"Subgoal marked as complete with status: {status}, context: {context}")
            return {"actions": ["Mark subgoal as complete"],
                    "reasoning": response.reasoning,
                    "subgoal_completed": True,
                    "back_to_planning": True}
        else: 
            logger.error(f"Unknown action type: {response.action}")
            return {"actions": ["WAIT"],
                    "reasoning": f"VLM Returned unknown action type: {response.action}",
                    "subgoal_completed": False,
                    "back_to_planning": True}

        # 3. determine if subgoal is completed, and whether to return to high-level planner
        judging_state_prompt = f"""You are a lower-level exploration and navigation agent for Pokemon Emerald speedrunning.

        You've just taken the following action to achieve the subgoal:
        {response.action} with details {response.action_detail}
        SUBGOAL: 
        {subgoal.description}, with context: {subgoal.context}

        CURRENT GAME STATE:
        {game_state}

        Please respond:
        1. Is the subgoal completed? (True/False)
        2. Should you return to the high-level planner for further instructions? (True/False)
        """
        judging_response = self.vlm.get_structured_query(
            text=judging_state_prompt,
            response_schema=JudgementResponse,
            module_name="explore_agent",
        )
        back_to_planning = judging_response.back_to_planning
        subgoal_completed = judging_response.subgoal_completed
        return {
            "actions": ["WAIT"],
            "message": output,
            "subgoal_completed": subgoal_completed,
            "back_to_planning": back_to_planning
        }

    def _determine_exploration_type(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,
        planning_context: Dict[str, Any]
    ) -> ExplorationTypeResponse:
        """
        Determine the type of exploration to perform by calling VLM.

        Returns an ExplorationTypeResponse indicating the type and details.
        """
        # Get player's location
        current_location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state.get("player", {}).get("position")
        world_map_info = planning_context.get("world_map_info", "No map info available.")
        nav_hints = planning_context.get("navigation_hints", "No navigation hints available.")

        # Build detailed prompt
        prompt = f"""You are a lower-level exploration and navigation agent for Pokemon Emerald speedrunning. 

SUBGOAL: {subgoal.description}, with context: {subgoal.context}

PLAYER STATE:
Location: {current_location}
Position: {player_pos}

WORLD MAP INFORMATION:
{world_map_info}

NAVIGATION HINTS:
{nav_hints}

You are given a specific navigation-related subgoal to achieve as part of a larger milestone. Your task is to determine which type of exploration is most appropriate to achieve the subgoal.

EXPLORATION TYPES:
1. NAVIGATION: Moving within the current area to specific coordinates
2. FINDING: Searching for a specific item or NPC in the current area
3. EXPLORATION: Thoroughly exploring the current area to uncover hidden items or paths
4. MOVE-AREA: Moving to a different area entirely

REQUIREMENTS:
1. Analyze the subgoal and context carefully to understand its nature
2. Consider the player's current location and position
3. Use the world map information to assess possible routes and areas
4. Use information from world map, navigation hints, and walkthrough
"""

        # Retry logic: 3 attempts with 1-second delays
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Calling VLM to determine exploration type (attempt {attempt + 1}/{max_retries})...")

                # Call VLM structured output with all sampled frames
                # (VLM backend automatically manages conversation history)
                response = self.vlm.get_structured_query(
                    text=prompt,
                    response_schema=ExplorationTypeResponse,
                    module_name="explore_agent",
                )
                # convert response to ExplorationType
                exploration_type = ExplorationTypes[response.exploration_type]
                return exploration_type

            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Attempt {attempt + 1} failed with error: {e}")

                if attempt < max_retries - 1:
                    # Sleep before retry
                    time.sleep(1.0)
                    logger.info(f"🔄 Retrying...")

        # All retries failed
        error_msg = f"Failed to generate subgoals after {max_retries} attempts. Last error: {last_error}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """
        Execute a tool based on its name and input.
        """
        if tool_name == "navigate_to":
            x = tool_input.get("x")
            y = tool_input.get("y")
            reason = tool_input.get("reason", "No reason provided")
            return self._execute_navigation(x, y, reason)
        elif tool_name == "navigate_interact":
            x = tool_input.get("x")
            y = tool_input.get("y")
            return self._execute_interaction(x, y)
        elif tool_name == "get_world_map":
            try:
                response = requests.post(f"{self.mcp_server_url}/mcp/get_world_map", json={})
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"Successfully retrieved world map information.")
                    return result
                else:
                    logger.error(f"Failed to retrieve world map: {result.get('error')}")
                    return result
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to call get_world_map: {e}")
                return {"success": False, "error": str(e)}
        elif tool_name == "get_navigation_hints":
            target_area_name = tool_input.get("target_area_name", "")
            try:
                response = requests.post(f"{self.mcp_server_url}/mcp/get_navigation_hints", json={
                    "target_area": target_area_name
                })
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"Successfully retrieved navigation hints.")
                    return result
                else:
                    logger.error(f"Failed to retrieve navigation hints: {result.get('error')}")
                    return result
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to call get_navigation_hints: {e}")
                return {"success": False, "error": str(e)}
        elif tool_name == "search_knowledge":
            query = tool_input.get("query", "")
            try:
                response = requests.post(f"{self.mcp_server_url}/mcp/search_knowledge", json={
                    "query": query
                })
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"Successfully searched knowledge base.")
                    return result
                else:
                    logger.error(f"Failed to search knowledge: {result.get('error')}")
                    return result
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to call search_knowledge: {e}")
                return {"success": False, "error": str(e)}
        elif tool_name == "add_knowledge":
            key = tool_input.get("key", "")
            value = tool_input.get("value", "")
            try:
                response = requests.post(f"{self.mcp_server_url}/mcp/add_knowledge", json={
                    "key": key,
                    "value": value
                })
                response.raise_for_status()
                result = response.json()
                if result.get("success"):
                    logger.info(f"Successfully added knowledge to the knowledge base.")
                    return result
                else:
                    logger.error(f"Failed to add knowledge: {result.get('error')}")
                    return result
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to call add_knowledge: {e}")
                return {"success": False, "error": str(e)}
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")
        
    def _get_navi_prompt(self, subgoal: Any, game_state: Dict[str, Any], exploration: ExplorationTypeResponse) -> str:
        """
        Generate navigation prompt for VLM.
        """
        current_location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state.get("player", {}).get("position")
        current_map_info = game_state.get("map", {}).get("stitched_map_info", "No map info available.")
        exploration_details = exploration.details
        return f"""You are a lower-level exploration and navigation agent for Pokemon Emerald speedrunning.
        Your are navigating to a specific area to achieve the subgoal: {subgoal.description}, with context: {subgoal.context}.
        DETAILS: {exploration_details}
        CURRENT GAME STATE:
        player location: {current_location}
        player position: {player_pos}
        CURRENT MAP INFORMATION:
        {current_map_info}
        Your task is to determine which coordinate to navigate to, and then navigate there.
        Return one action at a time.
        You can choose one of three action types:
        - High-level action: Use predefined tools.
        - press_buttons: Direct button inputs.
        - complete_subgoal: Mark subgoal as done with status.
        The following are the tools available to you:
        {self.tools}
        """

    def _get_find_prompt(self, subgoal: Any, game_state: Dict[str, Any], exploration: ExplorationTypeResponse) -> str:
        """
        Generate finding prompt for VLM.
        """
        exploration_details = exploration.details
        current_location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state.get("player", {}).get("position")
        current_map_info = game_state.get("map", {}).get("stitched_map_info", "No map info available.")
        return f"""You are a lower-level exploration and navigation agent for Pokemon Emerald speedrunning.
        Your are finding a specific item or NPC to achieve the subgoal: {subgoal.description}, with context: {subgoal.context}.
        DETAILS: {exploration_details}
        CURRENT GAME STATE:
        player location: {current_location}
        player position: {player_pos}
        CURRENT MAP INFORMATION:
        {current_map_info}
        Your task is to locate the item/NPC in the current area, navigating to the right coordinates and interacting with it.
        Return one action at a time.
        You can choose one of three action types:
        - High-level action: Use predefined tools.
        - press_buttons: Direct button inputs.
        - complete_subgoal: Mark subgoal as done with status.
        The following are the tools available to you:
        {self.tools}
        """

    def _get_explore_prompt(self, subgoal: Any, game_state: Dict[str, Any], exploration: ExplorationTypeResponse) -> str:
        """
        Generate exploration prompt for VLM.
        """
        exploration_details = exploration.details
        current_location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state.get("player", {}).get("position")
        current_map_info = game_state.get("map", {}).get("stitched_map_info", "No map info available.")
        return f"""You are a lower-level exploration and navigation agent for Pokemon Emerald speedrunning.
        Your are exploring a specific area to achieve the subgoal: {subgoal.description}, with context: {subgoal.context}.
        DETAILS: {exploration_details}
        CURRENT GAME STATE:
        player location: {current_location}
        player position: {player_pos}
        CURRENT MAP INFORMATION:
        {current_map_info}
        Your task is to explore the current area, navigating to the right coordinates and interacting with it.
        Return one action at a time.
        You can choose one of three action types:
        - High-level action: Use predefined tools.
        - press_buttons: Direct button inputs.
        - complete_subgoal: Mark subgoal as done with status.
        The following are the tools available to you:
        {self.tools}
        """

    def _get_move_area_prompt(self, subgoal: Any, game_state: Dict[str, Any], exploration: ExplorationTypeResponse) -> str:
        """
        Generate move-area prompt for VLM.
        """
        exploration_details = exploration.details
        current_location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state.get("player", {}).get("position")
        current_map_info = game_state.get("map", {}).get("stitched_map_info", "No map info available.")
        return f"""You are a lower-level exploration and navigation agent for Pokemon Emerald speedrunning.
        Your are to achieve the subgoal: {subgoal.description}, with context: {subgoal.context} by moving to another area in the world map.
        DETAILS: {exploration_details}
        CURRENT GAME STATE:
        player location: {current_location}
        player position: {player_pos}
        CURRENT MAP INFORMATION:
        {current_map_info}
        Your task is to navigate to the target area specified in or deduced from the subgoal.
        Return one action at a time.
        You can choose one of three action types:
        - High-level action: Use predefined tools.
        - press_buttons: Direct button inputs.
        - complete_subgoal: Mark subgoal as done with status.
        The following are the tools available to you:
        {self.tools}
        """

    def _execute_navigation(self, x, y, reason) -> str:
        """Executes a navigation task using the pathfinder."""

        try:
            response = requests.post(f"{self.mcp_server_url}/mcp/navigate_to", json={
                "x": x,
                "y": y,
                "reason": reason,
            })
            response.raise_for_status()
            result = response.json()
            if result.get("success"):
                logger.info(f"Pathfinding successful")
                return {"success": True}
            else:
                logger.error(f"Pathfinding failed: {result.get('error')}")
                return {"success": False, "error": result.get("error")}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call navigate_to: {e}")
            return {"success": False, "error": str(e)}


    def _execute_interaction(self, x, y) -> List[str]:
        """Executes an interaction task using the pathfinder."""
        try:
            self._execute_navigation(x, y, "Navigate and interact with target")
            # wait for arrival
            response = requests.post(f"{self.mcp_server_url}/get_comprehensive_state", json={})
            response.raise_for_status()
            result = response.json()
            player_pos = result.get("player", {}).get("position")
            if player_pos:
                player_coords = (player_pos.get("x", 0), player_pos.get("y", 0))
            target_coords = {"x": x, "y": y}
            key = ""
            if target_coords and player_coords:
                # switch for 4 cases: target is on the left, right, above, below, and default (not within range)
                if (abs(player_coords[0] - target_coords["x"]) == 1 and player_coords[1] == target_coords["y"]):
                    key = "LEFT"
                elif (abs(player_coords[0] - target_coords["x"]) == -1 and player_coords[1] == target_coords["y"]):
                    key = "RIGHT"
                elif (player_coords[0] == target_coords["x"] and abs(player_coords[1] - target_coords["y"]) == 1):
                    key = "ABOVE"
                elif (player_coords[0] == target_coords["x"] and abs(player_coords[1] - target_coords["y"]) == -1):
                    key = "BELOW"
                else:
                    logger.error(f"Not in interaction range for target")
                    return False
            action_response = requests.post(f"{self.mcp_server_url}/mcp/press_buttons", json={
                "buttons": [key + "A"],
            })
            action_response.raise_for_status()
            result = action_response.json()
            if result.get("success"):
                logger.info(f"Successfully executed interaction.")
                return {"success": True}
            else:
                logger.error(f"Failed to execute interaction: {result.get('error')}")
                return {"success": False, "error": result.get("error")}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to call get_game_state after navigation: {e}")
            return {"success": False, "error": str(e)}

    def _navigate_in_area(self, subgoal: Any, game_state: Dict[str, Any], exploration: ExplorationTypeResponse, planning_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call VLM to figure out the coordinates to go to in the current area based on the subgoal.
        """
        current_location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state.get("player", {}).get("position")
        current_map_info = game_state.get("map", {}).get("stitched_map_info", "No map info available.")
        exploration_details = exploration.details

        # Build prompt
        prompt = f"""You are a lower-level exploration and navigation agent for Pokemon Emerald speedrunning. 
        
SUBGOAL: {subgoal.description}, with context: {subgoal.context}. 
DETAILS: {exploration_details}
PLAYER STATE:
Location: {current_location}
Position: {player_pos}
CURRENT MAP INFORMATION:
{current_map_info}
Your task is to determine the specific coordinates to navigate to in order to achieve the subgoal, and whether the coordinate is blocked or not.

REQUIREMENTS:
1. Analyze the subgoal and context carefully to understand its nature
2. Consider the player's current location and position
3. Use the current map information to determine the best coordinates to navigate to
"""
        # Retry logic: 3 attempts with 1-second delays
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Calling VLM to determine navigation coordinates (attempt {attempt + 1}/{max_retries})...")

                # Call VLM structured output with all sampled frames
                # (VLM backend automatically manages conversation history)
                response = self.vlm.get_structured_query(
                    text=prompt,
                    response_schema=CoordinateResponse,
                    module_name="explore_agent",
                )
                x, y, is_blocked = response.x, response.y, response.is_blocked
                response = requests.post(
                    f"{self.mcp_server_url}/mcp/navigate_to",
                    json={"x": x, "y": y, "reason": exploration_details},
                    timeout=5
                )
                response.raise_for_status()
                if response.ok:
                    logger.info(f"✅ Navigation command sent to MCP for coordinates ({x}, {y})")
                    return {"nav_result": "SUCCESS", 
                            "is_blocked": is_blocked}
                else:
                    logger.error(f"❌ Failed to send navigation command to MCP: {response.text}")
                    return {"nav_result": "FAILED",
                            "is_blocked": is_blocked}

            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Attempt {attempt + 1} failed with error: {e}")

                if attempt < max_retries - 1:
                    # Sleep before retry
                    time.sleep(1.0)
                    logger.info(f"🔄 Retrying...")
        
 
    def _find_in_area(self, subgoal: Any, game_state: Dict[str, Any], exploration: ExplorationTypeResponse) -> Dict[str, Any]:
        """
        Call VLM to figure out how to find the item/npc in the current area based on the subgoal.
        """
        current_location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state.get("player", {}).get("position")
        current_map_info = game_state.get("map", {}).get("stitched_map_info", "No map info available.")
        exploration_details = exploration.details

        # Build prompt
        prompt = f"""You are a lower-level exploration and navigation agent for Pokemon Emerald speedrunning.

SUBGOAL: {subgoal.description}, with context: {subgoal.context}

PLAYER STATE:
Location: {current_location}
Position: {player_pos}

CURRENT MAP INFORMATION:
{current_map_info}

EXPLORATION DETAILS:
{exploration_details}

Your task is to determine how to find the specified item or NPC in the current area.

FINDING STRATEGIES:
1. Use the current map information to locate the item/NPC
2. Make sure the player is facing the item/NPC when reaching the target coordinate
3. Output one action at a time to gradually approach and find the item/NPC

Output format:
Output one action at a time.
The sub-agent can choose one of three action types:
- High-level action: Use predefined tools as listed below:
    - navigate_to: Move to specified coordinates. 
- press_buttons: Direct button inputs
- complete_subgoal: Mark subgoal as done with status
"""
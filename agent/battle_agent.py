"""
Battle Agent - Handles battles and catching Pokemon in Pokemon Emerald.

Responsibilities:
- Battle wild Pokemon
- Battle trainers
- Catch Pokemon
- Manage battle strategy
"""

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


class SubAgentActionResponse(BaseModel):
    """Schema for sub-agent action response.

    The sub-agent can choose one of three action types:
    - High-level action: Use predefined tools (e.g., use_move, switch_pokemon)
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


class BattleAgent:
    """
    Sub-agent for battles and catching Pokemon.

    Responsibilities:
    - Battle wild Pokemon
    - Battle trainers
    - Catch Pokemon
    - Manage battle strategy
    """

    def __init__(self, mcp_server_url: str):
        from utils.vlm import VLM

        self.mcp_server_url = mcp_server_url
        self.vlm = VLM()  # Create own VLM instance with own conversation history

    def step(
        self,
        game_state: Dict[str, Any],
        subgoal: Any,  # Subgoal dataclass
        planning_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step for battle subgoal.
        """

        # TODO: Implement with VLM using SubAgentActionResponse schema
        # 1. Build prompt with subgoal, context, and recent frames
        # 2. Call self.vlm.get_structured_query() with SubAgentActionResponse schema
        # 3. Parse response and handle action type:
        #    - high_level_action: Call MCP tool and return WAIT
        #    - press_buttons: Return buttons from action_detail
        #    - complete_subgoal: Return with completed/failed/interrupted status

        pass

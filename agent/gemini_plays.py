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
import time
import requests
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque

import google.generativeai as genai

from utils.vlm import VLM
from utils.llm_logger import LLMLogger
from utils.state_formatter import format_state_for_llm
from utils.pathfinding import Pathfinder
from utils.agent_helpers import update_server_metrics

logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """Adapter to call MCP server tools via HTTP."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool via HTTP request to the game server."""
        try:
            # Map tool names to server endpoints
            endpoint_map = {
                "get_game_state": "/mcp/get_game_state",
                "press_buttons": "/mcp/press_buttons",
                "navigate_to": "/mcp/navigate_to",
                "complete_direct_objective": "/mcp/complete_direct_objective",
                "create_direct_objectives": "/mcp/create_direct_objectives",
                "get_progress_summary": "/mcp/get_progress_summary",
            }

            endpoint = endpoint_map.get(tool_name)
            if not endpoint:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

            url = f"{self.server_url}{endpoint}"
            logger.info(f"🔧 Calling MCP tool: {tool_name}")

            # Log arguments, but exclude large base64 data
            args_for_log = {k: f"<{len(v)} bytes>" if k == "screenshot_base64" and isinstance(v, str) and len(v) > 100 else v
                           for k, v in arguments.items()}
            logger.debug(f"   Args: {args_for_log}")

            response = requests.post(url, json=arguments, timeout=30)
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ Tool {tool_name} completed")
            return result

        except Exception as e:
            logger.error(f"❌ Tool {tool_name} failed: {e}")
            return {"success": False, "error": str(e)}


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
        response_text = self._extract_text_from_response(response) if hasattr(self, '_extract_text_from_response') else str(response)

        try:
            coords = response_text.strip().split(',')
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
        response_text = self._extract_text_from_response(response) if hasattr(self, '_extract_text_from_response') else str(response)

        # Parse response and convert to buttons
        buttons = []
        if "move" in response_text.lower():
            buttons = ["A"]  # Select fight
            if "2" in response_text:
                buttons.extend(["DOWN", "A"])
            elif "3" in response_text:
                buttons.extend(["RIGHT", "A"])
            elif "4" in response_text:
                buttons.extend(["DOWN", "RIGHT", "A"])
            else:
                buttons.append("A")  # Default to move 1
        elif "switch" in response_text.lower():
            buttons = ["RIGHT", "A"]  # Go to Pokemon menu
            # Add navigation for specific Pokemon
        elif "run" in response_text.lower():
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

        response_text = self._extract_text_from_response(response) if hasattr(self, '_extract_text_from_response') else str(response)

        # Parse button sequence
        buttons = []
        for part in response_text.upper().split(','):
            button = part.strip()
            if button in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R"]:
                buttons.append(button)
        
        return {
            "success": bool(buttons),
            "buttons": buttons[:10],  # Limit to 10 buttons
            "message": f"Puzzle moves: {', '.join(buttons[:10])}"
        }


def _create_gemini_plays_tools():
    """Create comprehensive function declarations for GeminiPlaysAgent tools."""
    tools = [
        # =====================================================================
        # GAME CONTROL TOOLS (require MCP server)
        # =====================================================================
        {
            "name": "get_game_state",
            "description": "Get the current game state including player position, party Pokemon, map, items, and a screenshot.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "press_buttons",
            "description": "Press Game Boy Advance buttons to interact with the game. Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R, WAIT.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "buttons": {
                        "type_": "ARRAY",
                        "items": {"type_": "STRING"},
                        "description": "List of buttons to press (e.g., ['A'], ['UP'])"
                    },
                    "reasoning": {
                        "type_": "STRING",
                        "description": "Explain why you are pressing these buttons"
                    }
                },
                "required": ["buttons", "reasoning"]
            }
        },
        {
            "name": "navigate_to",
            "description": "Automatically navigate to specific coordinates using A* pathfinding with porymap ground truth.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "x": {"type_": "INTEGER", "description": "Target X coordinate"},
                    "y": {"type_": "INTEGER", "description": "Target Y coordinate"},
                    "variance": {
                        "type_": "STRING",
                        "description": "Path variance level: 'none', 'low', 'medium', 'high', 'extreme'",
                        "enum": ["none", "low", "medium", "high", "extreme"]
                    },
                    "reason": {"type_": "STRING", "description": "Why you are navigating here"}
                },
                "required": ["x", "y", "variance", "reason"]
            }
        },

        # =====================================================================
        # GOAL MANAGEMENT TOOLS (native/local execution)
        # =====================================================================
        {
            "name": "update_primary_goal",
            "description": "Update or set the primary goal (main progression objective like defeating gym leader, reaching new city).",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "description": {"type_": "STRING", "description": "Clear description of the primary goal"},
                    "completion_conditions": {
                        "type_": "ARRAY",
                        "items": {"type_": "STRING"},
                        "description": "List of conditions that indicate goal completion"
                    }
                },
                "required": ["description", "completion_conditions"]
            }
        },
        {
            "name": "update_secondary_goal",
            "description": "Update or set the secondary goal (goal that enables the primary goal, like training Pokemon, getting items).",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "description": {"type_": "STRING", "description": "Clear description of the secondary goal"},
                    "completion_conditions": {
                        "type_": "ARRAY",
                        "items": {"type_": "STRING"},
                        "description": "List of conditions that indicate goal completion"
                    }
                },
                "required": ["description", "completion_conditions"]
            }
        },
        {
            "name": "update_tertiary_goal",
            "description": "Update or set the tertiary goal (opportunistic/exploratory goal like catching Pokemon, exploring areas).",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "description": {"type_": "STRING", "description": "Clear description of the tertiary goal"},
                    "completion_conditions": {
                        "type_": "ARRAY",
                        "items": {"type_": "STRING"},
                        "description": "List of conditions that indicate goal completion"
                    }
                },
                "required": ["description", "completion_conditions"]
            }
        },
        {
            "name": "mark_goal_complete",
            "description": "Mark a goal as completed when you've achieved its objectives.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "goal_type": {
                        "type_": "STRING",
                        "enum": ["primary", "secondary", "tertiary"],
                        "description": "Which goal to mark as complete"
                    },
                    "reasoning": {"type_": "STRING", "description": "Why this goal is now complete"}
                },
                "required": ["goal_type", "reasoning"]
            }
        },
        {
            "name": "get_current_goals",
            "description": "Get information about all current goals (primary, secondary, tertiary) and their status.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {},
                "required": []
            }
        },

        # =====================================================================
        # MEMORY/KNOWLEDGE TOOLS (native/local execution)
        # =====================================================================
        {
            "name": "add_to_notepad",
            "description": "Add important information to your long-term notepad for future reference.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "note": {"type_": "STRING", "description": "Important information to remember"}
                },
                "required": ["note"]
            }
        },
        {
            "name": "read_notepad",
            "description": "Read recent entries from your notepad to recall important information.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "last_n_entries": {
                        "type_": "INTEGER",
                        "description": "Number of recent entries to read (default: 5)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "mark_location_important",
            "description": "Mark a specific location as a point of interest (Pokemon Center, Gym, NPC, etc.).",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "x": {"type_": "INTEGER", "description": "X coordinate"},
                    "y": {"type_": "INTEGER", "description": "Y coordinate"},
                    "description": {"type_": "STRING", "description": "What makes this location important"}
                },
                "required": ["x", "y", "description"]
            }
        },
        {
            "name": "get_exploration_status",
            "description": "Get current exploration statistics (percentage explored, unexplored neighbors).",
            "parameters": {
                "type_": "OBJECT",
                "properties": {},
                "required": []
            }
        },

        # =====================================================================
        # SELF-CRITIQUE TOOLS (native/local execution)
        # =====================================================================
        {
            "name": "analyze_recent_actions",
            "description": "Trigger self-critique analysis of recent actions to identify if you're stuck or making poor decisions.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "get_stuck_analysis",
            "description": "Check if you appear to be stuck and get suggestions for recovery.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {},
                "required": []
            }
        },

        # =====================================================================
        # META TOOLS (native/local execution)
        # =====================================================================
        {
            "name": "define_custom_agent",
            "description": "Define a custom mini-agent for a specific recurring task.",
            "parameters": {
                "type_": "OBJECT",
                "properties": {
                    "name": {"type_": "STRING", "description": "Name of the custom agent"},
                    "task_description": {"type_": "STRING", "description": "What this agent should do"}
                },
                "required": ["name", "task_description"]
            }
        },
    ]
    return tools


class GeminiPlaysAgent:
    """
    Implementation of Gemini Plays Pokemon architecture with MCP tool integration.

    Key features:
    - Hierarchical goal management
    - Map memory with fog-of-war
    - MCP tools (press_buttons, navigate_to with porymap pathfinding)
    - Self-critique and improvement
    - Context reset with summarization
    - Exploration directives
    """

    def __init__(
        self,
        vlm_client: Optional[VLM] = None,
        server_url: str = "http://localhost:8000",
        context_reset_interval: int = 100,
        enable_self_critique: bool = True,
        enable_exploration: bool = True,
        enable_meta_tools: bool = False,  # Disabled by default for simplicity
        use_mcp_tools: bool = True,  # Enable MCP tools by default
        verbose: bool = True
    ):
        """
        Initialize the GeminiPlaysAgent.

        Args:
            vlm_client: Vision-language model client
            server_url: MCP server URL for tool integration
            context_reset_interval: Steps before context reset (default 100 like original)
            enable_self_critique: Enable self-critique mechanism
            enable_exploration: Enable forced exploration
            enable_meta_tools: Enable meta-tools (define_agent, execute_script, notepad)
            use_mcp_tools: Use MCP tools (press_buttons, navigate_to) instead of internal agents
            verbose: Detailed logging
        """
        self.server_url = server_url
        self.context_reset_interval = context_reset_interval
        self.enable_self_critique = enable_self_critique
        self.enable_exploration = enable_exploration
        self.enable_meta_tools = enable_meta_tools
        self.use_mcp_tools = use_mcp_tools
        self.verbose = verbose

        # Core components
        self.step_count = 0
        self.map_memory = MapMemory()
        self.context = AgentContext()
        self.llm_logger = LLMLogger()

        # Tool integration (MCP + native tools)
        if self.use_mcp_tools:
            self.mcp_adapter = MCPToolAdapter(server_url)
            self.tools = _create_gemini_plays_tools()
            logger.info(f"✅ Tools enabled: {len(self.tools)} tools available (3 MCP + {len(self.tools)-3} native)")
        else:
            self.mcp_adapter = None
            self.tools = []

        # Initialize VLM client (should already have tools configured)
        if vlm_client:
            self.vlm_client = vlm_client
            # Verify tools are configured in VLM
            if self.use_mcp_tools:
                if not hasattr(vlm_client, 'tools') or not vlm_client.tools:
                    logger.warning("⚠️ VLM client has no tools configured! Function calling may not work.")
                else:
                    logger.info(f"✅ VLM client has {len(vlm_client.tools)} tools configured")
        else:
            # Create default VLM with tools
            self.vlm_client = VLM(
                backend='gemini',
                model_name='gemini-2.0-flash-exp',
                tools=self.tools if self.use_mcp_tools else []
            )
            logger.info(f"🔧 Created default VLM with {len(self.tools) if self.use_mcp_tools else 0} tools")

        # Hierarchical goals
        self.primary_goal: Optional[Goal] = None
        self.secondary_goal: Optional[Goal] = None
        self.tertiary_goal: Optional[Goal] = None

        # Specialized agents (only used if MCP tools disabled)
        if not self.use_mcp_tools:
            self.pathfinding_agent = PathfindingAgent(vlm_client)
            self.battle_agent = BattleAgent(vlm_client)
            self.puzzle_agent = PuzzleAgent(vlm_client)
        else:
            self.pathfinding_agent = None
            self.battle_agent = None
            self.puzzle_agent = None

        # Action queue
        self.action_queue: deque = deque()

        # Performance tracking for self-critique
        self.recent_actions: deque = deque(maxlen=20)
        self.stuck_counter = 0
        self.last_position = (0, 0)

    def _extract_text_from_response(self, response) -> str:
        """Extract text from response (handles both string and GenerateContentResponse)."""
        if isinstance(response, str):
            return response

        # Handle response.candidates structure first (more reliable)
        if hasattr(response, 'candidates') and response.candidates:
            try:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        for part in candidate.content.parts:
                            # If it's a function call, skip it
                            if hasattr(part, 'function_call'):
                                continue
                            # Check if it's a text part (not a function_call)
                            # Use try/except to safely access text
                            try:
                                if hasattr(part, 'text'):
                                    text = part.text
                                    if text:
                                        return text
                            except (ValueError, AttributeError):
                                continue
            except Exception as e:
                logger.debug(f"Could not extract text from candidates: {e}")

        # Try accessing .text property (but catch ValueError for function calls)
        # CRITICAL: Don't use hasattr() - it can return True even for function_call responses
        try:
            text = response.text
            if text:
                return text
        except ValueError as e:
            # This happens when response contains function_call instead of text
            logger.debug(f"Response contains function call, not text: {e}")
            return ""
        except AttributeError:
            # Response doesn't have .text property
            pass
        except Exception as e:
            logger.debug(f"Could not access response.text: {e}")

        # Fallback to empty string (don't convert to str as it's too verbose)
        return ""

    def _convert_protobuf_args(self, proto_args) -> dict:
        """Convert protobuf arguments to JSON-serializable Python types."""
        arguments = {}
        for key, value in proto_args.items():
            # Convert protobuf types to native Python types
            if hasattr(value, '__class__') and 'proto' in value.__class__.__module__:
                # Check if it's a list-like type first
                if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                    try:
                        arguments[key] = list(value)
                    except:
                        arguments[key] = value
                else:
                    try:
                        arguments[key] = dict(value)
                    except:
                        arguments[key] = value
            else:
                arguments[key] = value
        return arguments

    def _execute_tool(self, function_call) -> str:
        """Execute a tool (MCP or native) and return the result as JSON string."""
        function_name = function_call.name

        # Parse arguments - convert protobuf types to native Python types
        try:
            arguments = self._convert_protobuf_args(function_call.args)
        except Exception as e:
            logger.error(f"Failed to parse function arguments: {e}")
            return json.dumps({"success": False, "error": f"Invalid arguments: {e}"})

        # Route to appropriate execution handler
        # MCP tools: get_game_state, press_buttons, navigate_to
        if function_name in ["get_game_state", "press_buttons", "navigate_to"]:
            if not self.use_mcp_tools or not self.mcp_adapter:
                return json.dumps({"success": False, "error": "MCP tools not enabled"})
            result = self.mcp_adapter.call_tool(function_name, arguments)
            return json.dumps(result, indent=2)

        # Native tools: Execute locally
        elif function_name == "update_primary_goal":
            return self._tool_update_primary_goal(**arguments)
        elif function_name == "update_secondary_goal":
            return self._tool_update_secondary_goal(**arguments)
        elif function_name == "update_tertiary_goal":
            return self._tool_update_tertiary_goal(**arguments)
        elif function_name == "mark_goal_complete":
            return self._tool_mark_goal_complete(**arguments)
        elif function_name == "get_current_goals":
            return self._tool_get_current_goals()
        elif function_name == "add_to_notepad":
            return self._tool_add_to_notepad(**arguments)
        elif function_name == "read_notepad":
            return self._tool_read_notepad(**arguments)
        elif function_name == "mark_location_important":
            return self._tool_mark_location_important(**arguments)
        elif function_name == "get_exploration_status":
            return self._tool_get_exploration_status()
        elif function_name == "analyze_recent_actions":
            return self._tool_analyze_recent_actions()
        elif function_name == "get_stuck_analysis":
            return self._tool_get_stuck_analysis()
        elif function_name == "define_custom_agent":
            return self._tool_define_custom_agent(**arguments)
        else:
            return json.dumps({"success": False, "error": f"Unknown tool: {function_name}"})

    # =========================================================================
    # NATIVE TOOL IMPLEMENTATIONS
    # =========================================================================

    def _tool_update_primary_goal(self, description: str, completion_conditions: List[str]) -> str:
        """Update primary goal."""
        self.primary_goal = Goal(
            type="primary",
            description=description,
            conditions=completion_conditions,
            priority=1,
            created_at=self.step_count
        )
        if self.verbose:
            print(f"[GOAL] Updated primary goal: {description}")
        return json.dumps({"success": True, "message": f"Primary goal updated: {description}"})

    def _tool_update_secondary_goal(self, description: str, completion_conditions: List[str]) -> str:
        """Update secondary goal."""
        self.secondary_goal = Goal(
            type="secondary",
            description=description,
            conditions=completion_conditions,
            priority=2,
            created_at=self.step_count
        )
        if self.verbose:
            print(f"[GOAL] Updated secondary goal: {description}")
        return json.dumps({"success": True, "message": f"Secondary goal updated: {description}"})

    def _tool_update_tertiary_goal(self, description: str, completion_conditions: List[str]) -> str:
        """Update tertiary goal."""
        self.tertiary_goal = Goal(
            type="tertiary",
            description=description,
            conditions=completion_conditions,
            priority=3,
            created_at=self.step_count
        )
        if self.verbose:
            print(f"[GOAL] Updated tertiary goal: {description}")
        return json.dumps({"success": True, "message": f"Tertiary goal updated: {description}"})

    def _tool_mark_goal_complete(self, goal_type: str, reasoning: str) -> str:
        """Mark a goal as complete."""
        if goal_type == "primary" and self.primary_goal:
            self.primary_goal.completed = True
            if self.verbose:
                print(f"[GOAL] Primary goal completed: {reasoning}")
            return json.dumps({"success": True, "message": f"Primary goal marked complete: {reasoning}"})
        elif goal_type == "secondary" and self.secondary_goal:
            self.secondary_goal.completed = True
            if self.verbose:
                print(f"[GOAL] Secondary goal completed: {reasoning}")
            return json.dumps({"success": True, "message": f"Secondary goal marked complete: {reasoning}"})
        elif goal_type == "tertiary" and self.tertiary_goal:
            self.tertiary_goal.completed = True
            if self.verbose:
                print(f"[GOAL] Tertiary goal completed: {reasoning}")
            return json.dumps({"success": True, "message": f"Tertiary goal marked complete: {reasoning}"})
        else:
            return json.dumps({"success": False, "error": f"No {goal_type} goal to mark complete"})

    def _tool_get_current_goals(self) -> str:
        """Get current goals and their status."""
        goals_info = {
            "primary": {
                "description": self.primary_goal.description if self.primary_goal else None,
                "completed": self.primary_goal.completed if self.primary_goal else False,
                "conditions": self.primary_goal.conditions if self.primary_goal else [],
                "progress": self.primary_goal.progress if self.primary_goal else ""
            },
            "secondary": {
                "description": self.secondary_goal.description if self.secondary_goal else None,
                "completed": self.secondary_goal.completed if self.secondary_goal else False,
                "conditions": self.secondary_goal.conditions if self.secondary_goal else [],
                "progress": self.secondary_goal.progress if self.secondary_goal else ""
            },
            "tertiary": {
                "description": self.tertiary_goal.description if self.tertiary_goal else None,
                "completed": self.tertiary_goal.completed if self.tertiary_goal else False,
                "conditions": self.tertiary_goal.conditions if self.tertiary_goal else [],
                "progress": self.tertiary_goal.progress if self.tertiary_goal else ""
            }
        }
        return json.dumps({"success": True, "goals": goals_info})

    def _tool_add_to_notepad(self, note: str) -> str:
        """Add note to notepad."""
        self.context.notepad.append(f"[Step {self.step_count}] {note}")
        # Keep notepad size manageable
        if len(self.context.notepad) > 20:
            self.context.notepad = self.context.notepad[-20:]
        if self.verbose:
            print(f"[NOTE] Added to notepad: {note[:50]}...")
        return json.dumps({"success": True, "message": "Note added to notepad"})

    def _tool_read_notepad(self, last_n_entries: int = 5) -> str:
        """Read notepad entries."""
        entries = self.context.notepad[-last_n_entries:] if self.context.notepad else []
        return json.dumps({
            "success": True,
            "entries": entries,
            "total_entries": len(self.context.notepad)
        })

    def _tool_mark_location_important(self, x: int, y: int, description: str) -> str:
        """Mark location as point of interest."""
        self.map_memory.points_of_interest[(x, y)] = description
        if self.verbose:
            print(f"[MAP] Marked ({x}, {y}) as important: {description}")
        return json.dumps({
            "success": True,
            "message": f"Marked ({x}, {y}) as: {description}"
        })

    def _tool_get_exploration_status(self) -> str:
        """Get exploration statistics."""
        exploration_pct = self.map_memory.get_exploration_percentage(1000)
        total_explored = len(self.map_memory.explored_tiles)
        total_pois = len(self.map_memory.points_of_interest)
        return json.dumps({
            "success": True,
            "exploration_percentage": round(exploration_pct, 2),
            "tiles_explored": total_explored,
            "points_of_interest": total_pois,
            "important_locations": list(self.map_memory.points_of_interest.values())
        })

    def _tool_analyze_recent_actions(self) -> str:
        """Trigger self-critique analysis."""
        recent_actions_str = ", ".join(list(self.recent_actions)[-10:])

        # This would normally call the LLM for analysis, but for now return a summary
        analysis = f"Recent actions: {recent_actions_str}. "
        analysis += f"Step count: {self.step_count}. "

        if self.stuck_counter > 5:
            analysis += f"WARNING: Stuck counter at {self.stuck_counter}!"

        return json.dumps({
            "success": True,
            "analysis": analysis,
            "stuck_counter": self.stuck_counter,
            "recent_actions": list(self.recent_actions)[-10:]
        })

    def _tool_get_stuck_analysis(self) -> str:
        """Check if stuck and get suggestions."""
        is_stuck = self.stuck_counter > 5
        suggestions = []

        if is_stuck:
            suggestions.append("Try a different direction or approach")
            suggestions.append("Consider using navigate_to with higher variance")
            suggestions.append("Check if you're trying to walk through an obstacle")

        return json.dumps({
            "success": True,
            "is_stuck": is_stuck,
            "stuck_counter": self.stuck_counter,
            "suggestions": suggestions,
            "last_position": list(self.last_position)
        })

    def _tool_define_custom_agent(self, name: str, task_description: str) -> str:
        """Define a custom mini-agent."""
        agent_def = f"""Custom agent '{name}' for: {task_description}
Created at step {self.step_count}"""

        self.context.custom_agents[name] = agent_def

        if self.verbose:
            print(f"[BOT] Created custom agent: {name}")

        return json.dumps({
            "success": True,
            "message": f"Custom agent '{name}' created",
            "definition": agent_def
        })

    def step(self, state: Dict[str, Any], screenshot: Any = None) -> str:
        """
        Execute one step following Gemini Plays Pokemon architecture.

        Args:
            state: Current game state
            screenshot: Current game screenshot

        Returns:
            Button command (e.g., "A", "UP", "NONE") or "MCP_TOOL_EXECUTED"
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

        # Update goals
        self._update_goals(state)

        # Decide next action using function calling
        action = self._decide_action(state, screenshot)

        # Update server with agent step and metrics (for agent thinking display)
        update_server_metrics()

        self.recent_actions.append(action)
        return action

    def run(self, max_steps: int = None) -> int:
        """
        Run the agent loop (similar to my_cli_agent.run()).

        Args:
            max_steps: Maximum number of steps to run (None for unlimited)

        Returns:
            0 if successful, 1 if error
        """
        import json as json_module
        import base64
        from PIL import Image
        import io
        import time

        logger.info("=" * 70)
        logger.info("🎮 GeminiPlaysAgent with Native Tools")
        logger.info("=" * 70)
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Tools: {len(self.tools)} tools (3 MCP + {len(self.tools)-3} native)")
        logger.info(f"Context reset: Every {self.context_reset_interval} steps")
        if max_steps:
            logger.info(f"Max steps: {max_steps}")
        logger.info("=" * 70)

        logger.info("\n🚀 Starting agent loop...")
        logger.info("Press Ctrl+C to stop")
        logger.info("-" * 70)

        try:
            while True:
                # Check max steps
                if max_steps and self.step_count >= max_steps:
                    logger.info(f"\n✅ Reached max steps ({max_steps})")
                    break

                logger.info(f"\n{'='*70}")
                logger.info(f"🤖 Step {self.step_count + 1}")
                logger.info(f"{'='*70}")

                # Fetch game state via MCP
                try:
                    game_state_result = self.mcp_adapter.call_tool("get_game_state", {})

                    # Parse game state
                    if not game_state_result.get("success"):
                        logger.error(f"❌ Failed to get game state: {game_state_result.get('error')}")
                        time.sleep(3)
                        continue

                    # Extract screenshot
                    screenshot_b64 = game_state_result.get("screenshot_base64")
                    screenshot = None
                    if screenshot_b64:
                        try:
                            image_data = base64.b64decode(screenshot_b64)
                            screenshot = Image.open(io.BytesIO(image_data))

                            # Check for black frame (transition screen)
                            if self._is_black_frame(screenshot):
                                logger.info("⏳ Black frame detected (likely a transition), waiting for next frame...")
                                time.sleep(1)
                                continue

                        except Exception as e:
                            logger.warning(f"Failed to decode screenshot: {e}")

                    # Use the state_text from get_game_state (includes porymap data)
                    # Plus extract key fields for internal use
                    state = {
                        'state_text': game_state_result.get('state_text', ''),  # Includes porymap!
                        'player_position': game_state_result.get('player_position', {}),
                        'location': game_state_result.get('location', 'Unknown'),
                        'team': game_state_result.get('team', []),
                        'badges': game_state_result.get('badges', 0),
                        'in_battle': game_state_result.get('in_battle', False),
                        'items': game_state_result.get('items', []),
                        'money': game_state_result.get('money', 0)
                    }

                    # Run step (calls _decide_action which executes tools)
                    action = self.step(state, screenshot)

                    # CRITICAL: Wait for action queue to complete if action tool was called
                    # This ensures button presses finish before the next step starts
                    if action == "TOOL_EXECUTED":
                        self._wait_for_actions_complete()

                    if action and action != "WAIT":
                        logger.info(f"✅ Step {self.step_count} completed: {action}")

                    # Update server metrics
                    try:
                        update_server_metrics(self.server_url)
                    except Exception as e:
                        logger.debug(f"Failed to update server metrics: {e}")

                    # Auto-save checkpoint every 10 steps
                    if self.step_count % 10 == 0:
                        try:
                            import requests
                            checkpoint_response = requests.post(
                                f"{self.server_url}/checkpoint",
                                json={"step_count": self.step_count},
                                timeout=10
                            )
                            if checkpoint_response.status_code == 200:
                                logger.info(f"💾 Checkpoint saved at step {self.step_count}")
                        except Exception as e:
                            logger.debug(f"Checkpoint save failed: {e}")

                    # Small delay between steps
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"❌ Step failed: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(3)
                    continue

        except KeyboardInterrupt:
            logger.info("\n\n🛑 Shutdown requested by user")
            return 0
        except Exception as e:
            logger.error(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        return 0

    def _is_black_frame(self, image) -> bool:
        """Check if frame is a black screen (transition)."""
        try:
            import numpy as np

            # Convert PIL Image to numpy array if needed
            if hasattr(image, 'save'):  # PIL Image
                frame_array = np.array(image)
            else:
                frame_array = image

            # Calculate mean brightness
            mean_brightness = frame_array.mean()

            # If mean brightness is very low, it's likely a black frame
            threshold = 10  # Very dark threshold
            is_black = mean_brightness < threshold

            if is_black:
                logger.debug(f"Black frame detected: mean brightness = {mean_brightness:.2f} < {threshold}")

            return is_black
        except Exception as e:
            logger.warning(f"Error checking for black frame: {e}")
            return False  # If we can't check, assume it's not black

    def _wait_for_actions_complete(self, timeout: int = 30) -> None:
        """Wait for all queued actions to complete before proceeding (same as my_cli_agent)."""
        import requests
        import time

        logger.info("⏳ Waiting for actions to complete...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/queue_status", timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    if status.get("queue_empty", False):
                        logger.info("✅ All actions completed")
                        return
                    else:
                        queue_len = status.get("queue_length", 0)
                        logger.debug(f"   Queue: {queue_len} actions remaining...")
                        time.sleep(0.5)  # Poll every 500ms
                else:
                    logger.warning(f"Failed to get queue status: {response.status_code}")
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error checking queue status: {e}")
                time.sleep(0.5)
        time.sleep(1)

        logger.warning(f"⚠️ Timeout waiting for actions to complete after {timeout}s")

    def _decide_action(self, state: Dict[str, Any], screenshot: Any) -> str:
        """Decide next action using function calling (MCP + native tools)."""
        # Build prompt with goals and state
        prompt = self._build_decision_prompt(state)
        start_time = time.time()
        try:
            # Use VLM with function calling (tools already configured in VLM constructor)
            if screenshot:
                response = self.vlm_client.get_query(screenshot, prompt, "gemini_plays")
            else:
                response = self.vlm_client.get_text_query(prompt, "gemini_plays")

            # Extract thinking and action for logging
            thinking_text = ""
            action_text = ""

            # Process function calls if present
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    content = candidate.content
                    if hasattr(content, 'parts'):
                        for part in content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                function_call = part.function_call
                                function_name = function_call.name

                                # Extract thinking/reasoning from function args
                                args = self._convert_protobuf_args(function_call.args)
                                thinking = args.get('reasoning') or args.get('reason') or ""
                                thinking_text = thinking

                                # Format action details
                                if function_name == "press_buttons":
                                    buttons = args.get('buttons', [])
                                    action_str = f"press_buttons({buttons})"
                                elif function_name == "navigate_to":
                                    x = args.get('x')
                                    y = args.get('y')
                                    variance = args.get('variance', 'none')
                                    action_str = f"navigate_to(x={x}, y={y}, variance={variance})"
                                else:
                                    action_str = f"{function_name}(...)"

                                action_text = action_str

                                # Display thinking and action
                                if self.verbose:
                                    print(f"\n{'='*70}")
                                    print(f"🧠 THINKING:")
                                    print(f"   {thinking}")
                                    print(f"\n🎮 ACTION:")
                                    print(f"   {action_str}")
                                    print(f"{'='*70}\n")

                                # Log interaction with clean format
                                log_response = f"THINKING: {thinking}\nACTION: {action_str}"
                                token_usage = {}
                                if hasattr(response, 'usage_metadata'):
                                    usage = response.usage_metadata
                                    token_usage = {
                                        "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                                        "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                                        "total_tokens": getattr(usage, 'total_token_count', 0)
                                    }
                                duration = time.time() - start_time
                                # Log the interaction
                                self.llm_logger.log_interaction(
                                    interaction_type=f"gemini_plays",
                                    prompt=log_response,
                                    response=result,
                                    duration=duration,
                                    metadata={"model": self.model_name, "backend": "gemini", "has_image": True, "token_usage": token_usage},
                                    model_info={"model": self.model_name, "backend": "gemini"}
                                )

                                # Execute the tool (MCP or native)
                                result = self._execute_tool(function_call)

                                # For press_buttons and navigate_to, return special marker
                                # The actual execution happens via MCP server
                                if function_name in ["press_buttons", "navigate_to"]:
                                    return "TOOL_EXECUTED"

                                # For other tools, continue processing
                                # (goals, notepad, etc. are informational)

            # If no function call, return WAIT
            return "WAIT"

        except Exception as e:
            logger.error(f"Error in tool-based decision making: {e}")
            return "WAIT"

    def _build_decision_prompt(self, state: Dict[str, Any]) -> str:
        """Build decision prompt with comprehensive tool descriptions."""
        goals_text = f"""Current Goals:
- Primary: {self.primary_goal.description if self.primary_goal else 'None'}
- Secondary: {self.secondary_goal.description if self.secondary_goal else 'None'}
- Tertiary: {self.tertiary_goal.description if self.tertiary_goal else 'None'}"""

        exploration_text = f"""Exploration: {self.map_memory.get_exploration_percentage(1000):.1f}% of estimated map
Points of Interest: {len(self.map_memory.points_of_interest)} marked"""

        context_text = ""
        if self.context.summary:
            context_text = f"Context: {self.context.summary}"

        notepad_text = ""
        if self.context.notepad:
            notepad_text = f"\nNotepad entries: {len(self.context.notepad)}"

        # Use state_text from get_game_state (includes porymap ground truth!)
        # Fall back to format_state_for_llm if state_text not available
        state_text = state.get('state_text', '')
        if not state_text:
            state_text = format_state_for_llm(state)
        
        return f"""You are playing Pokemon Emerald. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Emerald and eventually defeat the Elite Four. Make decisions based on what you see on the screen.

Before each action, explain your reasoning briefly, then use the tools to execute your chosen commands.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay.

Be strategic in your decisions - consider type advantages in battles, manage your Pokemon's health, and explore thoroughly to find items and trainers.


{goals_text}

{exploration_text}
{context_text}{notepad_text}

Current State (includes porymap ground truth):
{state_text}

Recent actions: {', '.join(list(self.recent_actions)[-5:])}

AVAILABLE TOOLS (use function calling):

🎮 GAME CONTROL:
- press_buttons(buttons, reasoning) - Press GBA buttons: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT, L, R, WAIT
- navigate_to(x, y, variance, reason) - A* pathfind with porymap ground truth
  variance: 'none', 'low', 'medium', 'high', 'extreme'

🎯 GOAL MANAGEMENT:
- update_primary_goal(description, completion_conditions) - Set main objective
- update_secondary_goal(description, completion_conditions) - Set enabling goal
- update_tertiary_goal(description, completion_conditions) - Set exploratory goal
- mark_goal_complete(goal_type, reasoning) - Mark goal as done
- get_current_goals() - View all goals and status

📝 MEMORY/KNOWLEDGE:
- add_to_notepad(note) - Store important information
- read_notepad(last_n_entries) - Recall stored notes
- mark_location_important(x, y, description) - Mark point of interest
- get_exploration_status() - Get exploration statistics

🧠 SELF-CRITIQUE:
- analyze_recent_actions() - Analyze if stuck or making poor choices
- get_stuck_analysis() - Check stuck status and get suggestions

🔧 META:
- define_custom_agent(name, task_description) - Create mini-agent

STRATEGY:
1. Use navigate_to for distant movement (ground-truth pathfinding!)
2. Use press_buttons for interactions and manual movement
3. Update goals as you progress through the game
4. Use notepad to remember important NPCs, locations, items
5. Mark important locations (Pokemon Centers, Gyms, etc.)
6. If stuck, use analyze_recent_actions or get_stuck_analysis
7. Complete goals when done, then set new ones
8. Check the image for dialog

IMPORTANT: Please think step by step before choosing your action. Structure your response like this:

ANALYSIS:
[Analyze what you see in the frame and current game state - what's happening? where are you? is there dialog that you need to interact with? what should you be doing? 
IMPORTANT: Look carefully at the game image for objects (clocks, pokeballs, bags) and NPCs (people, trainers) that might not be shown on the map. NPCs appear as sprite characters and can block movement or trigger battles/dialogue. When you see them try determine their location (X,Y) on the map relative to the player and any objects.]

OBJECTIVES:
[Review your current goals to follow the storyline of Pokemon Emerald]

PLAN:
[Think about your immediate goal - what do you want to accomplish in the next few actions? Consider your current objectives and recent history. 
Check MOVEMENT MEMORY for areas you've had trouble with before and plan your route accordingly.]

REASONING:
[Explain why you're choosing this specific action. Reference the MOVEMENT PREVIEW and MOVEMENT MEMORY sections. Check the visual frame for NPCs before moving. If you see NPCs in the image, avoid walking into them. Consider any failed movements or known obstacles from your memory.]

Action:
[Action deciding what tool and arguments]
"""

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
        critique_text = self._extract_text_from_response(critique)

        if self.verbose:
            print(f"   Self-critique: {critique_text[:150]}...")

        # Extract suggested actions
        self._extract_recovery_actions(critique_text)
    
    def _extract_recovery_actions(self, critique: str):
        """Extract recovery actions from self-critique using LLM."""
        prompt = f"""Based on this self-critique analysis:
{critique}

Provide a sequence of 3-5 button presses to help unstick the agent.
Reply with just the buttons separated by commas (e.g., "UP, A, LEFT, DOWN, A").
Valid buttons: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT"""
        
        response = self.vlm_client.get_text_query(prompt, "gemini_plays")
        response_text = self._extract_text_from_response(response)

        # Parse button sequence
        buttons_found = False
        for part in response_text.upper().split(','):
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
        response_text = self._extract_text_from_response(response)

        for word in response_text.upper().split():
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
        # Generate initial goals on first step
        if self.step_count == 1 and not self.primary_goal:
            self._generate_new_goals(state)
            return

        # Only check goals every 20 steps to avoid blocking
        if self.step_count % 20 != 0:
            return

        # Check if goals need updating
        should_update = False

        # Check primary goal completion (only if exists and not recently created)
        if self.primary_goal:
            if self.step_count - self.primary_goal.created_at > 20:
                if self._check_goal_completion(self.primary_goal, state):
                    self.primary_goal.completed = True
                    should_update = True
        else:
            should_update = True

        # Check secondary goal completion (only if exists and not recently created)
        if self.secondary_goal:
            if self.step_count - self.secondary_goal.created_at > 20:
                if self._check_goal_completion(self.secondary_goal, state):
                    self.secondary_goal.completed = True
                    should_update = True
        elif self.primary_goal:
            should_update = True

        # Check tertiary goal completion (only if exists and not recently created)
        if self.tertiary_goal:
            if self.step_count - self.tertiary_goal.created_at > 20:
                if self._check_goal_completion(self.tertiary_goal, state):
                    self.tertiary_goal.completed = True
                    should_update = True
        else:
            should_update = True

        # Update goals if needed (but not on every step)
        if should_update and self.step_count % 50 == 0:
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

        # Extract text from response (handles both string and GenerateContentResponse)
        response_text = self._extract_text_from_response(response)

        # Parse goals from response
        try:
            # Extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                goals_data = json.loads(response_text[start:end])
                
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
        response_text = self._extract_text_from_response(response)

        return "YES" in response_text.upper()
    
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
    """Factory function to create GeminiPlaysAgent with MCP tool support.

    Args:
        vlm_client: Optional VLM client
        server_url: MCP server URL (default: "http://localhost:8000")
        context_reset_interval: Steps before context reset (default: 100)
        enable_self_critique: Enable self-critique mechanism (default: True)
        enable_exploration: Enable forced exploration (default: True)
        enable_meta_tools: Enable meta-tools (default: False)
        use_mcp_tools: Use MCP tools for actions (default: True)
        verbose: Detailed logging (default: True)

    Returns:
        GeminiPlaysAgent instance
    """
    return GeminiPlaysAgent(**kwargs)
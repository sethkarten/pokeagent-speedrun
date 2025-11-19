#!/usr/bin/env python3
"""
Autonomous CLI Agent for Pokemon Emerald
This version creates its own objectives and has access to all available tools.
Uses Gemini API (or VertexAI) directly with MCP tools exposed as function declarations.
Maintains conversation history with automatic compaction over time.
"""

import os
import sys
import time
import json
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from utils.agent_helpers import update_server_metrics
from utils.llm_logger import get_llm_logger
from utils.vlm import VLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
                # Pokemon MCP tools
                "get_game_state": "/mcp/get_game_state",
                "press_buttons": "/mcp/press_buttons",
                "navigate_to": "/mcp/navigate_to",
                "add_knowledge": "/mcp/add_knowledge",
                "search_knowledge": "/mcp/search_knowledge",
                "get_knowledge_summary": "/mcp/get_knowledge_summary",
                "lookup_pokemon_info": "/mcp/lookup_pokemon_info",
                "list_wiki_sources": "/mcp/list_wiki_sources",
                "get_walkthrough": "/mcp/get_walkthrough",
                
                "complete_direct_objective": "/mcp/complete_direct_objective",
                "create_direct_objectives": "/mcp/create_direct_objectives",
                "get_progress_summary": "/mcp/get_progress_summary",
                "reflect": "/mcp/reflect",

                # Baseline MCP tools (file/shell/web)
                "read_file": "/mcp/read_file",
                "write_file": "/mcp/write_file",
                "list_directory": "/mcp/list_directory",
                "glob": "/mcp/glob",
                "search_file_content": "/mcp/search_file_content",
                "replace": "/mcp/replace",
                "read_many_files": "/mcp/read_many_files",
                "run_shell_command": "/mcp/run_shell_command",
                "web_fetch": "/mcp/web_fetch",
                "google_web_search": "/mcp/google_web_search",
                "save_memory": "/mcp/save_memory",
            }

            endpoint = endpoint_map.get(tool_name)
            if not endpoint:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

            url = f"{self.server_url}{endpoint}"
            logger.info(f"🔧 Calling MCP tool: {tool_name}")
            logger.debug(f"   URL: {url}")

            # Log arguments, but exclude large base64 data
            args_for_log = {k: f"<{len(v)} bytes>" if k == "screenshot_base64" and isinstance(v, str) and len(v) > 100 else v
                           for k, v in arguments.items()}
            logger.info(f"   Args: {args_for_log}")

            response = requests.post(url, json=arguments, timeout=30)
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ Tool {tool_name} completed")

            # Special handling for get_game_state - print the actual formatted text
            if tool_name == "get_game_state" and result.get("success") and "state_text" in result:
                logger.info("   Game State:")
                print("\n" + "="*70)
                print(result["state_text"])
                print("="*70 + "\n")
                if "screenshot_base64" in result:
                    logger.info(f"   Screenshot: <{len(result['screenshot_base64'])} bytes>")
            else:
                # Log result, but exclude large base64 data
                result_for_log = {k: f"<{len(v)} bytes>" if k == "screenshot_base64" and isinstance(v, str) and len(v) > 100 else v
                                 for k, v in result.items()}
                logger.info(f"   Result: {result_for_log}")
            return result

        except Exception as e:
            logger.error(f"❌ Tool {tool_name} failed: {e}")
            return {"success": False, "error": str(e)}


class AutonomousCLIAgent:
    """Autonomous CLI Agent using Gemini API directly with ALL MCP tools available."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model: str = "gemini-2.5-flash",
        backend: str = "gemini",
        max_steps: Optional[int] = None,
        system_instructions_file: str = "POKEAGENT.md",
        max_context_chars: int = 100000,
        target_context_chars: int = 50000
    ):
        print(f"🚀 Initializing AutonomousCLIAgent with backend={backend}, model={model}, server={server_url}")
        self.server_url = server_url
        self.model = model
        self.backend = backend
        self.max_steps = max_steps
        self.step_count = 0
        self.max_context_chars = max_context_chars
        self.target_context_chars = target_context_chars

        # Conversation history for tracking and compaction
        self.conversation_history = []

        # Recent function call results to add to next step's context
        # Format: [(function_name, result_json_string, timestamp), ...]
        self.recent_function_results = []

        # Load system instructions
        self.system_instructions = self._load_system_instructions(system_instructions_file)

        # Initialize MCP tool adapter
        self.mcp_adapter = MCPToolAdapter(server_url)

        # Initialize VLM for ALL backends (unified interface)
        # Create tool declarations for function calling
        self.tools = self._create_tool_declarations()
        self.vlm = VLM(
            backend=self.backend,
            model_name=self.model,
            tools=self.tools,
            system_instruction=self.system_instructions
        )
        print(f"✅ VLM initialized with {self.backend} backend using model: {self.model}")
        print(f"✅ {len(self.tools)} tools available")
        print(f"✅ System instructions loaded ({len(self.system_instructions)} chars)")

        # Initialize LLM logger
        from utils.llm_logger import get_llm_logger
        self.llm_logger = get_llm_logger()

    def _load_system_instructions(self, filename: str) -> str:
        """Load system instructions from file."""
        filepath = Path(__file__).parent.parent / filename
        if not filepath.exists():
            logger.warning(f"System instructions file not found: {filepath}")
            return "You are an AI agent playing Pokemon Emerald. Use the available tools to progress through the game."

        with open(filepath, 'r') as f:
            content = f.read()

        logger.info(f"✅ Loaded system instructions from {filename} ({len(content)} chars)")
        return content

    def _create_tool_declarations(self):
        """Create Gemini function declarations for ALL MCP tools (Pokemon + Baseline) - ALL ENABLED."""

        # Use Gemini's declaration format with proper types
        import google.generativeai.types as genai_types

        tools = [
            # ============================================================
            # POKEMON MCP TOOLS
            # ============================================================

            # Game Control Tools
            {
                "name": "get_game_state",
                "description": "Get the current game state including player position, party Pokemon, map, items, and a screenshot. Use this to understand where you are and what you can do.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "press_buttons",
                "description": "Press Game Boy Advance buttons to interact with the game. Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R, WAIT. Use WAIT to observe without pressing any button.",
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
                            "description": "REQUIRED FORMAT: Must include 'ANALYZE: [game screen, location, objective, situation]' and 'PLAN: [action, reason, expected result]'. Example: 'ANALYZE: Dialogue box visible. Location: Route 101 (5,8). Objective: Talk to Prof Birch. Situation: Birch asking for help. PLAN: Action: Press A. Reason: Advance dialogue. Expected: Dialogue continues or battle starts.'"
                        }
                    },
                    "required": ["buttons", "reasoning"]
                }
            },
            {
                "name": "navigate_to",
                "description": "Automatically navigate to specific coordinates using A* pathfinding. IMPORTANT: Always specify the variance parameter. If you get blocked repeatedly at the same position, increase variance to 'medium', 'high', or 'extreme' to explore alternative paths.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "x": {"type_": "INTEGER", "description": "Target X coordinate"},
                        "y": {"type_": "INTEGER", "description": "Target Y coordinate"},
                        "variance": {
                            "type_": "STRING",
                            "description": "REQUIRED. Path variance level: 'none' (optimal path, use first), 'low' (1-step variation), 'medium' (3-step variation, use if blocked), 'high' (5-step variation, use if very stuck), 'extreme' (8-step variation, use as last resort). Default: 'none'",
                            "enum": ["none", "low", "medium", "high", "extreme"]
                        },
                        "reason": {
                            "type_": "STRING",
                            "description": "REQUIRED FORMAT: Must include 'ANALYZE: [current location, objective, destination details]' and 'PLAN: [why navigating here, what to do when arrive]'. Example: 'ANALYZE: Currently at Littleroot (8,10). Objective: Meet Prof Birch. Destination: Route 101 entrance at (15,5). PLAN: Navigate to encounter Birch being attacked. Will save him to progress story.'"
                        }
                    },
                    "required": ["x", "y", "variance", "reason"]
                }
            },
            {
                "name": "complete_direct_objective",
                "description": "Complete the current direct objective and advance to the next one. Use this when you have successfully completed the current objective's task.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "reasoning": {
                            "type_": "STRING",
                            "description": "REQUIRED FORMAT: Must include 'ANALYZE: [current state, objective requirements, completion evidence]' and 'PLAN: [confirm completion, next objective]'. Example: 'ANALYZE: Objective was to reach Route 101. Current location shows Route 101 at (15,5). Evidence: Game text shows \"Route 101\". PLAN: Objective complete, marking as done. Next: Talk to Prof Birch.'"
                        }
                    },
                    "required": ["reasoning"]
                }
            },
            {
                "name": "reflect",
                "description": "Use this when you feel stuck, uncertain, or suspect your current approach/objectives are wrong. This tool helps you step back, analyze what's happening, and realign your strategy. Call this if: (1) repeating same actions without progress, (2) objectives don't match game state, (3) stuck for multiple steps, (4) unsure what to do next.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "situation": {
                            "type_": "STRING",
                            "description": "Describe what you've been trying to do and why you think something might be wrong. Include: recent actions, lack of progress, confusion about objectives, or any observations that seem off."
                        }
                    },
                    "required": ["situation"]
                }
            },

            # Knowledge Base Tools - NOW ENABLED
            {
                "name": "add_knowledge",
                "description": "Store important discoveries in your knowledge base.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "category": {
                            "type_": "STRING",
                            "enum": ["location", "npc", "item", "pokemon", "strategy", "custom"],
                            "description": "Category of knowledge"
                        },
                        "title": {"type_": "STRING", "description": "Short title"},
                        "content": {"type_": "STRING", "description": "Detailed content"},
                        "location": {"type_": "STRING", "description": "Map name (optional)"},
                        "coordinates": {"type_": "STRING", "description": "Coordinates as 'x,y' (optional)"},
                        "importance": {
                            "type_": "INTEGER",
                            "description": "Importance 1-5",
                        }
                    },
                    "required": ["category", "title", "content", "importance"]
                }
            },
            {
                "name": "search_knowledge",
                "description": "Search your knowledge base for stored information.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "category": {"type_": "STRING", "description": "Category (optional)"},
                        "query": {"type_": "STRING", "description": "Text to search (optional)"},
                        "location": {"type_": "STRING", "description": "Map name filter (optional)"},
                        "min_importance": {"type_": "INTEGER", "description": "Min importance 1-5 (optional)"}
                    },
                    "required": []
                }
            },
            {
                "name": "get_knowledge_summary",
                "description": "Get a summary of the most important things you've learned.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "min_importance": {"type_": "INTEGER", "description": "Min importance (default 3)"}
                    },
                    "required": []
                }
            },

            # Wiki Tools - NOW ENABLED
            {
                "name": "lookup_pokemon_info",
                "description": "Look up Pokemon, moves, locations, items, NPCs from wikis (Bulbapedia, Serebii, PokemonDB, Marriland).",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "topic": {"type_": "STRING", "description": "What to search (e.g., 'Mudkip', 'Route_101')"},
                        "source": {
                            "type_": "STRING",
                            "enum": ["bulbapedia", "serebii", "pokemondb", "marriland"],
                            "description": "Wiki source (default: bulbapedia)"
                        }
                    },
                    "required": ["topic"]
                }
            },
            {
                "name": "list_wiki_sources",
                "description": "List available Pokemon wiki sources.",
                "parameters": {"type_": "OBJECT", "properties": {}, "required": []}
            },
            {
                "name": "get_walkthrough",
                "description": "Get official Emerald walkthrough (Parts 1-21). CRITICAL: Match your knowledge base to milestones. Part 1: Starter+Norman. Part 2: Roxanne (1st gym). Part 3: Brawly (2nd gym). Part 4: Slateport/Team Aqua. Part 5: Wattson (3rd gym). Part 6: Fallarbor. Part 7: Flannery (4th gym). ALWAYS call get_knowledge_summary() FIRST, match to milestones, use HIGHEST completed + 1. Example: Knowledge shows 'Defeated Roxanne, Stone Badge' → Past Part 2, use Part 3.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "part": {
                            "type_": "INTEGER",
                            "description": "Walkthrough part 1-21. Determine by: 1) Review knowledge_summary 2) Match to milestones 3) Use highest completed + 1. VERIFY the walkthrough against knowledge before creating objectives.",
                        }
                    },
                    "required": ["part"]
                }
            },
            {
                "name": "create_direct_objectives",
                "description": "Create the next 3 direct objectives when you need new goals. Use this after consulting get_walkthrough() or wiki sources to plan your next steps. Provide exactly 3 objectives with id, description, action_type, target_location, navigation_hint, and completion_condition.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "objectives": {
                            "type_": "ARRAY",
                            "items": {
                                "type_": "OBJECT",
                                "properties": {
                                    "id": {"type_": "STRING", "description": "Unique identifier (e.g., 'dynamic_01_navigate_route')"},
                                    "description": {"type_": "STRING", "description": "Clear description of what to accomplish"},
                                    "action_type": {
                                        "type_": "STRING",
                                        "enum": ["navigate", "interact", "battle", "wait"],
                                        "description": "Type of action"
                                    },
                                    "target_location": {"type_": "STRING", "description": "Target location/map name"},
                                    "navigation_hint": {"type_": "STRING", "description": "Specific guidance on how to accomplish this"},
                                    "completion_condition": {"type_": "STRING", "description": "How to verify completion (e.g., 'location_contains_route_102')"}
                                },
                                "required": ["id", "description", "action_type"]
                            },
                            "description": "Array of exactly 3 objectives to create next"
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Explanation of why these objectives were chosen (referencing walkthrough/wiki sources)"
                        }
                    },
                    "required": ["objectives", "reasoning"]
                }
            },
            {
                "name": "get_progress_summary",
                "description": "Get comprehensive progress summary including completed milestones, objectives, current location, and knowledge base summary. Use this to understand what you've accomplished.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {},
                    "required": []
                }
            },

            # ============================================================
            # BASELINE MCP TOOLS (File/Shell/Web) - NOW ALL ENABLED
            # ============================================================

            {
                "name": "read_file",
                "description": "Read file contents. Supports text, images, PDFs. Returns content or base64 for binary files.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "file_path": {"type_": "STRING", "description": "Absolute path to file"}
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write file to .pokeagent_cache/cli/ directory or current run directory. If using relative path, writes to run directory (timestamped). Creates directories if needed.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "file_path": {"type_": "STRING", "description": "Path within .pokeagent_cache/cli/ (absolute) or relative path for run directory"},
                        "content": {"type_": "STRING", "description": "File content"}
                    },
                    "required": ["file_path", "content"]
                }
            },
            {
                "name": "list_directory",
                "description": "List files and directories at path.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "path": {"type_": "STRING", "description": "Directory path"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "glob",
                "description": "Find files matching glob pattern (e.g., '**/*.py', 'src/*.md').",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "pattern": {"type_": "STRING", "description": "Glob pattern"},
                        "path": {"type_": "STRING", "description": "Starting directory (optional)"}
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "search_file_content",
                "description": "Search files for regex pattern. Returns matching lines.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "pattern": {"type_": "STRING", "description": "Regex pattern"},
                        "path": {"type_": "STRING", "description": "Directory to search"},
                        "file_pattern": {"type_": "STRING", "description": "File glob (e.g., '*.py')"}
                    },
                    "required": ["pattern", "path"]
                }
            },
            {
                "name": "run_shell_command",
                "description": "Run shell command (42 safe commands allowed: ls, cat, grep, python, npm, etc. NO git, rm, sudo).",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "command": {"type_": "STRING", "description": "Shell command"},
                        "description": {"type_": "STRING", "description": "What this command does"}
                    },
                    "required": ["command", "description"]
                }
            },
            {
                "name": "web_fetch",
                "description": "Fetch and parse web pages (up to 20 URLs). Extracts text content.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "prompt": {"type_": "STRING", "description": "Prompt with URLs and instructions"}
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "google_web_search",
                "description": "Search web using DuckDuckGo (privacy-friendly, no API key). Returns 10 results.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "query": {"type_": "STRING", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "save_memory",
                "description": "Save facts to remember across sessions (stored in .pokeagent_cache/cli/AGENT.md).",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "fact": {"type_": "STRING", "description": "Clear, self-contained fact"}
                    },
                    "required": ["fact"]
                }
            },
        ]

        logger.info(f"✅ Created {len(tools)} tool declarations (ALL TOOLS ENABLED)")
        return tools

    def _execute_function_call_by_name(self, function_name: str, arguments: dict) -> str:
        """Execute a function by name with given arguments and return result as JSON string."""

        # Special handling for reflect tool - use agent's own VLM for analysis
        if function_name == "reflect":
            return self._execute_reflect(arguments)

        # Call the tool via MCP adapter
        result = self.mcp_adapter.call_tool(function_name, arguments)
        # Return as JSON string
        return json.dumps(result, indent=2)

    def _execute_reflect(self, arguments: dict) -> str:
        """Execute reflection using agent's own VLM to analyze situation."""
        try:
            # Get context from server
            context_result = self.mcp_adapter.call_tool("reflect", arguments)

            if not context_result.get("success"):
                return json.dumps({"success": False, "error": context_result.get("error", "Failed to get context")})

            context = context_result.get("context", {})

            # Build reflection prompt
            situation = context.get("situation", "")
            current_state = context.get("current_state", {})
            current_obj = context.get("current_objective", {})
            progress = context.get("progress", {})
            history = context.get("recent_history", [])

            # Format history for readability
            history_text = []
            for h in history:
                history_text.append(f"Step {h.get('step')}: [{h.get('action')}] {h.get('action_details')}")
                if h.get('coords'):
                    history_text.append(f"  Position: {h.get('coords')}")
                if h.get('thinking'):
                    history_text.append(f"  Thinking: {h.get('thinking')}")

            history_str = "\n".join(history_text)

            # Format objective
            obj_text = "None"
            if current_obj.get("objective"):
                obj = current_obj["objective"]
                if isinstance(obj, dict):
                    obj_text = obj.get("description", "Unknown")
                    if obj.get("navigation_hint"):
                        obj_text += f"\nHint: {obj['navigation_hint']}"
                else:
                    obj_text = str(obj)

            # Include porymap if available
            porymap_section = ""
            if current_state.get('porymap_ground_truth'):
                porymap_section = f"\n\nGROUND TRUTH MAP (PORYMAP):\n{current_state.get('porymap_ground_truth')}\n"

            # Fetch knowledge base summary for context
            knowledge_section = ""
            try:
                knowledge_result = self.mcp_adapter.call_tool("get_knowledge_summary", {"min_importance": 3})
                if knowledge_result.get("success") and knowledge_result.get("summary"):
                    knowledge_text = knowledge_result["summary"]
                    if knowledge_text and knowledge_text.strip():
                        knowledge_section = f"\n\nKNOWLEDGE BASE (GROUND TRUTH - what agent has actually accomplished):\n⚠️ IMPORTANT: The knowledge base is ALWAYS CORRECT. It represents actual accomplishments.\n   If objectives conflict with knowledge base, the OBJECTIVES are wrong, NOT the knowledge base.\n\n{knowledge_text}\n"
                        logger.info("📚 Loaded knowledge base for reflection")
                    else:
                        knowledge_section = "\n\nKNOWLEDGE BASE: No entries yet (agent hasn't stored any discoveries)\n"
                        logger.info("📚 Knowledge base is empty")
            except Exception as e:
                logger.warning(f"Could not load knowledge base for reflection: {e}")
                knowledge_section = "\n\nKNOWLEDGE BASE: Error loading knowledge base\n"

            reflection_prompt = f"""You are a strategic advisor analyzing an AI agent playing Pokemon Emerald. Provide direct, actionable guidance.

AGENT'S CONCERN:
{situation}

CURRENT GAME STATE:
Location: {current_state.get('location')}
Coordinates: ({current_state.get('coordinates', {}).get('x')}, {current_state.get('coordinates', {}).get('y')})
{current_state.get('state_text', '')}{porymap_section}{knowledge_section}

CURRENT OBJECTIVE:
Sequence: {current_obj.get('sequence')}
Objective: {obj_text}
Status: {'COMPLETE - needs new objectives' if current_obj.get('is_complete') else 'Active'}

PROGRESS:
- Milestones: {progress.get('milestones_completed', 0)}
- Objectives: {progress.get('objectives_completed', 0)}/{progress.get('total_objectives', 0)} in current sequence

RECENT ACTIONS (last 10 steps):
{history_str}

GROUND TRUTH SOURCES (trust these in priority order):
1. PORYMAP - Map layout, tile walkability (navigation ground truth)
2. KNOWLEDGE BASE - What agent has actually accomplished (never outdated, always correct)
3. WALKTHROUGH - Correct sequence of steps for the game (strategic ground truth)
4. Current objectives - May be WRONG if they conflict with above sources

OBJECTIVE MISMATCH
- if the agent is stuck it is likely that they pre-emptively completed an objective without actually doing the task!

ANALYZE (use ground truth sources to verify):
1. Is the agent stuck or repeating actions?
2. Does the objective match the game state?
3. Are target coordinates reachable based on porymap?
4. **CRITICAL**: Does the objective conflict with knowledge base? (If YES, objective is WRONG)
5. Is the agent trying to do something already accomplished (check knowledge base)?
6. Has the agent already learned information that makes the current objective obsolete?
7. Are there signs of confusion?
8. What should the agent do next?

PROVIDE (in this exact format):

**ASSESSMENT**:
[2-3 sentences analyzing what's happening. If objectives conflict with knowledge base, state that OBJECTIVES are wrong.]

**ISSUES**:
[List specific problems: stuck, wrong objective, unreachable coordinates, already completed per knowledge base, etc.]
⚠️ NEVER say "knowledge base is outdated" - knowledge base is ALWAYS correct. If there's conflict, the objectives are wrong.

**RECOMMENDATIONS**:
[Numbered list of specific actions to take - reference porymap AND knowledge base if relevant]
⚠️ IMPORTANT: If the agent is stuck/looping or the objective seems wrong:
   1. Check if knowledge base shows this task is already done - if YES, the objective may have been prematurely marked completed
   2. Recommend calling get_knowledge_summary() to review actual accomplishments
   3. DETERMINE THE CORRECT WALKTHROUGH PART by matching knowledge to milestones:
      → Part 1: Got starter Pokemon, met Norman at Petalburg
      → Part 2: Roxanne (Stone Badge - 1st gym)
      → Part 3: Brawly (Knuckle Badge - 2nd gym)
      → Part 4: Slateport Museum, Team Aqua
      → Part 5: Wattson (Dynamo Badge - 3rd gym)
      → Part 6: Routes 111-114, Fallarbor (NO gym)
      → Part 7: Flannery (Heat Badge - 4th gym)
      → Use HIGHEST milestone completed + 1
      → Example: Knowledge shows "Defeated Roxanne, Stone Badge" → Use Part 3
   4. Recommend calling get_walkthrough(part=X) with SPECIFIC part number from step 3
   5. Remind agent to VERIFY: Compare walkthrough to knowledge base before creating objectives
   6. If walkthrough describes tasks already in knowledge base → Recommend NEXT part number
   7. Suggest creating new objectives ONLY after finding correct walkthrough part

**SHOULD_REALIGN**: [YES or NO - whether to create new objectives]

Be direct and actionable. Trust the ground truth sources (porymap, walkthrough) over current objectives.
ALWAYS BE QUESTIONABLE OF RECENT OBJECTIVES. THEY ARE LIKELY TO HAVE NOT ACTUALLY BEEN COMPLETED.
NEVER dismiss knowledge base as "outdated" -- rather it may be prematurely marked completed. Trust the in-game features and NPC dialogue to learn what is happening.
If stuck or looping, ALWAYS recommend checking the walkthrough to verify objectives are correct."""

            logger.info("🤔 Agent performing self-reflection using VLM...")

            # Use agent's own VLM for reflection
            reflection_response = self.vlm.get_text_query(reflection_prompt, "Self_Reflection")

            logger.info(f"✅ Self-reflection complete ({len(reflection_response)} chars)")

            return json.dumps({
                "success": True,
                "reflection": reflection_response,
                "context_analyzed": {
                    "steps_reviewed": len(history),
                    "location": current_state.get('location'),
                    "objective_status": current_obj.get('status')
                }
            }, indent=2)

        except Exception as e:
            logger.error(f"Error in reflect execution: {e}")
            import traceback
            traceback.print_exc()
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _convert_protobuf_value(self, value):
        """Recursively convert a protobuf value to JSON-serializable Python types."""
        # Handle None
        if value is None:
            return None

        # Check if it's a protobuf type
        if hasattr(value, '__class__') and 'proto' in value.__class__.__module__:
            # First try to convert as dict (for MapComposite objects)
            # This must be checked BEFORE checking for __iter__ because MapComposite has both
            try:
                dict_value = dict(value)
                # Successfully converted to dict - recursively convert values
                return {k: self._convert_protobuf_value(v) for k, v in dict_value.items()}
            except (TypeError, ValueError):
                # Not a dict-like type, check if it's a list
                pass

            # Check if it's a list-like type (RepeatedComposite, RepeatedScalar)
            if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                # It's a list/array - recursively convert each item
                try:
                    return [self._convert_protobuf_value(item) for item in value]
                except:
                    return list(value)

            # Fallback: return as-is
            return value

        # Check if it's a regular dict (might contain nested protobuf values)
        elif isinstance(value, dict):
            return {k: self._convert_protobuf_value(v) for k, v in value.items()}
        # Check if it's a regular list (might contain nested protobuf values)
        elif isinstance(value, list):
            return [self._convert_protobuf_value(item) for item in value]
        # Otherwise return as-is (native Python type)
        else:
            return value

    def _convert_protobuf_args(self, proto_args) -> dict:
        """Convert protobuf arguments to JSON-serializable Python types."""
        arguments = {}
        for key, value in proto_args.items():
            arguments[key] = self._convert_protobuf_value(value)
        return arguments

    def _execute_function_call(self, function_call) -> str:
        """Execute a function call and return the result as JSON string."""
        function_name = function_call.name

        # Parse arguments - convert protobuf types to native Python types
        try:
            arguments = self._convert_protobuf_args(function_call.args)
        except Exception as e:
            logger.error(f"Failed to parse function arguments: {e}")
            return json.dumps({"success": False, "error": f"Invalid arguments: {e}"})

        # Special handling for reflect tool - use agent's own VLM
        if function_name == "reflect":
            return self._execute_reflect(arguments)

        # Call the tool via MCP adapter
        result = self.mcp_adapter.call_tool(function_name, arguments)

        # Return as JSON string
        return json.dumps(result, indent=2)

    def _add_to_history(self, prompt: str, response: str, tool_calls: List[Dict] = None, action_details: str = None, player_coords: tuple = None):
        """Add interaction to conversation history - ONLY stores LLM responses and actions."""
        # Strip whitespace from response to save tokens
        response_stripped = response.strip() if response else ""

        # CRITICAL SAFEGUARD: If response contains our prompt header, skip it entirely
        if "You are an autonomous AI agent" in response_stripped:
            logger.warning(f"⚠️ Skipping corrupted history entry at step {self.step_count} (contains prompt echo)")
            return

        entry = {
            "step": self.step_count,
            "llm_response": response_stripped,
            "timestamp": time.time()
        }

        logger.debug(f"📝 Storing history entry for step {self.step_count}: {response_stripped[:100]}...")

        # Extract action and action_details from tool_calls
        if tool_calls:
            last_call = tool_calls[-1]
            entry["action"] = last_call.get("name", "unknown")
            if action_details:
                entry["action_details"] = action_details
            elif last_call.get("name") == "navigate_to" and "x" in last_call.get("args", {}) and "y" in last_call.get("args", {}):
                variance = last_call['args'].get('variance', 'none')
                entry["action_details"] = f"navigate_to({last_call['args']['x']}, {last_call['args']['y']}, variance={variance})"
            elif last_call.get("name") == "press_buttons" and "buttons" in last_call.get("args", {}):
                entry["action_details"] = f"press_buttons({last_call['args']['buttons']})"
            else:
                entry["action_details"] = f"{last_call.get('name', 'unknown')}(...)"

        # Store player coordinates if available
        if player_coords:
            entry["player_coords"] = player_coords

        self.conversation_history.append(entry)

        # Keep only last 10 entries to prevent unbounded growth
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        logger.debug(f"✅ History now has {len(self.conversation_history)} entries")

    def _calculate_context_size(self) -> int:
        """Calculate total character count of conversation history."""
        total_chars = 0
        for entry in self.conversation_history:
            total_chars += len(entry.get("prompt", ""))
            total_chars += len(entry.get("response", ""))
            # Also count tool call strings
            for tool_call in entry.get("tool_calls", []):
                total_chars += len(str(tool_call))
        return total_chars

    def _format_history_for_display(self) -> str:
        """Format conversation history for display."""
        if not self.conversation_history:
            return "No conversation history yet."

        lines = [f"\n{'='*70}", "CONVERSATION HISTORY", '='*70]

        for entry in self.conversation_history[-10:]:
            try:
                lines.append(f"\nStep {entry['step']}:")
                lines.append(f"  Prompt: {entry['prompt'][:100]}...")
                if entry.get('tool_calls'):
                    lines.append(f"  Tools called: {', '.join(t['name'] for t in entry['tool_calls'])}")
                lines.append(f"  Response: {entry['response'][:100]}...")
            except:
                continue

        lines.append('='*70)
        return '\n'.join(lines)

    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        # Check if API key is set (only required for Gemini backend)
        if self.backend == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            logger.error("GEMINI_API_KEY environment variable not set")
            return False

        # Check if server is running
        logger.info(f"Checking if game server is ready at {self.server_url}...")
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.server_url}/status", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Game server is ready")
                    break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Server not ready yet (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Game server not running at {self.server_url}: {e}")
                    return False

        logger.info("✅ All prerequisites met")
        return True

    def _send_thinking_to_server(self, thinking_text: str, step_num: int):
        """Send agent thinking to server for display in stream."""
        try:
            if thinking_text:
                logger.debug(f"Sending thinking to server ({len(thinking_text)} chars)")
                response = requests.post(
                    f"{self.server_url}/agent_step",
                    json={
                        "metrics": {},
                        "thinking": thinking_text,
                        "step": step_num
                    },
                    timeout=1
                )
                logger.debug(f"Server response: {response.status_code}")
        except Exception as e:
            logger.debug(f"Could not send thinking to server: {e}")

    def run_step(self, prompt: str, max_tool_calls: int = 5, screenshot_b64: str = None) -> tuple[bool, str]:
        """Run a single agent step.

        Args:
            prompt: The prompt to send to model
            max_tool_calls: Maximum number of tool calls allowed per step (default: 5)
            screenshot_b64: Optional base64-encoded screenshot to include with prompt

        Returns:
            Tuple of (success: bool, response: str)
        """
        try:
            logger.info(f"📤 Sending prompt to {self.backend}...")
            logger.info(f"   Model: {self.model}")
            logger.info(f"   Prompt length: {len(prompt)} chars")
            if screenshot_b64:
                logger.info(f"   Including screenshot ({len(screenshot_b64)} bytes)")

            tool_calls_made = []
            reasoning_text = ""
            enforcement_retry_count = 0
            max_enforcement_retries = 3

            # Track duration
            start_time = time.time()
            vlm_call_start = time.time()
            try:
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

                if screenshot_b64:
                    import PIL.Image as PILImage
                    import io
                    import base64

                    # Decode base64 to image
                    image_data = base64.b64decode(screenshot_b64)
                    image = PILImage.open(io.BytesIO(image_data))

                    # Validate frame
                    if image is None:
                        logger.error("🚫 CRITICAL: run_step called with None image - cannot proceed")
                        return False, "ERROR: No valid image provided"

                    if not (hasattr(image, 'save') or hasattr(image, 'shape')):
                        logger.error(f"🚫 CRITICAL: run_step called with invalid image type {type(image)} - cannot proceed")
                        return False, "ERROR: Invalid image type"

                    if hasattr(image, 'size'):
                        width, height = image.size
                        if width <= 0 or height <= 0:
                            logger.error(f"🚫 CRITICAL: run_step called with invalid image size {width}x{height} - cannot proceed")
                            return False, "ERROR: Invalid image dimensions"

                    # Check for black frame
                    if self._is_black_frame(image):
                        logger.info("⏳ Black frame detected (likely a transition), waiting for next frame...")
                        return True, "WAIT"

                    def call_vlm_with_image():
                        return self.vlm.get_query(image, prompt, "Autonomous_CLI_Agent")

                    logger.info(f"📡 Calling VLM API with image (prompt: {len(prompt)} chars, image: {len(screenshot_b64)} bytes)")
                    logger.info(f"   ⏱️  Started at {time.strftime('%H:%M:%S')} - timeout set to 45s...")

                    max_retries = 3
                    retry_count = 0
                    response = None

                    while retry_count < max_retries:
                        executor = ThreadPoolExecutor(max_workers=1)
                        future = None
                        try:
                            future = executor.submit(call_vlm_with_image)
                            response = future.result(timeout=45)  # 45 second timeout for slower models
                            vlm_duration = time.time() - vlm_call_start
                            logger.info(f"   ✅ VLM call completed in {vlm_duration:.1f}s (attempt {retry_count + 1}/{max_retries})")
                            break
                        except FutureTimeoutError:
                            retry_count += 1
                            vlm_duration = time.time() - vlm_call_start
                            logger.error(f"   ⏱️ VLM call TIMED OUT after {vlm_duration:.1f}s (attempt {retry_count}/{max_retries})")
                            logger.error(f"   ⚠️  Abandoning timed-out thread and retrying immediately...")
                            if retry_count >= max_retries:
                                logger.error(f"   ❌ Max retries ({max_retries}) reached - giving up")
                                raise TimeoutError(f"VLM call timed out after {max_retries} attempts")
                        finally:
                            executor.shutdown(wait=False)
                else:
                    def call_vlm_with_text():
                        return self.vlm.get_text_query(prompt, "Autonomous_CLI_Agent")

                    logger.info(f"📡 Calling VLM API with text only (prompt: {len(prompt)} chars)")
                    logger.info(f"   ⏱️  Started at {time.strftime('%H:%M:%S')} - timeout set to 45s...")

                    max_retries = 3
                    retry_count = 0
                    response = None

                    while retry_count < max_retries:
                        executor = ThreadPoolExecutor(max_workers=1)
                        future = None
                        try:
                            future = executor.submit(call_vlm_with_text)
                            response = future.result(timeout=45)  # 45 second timeout for slower models
                            vlm_duration = time.time() - vlm_call_start
                            logger.info(f"   ✅ VLM call completed in {vlm_duration:.1f}s (attempt {retry_count + 1}/{max_retries})")
                            break
                        except FutureTimeoutError:
                            retry_count += 1
                            vlm_duration = time.time() - vlm_call_start
                            logger.error(f"   ⏱️ VLM call TIMED OUT after {vlm_duration:.1f}s (attempt {retry_count}/{max_retries})")
                            logger.error(f"   ⚠️  Abandoning timed-out thread and retrying immediately...")
                            if retry_count >= max_retries:
                                logger.error(f"   ❌ Max retries ({max_retries}) reached - giving up")
                                raise TimeoutError(f"VLM call timed out after {max_retries} attempts")
                        finally:
                            executor.shutdown(wait=False)

                is_function_calling = hasattr(response, 'candidates')

            except KeyboardInterrupt:
                vlm_duration = time.time() - vlm_call_start
                logger.warning(f"⚠️ VLM call interrupted by user after {vlm_duration:.1f}s")
                raise
            except TimeoutError as e:
                vlm_duration = time.time() - vlm_call_start
                logger.error(f"❌ VLM call TIMED OUT after {vlm_duration:.1f}s")
                return False, f"VLM API timeout after {vlm_duration:.1f}s: {str(e)}"
            except Exception as e:
                vlm_duration = time.time() - vlm_call_start
                error_type = type(e).__name__
                error_msg = str(e)
                logger.error(f"❌ VLM call failed after {vlm_duration:.1f}s")
                logger.error(f"   Error type: {error_type}")
                logger.error(f"   Error message: {error_msg[:500]}")
                import traceback
                traceback.print_exc()
                return False, f"VLM API error ({error_type}) after {vlm_duration:.1f}s: {error_msg[:200]}"

            # Process response - handle function calls
            tool_call_count = 0

            # Use unified VLM function calling for ALL backends
            function_calls_executed = self._handle_vlm_function_calls(
                response, tool_calls_made, tool_call_count, max_tool_calls
            )

            if function_calls_executed:
                duration = time.time() - start_time

                # Extract reasoning from the last tool call
                tool_reasoning = ""
                if tool_calls_made:
                    last_tool_call = tool_calls_made[-1]
                    try:
                        if last_tool_call['name'] == "navigate_to" and "reason" in last_tool_call["args"]:
                            tool_reasoning = last_tool_call["args"]["reason"]
                        elif last_tool_call['name'] == "press_buttons" and "reasoning" in last_tool_call["args"]:
                            tool_reasoning = last_tool_call["args"]["reasoning"]
                    except Exception as e:
                        logger.debug(f"Could not extract tool reasoning: {e}")

                full_response = self._extract_text_from_response(response)
                if tool_reasoning:
                    if full_response:
                        full_response += f"\n\n{tool_reasoning}"
                    else:
                        full_response = tool_reasoning

                if full_response:
                    full_response += f"\n\nAction executed: {last_tool_call['name']}"
                else:
                    full_response = f"Executed {last_tool_call['name']}"

                display_text = self._extract_text_from_response(response)
                if tool_reasoning:
                    if display_text:
                        display_text += f"\n\n{tool_reasoning}"
                    else:
                        display_text = tool_reasoning

                logger.info(f"✅ Step completed in {duration:.2f}s")
                logger.info(f"📝 Response: {display_text}")

                # Extract action details
                action_taken = last_tool_call['name']
                action_details = ""
                if last_tool_call['name'] == "press_buttons" and "buttons" in last_tool_call["args"]:
                    action_details = f"Pressed {last_tool_call['args']['buttons']}"
                elif last_tool_call['name'] == "navigate_to" and "x" in last_tool_call["args"] and "y" in last_tool_call["args"]:
                    target_x = last_tool_call["args"]["x"]
                    target_y = last_tool_call["args"]["y"]
                    self._wait_for_actions_complete()
                    final_pos = None
                    final_state_result = self._execute_function_call_by_name("get_game_state", {})
                    import json as json_module
                    final_state_data = json_module.loads(final_state_result)
                    if final_state_data.get("success"):
                        player_pos = final_state_data.get("player_position", {})
                        if player_pos:
                            final_pos = (player_pos.get("x"), player_pos.get("y"))

                    variance = last_tool_call.get('args', {}).get('variance', 'none')
                    if final_pos:
                        action_details = f"navigate_to({target_x}, {target_y}, variance={variance}) → Ended at ({final_pos[0]}, {final_pos[1]})"
                    else:
                        action_details = f"navigate_to({target_x}, {target_y}, variance={variance})"
                else:
                    action_details = f"Executed {last_tool_call['name']}"

                # Store function result for next step's context
                if tool_calls_made:
                    last_call = tool_calls_made[-1]
                    self._store_function_result_for_context(last_call['name'], last_call['result'])

                self._add_to_history(prompt, full_response, tool_calls_made, action_details=action_details)

                return True, full_response
            else:
                text_content = self._extract_text_from_response(response)
                if not text_content:
                    text_content = str(response)

                logger.info(f"📥 Received text response from {self.backend}:")
                logger.info(f"   {text_content}")

                duration = time.time() - start_time

                logger.info(f"✅ Step completed in {duration:.2f}s")

                self._add_to_history(prompt, text_content, tool_calls=[])

                return True, text_content

            # If we reach here, no response was generated
            logger.warning("⚠️ No response from model")
            return False, "No response"

        except Exception as e:
            logger.error(f"❌ Error in agent step: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _wait_for_actions_complete(self, timeout: int = 30) -> None:
        """Wait for all queued actions to complete before proceeding."""
        import requests

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
                        time.sleep(0.5)
                else:
                    logger.warning(f"Failed to get queue status: {response.status_code}")
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error checking queue status: {e}")
                time.sleep(0.5)
        time.sleep(1)

        logger.warning(f"⚠️ Timeout waiting for actions to complete after {timeout}s")

    def _log_thinking(self, prompt: str, response: str, duration: float = None, tool_calls: list = None) -> None:
        """Log interaction to LLM logger with full tool call history."""
        try:
            self.llm_logger.log_interaction(
                interaction_type="autonomous_gemini_cli",
                prompt=prompt,
                response=response,
                duration=duration,
                metadata={"tool_calls": tool_calls or []},
                model_info={"model": self.model}
            )
            logger.debug("✅ Logged to LLM logger")
        except Exception as e:
            logger.debug(f"Could not log to LLM logger: {e}")

    def _store_function_result_for_context(self, function_name: str, result_json: str):
        """Store function result to include in next step's context."""
        import time
        self.recent_function_results.append({
            "function_name": function_name,
            "result": result_json,
            "timestamp": time.time()
        })

        # Keep only last 3 function results to avoid context explosion
        if len(self.recent_function_results) > 3:
            self.recent_function_results = self.recent_function_results[-3:]

        logger.info(f"📝 Stored {function_name} result for next step's context")

    def _get_function_results_context(self) -> str:
        """Format recent function results for inclusion in prompt."""
        if not self.recent_function_results:
            return ""

        lines = ["\n" + "="*70, "📋 RESULTS FROM PREVIOUS STEP:", "="*70]

        for entry in self.recent_function_results:
            func_name = entry["function_name"]
            result = entry["result"]

            lines.append(f"\n🔧 Function: {func_name}")
            lines.append(f"Result:")

            # Truncate very long results
            if len(result) > 10000:
                lines.append(result[:10000] + "\n... (truncated)")
            else:
                lines.append(result)
            lines.append("")

        lines.append("="*70)

        # Clear the results after formatting (they've been used)
        self.recent_function_results = []

        return "\n".join(lines)

    def _build_structured_prompt(self, game_state_result: str, step_count: int) -> str:
        """Build an autonomous prompt that emphasizes creating your own objectives."""

        # Parse game state to extract relevant information
        import json as json_module
        try:
            game_state_data = json_module.loads(game_state_result)
        except:
            game_state_data = {}

        state_text = game_state_data.get("state_text", "")

        # Detect if in title sequence
        is_title_sequence = self._is_title_sequence(game_state_data)
        if is_title_sequence:
            logger.info("🎬 Title sequence detected - map information will be hidden")

        # Extract player coordinates for stuck detection
        player_position = game_state_data.get("player_position", {})
        current_coords = None
        if player_position and "x" in player_position and "y" in player_position:
            current_coords = (player_position["x"], player_position["y"])

        # Add stuck warning if detected
        if not is_title_sequence:
            stuck_warning = self._get_stuck_warning(current_coords)
            if stuck_warning:
                state_text = stuck_warning + state_text

        # Strip map information during title sequence
        if is_title_sequence:
            state_text = self._strip_map_info(state_text)

        direct_objective = game_state_data.get("direct_objective", "")
        direct_objective_status = game_state_data.get("direct_objective_status", "")
        direct_objective_context = game_state_data.get("direct_objective_context", "")

        # Format direct objective nicely if it's a dict
        if isinstance(direct_objective, dict):
            obj_id = direct_objective.get("id", "")
            desc = direct_objective.get("description", "")
            hint = direct_objective.get("navigation_hint", "")
            formatted_obj = f"🎯 CURRENT OBJECTIVE:\n  ID: {obj_id}\n  Description: {desc}"
            if hint:
                formatted_obj += f"\n  Hint: {hint}"
            direct_objective = formatted_obj

        # Format status nicely if it's a dict
        if isinstance(direct_objective_status, dict):
            seq = direct_objective_status.get("sequence_name", "")
            total = direct_objective_status.get("total_objectives", 0)
            current_idx = direct_objective_status.get("current_index", 0)
            completed = direct_objective_status.get("completed_count", 0)
            direct_objective_status = f"📊 PROGRESS: Objective {current_idx + 1}/{total} in sequence '{seq}' ({completed} completed)"

        # Build action history summary
        action_history = self._format_action_history()

        # Get function results from previous step
        function_results_context = self._get_function_results_context()

        # Log component sizes
        logger.info(f"📏 Pre-prompt component sizes:")
        logger.info(f"   state_text: {len(state_text):,} chars")
        logger.info(f"   action_history: {len(action_history):,} chars")
        logger.info(f"   function_results: {len(function_results_context):,} chars")
        logger.info(f"   direct_objective: {len(str(direct_objective)):,} chars")
        logger.info(f"   direct_objective_context: {len(direct_objective_context):,} chars")
        logger.info(f"   direct_objective_status: {len(direct_objective_status):,} chars")

        # Build autonomous prompt
        prompt = f"""You are an autonomous AI agent playing Pokémon Emerald on a Game Boy Advance emulator.

🧠 AUTONOMOUS MODE: You must create your own objectives and plan your progression through the game!

If you notice that you are repeating the same action sequences over and over again, you definitely need to try something different since what you are doing is wrong! Try exploring different new areas or interacting with different NPCs if you are stuck.

Some pointers to keep in mind (guard rails) as you problem solve:
1) You must think step-by-step when solving problems and making decisions.
2) Always provide detailed, context-aware responses that bias for ground-truth.
3) Consider the current situation in the game as well as what you've learned over time.
4) Do not fixate on the correctness of a particular solution, be flexible and adapt your strategy as needed.


ACTION HISTORY (last steps with thinking):
{action_history}
{function_results_context}

================================================================================
🎯🎯🎯 CURRENT DIRECT OBJECTIVE - READ THIS CAREFULLY 🎯🎯🎯
================================================================================

{direct_objective_context}

{direct_objective}

{direct_objective_status}

================================================================================
⚠️ CRITICAL: When you have completed the objective above:
1. FIRST: Call add_knowledge() to store what you learned (NPCs, items, locations, strategies)
   - Use importance=4 or 5 for critical information
   - Example: add_knowledge(category="npc", title="Gym Leader Norman", content="...", importance=5)
2. THEN: Call complete_direct_objective(reasoning="<explain why it's complete>")

This ensures your discoveries are remembered for future gameplay!

🔄 AUTONOMOUS OBJECTIVE CREATION:
When you see "All objectives completed!" or sequence_complete=True OR when you start fresh:

**STEP 1: DETERMINE YOUR PROGRESS**
1. Call get_knowledge_summary() to see what you've already accomplished
   → Review all entries to understand what you've done
   → Result appears in "RESULTS FROM PREVIOUS STEP" in next step

2. Call get_progress_summary() to see milestones, badges, and current location
   → Result appears in "RESULTS FROM PREVIOUS STEP" in next step

**STEP 2: FIGURE OUT WHICH WALKTHROUGH PART YOU'RE ON**
Analyze your accomplishments against these milestones (ACCURATE to Bulbapedia):
- Part 1: Got starter Pokemon, Routes 101-103, Oldale, Petalburg (met Norman)
- Part 2: Route 104, Petalburg Woods, Rustboro, **Roxanne (Stone Badge - 1st gym)**, Route 116
- Part 3: Dewford Town, **Brawly (Knuckle Badge - 2nd gym)**, Granite Cave, Slateport City
- Part 4: Slateport Museum, Team Aqua, Devon Goods to Captain Stern, Route 110
- Part 5: Mauville City, **Wattson (Dynamo Badge - 3rd gym)**, Route 117, Verdanturf
- Part 6: Routes 111-114, Fallarbor Town (NO gym)
- Part 7: Meteor Falls, Mt. Chimney, Lavaridge, **Flannery (Heat Badge - 4th gym)**
- Part 8+: Later gym leaders

**Use the HIGHEST milestone you've completed, then add 1 for next steps.**

Example: If knowledge shows "Defeated Roxanne, got Stone Badge" → You're past Part 2, need Part 3

**STEP 3: GET THE RIGHT WALKTHROUGH PART**
3. Call get_walkthrough(part=X) where X is determined from Step 2
   → Result appears in next step's context

**STEP 4: VERIFY IT'S THE RIGHT PART**
4. READ the walkthrough carefully
   - Compare it to your knowledge base
   - If walkthrough describes things you ALREADY did → INCREMENT part number and try again
   - If walkthrough describes things you HAVEN'T done yet → CORRECT, proceed

**STEP 5: CREATE OBJECTIVES**
5. Create the next 3 logical objectives using create_direct_objectives()
   → Base objectives on the walkthrough steps you haven't completed
   → Confirm success in next step

6. Once objectives are created, proceed with the first new objective

⭐ IMPORTANT: Function call results appear in the NEXT step!
   - Call ONE function per step (e.g., get_walkthrough)
   - The result will appear in "📋 RESULTS FROM PREVIOUS STEP" section above
   - Use that result to make your next decision
   - This applies to ALL functions - information gathering AND actions

Example format for create_direct_objectives:
create_direct_objectives(
    objectives=[
        {{
            "id": "dynamic_01_navigate_route_102",
            "description": "Travel to Route 102",
            "action_type": "navigate",
            "target_location": "Route 102",
            "navigation_hint": "Move east from Petalburg City to reach Route 102",
            "completion_condition": "location_contains_route_102"
        }},
        {{"id": "dynamic_02_...", "description": "...", ...}},
        {{"id": "dynamic_03_...", "description": "...", ...}}
    ],
    reasoning="Based on walkthrough Part 5, the next step is to travel to Route 102..."
)

🎯 YOUR MISSION: Progress through Pokemon Emerald by:
- Creating appropriate objectives based on the game walkthrough
- Exploring new areas and talking to NPCs
- Building a strong Pokemon team
- Defeating gym leaders and trainers
- Advancing the story
================================================================================

CURRENT GAME STATE:
{state_text}

**DIALOGUE CHECK**: Look at the game screen carefully - if you see a dialogue box with text, press_buttons(["A"], reasoning).

AVAILABLE TOOLS - Use these function calls to interact with the game:

🎮 **PRIMARY GAME TOOLS** :
- get_game_state() - Get current game state, player position, Pokemon, map, and screenshot
- complete_direct_objective(reasoning) - Mark current direct objective as complete. Provide strict justification before completing the objective.
- press_buttons(buttons, reasoning) - Press GBA buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R, WAIT
- navigate_to(x, y, variance, reason) - Automatically pathfind to coordinates using A* algorithm with porymap ground truth data.
- reflect(situation) - 🔄 SELF-CORRECTION TOOL: Use when stuck, repeating actions, or objectives seem wrong. Helps realign strategy and objectives.

🗺️ **NAVIGATION**: Use navigate_to(x, y, variance, reason) to automatically pathfind to a coordinate.

📚 **INFORMATION TOOLS** (use when you need info or planning objectives):
- lookup_pokemon_info(topic, source) - Look up Pokemon, moves, locations from wikis
- get_walkthrough(part) - Get official Emerald walkthrough (parts 1-21)
- search_knowledge(query, category) - Search your stored knowledge
- add_knowledge(category, title, content, importance) - Store important discoveries
- get_progress_summary() - Get comprehensive progress summary
- get_knowledge_summary(min_importance) - Get summary of important discoveries

💾 **KNOWLEDGE TOOLS**:
- save_memory(fact) - Save facts to remember across sessions

🎯 **OBJECTIVE MANAGEMENT** (use to create your own goals):
- create_direct_objectives(objectives, reasoning) - Create next 3 direct objectives dynamically
  Use this to plan your progression through the game autonomously!

📁 **FILE/WEB TOOLS** (use for research or saving notes):
- read_file(file_path) - Read file contents
- write_file(file_path, content) - Write file
- list_directory(path) - List files
- glob(pattern) - Find files by pattern
- search_file_content(pattern, path) - Search files
- run_shell_command(command) - Run shell commands
- web_fetch(prompt) - Fetch web pages
- google_web_search(query) - Search the web

** COORDINATE & MOVEMENT EXAMPLES **:
- Pressing LEFT decreases your X coordinate (moves you west)
- Pressing RIGHT increases your X coordinate (moves you east)
- Pressing UP decreases your Y coordinate (moves you north)
- Pressing DOWN increases your Y coordinate (moves you south)

** INTERACTION TIPS **:
- To interact with an object or NPC, you must be both 1) on an adjacent tile to the NPC or object and 2) facing the NPC or object.

STRATEGY - PRIORITY ORDER:
1. **CHECK OBJECTIVE COMPLETION FIRST**: Before doing ANYTHING, check if your current direct objective is complete. If yes:
   a. Call add_knowledge() to store what you learned
   b. THEN call complete_direct_objective(reasoning="...")
2. **CREATE OBJECTIVES IF NEEDED**: If you have no objectives or sequence is complete, use the tools above to research and create new objectives
3. **SELF-REFLECT WHEN STUCK/LOOPING**: If you notice you're repeating the same actions, not making progress, or objectives don't match reality:
   a. CALL reflect(situation="...") to analyze the situation
   b. If reflect suggests objectives are wrong, CALL get_walkthrough(part=X) to verify correct steps
   c. Create new objectives if current ones are misaligned
4. **DIALOGUE SECOND**: If you see a dialogue box on screen, ALWAYS use press_buttons(["A"], reasoning) to advance it
5. **MOVEMENT**: Preferentially use navigate_to(x, y, variance, reason) for pathfinding
6. **BATTLES**: Use press_buttons with battle moves carefully
7. **INFORMATION**: Use lookup_pokemon_info or get_walkthrough when you need to know something

🔄 **WHEN TO USE reflect()**:
- You've tried the same action 3+ times without progress
- Your coordinates haven't changed in multiple steps
- Current objective doesn't match what's actually happening in the game
- You're confused about what to do next
- Objectives seem misaligned with game state
- You feel like you're going in circles

⚠️ **CRITICAL - When stuck/looping:**
After calling reflect(), if it suggests objectives are wrong:
1. **First, call get_knowledge_summary()** to see what you've already accomplished
   → This helps determine which walkthrough part is appropriate
2. **Then call get_walkthrough(part=X)** to verify the correct next steps
   → Choose the part based on what you learned from knowledge base
3. Compare walkthrough instructions to your current objectives
4. If objectives are wrong, create new ones using create_direct_objectives()
5. The walkthrough is ground truth - trust it over your current plan

After calling reflect(), you'll receive guidance on whether to:
- Continue current approach
- **Recheck walkthrough to verify objectives** (do this when stuck!)
  → Remember to check knowledge base FIRST to pick the right walkthrough part
- Create new objectives with create_direct_objectives()
- Try a completely different strategy
- Gather more information with get_walkthrough() or get_progress_summary()

🔴 **REMEMBER**:
- You MUST create objectives yourself when needed!
- You MUST call complete_direct_objective() when objectives are done!
- You are autonomous - think, plan, and execute!

IMPORTANT: Always check the game screen for dialogue boxes before planning movement!

═══════════════════════════════════════════════════════════════════════
🧠 HOW TO STRUCTURE YOUR REASONING (MANDATORY FORMAT)
═══════════════════════════════════════════════════════════════════════

You MUST structure your reasoning parameter using this EXACT format:

**ANALYZE:**
- Game screen: [what you see]
- Location: [map name, coordinates X,Y]
- Current objective: [what you're trying to accomplish]
- Situation: [obstacles, NPCs, dialogue, items, etc.]
- IMPORTANT: IF THERE IS DIALOG, note down the dialog in your analysis text so you can refer to it later!!

**PLAN:**
- Action: [what you will do]
- Reason: [why this is the best choice]
- Expected result: [what will happen]

**EXAMPLE - Correct Format:**
press_buttons(["A"], reasoning="ANALYZE: Game screen shows dialogue box with Mom talking. Location: Player's House 2F at (7,6). Current objective: Talk to Mom and go downstairs. Situation: In dialogue with Mom about visiting Prof. Birch's lab. PLAN: Action: Press A to advance dialogue. Reason: Must complete dialogue before moving. Expected result: Dialogue advances or ends, allowing movement.")

**EXAMPLE - WRONG (too short):**
press_buttons(["A"], reasoning="Need to advance dialogue with Mom")  ❌ INCORRECT - Missing analysis!

🎯 **CRITICAL**: Every reasoning parameter must include BOTH "ANALYZE:" and "PLAN:" sections!

Step {step_count}"""

        # Log prompt size breakdown
        prompt_size = len(prompt)
        state_size = len(state_text)
        history_size = len(action_history)
        function_results_size = len(function_results_context)
        context_size = len(direct_objective_context)
        objective_size = len(str(direct_objective))
        status_size = len(direct_objective_status)

        dynamic_total = state_size + history_size + function_results_size + context_size + objective_size + status_size
        static_instructions = prompt_size - dynamic_total

        logger.info(f"📏 Final prompt size breakdown:")
        logger.info(f"   ═══════════════════════════════════════")
        logger.info(f"   TOTAL PROMPT: {prompt_size:,} chars (~{prompt_size//4:,} tokens)")
        logger.info(f"   ═══════════════════════════════════════")
        logger.info(f"   Dynamic content:")
        logger.info(f"     - State text: {state_size:,} chars")
        logger.info(f"     - Action history: {history_size:,} chars")
        logger.info(f"     - Function results: {function_results_size:,} chars")
        logger.info(f"     - Objective context: {context_size:,} chars")
        logger.info(f"     - Objective: {objective_size:,} chars")
        logger.info(f"     - Status: {status_size:,} chars")
        logger.info(f"     DYNAMIC TOTAL: {dynamic_total:,} chars")
        logger.info(f"   ───────────────────────────────────────")
        logger.info(f"   Static instructions: {static_instructions:,} chars")
        logger.info(f"   ═══════════════════════════════════════")

        return prompt


    def _is_title_sequence(self, game_state_data: Dict[str, Any]) -> bool:
        """Detect if in title sequence"""
        player_location = game_state_data.get("location", "")
        if player_location == "TITLE_SEQUENCE":
            return True

        player_name = game_state_data.get("player_name", "").strip()
        if not player_name or player_name == "????????":
            return True

        game_state_value = game_state_data.get("game_state", "").lower()
        if "title" in game_state_value or "intro" in game_state_value:
            return True

        return False

    def _strip_map_info(self, state_text: str) -> str:
        """Strip map/navigation information from state text during title sequence"""
        lines = state_text.split('\n')
        filtered_lines = []
        skip_section = False

        for line in lines:
            if any(marker in line for marker in [
                "🗺️ MAP:",
                "CURRENT MAP:",
                "PORYMAP ASCII:",
                "PORYMAP GROUND TRUTH MAP:",
                "🧭 MOVEMENT PREVIEW:",
                "MOVEMENT MEMORY:",
                "Player coordinates:",
                "Map dimensions:",
                "POSITION:",
                "LOCATION:"
            ]):
                skip_section = True
                continue

            if line.strip() == "" or (line.startswith("🎯") or line.startswith("📊") or line.startswith("⚠️")):
                skip_section = False

            if not skip_section:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _is_black_frame(self, image) -> bool:
        """Check if frame is a black screen (transition)"""
        try:
            if hasattr(image, 'save'):  # PIL Image
                frame_array = np.array(image)
            else:
                frame_array = image

            mean_brightness = frame_array.mean()
            threshold = 10
            is_black = mean_brightness < threshold

            if is_black:
                logger.debug(f"Black frame detected: mean brightness = {mean_brightness:.2f} < {threshold}")

            return is_black
        except Exception as e:
            logger.warning(f"Error checking for black frame: {e}")
            return False

    def _detect_stuck_pattern(self, current_coords: Optional[Tuple[int, int]]) -> bool:
        """Detect if agent is stuck (same position for multiple recent steps)"""
        if not current_coords or len(self.conversation_history) < 3:
            return False

        recent_positions = []
        for entry in self.conversation_history[-3:]:
            coords = entry.get("player_coords")
            if coords:
                recent_positions.append(coords)

        if len(recent_positions) >= 3:
            if all(pos == recent_positions[0] for pos in recent_positions):
                return True

        return False

    def _get_stuck_warning(self, coords: Optional[Tuple[int, int]]) -> str:
        """Generate warning text if stuck pattern detected"""
        if self._detect_stuck_pattern(coords):
            return "\n⚠️ WARNING: You appear to be stuck at this location. Try a different approach!\n" \
                   "💡 TIP: If you try an action like RIGHT but coordinates don't change from (X,Y) to (X+1,Y), there's likely an obstacle.\n"
        return ""

    def _format_action_history(self) -> str:
        """Format action history - shows only LLM thinking and actions taken."""
        if not self.conversation_history:
            logger.debug(f"📜 No conversation history to format")
            return "No previous actions recorded."

        recent_entries = self.conversation_history[-10:]
        logger.debug(f"📜 Formatting {len(recent_entries)} history entries")

        history_lines = []
        for entry in recent_entries:
            step = entry.get("step", "?")
            llm_response = entry.get("llm_response", "").strip()
            action_details = entry.get("action_details", "").strip()
            coords = entry.get("player_coords", None)

            coord_str = f"({coords[0]},{coords[1]})" if coords else "(?)"

            if llm_response or action_details:
                history_lines.append(f"[{step}] at {coord_str}:")
                if llm_response:
                    history_lines.append(f"  {llm_response}")
                if action_details:
                    history_lines.append(f"  → {action_details}")
                history_lines.append("")

        return "\n".join(history_lines).strip()

    def run(self) -> int:
        """Run the autonomous agent loop."""
        # Clear conversation history
        self.conversation_history = []
        logger.info("🧹 Cleared conversation history (fresh start)")

        logger.info("=" * 70)
        logger.info("🎮 Pokemon Emerald Autonomous CLI Agent")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model}")
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Server: {self.server_url}")
        if hasattr(self, 'tools') and self.tools:
            logger.info(f"Tools: {len(self.tools)} MCP tools (ALL ENABLED)")
        else:
            logger.info(f"Tools: MCP tools available via VLM backend ({self.backend})")
        logger.info(f"Context: Max {self.max_context_chars:,} chars (compact to {self.target_context_chars:,})")
        if self.max_steps:
            logger.info(f"Max Steps: {self.max_steps}")
        logger.info("=" * 70)

        # Check prerequisites
        logger.info("\n🔍 Checking prerequisites...")
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return 1

        logger.info("\n🚀 Starting autonomous agent loop...")
        logger.info("🧠 AUTONOMOUS MODE: Agent will create its own objectives!")
        logger.info("🔧 ALL MCP tools enabled")
        logger.info("Press Ctrl+C to stop")
        logger.info("-" * 70)

        try:
            while True:
                # Check max steps
                if self.max_steps and self.step_count >= self.max_steps:
                    logger.info(f"\n✅ Reached max steps ({self.max_steps})")
                    break

                logger.info(f"\n{'='*70}")
                logger.info(f"🤖 Step {self.step_count + 1}")
                context_size = self._calculate_context_size()
                logger.info(f"📚 History: {len(self.conversation_history)} turns ({context_size:,} chars)")
                logger.info(f"{'='*70}")

                # Fetch game state
                game_state_result = self._execute_function_call_by_name("get_game_state", {})

                # Parse for screenshot
                import json as json_module
                try:
                    game_state_data = json_module.loads(game_state_result)
                    screenshot_b64 = game_state_data.get("screenshot_base64")
                except:
                    screenshot_b64 = None

                # Build prompt
                prompt = self._build_structured_prompt(game_state_result, self.step_count)

                # Run step
                success, output = self.run_step(prompt, screenshot_b64=screenshot_b64)

                if not success:
                    logger.warning("⚠️ Step failed, waiting 5 seconds before retry...")
                    time.sleep(5)
                    continue

                # Increment step count
                self.step_count += 1
                logger.info(f"✅ Step {self.step_count} completed")

                # Update server metrics
                try:
                    update_server_metrics(self.server_url)
                except Exception as e:
                    logger.debug(f"Failed to update server metrics: {e}")

                # Auto-save checkpoint
                try:
                    checkpoint_response = requests.post(
                        f"{self.server_url}/checkpoint",
                        json={"step_count": self.step_count},
                        timeout=10
                    )

                    history_response = requests.post(
                        f"{self.server_url}/save_agent_history",
                        timeout=5
                    )

                    if checkpoint_response.status_code == 200 and history_response.status_code == 200:
                        if self.step_count % 10 == 0:
                            logger.info(f"💾 Checkpoint and history saved at step {self.step_count}")
                    else:
                        logger.warning(f"⚠️ Save failed - Checkpoint: {checkpoint_response.status_code}, History: {history_response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.debug(f"⚠️ Checkpoint/history save error: {e}")

                # Brief pause
                logger.info("⏸️  Waiting 1 second before next step...")
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n\n🛑 Agent stopped by user")
            logger.info(self._format_history_for_display())
            return 0
        except Exception as e:
            logger.error(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return 1

        logger.info(f"\n🎯 Agent completed {self.step_count} steps")
        logger.info(f"📚 Conversation history: {len(self.conversation_history)} turns")
        logger.info(self._format_history_for_display())
        return 0

    def _handle_vlm_function_calls(self, response, tool_calls_made, tool_call_count, max_tool_calls):
        """Handle function calls from VLM backend

        Function results are stored and will be included in the next step's context,
        allowing the agent to see and use the results in subsequent decisions.
        """
        if not hasattr(response, 'candidates') or not response.candidates:
            return False

        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not candidate.content:
            return False

        content = candidate.content
        if not hasattr(content, 'parts'):
            logger.warning("⚠️ Response content has no 'parts' attribute")
            return False

        logger.debug(f"🔍 Checking {len(content.parts)} response parts for function calls")
        function_calls_found = False
        for i, part in enumerate(content.parts):
            logger.debug(f"   Part {i}: has function_call={hasattr(part, 'function_call')}, has text={hasattr(part, 'text')}")
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                tool_call_count += 1
                logger.info(f"🔧 VLM wants to call: {function_call.name} ({tool_call_count}/{max_tool_calls})")

                # Execute the function
                function_result = self._execute_function_call(function_call)
                result_str = str(function_result)
                logger.info(f"📥 Function result: {result_str[:200]}...")

                # Track tool call
                tool_calls_made.append({
                    "name": function_call.name,
                    "args": self._convert_protobuf_args(function_call.args),
                    "result": function_result
                })

                # Wait for action queue to complete
                if function_call.name == "press_buttons":
                    self._wait_for_actions_complete()

                function_calls_found = True
            elif hasattr(part, 'text') and part.text:
                logger.debug(f"   Part {i} is text: {part.text[:100]}...")

        if not function_calls_found:
            logger.warning(f"⚠️ No function calls found in response (parts checked: {len(content.parts)})")
        elif len(tool_calls_made) == 0:
            logger.error(f"🚫 Function calls found but none were executed")

        result = function_calls_found and len(tool_calls_made) > 0
        logger.debug(f"   Returning: function_calls_found={function_calls_found}, tool_calls_made={len(tool_calls_made)}, result={result}")
        return result

    def _extract_text_from_response(self, response):
        """Extract text content from response"""
        if isinstance(response, str):
            return response.strip()

        try:
            if hasattr(response, 'text'):
                return response.text.strip()
            return ""
        except:
            return ""


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Pokemon Emerald Autonomous CLI Agent")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="Game server URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model to use"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to run"
    )
    parser.add_argument(
        "--system-instructions",
        type=str,
        default="POKEAGENT.md",
        help="System instructions file"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        help="VLM backend (gemini, vertex, openai, etc.)"
    )

    args = parser.parse_args()

    agent = AutonomousCLIAgent(
        server_url=args.server_url,
        model=args.model,
        backend=args.backend,
        max_steps=args.max_steps,
        system_instructions_file=args.system_instructions
    )

    return agent.run()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
My Custom CLI Agent for Pokemon Emerald
Based on the existing CLIAgent but with custom modifications.
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
from typing import Optional, Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai

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


class MyCLIAgent:
    """My Custom CLI Agent using Gemini API directly with MCP tools."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model: str = "gemini-2.5-flash",
        backend: str = "gemini",
        max_steps: Optional[int] = None,
        system_instructions_file: str = "POKEAGENT.md",
        max_context_chars: int = 100000,  # ~25k tokens for gemini-2.5-flash
        target_context_chars: int = 50000  # Compact down to this when exceeded
    ):
        print(f"🚀 Initializing MyCLIAgent with backend={backend}, model={model}, server={server_url}")
        self.server_url = server_url
        self.model = model
        self.backend = backend
        self.max_steps = max_steps
        self.step_count = 0
        self.max_context_chars = max_context_chars
        self.target_context_chars = target_context_chars

        # Conversation history for tracking and compaction
        self.conversation_history = []

        # Load system instructions
        self.system_instructions = self._load_system_instructions(system_instructions_file)

        # Initialize MCP tool adapter
        self.mcp_adapter = MCPToolAdapter(server_url)

        # Initialize VLM or Gemini based on backend
        if self.backend == "gemini":
            # Use original Gemini API for backward compatibility
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            genai.configure(api_key=api_key)

            # Define MCP tools as Gemini function declarations
            self.tools = self._create_tool_declarations()

            # Create the model with tools
            print(f"🔧 Creating Gemini model with {len(self.tools)} tools...")
            self.gemini_model = genai.GenerativeModel(
                model_name=self.model,
                tools=self.tools,
                system_instruction=self.system_instructions
            )
            print("✅ Gemini model created successfully")

            # Start chat session
            self.chat = self.gemini_model.start_chat(history=[])
        else:
            # Use VLM for other backends (like vertex)
            # Create tool declarations for function calling
            self.tools = self._create_tool_declarations()
            self.vlm = VLM(backend=self.backend, model_name=self.model, tools=self.tools)
            print(f"✅ VLM initialized with {self.backend} backend using model: {self.model}")

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
        """Create Gemini function declarations for ALL MCP tools (Pokemon + Baseline)."""

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
                            "description": "Explain why you are pressing these buttons"
                        }
                    },
                    "required": ["buttons", "reasoning"]
                }
            },
            {
                "name": "navigate_to",
                "description": "Automatically navigate to specific coordinates using A* pathfinding.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "x": {"type_": "INTEGER", "description": "Target X coordinate"},
                        "y": {"type_": "INTEGER", "description": "Target Y coordinate"},
                        "reason": {"type_": "STRING", "description": "Why you are navigating here"}
                    },
                    "required": ["x", "y", "reason"]
                }
            },
            {
                "name": "complete_direct_objective",
                "description": "Complete the current direct objective and advance to the next one. Use this when you have successfully completed the current objective's task.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "reasoning": {"type_": "STRING", "description": "Brief explanation of why the current direct objective is complete"}
                    },
                    "required": ["reasoning"]
                }
            },

            # Knowledge Base Tools
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

            # Wiki Tools
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
                "description": "Get official Emerald walkthrough (Parts 1-21). Part 1: Littleroot, Part 6: Roxanne, Part 21: Elite Four.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "part": {
                            "type_": "INTEGER",
                            "description": "Walkthrough part 1-21",
                        }
                    },
                    "required": ["part"]
                }
            },
            {
                "name": "create_direct_objectives",
                "description": "Create the next 3 direct objectives when a sequence completes. Use this after consulting get_walkthrough() or wiki sources to plan your next steps. Provide exactly 3 objectives with id, description, action_type, target_location, navigation_hint, and completion_condition.",
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
                "description": "Get comprehensive progress summary including completed milestones, objectives, current location, and knowledge base summary. Use this when a sequence completes to understand what you've accomplished before creating next objectives.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {},
                    "required": []
                }
            },

            # ============================================================
            # BASELINE MCP TOOLS (File/Shell/Web)
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

        logger.info(f"✅ Created {len(tools)} tool declarations (9 Pokemon + 11 Baseline)")
        return tools

    def _execute_function_call_by_name(self, function_name: str, arguments: dict) -> str:
        """Execute a function by name with given arguments and return result as JSON string."""
        # Call the tool via MCP adapter
        result = self.mcp_adapter.call_tool(function_name, arguments)
        # Return as JSON string
        return json.dumps(result, indent=2)

    def _convert_protobuf_args(self, proto_args) -> dict:
        """Convert protobuf arguments to JSON-serializable Python types.

        This recursively converts RepeatedComposite, RepeatedScalar, and other
        protobuf types to native Python lists and dicts.
        """
        arguments = {}
        for key, value in proto_args.items():
            # Convert protobuf types to native Python types
            if hasattr(value, '__class__') and 'proto' in value.__class__.__module__:
                # Check if it's a list-like type first (RepeatedComposite, RepeatedScalar)
                if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                    # It's a list/array - convert to Python list
                    try:
                        arguments[key] = list(value)
                    except:
                        arguments[key] = value
                else:
                    # It's a dict-like type - convert via dict
                    try:
                        arguments[key] = dict(value)
                    except:
                        arguments[key] = value
            else:
                arguments[key] = value
        return arguments

    def _execute_function_call(self, function_call) -> str:
        """Execute a function call and return the result as JSON string."""
        function_name = function_call.name

        # navigate_to is now enabled with porymap ground truth pathfinding

        # Parse arguments - convert protobuf types to native Python types
        try:
            arguments = self._convert_protobuf_args(function_call.args)
        except Exception as e:
            logger.error(f"Failed to parse function arguments: {e}")
            return json.dumps({"success": False, "error": f"Invalid arguments: {e}"})

        # Call the tool via MCP adapter
        result = self.mcp_adapter.call_tool(function_name, arguments)

        # Return as JSON string
        return json.dumps(result, indent=2)

    def _add_to_history(self, prompt: str, response: str, tool_calls: List[Dict] = None, action_details: str = None):
        """Add interaction to conversation history."""
        entry = {
            "step": self.step_count,
            "prompt": prompt,
            "response": response,
            "tool_calls": tool_calls or [],
            "timestamp": time.time()
        }
        
        # Extract action and action_details from tool_calls if available
        if tool_calls:
            last_call = tool_calls[-1]
            entry["action"] = last_call.get("name", "unknown")
            if action_details:
                entry["action_details"] = action_details
            elif last_call.get("name") == "navigate_to" and "x" in last_call.get("args", {}) and "y" in last_call.get("args", {}):
                # Check if we have pending action details from navigation
                if hasattr(self, '_pending_action_details') and 'navigate_to' in self._pending_action_details:
                    entry["action_details"] = self._pending_action_details['navigate_to']
                else:
                    entry["action_details"] = f"navigate_to({last_call['args']['x']}, {last_call['args']['y']})"
            elif last_call.get("name") == "press_buttons" and "buttons" in last_call.get("args", {}):
                entry["action_details"] = f"Pressed {last_call['args']['buttons']}"
            else:
                entry["action_details"] = f"Executed {last_call.get('name', 'unknown')}"
        
        self.conversation_history.append(entry)

        # Check if we need to compact history based on context length
        current_context_size = self._calculate_context_size()
        if current_context_size > self.max_context_chars:
            self._compact_history()

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

    def _compact_history(self):
        """Compact conversation history by removing oldest entries until under target size."""
        current_size = self._calculate_context_size()

        if current_size <= self.target_context_chars:
            return

        logger.info(f"📚 Compacting history: {current_size:,} chars → target {self.target_context_chars:,} chars")

        # Remove oldest entries until we're under target
        removed_count = 0
        while len(self.conversation_history) > 1 and self._calculate_context_size() > self.target_context_chars:
            self.conversation_history.pop(0)
            removed_count += 1

        new_size = self._calculate_context_size()
        logger.info(f"   Removed {removed_count} oldest turns")
        logger.info(f"   New size: {new_size:,} chars ({len(self.conversation_history)} turns remaining)")

    def _format_history_for_display(self) -> str:
        """Format conversation history for display."""
        if not self.conversation_history:
            return "No conversation history yet."

        lines = [f"\n{'='*70}", "CONVERSATION HISTORY", '='*70]

        for entry in self.conversation_history[-10:]:  # Show last 10
            lines.append(f"\nStep {entry['step']}:")
            lines.append(f"  Prompt: {entry['prompt'][:100]}...")
            if entry.get('tool_calls'):
                lines.append(f"  Tools called: {', '.join(t['name'] for t in entry['tool_calls'])}")
            lines.append(f"  Response: {entry['response'][:100]}...")

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
            prompt: The prompt to send to Gemini
            max_tool_calls: Maximum number of tool calls allowed per step (default: 5)
            screenshot_b64: Optional base64-encoded screenshot to include with prompt

        Returns:
            Tuple of (success: bool, response: str)
        """
        try:
            logger.info(f"📤 Sending prompt to {self.backend}...")
            logger.info(f"   Model: {self.model}")
            # Don't log the full prompt - it's too long with game state
            logger.info(f"   Prompt length: {len(prompt)} chars")
            if screenshot_b64:
                logger.info(f"   Including screenshot ({len(screenshot_b64)} bytes)")

            tool_calls_made = []
            reasoning_text = ""  # Capture any text reasoning before tool calls
            enforcement_retry_count = 0  # Track how many times we've tried to enforce action tool
            max_enforcement_retries = 3

            # Track duration
            start_time = time.time()

            # Handle different backends
            if self.backend == "gemini":
                # Use original Gemini function calling
                # Build message content with optional image
                if screenshot_b64:
                    # Send message with both text and image
                    import PIL.Image as PILImage
                    import io
                    import base64

                    # Decode base64 to image
                    image_data = base64.b64decode(screenshot_b64)
                    image = PILImage.open(io.BytesIO(image_data))

                    # Send message with image and text
                    response = self.chat.send_message([prompt, image])
                else:
                    # Send message with text only
                    response = self.chat.send_message(prompt)
            else:
                # Use VLM for other backends
                if screenshot_b64:
                    import PIL.Image as PILImage
                    import io
                    import base64
                    
                    # Decode base64 to image
                    image_data = base64.b64decode(screenshot_b64)
                    image = PILImage.open(io.BytesIO(image_data))
                    
                    # Use VLM with image
                    response = self.vlm.get_query(image, prompt, "CLI_Agent")
                else:
                    # Use VLM with text only
                    response = self.vlm.get_text_query(prompt, "CLI_Agent")
                
                # Determine if response is a GenerationResponse object (function calling) or string (text)
                is_function_calling = hasattr(response, 'candidates')

            # Process response - handle function calls
            # Limit the number of tool calls per step to prevent infinite loops
            tool_call_count = 0
            
            # For VLM backends with native function calling support
            if self.backend != "gemini":
                # Handle native function calls from VLM backends (VertexAI, etc.)
                function_calls_executed = self._handle_vertex_function_calls(
                    response, tool_calls_made, tool_call_count, max_tool_calls
                )
                
                # If function calls were found and executed, end the step
                if function_calls_executed:
                    # END THE STEP IMMEDIATELY after action execution
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
                    
                    # Build full response including reasoning + action summary
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
                    
                    # Display the reasoning to user
                    display_text = self._extract_text_from_response(response)
                    if tool_reasoning:
                        if display_text:
                            display_text += f"\n\n{tool_reasoning}"
                        else:
                            display_text = tool_reasoning
                    
                    logger.info(f"✅ Step completed in {duration:.2f}s")
                    logger.info(f"📝 Response: {display_text}")
                    
                    # Store in conversation history with action tracking
                    self.conversation_history.append({
                        "step": self.step_count,
                        "role": "user",
                        "content": prompt,
                        "timestamp": time.time()
                    })
                    
                    # Extract action details for better tracking
                    action_taken = last_tool_call['name']
                    action_details = ""
                    if last_tool_call['name'] == "press_buttons" and "buttons" in last_tool_call["args"]:
                        action_details = f"Pressed {last_tool_call['args']['buttons']}"
                    elif last_tool_call['name'] == "navigate_to" and "x" in last_tool_call["args"] and "y" in last_tool_call["args"]:
                        target_x = last_tool_call["args"]["x"]
                        target_y = last_tool_call["args"]["y"]
                        # Wait for actions to complete first, then get final position
                        self._wait_for_actions_complete()
                        final_pos = None
                        try:
                            # Get state to find final position after navigation completes
                            final_state_result = self._execute_function_call_by_name("get_game_state", {})
                            import json as json_module
                            final_state_data = json_module.loads(final_state_result)
                            if final_state_data.get("success"):
                                player_pos = final_state_data.get("player_position", {})
                                if player_pos:
                                    final_pos = (player_pos.get("x"), player_pos.get("y"))
                        except:
                            pass
                        
                        if final_pos:
                            action_details = f"navigate_to({target_x}, {target_y}) → Ended at ({final_pos[0]}, {final_pos[1]})"
                        else:
                            action_details = f"navigate_to({target_x}, {target_y})"
                    elif last_tool_call['name'] == "complete_direct_objective":
                        action_details = "Completed direct objective"
                    else:
                        action_details = f"Executed {last_tool_call['name']}"
                    
                    self.conversation_history.append({
                        "step": self.step_count,
                        "role": "assistant", 
                        "content": full_response,
                        "tool_calls": tool_calls_made,
                        "action": action_taken,
                        "action_details": action_details,
                        "reasoning": tool_reasoning,
                        "timestamp": time.time()
                    })
                    
                    # Compact history if needed
                    self._compact_history()
                    
                    return True, full_response
                else:
                    # No function calls found - this might indicate an issue with function calling setup
                    if self.backend != "gemini":
                        logger.warning(f"⚠️  No function calls found from {self.backend} backend. This might indicate:")
                        logger.warning(f"   - Function calling not properly configured")
                        logger.warning(f"   - Model not generating function calls")
                        logger.warning(f"   - Response format not as expected")
                    
                    # Treat as text response
                    text_content = self._extract_text_from_response(response)
                    if not text_content:
                        text_content = str(response)
                    
                    logger.info(f"📥 Received text response from {self.backend}:")
                    logger.info(f"   {text_content}")
                    
                    # For VLM backends, we can't enforce action tools like we do with Gemini
                    # Just return the text response
                    duration = time.time() - start_time
                    
                    logger.info(f"✅ Step completed in {duration:.2f}s")
                    
                    # Store in conversation history
                    self.conversation_history.append({
                        "step": self.step_count,
                        "role": "user",
                        "content": prompt,
                        "timestamp": time.time()
                    })
                    self.conversation_history.append({
                        "step": self.step_count,
                        "role": "assistant",
                        "content": text_content,
                        "timestamp": time.time()
                    })
                    
                    # Compact history if needed
                    self._compact_history()
                    
                    return True, text_content
            else:
                # Original Gemini function calling logic
                while response.parts and tool_call_count < max_tool_calls:
                    # First, extract any text parts for reasoning
                    for part in response.parts:
                        if hasattr(part, 'text') and part.text:
                            reasoning_text += part.text + "\n"

                    part = response.parts[0]

                    # Check if it's a function call
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        tool_call_count += 1
                        logger.info(f"🔧 Gemini wants to call: {function_call.name} ({tool_call_count}/{max_tool_calls})")

                        # Execute the function
                        function_result = self._execute_function_call(function_call)
                        logger.info(f"📥 Function result: {function_result[:200]}...")

                        # Track tool call with result (convert protobuf args to JSON-serializable types)
                        tool_calls_made.append({
                            "name": function_call.name,
                            "args": self._convert_protobuf_args(function_call.args),
                            "result": function_result
                        })

                        # Wait for action queue to complete if this was a button press or navigation
                        if function_call.name in ["press_buttons", "navigate_to"]:
                            # Send the function response back to Gemini to maintain chat state
                            # This is important to keep the conversation flowing
                            self.chat.send_message(
                                genai.protos.Content(
                                    parts=[genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=function_call.name,
                                            response={"result": function_result}
                                        )
                                    )]
                                )
                            )

                        # Now wait for actions to complete
                        self._wait_for_actions_complete()

                        # END THE STEP IMMEDIATELY after action execution
                        # Calculate duration
                        duration = time.time() - start_time

                        # Extract reasoning from function arguments
                        tool_reasoning = ""
                        final_position = None
                        try:
                            # Get args from the tool call that was made
                            if tool_calls_made:
                                last_call = tool_calls_made[-1]
                                if function_call.name == "navigate_to" and "reason" in last_call["args"]:
                                    tool_reasoning = last_call["args"]["reason"]
                                    # Get final position after navigation
                                    try:
                                        final_state = self._execute_function_call_by_name("get_game_state", {})
                                        import json as json_module
                                        final_state_data = json_module.loads(final_state)
                                        if final_state_data.get("success"):
                                            player_pos = final_state_data.get("player_position", {})
                                            if player_pos:
                                                final_position = (player_pos.get("x"), player_pos.get("y"))
                                    except Exception as e:
                                        logger.debug(f"Could not get final position after navigate_to: {e}")
                                elif function_call.name == "press_buttons" and "reasoning" in last_call["args"]:
                                    tool_reasoning = last_call["args"]["reasoning"]
                        except Exception as e:
                            logger.debug(f"Could not extract tool reasoning: {e}")

                        # Build full response including reasoning + action summary
                        full_response = reasoning_text.strip()
                        if tool_reasoning:
                            if full_response:
                                full_response += f"\n\n{tool_reasoning}"
                            else:
                                full_response = tool_reasoning

                        if full_response:
                            full_response += f"\n\nAction executed: {function_call.name}"
                        else:
                            full_response = f"Executed {function_call.name}"

                        # Display the reasoning to user
                        display_text = reasoning_text.strip()
                        if tool_reasoning:
                            if display_text:
                                display_text += f"\n\n{tool_reasoning}"
                            else:
                                display_text = tool_reasoning

                        if display_text:
                            logger.info("📥 Gemini reasoning:")
                            print("\n" + "="*70)
                            print(display_text)
                            print("="*70 + "\n")

                        # Build action_details for navigate_to showing final position
                        action_details_str = None
                        if function_call.name == "navigate_to" and tool_calls_made:
                            last_call = tool_calls_made[-1]
                            if "x" in last_call["args"] and "y" in last_call["args"]:
                                target_x = last_call["args"]["x"]
                                target_y = last_call["args"]["y"]
                                if final_position:
                                    final_x, final_y = final_position
                                    action_details_str = f"navigate_to({target_x}, {target_y}) → Ended at ({final_x}, {final_y})"
                                else:
                                    action_details_str = f"navigate_to({target_x}, {target_y})"
                        
                        # Add to history with action details
                        self._add_to_history(prompt, full_response, tool_calls_made, action_details=action_details_str)

                        # Log to LLM logger
                        self._log_thinking(prompt, full_response, duration, tool_calls_made)

                        # Send reasoning to server for display in stream
                        thinking_to_send = display_text if display_text else full_response
                        self._send_thinking_to_server(thinking_to_send, self.step_count + 1)

                        logger.info(f"✅ Step ended after {function_call.name} - will observe results in next step")
                        return True, full_response

                    # Check if we've hit the limit
                    if tool_call_count >= max_tool_calls:
                        logger.warning(f"⚠️ Reached max tool calls ({max_tool_calls}). Forcing text response.")
                        # Send function result with a prompt to respond with text
                        response = self.chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=function_call.name,
                                        response={"result": function_result}
                                    )
                                ), genai.protos.Part(
                                    text="You have reached the maximum number of tool calls for this step. Please provide a brief text response summarizing what you accomplished and what you plan to do next."
                                )]
                            )
                        )
                    else:
                        # Send function result back to Gemini
                        response = self.chat.send_message(
                            genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=function_call.name,
                                        response={"result": function_result}
                                    )
                                )]
                            )
                        )
                
                # Check if we got a text response instead of function call
                if not hasattr(part, 'function_call') or not part.function_call:
                    # Check if any part has text
                    for part in response.parts:
                        if hasattr(part, 'text') and part.text:
                            # Got text response
                            text_response = part.text
                            logger.info("📥 Received response from Gemini:")
                            print("\n" + "="*70)
                            print(text_response)
                            print("="*70 + "\n")

                            # Check if this is a text-only response without action tools
                            has_action_tool = any(
                                call["name"] == "press_buttons"
                                for call in tool_calls_made
                            )

                            if not has_action_tool:
                                enforcement_retry_count += 1

                                if enforcement_retry_count > max_enforcement_retries:
                                    logger.error(f"❌ Gemini failed to call action tool after {max_enforcement_retries} retries!")
                                    logger.error(f"   Reasoning provided: {reasoning_text[:200]}...")
                                    logger.error(f"   Falling back to WAIT action")

                                    # Force a WAIT action to avoid getting stuck
                                    tool_calls_made.append({
                                        "name": "press_buttons",
                                        "args": {"buttons": ["WAIT"], "reasoning": "Fallback: Agent stuck in text loop"},
                                        "result": "Forced WAIT action due to agent text loop"
                                    })

                                    # Don't continue the loop - break out and return
                                    break

                                logger.warning(f"⚠️ Gemini provided text but no action tool! (Retry {enforcement_retry_count}/{max_enforcement_retries})")

                                # Send a message that DEMANDS a function call with increasingly forceful language
                                if enforcement_retry_count == 1:
                                    enforcement_msg = "You MUST call press_buttons now. No more text analysis. Just call the tool based on your plan. Use press_buttons(['WAIT']) if unsure."
                                elif enforcement_retry_count == 2:
                                    enforcement_msg = "CRITICAL: Call press_buttons([...]) RIGHT NOW. Do not write any text. Only make the function call."
                                else:
                                    enforcement_msg = "EMERGENCY: You are stuck in a loop. Call press_buttons(['WAIT'], 'observing') immediately. This is your final chance."

                                try:
                                    response = self.chat.send_message(enforcement_msg)
                                    # Loop back to process the action tool call
                                    continue
                                except Exception as e:
                                    # Handle malformed function calls or other errors during enforcement
                                    logger.error(f"❌ Enforcement failed with error: {e}")
                                    logger.error(f"   Gemini attempted malformed function call")

                                    # Force WAIT action and break out
                                    tool_calls_made.append({
                                        "name": "press_buttons",
                                        "args": {"buttons": ["WAIT"], "reasoning": f"Fallback: Enforcement failed - {str(e)[:50]}"},
                                        "result": "Forced WAIT due to enforcement error"
                                    })
                                    break

                            # Calculate duration
                            duration = time.time() - start_time

                            # Add to history
                            self._add_to_history(prompt, text_response, tool_calls_made)

                            # Log to LLM logger with duration and tool calls
                            self._log_thinking(prompt, text_response, duration, tool_calls_made)

                            # Send thinking to server for display in stream
                            self._send_thinking_to_server(text_response, self.step_count + 1)

                            return True, text_response

            # If we get here, no text response was generated
            logger.warning("⚠️ No text response from Gemini")
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
                        time.sleep(0.5)  # Poll every 500ms
                else:
                    logger.warning(f"Failed to get queue status: {response.status_code}")
                    time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error checking queue status: {e}")
                time.sleep(0.5)

        logger.warning(f"⚠️ Timeout waiting for actions to complete after {timeout}s")

    def _log_thinking(self, prompt: str, response: str, duration: float = None, tool_calls: list = None) -> None:
        """Log interaction to LLM logger with full tool call history."""
        try:
            self.llm_logger.log_interaction(
                interaction_type="my_gemini_cli",
                prompt=prompt,
                response=response,
                duration=duration,
                metadata={"tool_calls": tool_calls or []},
                model_info={"model": self.model}
            )
            logger.debug("✅ Logged to LLM logger")
        except Exception as e:
            logger.debug(f"Could not log to LLM logger: {e}")

    def _build_structured_prompt(self, game_state_result: str, step_count: int) -> str:
        """Build a function-call focused prompt that clearly explains available tools."""
        
        # Parse game state to extract relevant information
        import json as json_module
        try:
            game_state_data = json_module.loads(game_state_result)
        except:
            game_state_data = {}
        
        # Extract key information from game state
        state_text = game_state_data.get("state_text", "")
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
        
        # Format objective context to highlight previous, current, and next
        if direct_objective_context:
            # The context is already formatted nicely by DirectObjectiveManager
            # Just make it more prominent
            pass  # Keep as-is
        
        # Build action history summary for better context
        action_history = self._format_action_history()
        
        # Build function-call focused prompt
        prompt = f"""You are an expert navigator and battle strategist playing Pokémon Emerald on a Game Boy Advance emulator.

Some pointers to keep in mind (guard rails) as you problem solve:
1) You must think step-by-step when solving problems and making decisions. 
2) Always provide detailed, context-aware responses that bias for ground-truth.
3) Consider the current situation in the game as well as what you've learned over time.
4) Do not fixate on the correctness of a particular solution, be flexible and adapt your strategy as needed.
5) **CRITICAL**: Always check the game screen for dialogue boxes first - if you see dialogue, advance it with press_buttons(["A"]) before doing anything else.
Especially If a current approach is leading to consistent failure without providing knowledge on how to improve.

ACTION HISTORY (last 20 steps):
{action_history}

================================================================================
🎯🎯🎯 CURRENT DIRECT OBJECTIVE - READ THIS CAREFULLY 🎯🎯🎯
================================================================================

{direct_objective_context}

{direct_objective}

{direct_objective_status}

================================================================================
⚠️ CRITICAL: When you have completed the objective above, you MUST call (prioritize this before progressing through dialogue):
   complete_direct_objective(reasoning="<explain why it's complete>")
   
🔄 SEQUENCE COMPLETION HANDLING:
When you see "All objectives completed!" or sequence_complete=True in the response:
1. Call get_progress_summary() to see what you've accomplished (milestones, objectives, location, knowledge)
2. Use get_walkthrough(part=X) to find the next relevant walkthrough part based on your current location/progress
3. Optionally use lookup_pokemon_info() for specific location/NPC information
4. Create the next 3 logical objectives using create_direct_objectives():
   - Base them on walkthrough/wiki information
   - Format them with clear descriptions, action_types, target_locations, navigation_hints
   - Use completion_condition to specify how to verify completion
5. Once created, proceed with the first new objective

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
================================================================================

CURRENT GAME STATE:
{state_text}

**DIALOGUE CHECK**: Look at the game screen carefully - if you see a dialogue box with text, press_buttons(["A"], reasoning).

AVAILABLE TOOLS - Use these function calls to interact with the game:

🎮 **PRIMARY GAME TOOLS** :
- get_game_state() - Get current game state, player position, Pokemon, map, and screenshot
- complete_direct_objective(reasoning) - Mark current direct objective as complete. (prioritize this before progressing through dialogue)
- press_buttons(buttons, reasoning) - Press GBA buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R, WAIT
- navigate_to(x, y, reason) - Automatically pathfind to coordinates using A* algorithm with porymap ground truth data

🗺️ **NAVIGATION**: Use navigate_to(x, y, reason) to automatically pathfind to a coordinate. It uses A* pathfinding on the porymap ground truth map. You can also use press_buttons() for manual movement if navigate_to isn't working.

📚 **INFORMATION TOOLS** (use when you need info or are stuck in a loop):
- lookup_pokemon_info(topic, source) - Look up Pokemon, moves, locations from wikis
- get_walkthrough(part) - Get official Emerald walkthrough (parts 1-21)
- search_knowledge(query, category) - Search your stored knowledge
- add_knowledge(category, title, content, importance) - Store important discoveries
- get_progress_summary() - Get comprehensive progress summary (milestones, objectives, location, knowledge)

💾 **KNOWLEDGE TOOLS** (use to remember things):
- get_knowledge_summary(min_importance) - Get summary of important discoveries
- save_memory(fact) - Save facts to remember across sessions

🎯 **OBJECTIVE MANAGEMENT** (use when sequences complete):
- create_direct_objectives(objectives, reasoning) - Create next 3 direct objectives dynamically
  Use this after get_progress_summary() and get_walkthrough() to plan your next steps

** COORDINATE & MOVEMENT EXAMPLES **:
- Pressing LEFT decreases your X coordinate (moves you west)
- Pressing RIGHT increases your X coordinate (moves you east)  
- Pressing UP decreases your Y coordinate (moves you north)
- Pressing DOWN increases your Y coordinate (moves you south)

Example: If you are at position (5, 5) and press RIGHT, you will move to (6, 5)
Example: If you are at position (3, 8) and press UP, you will move to (3, 7)

** INTERACTION TIPS **:
- To interact with an object or NPC, you must be both 1) on an adjacent tile to the NPC or object and 2) facing the NPC or object.

STRATEGY - PRIORITY ORDER:
1. **CHECK OBJECTIVE COMPLETION FIRST**: Before doing ANYTHING, check if your current direct objective is complete. If you've accomplished what the objective asks for, IMMEDIATELY call complete_direct_objective(reasoning="...") 
2. **DIALOGUE SECOND**: If you see a dialogue box on screen, ALWAYS use press_buttons(["A"], reasoning) to advance it
3. **MOVEMENT**: Preferentially use navigate_to(x, y, reason) to automatically pathfind to a coordinate. Use press_buttons(["UP"], reasoning) etc. for manual movement only if navigate_to is not working.
4. **BATTLES**: Use press_buttons with battle moves. Select moves carefully based on the current situation and the enemy's Pokemon. If below 75% health, prioritize using healing moves that also cause damage like (absorb, gigadrain, etc) if these moves are available.
5. **INFORMATION**: Use lookup_pokemon_info or get_walkthrough when you need to know something
6. **STUCK DETECTION**: If you've been attempting the same move (UP, DOWN, LEFT, RIGHT) for an extended period of time without your player coordinates changing, try a different direction to move around the obstacle that still conforms to the objective.

🔴 **REMEMBER**: You MUST call complete_direct_objective() as soon as you've completed the current objective! The agent will NOT automatically know you're done - you must explicitly call the function.

IMPORTANT: Always check the game screen for dialogue boxes before planning movement!
**CRITICAL**: After performing any action, proactively check if your current direct objective is complete!

═══════════════════════════════════════════════════════════════════════
🧠 HOW TO STRUCTURE YOUR REASONING 
═══════════════════════════════════════════════════════════════════════

You MUST follow this decision-making process for EVERY action:

**STEP 1 - ANALYZE**: Examine the current situation
   - What do I see on the game screen?
   - Where am I located? (coordinates and map name)
   - What is my current objective?
   - What obstacles or opportunities are present?
   - Is there dialogue I need to advance?

**STEP 2 - PLAN**: Decide what to do next
   - What action will help achieve my objective?
   - Why is this action the best choice right now?
   - What do I expect to happen after this action?
   - Are there any risks or alternative approaches?

**STEP 3 - EXECUTE**: Call the function with DETAILED reasoning

🎯 **CRITICAL REQUIREMENT**: 
Your `reasoning` parameter MUST contain your FULL analysis and plan!
Format it with clear sections so your thinking is visible and traceable.

**GOOD EXAMPLE** (detailed reasoning):
press_buttons(
    buttons=["UP"],
    reasoning='''ANALYSIS: I'm at position (10, 2) in Littleroot Town. The movement preview shows UP leads to (10, 1) which is walkable terrain. My current objective is to explore the town and find the Pokemon Lab. Looking at the map, there are buildings to the north.

PLAN: I'll move UP one tile to get closer to the northern buildings. This is safe according to the movement preview - no obstacles or NPCs blocking the path AND I don't see any NPCs in my way in the visual frame. After moving, I'll reassess and continue exploring north.

ACTION: Pressing UP to move north toward potential buildings/NPCs.'''
)

**BAD EXAMPLE** (too brief):
press_buttons(
    buttons=["UP"],
    reasoning="Moving north"
)

**ANOTHER GOOD EXAMPLE** (dialogue):
press_buttons(
    buttons=["A"],
    reasoning='''ANALYSIS: I can see a dialogue box on screen with text from Professor Birch. He's explaining something about Pokemon. The dialogue box is visible in the bottom portion of the screen.

PLAN: I need to advance this dialogue by pressing A to read what he says and continue the conversation. This is standard Pokemon game interaction.

ACTION: Pressing A to advance the dialogue and hear what Professor Birch has to say.'''
)

**WHY THIS MATTERS**: 
- Detailed reasoning helps track your decision-making process
- It makes debugging easier when things go wrong
- It provides context for future decisions
- It demonstrates goal-oriented thinking

Think step-by-step through ANALYZE → PLAN → EXECUTE, then call the appropriate function with your detailed reasoning.

Step {step_count}"""
        
        return prompt


    def _format_action_history(self) -> str:
        """Format action history for better context awareness"""
        if not self.conversation_history:
            return "No previous actions recorded."
        
        # Get last 20 conversation entries
        recent_entries = self.conversation_history[-20:]
        
        history_lines = []
        for i, entry in enumerate(recent_entries, 1):
            step = entry.get("step", "?")
            timestamp = entry.get("timestamp", "")
            action = entry.get("action", "Unknown")
            action_details = entry.get("action_details", "")
            reasoning = entry.get("reasoning", "")
            
            # Format timestamp nicely
            if timestamp:
                try:
                    from datetime import datetime
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                    else:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = str(timestamp)[:8] if len(str(timestamp)) > 8 else str(timestamp)
            else:
                time_str = "??:??:??"
            
            # Format action and reasoning
            action_display = action_details if action_details else action
            action_str = f"Step {step} ({time_str}): {action_display}"
            if reasoning and reasoning != "No reasoning provided":
                action_str += f" | Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}"
            
            history_lines.append(action_str)
        
        return "\n".join(history_lines)

    def run(self) -> int:
        """Run the agent loop."""
        logger.info("=" * 70)
        logger.info("🎮 Pokemon Emerald My Custom CLI Agent (Gemini API)")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model}")
        logger.info(f"Server: {self.server_url}")
        if hasattr(self, 'tools') and self.tools:
            logger.info(f"Tools: {len(self.tools)} MCP tools (9 Pokemon + 11 Baseline)")
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

        logger.info("\n🚀 Starting agent loop...")
        logger.info("📝 Gemini API with native function calling")
        logger.info("💾 Conversation history with auto-compaction")
        logger.info("🔧 20 MCP tools available via HTTP (Pokemon + Baseline)")
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

                # Automatically fetch game state at the beginning of each step
                game_state_result = self._execute_function_call_by_name("get_game_state", {})

                # Parse game state result to extract screenshot if available
                import json as json_module
                try:
                    game_state_data = json_module.loads(game_state_result)
                    screenshot_b64 = game_state_data.get("screenshot_base64")
                except:
                    screenshot_b64 = None

                # Build structured prompt for this step
                prompt = self._build_structured_prompt(game_state_result, self.step_count)

                # Run step with optional screenshot
                success, output = self.run_step(prompt, screenshot_b64=screenshot_b64)

                if not success:
                    logger.warning("⚠️ Step failed, waiting 5 seconds before retry...")
                    time.sleep(5)
                    continue

                # Increment step count
                self.step_count += 1
                logger.info(f"✅ Step {self.step_count} completed")

                # Update server metrics with LLM usage
                try:
                    update_server_metrics(self.server_url)
                except Exception as e:
                    logger.debug(f"Failed to update server metrics: {e}")

                # Auto-save checkpoint after each step for persistence
                try:
                    # Save game state checkpoint
                    checkpoint_response = requests.post(
                        f"{self.server_url}/checkpoint",
                        json={"step_count": self.step_count},
                        timeout=10
                    )
                    
                    # Save agent history to checkpoint_llm.txt
                    history_response = requests.post(
                        f"{self.server_url}/save_agent_history",
                        timeout=5
                    )
                    
                    if checkpoint_response.status_code == 200 and history_response.status_code == 200:
                        if self.step_count % 10 == 0:  # Log every 10 steps to avoid spam
                            logger.info(f"💾 Checkpoint and history saved at step {self.step_count}")
                    else:
                        logger.warning(f"⚠️ Save failed - Checkpoint: {checkpoint_response.status_code}, History: {history_response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.debug(f"⚠️ Checkpoint/history save error: {e}")

                # Brief pause between steps
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

    def _handle_vertex_function_calls(self, response, tool_calls_made, tool_call_count, max_tool_calls):
        """Handle function calls from VertexAI backend
        
        Args:
            response: Response object (GenerationResponse or string)
            tool_calls_made: List to append executed calls to
            tool_call_count: Current count of tool calls
            max_tool_calls: Maximum allowed tool calls
            
        Returns:
            bool: True if function calls were executed, False otherwise
        """
        if not hasattr(response, 'candidates') or not response.candidates:
            return False
        
        candidate = response.candidates[0]
        if not hasattr(candidate, 'content') or not candidate.content:
            return False
            
        content = candidate.content
        if not hasattr(content, 'parts'):
            return False
        
        function_calls_found = False
        for part in content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                tool_call_count += 1
                logger.info(f"🔧 VLM wants to call: {function_call.name} ({tool_call_count}/{max_tool_calls})")
                
                # Execute the function
                function_result = self._execute_function_call(function_call, self.mcp_adapter)
                result_str = str(function_result)
                logger.info(f"📥 Function result: {result_str[:200]}...")
                
                # Track tool call with result
                tool_calls_made.append({
                    "name": function_call.name,
                    "args": dict(function_call.args),
                    "result": function_result
                })
                
                # Wait for action queue to complete if this was a button press
                if function_call.name == "press_buttons":
                    self._wait_for_actions_complete()
                
                function_calls_found = True
        
        return function_calls_found and len(tool_calls_made) > 0
    
    def _extract_text_from_response(self, response):
        """Extract text content from response (handles both string and GenerationResponse)
        
        Args:
            response: Response object (string or GenerationResponse)
            
        Returns:
            str: Extracted text content
        """
        if isinstance(response, str):
            return response.strip()
        
        # Try to extract text from GenerationResponse
        try:
            if hasattr(response, 'text'):
                return response.text.strip()
            return ""
        except:
            return ""
    
    def _execute_function_call(self, function_call, mcp_adapter):
        """Execute a function call using the MCP adapter."""
        try:
            # Convert function call to the format expected by MCP adapter
            function_name = function_call.name
            function_args = dict(function_call.args)
            
            # Call the MCP tool
            result = mcp_adapter.call_tool(function_name, function_args)
            return result
        except Exception as e:
            logger.error(f"Error executing function call {function_call.name}: {e}")
            return {"error": str(e)}


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Pokemon Emerald My Custom CLI Agent")
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
        help="Gemini model to use"
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

    agent = MyCLIAgent(
        server_url=args.server_url,
        model=args.model,
        backend=args.backend,
        max_steps=args.max_steps,
        system_instructions_file=args.system_instructions
    )

    return agent.run()


# DEPRECATED: Text parsing functions have been removed in favor of native function calling
# The following functions were used for parsing text responses but are no longer needed:
# - _parse_tool_call_from_text()
# - _execute_tool_call_from_text()
# 
# All VLM backends now use native function calling through proper tool declarations.


if __name__ == "__main__":
    sys.exit(main())

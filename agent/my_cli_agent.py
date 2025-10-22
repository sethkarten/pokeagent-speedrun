#!/usr/bin/env python3
"""
My Custom CLI Agent for Pokemon Emerald
Based on the existing CLIAgent but with custom modifications.
Uses Gemini API directly with MCP tools exposed as function declarations.
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
        max_steps: Optional[int] = None,
        system_instructions_file: str = "POKEAGENT.md",
        max_context_chars: int = 100000,  # ~25k tokens for gemini-2.5-flash
        target_context_chars: int = 50000  # Compact down to this when exceeded
    ):
        print(f"🚀 Initializing MyCLIAgent with model={model}, server={server_url}")
        self.server_url = server_url
        self.model = model
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

        # Initialize Gemini
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
                "description": "Write file to .pokeagent_cache/cli/ directory ONLY. Creates directories if needed.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "file_path": {"type_": "STRING", "description": "Path within .pokeagent_cache/cli/"},
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

    def _add_to_history(self, prompt: str, response: str, tool_calls: List[Dict] = None):
        """Add interaction to conversation history."""
        self.conversation_history.append({
            "step": self.step_count,
            "prompt": prompt,
            "response": response,
            "tool_calls": tool_calls or [],
            "timestamp": time.time()
        })

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
        # Check if API key is set
        if not os.environ.get("GEMINI_API_KEY"):
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
            logger.info(f"📤 Sending prompt to Gemini...")
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

            # Process response - handle function calls
            # Limit the number of tool calls per step to prevent infinite loops
            tool_call_count = 0
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
                        try:
                            # Get args from the tool call that was made
                            if tool_calls_made:
                                last_call = tool_calls_made[-1]
                                if function_call.name == "navigate_to" and "reason" in last_call["args"]:
                                    tool_reasoning = last_call["args"]["reason"]
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

                        # Add to history
                        self._add_to_history(prompt, full_response, tool_calls_made)

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
                elif hasattr(part, 'text') and part.text:
                    # Got text response
                    text_response = part.text
                    logger.info("📥 Received response from Gemini:")
                    print("\n" + "="*70)
                    print(text_response)
                    print("="*70 + "\n")

                    # Check if this is a text-only response without action tools
                    has_action_tool = any(
                        call["name"] in ["navigate_to", "press_buttons"]
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
                            enforcement_msg = "You MUST call either navigate_to or press_buttons now. No more text analysis. Just call the tool based on your plan. Use press_buttons(['WAIT']) if unsure."
                        elif enforcement_retry_count == 2:
                            enforcement_msg = "CRITICAL: Call navigate_to(...) or press_buttons([...]) RIGHT NOW. Do not write any text. Only make the function call."
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
                else:
                    # Unknown part type, skip it
                    logger.warning(f"⚠️ Unknown part type: {part}")
                    break

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

    def run(self) -> int:
        """Run the agent loop."""
        logger.info("=" * 70)
        logger.info("🎮 Pokemon Emerald My Custom CLI Agent (Gemini API)")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model}")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Tools: {len(self.tools)} MCP tools (9 Pokemon + 11 Baseline)")
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

                # Build prompt for this step with game state included
                if self.step_count == 0:
                    prompt = f"Here is the current game state:\n\n{game_state_result}\n\nBased on this state, decide on and execute the next action to progress through the game."
                else:
                    prompt = f"Here is the current game state:\n\n{game_state_result}\n\nBased on this state and the previous actions, decide on and execute the next action to progress through the game."

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

                # Brief pause
                logger.info("⏸️  Waiting 3 seconds before next step...")
                time.sleep(3)

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

    args = parser.parse_args()

    agent = MyCLIAgent(
        server_url=args.server_url,
        model=args.model,
        max_steps=args.max_steps,
        system_instructions_file=args.system_instructions
    )

    return agent.run()


if __name__ == "__main__":
    sys.exit(main())

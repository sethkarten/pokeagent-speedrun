#!/usr/bin/env python3
"""
Vision-Only CLI Agent for Pokemon Emerald
This agent relies purely on visual input (screenshots) without map information or pathfinding.
Uses Gemini API (or VertexAI) directly with MCP tools exposed as function declarations.
Navigates using only directional buttons and visual observations.
"""

import os
import sys
import time
import json
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np

import json as json_module

import PIL.Image as PILImage
import io
import base64

import traceback


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


# Local imports
from utils.agent_helpers import update_server_metrics
from utils.llm_logger import get_llm_logger
from utils.vlm import VLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPToolAdapter:
    """Adapter to call MCP server tools via HTTP."""

    def __init__(self, server_url: str, print_game_state: bool = True):
        self.server_url = server_url
        self.print_game_state = print_game_state  # Control whether to print full game state

    def _convert_protobuf_to_native(self, value):
        """Recursively convert protobuf objects to native Python types."""
        # Check if it's a protobuf object
        if hasattr(value, '__class__') and 'proto' in value.__class__.__module__:
            try:
                # Check if it's dict-like first (has .items() method) - MapComposite
                if hasattr(value, 'items'):
                    return {k: self._convert_protobuf_to_native(v) for k, v in value.items()}
                # Check if it's list-like (RepeatedComposite, RepeatedScalar)
                elif hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                    return [self._convert_protobuf_to_native(item) for item in value]
                else:
                    # Fallback to string conversion
                    return str(value)
            except Exception as e:
                logger.warning(f"Failed to convert protobuf object: {e}")
                return str(value)
        # Check if it's a list - recurse into it
        elif isinstance(value, list):
            return [self._convert_protobuf_to_native(item) for item in value]
        # Check if it's a dict - recurse into it
        elif isinstance(value, dict):
            return {k: self._convert_protobuf_to_native(v) for k, v in value.items()}
        else:
            return value

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool via HTTP request to the game server."""
        try:
            # Map tool names to server endpoints
            endpoint_map = {
                # Pokemon MCP tools
                "get_game_state": "/mcp/get_game_state",
                "press_buttons": "/mcp/press_buttons",
                # navigate_to REMOVED - no pathfinding in this version
                "add_knowledge": "/mcp/add_knowledge",
                "search_knowledge": "/mcp/search_knowledge",
                "get_knowledge_summary": "/mcp/get_knowledge_summary",
                "lookup_pokemon_info": "/mcp/lookup_pokemon_info",
                "list_wiki_sources": "/mcp/list_wiki_sources",
                "get_walkthrough": "/mcp/get_walkthrough",
                "complete_direct_objective": "/mcp/complete_direct_objective",
                "create_direct_objectives": "/mcp/create_direct_objectives",
                # "get_progress_summary": "/mcp/get_progress_summary",
                "reflect": "/mcp/reflect",
                # SLAM tools
                "save_map": "/mcp/save_map",
                "load_map": "/mcp/load_map",
            }

            endpoint = endpoint_map.get(tool_name)
            if not endpoint:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

            url = f"{self.server_url}{endpoint}"
            logger.info(f"🔧 Calling MCP tool: {tool_name}")
            logger.debug(f"   URL: {url}")

            # Convert all arguments recursively (handles nested protobuf objects)
            converted_args = {}
            args_for_log = {}
            for k, v in arguments.items():
                # Recursively convert protobuf objects to native types
                converted_value = self._convert_protobuf_to_native(v)
                converted_args[k] = converted_value

                # For logging, abbreviate large base64 data
                if k == "screenshot_base64" and isinstance(converted_value, str) and len(converted_value) > 100:
                    args_for_log[k] = f"<{len(converted_value)} bytes>"
                else:
                    args_for_log[k] = converted_value

            logger.info(f"   Args: {args_for_log}")

            # Note: The server already handles button sequences properly via action_queue.extend()
            # No need to split sequences - just send them as-is
            response = requests.post(url, json=converted_args, timeout=30)
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ Tool {tool_name} completed")

            # Special handling for get_game_state - optionally print the formatted text
            if tool_name == "get_game_state" and result.get("success") and "state_text" in result:
                if self.print_game_state:
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


class VisionOnlyAgent:
    """Vision-Only CLI Agent - uses only visual input without map data or pathfinding."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model: str = "gemini-2.5-flash",
        backend: str = "gemini",
        max_steps: Optional[int] = None,
        system_instructions_file: str = "POKEAGENT.md",
        max_context_chars: int = 100000,  # ~25k tokens for gemini-2.5-flash
        target_context_chars: int = 50000,  # Compact down to this when exceeded
        allow_walkthrough: bool = False,  # Enable get_walkthrough tool
        allow_slam: bool = False  # Enable SLAM (map building) mode
    ):
        print(f"🚀 Initializing VisionOnlyAgent with backend={backend}, model={model}, server={server_url}")
        self.server_url = server_url
        self.model = model
        self.backend = backend
        self.max_steps = max_steps
        self.step_count = 0
        self.max_context_chars = max_context_chars
        self.target_context_chars = target_context_chars
        self.allow_walkthrough = allow_walkthrough
        self.allow_slam = allow_slam

        # SLAM mode (maps handled by server endpoints)
        if self.allow_slam:
            print(f"🗺️  SLAM mode enabled - maps saved to .pokeagent_cache/maps/")

        # Conversation history for tracking and compaction
        self.conversation_history = []

        # Load system instructions
        self.system_instructions = "" #self._load_system_instructions(system_instructions_file)

        # Initialize MCP tool adapter (don't print full game state since we strip map info)
        self.mcp_adapter = MCPToolAdapter(server_url, print_game_state=False)

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
        print(f"✅ {len(self.tools)} tools available (NO MAP / NO PATHFINDING)")
        print(f"✅ System instructions loaded ({len(self.system_instructions)} chars)")

        # Initialize LLM logger
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
        """Create Gemini function declarations for MCP tools (NO NAVIGATE_TO)."""

        # Use Gemini's declaration format with proper types

        tools = [
            # ============================================================
            # POKEMON MCP TOOLS (NO NAVIGATE_TO)
            # ============================================================

            # Game Control Tools
            # {
            #     "name": "get_game_state",
            #     "description": "Get the current game state including player position, party Pokemon, items, and a screenshot. Use this to understand where you are and what you can do.",
            #     "parameters": {
            #         "type_": "OBJECT",
            #         "properties": {},
            #         "required": []
            #     }
            # },
            {
                "name": "press_buttons",
                "description": "Press Game Boy Advance buttons to interact with the game. Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R, WAIT. You can pass multiple buttons in sequence (e.g., ['A', 'A', 'A'] to press A three times).",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "buttons": {
                            "type_": "ARRAY",
                            "items": {"type_": "STRING"},
                            "description": "List of buttons to press in sequence (e.g., ['A'], ['UP', 'UP', 'UP'] for multiple presses)"
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Explain why you are pressing these buttons"
                        }
                    },
                    "required": ["buttons", "reasoning"]
                }
            },
            # NOTE: navigate_to has been REMOVED - no pathfinding in this version
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
            {
                "name": "create_direct_objectives",
                "description": "Create the next 3 direct objectives when a sequence completes. Use this after consulting get_walkthrough() or wiki sources to plan your next steps.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "objectives": {
                            "type_": "ARRAY",
                            "items": {
                                "type_": "OBJECT",
                                "properties": {
                                    "id": {"type_": "STRING", "description": "Unique identifier"},
                                    "description": {"type_": "STRING", "description": "Clear description of what to accomplish"},
                                    "action_type": {
                                        "type_": "STRING",
                                        "enum": ["navigate", "interact", "battle", "wait"],
                                        "description": "Type of action"
                                    },
                                    "target_location": {"type_": "STRING", "description": "Target location/map name"},
                                    "navigation_hint": {"type_": "STRING", "description": "Specific guidance on how to accomplish this"},
                                    "completion_condition": {"type_": "STRING", "description": "How to verify completion"}
                                },
                                "required": ["id", "description", "action_type"]
                            },
                            "description": "Array of exactly 3 objectives to create next"
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Explanation of why these objectives were chosen"
                        }
                    },
                    "required": ["objectives", "reasoning"]
                }
            },
            # {
            #     "name": "get_progress_summary",
            #     "description": "Get comprehensive progress summary including completed milestones, objectives, current location, and knowledge base summary.",
            #     "parameters": {
            #         "type_": "OBJECT",
            #         "properties": {},
            #         "required": []
            #     }
            # },

            # Knowledge & Information Tools
            {
                "name": "add_knowledge",
                "description": "Store information in the persistent knowledge base for later recall.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "category": {"type_": "STRING", "description": "Category (e.g., 'npcs', 'locations', 'items')"},
                        "key": {"type_": "STRING", "description": "Unique identifier"},
                        "value": {"type_": "STRING", "description": "Information to store"}
                    },
                    "required": ["category", "key", "value"]
                }
            },
            {
                "name": "search_knowledge",
                "description": "Search the knowledge base for previously stored information.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "query": {"type_": "STRING", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_knowledge_summary",
                "description": "Get a summary of all knowledge stored in the database.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "lookup_pokemon_info",
                "description": "Look up Pokemon information from Bulbapedia (stats, moves, evolution, locations).",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "pokemon_name": {"type_": "STRING", "description": "Pokemon name to look up"}
                    },
                    "required": ["pokemon_name"]
                }
            },
            {
                "name": "list_wiki_sources",
                "description": "List all available wiki article sources.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "reflect",
                "description": "Perform self-reflection by analyzing recent actions, current situation, and progress. Returns context for introspection including recent history, current state, objectives, and progress metrics.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "situation": {
                            "type_": "STRING",
                            "description": "Brief description of the current situation requiring reflection (e.g., 'stuck in same location', 'repeated failed attempts', 'unclear next steps')"
                        }
                    },
                    "required": ["situation"]
                }
            },
        ]

        # Conditionally add get_walkthrough if enabled
        if self.allow_walkthrough:
            tools.append({
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
            })

        # Conditionally add SLAM tools if enabled
        if self.allow_slam:
            tools.extend([
                {
                    "name": "save_map",
                    "description": "Save your mental map of the current location. The map should be a text representation showing walls (#), walkable tiles (.), your position (@), NPCs (N), items (I), and doors (D). ONLY use in overworld.",
                    "parameters": {
                        "type_": "OBJECT",
                        "properties": {
                            "location_name": {
                                "type_": "STRING",
                                "description": "Name of the location (e.g., 'Littleroot Town', 'Route 101')"
                            },
                            "map_data": {
                                "type_": "STRING",
                                "description": "ASCII map representation with legend. Include your observations about layout, obstacles, NPCs, and points of interest."
                            }
                        },
                        "required": ["location_name", "map_data"]
                    }
                },
                {
                    "name": "load_map",
                    "description": "Load your previously saved map for the current location to remember the layout and your observations.",
                    "parameters": {
                        "type_": "OBJECT",
                        "properties": {
                            "location_name": {
                                "type_": "STRING",
                                "description": "Name of the location to load map for"
                            }
                        },
                        "required": ["location_name"]
                    }
                }
            ])

        feature_flags = []
        if self.allow_walkthrough:
            feature_flags.append("+WALKTHROUGH")
        if self.allow_slam:
            feature_flags.append("+SLAM")
        flags_str = f" ({', '.join(feature_flags)})" if feature_flags else ""

        logger.info(f"✅ Created {len(tools)} tool declarations (NO PATHFINDING{flags_str})")
        return tools

    def _execute_function_call_by_name(self, function_name: str, arguments: dict) -> str:
        """Execute a function by name with given arguments and return result as JSON string."""
        # All tools (including SLAM) now go through MCP adapter for consistency
        result = self.mcp_adapter.call_tool(function_name, arguments)
        # Return as JSON string
        return json.dumps(result, indent=2)

    def _convert_protobuf_args(self, proto_args) -> dict:
        """Convert protobuf arguments to JSON-serializable Python types."""
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

        # Route through _execute_function_call_by_name to handle local SLAM tools
        return self._execute_function_call_by_name(function_name, arguments)

    def _add_to_history(self, prompt: str, response: str, tool_calls: List[Dict] = None, action_details: str = None, player_coords: tuple = None):
        """Add interaction to conversation history - ONLY stores LLM responses and actions."""
        # Strip whitespace from response to save tokens
        response_stripped = response.strip() if response else ""

        # CRITICAL SAFEGUARD: If response contains our prompt header, skip it entirely
        if "You are an expert navigator and battle strategist" in response_stripped:
            logger.warning(f"⚠️ Skipping corrupted history entry at step {self.step_count} (contains prompt echo)")
            return  # Don't store corrupted data

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

        for entry in self.conversation_history[-10:]:  # Show last 10
            try:
                lines.append(f"\nStep {entry['step']}:")
                lines.append(f"  Prompt: {entry['prompt'][:100]}...")
                if entry.get('tool_calls'):
                    lines.append(f"  Tools called: {', '.join(t['name'] for t in entry['tool_calls'])}")
                lines.append(f"  Response: {entry['response'][:100]}...")
            except:
                print("Error vision_only_agent.py formatting history")
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
            prompt: The prompt to send to the LLM
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
                # Add timeout wrapper to prevent indefinite hangs

                if screenshot_b64:

                    # Decode base64 to image
                    image_data = base64.b64decode(screenshot_b64)
                    image = PILImage.open(io.BytesIO(image_data))

                    # CRITICAL: Validate frame before any VLM processing
                    if image is None:
                        logger.error("🚫 CRITICAL: run_step called with None image - cannot proceed")
                        return False, "ERROR: No valid image provided"

                    # Validate frame is a proper image
                    if not (hasattr(image, 'save') or hasattr(image, 'shape')):
                        logger.error(f"🚫 CRITICAL: run_step called with invalid image type {type(image)} - cannot proceed")
                        return False, "ERROR: Invalid image type"

                    # Additional PIL Image validation
                    if hasattr(image, 'size'):
                        width, height = image.size
                        if width <= 0 or height <= 0:
                            logger.error(f"🚫 CRITICAL: run_step called with invalid image size {width}x{height} - cannot proceed")
                            return False, "ERROR: Invalid image dimensions"

                    # Check for black frame (transition screen)
                    if self._is_black_frame(image):
                        logger.info("⏳ Black frame detected (likely a transition), waiting for next frame...")
                        return True, "WAIT"

                    # Define function after image is created and validated
                    def call_vlm_with_image():
                        return self.vlm.get_query(image, prompt, "VisionOnlyAgent")

                    # Use VLM with image
                    timeout = 180 if 'preview' in self.model or '3-pro' in self.model else 60
                    logger.info(f"📡 Calling VLM API with image (timeout: {timeout}s)")

                    # Retry loop for timeouts
                    max_retries = 3
                    retry_count = 0
                    response = None

                    while retry_count < max_retries:
                        executor = ThreadPoolExecutor(max_workers=1)
                        future = None
                        try:
                            future = executor.submit(call_vlm_with_image)
                            response = future.result(timeout=timeout)
                            vlm_duration = time.time() - vlm_call_start
                            logger.info(f"   ✅ VLM call completed in {vlm_duration:.1f}s")
                            break
                        except FutureTimeoutError:
                            retry_count += 1
                            vlm_duration = time.time() - vlm_call_start
                            logger.error(f"   ⏱️ VLM call TIMED OUT after {vlm_duration:.1f}s (attempt {retry_count}/{max_retries})")
                            if retry_count >= max_retries:
                                raise TimeoutError(f"VLM call timed out after {max_retries} attempts")
                        finally:
                            executor.shutdown(wait=False)
                else:
                    # Define function for text-only call
                    def call_vlm_with_text():
                        return self.vlm.get_text_query(prompt, "VisionOnlyAgent")

                    # Use VLM with text only
                    timeout = 180 if 'preview' in self.model or '3-pro' in self.model else 60
                    logger.info(f"📡 Calling VLM API with text only (timeout: {timeout}s)")

                    # Retry loop for timeouts
                    max_retries = 3
                    retry_count = 0
                    response = None

                    while retry_count < max_retries:
                        executor = ThreadPoolExecutor(max_workers=1)
                        future = None
                        try:
                            future = executor.submit(call_vlm_with_text)
                            response = future.result(timeout=timeout)
                            vlm_duration = time.time() - vlm_call_start
                            logger.info(f"   ✅ VLM call completed in {vlm_duration:.1f}s")
                            break
                        except FutureTimeoutError:
                            retry_count += 1
                            vlm_duration = time.time() - vlm_call_start
                            logger.error(f"   ⏱️ VLM call TIMED OUT after {vlm_duration:.1f}s (attempt {retry_count}/{max_retries})")
                            if retry_count >= max_retries:
                                raise TimeoutError(f"VLM call timed out after {max_retries} attempts")
                        finally:
                            executor.shutdown(wait=False)

                # Determine if response is a GenerationResponse object (function calling) or string (text)
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
                traceback.print_exc()
                return False, f"VLM API error ({error_type}) after {vlm_duration:.1f}s: {error_msg[:200]}"

            # Process response - handle function calls
            tool_call_count = 0

            # For VLM backends with native function calling support
            if self.backend != "gemini":
                # Handle native function calls from VLM backends (VertexAI, etc.)
                function_calls_executed = self._handle_vertex_function_calls(
                    response, tool_calls_made, tool_call_count, max_tool_calls
                )

                if function_calls_executed:
                    duration = time.time() - start_time

                    # Extract reasoning
                    tool_reasoning = ""
                    if tool_calls_made:
                        last_tool_call = tool_calls_made[-1]
                        try:
                            if last_tool_call['name'] == "press_buttons" and "reasoning" in last_tool_call["args"]:
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
                    else:
                        action_details = f"Executed {last_tool_call['name']}"

                    self._add_to_history(prompt, full_response, tool_calls_made, action_details=action_details)

                    return True, full_response
                else:
                    # No function calls - treat as text response
                    text_content = self._extract_text_from_response(response)
                    if not text_content:
                        text_content = str(response)

                    logger.info(f"📥 Received text response from {self.backend}:")
                    logger.info(f"   {text_content}")

                    duration = time.time() - start_time
                    logger.info(f"✅ Step completed in {duration:.2f}s")

                    self._add_to_history(prompt, text_content, tool_calls=[])

                    return True, text_content
            else:
                # Original Gemini function calling logic
                part = None
                function_call = None
                function_result = None

                # Extract parts from Gemini response structure
                parts = None
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            parts = candidate.content.parts

                logger.info(f"🔍 Gemini response has {len(parts) if parts else 0} parts")

                if not parts:
                    logger.error("❌ No parts found in Gemini response")

                # Process function calls from parts
                if parts and tool_call_count < max_tool_calls:
                    # First, extract any text parts for reasoning
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            reasoning_text += part.text + "\n"

                    # Check each part for function calls
                    for part in parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            tool_call_count += 1
                            logger.info(f"🔧 Gemini wants to call: {function_call.name} ({tool_call_count}/{max_tool_calls})")

                            # Execute the function
                            function_result = self._execute_function_call(function_call)
                            logger.info(f"📥 Function result: {function_result[:200]}...")

                            # Track tool call
                            tool_calls_made.append({
                                "name": function_call.name,
                                "args": self._convert_protobuf_args(function_call.args),
                                "result": function_result
                            })

                            # Wait for actions to complete
                            self._wait_for_actions_complete()

                            # END THE STEP IMMEDIATELY
                            duration = time.time() - start_time

                            # Extract reasoning from function args (any tool)
                            tool_reasoning = ""
                            try:
                                if tool_calls_made:
                                    last_call = tool_calls_made[-1]
                                    # Extract reasoning from any tool that has it
                                    if "reasoning" in last_call["args"]:
                                        tool_reasoning = last_call["args"]["reasoning"]
                                    # For save_map, include location and map preview
                                    elif function_call.name == "save_map":
                                        loc = last_call["args"].get("location_name", "")
                                        map_preview = last_call["args"].get("map_data", "")[:200]
                                        tool_reasoning = f"Saving map for {loc}:\n{map_preview}..."
                                    # For load_map, include location
                                    elif function_call.name == "load_map":
                                        loc = last_call["args"].get("location_name", "")
                                        tool_reasoning = f"Loading map for {loc}"
                            except Exception as e:
                                logger.debug(f"Could not extract tool reasoning: {e}")

                            # Build full response
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

                            # Display reasoning
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

                            # Send thinking to server
                            thinking_to_send = display_text if display_text else full_response
                            self._send_thinking_to_server(thinking_to_send, self.step_count + 1)

                            logger.info(f"✅ Step ended after {function_call.name}")
                            return True, full_response

                # Check if we got a text response
                if part is not None and (not hasattr(part, 'function_call') or not part.function_call):
                    if parts:
                        for part in parts:
                            if hasattr(part, 'text') and part.text:
                                text_response = part.text
                                logger.info("📥 Received response from Gemini:")
                                print("\n" + "="*70)
                                print(text_response)
                                print("="*70 + "\n")

                                # Check if action tool was used
                                has_action_tool = any(
                                    call["name"] == "press_buttons"
                                    for call in tool_calls_made
                                )

                                if not has_action_tool:
                                    enforcement_retry_count += 1
                                    logger.error(f"❌ Gemini provided text but no action tool! (Retry {enforcement_retry_count}/{max_enforcement_retries})")
                                    logger.error(f"   This should not happen - the prompt explicitly requires function calls!")
                                    logger.error(f"   Text response: {text_response[:200]}...")

                                    # Force a WAIT action as last resort
                                    logger.warning("⚠️ Forcing WAIT action due to missing action tool (prompt should prevent this)")
                                    tool_calls_made.append({
                                        "name": "press_buttons",
                                        "args": {"buttons": ["WAIT"], "reasoning": f"Fallback: Model ignored function call requirement"},
                                        "result": "Forced WAIT action"
                                    })

                                    # Build response
                                    full_response = f"[ERROR: Text response instead of function call]\n{text_response[:500]}"

                                    # Calculate duration
                                    duration = time.time() - start_time

                                    # Add to history
                                    self._add_to_history(prompt, full_response, tool_calls_made)

                                    return True, full_response

                                # Calculate duration
                                duration = time.time() - start_time

                                # Add to history
                                self._add_to_history(prompt, text_response, tool_calls_made)

                                # Send thinking to server
                                self._send_thinking_to_server(text_response, self.step_count + 1)

                                return True, text_response

            # If we get here, no response was generated
            logger.warning("⚠️ No response generated")
            return False, "No response"

        except Exception as e:
            logger.error(f"❌ Error in agent step: {e}")
            traceback.print_exc()
            return False, str(e)

    def _wait_for_actions_complete(self, timeout: int = 30) -> None:
        """Wait for all queued actions to complete before proceeding."""

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

    def _build_structured_prompt(self, game_state_result: str, step_count: int) -> str:
        """Build a prompt WITHOUT map information."""

        # Parse game state
        try:
            game_state_data = json_module.loads(game_state_result)
        except:
            game_state_data = {}

        direct_objective = game_state_data.get("direct_objective", "")
        direct_objective_status = game_state_data.get("direct_objective_status", "")
        direct_objective_context = game_state_data.get("direct_objective_context", "")

        # Format objective nicely if it's a dict
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

        # Build action history
        action_history = self._format_action_history()

        # Load saved SLAM map if available and SLAM is enabled
        saved_map_info = ""
        if self.allow_slam:
            try:
                # Get current location from game state
                location = game_state_data.get("player", {}).get("location", "")
                if location and location != "Unknown":
                    # Try to load the saved map
                    from pathlib import Path
                    maps_dir = Path(".pokeagent_cache/maps")
                    # Normalize to title case for consistent filenames
                    normalized_location = location.title()
                    safe_name = "".join(c for c in normalized_location if c.isalnum() or c in (' ', '_', '-')).strip()
                    safe_name = safe_name.replace(' ', '_')
                    map_file = maps_dir / f"{safe_name}.txt"

                    if map_file.exists():
                        saved_map_content = map_file.read_text()
                        saved_map_info = f"""

╔═══════════════════════════════════════════════════════════════════════╗
║                  📍 PREVIOUSLY SAVED MAP - {location.upper()}
║═══════════════════════════════════════════════════════════════════════║

{saved_map_content}

╚═══════════════════════════════════════════════════════════════════════╝

⚠️ USE THIS MAP: Compare current frame with the saved map above.
⚠️ STITCH: Identify overlapping features and update the global map.
"""
                        logger.info(f"📍 Loaded saved map for {location} ({len(saved_map_content)} chars)")
                    else:
                        saved_map_info = f"""

📍 NO SAVED MAP for {location} - This is your first time here!
⚠️ START FRESH: Create a new map from the current frame.
"""
                        logger.info(f"📍 No saved map found for {location}")
            except Exception as e:
                logger.error(f"Error loading saved map: {e}")

        # Log sizes
        logger.info(f"📏 Prompt component sizes:")
        logger.info(f"   action_history: {len(action_history):,} chars")
        logger.info(f"   saved_map_info: {len(saved_map_info):,} chars")

        # Build prompt WITHOUT navigation/pathfinding instructions
        prompt = f"""You are an expert Pokemon player and battle strategist playing Pokémon Emerald on a Game Boy Advance emulator."""

        # Add MANDATORY SLAM check at the very top if enabled
        if self.allow_slam:
            prompt += """

🚨🚨🚨 MANDATORY SLAM CHECK (VISION-ONLY) 🚨🚨🚨

You are the **SLAM (Simultaneous Localization and Mapping) Engine** for Vision-Only navigation.

Look at the screenshot. Are you in OVERWORLD (not battle/menu)?

IF YES - MAP EVERY FRAME:

📐 **THE "PLAYER CENTER" ANCHOR (CRITICAL)**:
   - Camera is locked to player in overworld
   - Player (@) is **ALWAYS** at Row 5, Column 7 (center of 15×10 grid)
   - Do NOT search for player - PLACE @ at (5,7) FIRST
   - Map all other objects RELATIVE to this fixed center point

🗺️ **SYMBOL LEGEND** (classify by FUNCTION, not just visuals):
   @ = Player (ALWAYS at Row 5, Col 7)
   . = Walkable (grass, dirt, floor, sand)
   # = Blocked (trees, walls, rocks, furniture, map edge)
   G = Danger (tall grass with encounters)
   W = Fluid (water, lava - requires Surf)
   ~ = Shore (transition between land and water)
   N = Entity (NPCs, trainers, moving sprites)
   D = Transition (doors, cave entrances, stairs, warps)
   L = Ledge (one-way jumpable)
   X = Void (black empty space in interiors)

👁️ **SCAN PROTOCOL**:
   1. Anchor @ to (5,7)
   2. Inspect OUTER RING (Rows 0, 9 and Cols 0, 14) for boundaries
   3. Identify features relative to (5,7) anchor
   4. Respect actual geometry - don't draw straight lines unless visible

⚠️ SAVE FREQUENTLY - Every 2-3 movements!
⚠️ Call save_map() with the 15×10 grid (player at center)

════════════════════════════════════════════════════════════════"""

        prompt += f"""

ACTION HISTORY (last steps):
{action_history}
{saved_map_info}

================================================================================
🎯🎯🎯 CURRENT DIRECT OBJECTIVE - READ THIS CAREFULLY 🎯🎯🎯
================================================================================

{direct_objective_context}

{direct_objective}

{direct_objective_status}

================================================================================
⚠️ CRITICAL: When you have completed the objective above, you MUST call:
   complete_direct_objective(reasoning="<explain why it's complete>")
================================================================================

**DIALOGUE CHECK**: If you see a dialogue box with text, press_buttons(["A"], reasoning).

AVAILABLE TOOLS:

🎮 **PRIMARY GAME TOOLS**:
- complete_direct_objective(reasoning) - Mark current objective as complete
- press_buttons(buttons, reasoning) - Press GBA buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R, WAIT
  * You can pass multiple buttons for sequences: ['UP', 'UP', 'UP'] presses UP three times
  * Each button in the sequence will be sent as a separate action to the game

🎯 **OBJECTIVE MANAGEMENT**:
- create_direct_objectives(objectives, reasoning) - Create next 3 objectives when sequences complete

** MOVEMENT **:
- Use press_buttons with directional buttons: UP, DOWN, LEFT, RIGHT
- Pressing LEFT moves you west, RIGHT moves you east
- Pressing UP moves you north, DOWN moves you south
- You can chain multiple movements: press_buttons(['UP', 'UP', 'RIGHT'], reasoning)

** INTERACTION TIPS **:
- To interact with an object or NPC, you must be adjacent and facing them
- Press A to interact, B to cancel, START for menu

"""

        # Add SLAM to strategy if enabled
        if self.allow_slam:
            prompt += """
STRATEGY - PRIORITY ORDER:
1. **CHECK OBJECTIVE COMPLETION FIRST**: If your objective is complete, call complete_direct_objective()
2. **🗺️ MAP CHECK (SLAM MODE)**:
   - In overworld? Check: load_map(location_name) to see if you have a map
   - After moving 5+ times? Save: save_map(location_name, map_data)
3. **DIALOGUE**: If you see a dialogue box, press A to advance it
4. **MOVEMENT**: Use press_buttons with directional buttons to navigate
5. **BATTLES**: Select moves carefully, heal when below 50% HP
6. **STUCK**: If stuck, try different directions or look for alternative routes"""
        else:
            prompt += """
STRATEGY - PRIORITY ORDER:
1. **CHECK OBJECTIVE COMPLETION FIRST**: If your objective is complete, call complete_direct_objective()
2. **DIALOGUE SECOND**: If you see a dialogue box, press A to advance it
3. **MOVEMENT**: Use press_buttons with directional buttons to navigate
4. **BATTLES**: Select moves carefully, heal when below 50% HP
5. **STUCK**: If stuck, try different directions or look for alternative routes"""

        prompt += """

🔴 **REMEMBER**: Call complete_direct_objective() as soon as you complete the current objective!"""

        # Add SLAM instructions if enabled - make them PROMINENT
        if self.allow_slam:
            prompt += """

╔═══════════════════════════════════════════════════════════════════════╗
║                     🗺️ SLAM MODE ACTIVE 🗺️                            ║
║         SIMULTANEOUS LOCALIZATION AND MAPPING REQUIRED!               ║
╚═══════════════════════════════════════════════════════════════════════╝

🚨 **SLAM WORKFLOW - MAP & STITCH EVERY FRAME**:

**1. ANCHOR**: Player (@) is ALWAYS at Row 5, Column 7 in CURRENT viewport
   - Camera is locked to player
   - Do NOT search for player position in current frame
   - PLACE @ at (5,7) FIRST, then map everything else relative to it

**2. SCAN**: Inspect the current 15×10 frame
   - Check OUTER RING first (Rows 0, 9 and Cols 0, 14) for boundaries
   - Identify features relative to (5,7) anchor
   - Classify by FUNCTION: walkable, blocked, danger, entities, transitions
   - Respect actual geometry - no straight lines unless visible

**3. STITCH**: Maintain a persistent global map
   - Load previous map with load_map(location_name)
   - Compare NEW frame vs PREVIOUS map
   - Identify OVERLAPPING features (tree patterns, shores, paths) to align
   - ADD new data from current frame to global map
   - CORRECT old data if new visual info contradicts it
   - Update player position in global coordinates (not always at 5,7!)

**4. SAVE**: Call save_map() every 2-3 movements
   - Save the FULL STITCHED GLOBAL MAP (not just current 15×10)
   - Player position in global map is DYNAMIC (moves as you explore)
   - Map grows larger than 15×10 as you explore
   - Include alignment notes (e.g., "Matched trees at Row 2")

**YOUR MAP TOOLS**:
- load_map(location_name) - Get previous global map for stitching
- save_map(location_name, map_data) - Save expanded global map

**MAP FORMAT - CRITICAL**:

🔴 **CURRENT VIEWPORT**: Always 15 columns × 10 rows (player at 5,7)
🔴 **GLOBAL MAP**: Grows larger as you explore (20×15, 30×20, etc.)
🔴 **PLAYER IN GLOBAL**: Dynamic position, moves with exploration
🔴 **TILE SIZE**: Each tile is 16×16 pixels in 240×160 frame

**REQUIRED OUTPUT FORMAT** (Global Stitched Map):
```
save_map(
    location_name="<Location>",
    map_data='''<Location> (Global)
...............   <- Rows from previous exploration
...............
..GGGG##WWWWWWW   <- Current viewport starts here
..GGGG##WWWWWWW
..GGGG##WWWWWWW
..GGGG##~WWWWWW
..GGGG##~WWWWWW
..GGG.@#~NWWWWW   <- Player @ (Global Row X, Col Y)
...GG.##~WWWWWW
...GG.###WWWWWW
....G.###WWWWWW
.....####WWWWWW   <- Current viewport ends here
...............   <- Rows from previous exploration
...............
''' # Map grows as you explore
)
```

**EXAMPLE - Route 103 (After exploring 3 moves south)**:
```
# [ANALYSIS]: Moved south 3 tiles. Matched shore pattern and tree line.
# [COORDINATE CHECK]: Player now @ Global Row 8, Col 7

save_map(
    location_name="Route 103",
    map_data='''Route 103 (Global Map - Explored)
..GGGG##WWWWWWW   <- Row 0 (From previous exploration)
..GGGG##WWWWWWW   <- Row 1
..GGGG##WWWWWWW   <- Row 2
..GGGG##~WWWWWW   <- Row 3
..GGGG##~WWWWWW   <- Row 4
..GGG..#~NWWWWW   <- Row 5 (NPC was here)
...GG..##WWWWWW   <- Row 6 -- Current viewport starts here
...GG.###WWWWWW   <- Row 7
....G.###WWWWWW   <- Row 8
.....####WWWWWW   <- Row 9
......@##WWWWWW   <- Row 10 (Player NOW here - Global pos)
.......##WWWWWW   <- Row 11
.......##WWWWWW   <- Row 12
......###WWWWWW   <- Row 13
.....####WWWWWW   <- Row 14
....#####WWWWWW   <- Row 15 -- Current viewport ends here

Legend:
@ = Player (Global Row 10, Col 7)
. = Walkable path
G = Tall grass (danger)
# = Trees (blocked)
~ = Shore
W = Water (need Surf)
N = NPC/Trainer (stayed at Row 5, Col 10)

Map grew from 15x10 to 15x16 after moving south!
'''
)

# SUGGESTED ACTION: Continue south to explore more
```

**INTEGRATION WITH ANALYSIS**:
```
ANALYSIS: Loading map... I've explored the northern area (Rows 0-5).
Current viewport shows I'm at Global Row 10. Looking at saved map:
- Water continues east (columns 11-15)
- Path opens south (current direction)
- NPC remains at Row 5, Col 10 (north of me now)
PLAN: Continue south to map unexplored area, avoid tall grass to west.
```

**🔄 STITCHING TIPS**:
- Use DISTINCTIVE FEATURES to align: shore lines, tree clusters, NPC positions
- When moving NORTH: Add rows ABOVE previous map (rows 0, -1, -2...)
- When moving SOUTH: Add rows BELOW previous map (rows N+1, N+2...)
- When moving EAST/WEST: Expand columns accordingly
- If features don't match: Trust NEW visual data over old memory

**SLAM is part of your THINKING** - no additional VLM calls needed!"""

        prompt += f"""

═══════════════════════════════════════════════════════════════════════
🧠 REASONING STRUCTURE (VISION-ONLY - NO MAP/COORDINATES)
═══════════════════════════════════════════════════════════════════════

🔴 **CRITICAL**: Describe what you SEE in the screenshot - you have NO MAP!

Process: ANALYZE visuals → PLAN action → EXECUTE with detailed reasoning

**✅ GOOD** (describes visuals):
press_buttons(
    buttons=["UP", "UP"],
    reasoning='''ANALYSIS: I see buildings to the north and a clear path leading upward. No visible obstacles or NPCs blocking. Objective: Find Pokemon Lab.
PLAN: Move UP twice toward buildings. Path appears open.
ACTION: Pressing UP twice along visible path.'''
)

**❌ BAD** (no visual analysis):
press_buttons(buttons=["UP"], reasoning="Moving north")

**❌ BAD** (no tool executed):
PLAN: Move UP twice toward buildings. Path appears open.
ACTION: Pressing UP twice along visible path.

═══════════════════════════════════════════════════════════════════════
🚨 RESPONSE FORMAT - CRITICAL 🚨
═══════════════════════════════════════════════════════════════════════

Your response should follow this TWO-PART structure:

**PART 1 - Your Analysis (TEXT OUTPUT)**:
Write your ANALYSIS and PLAN as text:
```
ANALYSIS: What I see in the screenshot...
PLAN: What I'm going to do...
```

**PART 2 - Function Call (REQUIRED)**:
After your text, you MUST **ACTUALLY CALL** the function using the function calling system.

🔴 **CRITICAL - DO NOT WRITE THIS**:
❌ BAD: "ACTION: press_buttons(buttons=['A'], reasoning='...')"  ← This is TEXT, not a function call!
❌ BAD: Writing what the function call would look like

🔴 **CRITICAL - DO THIS INSTEAD**:
✅ GOOD: Actually invoke the press_buttons function through the function calling mechanism
✅ GOOD: The function should appear in your response as a proper function call, not as text

**Every response MUST**:
1. ✅ Text: ANALYSIS and PLAN sections
2. ✅ Actual function call: Use the function calling system to invoke press_buttons, complete_direct_objective, etc.
3. ❌ DO NOT write text that looks like a function call - ACTUALLY CALL the function

If you write "ACTION: press_buttons(...)" as text, you did it WRONG. Use the actual function calling mechanism.

Think step-by-step through ANALYZE → PLAN → **CALL FUNCTION** (don't write about calling it - actually call it).

Step {step_count}"""

        logger.info(f"📏 Final prompt: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")
        return prompt

    def _is_black_frame(self, image) -> bool:
        """Check if frame is a black screen (transition)"""
        try:
            # Convert PIL Image to numpy array if needed
            if hasattr(image, 'save'):
                frame_array = np.array(image)
            else:
                frame_array = image

            # Calculate mean brightness
            mean_brightness = frame_array.mean()

            # If mean brightness is very low, it's likely a black frame
            threshold = 10
            is_black = mean_brightness < threshold

            if is_black:
                logger.debug(f"Black frame detected: mean brightness = {mean_brightness:.2f} < {threshold}")

            return is_black
        except Exception as e:
            logger.warning(f"Error checking for black frame: {e}")
            return False

    def _format_action_history(self) -> str:
        """Format action history - shows only LLM thinking and actions taken."""
        if not self.conversation_history:
            return "No previous actions recorded."

        recent_entries = self.conversation_history[-10:]
        history_lines = []

        for entry in recent_entries:
            step = entry.get("step", "?")
            llm_response = entry.get("llm_response", "").strip()
            action_details = entry.get("action_details", "").strip()

            if llm_response or action_details:
                history_lines.append(f"[Step {step}]:")
                if llm_response:
                    history_lines.append(f"  {llm_response}")
                if action_details:
                    history_lines.append(f"  → {action_details}")
                history_lines.append("")

        return "\n".join(history_lines).strip()

    def run(self) -> int:
        """Run the agent loop."""
        # Clear conversation history
        self.conversation_history = []
        logger.info("🧹 Cleared conversation history (fresh start)")

        logger.info("=" * 70)
        logger.info("🎮 Pokemon Emerald Vision-Only Agent")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model}")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Tools: {len(self.tools)} (NO PATHFINDING)")
        if self.max_steps:
            logger.info(f"Max Steps: {self.max_steps}")
        logger.info("=" * 70)

        # Check prerequisites
        logger.info("\n🔍 Checking prerequisites...")
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return 1

        logger.info("\n🚀 Starting agent loop...")
        logger.info("🚫 NO MAP INFO | NO PATHFINDING")
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
                logger.info(f"{'='*70}")

                # Fetch game state
                game_state_result = self._execute_function_call_by_name("get_game_state", {})

                # Extract screenshot
                try:
                    game_state_data = json_module.loads(game_state_result)
                    screenshot_b64 = game_state_data.get("screenshot_base64")
                except:
                    screenshot_b64 = None
                
                if screenshot_b64 is None:
                    logger.warning("⚠️ Step failed, waiting 5 seconds before retry...")
                    time.sleep(5)
                    continue

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
                            logger.info(f"💾 Checkpoint saved at step {self.step_count}")
                except requests.exceptions.RequestException as e:
                    logger.debug(f"⚠️ Checkpoint save error: {e}")

                # Brief pause
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n\n🛑 Agent stopped by user")
            return 0
        except Exception as e:
            logger.error(f"\n❌ Fatal error: {e}")
            traceback.print_exc()
            return 1

        logger.info(f"\n🎯 Agent completed {self.step_count} steps")
        return 0

    def _handle_vertex_function_calls(self, response, tool_calls_made, tool_call_count, max_tool_calls):
        """Handle function calls from VertexAI backend"""
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
                logger.info(f"🔧 VLM wants to call: {function_call.name}")

                # Execute the function
                function_result = self._execute_function_call(function_call)
                logger.info(f"📥 Function result: {str(function_result)[:200]}...")

                # Track tool call
                tool_calls_made.append({
                    "name": function_call.name,
                    "args": self._convert_protobuf_args(function_call.args),
                    "result": function_result
                })

                # Wait for actions
                if function_call.name == "press_buttons":
                    self._wait_for_actions_complete()

                function_calls_found = True

        return function_calls_found and len(tool_calls_made) > 0

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

    parser = argparse.ArgumentParser(description="Pokemon Emerald Vision-Only Agent")
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
        help="VLM backend (gemini, vertex, etc.)"
    )

    args = parser.parse_args()

    agent = VisionOnlyAgent(
        server_url=args.server_url,
        model=args.model,
        backend=args.backend,
        max_steps=args.max_steps,
        system_instructions_file=args.system_instructions
    )

    return agent.run()


if __name__ == "__main__":
    sys.exit(main())

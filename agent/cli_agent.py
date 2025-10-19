#!/usr/bin/env python3
"""
CLI Agent for Pokemon Emerald
Uses Gemini API directly with MCP tools exposed as function declarations.
Supports both Pokemon MCP tools AND baseline file/shell/web tools.
Maintains conversation history with automatic compaction over time.
"""

import os
import sys
import time
import json
import logging
import re
import traceback
import requests
import io
import base64
from pathlib import Path
from typing import Optional, Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai
import google.generativeai.types as genai_types
import PIL.Image as PILImage

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
                # print(result["state_text"])
                # print(self._truncate_json_map_for_console(result["state_text"]))
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


class CLIAgent:
    """CLI Agent using Gemini API directly with MCP tools."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model: str = "gemini-2.5-flash",
        max_steps: Optional[int] = None,
        system_instructions_file: str = "POKEAGENT.md",
        max_context_chars: int = 800000,  # ~200k tokens for gemini-2.5-flash (1M token limit)
        target_context_chars: int = 400000,  # Compact down to this when exceeded
        include_story_objectives: bool = False  # Toggle for storyline objectives
    ):
        print(f"🚀 Initializing CLIAgent with model={model}, server={server_url}")
        self.server_url = server_url
        self.model = model
        self.max_steps = max_steps
        self.step_count = 0
        self.max_context_chars = max_context_chars
        self.target_context_chars = target_context_chars
        self.include_story_objectives = include_story_objectives

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

        # Initialize LLM logger
        self.llm_logger = get_llm_logger()

        # Gemini chat history (manually managed for better control)
        # Each entry: {"role": "user"/"model", "parts": [...]}
        self.gemini_history = []

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

    def _truncate_json_map_for_console(self, state_text: str, max_tiles: int = 5) -> str:
        """Truncate JSON map tiles for console display only.

        This only affects console printing - the LLM still receives the full map.
        """
        # Find JSON map sections
        json_map_pattern = r'(--- MAP DATA \(JSON.*?\) ---\n)(\{[\s\S]*?\n\})'

        def truncate_tiles(match):
            header = match.group(1)
            json_str = match.group(2)

            try:
                map_data = json.loads(json_str)
                tiles = map_data.get('tiles', [])

                if len(tiles) > max_tiles:
                    # Keep first max_tiles tiles
                    truncated_tiles = tiles[:max_tiles]
                    map_data['tiles'] = truncated_tiles

                    # Add truncation note
                    truncated_json = json.dumps(map_data, indent=2)
                    truncated_json = truncated_json.rstrip('\n}') + f',\n  "_truncated": "Showing {max_tiles}/{len(tiles)} tiles (console display only)"\n}}'

                    return header + truncated_json
                else:
                    return match.group(0)
            except:
                # If parsing fails, return original
                return match.group(0)

        return re.sub(json_map_pattern, truncate_tiles, state_text)

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
        """Calculate total character count of conversation history.

        NOTE: This is an approximation. Actual token count is ~4x characters for Gemini.
        Gemini 2.5 Flash has a 1,048,576 input token limit (1M tokens).
        """
        total_chars = 0
        for entry in self.conversation_history:
            total_chars += len(entry.get("prompt", ""))
            total_chars += len(entry.get("response", ""))
            # Also count tool call strings
            for tool_call in entry.get("tool_calls", []):
                total_chars += len(str(tool_call))
        return total_chars

    def _compact_history(self):
        """Compact conversation history using LLM to create intelligent summaries.

        CRITICAL: This uses the LLM to summarize old conversation turns, then recreates
        the Gemini chat session with the compacted history. This is similar to how
        Claude Code handles context window management.
        """
        current_size = self._calculate_context_size()

        if current_size <= self.target_context_chars:
            return

        logger.info(f"📚 Compacting history: {current_size:,} chars → target {self.target_context_chars:,} chars")

        # Determine how many old entries to summarize
        # Keep the most recent entries intact, summarize the older ones
        total_entries = len(self.conversation_history)

        # Keep at least the last 5 turns intact for immediate context
        keep_recent = min(5, total_entries)

        # Summarize everything older than the recent entries
        entries_to_summarize = self.conversation_history[:-keep_recent] if keep_recent < total_entries else []
        recent_entries = self.conversation_history[-keep_recent:] if keep_recent > 0 else []

        if not entries_to_summarize:
            logger.info(f"   No old entries to summarize (only {total_entries} turns)")
            return

        logger.info(f"   Summarizing {len(entries_to_summarize)} old turns, keeping {len(recent_entries)} recent turns intact")

        # Build text to summarize
        history_text = ""
        for entry in entries_to_summarize:
            history_text += f"\n--- Turn {entry['step']} ---\n"
            history_text += f"User: {entry['prompt'][:500]}...\n"
            history_text += f"Assistant: {entry['response'][:500]}...\n"
            if entry.get('tool_calls'):
                tools = ', '.join(t['name'] for t in entry['tool_calls'])
                history_text += f"Tools used: {tools}\n"

        # Ask LLM to create a concise summary
        summary_prompt = f"""Please create a concise summary of this Pokemon Emerald gameplay session history.
Focus on:
- Key progress made (locations visited, Pokemon caught, battles won, items obtained)
- Current objectives and goals
- Important context needed to continue playing

Keep the summary under 500 words.

History to summarize:
{history_text}
"""

        try:
            logger.info(f"   Asking LLM to summarize {len(entries_to_summarize)} turns...")

            # Create a temporary chat for summarization (don't use main chat)
            temp_chat = self.gemini_model.start_chat(history=[])
            summary_response = temp_chat.send_message(summary_prompt)
            summary = summary_response.text

            logger.info(f"   ✅ Generated summary: {len(summary)} chars")
            logger.info(f"   Summary preview: {summary[:200]}...")

            # Replace old entries with a single summarized entry
            self.conversation_history = [{
                "step": 0,
                "prompt": "Previous session context",
                "response": f"**SUMMARY OF PREVIOUS GAMEPLAY:**\n\n{summary}",
                "tool_calls": [],
                "timestamp": time.time()
            }] + recent_entries

        except Exception as e:
            logger.error(f"   ❌ Failed to generate summary: {e}")
            logger.info(f"   Falling back to simple truncation")

            # Fallback: Just keep recent entries
            self.conversation_history = recent_entries

        new_size = self._calculate_context_size()
        logger.info(f"   New history size: {new_size:,} chars ({len(self.conversation_history)} turns)")

        # Update gemini_history to match compacted conversation_history
        logger.info(f"🔄 Updating Gemini history with compacted data...")

        self.gemini_history = []
        for entry in self.conversation_history:
            # Add user message
            self.gemini_history.append({
                "role": "user",
                "parts": [{"text": entry["prompt"]}]
            })
            # Add model response
            response_text = entry["response"] if entry["response"] else "[Tool execution only]"
            self.gemini_history.append({
                "role": "model",
                "parts": [{"text": response_text}]
            })

        logger.info(f"✅ Gemini history updated with {len(self.gemini_history)} messages ({len(self.conversation_history)} turns)")

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

    def _load_checkpoint(self):
        """Load conversation history from checkpoint file."""
        checkpoint_file = ".pokeagent_cache/checkpoint_llm.txt"

        if not os.path.exists(checkpoint_file):
            logger.warning(f"   ⚠️ No checkpoint file found at {checkpoint_file}")
            return

        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Restore step counter
            if 'step_counter' in checkpoint_data:
                self.step_count = checkpoint_data['step_counter']
                logger.info(f"   ✅ Restored step counter: {self.step_count}")

            # Restore conversation history
            if 'history' in checkpoint_data:
                # Convert checkpoint history to our conversation_history format
                self.conversation_history = checkpoint_data['history']
                logger.info(f"   ✅ Restored conversation history: {len(self.conversation_history)} turns")

                # Rebuild Gemini history for API calls
                self.gemini_history = []
                for entry in self.conversation_history:
                    # Add user message
                    self.gemini_history.append({
                        "role": "user",
                        "parts": [{"text": entry["prompt"]}]
                    })
                    # Add model response
                    response_text = entry["response"] if entry["response"] else "[Tool execution only]"
                    self.gemini_history.append({
                        "role": "model",
                        "parts": [{"text": response_text}]
                    })

                logger.info(f"   ✅ Restored Gemini history with {len(self.gemini_history)} messages")

            logger.info(f"✅ Checkpoint loaded from {checkpoint_file}")

        except Exception as e:
            logger.error(f"   ❌ Failed to load checkpoint: {e}")
            traceback.print_exc()

    def _strip_json_and_reminders_from_history_text(self, text: str) -> str:
        """Strip JSON maps and format_reminders from old history text to save tokens.

        Keeps:
        - Player info, location, party, items
        - VLM analysis and reasoning
        - Tool call results

        Removes:
        - JSON map data (within {})
        - format_reminder sections
        - scene_directions
        """
        # Remove everything between CRITICAL STUCK DETECTION and Please think step by step
        # This removes format_reminder and scene_directions
        import re
        text = re.sub(
            r'CRITICAL STUCK DETECTION:.*?(?=\n\n|$)',
            '',
            text,
            flags=re.DOTALL
        )
        text = re.sub(
            r'Please think step by step before choosing your action\..*?(?=\n\n|$)',
            '',
            text,
            flags=re.DOTALL
        )
        text = re.sub(
            r'ANALYSIS:.*?REASONING:.*?ACTION:',
            '',
            text,
            flags=re.DOTALL
        )

        # Remove JSON map data
        text = self._strip_json_from_state(text)

        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    def _strip_json_from_state(self, state_text: str) -> str:
        """Strip JSON map data (within {}) from state text for console display.

        The JSON map is useful for LLM but too verbose for console.
        """
        # Remove JSON objects (matching balanced braces)
        # This regex finds {...} blocks with proper nesting
        result = []
        brace_depth = 0
        in_json = False

        for char in state_text:
            if char == '{':
                if brace_depth == 0:
                    in_json = True
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    in_json = False
                continue

            if not in_json and brace_depth == 0:
                result.append(char)

        # Clean up extra whitespace
        cleaned = ''.join(result)
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Remove triple+ newlines

        return cleaned.strip()

    def _save_checkpoint(self):
        """Save conversation history to checkpoint file."""
        try:
            os.makedirs(".pokeagent_cache", exist_ok=True)
            checkpoint_file = ".pokeagent_cache/checkpoint_llm.txt"

            checkpoint_data = {
                "step_counter": self.step_count,
                "history": self.conversation_history,
                "timestamp": time.time()
            }

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.info(f"💾 Checkpoint saved to {checkpoint_file}")
            logger.info(f"   Steps: {self.step_count}")
            logger.info(f"   History: {len(self.conversation_history)} turns")

        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            traceback.print_exc()

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
            # Calculate total context being sent
            total_prompt_chars = len(prompt)
            approx_prompt_tokens = total_prompt_chars // 4

            logger.info(f"📤 Sending prompt to Gemini...")
            logger.info(f"   Model: {self.model}")
            logger.info(f"   Prompt: {total_prompt_chars:,} chars (~{approx_prompt_tokens:,} tokens)")
            if screenshot_b64:
                logger.info(f"   Screenshot: {len(screenshot_b64)} bytes")

            # Log context window usage
            context_usage_pct = (approx_prompt_tokens / 250000) * 100  # 1M token limit for gemini-2.5
            logger.info(f"   Context usage: ~{context_usage_pct:.1f}% of 1M token limit")

            tool_calls_made = []
            reasoning_text = ""  # Capture any text reasoning before tool calls
            enforcement_retry_count = 0  # Track how many times we've tried to enforce action tool
            max_enforcement_retries = 3

            # Track duration
            start_time = time.time()

            # Strip JSON maps and reminders from history to save tokens
            cleaned_history = []
            for idx, entry in enumerate(self.gemini_history):
                if entry["role"] == "user":
                    # For all user messages except the current one, strip JSON/reminders
                    cleaned_entry = {"role": "user", "parts": []}
                    for part in entry["parts"]:
                        if "text" in part:
                            # Strip JSON and reminders from old prompts
                            cleaned_text = self._strip_json_and_reminders_from_history_text(part["text"])
                            if cleaned_text:  # Only add if there's content left
                                cleaned_entry["parts"].append({"text": cleaned_text})
                        else:
                            # Keep images as-is
                            cleaned_entry["parts"].append(part)

                    if cleaned_entry["parts"]:  # Only add if there are parts
                        cleaned_history.append(cleaned_entry)
                else:
                    # Keep model responses as-is
                    cleaned_history.append(entry)

            # Build message content with optional image
            user_parts = []
            if screenshot_b64:
                # Decode base64 to image
                image_data = base64.b64decode(screenshot_b64)
                image = PILImage.open(io.BytesIO(image_data))
                user_parts = [prompt, image]
            else:
                user_parts = [prompt]

            # Create chat session with cleaned history
            chat = self.gemini_model.start_chat(history=cleaned_history)

            # Send current message
            response = chat.send_message(user_parts)

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

                        # Log to LLM logger (stream endpoint reads from llm_logs)
                        self._log_thinking(prompt, full_response, duration, tool_calls_made)

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
                            response = chat.send_message(enforcement_msg)
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

                    # Add to conversation history (for tracking)
                    self._add_to_history(prompt, text_response, tool_calls_made)

                    # Add to Gemini history (for next API call with cleaned history)
                    # Store the FULL prompt (with JSON maps) so we can strip it later
                    user_message = {"role": "user", "parts": []}
                    if screenshot_b64:
                        user_message["parts"] = [{"text": prompt}, {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}]
                    else:
                        user_message["parts"] = [{"text": prompt}]

                    model_message = {"role": "model", "parts": [{"text": text_response}]}

                    self.gemini_history.append(user_message)
                    self.gemini_history.append(model_message)

                    # Log to LLM logger (stream endpoint reads from llm_logs)
                    self._log_thinking(prompt, text_response, duration, tool_calls_made)

                    return True, text_response
                else:
                    # Unknown part type, skip it
                    logger.warning(f"⚠️ Unknown part type: {part}")
                    break

            # If we get here, no text response was generated
            logger.warning("⚠️ No text response from Gemini")
            return False, "No response"

        except genai_types.StopCandidateException as e:
            # Handle MALFORMED_FUNCTION_CALL and other stop reasons
            logger.error(f"❌ Gemini stopped generation: {e}")

            if "MALFORMED_FUNCTION_CALL" in str(e):
                logger.error("🔧 Gemini tried to call a function but formatted it incorrectly")
                logger.error("   This usually means the model is confused about tool parameters")
                logger.error("   Treating as a failed step - will retry next iteration")

                # Return a message asking the agent to try again
                error_msg = "Error: Function call was malformed. Please try again."
                return False, error_msg
            else:
                logger.error(f"   Stop reason: {e}")
                traceback.print_exc()
                return False, f"Generation stopped: {e}"

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
                interaction_type="gemini_cli",
                prompt=prompt,
                response=response,
                duration=duration,
                metadata={"tool_calls": tool_calls or []},
                model_info={"model": self.model}
            )
            logger.debug("✅ Logged to LLM logger")
        except Exception as e:
            logger.debug(f"Could not log to LLM logger: {e}")

    def _get_story_objectives(self) -> str:
        """Fetch and format story objectives from the server (if enabled)."""
        if not self.include_story_objectives:
            return ""

        try:
            # Get milestones from server
            response = requests.get(f"{self.server_url}/milestones", timeout=5)
            if response.status_code != 200:
                return ""

            milestones_data = response.json()
            milestones = milestones_data.get("milestones", {})

            if not milestones:
                return ""

            # Format storyline objectives
            lines = ["\n=== STORY OBJECTIVES ==="]
            lines.append("Main storyline progression (auto-verified after completed; must be completed in-order):")
            lines.append("")

            # Define storyline objectives matching simple.py
            storyline_objectives = [
                {"id": "GAME_RUNNING", "desc": "Start the game"},
                {"id": "INTRO_CUTSCENE_COMPLETE", "desc": "Complete intro cutscene"},
                {"id": "PLAYER_HOUSE_ENTERED", "desc": "Enter player's house"},
                {"id": "PLAYER_BEDROOM", "desc": "Enter player's bedroom"},
                {"id": "CLOCK_SET", "desc": "Set the clock"},
                {"id": "RIVAL_HOUSE", "desc": "Visit rival's house"},
                {"id": "RIVAL_BEDROOM", "desc": "Visit rival's bedroom"},
                {"id": "ROUTE_101", "desc": "Reach Route 101"},
                {"id": "STARTER_CHOSEN", "desc": "Choose starter Pokemon"},
                {"id": "BIRCH_LAB_VISITED", "desc": "Visit Professor Birch's lab"},
                {"id": "OLDALE_TOWN", "desc": "Reach Oldale Town"},
                {"id": "ROUTE_103", "desc": "Reach Route 103"},
                {"id": "RECEIVED_POKEDEX", "desc": "Receive Pokedex"},
                {"id": "ROUTE_102", "desc": "Reach Route 102"},
                {"id": "PETALBURG_CITY", "desc": "Reach Petalburg City"},
                {"id": "DAD_FIRST_MEETING", "desc": "Meet Dad at Petalburg Gym"},
                {"id": "GYM_EXPLANATION", "desc": "Learn about gyms"},
                {"id": "ROUTE_104_SOUTH", "desc": "Reach Route 104 (South)"},
                {"id": "PETALBURG_WOODS", "desc": "Enter Petalburg Woods"},
                {"id": "TEAM_AQUA_GRUNT_DEFEATED", "desc": "Defeat Team Aqua Grunt"},
                {"id": "ROUTE_104_NORTH", "desc": "Reach Route 104 (North)"},
                {"id": "RUSTBORO_CITY", "desc": "Reach Rustboro City"},
                {"id": "RUSTBORO_GYM_ENTERED", "desc": "Enter Rustboro Gym"},
                {"id": "ROXANNE_DEFEATED", "desc": "Defeat Roxanne"},
                {"id": "FIRST_GYM_COMPLETE", "desc": "Complete first gym"},
            ]

            completed_count = 0
            next_objective_idx = None

            # Find which objectives are completed and which is next
            for idx, obj in enumerate(storyline_objectives):
                milestone = milestones.get(obj["id"], {})
                is_completed = milestone.get("completed", False)

                if is_completed:
                    completed_count += 1
                elif next_objective_idx is None:
                    next_objective_idx = idx

            # Show recently completed (last 3) and upcoming (next 5)
            lines.append(f"Progress: {completed_count}/{len(storyline_objectives)} milestones completed\n")

            # Show recently completed objectives
            if completed_count > 0:
                lines.append("Recently Completed:")
                start_idx = max(0, completed_count - 3)
                for idx in range(start_idx, completed_count):
                    obj = storyline_objectives[idx]
                    milestone = milestones.get(obj["id"], {})
                    split_time = milestone.get("split_formatted", "??:??:??")
                    lines.append(f"  ✓ {obj['desc']} ({split_time})")
                lines.append("")

            # Highlight next objective
            if next_objective_idx is not None:
                lines.append("🎯 NEXT OBJECTIVE:")
                obj = storyline_objectives[next_objective_idx]
                lines.append(f"  ➡️  {obj['desc']}")
                lines.append("")

                # Show upcoming objectives (next 4 after the current one)
                # lines.append("Upcoming Objectives:")
                # for idx in range(next_objective_idx + 1, min(next_objective_idx + 5, len(storyline_objectives))):
                #     obj = storyline_objectives[idx]
                #     lines.append(f"  ○ {obj['desc']}")
            else:
                lines.append("🎉 All story objectives completed!")

            lines.append("\nNOTE: Story objectives auto-complete via emulator verification. Focus on game progression!")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Failed to fetch story objectives: {e}")
            return ""

    def run(self) -> int:
        """Run the agent loop."""
        logger.info("=" * 70)
        logger.info("🎮 Pokemon Emerald CLI Agent (Gemini API)")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model}")
        logger.info(f"Server: {self.server_url}")
        logger.info(f"Tools: {len(self.tools)} MCP tools (9 Pokemon + 11 Baseline)")
        logger.info(f"Context: Max {self.max_context_chars:,} chars (compact to {self.target_context_chars:,})")
        logger.info(f"Story Objectives: {'Enabled' if self.include_story_objectives else 'Disabled'}")
        if self.max_steps:
            logger.info(f"Max Steps: {self.max_steps}")
        logger.info("=" * 70)

        # Check prerequisites
        logger.info("\n🔍 Checking prerequisites...")
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return 1

        # Load checkpoint if LOAD_CHECKPOINT_MODE is set
        if os.environ.get("LOAD_CHECKPOINT_MODE") == "true":
            logger.info("\n🔄 Loading checkpoint...")
            self._load_checkpoint()

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
                # Calculate approximate token count (rough estimate: 1 token ≈ 4 chars)
                approx_tokens = context_size // 4
                logger.info(f"📚 History: {len(self.conversation_history)} turns ({context_size:,} chars / ~{approx_tokens:,} tokens)")
                logger.info(f"{'='*70}")

                # Automatically fetch game state at the beginning of each step
                game_state_result = self._execute_function_call_by_name("get_game_state", {})

                # Parse game state result to extract state_text and screenshot
                try:
                    game_state_data = json.loads(game_state_result)
                    screenshot_b64 = game_state_data.get("screenshot_base64")
                    state_text = game_state_data.get("state_text", game_state_result)
                    location = game_state_data.get("location")
                    debug_text = game_state_data.get("debug")

                    # Print state_text but strip out JSON map data (within {})
                    print("=" * 80)
                    print("📊 COMPREHENSIVE STATE (LLM View)")
                    print("=" * 80)

                    # Strip JSON map sections from state_text for console display
                    display_text = self._strip_json_from_state(state_text)
                    print(display_text)
                    print("=" * 80)
                    print(flush=True)
                except:
                    print("Error: No screenshot\n")
                    screenshot_b64 = None
                    state_text = game_state_result

                # Get story objectives if enabled
                story_objectives_text = self._get_story_objectives()

                # Build prompt for this step with game state included
                # Include a reminder of the decision-making format
                scene_directions = """
When STUCK in dialogue or repeating actions:
✅ Try pressing B to exit dialogue/menus
✅ Move to a DIFFERENT location (LEFT/RIGHT/UP/DOWN)
✅ Interact with a DIFFERENT NPC or object
✅ Check the map for stairs (S), doors (D), or unexplored areas (?)
✅ Walk to a different part of the town/map
❌ DO NOT keep pressing A if it's not advancing dialogue

STORY-BLOCKED LOCATIONS:
⚠️ Some buildings/areas are BLOCKED BY STORY PROGRESSION, not just walls!
- If you try entering a door 3+ times and it doesn't work → It's story-locked
- Story-locked locations: Professor's Lab (before getting Pokemon), certain buildings
- STOP trying story-locked locations - explore OTHER areas instead
- The game will naturally unlock these as you progress

EXPLORATION IS KEY:
🔍 Look for UNEXPLORED TILES marked as "?" or "type": "unknown" in the map
🔍 These are adjacent to known areas but haven't been visited yet
🔍 Prioritize exploring "?" tiles - they may have NPCs, items, or story triggers
🔍 If stuck for 3+ steps at same coordinates → Find and navigate to nearest "?" tile

WHEN TRULY STUCK:
- Look at the ENTIRE map JSON for "type": "unknown" tiles
- Navigate to unexplored (?) areas
- Try talking to ALL NPCs in town (not just the same one)
- Explore EVERY building entrance you haven't tried
- Check for routes/paths leading OUT of the current town
- Doors/stairs with "leads_to" show where they go - use this to plan your route!
"""
                # Check if we're in title sequence
                if 'TITLE_SEQUENCE' in location:
                    scene_directions = """
TITLE SEQUENCE NAVIGATION:
📺 You are on the title screen or in intro cutscenes
📺 Press START or A to begin a new game or continue
📺 If in opening cutscene, press A repeatedly to advance dialogue
📺 Your goal is to reach the MOVING_VAN (start of actual gameplay)
📺 Do NOT use exploration commands - just advance through the intro

CHARACTER NAMING SCREEN:
✏️ You need to name your character
✏️ Navigate the on-screen keyboard using D-PAD (UP/DOWN/LEFT/RIGHT)
✏️ Press A to select a letter
✏️ Press B to remove a letter
✏️ Press START and A when done naming (usually after 3-7 characters)
✏️ Be creative with your name!
✏️ Take your time - use one button press at a time to navigate the keyboard
✏️ Watch the cursor position to know where you are on the keyboard
"""
                
                format_reminder = f"""
CRITICAL STUCK DETECTION: Check your recent history! If you pressed the SAME button 3+ times in a row with NO position change or NO new dialogue text, you ARE STUCK!

{scene_directions}

Please think step by step before choosing your action. Structure your response like this:

ANALYSIS:
[Analyze what you see in the frame and current game state - what's happening? where are you? what should you be doing? What tiles (S, D, etc.) are visible on the map and at what coordinates?
IMPORTANT: Look carefully at the game image for objects (clocks, pokeballs, bags) and NPCs (people, trainers) that might not be shown on the map. NPCs appear as sprite characters and can block movement or trigger battles/dialogue. When you see them try determine their location (X,Y) on the map relative to the player and any objects.
CRITICAL: Are you seeing the SAME dialogue text as before? If yes, you're stuck in a loop!]

OBJECTIVES:
[Review your current objectives. You have main storyline objectives (story_*) that track overall Emerald progression - these are automatically verified and you CANNOT manually complete them.  There may be sub-objectives that you need to complete before the main milestone. You can create your own sub-objectives to help achieve the main goals. Do any need to be updated, added, or marked as complete?
- What is your current objective?
- Add sub-objectives: ADD_OBJECTIVE: type:description:target_value (e.g., "ADD_OBJECTIVE: location:Find Pokemon Center in town:(15,20)" or "ADD_OBJECTIVE: item:Buy Pokeballs:5")
- Complete sub-objectives only: COMPLETE_OBJECTIVE: objective_id:notes (e.g., "COMPLETE_OBJECTIVE: my_sub_obj_123:Successfully bought Pokeballs").
- NOTE: Do NOT try to complete storyline objectives (story_*) - they auto-complete when milestones are reached]

PLAN:
[Think about your immediate goal - what do you want to accomplish in the next few actions? Consider your current objectives and recent history.
Check MOVEMENT MEMORY for areas you've had trouble with before and plan your route accordingly.
IMPORTANT: If you've been doing the same action repeatedly, CHANGE YOUR PLAN! Try something different!]

REASONING:
[Explain why you're choosing this specific action. Reference the MOVEMENT PREVIEW and MOVEMENT MEMORY sections. Check the visual frame for NPCs before moving. If you see NPCs in the image, avoid walking into them. Consider any failed movements or known obstacles from your memory.
CRITICAL: If your last 3+ actions were the same (e.g., pressing A repeatedly), explain WHY you're now trying something DIFFERENT.]

ACTION:
   - REQUIRED: Call either navigate_to(x, y) OR press_buttons([...])
   - Look at the ENTIRE map for S/D tiles and their (X,Y) coordinates
   - If stuck: Try B, or move to a different area, or interact with different NPCs
   - For hard control sequences (such as naming your character or moving around NPCs), it is best to choose 1 action at a time and view the result before continuing.
"""

                if self.step_count == 0:
                    prompt = f"Here is the current game state:\n\n{state_text}\n{story_objectives_text}\n\n{format_reminder}\nBased on this state, analyze, plan, and execute the next action to progress through the game."
                else:
                    prompt = f"Here is the current game state:\n\n{state_text}\n{story_objectives_text}\n\n{format_reminder}\nBased on this state and previous actions, analyze, plan, and execute the next action to progress through the game."

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
            self._save_checkpoint()
            return 0
        except Exception as e:
            logger.error(f"\n❌ Fatal error: {e}")
            traceback.print_exc()
            self._save_checkpoint()
            return 1

        logger.info(f"\n🎯 Agent completed {self.step_count} steps")
        logger.info(f"📚 Conversation history: {len(self.conversation_history)} turns")
        logger.info(self._format_history_for_display())
        self._save_checkpoint()
        return 0

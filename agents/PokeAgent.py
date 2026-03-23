#!/usr/bin/env python3
"""
PokeAgent for Pokemon Emerald
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
import traceback
import PIL.Image as PILImage
import io
import base64

import json as json_module

import google.generativeai.types as genai_types
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from utils.metric_tracking.server_metrics import update_server_metrics
from utils.data_persistence.llm_logger import get_llm_logger
from utils.agent_infrastructure.vlm_backends import VLM
from utils.data_persistence.run_data_manager import get_run_data_manager, initialize_run_data_manager
from agents.utils.prompt_optimizer import create_prompt_optimizer
from agents.subagents import (
    DEFAULT_TRAJECTORY_WINDOW,
    DEFAULT_SUMMARY_WINDOW,
    PokeAgentRuntime,
    allowed_battler_tool_names,
    build_battler_prompt,
    build_gym_puzzle_prompt,
    build_local_subagent_tool_declarations,
    build_reflect_prompt,
    build_summarize_prompt,
    build_verify_prompt,
    clamp_trajectory_window,
    decode_screenshot_base64,
    extract_key_events_from_summary,
    format_battler_history,
    get_gym_puzzle_info,
    get_local_subagent_spec,
    is_local_subagent_tool,
    load_subagent_context,
    parse_verify_response,
    resolve_gym_name,
    resolve_verification_target,
)
from agents.subagents.planner import (
    PLANNER_HISTORY_CAP,
    PLANNER_SAFETY_CAP,
    REPLAN_OBJECTIVES_TOOL_DECLARATION,
    allowed_planner_tool_names,
    build_planner_prompt,
    format_full_objective_sequence,
    format_planner_history,
)
from agents.prompts.paths import (
    POKEAGENT_BASE_PROMPT_PATH,
    POKEAGENT_PROMPT_PATH,
    POKEAGENT_SYSTEM_PROMPT_PATH,
    resolve_repo_path,
)

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

            # Use longer timeout for initial MCP calls that may need to load data from disk
            # (knowledge base, porymap data, etc.)
            timeout = 90  # Increased from 30 to handle startup initialization
            response = requests.post(url, json=arguments, timeout=timeout)
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ Tool {tool_name} completed")

            # Special handling for get_game_state - print the actual formatted text
            if tool_name == "get_game_state" and result.get("success") and "state_text" in result:
                logger.info("   Game State:")
                logger.info("\n" + "="*70)
                logger.info(result["state_text"])
                logger.info("="*70 + "\n")
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


class PokeAgent:
    """Custom benchmark agent using Gemini API directly with all MCP tools available."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model: str = "gemini-2.5-flash",
        backend: str = "gemini",
        max_steps: Optional[int] = None,
        system_instructions_file: str = None,  # Will be set based on optimization flag
        max_context_chars: int = 100000,
        target_context_chars: int = 50000,
        enable_prompt_optimization: bool = False,
        optimization_frequency: int = 10
    ):
        logger.info(f"🚀 Initializing PokeAgent with backend={backend}, model={model}, server={server_url}")
        self.server_url = server_url
        self.model = model
        self.backend = backend
        self.max_steps = max_steps
        self.step_count = 0
        self.max_context_chars = max_context_chars
        self.target_context_chars = target_context_chars
        self.optimization_enabled = enable_prompt_optimization
        self.optimization_frequency = optimization_frequency

        # Conversation history for tracking and compaction
        self.conversation_history = []

        # Recent function call results to add to next step's context
        # Format: [(function_name, result_json_string, timestamp), ...]
        self.recent_function_results = []
        self._subagent_vlm_cache: Dict[Tuple[str, ...], Any] = {}
        self._local_subagent_vlm = None
        self.runtime = PokeAgentRuntime(
            initial_step=self.step_count,
            publish_history=self._add_to_history,
            publish_function_result=self._store_function_result_for_context,
            on_step_change=lambda step: setattr(self, "step_count", step),
        )

        # Determine which system instructions file to use
        if system_instructions_file is None:
            if self.optimization_enabled:
                system_instructions_file = POKEAGENT_SYSTEM_PROMPT_PATH  # Tools + hard constraints
            else:
                system_instructions_file = POKEAGENT_PROMPT_PATH  # Full single-file prompt

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
        logger.info(f"✅ VLM initialized with {self.backend} backend using model: {self.model}")
        logger.info(f"✅ {len(self.tools)} tools available")
        logger.info(f"✅ System instructions loaded ({len(self.system_instructions)} chars)")

        # Initialize LLM logger
        self.llm_logger = get_llm_logger()
        
        # Initialize prompt optimizer if enabled
        self.prompt_optimizer = None
        if self.optimization_enabled:
            run_manager = get_run_data_manager()
            if not run_manager:
                raise RuntimeError(
                    "enable_prompt_optimization=True requires an initialized run_data_manager "
                    "(experiment run directory). Use run.py (or equivalent) so run data is set up "
                    "before starting PokeAgent, or disable prompt optimization."
                )
            self.prompt_optimizer = create_prompt_optimizer(
                vlm=self.vlm,
                run_data_manager=run_manager,
                base_prompt_path=POKEAGENT_BASE_PROMPT_PATH,
            )
            logger.info(f"🔄 Prompt optimization ENABLED (frequency: every {optimization_frequency} steps)")

    def _load_system_instructions(self, filename: str) -> str:
        """Load system instructions from file."""
        filepath = resolve_repo_path(filename)
        if not filepath.exists():
            logger.warning(f"System instructions file not found: {filepath}")
            return "You are an AI agent playing Pokemon Emerald. Use the available tools to progress through the game."

        with open(filepath, 'r') as f:
            content = f.read()

        logger.info(f"✅ Loaded system instructions from {filename} ({len(content)} chars)")
        return content
    
    def _load_base_prompt(self) -> str:
        """Load base prompt (strategic guidance) from file.
        
        This prompt can be optimized by the prompt optimizer.
        If prompt optimization is enabled, this will load the optimized version.
        """
        # Check if we have an optimizer with a current prompt
        if hasattr(self, 'prompt_optimizer') and self.prompt_optimizer:
            prompt = self.prompt_optimizer.get_current_prompt()
            logger.info(f"📋 Loaded base prompt from optimizer ({len(prompt)} chars)")
            # Log first 200 chars to verify it's the optimized version
            preview = prompt[:200].replace('\n', ' ')
            logger.info(f"   Preview: {preview}...")
            return prompt
        else:
            if hasattr(self, 'prompt_optimizer'):
                logger.warning(f"⚠️ prompt_optimizer exists but is {self.prompt_optimizer}")
            else:
                logger.info(f"📋 No prompt_optimizer attribute found")
        
        # Otherwise load from canonical repo path (e.g. optimization off but code path hit)
        filepath = resolve_repo_path(POKEAGENT_BASE_PROMPT_PATH)
        if not filepath.exists():
            logger.warning(f"Base prompt file not found: {filepath}, using minimal default")
            return """# Strategic Guidance
## Make intelligent decisions to progress through Pokemon Emerald.
- Think step-by-step
- Use tools effectively
- Store knowledge
- Complete objectives"""
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        logger.info(f"📋 Loaded base prompt from file ({len(content)} chars)")
        return content

    def _create_tool_declarations(self):
        """Create Gemini function declarations for ALL MCP tools (Pokemon + Baseline) - ALL ENABLED."""

        # Use Gemini's declaration format with proper types

        tools = [
            # ============================================================
            # POKEMON MCP TOOLS
            # ============================================================

            # Game Control Tools
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
                        },
                        "consider_npcs": {
                            "type_": "BOOLEAN",
                            "description": "Whether to treat NPCs as obstacles during pathfinding. Set to true to avoid NPCs (recommended for most navigation). Set to false only if NPCs are wandering/moving and you want to ignore them."
                        }
                    },
                    "required": ["x", "y", "variance", "reason", "consider_npcs"]
                }
            },
            {
                "name": "complete_direct_objective",
                "description": "Complete the current direct objective and advance to the next one. In CATEGORIZED mode, you must specify which category objective to complete (story, battling, or dynamics). In LEGACY mode, category is ignored.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "reasoning": {
                            "type_": "STRING",
                            "description": "REQUIRED FORMAT: Must include 'ANALYZE: [current state, objective requirements, completion evidence]' and 'PLAN: [confirm completion, next objective]'. Example: 'ANALYZE: Objective was to reach Route 101. Current location shows Route 101 at (15,5). Evidence: Game text shows \"Route 101\". PLAN: Objective complete, marking as done. Next: Talk to Prof Birch.'"
                        },
                        "category": {
                            "type_": "STRING",
                            "enum": ["story", "battling", "dynamics"],
                            "description": "Which category objective to complete (required in CATEGORIZED mode). 'story' for narrative objectives, 'battling' for team objectives, 'dynamics' for agent-created objectives."
                        }
                    },
                    "required": ["reasoning"]
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
            # COMMENTED OUT FOR NOW AS OBJECTIVE CREATION IS HANDLED BY THE PLANNER SUBAGENT
            #  {
            #     "name": "create_direct_objectives",
            #     "description": "Create the next 3 direct objectives when you need new goals. In LEGACY mode, creates general objectives. In CATEGORIZED mode, you MUST choose a category (story, battling, or dynamics). Use 'story' for walkthrough progression, 'battling' for training prep, and 'dynamics' for short-term navigation/cleanup. Provide exactly 3 objectives with id, description, action_type, target_location, navigation_hint, and completion_condition.",
            #     "parameters": {
            #         "type_": "OBJECT",
            #         "properties": {
            #             "objectives": {
            #                 "type_": "ARRAY",
            #                 "items": {
            #                     "type_": "OBJECT",
            #                     "properties": {
            #                         "id": {"type_": "STRING", "description": "Unique identifier (e.g., 'dynamic_01_navigate_route')"},
            #                         "description": {"type_": "STRING", "description": "Clear description of what to accomplish"},
            #                         "action_type": {
            #                             "type_": "STRING",
            #                             "enum": ["navigate", "interact", "battle", "wait"],
            #                             "description": "Type of action"
            #                         },
            #                         "target_location": {"type_": "STRING", "description": "Target location/map name"},
            #                         "navigation_hint": {"type_": "STRING", "description": "Specific guidance on how to accomplish this"},
            #                         "completion_condition": {"type_": "STRING", "description": "How to verify completion (e.g., 'location_contains_route_102')"}
            #                     },
            #                     "required": ["id", "description", "action_type"]
            #                 },
            #                 "description": "Array of exactly 3 objectives to create next"
            #             },
            #             "category": {
            #                 "type_": "STRING",
            #                 "enum": ["dynamics", "story", "battling"],
            #                 "description": "Category for objectives: 'story' (walkthrough progression), 'battling' (training/prep), or 'dynamics' (short-term navigation/cleanup). Choose the category that matches the goal."
            #             },
            #             "reasoning": {
            #                 "type_": "STRING",
            #                 "description": "Explanation of why these objectives were chosen (referencing walkthrough/wiki sources)"
            #             }
            #         },
            #         "required": ["objectives", "reasoning"]
            #     }
            # },
            {
                "name": "get_progress_summary",
                "description": "Get comprehensive progress summary including completed milestones, objectives, current location, and knowledge base summary.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_walkthrough",
                "description": "Get official Emerald walkthrough (Parts 1-21). Part 1: Littleroot, Part 6: Roxanne, Part 21: Elite Four.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "part": {
                            "type_": "INTEGER",
                            "description": "Walkthrough part 1-21"
                        }
                    },
                    "required": ["part"]
                }
            },
            {
                "name": "lookup_pokemon_info",
                "description": "Look up Pokemon information from Bulbapedia (stats, moves, evolution, locations).",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "topic": {
                            "type_": "STRING",
                            "description": "Pokemon name or topic to look up"
                        },
                        "source": {
                            "type_": "STRING",
                            "description": "Wiki source (default: bulbapedia)"
                        }
                    },
                    "required": ["topic"]
                }
            },
        ]

        tools.extend(build_local_subagent_tool_declarations())

        logger.info(f"✅ Created {len(tools)} tool declarations (ALL TOOLS ENABLED)")
        return tools

    def _execute_function_call_by_name(self, function_name: str, arguments: dict) -> str:
        """Execute a function by name with given arguments and return result as JSON string."""
        if is_local_subagent_tool(function_name):
            spec = get_local_subagent_spec(function_name)
            return getattr(self, spec.handler_method)(arguments)

        # REGULAR MCP TOOL CALL VIA MCP ADAPTER
        result = self.mcp_adapter.call_tool(function_name, arguments)
        # Return as JSON string
        return json.dumps(result, indent=2)

    def _get_subagent_vlm(
        self,
        tool_names: Optional[set[str]] = None,
        supplemental_tools: Optional[list[dict]] = None,
    ):
        """Lazily create cached VLMs for local subagents.

        Tool-less subagents (reflect, verify, summarize, gym_puzzle) get a bare
        VLM without tools or system instructions.  Tool-using subagents (battler,
        planner) get a VLM with only their allowed tool declarations AND the
        orchestrator's system instructions so they inherit the same behavioural
        grounding.

        ``supplemental_tools`` allows injecting extra tool declarations that are
        not part of the orchestrator's own tool set (e.g. ``replan_objectives``
        which is planner-exclusive).
        """
        normalized = tuple(sorted(tool_names or ()))
        supp_key = tuple(t["name"] for t in (supplemental_tools or []))
        cache_key = (normalized, supp_key)

        if not normalized and not supp_key:
            if self._local_subagent_vlm is None:
                self._local_subagent_vlm = VLM(
                    model_name=self.model,
                    backend=self.backend,
                    tools=None,
                )
            return self._local_subagent_vlm

        cached = self._subagent_vlm_cache.get(cache_key)
        if cached is None:
            allowed_tools = [tool for tool in self.tools if tool.get("name") in set(normalized)]
            if supplemental_tools:
                allowed_tools.extend(supplemental_tools)
            cached = VLM(
                model_name=self.model,
                backend=self.backend,
                tools=allowed_tools,
                system_instruction=self.system_instructions,
            )
            self._subagent_vlm_cache[cache_key] = cached
        return cached

    def _run_one_step_subagent(
        self,
        *,
        prompt: str,
        interaction_name: str,
        current_image=None,
    ) -> tuple[int, str]:
        """Run a one-step local subagent with its own claimed global step."""
        step_number = self.runtime.claim_step(owner="subagent", interaction_name=interaction_name)
        subagent_vlm = self._get_subagent_vlm()
        if current_image is not None:
            response = subagent_vlm.get_query(current_image, prompt, interaction_name)
        else:
            response = subagent_vlm.get_text_query(prompt, interaction_name)

        # Tag the step in cumulative_metrics so one-step subagent calls are
        # identifiable (they produce no tool calls, so without this the entry
        # would appear as an anonymous ghost step).
        self.llm_logger.add_step_tool_calls(
            step_number,
            [{"name": interaction_name, "args": {}}],
        )

        text = self._extract_text_from_response(response)
        if text:
            return step_number, text
        raise RuntimeError(f"{interaction_name} did not return text output")

    def _extract_recommended_next_action(self, text: str) -> str:
        for line in (text or "").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("**") and not stripped.startswith("- "):
                if "RECOMMENDED_NEXT_ACTION" in stripped.upper():
                    continue
            if stripped and "RECOMMENDED_NEXT_ACTION" not in stripped.upper():
                return stripped
        return ""

    def _execute_subagent_reflect(self, arguments: dict) -> str:
        """Execute reflection using a local, tool-less subagent."""
        try:
            last_n_steps = clamp_trajectory_window(arguments.get("last_n_steps"))
            context = load_subagent_context(
                self.mcp_adapter,
                get_run_data_manager(),
                last_n_steps=last_n_steps,
                include_current_image=True,
            )
            prompt = build_reflect_prompt(
                situation=arguments.get("situation", "Agent requested reflection"),
                context=context,
                last_n_steps=last_n_steps,
            )
            step_number, reflection_text = self._run_one_step_subagent(
                prompt=prompt,
                interaction_name="Subagent_Reflect",
                current_image=context.get("current_image"),
            )
            logger.info(f"✅ Self-reflection complete ({len(reflection_text)} chars)")
            return json.dumps(
                {
                    "success": True,
                    "reflection": reflection_text,
                    "step_number": step_number,
                    "context_analyzed": {
                        "steps_reviewed": len(context.get("trajectory_window", [])),
                        "location": context.get("current_state", {}).get("location"),
                        "objective_status": context.get("objective_state", {}).get("status"),
                    },
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error in reflect execution: {e}")
            traceback.print_exc()
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _execute_subagent_verify(self, arguments: dict) -> str:
        """Execute objective verification using a local, tool-less subagent."""
        try:
            last_n_steps = clamp_trajectory_window(arguments.get("last_n_steps"))
            context = load_subagent_context(
                self.mcp_adapter,
                get_run_data_manager(),
                last_n_steps=last_n_steps,
                include_current_image=True,
            )
            target = resolve_verification_target(
                context.get("objective_state", {}),
                category=arguments.get("category"),
            )
            prompt = build_verify_prompt(
                context=context,
                target=target,
                last_n_steps=last_n_steps,
                reasoning=arguments.get("reasoning", ""),
            )
            step_number, verify_text = self._run_one_step_subagent(
                prompt=prompt,
                interaction_name="Subagent_Verify",
                current_image=context.get("current_image"),
            )
            verdict = parse_verify_response(verify_text, target=target)
            verdict["success"] = True
            verdict["step_number"] = step_number
            verdict["steps_reviewed"] = len(context.get("trajectory_window", []))
            verdict["location"] = context.get("current_state", {}).get("location")
            return json.dumps(verdict, indent=2)
        except Exception as e:
            logger.error(f"Error in verify execution: {e}")
            traceback.print_exc()
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _execute_subagent_gym_puzzle(self, arguments: dict) -> str:
        """Execute gym puzzle solving using a lightweight local subagent."""
        try:
            game_state = self.mcp_adapter.call_tool("get_game_state", {})
            if not game_state.get("success"):
                return json.dumps({"success": False, "error": "Failed to get game state"})

            state_text = game_state.get("state_text", "")
            gym_name = resolve_gym_name(arguments, state_text)
            gym_info = get_gym_puzzle_info(gym_name)
            current_image = decode_screenshot_base64(game_state.get("screenshot_base64"))
            if current_image is None:
                return json.dumps({"success": False, "error": "No frame available in game state"})

            step_number, puzzle_text = self._run_one_step_subagent(
                prompt=build_gym_puzzle_prompt(
                    gym_name=gym_name,
                    gym_info=gym_info,
                    state_text=state_text,
                    action_history=self._format_action_history(),
                    function_results=self._get_function_results_context(),
                ),
                interaction_name="Gym_Puzzle_Analysis",
                current_image=current_image,
            )

            logger.info(f"✅ Gym puzzle analysis complete ({len(puzzle_text)} chars)")
            return json.dumps(
                {
                    "success": True,
                    "gym": gym_name,
                    "analysis": puzzle_text,
                    "step_number": step_number,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error in solve_gym_puzzle execution: {e}")
            traceback.print_exc()
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _execute_subagent_summarize(self, arguments: dict) -> str:
        """Summarize the latest trajectory window without an extra compaction layer."""
        try:
            requested_window = arguments.get("last_n_steps", DEFAULT_SUMMARY_WINDOW)
            last_n_steps = clamp_trajectory_window(requested_window)
            context = load_subagent_context(
                self.mcp_adapter,
                get_run_data_manager(),
                last_n_steps=last_n_steps,
                include_current_image=False,
            )
            step_number, summary_text = self._run_one_step_subagent(
                prompt=build_summarize_prompt(
                    context=context,
                    last_n_steps=last_n_steps,
                    reasoning=arguments.get("reasoning", ""),
                ),
                interaction_name="Subagent_Summarize",
                current_image=None,
            )
            return json.dumps(
                {
                    "success": True,
                    "summary": summary_text,
                    "recommended_next_action": self._extract_recommended_next_action(summary_text),
                    "steps_reviewed": len(context.get("trajectory_window", [])),
                    "location": context.get("current_state", {}).get("location"),
                    "step_number": step_number,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error in summarize execution: {e}")
            traceback.print_exc()
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _execute_subagent_battler(self, arguments: dict) -> str:
        """Delegate the active battle to a local looping battler."""
        try:
            initial_state = self.mcp_adapter.call_tool("get_game_state", {})
            raw_state = initial_state.get("raw_state", {}) if isinstance(initial_state, dict) else {}
            is_in_battle = bool(raw_state.get("game", {}).get("is_in_battle") or initial_state.get("is_in_battle"))
            if not initial_state.get("success"):
                return json.dumps({"success": False, "error": "Failed to get game state"}, indent=2)
            if not is_in_battle:
                return json.dumps(
                    {
                        "success": True,
                        "entered_battle_loop": False,
                        "battle_resolved": False,
                        "turns_taken": 0,
                        "battle_summary": "Delegation skipped because the agent is not currently in battle.",
                        "key_events": [],
                        "recommended_next_action": "Continue with the overworld orchestrator.",
                    },
                    indent=2,
                )

            last_n_steps = clamp_trajectory_window(arguments.get("last_n_steps", DEFAULT_SUMMARY_WINDOW))
            handoff = json.loads(
                self._execute_subagent_summarize(
                    {
                        "reasoning": "Summarize the recent context leading into this battle for a delegated battler handoff.",
                        # Respect orchestrator's last_n_steps (already clamped); do not force a minimum.
                        "last_n_steps": last_n_steps,
                    }
                )
            )
            battle_result = self._run_battler_loop(
                handoff_summary=handoff.get("summary", ""),
            )
            battle_result["success"] = True
            battle_result["entered_battle_loop"] = True
            return json.dumps(battle_result, indent=2)
        except Exception as e:
            logger.error(f"Error in battler execution: {e}")
            traceback.print_exc()
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _run_battler_loop(self, *, handoff_summary: str) -> Dict[str, Any]:
        run_manager = get_run_data_manager()
        battler_vlm = self._get_subagent_vlm(allowed_battler_tool_names())
        turns_taken = 0
        key_events: List[str] = []
        battle_history: List[Dict[str, Any]] = []
        resolved = False
        SAFETY_CAP = 200

        while turns_taken < SAFETY_CAP:
            current_context = load_subagent_context(
                self.mcp_adapter,
                run_manager,
                last_n_steps=1,
                include_current_image=True,
            )
            game_state_result = current_context.get("game_state_result", {})
            raw_state = game_state_result.get("raw_state", {}) or {}
            in_battle = bool(
                raw_state.get("game", {}).get("is_in_battle")
                or game_state_result.get("is_in_battle")
            )
            if not in_battle:
                resolved = True
                break

            current_image = current_context.get("current_image")
            step_number = self.runtime.claim_step(
                owner="subagent_battler", interaction_name="Subagent_Battler"
            )
            prompt = build_battler_prompt(
                current_state_text=current_context.get("current_state", {}).get("state_text", ""),
                location=current_context.get("current_state", {}).get("location", "Unknown"),
                objective_state=current_context.get("objective_state", {}),
                progress=current_context.get("progress", {}),
                knowledge_summary=current_context.get("knowledge_summary", ""),
                handoff_summary=handoff_summary,
                battle_history=format_battler_history(battle_history),
                turn_index=turns_taken + 1,
            )

            if current_image is not None:
                response = battler_vlm.get_query(current_image, prompt, "Subagent_Battler")
            else:
                response = battler_vlm.get_text_query(prompt, "Subagent_Battler")
            reasoning_text = self._extract_text_from_response(response)

            tool_calls_made: List[Dict[str, Any]] = []
            used_tools = self._handle_vlm_function_calls(
                response,
                tool_calls_made,
                0,
                4,
                allowed_tool_names=allowed_battler_tool_names(),
            )
            if not used_tools or not tool_calls_made:
                key_events.append("Battler produced no executable tool call.")
                break

            self.llm_logger.add_step_tool_calls(step_number, tool_calls_made)
            if run_manager:
                pre_state = run_manager.create_state_snapshot(raw_state) if raw_state else None
                if pre_state:
                    self._log_trajectory_for_step(
                        run_manager=run_manager,
                        step_num=step_number,
                        pre_state=pre_state,
                        prompt=prompt,
                        reasoning=reasoning_text,
                        tool_calls=tool_calls_made,
                        response=reasoning_text,
                    )

            battle_history.append({
                "step": turns_taken + 1,
                "reasoning": reasoning_text,
                "tool_calls": tool_calls_made,
            })

            last_call = tool_calls_made[-1]
            key_events.append(f"Turn {turns_taken + 1}: {last_call['name']}")
            turns_taken += 1

        if turns_taken >= SAFETY_CAP and not resolved:
            key_events.append("Battler hit the safety turn cap before returning to the overworld.")

        # Exit summary: cover roughly the battler's own turns in global trajectory (cap 50).
        exit_trajectory_window = clamp_trajectory_window(turns_taken)
        exit_summary = json.loads(
            self._execute_subagent_summarize(
                {
                    "reasoning": (
                        "Compact the completed delegated battle into a concise handoff for the main orchestrator. "
                        f"Focus on the last ~{exit_trajectory_window} global steps (battler inner turns: {turns_taken})."
                    ),
                    "last_n_steps": exit_trajectory_window,
                }
            )
        )
        summary_text = exit_summary.get("summary", "")
        key_events.extend(extract_key_events_from_summary(summary_text))
        return {
            "battle_resolved": resolved,
            "turns_taken": turns_taken,
            "battle_summary": summary_text,
            "key_events": key_events[:8],
            "recommended_next_action": exit_summary.get("recommended_next_action")
            or "Resume overworld planning using the compacted battle summary.",
        }

    # ------------------------------------------------------------------
    # Objectives planner subagent (looping)
    # ------------------------------------------------------------------

    def _execute_subagent_plan_objectives(self, arguments: dict) -> str:
        """Delegate objective planning/replanning to the local planner loop."""
        try:
            reason = arguments.get("reason", "Orchestrator requested objective planning.")
            last_n_steps = clamp_trajectory_window(arguments.get("last_n_steps", DEFAULT_SUMMARY_WINDOW))

            handoff = json.loads(
                self._execute_subagent_summarize(
                    {
                        "reasoning": (
                            "Summarize the recent context for a delegated objective-planning handoff. "
                            "Emphasize current progress, what objectives have been completed, and "
                            "any issues the orchestrator is experiencing."
                        ),
                        # Respect orchestrator's last_n_steps (already clamped); do not force a minimum.
                        "last_n_steps": last_n_steps,
                    }
                )
            )

            full_seq_raw = self.mcp_adapter.call_tool("get_full_objective_sequence", {})
            if isinstance(full_seq_raw, str):
                cached_sequence = json.loads(full_seq_raw)
            elif isinstance(full_seq_raw, dict):
                cached_sequence = full_seq_raw
            else:
                cached_sequence = {}

            if not cached_sequence.get("success", True):
                return json.dumps(
                    {"success": False, "error": cached_sequence.get("error", "Failed to load objective sequence")},
                    indent=2,
                )

            planner_result = self._run_planner_loop(
                handoff_summary=handoff.get("summary", ""),
                reason=reason,
                cached_sequence=cached_sequence,
            )
            planner_result["success"] = True
            return json.dumps(planner_result, indent=2)

        except Exception as e:
            logger.error(f"Error in plan_objectives execution: {e}")
            traceback.print_exc()
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    def _run_planner_loop(
        self,
        *,
        handoff_summary: str,
        reason: str,
        cached_sequence: Dict[str, Any],
    ) -> Dict[str, Any]:
        run_manager = get_run_data_manager()
        planner_vlm = self._get_subagent_vlm(
            allowed_planner_tool_names(),
            supplemental_tools=[REPLAN_OBJECTIVES_TOOL_DECLARATION],
        )

        planner_history: List[Dict[str, Any]] = []
        replan_results: List[Dict[str, Any]] = []
        turns_taken = 0
        should_return = False

        while turns_taken < PLANNER_SAFETY_CAP and not should_return:
            current_context = load_subagent_context(
                self.mcp_adapter,
                run_manager,
                last_n_steps=1,
                include_current_image=True,
            )
            current_image = current_context.get("current_image")

            step_number = self.runtime.claim_step(
                owner="subagent_plan_objectives",
                interaction_name="Subagent_Plan_Objectives",
            )

            prompt = build_planner_prompt(
                reason=reason,
                objective_sequence=format_full_objective_sequence(cached_sequence),
                current_state_text=current_context.get("current_state", {}).get("state_text", ""),
                location=current_context.get("current_state", {}).get("location", "Unknown"),
                progress=current_context.get("progress", {}),
                knowledge_summary=current_context.get("knowledge_summary", ""),
                planner_history=format_planner_history(planner_history[-PLANNER_HISTORY_CAP:]),
                handoff_summary=handoff_summary,
                turn_index=turns_taken + 1,
            )

            if current_image is not None:
                response = planner_vlm.get_query(current_image, prompt, "Subagent_Plan_Objectives")
            else:
                response = planner_vlm.get_text_query(prompt, "Subagent_Plan_Objectives")
            reasoning_text = self._extract_text_from_response(response)

            tool_calls_made: List[Dict[str, Any]] = []
            used_tools = self._handle_vlm_function_calls(
                response,
                tool_calls_made,
                0,
                4,
                allowed_tool_names=allowed_planner_tool_names(),
            )

            self.llm_logger.add_step_tool_calls(step_number, tool_calls_made)
            if run_manager:
                game_state_result = current_context.get("game_state_result", {})
                raw_state = game_state_result.get("raw_state", {}) or {}
                pre_state = run_manager.create_state_snapshot(raw_state) if raw_state else None
                if pre_state:
                    self._log_trajectory_for_step(
                        run_manager=run_manager,
                        step_num=step_number,
                        pre_state=pre_state,
                        prompt=prompt,
                        reasoning=reasoning_text,
                        tool_calls=tool_calls_made,
                        response=reasoning_text,
                    )

            planner_history.append({
                "step": turns_taken + 1,
                "reasoning": reasoning_text,
                "tool_calls": tool_calls_made,
            })

            for tc in tool_calls_made:
                if tc.get("name") == "replan_objectives":
                    result_raw = tc.get("result", "{}")
                    try:
                        result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
                    except (json.JSONDecodeError, TypeError):
                        result = {}
                    if result.get("success"):
                        replan_results.append({
                            "category": tc.get("args", {}).get("category"),
                            "edits_applied": result.get("edits_applied", []),
                            "rationale": tc.get("args", {}).get("rationale", ""),
                            "new_sequence_length": result.get("new_sequence_length"),
                        })
                        fresh_seq = self.mcp_adapter.call_tool("get_full_objective_sequence", {})
                        if isinstance(fresh_seq, str):
                            try:
                                cached_sequence = json.loads(fresh_seq)
                            except (json.JSONDecodeError, TypeError):
                                pass
                        elif isinstance(fresh_seq, dict):
                            cached_sequence = fresh_seq

                        if tc.get("args", {}).get("return_to_orchestrator"):
                            should_return = True

            if not used_tools or not tool_calls_made:
                logger.warning("Planner produced no executable tool call — breaking loop.")
                break

            turns_taken += 1

        if turns_taken >= PLANNER_SAFETY_CAP and not should_return:
            logger.warning("Planner hit the safety turn cap (%d) without returning.", PLANNER_SAFETY_CAP)

        final_rationale = (
            replan_results[-1]["rationale"] if replan_results else "No changes applied."
        )
        return {
            "changes": replan_results,
            "turns_taken": turns_taken,
            "steps_consumed": turns_taken,
            "rationale": final_rationale,
            "recommended_next_action": reasoning_text if reasoning_text else "Resume normal execution.",
        }

    def _convert_protobuf_value(self, value):
        """Recursively convert a protobuf value to JSON-serializable Python types."""
        # Handle None
        if value is None:
            return None

        # Check if it's a protobuf type
        if hasattr(value, '__class__') and 'proto' in value.__class__.__module__:
            value_type = value.__class__.__name__
            logger.debug(f"      Converting protobuf value: type={value_type}, module={value.__class__.__module__}")
            
            # First try to convert as dict (for MapComposite objects)
            # This must be checked BEFORE checking for __iter__ because MapComposite has both
            try:
                dict_value = dict(value)
                logger.debug(f"      ✅ Converted to dict with {len(dict_value)} keys")
                # Successfully converted to dict - recursively convert values
                return {k: self._convert_protobuf_value(v) for k, v in dict_value.items()}
            except (TypeError, ValueError) as e:
                logger.debug(f"      Not dict-like: {e}")
                # Not a dict-like type, check if it's a list
                pass

            # Check if it's a list-like type (RepeatedComposite, RepeatedScalar)
            if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                logger.debug(f"      Detected iterable (likely list/array)")
                # It's a list/array - recursively convert each item
                try:
                    items = list(value)
                    logger.debug(f"      ✅ Converted to list with {len(items)} items")
                    converted = [self._convert_protobuf_value(item) for item in items]
                    logger.debug(f"      ✅ Recursively converted all {len(converted)} items")
                    return converted
                except Exception as e:
                    logger.warning(f"      ⚠️ Error converting list items: {e}")
                    try:
                        fallback = list(value)
                        logger.debug(f"      Using fallback list conversion: {len(fallback)} items")
                        return fallback
                    except Exception as e2:
                        logger.error(f"      ❌ Fallback also failed: {e2}")
                        return value

            # Fallback: return as-is
            logger.debug(f"      Returning protobuf value as-is (type: {value_type})")
            return value

        # Check if it's a regular dict (might contain nested protobuf values)
        elif isinstance(value, dict):
            logger.debug(f"      Converting regular dict with {len(value)} keys")
            return {k: self._convert_protobuf_value(v) for k, v in value.items()}
        # Check if it's a regular list (might contain nested protobuf values)
        elif isinstance(value, list):
            logger.debug(f"      Converting regular list with {len(value)} items")
            return [self._convert_protobuf_value(item) for item in value]
        # Otherwise return as-is (native Python type)
        else:
            logger.debug(f"      Returning native Python value: type={type(value).__name__}")
            return value

    def _convert_protobuf_args(self, proto_args) -> dict:
        """Convert protobuf arguments to JSON-serializable Python types."""
        logger.debug(f"   Converting protobuf args: {len(proto_args)} keys")
        arguments = {}
        for key, value in proto_args.items():
            logger.debug(f"   Converting key '{key}': type={type(value).__name__}")
            try:
                converted = self._convert_protobuf_value(value)
                arguments[key] = converted
                logger.debug(f"   ✅ Key '{key}' converted successfully: type={type(converted).__name__}")
            except Exception as e:
                logger.error(f"   ❌ Error converting key '{key}': {e}")
                logger.error(f"   Traceback: {traceback.format_exc()}")
                # Try to include the raw value as fallback
                arguments[key] = str(value)
        logger.debug(f"   ✅ Converted {len(arguments)} arguments")
        return arguments

    def _execute_function_call(self, function_call, allowed_tool_names: Optional[set[str]] = None) -> str:
        """Execute a function call and return the result as JSON string."""
        function_name = function_call.name
        logger.info(f"🔧 Executing function: {function_name}")
        if allowed_tool_names is not None and function_name not in allowed_tool_names:
            raise ValueError(f"Tool {function_name} is not allowed in this context")

        # Parse arguments - convert protobuf types to native Python types
        try:
            logger.debug(f"   Converting protobuf args...")
            arguments = self._convert_protobuf_args(function_call.args)
            logger.info(f"   ✅ Successfully parsed arguments: {list(arguments.keys())}")
            
            # Special validation for create_direct_objectives
            if function_name == "create_direct_objectives":
                logger.info(f"   🎯 Validating create_direct_objectives arguments...")
                if "objectives" not in arguments:
                    logger.error(f"   ❌ Missing 'objectives' key in arguments!")
                    logger.error(f"   Available keys: {list(arguments.keys())}")
                    return json.dumps({"success": False, "error": "Missing 'objectives' parameter"})
                
                obj_list = arguments["objectives"]
                if not isinstance(obj_list, list):
                    logger.error(f"   ❌ 'objectives' is not a list! Type: {type(obj_list)}")
                    logger.error(f"   Value: {str(obj_list)[:500]}")
                    return json.dumps({"success": False, "error": f"'objectives' must be a list, got {type(obj_list)}"})
                
                if len(obj_list) != 3:
                    logger.warning(f"   ⚠️ Expected 3 objectives, got {len(obj_list)}")
                
                for i, obj in enumerate(obj_list):
                    if not isinstance(obj, dict):
                        logger.error(f"   ❌ Objective {i} is not a dict! Type: {type(obj)}")
                        logger.error(f"   Value: {str(obj)[:200]}")
                        return json.dumps({"success": False, "error": f"Objective {i} must be a dict, got {type(obj)}"})
                    
                    required_fields = ["id", "description", "action_type"]
                    missing = [f for f in required_fields if f not in obj]
                    if missing:
                        logger.error(f"   ❌ Objective {i} missing required fields: {missing}")
                        logger.error(f"   Available fields: {list(obj.keys())}")
                        return json.dumps({"success": False, "error": f"Objective {i} missing required fields: {missing}"})
                    
                    logger.info(f"   ✅ Objective {i} valid: id={obj.get('id')}, action_type={obj.get('action_type')}")
                
                logger.info(f"   ✅ All objectives validated successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to parse function arguments: {e}")
            logger.error(f"   Function: {function_name}")
            logger.error(f"   Args type: {type(function_call.args) if hasattr(function_call, 'args') else 'NO ARGS'}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return json.dumps({"success": False, "error": f"Invalid arguments: {e}"})

        if is_local_subagent_tool(function_name):
            spec = get_local_subagent_spec(function_name)
            return getattr(self, spec.handler_method)(arguments)

        # Call the tool via MCP adapter
        result = self.mcp_adapter.call_tool(function_name, arguments)

        # Return as JSON string
        return json.dumps(result, indent=2)

    def _add_to_history(
        self,
        prompt: str,
        response: str,
        tool_calls: List[Dict] = None,
        action_details: str = None,
        player_coords: tuple = None,
        start_coords: tuple = None,
        end_coords: tuple = None,
        step_number: Optional[int] = None,
    ):
        """Add interaction to conversation history - ONLY stores LLM responses and actions."""
        # Strip whitespace from response to save tokens
        response_stripped = response.strip() if response else ""

        # CRITICAL SAFEGUARD: If response contains our prompt header, skip it entirely
        if "You are an autonomous AI agent" in response_stripped:
            logger.warning(f"⚠️ Skipping corrupted history entry at step {self.step_count} (contains prompt echo)")
            return

        entry = {
            "step": step_number if step_number is not None else self.step_count,
            "llm_response": response_stripped,
            "timestamp": time.time()
        }

        logger.debug(f"📝 Storing history entry for step {entry['step']}: {response_stripped[:100]}...")

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

        if tool_calls:
            entry["tool_calls"] = tool_calls

        # Store explicit step start/end coordinates when available
        if start_coords:
            entry["start_coords"] = start_coords
        if end_coords:
            entry["end_coords"] = end_coords

        # Backward-compatible coordinate field (treated as end-of-step position)
        if player_coords:
            entry["player_coords"] = player_coords
        elif end_coords:
            entry["player_coords"] = end_coords

        self.conversation_history.append(entry)

        # Keep only last 20 entries to prevent unbounded growth
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

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

        for entry in self.conversation_history[-20:]:
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
        max_retries = 20  # Increased from 10 to allow more time for server initialization
        retry_delay = 5  # Increased from 3 to give server more time between checks

        for attempt in range(max_retries):
            try:
                # Use /health endpoint for faster startup checks (doesn't trigger state reads)
                response = requests.get(f"{self.server_url}/health", timeout=10)  # Increased from 5 to 10
                if response.status_code == 200:
                    logger.info(f"✅ Game server is ready")
                    break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Server not ready yet (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s...")
                    logger.info(f"   (Server may be loading porymap data, knowledge base, etc.)")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Game server not running at {self.server_url} after {max_retries * retry_delay}s: {e}")
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

    def run_step(
        self,
        prompt: str,
        max_tool_calls: int = 5,
        screenshot_b64: str = None,
        step_number: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Run a single agent step.

        Args:
            prompt: The prompt to send to model
            max_tool_calls: Maximum number of tool calls allowed per step (default: 5)
            screenshot_b64: Optional base64-encoded screenshot to include with prompt

        Returns:
            Tuple of (success: bool, response: str)
        """
        try:
            preview_step = step_number if step_number is not None else self.runtime.peek_next_step()

            # Capture pre-state for trajectory logging
            run_manager = get_run_data_manager()
            
            # DEBUG: Log run_manager availability
            if not run_manager:
                logger.warning(f"🔍 [DEBUG] Step {preview_step}: run_manager is None - trajectory logging will be skipped")
                # Try to initialize if not available
                run_id = os.environ.get("RUN_DATA_ID")
                if run_id:
                    logger.info(f"🔍 [DEBUG] Attempting to initialize run_manager with run_id: {run_id}")
                    run_manager = initialize_run_data_manager(run_id=run_id)
                    if run_manager:
                        logger.info(f"🔍 [DEBUG] Successfully initialized run_manager")
                    else:
                        logger.error(f"🔍 [DEBUG] Failed to initialize run_manager")
                else:
                    logger.warning(f"🔍 [DEBUG] RUN_DATA_ID environment variable not set")
            else:
                logger.debug(f"🔍 [DEBUG] Step {preview_step}: run_manager available: {run_manager.run_id}")
            
            pre_state = None
            if run_manager:
                # Get current game state for pre-state snapshot
                try:
                    game_state_result = self.mcp_adapter.call_tool("get_game_state", {})
                    if isinstance(game_state_result, str):
                        game_state_result = json.loads(game_state_result)
                    if game_state_result.get("success"):
                        raw_state = game_state_result.get("raw_state", {})
                        pre_state = run_manager.create_state_snapshot(raw_state)
                        logger.debug(f"🔍 [DEBUG] Step {preview_step}: pre_state captured: {pre_state.get('location', 'Unknown')}")
                    else:
                        logger.warning(f"🔍 [DEBUG] Step {preview_step}: get_game_state returned success=False")
                except Exception as e:
                    logger.error(f"🔍 [DEBUG] Step {preview_step}: Could not capture pre-state: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"🔍 [DEBUG] Step {preview_step}: run_manager is None, skipping pre_state capture")
            
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
            claimed_step = None
            try:

                if screenshot_b64:
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

                    claimed_step = self.runtime.claim_step(
                        owner="orchestrator",
                        interaction_name="Autonomous_CLI_Agent",
                    )

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
                            response = future.result(timeout=70)  # 70 second timeout for slower models
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
                    claimed_step = self.runtime.claim_step(
                        owner="orchestrator",
                        interaction_name="Autonomous_CLI_Agent",
                    )

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
                            response = future.result(timeout=70)  # 70 second timeout for slower models
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
                
                # Log detailed response information for debugging
                logger.info(f"🔍 Response analysis:")
                logger.info(f"   Response received: {type(response).__name__}")
                logger.info(f"   Response type: {type(response).__name__}")
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    logger.info(f"   Candidates: {len(response.candidates)}")
                    logger.info(f"   Candidate type: {type(candidate).__name__}")
                    
                    # Check for finish_reason (this is where MALFORMED_FUNCTION_CALL appears)
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        logger.warning(f"   ⚠️ FINISH_REASON: {finish_reason}")
                        if finish_reason and finish_reason != 1:  # 1 = STOP, other values indicate issues
                            logger.error(f"   🚨 NON-STOP FINISH REASON DETECTED: {finish_reason}")
                            logger.error(f"   This may indicate a malformed function call or other issue")
                    
                    # Log content structure
                    if hasattr(candidate, 'content') and candidate.content:
                        content = candidate.content
                        logger.info(f"   Content type: {type(content).__name__}")
                        if hasattr(content, 'parts'):
                            logger.info(f"   Content parts: {len(content.parts)}")
                            for i, part in enumerate(content.parts):
                                part_info = []
                                if hasattr(part, 'text') and part.text:
                                    part_info.append(f"text({len(part.text)} chars)")
                                if hasattr(part, 'function_call') and part.function_call:
                                    fc = part.function_call
                                    part_info.append(f"function_call(name={fc.name})")
                                    # Try to log function call args (may fail if malformed)
                                    try:
                                        if hasattr(fc, 'args'):
                                            args_dict = dict(fc.args) if hasattr(fc.args, '__iter__') else {}
                                            logger.info(f"      Function call args keys: {list(args_dict.keys())}")
                                            # For create_direct_objectives, log objectives structure
                                            if fc.name == "create_direct_objectives" and "objectives" in args_dict:
                                                obj_list = args_dict.get("objectives", [])
                                                logger.info(f"      Objectives array length: {len(obj_list) if isinstance(obj_list, list) else 'NOT A LIST'}")
                                                if isinstance(obj_list, list) and len(obj_list) > 0:
                                                    logger.info(f"      First objective keys: {list(obj_list[0].keys()) if isinstance(obj_list[0], dict) else type(obj_list[0])}")
                                    except Exception as e:
                                        logger.error(f"      ⚠️ Could not parse function call args: {e}")
                                        logger.error(f"      Args type: {type(fc.args) if hasattr(fc, 'args') else 'NO ARGS ATTR'}")
                                        logger.error(f"      Traceback: {traceback.format_exc()}")
                                logger.info(f"   Part {i}: {', '.join(part_info) if part_info else 'empty'}")
                        else:
                            logger.warning(f"   Content has no 'parts' attribute")
                    else:
                        logger.warning(f"   Candidate has no 'content' attribute")
                else:
                    logger.warning(f"   Response has no candidates")

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

            active_step = claimed_step if claimed_step is not None else preview_step

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
                elif last_tool_call['name'] == "subagent_gym_puzzle":
                    try:
                        result_data = json_module.loads(last_tool_call['result'])
                        if result_data.get("success"):
                            gym = result_data.get("gym", "Unknown")
                            analysis = result_data.get("analysis", "")
                            action_details = f"subagent_gym_puzzle({gym})\nAnalysis: {analysis}"
                        else:
                            action_details = f"subagent_gym_puzzle failed: {result_data.get('error', 'Unknown error')}"
                    except Exception as e:
                        logger.debug(f"Could not extract subagent_gym_puzzle details: {e}")
                        action_details = "Executed subagent_gym_puzzle"
                else:
                    action_details = f"Executed {last_tool_call['name']}"

                # Capture explicit start/end coordinates for short-term memory.
                start_coords = pre_state.get("player_coords") if pre_state else None
                end_coords = None
                last_tool_name = last_tool_call.get("name")
                if last_tool_name in {"press_buttons", "navigate_to"}:
                    self._wait_for_actions_complete()
                try:
                    final_state_result = self._execute_function_call_by_name("get_game_state", {})
                    final_state_data = json_module.loads(final_state_result)
                    if final_state_data.get("success"):
                        player_pos = final_state_data.get("player_position", {})
                        if player_pos and "x" in player_pos and "y" in player_pos:
                            end_coords = (player_pos.get("x"), player_pos.get("y"))
                except Exception as e:
                    logger.debug(f"Could not capture post-step coordinates for history: {e}")

                # Store function result for next step's context
                if tool_calls_made:
                    last_call = tool_calls_made[-1]
                    self.runtime.publish_function_result(
                        step_number=active_step,
                        function_name=last_call['name'],
                        result_json=last_call['result'],
                    )

                if tool_calls_made:
                    self.llm_logger.add_step_tool_calls(active_step, tool_calls_made)
                self.runtime.publish_history(
                    step_number=active_step,
                    prompt=prompt,
                    response=full_response,
                    tool_calls=tool_calls_made,
                    action_details=action_details,
                    start_coords=start_coords,
                    end_coords=end_coords,
                )
                
                # Log trajectory for this step
                if run_manager and pre_state:
                    logger.debug(f"🔍 [DEBUG] Step {active_step}: Attempting to log trajectory (run_manager={run_manager is not None}, pre_state={pre_state is not None})")
                    self._log_trajectory_for_step(
                        run_manager=run_manager,
                        step_num=active_step,
                        pre_state=pre_state,
                        prompt=prompt,
                        reasoning=tool_reasoning or full_response,
                        tool_calls=tool_calls_made,
                        response=full_response
                    )
                else:
                    logger.warning(f"🔍 [DEBUG] Step {active_step}: Skipping trajectory logging (run_manager={run_manager is not None}, pre_state={pre_state is not None})")

                # Check if prompt optimization should run
                if self.optimization_enabled and self.prompt_optimizer:
                    if self.prompt_optimizer.should_optimize(active_step, self.optimization_frequency):
                        logger.info(f"🔄 Triggering prompt optimization at step {active_step}")
                        try:
                            new_base_prompt = self.prompt_optimizer.optimize_prompt(
                                current_step=active_step,
                                num_trajectory_steps=self.optimization_frequency
                            )
                            logger.info(f"✅ Base prompt optimized (new length: {len(new_base_prompt)} chars)")
                        except Exception as e:
                            logger.error(f"❌ Prompt optimization failed: {e}", exc_info=True)

                return True, full_response
            else:
                text_content = self._extract_text_from_response(response)
                if not text_content:
                    text_content = str(response)

                logger.info(f"📥 Received text response from {self.backend}:")
                logger.info(f"   {text_content}")

                duration = time.time() - start_time

                logger.info(f"✅ Step completed in {duration:.2f}s")

                self.runtime.publish_history(
                    step_number=active_step,
                    prompt=prompt,
                    response=text_content,
                    tool_calls=[],
                )
                
                # Log trajectory for this step (text response, no tool calls)
                if run_manager and pre_state:
                    logger.debug(f"🔍 [DEBUG] Step {active_step}: Attempting to log trajectory (text response)")
                    self._log_trajectory_for_step(
                        run_manager=run_manager,
                        step_num=active_step,
                        pre_state=pre_state,
                        prompt=prompt,
                        reasoning=text_content,
                        tool_calls=[],
                        response=text_content
                    )
                else:
                    logger.warning(f"🔍 [DEBUG] Step {active_step}: Skipping trajectory logging (text response, run_manager={run_manager is not None}, pre_state={pre_state is not None})")

                # Check if prompt optimization should run
                if self.optimization_enabled and self.prompt_optimizer:
                    if self.prompt_optimizer.should_optimize(active_step, self.optimization_frequency):
                        logger.info(f"🔄 Triggering prompt optimization at step {active_step}")
                        try:
                            new_base_prompt = self.prompt_optimizer.optimize_prompt(
                                current_step=active_step,
                                num_trajectory_steps=self.optimization_frequency
                            )
                            logger.info(f"✅ Base prompt optimized (new length: {len(new_base_prompt)} chars)")
                        except Exception as e:
                            logger.error(f"❌ Prompt optimization failed: {e}", exc_info=True)

                return True, text_content

            # If we reach here, no response was generated
            logger.warning("⚠️ No response from model")
            return False, "No response"

        except Exception as e:
            logger.error(f"❌ Error in agent step: {e}")
            traceback.print_exc()
            return False, str(e)

    def _wait_for_actions_complete(self, timeout: int = 30) -> None:
        """Wait for all queued actions to complete before proceeding.

        Optimization: For single actions, just wait a fixed time instead of polling.
        This avoids timeout errors when the queue_status endpoint is slow.

        Wait time is 1.0s to allow game state to stabilize after action completes.
        Actions execute faster (0.09-0.32s) but game state updates take additional time.
        """
        # First, check initial queue length
        try:
            response = requests.get(f"{self.server_url}/queue_status", timeout=5)  # Increased from 2s to 5s
            if response.status_code == 200:
                status = response.json()
                initial_queue_len = status.get("queue_length", 0)

                # If only 1-3 actions, use fixed wait instead of polling
                # Avoids timeout errors when server is under load
                if initial_queue_len <= 3:
                    wait_time = max(1.0, initial_queue_len * 0.5)  # 0.5s per action + 0.5s buffer
                    logger.info(f"⏳ {initial_queue_len} action(s) queued, waiting {wait_time:.1f}s (fixed wait)...")
                    time.sleep(wait_time)
                    logger.info("✅ Actions completed (fixed wait)")
                    return

                logger.info(f"⏳ Waiting for {initial_queue_len} actions to complete...")
        except Exception as e:
            # If we can't check queue, fall back to fixed wait
            logger.debug(f"Could not check queue length: {e}, using fixed wait")
            time.sleep(1.0)
            return

        # For 4+ actions, poll the queue with longer intervals
        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 3

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/queue_status", timeout=5)  # Increased from 2s to 5s
                if response.status_code == 200:
                    consecutive_errors = 0  # Reset error counter on success
                    status = response.json()
                    if status.get("queue_empty", False):
                        logger.info("✅ All actions completed")
                        return
                    else:
                        queue_len = status.get("queue_length", 0)
                        logger.debug(f"   Queue: {queue_len} actions remaining...")
                        time.sleep(1.0)  # Increased from 0.5s to 1.0s to reduce polling frequency
                else:
                    logger.warning(f"Failed to get queue status: {response.status_code}")
                    time.sleep(1.0)
            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"Error checking queue status ({consecutive_errors}/{max_consecutive_errors}): {e}")

                # If we get too many consecutive errors, assume actions completed
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning(f"⚠️ Too many errors, assuming actions completed")
                    return

                time.sleep(1.0)

        logger.warning(f"⚠️ Timeout waiting for actions to complete after {timeout}s")

    def _log_thinking(
        self,
        prompt: str,
        response: str,
        duration: float = None,
        tool_calls: list = None,
        step_number: Optional[int] = None,
    ) -> None:
        """Log interaction to LLM logger with full tool call history."""
        try:
            self.llm_logger.log_interaction(
                interaction_type="autonomous_gemini_cli",
                prompt=prompt,
                response=response,
                duration=duration,
                metadata={"tool_calls": tool_calls or []},
                model_info={"model": self.model},
                step_number=step_number if step_number is not None else self.step_count,
            )
            logger.debug("✅ Logged to LLM logger")
        except Exception as e:
            logger.debug(f"Could not log to LLM logger: {e}")

    def _store_function_result_for_context(self, function_name: str, result_json: str):
        """Store function result to include in next step's context."""
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

    def _get_knowledge_base_context(self, max_entries: int = 15, min_importance: int = 3) -> str:
        """
        Fetch and format knowledge base entries for inclusion in the prompt.
        Returns the N most recent entries (by importance first, then recency).

        Args:
            max_entries: Maximum number of entries to include (default 15)
            min_importance: Minimum importance level (1-5, default 3)

        Returns:
            Formatted knowledge base string for prompt injection
        """
        try:
            # Call the get_knowledge_summary tool via MCP
            kb_result = self.mcp_adapter.call_tool("get_knowledge_summary", {"min_importance": min_importance})

            if not kb_result.get("success"):
                logger.debug("Knowledge base summary not available")
                return "No knowledge entries yet. Use add_knowledge() to store important discoveries!"

            summary = kb_result.get("summary", "")

            if not summary or summary.strip() == "No knowledge entries yet.":
                return "No knowledge entries yet. Use add_knowledge() to store important discoveries!"

            # Parse the summary to limit entries
            lines = summary.split("\n")

            # Count actual entries (lines starting with "  • ")
            entry_count = sum(1 for line in lines if line.strip().startswith("•"))

            if entry_count == 0:
                return "No knowledge entries yet. Use add_knowledge() to store important discoveries!"

            # If we have too many entries, keep only the most recent N entries
            if entry_count > max_entries:
                # Collect all entries with their content
                entries = []
                current_entry = []
                in_entry = False
                category_headers = []
                header_lines = []

                for line in lines:
                    # Save header/preamble lines
                    if not line.strip().startswith("•") and not in_entry and not line.strip().startswith("["):
                        if "===" in line or "Total:" in line:
                            continue  # Skip header/footer lines
                        header_lines.append(line)
                    # Category header
                    elif line.strip().startswith("["):
                        if current_entry:
                            entries.append("\n".join(current_entry))
                            current_entry = []
                        category_headers.append(line)
                        in_entry = False
                    # Start of new entry
                    elif line.strip().startswith("•"):
                        if current_entry:
                            entries.append("\n".join(current_entry))
                        current_entry = [line]
                        in_entry = True
                    # Entry content line
                    elif in_entry:
                        current_entry.append(line)

                # Add last entry
                if current_entry:
                    entries.append("\n".join(current_entry))

                # Keep only the last N entries (most recent)
                recent_entries = entries[-max_entries:]

                # Rebuild summary with recent entries
                result_lines = []
                if header_lines:
                    result_lines.extend([h for h in header_lines if h.strip()])
                result_lines.extend(recent_entries)
                result_lines.append(f"\n(Showing {len(recent_entries)} most recent entries - {entry_count - len(recent_entries)} older entries available via get_knowledge_summary())")

                return "\n".join(result_lines)

            return summary

        except Exception as e:
            logger.warning(f"Failed to fetch knowledge base for prompt: {e}")
            return "Knowledge base temporarily unavailable."

    def _build_optimized_prompt(self, game_state_result: str, step_count: int) -> str:
        """Build optimized prompt by combining base_prompt.md with current game context.
        
        Used when prompt optimization is enabled.
        This function:
        1. Loads the base_prompt.md (strategic guidance - can be optimized)
        2. Extracts current game context (state, objectives, history)
        3. Combines them into a complete prompt for the VLM
        """
        
        # Parse game state to extract relevant information
        try:
            game_state_data = json_module.loads(game_state_result)
        except:
            game_state_data = {}
        
        # Extract key information from game state
        state_text = game_state_data.get("state_text", "")

        # Detect if in title sequence
        is_title_sequence = self._is_title_sequence(game_state_data)

        # Extract player coordinates for stuck detection
        player_position = game_state_data.get("player_position", {})
        current_coords = None
        if player_position and "x" in player_position and "y" in player_position:
            current_coords = (player_position["x"], player_position["y"])

        # Add stuck warning if detected (but not during title sequence)
        if not is_title_sequence:
            stuck_warning = self._get_stuck_warning(current_coords)
            if stuck_warning:
                state_text = stuck_warning + state_text

        # Strip map information during title sequence
        if is_title_sequence:
            state_text = self._strip_map_info(state_text)

        # Check if we're in categorized mode
        objectives_mode = game_state_data.get("objectives_mode", "legacy")

        if objectives_mode == "categorized":
            # NEW: Handle 3-category objectives
            categorized_objs = game_state_data.get("categorized_objectives", {})
            story_obj = categorized_objs.get("story")
            battling_group = categorized_objs.get("battling_group", [])
            dynamics_obj = categorized_objs.get("dynamics")
            recommended_battling = categorized_objs.get("recommended_battling_objectives", [])
            categorized_status = game_state_data.get("categorized_status", {})

            # Log whether 3-category status was received (so user can confirm in out.txt)
            if not categorized_status:
                logger.info("   - categorized_status: No (missing from get_game_state - 3-category progress will be empty)")

            # Format single objective
            def format_objective(obj_dict, category_emoji, category_name):
                if not obj_dict:
                    return f"{category_emoji} {category_name.upper()} OBJECTIVE: None"

                obj_id = obj_dict.get("id", "")
                desc = obj_dict.get("description", "")
                action_type = obj_dict.get("action_type", "")
                target_location = obj_dict.get("target_location")
                target_coords = obj_dict.get("target_coords")
                hint = obj_dict.get("navigation_hint", "")
                completion_condition = obj_dict.get("completion_condition", "")

                formatted = f"{category_emoji} {category_name.upper()} OBJECTIVE:\n  ID: {obj_id}\n  Description: {desc}"
                if action_type:
                    formatted += f"\n  Action Type: {action_type}"
                if target_location:
                    formatted += f"\n  Target Location: {target_location}"
                if target_coords:
                    formatted += f"\n  Target Coordinates: {target_coords}"
                if hint:
                    formatted += f"\n  Navigation Hint: {hint}"
                if completion_condition:
                    formatted += f"\n  Completion Condition: {completion_condition}"
                return formatted

            # Format battling objectives group
            def format_battling_group(group_list):
                if not group_list:
                    return "⚔️  BATTLING OBJECTIVES: None"

                formatted = f"⚔️  BATTLING OBJECTIVES ({len(group_list)} in current group):"
                for i, obj_dict in enumerate(group_list, 1):
                    obj_id = obj_dict.get("id", "")
                    desc = obj_dict.get("description", "")
                    formatted += f"\n  [{i}] {obj_id}: {desc}"

                    # Add details for first objective only to keep prompt concise
                    if i == 1:
                        target_location = obj_dict.get("target_location")
                        hint = obj_dict.get("navigation_hint", "")
                        if target_location:
                            formatted += f"\n      Target Location: {target_location}"
                        if hint:
                            formatted += f"\n      Hint: {hint}"

                formatted += "\n\n  💡 TIP: Complete these in any order. Use complete_direct_objective(category=\"battling\") for each."
                return formatted

            # Format recommended battling objectives if they exist
            recommended_text = ""
            if recommended_battling:
                recommended_text = f"\n\n💡 RECOMMENDED PREPARATION:\n  Complete these battling objectives before the story objective:\n  → {', '.join(recommended_battling)}"

            direct_objective = f"""{format_objective(story_obj, "📖", "story")}{recommended_text}

{format_battling_group(battling_group)}

{format_objective(dynamics_obj, "🎯", "dynamics")}"""

            # Format categorized status
            if categorized_status:
                story_status = categorized_status.get("story", {})
                battling_status = categorized_status.get("battling", {})
                dynamics_status = categorized_status.get("dynamics", {})

                direct_objective_status = f"""📊 PROGRESS (3 Categories):
  📖 Story: {story_status.get('current_index', 0) + 1}/{story_status.get('total', 0)} ({story_status.get('completed', 0)} completed)
  ⚔️  Battling: {battling_status.get('current_index', 0) + 1}/{battling_status.get('total', 0)} ({battling_status.get('completed', 0)} completed)
  🎯 Dynamics: {dynamics_status.get('current_index', 0) + 1}/{dynamics_status.get('total', 0)} ({dynamics_status.get('completed', 0)} completed)"""
            else:
                direct_objective_status = ""

            direct_objective_context = ""  # No context needed in categorized mode

        else:
            # LEGACY: Single objective mode (backward compatible)
            direct_objective = game_state_data.get("direct_objective", "")
            direct_objective_status = game_state_data.get("direct_objective_status", "")
            direct_objective_context = game_state_data.get("direct_objective_context", "")

            # Format direct objective nicely if it's a dict - show ALL fields
            if isinstance(direct_objective, dict):
                obj_id = direct_objective.get("id", "")
                desc = direct_objective.get("description", "")
                action_type = direct_objective.get("action_type", "")
                target_location = direct_objective.get("target_location")
                target_coords = direct_objective.get("target_coords")
                hint = direct_objective.get("navigation_hint", "")
                completion_condition = direct_objective.get("completion_condition", "")

                formatted_obj = f"🎯 CURRENT OBJECTIVE:\n  ID: {obj_id}\n  Description: {desc}"
                if action_type:
                    formatted_obj += f"\n  Action Type: {action_type}"
                if target_location:
                    formatted_obj += f"\n  Target Location: {target_location}"
                if target_coords:
                    formatted_obj += f"\n  Target Coordinates: {target_coords}"
                if hint:
                    formatted_obj += f"\n  Navigation Hint: {hint}"
                if completion_condition:
                    formatted_obj += f"\n  Completion Condition: {completion_condition}"
                direct_objective = formatted_obj

            # Format status nicely if it's a dict
            if isinstance(direct_objective_status, dict):
                seq = direct_objective_status.get("sequence_name", "")
                total = direct_objective_status.get("total_objectives", 0)
                current_idx = direct_objective_status.get("current_index", 0)
                completed = direct_objective_status.get("completed_count", 0)
                direct_objective_status = f"📊 PROGRESS: Objective {current_idx + 1}/{total} in sequence '{seq}' ({completed} completed)"

        # Build action history summary for better context
        action_history = self._format_action_history()

        # Get function results from previous step
        function_results_context = self._get_function_results_context()

        # Get knowledge base summary for context
        knowledge_context = self._get_knowledge_base_context(max_entries=15, min_importance=3)

        # Load base prompt (strategic guidance - can be optimized)
        base_prompt = self._load_base_prompt()

        # Log component sizes BEFORE building prompt
        logger.info(f"📏 Pre-prompt component sizes:")
        logger.info(f"   base_prompt: {len(base_prompt):,} chars")
        logger.info(f"   state_text: {len(state_text):,} chars")
        logger.info(f"   action_history: {len(action_history):,} chars")
        logger.info(f"   function_results: {len(function_results_context):,} chars")
        logger.info(f"   knowledge_base: {len(knowledge_context):,} chars")
        logger.info(f"   direct_objective: {len(str(direct_objective)):,} chars")
        logger.info(f"   direct_objective_context: {len(direct_objective_context):,} chars")
        logger.info(f"   direct_objective_status: {len(direct_objective_status):,} chars")

        # Build complete prompt by combining base prompt with context
        prompt = f"""# Current Step: {step_count}

{base_prompt}

## CONTEXT FOR THIS STEP

### ACTION HISTORY (Recent Steps):
{action_history}
{function_results_context}

### CURRENT DIRECT OBJECTIVE:
{direct_objective_context}

{direct_objective}

{direct_objective_status}

⚠️ **CRITICAL**: When you complete the objective, IMMEDIATELY call:
   complete_direct_objective(category="<story/battling/dynamics>", reasoning="<explain why it's complete>")

### CURRENT GAME STATE:
{state_text}

### KNOWLEDGE BASE - What You've Learned:
{knowledge_context}

Step {step_count}"""

        # Log prompt size breakdown to debug token issues
        prompt_size = len(prompt)
        state_size = len(state_text)
        history_size = len(action_history)
        function_results_size = len(function_results_context)
        context_size = len(direct_objective_context)
        objective_size = len(str(direct_objective))
        status_size = len(direct_objective_status)

        # Calculate static instruction size (approximate)
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

    def _build_structured_prompt(self, game_state_result: str, step_count: int) -> str:
        """Build an autonomous prompt that emphasizes creating your own objectives."""

        # Parse game state to extract relevant information
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

        # Check if we're in categorized mode
        objectives_mode = game_state_data.get("objectives_mode", "legacy")
        logger.info(f"🎯 Objectives mode: {objectives_mode}")

        if objectives_mode == "categorized":
            # NEW: Handle 3-category objectives
            categorized_objs = game_state_data.get("categorized_objectives", {})
            story_obj = categorized_objs.get("story")
            battling_group = categorized_objs.get("battling_group", [])
            dynamics_obj = categorized_objs.get("dynamics")
            recommended_battling = categorized_objs.get("recommended_battling_objectives", [])
            categorized_status = game_state_data.get("categorized_status", {})

            # DEBUG: Log what we received
            logger.info(f"📊 Categorized objectives received:")
            logger.info(f"   - story_obj: {'Yes' if story_obj else 'None'}")
            logger.info(f"   - battling_group: {len(battling_group)} objectives")
            logger.info(f"   - dynamics_obj: {'Yes' if dynamics_obj else 'None'}")
            logger.info(f"   - recommended: {len(recommended_battling)} IDs")
            logger.info(f"   - categorized_status: {'Yes' if categorized_status else 'No (missing from get_game_state)'}")

            # Format single objective
            def format_objective(obj_dict, category_emoji, category_name):
                if not obj_dict:
                    return f"{category_emoji} {category_name.upper()} OBJECTIVE: None"

                obj_id = obj_dict.get("id", "")
                desc = obj_dict.get("description", "")
                action_type = obj_dict.get("action_type", "")
                target_location = obj_dict.get("target_location")
                target_coords = obj_dict.get("target_coords")
                hint = obj_dict.get("navigation_hint", "")
                completion_condition = obj_dict.get("completion_condition", "")

                formatted = f"{category_emoji} {category_name.upper()} OBJECTIVE:\n  ID: {obj_id}\n  Description: {desc}"
                if action_type:
                    formatted += f"\n  Action Type: {action_type}"
                if target_location:
                    formatted += f"\n  Target Location: {target_location}"
                if target_coords:
                    formatted += f"\n  Target Coordinates: {target_coords}"
                if hint:
                    formatted += f"\n  Navigation Hint: {hint}"
                if completion_condition:
                    formatted += f"\n  Completion Condition: {completion_condition}"
                return formatted

            # Format battling objectives group
            def format_battling_group(group_list):
                if not group_list:
                    return "⚔️  BATTLING OBJECTIVES: None"

                formatted = f"⚔️  BATTLING OBJECTIVES ({len(group_list)} in current group):"
                for i, obj_dict in enumerate(group_list, 1):
                    obj_id = obj_dict.get("id", "")
                    desc = obj_dict.get("description", "")
                    formatted += f"\n  [{i}] {obj_id}: {desc}"

                    # Add details for first objective only to keep prompt concise
                    if i == 1:
                        target_location = obj_dict.get("target_location")
                        hint = obj_dict.get("navigation_hint", "")
                        if target_location:
                            formatted += f"\n      Target Location: {target_location}"
                        if hint:
                            formatted += f"\n      Hint: {hint}"

                formatted += "\n\n  💡 TIP: Complete these in any order. Use complete_direct_objective(category=\"battling\") for each."
                return formatted

            # Format recommended battling objectives if they exist
            recommended_text = ""
            if recommended_battling:
                recommended_text = f"\n\n💡 RECOMMENDED PREPARATION:\n  Complete these battling objectives before the story objective:\n  → {', '.join(recommended_battling)}"

            direct_objective = f"""{format_objective(story_obj, "📖", "story")}{recommended_text}

{format_battling_group(battling_group)}

{format_objective(dynamics_obj, "🎯", "dynamics")}"""

            # Format categorized status
            if categorized_status:
                story_status = categorized_status.get("story", {})
                battling_status = categorized_status.get("battling", {})
                dynamics_status = categorized_status.get("dynamics", {})

                direct_objective_status = f"""📊 PROGRESS (3 Categories):
  📖 Story: {story_status.get('current_index', 0) + 1}/{story_status.get('total', 0)} ({story_status.get('completed', 0)} completed)
  ⚔️  Battling: {battling_status.get('current_index', 0) + 1}/{battling_status.get('total', 0)} ({battling_status.get('completed', 0)} completed)
  🎯 Dynamics: {dynamics_status.get('current_index', 0) + 1}/{dynamics_status.get('total', 0)} ({dynamics_status.get('completed', 0)} completed)"""
                logger.info(f"📊 PROGRESS (3 Categories) in prompt: Story {story_status.get('current_index', 0) + 1}/{story_status.get('total', 0)}, Battling {battling_status.get('current_index', 0) + 1}/{battling_status.get('total', 0)}, Dynamics {dynamics_status.get('current_index', 0) + 1}/{dynamics_status.get('total', 0)}")
            else:
                direct_objective_status = ""

            direct_objective_context = ""  # No context needed in categorized mode

        else:
            # LEGACY: Single objective mode (backward compatible)
            direct_objective = game_state_data.get("direct_objective", "")
            direct_objective_status = game_state_data.get("direct_objective_status", "")
            direct_objective_context = game_state_data.get("direct_objective_context", "")

            # Format direct objective nicely if it's a dict - show ALL fields
            if isinstance(direct_objective, dict):
                obj_id = direct_objective.get("id", "")
                desc = direct_objective.get("description", "")
                action_type = direct_objective.get("action_type", "")
                target_location = direct_objective.get("target_location")
                target_coords = direct_objective.get("target_coords")
                hint = direct_objective.get("navigation_hint", "")
                completion_condition = direct_objective.get("completion_condition", "")

                formatted_obj = f"🎯 CURRENT OBJECTIVE:\n  ID: {obj_id}\n  Description: {desc}"
                if action_type:
                    formatted_obj += f"\n  Action Type: {action_type}"
                if target_location:
                    formatted_obj += f"\n  Target Location: {target_location}"
                if target_coords:
                    formatted_obj += f"\n  Target Coordinates: {target_coords}"
                if hint:
                    formatted_obj += f"\n  Navigation Hint: {hint}"
                if completion_condition:
                    formatted_obj += f"\n  Completion Condition: {completion_condition}"
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

        # Get knowledge base summary for context
        knowledge_context = self._get_knowledge_base_context(max_entries=15, min_importance=3)

        # Build autonomous prompt
        prompt = f"""# Step: {step_count}
### SHORT-TERM MEMORY (last 10 steps)
{action_history}
{function_results_context}
### OBJECTIVES
{direct_objective_context}
{direct_objective}
{direct_objective_status}
### STATE
{state_text}
### KNOWLEDGE BASE
{knowledge_context}
"""

        return prompt


    def _is_title_sequence(self, game_state_data: Dict[str, Any]) -> bool:
        """Detect if in title sequence"""
        # Check location (returned by MCP get_game_state)
        location = game_state_data.get("location", "")
        if location == "TITLE_SEQUENCE":
            return True

        # Check if state_text contains player name (more reliable than parsing JSON)
        state_text = game_state_data.get("state_text", "")
        if "Player Name:" in state_text:
            # Extract player name from state text
            import re
            match = re.search(r"Player Name:\s*(\S+)", state_text)
            if match:
                player_name = match.group(1).strip()
                if not player_name or player_name == "????????":
                    return True
            else:
                # No player name found in text
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
        """Check if frame is a black screen (transition)

        Uses both mean brightness AND variance to avoid false positives in caves.
        A true black transition frame has very low brightness AND very low variance.
        A cave scene has low brightness but higher variance (lit areas vs dark areas).
        """
        try:
            if hasattr(image, 'save'):  # PIL Image
                frame_array = np.array(image)
            else:
                frame_array = image

            mean_brightness = frame_array.mean()
            std_brightness = frame_array.std()

            # True black frame: very low brightness AND very low variance
            brightness_threshold = 10
            variance_threshold = 5  # Low variance = uniform darkness

            is_black = (mean_brightness < brightness_threshold and
                       std_brightness < variance_threshold)

            if is_black:
                logger.debug(f"Black frame detected: mean={mean_brightness:.2f}, std={std_brightness:.2f}")
            elif mean_brightness < brightness_threshold:
                logger.debug(f"Dark frame (cave?) but not black: mean={mean_brightness:.2f}, std={std_brightness:.2f}")

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
    
    def _log_trajectory_for_step(self, run_manager, step_num: int, pre_state: dict, 
                                  prompt: str, reasoning: str, tool_calls: list, response: str):
        """Log trajectory for a CLI agent step
        
        Args:
            run_manager: RunDataManager instance
            step_num: Step number
            pre_state: Pre-state snapshot
            prompt: Prompt sent to LLM
            reasoning: Reasoning from LLM
            tool_calls: List of tool calls made
            response: Full response from LLM
        """
        try:
            # Get post-state after tool calls
            try:
                game_state_result = self.mcp_adapter.call_tool("get_game_state", {})
                if isinstance(game_state_result, str):
                    game_state_result = json.loads(game_state_result)
                if game_state_result.get("success"):
                    raw_state = game_state_result.get("raw_state", {})
                    post_state = run_manager.create_state_snapshot(raw_state)
                else:
                    post_state = pre_state  # Fallback to pre-state
            except Exception as e:
                logger.debug(f"Could not capture post-state: {e}")
                post_state = pre_state  # Fallback to pre-state
            
            # Build action dict from tool calls
            action = {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "name": tc.get("name"),
                        "args": {k: v for k, v in tc.get("args", {}).items() 
                                if k not in ["screenshot_base64"]}  # Exclude large base64 data
                    }
                    for tc in tool_calls
                ],
                "total_tool_calls": len(tool_calls)
            }
            
            # Determine outcome
            outcome = {
                "success": True,
                "objectives_completed": []
            }
            
            # Check if location/coordinates changed
            if pre_state.get("location") != post_state.get("location"):
                outcome["observations"] = f"Moved from {pre_state.get('location')} to {post_state.get('location')}"
            elif pre_state.get("player_coords") != post_state.get("player_coords"):
                outcome["observations"] = f"Position changed from {pre_state.get('player_coords')} to {post_state.get('player_coords')}"
            else:
                outcome["observations"] = "No significant state change"
            
            # Log trajectory
            logger.debug(f"🔍 [DEBUG] Calling run_manager.log_trajectory for step {step_num}")
            run_manager.log_trajectory(
                step=step_num,
                reasoning=reasoning,
                action=action,
                pre_state=pre_state,
                post_state=post_state,
                outcome=outcome,
                llm_prompt=prompt
            )
            logger.info(f"🔍 [DEBUG] Successfully logged trajectory for step {step_num}")
        except Exception as e:
            logger.error(f"🔍 [DEBUG] Failed to log trajectory at step {step_num}: {e}")
            logger.error(traceback.format_exc())

    def _format_action_history(self) -> str:
        """Format short-term memory (last 20 steps) with full tool details."""
        if not self.conversation_history:
            return "No previous actions recorded."

        recent_entries = self.conversation_history[-20:]

        history_lines = []
        for entry in recent_entries:
            step = entry.get("step", "?")
            llm_response = entry.get("llm_response", "").strip()
            start_coords = entry.get("start_coords")
            end_coords = entry.get("end_coords")
            coords = entry.get("player_coords", None)

            if start_coords is None and coords is not None:
                start_coords = coords
            if end_coords is None:
                end_coords = coords

            start_str = f"({start_coords[0]},{start_coords[1]})" if start_coords else "(?)"
            end_str = f"({end_coords[0]},{end_coords[1]})" if end_coords else "(?)"

            history_lines.append(f"[{step}] start={start_str} end={end_str}")
            if llm_response:
                history_lines.append(f"  THINKING: {llm_response}")

            tool_calls = entry.get("tool_calls", [])
            if tool_calls:
                history_lines.append("  TOOLS:")
                for tool_call in tool_calls:
                    name = tool_call.get("name", "unknown")
                    args = tool_call.get("args", {})
                    result = tool_call.get("result", "")
                    history_lines.append(f"    - {name}")
                    history_lines.append(f"      args: {json.dumps(args, ensure_ascii=False)}")
                    if result != "":
                        try:
                            history_lines.append(f"      result: {json.dumps(result, ensure_ascii=False)}")
                        except TypeError:
                            history_lines.append(f"      result: {str(result)}")
            history_lines.append("")

        return "\n".join(history_lines).strip()

    def run(self) -> int:
        """Run the autonomous agent loop."""
        # Clear conversation history
        self.conversation_history = []
        logger.info("🧹 Cleared conversation history (fresh start)")

        logger.info("=" * 70)
        logger.info("🎮 Pokemon Emerald PokeAgent")
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

                next_step = self.runtime.peek_next_step()

                logger.info(f"\n{'='*70}")
                logger.info(f"🤖 Step {next_step}")
                context_size = self._calculate_context_size()
                logger.info(f"📚 History: {len(self.conversation_history)} turns ({context_size:,} chars)")
                logger.info(f"{'='*70}")

                # Fetch game state
                game_state_result = self._execute_function_call_by_name("get_game_state", {})

                # Parse for screenshot
                try:
                    game_state_data = json_module.loads(game_state_result)
                    screenshot_b64 = game_state_data.get("screenshot_base64")
                except:
                    screenshot_b64 = None

                # Build prompt (conditional based on optimization mode)
                if self.optimization_enabled:
                    prompt = self._build_optimized_prompt(game_state_result, next_step)
                else:
                    prompt = self._build_structured_prompt(game_state_result, next_step)

                # Run step
                success, output = self.run_step(prompt, screenshot_b64=screenshot_b64, step_number=next_step)

                if not success:
                    logger.warning("⚠️ Step failed, waiting 5 seconds before retry...")
                    time.sleep(5)
                    continue

                logger.info(f"✅ Global step counter now {self.step_count}")

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
            traceback.print_exc()
            return 1

        logger.info(f"\n🎯 Agent completed {self.step_count} steps")
        logger.info(f"📚 Conversation history: {len(self.conversation_history)} turns")
        logger.info(self._format_history_for_display())
        return 0

    def _handle_vlm_function_calls(
        self,
        response,
        tool_calls_made,
        tool_call_count,
        max_tool_calls,
        allowed_tool_names: Optional[set[str]] = None,
    ):
        """Handle function calls from VLM backend

        Function results are stored and will be included in the next step's context,
        allowing the agent to see and use the results in subsequent decisions.
        """
        if not hasattr(response, 'candidates') or not response.candidates:
            logger.warning("⚠️ Response has no candidates")
            return False

        candidate = response.candidates[0]
        
        # Check finish_reason first - this tells us if there was a problem
        if hasattr(candidate, 'finish_reason'):
            finish_reason = candidate.finish_reason
            if finish_reason and finish_reason != 1:  # 1 = STOP (normal)
                logger.error(f"🚨 FINISH_REASON indicates problem: {finish_reason}")
                logger.error(f"   finish_reason value: {finish_reason}")
                logger.error(f"   finish_reason type: {type(finish_reason)}")
                # Try to get more info about the finish reason
                try:
                    if hasattr(genai_types, 'FinishReason'):
                        logger.error(f"   FinishReason enum values: {[e.name for e in genai_types.FinishReason]}")
                except:
                    pass
        
        if not hasattr(candidate, 'content') or not candidate.content:
            logger.warning("⚠️ Candidate has no content")
            return False

        content = candidate.content
        if not hasattr(content, 'parts'):
            logger.warning("⚠️ Response content has no 'parts' attribute")
            return False

        logger.info(f"🔍 Checking {len(content.parts)} response parts for function calls")
        function_calls_found = False
        for i, part in enumerate(content.parts):
            logger.debug(f"   Part {i}: has function_call={hasattr(part, 'function_call')}, has text={hasattr(part, 'text')}")
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                tool_call_count += 1
                logger.info(f"🔧 VLM wants to call: {function_call.name} ({tool_call_count}/{max_tool_calls})")
                
                # Log detailed function call information
                logger.info(f"   📋 Function call details:")
                logger.info(f"      Name: {function_call.name}")
                logger.info(f"      Args attribute exists: {hasattr(function_call, 'args')}")
                
                # Try to extract and log arguments
                try:
                    if hasattr(function_call, 'args'):
                        logger.info(f"      Args type: {type(function_call.args)}")
                        # Try to convert to dict
                        try:
                            args_dict = self._convert_protobuf_args(function_call.args)
                            logger.info(f"      ✅ Successfully converted args to dict")
                            logger.info(f"      Args keys: {list(args_dict.keys())}")
                            
                            # Special logging for create_direct_objectives
                            if function_call.name == "create_direct_objectives":
                                logger.info(f"      🎯 create_direct_objectives args analysis:")
                                if "objectives" in args_dict:
                                    obj_list = args_dict["objectives"]
                                    logger.info(f"         objectives type: {type(obj_list)}")
                                    logger.info(f"         objectives is list: {isinstance(obj_list, list)}")
                                    if isinstance(obj_list, list):
                                        logger.info(f"         objectives length: {len(obj_list)}")
                                        for j, obj in enumerate(obj_list[:3]):  # Log first 3
                                            logger.info(f"         Objective {j}: type={type(obj)}, keys={list(obj.keys()) if isinstance(obj, dict) else 'NOT DICT'}")
                                    else:
                                        logger.error(f"         ⚠️ objectives is NOT a list! Type: {type(obj_list)}")
                                        logger.error(f"         Value: {str(obj_list)[:500]}")
                                else:
                                    logger.error(f"         ⚠️ 'objectives' key not found in args!")
                                    logger.error(f"         Available keys: {list(args_dict.keys())}")
                                if "reasoning" in args_dict:
                                    reasoning = args_dict["reasoning"]
                                    logger.info(f"         reasoning length: {len(str(reasoning))} chars")
                        except Exception as e:
                            logger.error(f"      ❌ Failed to convert args: {e}")
                            logger.error(f"      Args raw value: {str(function_call.args)[:500]}")
                            logger.error(f"      Traceback: {traceback.format_exc()}")
                    else:
                        logger.warning(f"      ⚠️ Function call has no 'args' attribute")
                except Exception as e:
                    logger.error(f"      ❌ Error examining function call: {e}")
                    logger.error(f"      Traceback: {traceback.format_exc()}")

                # Execute the function
                try:
                    function_result = self._execute_function_call(
                        function_call,
                        allowed_tool_names=allowed_tool_names,
                    )
                    result_str = str(function_result)
                    logger.info(f"📥 Function result: {result_str[:200]}...")
                except Exception as e:
                    logger.error(f"❌ Error executing function call: {e}")
                    logger.error(f"   Traceback: {traceback.format_exc()}")
                    # Still try to track the failed call
                    try:
                        tool_calls_made.append({
                            "name": function_call.name,
                            "args": self._convert_protobuf_args(function_call.args) if hasattr(function_call, 'args') else {},
                            "result": json.dumps({"success": False, "error": str(e)}),
                            "error": str(e)
                        })
                    except:
                        pass
                    raise

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
                text = response.text.strip()
                if text:
                    return text
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    text_parts = [part.text for part in parts if hasattr(part, "text") and part.text]
                    if text_parts:
                        return "\n".join(text_parts).strip()
            return ""
        except:
            return ""


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Pokemon Emerald PokeAgent")
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
        default=None,
        help=(
            "Repo-relative path to system instructions markdown. "
            "Omit for automatic selection: POKEAGENT.md by default, or system_prompt.md when "
            "--enable-prompt-optimization is set (fails at startup if run_data_manager is missing)."
        ),
    )
    parser.add_argument(
        "--enable-prompt-optimization",
        action="store_true",
        help=(
            "Lean system prompt + optimizable base prompt. Requires initialized run_data_manager "
            "(e.g. via run.py); otherwise construction raises RuntimeError."
        ),
    )
    parser.add_argument(
        "--optimization-frequency",
        type=int,
        default=10,
        help="Run prompt optimization every N steps when optimization is enabled.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        help="VLM backend (gemini, vertex, openai, etc.)"
    )

    args = parser.parse_args()

    agent_kwargs = {
        "server_url": args.server_url,
        "model": args.model,
        "backend": args.backend,
        "max_steps": args.max_steps,
        "enable_prompt_optimization": args.enable_prompt_optimization,
        "optimization_frequency": args.optimization_frequency,
    }
    if args.system_instructions is not None:
        agent_kwargs["system_instructions_file"] = args.system_instructions

    agent = PokeAgent(**agent_kwargs)

    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
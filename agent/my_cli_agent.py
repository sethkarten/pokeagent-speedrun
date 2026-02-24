#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import requests
import base64
import io
import threading
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import google.generativeai as genai
from utils.agent_helpers import update_server_metrics
from utils.llm_logger import get_llm_logger
from utils.vlm_backends import VLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MCPToolAdapter:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            endpoint_map = {
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
                "save_memory": "/mcp/save_memory",
            }
            endpoint = endpoint_map.get(tool_name)
            if not endpoint:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
            url = f"{self.server_url}{endpoint}"
            logger.info(f"🔧 Calling MCP tool: {tool_name}")
            response = requests.post(url, json=arguments, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Tool {tool_name} failed: {e}")
            return {"success": False, "error": str(e)}


class MyCLIAgent:
    def __init__(
        self,
        server_url="http://localhost:8000",
        model="gemini-2.5-flash",
        backend="gemini",
        max_steps=None,
        system_instructions_file=None,
        max_context_chars=100000,
        target_context_chars=50000,
        enable_prompt_optimization=False,
        optimization_frequency=10,
    ):
        self._game_type = os.environ.get("GAME_TYPE", "emerald")
        if system_instructions_file is None:
            system_instructions_file = (
                "agent/prompts/POKEAGENT_RED.md" if self._game_type == "red"
                else "agent/prompts/POKEAGENT.md"
            )
        print(f"🚀 Initializing MyCLIAgent with backend={backend}, model={model}, server={server_url}, game={self._game_type}")
        self.server_url, self.model, self.backend, self.max_steps = server_url, model, backend, max_steps
        self.step_count, self.max_context_chars, self.target_context_chars = 0, max_context_chars, target_context_chars
        self.optimization_enabled, self.optimization_frequency = enable_prompt_optimization, optimization_frequency
        self.conversation_history, self.frame_buffer, self.max_frame_buffer_size = [], [], 10
        self.frame_buffer_lock, self.sampling_interval, self.stop_sampling = threading.Lock(), 1.0, threading.Event()
        self.recent_function_results = []
        self.defeated_trainers = set()
        self.blocked_coords = set()
        if self._game_type != "red":
            self.turnstile_states = {
                (15, 21): "H",
                (13, 5): "H",
                (5, 6): "H",
                (9, 11): "H",
                (9, 13): "V",
                (7, 17): "H",
                (4, 18): "V",
                (4, 3): "H",
                (12, 13): "H",
                (12, 11): "V",
            }
        else:
            self.turnstile_states = {}

        self.system_instructions = self._load_system_instructions(system_instructions_file)
        self.mcp_adapter = MCPToolAdapter(server_url)
        self.tools = self._create_tool_declarations()
        self.vlm = VLM(
            backend=self.backend, model_name=self.model, tools=self.tools, system_instruction=self.system_instructions
        )
        self.llm_logger = get_llm_logger()
        self.sampling_thread = threading.Thread(target=self._sample_frames_loop, daemon=True)
        self.sampling_thread.start()

    def _load_system_instructions(self, f):
        p = Path(__file__).resolve().parent.parent / f
        game_name = "Pokemon Red" if self._game_type == "red" else "Pokemon Emerald"
        return p.read_text() if p.exists() else f"AI agent playing {game_name}."

    def _load_base_prompt(self):
        prompt_file = "base_prompt_red.md" if self._game_type == "red" else "base_prompt.md"
        p = Path(__file__).resolve().parent.parent / "agent" / "prompts" / prompt_file
        return p.read_text() if p.exists() else "Make intelligent decisions."

    def _sample_frames_loop(self):
        while not self.stop_sampling.is_set():
            try:
                r = requests.get(f"{self.server_url}/screenshot", timeout=2)
                if r.status_code == 200:
                    b64 = r.json().get("screenshot_base64")
                    if b64:
                        img = Image.open(io.BytesIO(base64.b64decode(b64)))
                        with self.frame_buffer_lock:
                            self.frame_buffer.append(img)
                            if len(self.frame_buffer) > self.max_frame_buffer_size:
                                self.frame_buffer.pop(0)
                time.sleep(self.sampling_interval)
            except:
                time.sleep(2)

    def _create_tool_declarations(self):
        return [
            {
                "name": "get_game_state",
                "description": "Get the current game state including player position, party Pokemon, map, items, and a screenshot. Use this to understand where you are and what you can do.",
                "parameters": {"type_": "OBJECT", "properties": {}, "required": []},
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
                            "description": "List of buttons to press (e.g., ['A'], ['UP'])",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "REQUIRED FORMAT: Must include 'ANALYZE: [game screen, location, objective, situation]' and 'PLAN: [action, reason, expected result]'.",
                        },
                    },
                    "required": ["buttons", "reasoning"],
                },
            },
            {
                "name": "navigate_to",
                "description": "Automatically navigate to specific coordinates using A* pathfinding. IMPORTANT: Always specify the variance parameter.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "x": {"type_": "INTEGER", "description": "Target X coordinate"},
                        "y": {"type_": "INTEGER", "description": "Target Y coordinate"},
                        "variance": {
                            "type_": "STRING",
                            "description": "Path variance level: 'none', 'low', 'medium', 'high', 'extreme'.",
                            "enum": ["none", "low", "medium", "high", "extreme"],
                        },
                        "reason": {
                            "type_": "STRING",
                            "description": "REQUIRED FORMAT: Must include 'ANALYZE:' and 'PLAN:' sections.",
                        },
                        "blocked_coords": {
                            "type_": "ARRAY",
                            "items": {"type_": "ARRAY", "items": {"type_": "INTEGER"}},
                        },
                        "consider_npcs": {
                            "type_": "BOOLEAN",
                            "description": "Whether to treat NPCs as obstacles during pathfinding.",
                        },
                    },
                    "required": ["x", "y", "variance", "reason"],
                },
            },
            {
                "name": "complete_direct_objective",
                "description": "Complete the current direct objective and advance to the next one. In CATEGORIZED mode, you must specify which category objective to complete (story, battling, or dynamics). In LEGACY mode, category is ignored.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "reasoning": {
                            "type_": "STRING",
                            "description": "REQUIRED FORMAT: Must include 'ANALYZE: [current state, objective requirements, completion evidence]' and 'PLAN: [confirm completion, next objective]'.",
                        },
                        "category": {
                            "type_": "STRING",
                            "enum": ["story", "battling", "dynamics"],
                            "description": "Which category objective to complete (required in CATEGORIZED mode).",
                        },
                    },
                    "required": ["reasoning"],
                },
            },
            {
                "name": "reflect",
                "description": "Use this when you feel stuck, uncertain, or suspect your current approach/objectives are wrong. This tool helps you step back, analyze what's happening, and realign your strategy.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "situation": {
                            "type_": "STRING",
                            "description": "Describe what you've been trying to do and why you think something might be wrong.",
                        }
                    },
                    "required": ["situation"],
                },
            },
            {
                "name": "gym_puzzle_agent",
                "description": "Get expert guidance on solving gym puzzles. Use this when you're in a gym and need help understanding the puzzle mechanics or finding the solution.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "gym_name": {
                            "type_": "STRING",
                            "description": "Name of the gym you're currently in (e.g., 'MOSSDEEP_CITY_GYM').",
                        }
                    },
                    "required": ["gym_name"],
                },
            },
            {
                "name": "add_knowledge",
                "description": "Store important discoveries in your knowledge base.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "category": {
                            "type_": "STRING",
                            "enum": ["location", "npc", "item", "pokemon", "strategy", "custom"],
                            "description": "Category of knowledge",
                        },
                        "title": {"type_": "STRING", "description": "Short title"},
                        "content": {"type_": "STRING", "description": "Detailed content"},
                        "location": {"type_": "STRING", "description": "Map name (optional)"},
                        "coordinates": {"type_": "STRING", "description": "Coordinates as 'x,y' (optional)"},
                        "importance": {"type_": "INTEGER", "description": "Importance 1-5"},
                    },
                    "required": ["category", "title", "content", "importance"],
                },
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
                        "min_importance": {"type_": "INTEGER", "description": "Min importance 1-5 (optional)"},
                    },
                    "required": [],
                },
            },
            {
                "name": "get_knowledge_summary",
                "description": "Get a summary of the most important things you've learned.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "min_importance": {"type_": "INTEGER", "description": "Min importance (default 3)"}
                    },
                    "required": [],
                },
            },
            {
                "name": "create_direct_objectives",
                "description": "Create the next 3 direct objectives when you need new goals. In LEGACY mode, creates general objectives. In CATEGORIZED mode, you MUST choose a category (story, battling, or dynamics). Use 'story' for main walkthrough progression, 'battling' for training prep, and 'dynamics' for ad-hoc navigation or cleanup.",
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
                                        "description": "Type of action",
                                    },
                                    "target_location": {"type_": "STRING", "description": "Target location/map name"},
                                    "navigation_hint": {"type_": "STRING", "description": "Specific guidance on how to accomplish this"},
                                    "completion_condition": {"type_": "STRING", "description": "How to verify completion (e.g., 'location_contains_route_102')"},
                                },
                                "required": ["id", "description", "action_type"],
                            },
                            "description": "Array of exactly 3 objectives to create next",
                        },
                        "category": {
                            "type_": "STRING",
                            "enum": ["dynamics", "story", "battling"],
                            "description": "Category for objectives: 'story' (walkthrough progression), 'battling' (training/prep), or 'dynamics' (short-term navigation/cleanup). Choose the category that matches the goal.",
                        },
                        "reasoning": {
                            "type_": "STRING",
                            "description": "Explanation of why these objectives were chosen (referencing walkthrough/wiki sources)",
                        },
                    },
                    "required": ["objectives", "reasoning"],
                },
            },
            {
                "name": "get_progress_summary",
                "description": "Get comprehensive progress summary including completed milestones, objectives, current location, and knowledge base summary.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_walkthrough",
                "description": (
                    "Get official Red walkthrough (Parts 1-17). Part 1: Pallet Town, Part 2: Viridian City, Part 16: Indigo Plateau (Elite Four)."
                    if self._game_type == "red" else
                    "Get official Emerald walkthrough (Parts 1-21). Part 1: Littleroot, Part 6: Roxanne, Part 21: Elite Four."
                ),
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "part": {
                            "type_": "INTEGER",
                            "description": (
                                "Walkthrough part 1-17" if self._game_type == "red"
                                else "Walkthrough part 1-21"
                            ),
                        }
                    },
                    "required": ["part"],
                },
            },
            {
                "name": "lookup_pokemon_info",
                "description": "Look up Pokemon information from Bulbapedia (stats, moves, evolution, locations).",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "topic": {
                            "type_": "STRING",
                            "description": "Pokemon name or topic to look up",
                        },
                        "source": {
                            "type_": "STRING",
                            "description": "Wiki source (default: bulbapedia)",
                        },
                    },
                    "required": ["topic"],
                },
            },
            {
                "name": "list_wiki_sources",
                "description": "List all available wiki article sources.",
                "parameters": {"type_": "OBJECT", "properties": {}, "required": []},
            },
            {
                "name": "save_memory",
                "description": "Save important facts to remember across sessions.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {"fact": {"type_": "STRING", "description": "Fact to remember"}},
                    "required": ["fact"],
                },
            },
        ]

    def _execute_function_call_by_name(self, n, a):
        return json.dumps(self.mcp_adapter.call_tool(n, a))

    def _execute_function_call(self, fc):
        return self._execute_function_call_by_name(fc.name, self._convert_protobuf_args(fc.args))

    def _convert_protobuf_args(self, proto_args):
        result = {}
        for k, v in proto_args.items():
            if hasattr(v, "items"):
                result[k] = self._convert_protobuf_args(v)
            elif hasattr(v, "__iter__") and not isinstance(v, (str, dict)):
                converted_list = []
                for item in v:
                    if hasattr(item, "items"):
                        converted_list.append(self._convert_protobuf_args(item))
                    else:
                        converted_list.append(item)
                result[k] = converted_list
            else:
                result[k] = v
        return result

    def _add_to_history(self, p, r, tc=None, ad=None, pc=None):
        response_stripped = r.strip() if isinstance(r, str) else str(r)
        entry = {
            "step": self.step_count,
            "llm_response": response_stripped,
            "timestamp": time.time(),
        }

        if tc:
            entry["tool_calls"] = tc
            last_call = tc[-1]
            entry["action"] = last_call.get("name", "unknown")
            if ad:
                entry["action_details"] = ad
            elif last_call.get("name") == "navigate_to" and "x" in last_call.get("args", {}) and "y" in last_call.get(
                "args", {}
            ):
                variance = last_call.get("args", {}).get("variance", "none")
                entry["action_details"] = f"navigate_to({last_call['args']['x']}, {last_call['args']['y']}, variance={variance})"
            elif last_call.get("name") == "press_buttons" and "buttons" in last_call.get("args", {}):
                entry["action_details"] = f"press_buttons({last_call['args']['buttons']})"
            else:
                entry["action_details"] = f"{last_call.get('name', 'unknown')}(...)"
        elif ad:
            entry["action_details"] = ad

        if pc:
            entry["player_coords"] = pc

        self.conversation_history.append(entry)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _store_function_result_for_context(self, n, r):
        self.recent_function_results.append({"name": n, "result": r})
        if len(self.recent_function_results) > 3:
            self.recent_function_results.pop(0)

    def _get_function_results_context(self):
        return "\n".join([f"🔧 {r['name']}: {r['result'][:500]}" for r in self.recent_function_results])

    def _patch_gym_grid(self, gs_data):
        """Patch the grid and state_text to reflect current turnstile states."""
        try:
            loc = gs_data.get("location", "")
            if "GYM" not in loc.upper():
                return gs_data

            grid = gs_data.get("raw_state", {}).get("map", {}).get("porymap", {}).get("grid")
            if not grid:
                st = gs_data.get("state_text", "")
                if "ASCII Map:" in st:
                    # Extract the map part accurately
                    map_part = st.split("ASCII Map:")[1]
                    # Cut off at Warps or Objects
                    if "Warps (" in map_part:
                        map_part = map_part.split("Warps (")[0]
                    elif "Objects/" in map_part:
                        map_part = map_part.split("Objects/")[0]

                    # Split lines and strip each
                    map_lines = [line.strip() for line in map_part.strip().split("\n") if line.strip()]
                    grid = [list(line) for line in map_lines]
            if not grid:
                return gs_data

            # 2. Clear all arms (orthogonal neighbors of pivots)
            for px, py in self.turnstile_states.keys():
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = px + dx, py + dy
                    if 0 <= ny < len(grid) and 0 <= nx < len(grid[ny]):
                        if grid[ny][nx] == "#":
                            grid[ny][nx] = "."

            # 3. Add arms based on toggle states
            for (px, py), state in self.turnstile_states.items():
                arms = [(px - 1, py), (px + 1, py)] if state == "H" else [(px, py - 1), (px, py + 1)]
                for ax, ay in arms:
                    if 0 <= ay < len(grid) and 0 <= ax < len(grid[ay]):
                        grid[ay][ax] = "#"

            # 4. Re-format ASCII map in state_text
            st = gs_data.get("state_text", "")
            if "ASCII Map:" in st:
                parts = st.split("ASCII Map:")
                header = parts[0]
                rest = parts[1].split("Warps (")
                footer = "Warps (" + rest[1] if len(rest) > 1 else ""

                # Build new ASCII map from patched grid
                new_ascii_lines = []
                for y, row in enumerate(grid):
                    line_chars = list(row)
                    # Add player if at this Y
                    pp = gs_data.get("player_position", {})
                    if pp and pp.get("y") == y:
                        px = pp.get("x")
                        if 0 <= px < len(line_chars):
                            line_chars[px] = "P"
                    new_ascii_lines.append("".join(line_chars))

                new_ascii = "\n".join(new_ascii_lines)
                gs_data["state_text"] = f"{header}ASCII Map (PATCHED):\n{new_ascii}\n\n{footer}"
                logger.info("🗺️ Successfully patched gym ASCII map")

            return gs_data
        except Exception as e:
            logger.warning(f"Failed to patch gym grid: {e}")
            return gs_data

    def run_step(self, prompt, max_tool_calls=5, screenshot_b64=None):
        try:
            # Make current step available for per-step metrics logging
            os.environ["LLM_STEP_NUMBER"] = str(self.step_count)

            gs = self.mcp_adapter.call_tool("get_game_state", {})
            loc = gs.get("location", "Unknown")
            self.vlm.backend.tools = self.tools
            if hasattr(self.vlm.backend, "_setup_function_calling"):
                self.vlm.backend._setup_function_calling()
            last_coords = self.conversation_history[-1].get("player_coords") if self.conversation_history else None
            with self.frame_buffer_lock:
                frames = list(self.frame_buffer)
                if screenshot_b64:
                    frames.append(Image.open(io.BytesIO(base64.b64decode(screenshot_b64))))
            if not frames:
                return False, "No frames"
            if self._is_black_frame(frames[-1]):
                return True, "WAIT"
            res = self.vlm.get_query(frames, prompt, "CLI_Agent")
            thinking_text, parts = "", []
            if self.backend == "gemini":
                parts = res.candidates[0].content.parts if hasattr(res, "candidates") else []
                for p in parts:
                    if hasattr(p, "text") and p.text:
                        thinking_text += p.text + " "
                thinking_text = thinking_text.strip()
                for p in parts:
                    if hasattr(p, "function_call"):
                        fc = p.function_call
                        args = self._convert_protobuf_args(fc.args)
                        if fc.name == "navigate_to" and self.blocked_coords:
                            if not args.get("blocked_coords"):
                                args["blocked_coords"] = [list(c) for c in self.blocked_coords]
                                fr = self._execute_function_call_by_name(fc.name, args)
                            else:
                                fr = self._execute_function_call(fc)
                        else:
                            fr = self._execute_function_call(fc)
                        self._store_function_result_for_context(fc.name, fr)
                        if fc.name in ["press_buttons", "navigate_to"]:
                            self._wait_for_actions_complete()
                        tr = args.get("reasoning") or args.get("reason") or ""
                        cr = f"{thinking_text}\nTool Reasoning: {tr}" if thinking_text and tr else (thinking_text or tr)
                        try:
                            s = json.loads(self._execute_function_call_by_name("get_game_state", {}))
                            coords = (s.get("player_position", {}).get("x"), s.get("player_position", {}).get("y"))
                        except:
                            coords = None
                        tool_call = {"name": fc.name, "args": args, "result": fr}
                        if fc.name == "navigate_to" and "x" in args and "y" in args:
                            variance = args.get("variance", "none")
                            action_details = f"navigate_to({args['x']}, {args['y']}, variance={variance})"
                        elif fc.name == "press_buttons" and "buttons" in args:
                            action_details = f"press_buttons({args['buttons']})"
                        else:
                            action_details = f"{fc.name}(...)"
                        self.llm_logger.add_step_tool_calls(self.step_count, [tool_call])
                        self._add_to_history(prompt, cr, [tool_call], action_details, pc=coords)
                        if coords and last_coords and coords == last_coords and fc.name == "press_buttons":
                            try:
                                btn = args.get("buttons", [])[-1]
                                tx, ty = coords
                                if btn == "UP":
                                    ty -= 1
                                elif btn == "DOWN":
                                    ty += 1
                                elif btn == "LEFT":
                                    tx -= 1
                                elif btn == "RIGHT":
                                    tx += 1
                                for px, py in self.turnstile_states.keys():
                                    if abs(tx - px) + abs(ty - py) == 1:
                                        self.turnstile_states[(px, py)] = (
                                            "H" if self.turnstile_states[(px, py)] == "V" else "V"
                                        )
                                self.blocked_coords.add((tx, ty))
                            except:
                                pass
                        if isinstance(cr, str) and any(
                            p in cr.lower() for p in ["already fought", "already battled", "repeating", "defeated"]
                        ):
                            try:
                                gs_data = json.loads(self._execute_function_call_by_name("get_game_state", {}))
                                if coords:
                                    for obj in gs_data.get("raw_state", {}).get("objects", []):
                                        if obj.get("trainer_type") != "TRAINER_TYPE_NONE" and (
                                            abs(obj["x"] - coords[0]) + abs(obj["y"] - coords[1]) <= 1
                                        ):
                                            self.defeated_trainers.add(f"{obj['graphics_id']}@{obj['x']},{obj['y']}")
                            except:
                                pass
                        return True, cr or "Action executed"
            self._add_to_history(prompt, str(res), [])
            return True, str(res)
        except Exception as e:
            return False, str(e)

    def _build_structured_prompt(self, gs_res, sc):
        try:
            gd = json.loads(gs_res)
        except:
            gd = {}
        st, loc = gd.get("state_text", ""), gd.get("location", "Unknown")
        if self._is_title_sequence(gd):
            st = self._strip_map_info(st)
        elif "Gym" in loc:
            st = self._gym_strip(st)
        def _fmt_obj(obj, label):
            if not obj:
                return [f"{label}: None"]
            return [
                f"{label}:",
                f"  id: {obj.get('id')}",
                f"  description: {obj.get('description')}",
                f"  action_type: {obj.get('action_type')}",
                f"  target_location: {obj.get('target_location')}",
                f"  navigation_hint: {obj.get('navigation_hint')}",
                f"  completion_condition: {obj.get('completion_condition')}",
            ]

        do, ds = gd.get("direct_objective", ""), gd.get("direct_objective_status", "")
        co = gd.get("categorized_objectives", {})
        if co:
            story_obj = co.get("story")
            battling_group = co.get("battling_group", [])
            dynamics_obj = co.get("dynamics")

            parts = []
            parts.extend(_fmt_obj(story_obj, "STORY"))

            if battling_group:
                parts.append("BATTLING:")
                for i, obj in enumerate(battling_group, 1):
                    parts.extend(
                        [
                            f"  [{i}] id: {obj.get('id')}",
                            f"  [{i}] description: {obj.get('description')}",
                            f"  [{i}] action_type: {obj.get('action_type')}",
                            f"  [{i}] target_location: {obj.get('target_location')}",
                            f"  [{i}] navigation_hint: {obj.get('navigation_hint')}",
                            f"  [{i}] completion_condition: {obj.get('completion_condition')}",
                        ]
                    )
            else:
                parts.append("BATTLING: None")

            parts.extend(_fmt_obj(dynamics_obj, "DYNAMICS"))
            do = "\n".join(parts)

            cs = gd.get("categorized_status", {})
            if cs:
                ds = "📊 PROGRESS: " + " | ".join(
                    [
                        f"{c.capitalize()}: {i.get('current_index') + 1}/{i.get('total')}"
                        for c, i in cs.items()
                        if i.get("total") > 0
                    ]
                )
        if do and isinstance(do, dict):
            do = "\n".join(_fmt_obj(do, "OBJECTIVE"))

        # Enhanced history for LLM
        lines, last, trail = [], None, []
        for e in self.conversation_history[-20:]:
            c = e.get("player_coords")
            c_str = f"({c[0]},{c[1]})" if c else "(?)"
            if c:
                trail.append(c_str)
            s = (
                " [STAYED AT SAME POS]"
                if last and c == last
                else (
                    " [LOOP]"
                    if c and last and any(p.get("player_coords") == c for p in self.conversation_history[:-1])
                    else ""
                )
            )
            res = str(e.get("llm_response", ""))
            res_preview = res.split("\n")[0][:100]
            if s and "Executed press_buttons" in e.get("action_details", ""):
                s = " [STAYED AT SAME POS - LIKELY TOGGLED GATE]"
            tools_called = ""
            if e.get("tool_calls"):
                tools_called = f" [Tools: {', '.join(t.get('name', 'unknown') for t in e['tool_calls'])}]"
            action_details = e.get("action_details", "No action")
            lines.append(f"[{e['step']}] {c_str}{s}: {res_preview}... -> {action_details}{tools_called}")
            last = c
        hist = "\n".join(lines) + "\n### TRAIL: " + " -> ".join(trail[-10:])

        func = self._get_function_results_context()
        defeated = "\n### DEFEATED TRAINERS:\n" + "\n".join(self.defeated_trainers) if self.defeated_trainers else ""
        blocked = (
            "\n### BLOCKED TILES:\n" + ", ".join([f"({x},{y})" for x, y in self.blocked_coords])
            if self.blocked_coords
            else ""
        )
        prog = ""
        if self._game_type != "red":
            try:
                pp = gd.get("player_position", {})
                if pp:
                    dist = abs(pp["x"] - 15) + abs(pp["y"] - 2)
                    radar = self._get_local_radar(gd)
                    prog = f"\n### PROGRESS METRICS:\n- Distance to Winona (15,2): {dist} tiles\n- DONT BACKTRACK!\n{radar}"
            except:
                pass
        warning = ""
        if "Gym" in loc and self.conversation_history:
            if "STAYED AT SAME POS" in self.conversation_history[-1].get("llm_response", ""):
                warning = "\n⚠️ ALERT: Last move toggled a gate. DO NOT repeat it! Look for a NEW path."
        return f"# Step: {sc}\n{self._load_base_prompt()}\n## CONTEXT\n{warning}\n### HISTORY:\n{hist}\n{func}{defeated}{blocked}{prog}\n### OBJECTIVE:\n{do}\n{ds}\n### STATE:\n{st}\n"

    def _is_black_frame(self, img):
        try:
            a = np.array(img) if hasattr(img, "save") else img
            return a.mean() < 10 and a.std() < 5
        except:
            return False

    def _is_title_sequence(self, gd):
        if gd.get("location") == "TITLE_SEQUENCE":
            return True
        st = gd.get("state_text", "")
        if "Player Name:" in st:
            m = re.search(r"Player Name:\s*(\S+)", st)
            return True if m and (not m.group(1).strip() or m.group(1).strip() == "????????") else not m
        return False

    def _strip_map_info(self, st):
        lines, filtered, skip = st.split("\n"), [], False
        for l in lines:
            if any(
                m in l
                for m in [
                    "🗺️ MAP:",
                    "CURRENT MAP:",
                    "PORYMAP ASCII:",
                    "PORYMAP GROUND TRUTH MAP:",
                    "🧭 MOVEMENT PREVIEW:",
                    "POSITION:",
                    "LOCATION:",
                ]
            ):
                skip = True
            if l.strip() == "" or l.startswith(("🎯", "📊", "⚠️")):
                skip = False
            if not skip:
                filtered.append(l)
        return "\n".join(filtered)

    def _gym_strip(self, st):
        lines, filtered, skip = st.split("\n"), [], False
        for l in lines:
            if "MOVEMENT PREVIEW:" in l:
                skip = True
            if l.strip() == "" or "### TOOLS:" in l:
                skip = False
            if not skip:
                filtered.append(l)
        return "\n".join(filtered)

    def _get_local_radar(self, gd):
        try:
            player_pos = gd.get("player_position", {})
            if not (player_pos and "x" in player_pos and "y" in player_pos):
                return ""
            px, py = player_pos["x"], player_pos["y"]
            grid = gd.get("raw_state", {}).get("map", {}).get("porymap", {}).get("grid")
            if not grid:
                return ""
            radar_text = "\n### LOCAL COLLISION RADAR (5x5, P=You, #=Gate/Wall, .=Walkable):\n"
            for y in range(py - 2, py + 3):
                row = []
                for x in range(px - 2, px + 3):
                    if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
                        char = "P" if x == px and y == py else grid[y][x]
                        row.append(f"{char}")
                    else:
                        row.append("X")
                radar_text += " ".join(row) + "\n"
            return radar_text
        except:
            return ""

    def _wait_for_actions_complete(self):
        time.sleep(1.8)

    def run(self) -> int:
        self.conversation_history = []
        logger.info("🚀 Starting MyCLIAgent loop...")
        try:
            while True:
                if self.max_steps and self.step_count >= self.max_steps:
                    break
                logger.info(f"🤖 Step {self.step_count + 1}")
                gs_res = self._execute_function_call_by_name("get_game_state", {})
                try:
                    gs_json = json.loads(gs_res)
                    gs_patched = self._patch_gym_grid(gs_json)
                    gs_res = json.dumps(gs_patched)
                except:
                    pass
                try:
                    b64 = json.loads(gs_res).get("screenshot_base64")
                except:
                    b64 = None
                p = (
                    self._build_structured_prompt(gs_res, self.step_count)
                    if self.optimization_enabled
                    else self._build_structured_prompt(gs_res, self.step_count)
                )
                success, out = self.run_step(p, screenshot_b64=b64)
                if not success:
                    time.sleep(5)
                    continue
                self.step_count += 1
                try:
                    update_server_metrics(self.server_url)
                    requests.post(f"{self.server_url}/checkpoint", json={"step_count": self.step_count}, timeout=10)
                    requests.post(f"{self.server_url}/save_agent_history", timeout=5)
                except:
                    pass
                time.sleep(1)
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            return 1
        finally:
            self.stop_sampling.set()
            if self.sampling_thread:
                self.sampling_thread.join(timeout=2)
        return 0


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--server-url", default="http://localhost:8000")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--system-instructions", default="agent/prompts/POKEAGENT.md")
    p.add_argument("--backend", default="gemini")
    args = p.parse_args()
    agent = MyCLIAgent(
        server_url=args.server_url,
        model=args.model,
        backend=args.backend,
        max_steps=args.max_steps,
        system_instructions_file=args.system_instructions,
    )
    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
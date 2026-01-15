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
from utils.vlm import VLM

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
        system_instructions_file="POKEAGENT.md",
        max_context_chars=100000,
        target_context_chars=50000,
        enable_prompt_optimization=False,
        optimization_frequency=10,
    ):
        print(f"🚀 Initializing MyCLIAgent with backend={backend}, model={model}, server={server_url}")
        self.server_url, self.model, self.backend, self.max_steps = server_url, model, backend, max_steps
        self.step_count, self.max_context_chars, self.target_context_chars = 0, max_context_chars, target_context_chars
        self.optimization_enabled, self.optimization_frequency = enable_prompt_optimization, optimization_frequency
        self.conversation_history, self.frame_buffer, self.max_frame_buffer_size = [], [], 15
        self.frame_buffer_lock, self.sampling_interval, self.stop_sampling = threading.Lock(), 1.0, threading.Event()
        self.recent_function_results = []
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
        p = Path(__file__).parent.parent / f
        return p.read_text() if p.exists() else "AI agent playing Pokemon Emerald."

    def _load_base_prompt(self):
        p = Path(__file__).parent.parent / "base_prompt.md"
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
                "description": "Get state",
                "parameters": {"type_": "OBJECT", "properties": {}, "required": []},
            },
            {
                "name": "press_buttons",
                "description": "Press buttons",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "buttons": {"type_": "ARRAY", "items": {"type_": "STRING"}},
                        "reasoning": {"type_": "STRING"},
                    },
                    "required": ["buttons", "reasoning"],
                },
            },
            {
                "name": "navigate_to",
                "description": "Navigate",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "x": {"type_": "INTEGER"},
                        "y": {"type_": "INTEGER"},
                        "variance": {"type_": "STRING"},
                        "reason": {"type_": "STRING"},
                    },
                    "required": ["x", "y", "variance", "reason"],
                },
            },
            {
                "name": "complete_direct_objective",
                "description": "Complete objective",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {"reasoning": {"type_": "STRING"}},
                    "required": ["reasoning"],
                },
            },
        ]

    def _execute_function_call_by_name(self, n, a):
        return json.dumps(self.mcp_adapter.call_tool(n, a))

    def _execute_function_call(self, fc):
        return self._execute_function_call_by_name(fc.name, self._convert_protobuf_args(fc.args))

    def _convert_protobuf_args(self, pa):
        return {k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, (str, dict)) else v) for k, v in pa.items()}

    def _add_to_history(self, p, r, tc=None, ad=None, pc=None):
        self.conversation_history.append(
            {"step": self.step_count, "llm_response": r, "action_details": ad, "player_coords": pc}
        )
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)

    def _store_function_result_for_context(self, n, r):
        self.recent_function_results.append({"name": n, "result": r})
        if len(self.recent_function_results) > 3:
            self.recent_function_results.pop(0)

    def _get_function_results_context(self):
        return "\n".join([f"🔧 {r['name']}: {r['result'][:500]}" for r in self.recent_function_results])

    def run_step(self, prompt, max_tool_calls=5, screenshot_b64=None):
        try:
            gs = self.mcp_adapter.call_tool("get_game_state", {})
            loc = gs.get("location", "Unknown")
            is_gym = "Gym" in loc or "GYM" in loc
            logger.info(f"📍 Current location: {loc} (is_gym={is_gym})")

            at = [t for t in self.tools if t["name"] != "navigate_to"] if is_gym else self.tools
            if is_gym:
                logger.info(f"🚫 Disabled navigate_to tool for {loc}")

            self.vlm.backend.tools = at

            if hasattr(self.vlm.backend, "_setup_function_calling"):
                self.vlm.backend._setup_function_calling()
            with self.frame_buffer_lock:
                frames = list(self.frame_buffer)
                if screenshot_b64:
                    frames.append(Image.open(io.BytesIO(base64.b64decode(screenshot_b64))))
            if not frames:
                return False, "No frames"
            if self._is_black_frame(frames[-1]):
                return True, "WAIT"
            res = self.vlm.get_query(frames, prompt, "CLI_Agent")
            if self.backend == "gemini":
                parts = res.candidates[0].content.parts if hasattr(res, "candidates") else []
                for p in parts:
                    if hasattr(p, "function_call"):
                        fr = self._execute_function_call(p.function_call)
                        self._store_function_result_for_context(p.function_call.name, fr)
                        try:
                            s = json.loads(self._execute_function_call_by_name("get_game_state", {}))
                            coords = (s.get("player_position", {}).get("x"), s.get("player_position", {}).get("y"))
                        except:
                            coords = None
                        self._add_to_history(prompt, "", None, f"Executed {p.function_call.name}", pc=coords)
                        return True, "Action executed"
            self._add_to_history(prompt, str(res))
            return True, str(res)
        except Exception as e:
            return False, str(e)

    def _build_optimized_prompt(self, gs_res, sc):
        try:
            gd = json.loads(gs_res)
        except:
            gd = {}
        st = gd.get("state_text", "")
        if self._is_title_sequence(gd):
            st = self._strip_map_info(st)
        do, ds = gd.get("direct_objective", ""), gd.get("direct_objective_status", "")
        co = gd.get("categorized_objectives", {})
        if co:
            parts = []
            for c in ["story", "battling", "dynamics"]:
                o = co.get(c, {})
                if o:
                    parts.append(
                        f"{c.upper()}: {o.get('description')} (@ {o.get('target_location')})\n   Hint: {o.get('navigation_hint')}"
                    )
            if parts:
                do = "\n\n".join(parts)
            cs = gd.get("categorized_status", {})
            if cs:
                ds = "📊 PROGRESS: " + " | ".join(
                    [
                        f"{c.capitalize()}: {i.get('current_index') + 1}/{i.get('total')}"
                        for c, i in cs.items()
                        if i.get("total") > 0
                    ]
                )
        loc = gd.get("location", "Unknown")
        is_gym = "Gym" in loc or "GYM" in loc
        tools = "🎮 **TOOLS**:\n- get_game_state()\n- complete_direct_objective()\n- press_buttons()"
        if not is_gym:
            tools += "\n- navigate_to()"
        return f"# Step: {sc}\n{self._load_base_prompt()}\n## CONTEXT\n### HISTORY:\n{self._format_action_history()}\n{self._get_function_results_context()}\n### OBJECTIVE:\n{do}\n{ds}\n### STATE:\n{st}\n### TOOLS:\n{tools}\n"

    def _build_structured_prompt(self, gs_res, sc):
        return self._build_optimized_prompt(gs_res, sc)

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

    def _format_action_history(self):
        if not self.conversation_history:
            return "None."
        lines = []
        for e in self.conversation_history[-10:]:
            c = f"({e['player_coords'][0]},{e['player_coords'][1]})" if e.get("player_coords") else "(?)"
            res = e.get("llm_response", "")
            # Truncate long thinking to keep history clean
            if len(res) > 200:
                res = res[:200] + "..."
            lines.append(f"[{e['step']}] at {c}: {res} -> {e['action_details']}")
        return "\n".join(lines)

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
                    b64 = json.loads(gs_res).get("screenshot_base64")
                except:
                    b64 = None
                p = (
                    self._build_optimized_prompt(gs_res, self.step_count)
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
            logger.info("🛑 Stopped")
            return 0
        except Exception as e:
            logger.error(f"❌ Error: {e}")
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
    p.add_argument("--system-instructions", default="POKEAGENT.md")
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

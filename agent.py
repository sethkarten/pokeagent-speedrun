#!/usr/bin/env python3
"""
Direct agent implementation that runs agent and emulator in the same process
with visualization and real-time server interface like server.app
"""

import os
import pygame
import numpy as np
import time
import datetime
import base64
import io
import signal
import sys
import threading
from PIL import Image
import argparse
import logging
import cv2
import subprocess
import multiprocessing
import requests

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import json

from pokemon_env.emulator import EmeraldEmulator
from pokemon_env.enums import MetatileBehavior
from agent.perception import perception_step
from agent.planning import planning_step
from agent.memory import memory_step
from agent.action import action_step
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm
from utils.map_formatter import format_map_for_display
from utils.anticheat import AntiCheatTracker
from utils.llm_logger import get_llm_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
emulator = None
agent_modules = None
anticheat_tracker = None  # Anti-cheat tracker for submission logging
llm_logger = None  # LLM logger for tracking interactions
running = True
step_count = 0  # Display/frame counter
agent_step_count = 0  # Actual agent decision counter
current_obs = None
fps = 120
agent_thinking = False
last_agent_action = None
agent_mode = True   # True = agent (default), False = manual
agent_auto_enabled = False  # Auto agent actions
last_game_state = None  # Cache for web interface
recent_manual_actions = []  # Track recent manual button presses with timestamps

# Checkpoint system
CHECKPOINT_INTERVAL = 1  # Save checkpoint every N agent actions
last_checkpoint_step = 0  # Track when we last saved a checkpoint
agent_step_count = 0  # Track total agent steps

# Error recovery system
consecutive_errors = 0  # Track consecutive errors for recovery
MAX_CONSECUTIVE_ERRORS = 5  # Maximum errors before trying recovery
SERVER_RESTART_THRESHOLD = 10  # Restart server after this many consecutive errors

# LLM logging
llm_logger = None  # Will be initialized when needed

# Video recording state
video_writer = None
video_recording = False
video_filename = None
video_frame_skip = 4  # Record every 4th frame (120/4 = 30 FPS)
video_frame_counter = 0  # Counter for frame skipping

# Simple mode settings
simple_mode = False
no_ocr_mode = False
multiprocess_mode = False

# Agent processing queues for async processing
agent_processing_queue = []
agent_result_queue = []
agent_processing_thread = None

# Pygame display
screen_width = 480  # 240 * 2 (upscaled)
screen_height = 320  # 160 * 2 (upscaled)
screen = None
font = None
clock = None

# Threading locks
obs_lock = threading.Lock()
step_lock = threading.Lock()
agent_lock = threading.Lock()

# WebSocket connections
websocket_connections = set()

# Button mapping for manual control
button_map = {
    pygame.K_z: 'A',
    pygame.K_x: 'B', 
    pygame.K_RETURN: 'START',
    pygame.K_RSHIFT: 'SELECT',
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
}

# FastAPI app for web interface
app = FastAPI(
    title="Direct Agent Pokemon Emerald",
    description="Agent and emulator running in same process with real-time interface",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for streaming data (initialized for HTTP endpoints)
latest_frame = ""
latest_text = ""  
latest_plan = ""
latest_memory_status = "idle"

# Debug logging rate limiting
last_debug_log_time = {
    "frame": 0,
    "text": 0,
    "plan": 0,
    "memory": 0
}

# API Models
class AgentActionRequest(BaseModel):
    buttons: list = []
    manual: bool = False

class AgentStateResponse(BaseModel):
    visual: dict
    player: dict
    game: dict
    map: dict
    agent: dict
    step_number: int
    status: str

class AgentModules:
    """Container for agent modules using function-based approach"""
    def __init__(self, backend="openai", model_name="gpt-4o"):
        self.backend = backend
        self.model_name = model_name
        self.vlm = None  # Will be initialized lazily
        self._vlm_initializing = False
        self._vlm_init_error = None
        
        # Agent state
        self.memory_context = []
        self.observation_buffer = []
        self.current_plan = None
        self.recent_actions = []
        self.last_observation = None
        self.last_plan = None
        self.last_action = None
        self.thinking = False
        self.step_counter = 0  # Track frame IDs for observations
    
    def _ensure_vlm_initialized(self):
        """Initialize VLM if not already done (lazy initialization for heavy models)"""
        if self.vlm is not None:
            return True
        
        if self._vlm_init_error:
            raise Exception(f"VLM initialization failed previously: {self._vlm_init_error}")
        
        if self._vlm_initializing:
            # Wait for initialization to complete
            import time
            timeout = 60  # 60 seconds timeout
            start = time.time()
            while self._vlm_initializing and time.time() - start < timeout:
                time.sleep(0.1)
            
            if self.vlm is None:
                raise Exception("VLM initialization timed out")
            return True
        
        # Initialize VLM
        self._vlm_initializing = True
        try:
            print(f"Initializing {self.backend} VLM model {self.model_name}...")
            self.vlm = VLM(backend=self.backend, model_name=self.model_name)
            print(f"VLM initialized successfully")
            return True
        except Exception as e:
            self._vlm_init_error = str(e)
            raise
        finally:
            self._vlm_initializing = False
    
    def process_game_state(self, game_state):
        """Process game state through agent modules"""
        try:
            # Ensure VLM is initialized (will happen in background thread on first call)
            self._ensure_vlm_initialized()
            
            with agent_lock:
                self.thinking = True
                
                # Get screenshot from game state
                screenshot_obj = game_state["visual"]["screenshot"]
                frame = screenshot_obj if screenshot_obj is not None and hasattr(screenshot_obj, 'save') else None
                
                # Increment step counter
                self.step_counter += 1
                
                # Simple mode: skip all modules except action selection
                if simple_mode:
                    return self._simple_mode_processing(frame, game_state)
                
                # Full mode: use all four agent modules
                # 1. Perception - analyze current game state
                observation, slow_thinking_needed = perception_step(frame, game_state, self.vlm)
                self.last_observation = observation
                
                # Store observation with frame_id like in agent.py
                self.observation_buffer.append({
                    "frame_id": self.step_counter,
                    "observation": observation,
                    "state": game_state
                })
                
                # Keep observation buffer reasonable size
                if len(self.observation_buffer) > 10:
                    self.observation_buffer = self.observation_buffer[-10:]
                
                # 2. Memory - update with new observations
                self.memory_context = memory_step(
                    self.memory_context, 
                    self.current_plan, 
                    self.recent_actions,
                    self.observation_buffer,
                    self.vlm
                )
                
                # 3. Planning - create high-level plan
                plan_result = planning_step(
                    self.memory_context,
                    self.current_plan,
                    slow_thinking_needed,
                    game_state,
                    self.vlm
                )
                self.current_plan = plan_result
                self.last_plan = plan_result
                
                # 4. Action - select specific button input
                action_list = action_step(
                    self.memory_context,
                    self.current_plan,
                    observation,
                    frame,
                    game_state,
                    self.recent_actions,
                    self.vlm
                )
                
                # Convert action list to expected dictionary format
                if isinstance(action_list, list):
                    if action_list:
                        # Take first action and store rest for sequential execution
                        action = action_list[0]
                        remaining = action_list[1:] if len(action_list) > 1 else []
                        action_result = {
                            "action": action,
                            "reasoning": f"Selected action from sequence: {', '.join(action_list)}"
                        }
                        if remaining:
                            action_result["remaining_actions"] = remaining
                    else:
                        # Empty list, default to 'A'
                        action_result = {
                            "action": 'A',
                            "reasoning": "Default action (empty list)"
                        }
                else:
                    # If somehow not a list, handle gracefully
                    action_result = {
                        "action": str(action_list) if action_list else 'A',
                        "reasoning": "Direct action"
                    }
                
                # Store action result
                self.last_action = action_result
                self.recent_actions.append(action_result["action"])
                
                # Keep recent actions reasonable size
                if len(self.recent_actions) > 20:
                    self.recent_actions = self.recent_actions[-20:]
                
                self.thinking = False
                
                return action_result
                
        except Exception as e:
            logger.error(f"Error in agent processing: {e}")
            self.thinking = False
            return {"action": "A", "reasoning": f"Error: {e}"}  # Default safe action
    
#     def _simple_mode_processing(self, frame, game_state):
#         """Simple mode: direct frame + formatted state -> action"""
#         try:
#             from utils.state_formatter import format_state_for_llm
            
#             # Format the current state for LLM
#             formatted_state = format_state_for_llm(game_state)
            
#             # Create simple prompt with just frame and comprehensive state
#             prompt = f"""You are playing Pokemon Emerald. Based on the current game frame and state information, choose the best button action.

# CURRENT GAME STATE:
# {formatted_state}

# ACTION HISTORY (last 20 actions):
# {', '.join(self.recent_actions[-20:]) if self.recent_actions else 'None'}

# Available actions: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT

# Respond with just the button name (e.g., 'A' or 'RIGHT'). Be decisive and avoid getting stuck."""
            
#             # Make VLM call directly
#             if frame:
#                 response = self.vlm.get_query(frame, prompt, "simple_mode")
#             else:
#                 response = self.vlm.get_text_query(prompt, "simple_mode")
            
#             # Extract action from response
#             response_upper = response.upper().strip()
#             valid_actions = ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT']
            
#             # Find the action in the response
#             action = 'A'  # Default
#             for valid_action in valid_actions:
#                 if valid_action in response_upper:
#                     action = valid_action
#                     break
            
#             action_result = {
#                 "action": action,
#                 "reasoning": f"Simple mode: {response}"
#             }
            
#             # Store action result
#             self.last_action = action_result
#             self.recent_actions.append(action_result["action"])
            
#             # Keep recent actions reasonable size
#             if len(self.recent_actions) > 20:
#                 self.recent_actions = self.recent_actions[-20:]
            
#             self.thinking = False
#             return action_result
            
#         except Exception as e:
#             logger.error(f"Error in simple mode processing: {e}")
#             self.thinking = False
#             return {"action": "A", "reasoning": f"Simple mode error: {e}"}
    
    def get_agent_status(self):
        """Get current agent status for API"""
        return {
            "thinking": self.thinking,
            "last_observation": str(self.last_observation) if self.last_observation else "",
            "last_plan": str(self.last_plan) if self.last_plan else "",
            "last_action": str(self.last_action) if self.last_action else "",
            "reasoning": self.last_action.get("reasoning", "") if isinstance(self.last_action, dict) else "",
            "memory_size": len(self.memory_context),
            "backend": self.backend,
            "model": self.model_name
        }

async def broadcast_state_update():
    """Broadcast current game state to all connected WebSocket clients"""
    global last_game_state, agent_modules
    
    if not websocket_connections or not last_game_state:
        return
    
    try:
        # Get agent status if available
        agent_status = {}
        if agent_modules:
            agent_status = {
                "thinking": agent_modules.thinking,
                "last_observation": str(agent_modules.last_observation)[:500] if agent_modules.last_observation else "",
                "last_plan": str(agent_modules.last_plan)[:500] if agent_modules.last_plan else "",
                "last_action": agent_modules.last_action if agent_modules.last_action else {},
                "step_counter": agent_modules.step_counter
            }
        
        # Extract party data properly
        party_data = []
        if "player" in last_game_state:
            player_data = last_game_state["player"]
            if "party" in player_data and player_data["party"] is not None:
                party = player_data["party"]
                if isinstance(party, dict) and "pokemon" in party:
                    party_data = party["pokemon"]
                elif isinstance(party, list):
                    party_data = party
        
        message = json.dumps({
            "type": "state_update",
            "data": {
                "screenshot": last_game_state.get("visual", {}).get("screenshot_base64", ""),
                "player": last_game_state.get("player", {}),
                "game": last_game_state.get("game", {}),
                "party": party_data,
                "map": last_game_state.get("map", {}),
                "location": last_game_state.get("player", {}).get("location", "Unknown"),
                "agent_mode": agent_mode,
                "agent_auto": agent_auto_enabled,
                "agent": agent_status,
                "agent_thinking": agent_thinking,
                "last_action": last_agent_action,
                "step": step_count,
                "fps": fps
            }
        })
        
        # Send to all connected WebSocket clients
        disconnected = set()
        for websocket in websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            websocket_connections.discard(ws)
            
        # Update global streaming data for HTTP endpoints
        global latest_frame, latest_text, latest_plan, latest_memory_status
        
        screenshot = last_game_state.get("visual", {}).get("screenshot_base64", "")
        if screenshot:
            latest_frame = screenshot
        
        # Update agent status 
        if agent_status.get('last_observation'):
            latest_text = agent_status['last_observation']
        
        if agent_status.get('last_plan'):  
            latest_plan = agent_status['last_plan']
            
        latest_memory_status = "refreshed"
            
    except Exception as e:
        logger.error(f"Error broadcasting state: {e}")

def reset_error_counter():
    """Reset the consecutive error counter on successful operation"""
    global consecutive_errors
    consecutive_errors = 0

def handle_agent_error(error):
    """Handle agent processing errors with graceful recovery"""
    global consecutive_errors, running
    
    consecutive_errors += 1
    error_msg = str(error)
    
    if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
        if consecutive_errors <= MAX_CONSECUTIVE_ERRORS:
            wait_time = min(consecutive_errors * 2, 10)  # Exponential backoff, max 10s
            print(f"⚠️  Connection error {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}: {error}")
            if "read timed out" in error_msg.lower():
                print(f"   Server is not responding. Check if server/app.py is running on port 8000")
            print(f"   Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        elif consecutive_errors <= SERVER_RESTART_THRESHOLD:
            print(f"🔄 Too many consecutive errors ({consecutive_errors}), attempting recovery...")
            time.sleep(5)
        else:
            print(f"💥 CRITICAL: Too many consecutive errors ({consecutive_errors})")
            print("   Saving emergency checkpoint and continuing...")
            try:
                save_checkpoint()
            except:
                pass
            time.sleep(10)
    else:
        # Non-connection error, just log and continue
        print(f"Agent processing error: {error}")
        time.sleep(1)

def signal_handler(signum, _frame):
    """Handle shutdown signals gracefully"""
    global running, video_writer
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_filename}")
    if emulator:
        emulator.stop()
    pygame.quit()
    sys.exit(0)

def save_checkpoint(step_count=None):
    """Save checkpoint with current state, milestones, and LLM history"""
    global emulator, agent_modules, llm_logger, agent_step_count, last_checkpoint_step
    
    # Use provided step count or global one
    current_step = step_count if step_count is not None else agent_step_count
    # Update global step count if local one is provided
    if step_count is not None:
        agent_step_count = step_count
    
    try:
        checkpoint_base = "checkpoint"
        
        # 1. Save emulator state
        state_file = f"{checkpoint_base}.state"
        if emulator:
            emulator.save_state(state_file)
            print(f"💾 Checkpoint state saved: {state_file}")
            
            # 2. Save milestones (this is automatic with state saving)
            milestone_file = f"{checkpoint_base}_milestones.json"
            if emulator.milestone_tracker:
                emulator.milestone_tracker.save_milestones_for_state(state_file)
                print(f"💾 Checkpoint milestones saved: {milestone_file}")
        
        # 3. Save LLM interaction history
        llm_file = f"{checkpoint_base}_llm.txt"
        if llm_logger:
            llm_logger.save_checkpoint(llm_file)
            print(f"💾 Checkpoint LLM history saved: {llm_file}")
        
        # Update checkpoint tracking
        last_checkpoint_step = current_step
        print(f"✅ Checkpoint completed at step {current_step}")
        
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()

def load_checkpoint():
    """Load checkpoint with state, milestones, and LLM history"""
    global emulator, agent_modules, llm_logger, agent_step_count, last_checkpoint_step
    
    try:
        checkpoint_base = "checkpoint"
        
        # 1. Load emulator state
        state_file = f"{checkpoint_base}.state"
        if os.path.exists(state_file) and emulator:
            emulator.load_state(state_file)
            print(f"📂 Checkpoint state loaded: {state_file}")
        
        # 2. Load LLM interaction history
        llm_file = f"{checkpoint_base}_llm.txt"
        if os.path.exists(llm_file) and llm_logger:
            loaded_step_count = llm_logger.load_checkpoint(llm_file)
            if loaded_step_count:
                agent_step_count = loaded_step_count
                last_checkpoint_step = (agent_step_count // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
                print(f"📂 Checkpoint LLM history loaded: {llm_file} (step {agent_step_count})")
        
        print(f"✅ Checkpoint loaded at step {agent_step_count}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

def save_agent_state(simple_agent, filename):
    """Save simple agent state to JSON"""
    try:
        import json
        from collections import deque
        from datetime import datetime
        
        # Convert SimpleAgent state to serializable format
        state_data = {
            "step_counter": simple_agent.state.step_counter,
            "history": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "player_coords": entry.player_coords,
                    "map_id": entry.map_id,
                    "context": entry.context,
                    "action_taken": entry.action_taken,
                    "game_state_summary": entry.game_state_summary
                }
                for entry in simple_agent.state.history
            ],
            "recent_actions": list(simple_agent.state.recent_actions),
            "stuck_detection": dict(simple_agent.state.stuck_detection),
            "objectives": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "objective_type": obj.objective_type,
                    "target_value": obj.target_value,
                    "completed": obj.completed,
                    "created_at": obj.created_at.isoformat(),
                    "completed_at": obj.completed_at.isoformat() if obj.completed_at else None,
                    "progress_notes": obj.progress_notes,
                    "storyline": obj.storyline,
                    "milestone_id": obj.milestone_id
                }
                for obj in simple_agent.state.objectives
            ],
            "objectives_updated": simple_agent.state.objectives_updated,
            "history_display_count": simple_agent.history_display_count,
            "actions_display_count": simple_agent.actions_display_count
        }
        
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
            
    except Exception as e:
        print(f"❌ Failed to save SimpleAgent state: {e}")
        import traceback
        traceback.print_exc()

def load_agent_state(simple_agent, filename):
    """Load simple agent state from JSON"""
    try:
        import json
        from collections import deque
        from datetime import datetime
        from agent.simple import HistoryEntry, Objective
        
        with open(filename, 'r') as f:
            state_data = json.load(f)
        
        # Restore basic counters
        simple_agent.state.step_counter = state_data.get("step_counter", 0)
        simple_agent.state.stuck_detection = state_data.get("stuck_detection", {})
        simple_agent.state.objectives_updated = state_data.get("objectives_updated", False)
        
        # Restore display counts
        simple_agent.history_display_count = state_data.get("history_display_count", 15)
        simple_agent.actions_display_count = state_data.get("actions_display_count", 20)
        
        # Restore recent actions
        recent_actions = state_data.get("recent_actions", [])
        simple_agent.state.recent_actions = deque(recent_actions, maxlen=simple_agent.state.recent_actions.maxlen)
        
        # Restore history entries
        history_data = state_data.get("history", [])
        restored_history = []
        for entry_data in history_data:
            entry = HistoryEntry(
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                player_coords=tuple(entry_data["player_coords"]) if entry_data["player_coords"] else None,
                map_id=entry_data["map_id"],
                context=entry_data["context"],
                action_taken=entry_data["action_taken"],
                game_state_summary=entry_data["game_state_summary"]
            )
            restored_history.append(entry)
        simple_agent.state.history = deque(restored_history, maxlen=simple_agent.state.history.maxlen)
        
        # Restore objectives
        objectives_data = state_data.get("objectives", [])
        restored_objectives = []
        for obj_data in objectives_data:
            obj = Objective(
                id=obj_data["id"],
                description=obj_data["description"],
                objective_type=obj_data["objective_type"],
                target_value=obj_data["target_value"],
                completed=obj_data["completed"],
                created_at=datetime.fromisoformat(obj_data["created_at"]),
                completed_at=datetime.fromisoformat(obj_data["completed_at"]) if obj_data["completed_at"] else None,
                progress_notes=obj_data["progress_notes"],
                storyline=obj_data["storyline"],
                milestone_id=obj_data["milestone_id"]
            )
            restored_objectives.append(obj)
        simple_agent.state.objectives = restored_objectives
        
        print(f"✅ SimpleAgent state loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load SimpleAgent state: {e}")
        import traceback
        traceback.print_exc()

def setup_emulator(rom_path="Emerald-GBAdvance/rom.gba", load_state=None):
    """Initialize the emulator"""
    global emulator, current_obs, no_ocr_mode
    
    try:
        if not os.path.exists(rom_path):
            raise RuntimeError(f"ROM not found at {rom_path}")
        
        # Suppress debug logging
        logging.getLogger('pokemon_env.memory_reader').setLevel(logging.WARNING)
        
        emulator = EmeraldEmulator(rom_path=rom_path, headless=False, sound=False)
        emulator.initialize()
        
        # Disable OCR and all dialog detection if no_ocr_mode is enabled
        if no_ocr_mode and emulator.memory_reader:
            emulator.memory_reader._ocr_enabled = False
            emulator.memory_reader._ocr_detector = None
            emulator.memory_reader._dialog_detection_enabled = False
            print("🚫 All dialogue detection disabled (--no-ocr flag)")
        
        if load_state and os.path.exists(load_state):
            emulator.load_state(load_state)
            print(f"✅ Loaded state from: {load_state}")
            
            # Verify state loaded correctly
            state = emulator.get_comprehensive_state()
            player_info = state.get("player", {})
            print(f"📍 Player: {player_info.get('name', 'Unknown')} at ({player_info.get('map_x', '?')}, {player_info.get('map_y', '?')})")
            print(f"🗺️  Map: {state.get('player', {}).get('location', 'Unknown')}")
        
        screenshot = emulator.get_screenshot()
        if screenshot is not None and hasattr(screenshot, 'save'):
            with obs_lock:
                current_obs = np.array(screenshot)
        else:
            with obs_lock:
                current_obs = np.zeros((emulator.height, emulator.width, 3), dtype=np.uint8)

        print("✅ Emulator initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize emulator: {e}")
        return False

def display_comprehensive_state():
    """Display the comprehensive state exactly as the LLM sees it"""
    global emulator
    
    if not emulator:
        print("❌ Emulator not initialized")
        return
    
    try:
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE STATE (What the LLM Sees)")
        print("="*80)
        
        # Get the comprehensive state with screenshot for OCR
        screenshot = emulator.get_screenshot()
        state = emulator.get_comprehensive_state(screenshot)
        
        # Format the state using the same formatter the agent uses
        from utils.state_formatter import format_state_for_llm
        formatted_state = format_state_for_llm(state)
        
        print(formatted_state)
        
        # Show additional debug info
        print("\n" + "-"*80)
        print("🔍 DEBUG INFO")
        print("-"*80)
        
        # Show dialogue detection details
        game_data = state.get('game', {})
        dialog_text = game_data.get('dialog_text', '')
        dialogue_detected = game_data.get('dialogue_detected', {})
        
        print(f"Dialog Text (raw): '{dialog_text}'")
        print(f"Dialogue Detection: {dialogue_detected}")
        
        # If we have OCR enabled, show OCR vs Memory comparison
        if emulator.memory_reader and emulator.memory_reader._ocr_enabled and screenshot:
            print("\n📖 OCR vs Memory Comparison:")
            memory_only = emulator.memory_reader.read_dialog()
            
            # Get OCR reading
            ocr_only = ""
            if emulator.memory_reader._ocr_detector:
                ocr_only = emulator.memory_reader._ocr_detector.detect_dialogue_from_screenshot(screenshot)
            
            combined = emulator.memory_reader.read_dialog_with_ocr_fallback(screenshot)
            
            print(f"  Memory: '{memory_only}'")
            print(f"  OCR:    '{ocr_only}'")
            print(f"  Combined: '{combined}'")
            
            # Show which case we're in using the same logic as the OCR fallback
            memory_clean = memory_only.strip() if memory_only else ""
            ocr_clean = ocr_only.strip() if ocr_only else ""
            
            # Apply the same meaningfulness detection as the OCR fallback
            ocr_is_meaningful = False
            if ocr_clean and emulator.memory_reader:
                ocr_is_meaningful = emulator.memory_reader._is_ocr_meaningful_dialogue(ocr_clean)
            
            # Apply the same residual text filtering as the OCR fallback  
            memory_filtered = memory_clean
            if memory_clean:
                cleaned_text = memory_clean.lower()
                residual_indicators = [
                    "got away safely", "fled from", "escaped", "ran away",
                    "fainted", "defeated", "victory", "experience points", 
                    "gained", "grew to", "learned"
                ]
                if any(indicator in cleaned_text for indicator in residual_indicators):
                    memory_filtered = ""  # Filter out residual text
            
            # Use the same case logic as the OCR fallback
            if memory_filtered and ocr_clean and ocr_is_meaningful:
                print("  ✅ Case 1: Both meaningful -> Using memory")
            elif not memory_filtered and ocr_clean and ocr_is_meaningful:
                print("  🔄 Case 2: OCR only -> Using OCR")
            elif memory_filtered and (not ocr_clean or not ocr_is_meaningful):
                if not ocr_clean:
                    print("  🚨 Case 3: Memory only -> SUPPRESSED (OCR found nothing)")
                else:
                    print(f"  🚨 Case 3: Memory only -> SUPPRESSED (OCR meaningless: '{ocr_clean}')")
            else:
                print("  ❌ Case 4: Neither detected")
        
        # Show in-battle status
        print(f"\nIn Battle: {game_data.get('is_in_battle', False)}")
        print(f"Game State: {game_data.get('game_state', 'unknown')}")
        
        # Show player location
        player_data = state.get('player', {})
        position = player_data.get('position', {})
        location = player_data.get('location', {})
        print(f"\nPlayer Position: X={position.get('x') if isinstance(position, dict) else position}, Y={position.get('y') if isinstance(position, dict) else 'N/A'}")
        
        # Handle location being either dict or string
        if isinstance(location, dict):
            print(f"Location: Bank={location.get('map_bank')}, Map={location.get('map_number')}")
        else:
            print(f"Location: {location}")
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ Failed to display comprehensive state: {e}")
        import traceback
        traceback.print_exc()

def setup_agent(backend="openai", model_name="gpt-4o"):
    """Initialize agent modules"""
    global agent_modules, anticheat_tracker, llm_logger
    
    try:
        agent_modules = AgentModules(backend=backend, model_name=model_name)
        print(f"Agent initialized with {backend} backend using {model_name}")
        
        # Initialize anti-cheat tracker for submission logging
        anticheat_tracker = AntiCheatTracker()
        anticheat_tracker.initialize_submission_log(model_name)
        print(f"Submission logging initialized to submission.log")
        
        # Initialize LLM logger
        llm_logger = get_llm_logger()
        print(f"LLM interaction logging initialized")
        
        return True
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return False

def handle_input(manual_mode=True):
    """Handle keyboard input"""
    global running
    actions_pressed = []
    
    if not manual_mode:
        return True, []
    
    keys = pygame.key.get_pressed()
    for key, button in button_map.items():
        if keys[key]:
            actions_pressed.append(button)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, []
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False, []
            elif event.key == pygame.K_s:
                save_screenshot()
            elif event.key == pygame.K_m:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # Shift+M for map
                    display_map()
                else:
                    # Regular 'M' for comprehensive state (what LLM sees)
                    display_comprehensive_state()
            elif event.key == pygame.K_SPACE:  # Spacebar to trigger agent action
                return True, ["AGENT_STEP"]
            elif event.key == pygame.K_TAB:  # Tab to toggle agent/manual mode
                global agent_mode
                agent_mode = not agent_mode
                mode_text = "AGENT" if agent_mode else "MANUAL"
                control_text = "(LLM controls game)" if agent_mode else "(Keyboard controls game)"
                print(f"🔄 Switched to {mode_text} mode {control_text}")
            elif event.key == pygame.K_a:  # 'A' key to toggle auto agent
                global agent_auto_enabled
                agent_auto_enabled = not agent_auto_enabled
                auto_text = "ENABLED" if agent_auto_enabled else "DISABLED"
                trigger_text = "(acts automatically)" if agent_auto_enabled else "(press Space to act)"
                print(f"🤖 Auto agent {auto_text} {trigger_text}")
                
                # Show current combined state for clarity
                if agent_mode and agent_auto_enabled:
                    print("  ✅ Agent will act automatically every few seconds")
                elif agent_mode and not agent_auto_enabled:
                    print("  🎯 Agent will act when you press Spacebar")
                elif not agent_mode:
                    print("  ⌨️  Keyboard control active (auto setting ignored)")
            elif event.key == pygame.K_1:
                if emulator:
                    save_file = "agent_direct_save.state"
                    emulator.save_state(save_file)
                    print(f"State saved to: {save_file}")
            elif event.key == pygame.K_2:
                if emulator:
                    load_file = "agent_direct_save.state"
                    if os.path.exists(load_file):
                        emulator.load_state(load_file)
                        print(f"State loaded from: {load_file}")
    
    return True, actions_pressed

# Global action queue for 120 FPS background loop
current_actions = []
action_lock = threading.Lock()
pending_agent_action = False

def queue_action(actions):
    """Queue actions to be processed by background loop"""
    global current_actions
    with action_lock:
        current_actions = actions.copy() if actions else []

def agent_processing_worker():
    """Dedicated thread for agent processing to avoid blocking emulator"""
    global agent_processing_queue, agent_result_queue, agent_modules, running
    
    print("🧠 Starting agent processing thread...")
    
    while running:
        # Check for agent processing requests
        if agent_processing_queue and agent_modules:
            # Get the latest game state to process
            game_state = agent_processing_queue.pop(0)
            
            try:
                print("🤖 Agent processing started (async)...")
                start_time = time.time()
                agent_action = agent_modules.process_game_state(game_state)
                decision_time = time.time() - start_time
                
                # Add decision time and original game state to the action result
                if agent_action:
                    agent_action["decision_time"] = decision_time
                    agent_action["game_state"] = game_state  # Include the game state with hash
                    
                    # Log to submission.log for anti-cheat tracking
                    if anticheat_tracker and game_state:
                        try:
                            current_agent_step = game_state.get("agent_step", agent_step_count)
                            state_hash = game_state.get("state_hash", "unknown")
                            action_taken = agent_action.get("action", "UNKNOWN")
                            
                            # Log to anti-cheat submission file
                            anticheat_tracker.log_submission_data(
                                step=current_agent_step,
                                state_data=game_state,
                                action_taken=action_taken,
                                decision_time=decision_time,
                                state_hash=state_hash
                            )
                            
                        except Exception as e:
                            print(f"❌ Submission logging error: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Put result in result queue
                agent_result_queue.append(agent_action)
                print(f"✅ Agent processing complete: {agent_action.get('action', 'NO_ACTION')}")
                print(f"📝 Full reasoning: {agent_action.get('reasoning', 'No reason')}")
                
            except Exception as e:
                print(f"❌ Agent processing error: {e}")
                import traceback
                traceback.print_exc()
                agent_result_queue.append({"action": "A", "reasoning": f"Error: {e}"})
        else:
            time.sleep(0.1)  # Check every 100ms
    
    print("🧠 Agent processing thread stopped")

def queue_agent_step():
    """Queue an agent step to be processed asynchronously"""
    global agent_processing_queue, emulator, anticheat_tracker, llm_logger, agent_step_count
    
    if emulator and agent_modules:
        # Increment agent step counter for this decision
        agent_step_count += 1
        
        # Log step start
        if llm_logger:
            llm_logger.log_step_start(agent_step_count, "agent_step")
        
        # Get current screenshot and use it for comprehensive state to ensure OCR uses latest frame
        screenshot = emulator.get_screenshot()
        game_state = emulator.get_comprehensive_state(screenshot)
        
        # Create state hash for integrity verification
        if anticheat_tracker:
            state_hash = anticheat_tracker.create_state_hash(game_state)
            game_state["state_hash"] = state_hash
            game_state["step_start_time"] = time.time()
            game_state["agent_step"] = agent_step_count  # Store the agent step number
            
            # Log state snapshot
            if llm_logger:
                llm_logger.log_state_snapshot(game_state, agent_step_count)
        
        agent_processing_queue.append(game_state)
        print(f"📝 Agent step {agent_step_count} queued for async processing")
    else:
        print("❌ Cannot queue agent step - emulator or agent not ready")

def init_video_recording(record_enabled=False):
    """Initialize video recording if enabled"""
    global video_writer, video_recording, video_filename, fps, video_frame_skip
    
    if not record_enabled:
        return
    
    try:
        # Create video filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"pokegent_recording_{timestamp}.mp4"
        
        # Video settings (GBA resolution is 240x160)
        # Record at 30 FPS (skip every 4th frame from 120 FPS emulator)
        recording_fps = fps / video_frame_skip  # 120 / 4 = 30 FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, float(recording_fps), (240, 160))
        
        if video_writer.isOpened():
            video_recording = True
            print(f"📹 Video recording started: {video_filename} at {recording_fps:.0f} FPS (recording every {video_frame_skip} frames)")
        else:
            print("❌ Failed to initialize video recording")
            video_writer = None
            
    except Exception as e:
        print(f"❌ Video recording initialization error: {e}")
        video_writer = None

def record_frame(screenshot):
    """Record frame to video if recording is enabled with frame skipping"""
    global video_writer, video_recording, video_frame_counter, video_frame_skip
    
    if not video_recording or video_writer is None or screenshot is None:
        return
    
    # Increment frame counter
    video_frame_counter += 1
    
    # Only record every Nth frame based on frame skip
    if video_frame_counter % video_frame_skip != 0:
        return
        
    try:
        # Convert PIL image to OpenCV format
        if hasattr(screenshot, 'save'):  # PIL image
            # Convert PIL to numpy array
            frame_array = np.array(screenshot)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        elif isinstance(screenshot, np.ndarray):  # Already numpy array
            # Convert RGB to BGR for OpenCV if needed
            if screenshot.shape[2] == 3:  # RGB
                frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = screenshot
            video_writer.write(frame_bgr)
            
    except Exception as e:
        logger.debug(f"Video recording frame error: {e}")

def background_emulator_loop():
    """Background 120 FPS emulator loop"""
    global current_obs, last_agent_action, pending_agent_action, running, current_actions, last_game_state, fps
    
    print(f"🎮 Starting {fps} FPS background emulator loop...")
    last_broadcast_time = 0
    broadcast_interval = 1/10  # Broadcast at 10 FPS to reduce overhead
    
    # Button press timing state
    current_button = None  # Currently pressed button
    button_hold_frames = 0  # Frames to hold button
    button_release_frames = 0  # Frames to wait after release
    BUTTON_HOLD_DURATION = 6  # Hold button for 6 frames (~50ms at 120 FPS)
    BUTTON_RELEASE_DELAY = 12  # Wait 12 frames (~100ms at 120 FPS) between presses
    
    # Wait for emulator to be properly initialized
    while running:
        if not emulator:
            time.sleep(0.01)
            continue
        
        # Ensure we have an initial screenshot before starting the main loop
        if current_obs is None:
            screenshot = emulator.get_screenshot()
            if screenshot:
                with obs_lock:
                    current_obs = np.array(screenshot)
                print("📸 Initial screenshot captured for background loop")
            else:
                time.sleep(0.01)
                continue
        
        break
    
    # Main emulator loop
    while running:
        if not emulator:
            time.sleep(0.01)
            continue
        
        actions_to_execute = []
        agent_step_needed = False
        
        # Get queued manual actions
        manual_actions = []
        with action_lock:
            manual_actions = current_actions.copy() if current_actions else []
            agent_step_needed = pending_agent_action
            current_actions = []
            pending_agent_action = False
        
        # Handle button press timing state machine
        if manual_actions:
            # Manual actions bypass timing (for immediate response)
            actions_to_execute = manual_actions
            # Track manual actions for web interface
            global recent_manual_actions
            current_time = time.time()
            for action in manual_actions:
                recent_manual_actions.append({
                    "action": action,
                    "timestamp": current_time,
                    "type": "manual"
                })
            # Keep only last 20 actions
            if len(recent_manual_actions) > 20:
                recent_manual_actions = recent_manual_actions[-20:]
            # Clear any pending button states
            current_button = None
            button_hold_frames = 0
            button_release_frames = 0
        elif current_button:
            # We're currently holding a button
            if button_hold_frames > 0:
                # Continue holding the button
                actions_to_execute = [current_button]
                button_hold_frames -= 1
            else:
                # Release the button and start delay
                actions_to_execute = []  # No buttons pressed
                current_button = None
                button_release_frames = BUTTON_RELEASE_DELAY
        elif button_release_frames > 0:
            # We're in the delay period after releasing a button
            actions_to_execute = []  # No buttons pressed
            button_release_frames -= 1
        else:
            # Ready for a new button press - check for agent results
            if agent_result_queue:
                agent_action = agent_result_queue.pop(0)
                if agent_action and "action" in agent_action:
                    button_action = agent_action["action"]
                    print(f"🎮 Agent button press: {button_action}")
                    last_agent_action = agent_action
                    # Track agent actions for web interface
                    recent_manual_actions.append({
                        "action": button_action,
                        "timestamp": time.time(),
                        "type": "agent"
                    })
                    if len(recent_manual_actions) > 20:
                        recent_manual_actions = recent_manual_actions[-20:]
                    # Start holding the new button
                    current_button = button_action
                    button_hold_frames = BUTTON_HOLD_DURATION
                    actions_to_execute = [button_action]
                    
                    # Submission logging now happens in agent processing, not here
                    
                    # Store remaining actions if this was from a sequence
                    if "remaining_actions" in agent_action:
                        # Queue remaining actions for next frames - preserve game_state for logging
                        for action in agent_action["remaining_actions"]:
                            remaining_action = {
                                "action": action, 
                                "reasoning": "Continued sequence",
                                "decision_time": agent_action.get("decision_time", 0.0),
                                "game_state": agent_action.get("game_state")  # Preserve the game state
                            }
                            agent_result_queue.append(remaining_action)
                else:
                    print(f"❌ Invalid agent result: {agent_action}")
                    actions_to_execute = []
            else:
                # No new actions, just run empty frame
                actions_to_execute = []
        
        # Handle manual agent step requests
        if agent_step_needed:
            print("🤖 Manual agent step requested")
            queue_agent_step()
        
        # In agent mode, automatically let agent act (when --agent-auto flag is used)
        if (agent_mode and not agent_step_needed and agent_modules and agent_auto_enabled):
            # Agent auto mode: automatically trigger agent action if no manual actions
            current_time = time.time()
            agent_not_processing = len(agent_processing_queue) == 0  # Check if agent is not already processing
            
            if (not actions_to_execute and agent_not_processing and
                (not hasattr(emulator, '_last_agent_action_time') or 
                current_time - emulator._last_agent_action_time >= 2.0)):  # Agent acts every 2 seconds
                
                # Trigger async agent processing to avoid blocking emulator loop
                emulator._last_agent_action_time = current_time
                print("🤖 Auto agent triggered - starting async processing...")
                queue_agent_step()
        
        # Run frame with actions (or no actions)
        emulator.run_frame_with_buttons(actions_to_execute)
        
        # Update screenshot (very lightweight - just get the PIL image)
        screenshot = emulator.get_screenshot()
        if screenshot is not None and hasattr(screenshot, 'save'):
            # Store PIL image directly, convert to numpy only when display needs it
            with obs_lock:
                current_obs = screenshot  # Store PIL image directly
            
            # Record frame for video if enabled
            record_frame(screenshot)
        else:
            # Debug: Log when screenshot is None
            if hasattr(emulator, '_screenshot_fail_count'):
                emulator._screenshot_fail_count += 1
                if emulator._screenshot_fail_count % 60 == 0:  # Log every second
                    print(f"⚠️  Screenshot is None (fail count: {emulator._screenshot_fail_count})")
            else:
                emulator._screenshot_fail_count = 1
                print("⚠️  First screenshot failure")
        
        # Cache game state and broadcast updates periodically (expensive operation)
        current_time = time.time()
        if current_time - last_broadcast_time > broadcast_interval:
            try:
                # Only get comprehensive state when we actually need to broadcast (expensive)
                if websocket_connections and len(websocket_connections) > 0:
                    # Use the current screenshot we already have to avoid redundant get_screenshot() calls
                    try:
                        game_state = emulator.get_comprehensive_state(screenshot)
                        last_game_state = game_state
                    except ValueError as e:
                        if "ambiguous" in str(e):
                            print(f"AMBIGUOUS ERROR: Screenshot type: {type(screenshot)}")
                            if hasattr(screenshot, 'shape'):
                                print(f"Screenshot has shape: {screenshot.shape}")
                            if hasattr(screenshot, 'size'):
                                print(f"Screenshot has size: {screenshot.size}")
                            print(f"Screenshot value: {screenshot}")
                            import traceback
                            traceback.print_exc()
                        raise
                    except Exception as e:
                        logger.error(f"Error getting comprehensive state: {e}")
                        import traceback
                        import sys
                        traceback.print_exc()
                        sys.stderr.flush()
                
                # Only broadcast if we have WebSocket clients (skip expensive operations if not needed)
                if websocket_connections and len(websocket_connections) > 0:
                    # Move expensive encoding to background thread to avoid blocking main loop
                    def background_broadcast():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(broadcast_state_update())
                            loop.close()
                        except Exception as e:
                            logger.debug(f"Background broadcast error: {e}")
                    
                    # Run in background thread to not block main loop
                    threading.Thread(target=background_broadcast, daemon=True).start()
                
                last_broadcast_time = current_time
            except Exception as e:
                logger.debug(f"State update error: {e}")
        
        # Run at configured FPS
        time.sleep(1.0 / fps)

def step_emulator(actions_pressed):
    """Queue actions for the background emulator loop"""
    if "AGENT_STEP" in actions_pressed:
        queue_agent_step()
    else:
        queue_action(actions_pressed)

def update_display():
    """Update the pygame display"""
    global current_obs, screen, step_count, font
    
    if not screen:
        return
    
    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None
    
    if obs_copy is not None:
        # Convert PIL image to numpy only when needed for display
        if hasattr(obs_copy, 'size'):  # PIL image
            obs_array = np.array(obs_copy)
            obs_surface = pygame.surfarray.make_surface(obs_array.swapaxes(0, 1))
        else:  # Already numpy array (fallback)
            obs_surface = pygame.surfarray.make_surface(obs_copy.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(obs_surface, (screen_width, screen_height))
        screen.blit(scaled_surface, (0, 0))
    else:
        # Fill with black if no screenshot available and add debug text
        screen.fill((0, 0, 0))
        if font and step_count % 30 == 0:  # Debug message every second
            debug_text = font.render("No screenshot available", True, (255, 255, 255))
            screen.blit(debug_text, (10, 10))
        
        # Draw info overlay
        if font:
            mode_text = "AGENT" if agent_mode else "MANUAL"
            
            # Create clearer status display
            if agent_mode and agent_auto_enabled:
                status_text = "AGENT (AUTO)"
                control_info = "Agent acts automatically"
            elif agent_mode and not agent_auto_enabled:
                status_text = "AGENT (MANUAL)"
                control_info = "Press Space for agent action"
            else:
                status_text = "MANUAL"
                control_info = "Keyboard controls active"
            
            info_lines = [
                f"Step: {step_count} | Mode: {status_text}",
                f"Status: {control_info}",
                f"Controls: WASD/Arrows=Move, Z=A, X=B, Space=Agent Step",
                f"Special: Tab=Agent/Manual, A=Auto On/Off, S=Screenshot, M=Map, Esc=Quit"
            ]
            
            if agent_modules:
                agent_status = agent_modules.get_agent_status()
                if agent_status["thinking"]:
                    info_lines.append("🤖 Agent: THINKING...")
                elif agent_status["last_action"]:
                    info_lines.append(f"🤖 Last: {agent_status['last_action']} - {agent_status['reasoning']}")
            
            y_offset = 10
            for line in info_lines:
                text_surface = font.render(line, True, (255, 255, 255))
                # Add background for readability
                text_rect = text_surface.get_rect()
                bg_rect = pygame.Rect(10, y_offset-2, text_rect.width+4, text_rect.height+4)
                pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect)
                screen.blit(text_surface, (10, y_offset))
                y_offset += 25
    
    pygame.display.flip()

def save_screenshot():
    """Save current screenshot"""
    global current_obs
    
    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None
    
    if obs_copy is not None:
        timestamp = int(time.time())
        filename = f"agent_direct_screenshot_{timestamp}.png"
        
        # Handle both PIL images and numpy arrays
        if hasattr(obs_copy, 'save'):  # PIL image
            obs_copy.save(filename)
        else:  # numpy array
            img = Image.fromarray(obs_copy)
            img.save(filename)
        print(f"Screenshot saved: {filename}")

def display_map():
    """Display current map in terminal - showing both raw and agent views"""
    global emulator
    
    if not emulator:
        print("❌ Emulator not initialized")
        return
    
    try:
        # Suppress debug logs temporarily
        original_level = logging.getLogger('pokemon_env.memory_reader').level
        logging.getLogger('pokemon_env.memory_reader').setLevel(logging.WARNING)
        
        # Clear any cached state to ensure fresh data
        # if hasattr(emulator, '_cached_state'):
        #     delattr(emulator, '_cached_state')
        # if hasattr(emulator, '_cached_state_time'):
        #     delattr(emulator, '_cached_state_time')
        
        # Get raw map data FIRST (this should be clean)
        raw_map_data = emulator.memory_reader.read_map_around_player(radius=7)
        
        # Get comprehensive state using latest screenshot - this is what the agent receives
        screenshot = emulator.get_screenshot()
        state = emulator.get_comprehensive_state(screenshot)
        
        # Get the formatted state that the agent receives
        agent_view = format_state_for_llm(state)
        
        player_data = state.get("player", {})
        
        # Restore logging level
        logging.getLogger('pokemon_env.memory_reader').setLevel(original_level)
        
        print("\n" + "="*70)
        print("🎮 RAW MAP DATA (Direct from memory)")
        print("="*70)
        
        # Use unified formatter for raw map display
        if raw_map_data:
            facing = player_data.get('facing', 'South')
            npcs = state.get('map', {}).get('object_events', [])
            player_coords = state.get('map', {}).get('player_coords')
            formatted_map = format_map_for_display(raw_map_data, facing, "15x15 Map", npcs, player_coords)
            print(formatted_map)
        else:
            print("No raw map data available")
        
        print("\n" + "="*70)
        print("🤖 AGENT'S FORMATTED VIEW (What LLM sees)")
        print("="*70)
        
        # Show what the agent sees (which might be corrupted)
        if "=== LOCATION & MAP INFO ===" in agent_view:
            map_section_start = agent_view.index("=== LOCATION & MAP INFO ===")
            map_section = agent_view[map_section_start:]
            
            # Find the end of the map section (next === or end of string)
            next_section = map_section.find("\n===", 10)
            if next_section > 0:
                map_section = map_section[:next_section]
            
            # Print the exact map view the agent sees
            print(map_section)
        else:
            print("No map data in agent view")
        
        
        # Also show debug info
        print("\n" + "-"*70)
        print("📊 DEBUG: Additional State Info")
        print("-"*70)
        print(f"Player Name: {player_data.get('name', 'Unknown')}")
        print(f"Position: {player_data.get('position', 'Unknown')}")
        print(f"Facing: {player_data.get('facing', 'Unknown')}")
        print(f"Location: {player_data.get('location', 'Unknown')}")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error displaying map: {e}")

def init_pygame():
    """Initialize pygame"""
    global screen, font, clock
    
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Direct Agent Pokemon Emerald")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()

def game_loop(manual_mode=True, agent_auto=False):
    """Main game loop - handles input and display while background loop runs emulator at 120 FPS"""
    global running, step_count, fps
    
    mode_text = "AGENT" if agent_mode else "MANUAL"
    print(f"Starting Direct Agent game loop in {mode_text} mode...")
    print("Controls: WASD/Arrows=Move, Z=A, X=B, Space=Agent Step")
    print("Special: Tab=Mode Toggle, A=Auto Toggle, S=Screenshot, M=LLM State, Shift+M=Map, 1=Save, 2=Load, Esc=Quit")
    
    if agent_auto:
        print("Agent auto mode: Agent will act automatically every few seconds")
    
    # Start background emulator loop at specified FPS
    emulator_thread = threading.Thread(target=background_emulator_loop, daemon=True)
    emulator_thread.start()
    
    last_agent_time = time.time()
    agent_interval = 3.0  # Agent acts every 3 seconds in auto mode
    display_fps = 120  # Display updates at 120 FPS to match emulator performance
    
    while running:
        # Handle input
        should_continue, actions_pressed = handle_input(manual_mode)
        if not should_continue:
            break
        
        # Auto agent mode is now handled by background_emulator_loop
        # This avoids conflicts between two different auto-agent mechanisms
        
        # Queue actions for background emulator loop
        if actions_pressed:
            step_emulator(actions_pressed)
        
        # Update display (independent of emulator speed)
        update_display()
        
        # Update step count (this is now just for display purposes)
        with step_lock:
            step_count += 1
        
        # Display loop runs at 120 FPS to match emulator performance
        if clock:  # Only tick if pygame was initialized
            clock.tick(display_fps)

def run_fastapi_server(port):
    """Run FastAPI server in background thread"""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

# FastAPI endpoints
@app.get("/status")
async def get_status():
    """Get server status"""
    with step_lock:
        current_display_step = step_count
    
    agent_status = agent_modules.get_agent_status() if agent_modules else {"thinking": False}
    
    return {
        "status": "running",
        "step_count": agent_step_count,  # Use agent decision count for main step display
        "display_step": current_display_step,  # Keep display frames for reference
        "fps": fps,
        "agent_initialized": agent_modules is not None,
        "agent_thinking": agent_status["thinking"]
    }

@app.get("/state")
async def get_comprehensive_state():
    """Get comprehensive game state"""
    if emulator is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Use current_obs (latest frame from 120 FPS loop) for OCR performance
        screenshot_for_state = None
        with obs_lock:
            if current_obs is not None:
                # current_obs is now a PIL image directly
                screenshot_for_state = current_obs.copy() if hasattr(current_obs, 'copy') else current_obs
        
        # Get game state using the latest screenshot
        state = emulator.get_comprehensive_state(screenshot_for_state)
        
        # If screenshot is missing in state, use current_obs as fallback
        if not state.get("visual", {}).get("screenshot"):
            with obs_lock:
                if current_obs is not None:
                    try:
                        # current_obs is now a PIL image, handle both cases for backward compatibility
                        if hasattr(current_obs, 'save'):  # PIL image
                            img = current_obs
                        else:  # numpy array (fallback)
                            img = Image.fromarray(current_obs.astype('uint8'), 'RGB')
                        if "visual" not in state:
                            state["visual"] = {}
                        state["visual"]["screenshot"] = img
                        logger.debug("Used current_obs as fallback screenshot")
                    except Exception as fallback_error:
                        logger.debug(f"Fallback screenshot error: {fallback_error}")
        
        # Add agent information
        agent_status = agent_modules.get_agent_status() if agent_modules else {
            "thinking": False,
            "last_observation": "",
            "last_plan": "",
            "last_action": "",
            "reasoning": "",
            "memory_size": 0
        }
        
        state["agent"] = agent_status
        
        # Convert screenshot to base64 if available
        screenshot_obj = state.get("visual", {}).get("screenshot")
        if screenshot_obj is not None and hasattr(screenshot_obj, 'save'):
            try:
                buffer = io.BytesIO()
                state["visual"]["screenshot"].save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                state["visual"]["screenshot_base64"] = img_str
                del state["visual"]["screenshot"]
            except Exception as img_error:
                logger.debug(f"Screenshot conversion error: {img_error}")
                state["visual"]["screenshot_base64"] = ""
        else:
            # Ensure visual dict exists with empty screenshot
            if "visual" not in state:
                state["visual"] = {}
            state["visual"]["screenshot_base64"] = ""
        
        # Update global streaming variables for HTTP endpoints
        global latest_frame, latest_text, latest_plan, latest_memory_status
        
        # Update frame data
        screenshot = state.get("visual", {}).get("screenshot_base64", "")
        if screenshot:
            latest_frame = screenshot
        
        # Update agent status data
        if agent_status.get('last_observation'):
            latest_text = agent_status['last_observation']
        
        if agent_status.get('last_plan'):  
            latest_plan = agent_status['last_plan']
            
        latest_memory_status = "refreshed"
        
        with step_lock:
            current_step = step_count
        
        return AgentStateResponse(
            visual=state["visual"],
            player=state["player"],
            game=state["game"],
            map=state["map"],
            agent=state["agent"],
            step_number=current_step,
            status="running"
        )
        
    except Exception as e:
        logger.error(f"Error getting comprehensive state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/action")
async def take_action(request: AgentActionRequest):
    """Take an action (manual or agent)"""
    if emulator is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        if request.manual:
            # Manual action
            step_emulator(request.buttons)
        else:
            # Agent action
            if agent_modules:
                step_emulator(["AGENT_STEP"])
            else:
                raise HTTPException(status_code=400, detail="Agent not initialized")
        
        with step_lock:
            step_count += 1
        
        # Return updated state
        state = await get_comprehensive_state()
        return state
        
    except Exception as e:
        logger.error(f"Error taking action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent")
async def get_agent_status():
    """Get agent status with thinking data for stream.html compatibility"""
    # Get the agent thinking data
    thinking_response = await get_agent_thinking()
    
    # Base response
    response = {
        "status": "initialized" if agent_modules else "manual_mode",
        "current_step": thinking_response.get("current_step", 0),
        "recent_interactions": thinking_response.get("recent_interactions", []),
        "current_thought": thinking_response.get("current_thought", ""),
        "confidence": thinking_response.get("confidence", 0.0)
    }
    
    # Add agent module status if available
    if agent_modules is not None:
        agent_status = agent_modules.get_agent_status()
        response.update(agent_status)
    else:
        response.update({
            "message": "Manual mode active - Agent not running",
            "last_action": "",
            "reasoning": ""
        })
    
    return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time state updates"""
    await websocket.accept()
    websocket_connections.add(websocket)
    print(f"WebSocket client connected. Total: {len(websocket_connections)}")
    
    try:
        while True:
            # Keep connection alive and listen for client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.discard(websocket)
        print(f"WebSocket client disconnected. Remaining: {len(websocket_connections)}")

@app.post("/toggle_mode")
async def toggle_mode():
    """Toggle between manual and agent mode"""
    global agent_mode
    agent_mode = not agent_mode
    mode_text = "AGENT" if agent_mode else "MANUAL"
    print(f"🔄 Mode switched to {mode_text} via API")
    return {"mode": mode_text, "agent_mode": agent_mode}

@app.post("/toggle_auto")
async def toggle_auto():
    """Toggle auto agent mode"""
    global agent_auto_enabled
    agent_auto_enabled = not agent_auto_enabled
    auto_text = "ENABLED" if agent_auto_enabled else "DISABLED"
    print(f"🤖 Auto agent {auto_text} via API")
    return {"auto_enabled": agent_auto_enabled, "status": auto_text}

# Streaming endpoints for stream.html compatibility
@app.get("/api/frame")
async def get_latest_frame():
    """Get latest game frame"""
    global latest_frame, last_debug_log_time
    frame_len = len(latest_frame) if latest_frame else 0
    
    # Rate limited debug logging - only log once every 5 seconds
    current_time = time.time()
    if current_time - last_debug_log_time["frame"] > 5:
        logger.debug(f"Frame endpoint: {frame_len} chars")
        last_debug_log_time["frame"] = current_time
    
    return {"frame": latest_frame}

@app.get("/api/text")  
async def get_latest_text():
    """Get latest agent observation"""
    global latest_text, last_debug_log_time
    text_len = len(latest_text) if latest_text else 0
    
    # Rate limited debug logging
    current_time = time.time()
    if current_time - last_debug_log_time["text"] > 5:
        logger.debug(f"Text endpoint: {text_len} chars")
        last_debug_log_time["text"] = current_time
    
    return {"text": latest_text}

@app.get("/api/plan")
async def get_latest_plan():
    """Get latest agent plan"""
    global latest_plan, last_debug_log_time
    plan_len = len(latest_plan) if latest_plan else 0
    
    # Rate limited debug logging
    current_time = time.time()
    if current_time - last_debug_log_time["plan"] > 5:
        logger.debug(f"Plan endpoint: {plan_len} chars")
        last_debug_log_time["plan"] = current_time
    
    return {"plan": latest_plan}

@app.get("/api/memory")
async def get_memory_status():
    """Get memory update status"""
    global latest_memory_status, last_debug_log_time
    
    # Rate limited debug logging
    current_time = time.time()
    if current_time - last_debug_log_time["memory"] > 5:
        logger.debug(f"Memory endpoint: {latest_memory_status}")
        last_debug_log_time["memory"] = current_time
    
    return {"status": latest_memory_status}

@app.get("/milestones")
async def get_milestones():
    """Get milestone tracking data for stream.html"""
    if emulator is None:
        return {
            "milestones": [],
            "completed": 0,
            "progress": 0,
            "tracking_system": "agent_direct",
            "milestone_file": None
        }
    
    try:
        # Use the emulator's built-in milestone tracker
        return emulator.get_milestones()
        
    except Exception as e:
        logger.error(f"Error getting milestones: {e}")
        return {
            "milestones": [],
            "completed": 0,
            "progress": 0,
            "tracking_system": "agent_direct",
            "error": str(e)
        }

@app.get("/recent_actions")
async def get_recent_actions():
    """Get recent button presses for stream.html action queue"""
    try:
        recent_buttons = []
        
        # Primary: Use the new unified action tracking
        global recent_manual_actions
        if recent_manual_actions:
            for action_data in recent_manual_actions[-20:]:  # Last 20 actions
                if isinstance(action_data, dict):
                    recent_buttons.append({
                        "button": action_data["action"],
                        "timestamp": action_data["timestamp"]
                    })
                else:
                    # Fallback for old string format
                    button_name = action_data.split(": ", 1)[-1] if ": " in action_data else action_data
                    recent_buttons.append({
                        "button": button_name,
                        "timestamp": time.time()
                    })
        
        # Fallback: get from agent modules if available
        if not recent_buttons and agent_modules is not None and hasattr(agent_modules, 'recent_actions'):
            current_time = time.time()
            recent = agent_modules.recent_actions[-20:] if agent_modules.recent_actions else []
            for i, action in enumerate(recent):
                button_name = action.get('action', '') if isinstance(action, dict) else str(action)
                if button_name:
                    recent_buttons.append({
                        "button": button_name,
                        "timestamp": current_time - (len(recent) - i)
                    })
        
        # Fallback: get from global last_agent_action if available
        if not recent_buttons and last_agent_action:
            button_name = ""
            if isinstance(last_agent_action, dict):
                button_name = last_agent_action.get("action", "")
            else:
                button_name = str(last_agent_action)
                
            if button_name:
                recent_buttons.append({
                    "button": button_name,
                    "timestamp": time.time()
                })
        
        # Return format expected by stream.html
        return {
            "recent_buttons": recent_buttons,
            "count": len(recent_buttons)
        }
    except Exception as e:
        logger.error(f"Error getting recent actions: {e}")
        return {"recent_buttons": [], "count": 0, "error": str(e)}

@app.get("/agent_thinking")
async def get_agent_thinking():
    """Get recent LLM interactions for stream.html"""
    try:
        # Check if llm_logs directory exists
        if not os.path.exists("llm_logs"):
            return {
                "status": "inactive",
                "current_thought": "Manual mode - No agent thinking. Use 'M' to toggle to agent mode.",
                "confidence": 0.0,
                "interactions_found": 0,
                "step": 0
            }
        
        # Find LLM log files
        import glob
        log_files = glob.glob("llm_logs/llm_log_*.jsonl")
        
        # Get recent interactions from the most recent log files (limit to last 3 files for performance)
        log_files.sort(reverse=True)  # Most recent first
        recent_interactions = []
        
        for log_file in log_files[:3]:  # Only check last 3 log files
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Get interactions from this file (last 10 lines for performance)
                        for line in lines[-20:]:  # Get more lines for more interactions
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("type") == "interaction":
                                    recent_interactions.append({
                                        "type": entry.get("interaction_type", "unknown"),
                                        "prompt": entry.get("prompt", ""),  # Full prompt, no truncation
                                        "response": entry.get("response", ""),  # Full response, no truncation
                                        "duration": entry.get("duration", 0),
                                        "timestamp": entry.get("timestamp", "")
                                    })
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.error(f"Error reading LLM log {log_file}: {e}")
        
        # Sort by timestamp and keep more recent interactions (show last 5)
        recent_interactions.sort(key=lambda x: x.get("timestamp", ""))
        recent_interactions = recent_interactions[-5:]  # Keep last 5 interactions
        
        # Provide current_thought for fallback display
        if recent_interactions:
            current_thought = "Agent is thinking... (see interactions below)"
        else:
            # Provide more detailed status based on what we found
            log_count = len(log_files)
            if log_count == 0:
                current_thought = f"No LLM log files found. Agent hasn't made any VLM calls yet."
            else:
                current_thought = f"Found {log_count} log files but no recent interactions. Agent may be in manual mode."
            
            # Add mode-specific guidance
            mode_status = "Manual" if not agent_mode else "Agent"
            current_thought += f"\nCurrent mode: {mode_status}. Press 'M' to toggle modes."
        
        with step_lock:
            current_step = step_count
        
        return {
            "status": "active" if recent_interactions else "manual",
            "current_thought": current_thought,
            "confidence": 0.95 if recent_interactions else 0.0,
            "timestamp": time.time(),
            "recent_interactions": recent_interactions,
            "current_step": current_step
        }
        
    except Exception as e:
        logger.error(f"Error in agent thinking: {e}")
        return {
            "status": "error",
            "current_thought": f"Error getting agent thinking: {str(e)}",
            "confidence": 0.0,
            "timestamp": time.time()
        }

async def serve_stream_html():
    """Common function to serve stream.html with HTTP polling"""
    try:
        with open("server/stream.html", "r") as f:
            html_content = f.read()
        # Replace Socket.IO with HTTP polling
        socket_io_code = """        // Initialize Socket.IO connection
        const socket = io('http://127.0.0.1:8000');
        
        // WebSocket event handlers
        socket.on('frame_update', function(data) {
            document.getElementById('frame').src = `data:image/png;base64,${data.frame}`;
        });
        
        socket.on('text_update', function(data) {
            // Text updates can be logged to console if needed
            console.log('Text update:', data.text);
        });
        
        socket.on('plan_update', function(data) {
            // Plan updates can be logged to console if needed
            console.log('Plan update:', data.plan);
        });
        
        socket.on('memory_update', function(data) {
            // Memory updates can be logged to console if needed
            console.log('Memory update: Memory refreshed');
        });
        
        // Connection status indicators
        socket.on('connect', function() {
            console.log('WebSocket connected');
            document.querySelector('.header div[style*="background-color"]').style.backgroundColor = '#33ff33';
        });
        
        socket.on('disconnect', function() {
            console.log('WebSocket disconnected');
            document.querySelector('.header div[style*="background-color"]').style.backgroundColor = '#ff6177';
        });"""
        
        http_polling_code = """        // HTTP polling for real-time updates
        let connected = true;
        
        async function pollFrame() {
            try {
                const response = await fetch('/api/frame');
                const data = await response.json();
                if (data.frame) {
                    document.getElementById('frame').src = `data:image/png;base64,${data.frame}`;
                }
                if (!connected) {
                    connected = true;
                    document.querySelector('.header div[style*="background-color"]').style.backgroundColor = '#33ff33';
                    console.log('Connection restored');
                }
            } catch (error) {
                if (connected) {
                    connected = false;
                    document.querySelector('.header div[style*="background-color"]').style.backgroundColor = '#ff6177';
                    console.log('Connection lost');
                }
            }
        }
        
        async function pollUpdates() {
            try {
                // Poll text updates
                const textResponse = await fetch('/api/text');
                const textData = await textResponse.json();
                if (textData.text) {
                    console.log('Text update:', textData.text);
                }
                
                // Poll plan updates  
                const planResponse = await fetch('/api/plan');
                const planData = await planResponse.json();
                if (planData.plan) {
                    console.log('Plan update:', planData.plan);
                }
                
                // Poll memory updates
                const memoryResponse = await fetch('/api/memory');
                const memoryData = await memoryResponse.json();
                if (memoryData.status === 'refreshed') {
                    console.log('Memory update: Memory refreshed');
                }
            } catch (error) {
                console.log('Update polling error:', error);
            }
        }
        
        // Start polling
        setInterval(pollFrame, 33); // 30 FPS
        setInterval(pollUpdates, 100); // 10 Hz for text updates
        
        // Initial connection status
        document.querySelector('.header div[style*="background-color"]').style.backgroundColor = '#33ff33';"""
        
        html_content = html_content.replace(socket_io_code, http_polling_code)
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Web interface not found</h1><p>Please ensure server/stream.html exists</p>")

@app.get("/")
async def get_web_interface_root():
    """Serve the web interface at root path"""
    return await serve_stream_html()

@app.get("/server/stream.html")
async def get_web_interface_server_path():
    """Serve the web interface at /server/stream.html path for compatibility"""
    return await serve_stream_html()

def run_multiprocess_server(args):
    """Run the server component in multiprocess mode"""
    try:
        # Import server dependencies
        from server.app import app
        import uvicorn
        
        # Configure the server app with the provided arguments
        os.environ["ROM_PATH"] = args.rom
        if args.load_checkpoint:
            # Check if checkpoint files exist
            checkpoint_state = "checkpoint.state"
            if os.path.exists(checkpoint_state):
                os.environ["LOAD_STATE"] = checkpoint_state
                print(f"🔄 Server will load from checkpoint: {checkpoint_state}")
            else:
                print(f"⚠️ Checkpoint file not found: {checkpoint_state}")
        elif args.load_state:
            os.environ["LOAD_STATE"] = args.load_state
        os.environ["VLM_BACKEND"] = args.backend
        os.environ["VLM_MODEL"] = args.model_name
        if args.no_ocr:
            os.environ["NO_OCR"] = "1"
        if args.record:
            os.environ["RECORD_VIDEO"] = "1"
        if args.simple:
            os.environ["SIMPLE_MODE"] = "1"
        
        print(f"📝 Server environment configured:")
        print(f"   ROM: {args.rom}")
        print(f"   Backend: {args.backend} / {args.model_name}")
        if args.load_checkpoint:
            print(f"   Load checkpoint: checkpoint.state")
        elif args.load_state:
            print(f"   Load state: {args.load_state}")
        if args.simple:
            print(f"   Simple mode enabled")
        if args.no_ocr:
            print(f"   No-OCR mode enabled")
        if args.record:
            print(f"   Video recording enabled")
        
        print(f"🌐 Starting server process on port {args.port}")
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
        
    except ImportError:
        print("❌ Server components not available for multiprocess mode")
        print("Please ensure server/app.py exists and dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Server process error: {e}")
        return False

def run_multiprocess_client_headless(server_port=8000, args=None):
    """Run headless client that just processes agent logic and sends commands to server"""
    global agent_step_count, last_checkpoint_step, llm_logger, consecutive_errors, anticheat_tracker
    
    server_url = f"http://127.0.0.1:{server_port}"
    
    print(f"🤖 Starting headless agent client, connecting to server at {server_url}")
    
    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{server_url}/status", timeout=2)
            if response.status_code == 200:
                print("✅ Connected to server")
                break
        except:
            print(f"⏳ Waiting for server... ({i+1}/{max_retries})")
            time.sleep(1)
    else:
        print("❌ Could not connect to server")
        return False
    
    # Note: Server checkpoint loading is handled during server startup
    # Client will only load LLM history below
    
    # Initialize agent if auto mode is enabled
    vlm = None
    agent_step_count = 0
    last_agent_time = 0
    
    if args and args.agent_auto:
        # Initialize VLM for agent processing
        try:
            vlm = VLM(backend=args.backend, model_name=args.model_name)
            print(f"✅ Agent initialized with {args.backend}/{args.model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            return False
            
        # Initialize LLM logger for multiprocess mode
        from utils.llm_logger import get_llm_logger
        llm_logger = get_llm_logger()
        print(f"✅ LLM interaction logging initialized")
        
        # Initialize anti-cheat tracker for submission logging
        from utils.anticheat import AntiCheatTracker
        anticheat_tracker = AntiCheatTracker()
        anticheat_tracker.initialize_submission_log(args.model_name if args else "unknown")
        print(f"✅ Anti-cheat submission logging initialized to submission.log")
        
        # Load LLM history if checkpoint was requested
        if args and getattr(args, 'load_checkpoint', False):
            llm_checkpoint_file = "checkpoint_llm.txt"
            if os.path.exists(llm_checkpoint_file):
                try:
                    loaded_step_count = llm_logger.load_checkpoint(llm_checkpoint_file)
                    if loaded_step_count:
                        agent_step_count = loaded_step_count
                        last_checkpoint_step = (agent_step_count // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
                        print(f"📂 Client: Checkpoint LLM history loaded: {llm_checkpoint_file} (step {agent_step_count})")
                    else:
                        print(f"📂 Client: Checkpoint LLM history loaded: {llm_checkpoint_file}")
                except Exception as e:
                    print(f"⚠️ Client: Failed to load checkpoint LLM history: {e}")
            else:
                print(f"⚠️ Client: No checkpoint LLM history found: {llm_checkpoint_file}")
    
    # Initialize agent modules if not in simple mode
    agent_modules = None
    if vlm and not (args and getattr(args, 'simple', False)):
        try:
            # Initialize agent modules like single process mode
            agent_modules = {
                'perception': True,  # We'll use the functions directly
                'planning': True,
                'memory': True,
                'action': True
            }
            print("✅ Agent modules initialized")
        except Exception as e:
            print(f"❌ Failed to initialize agent modules: {e}")
    
    running = True
    print("🤖 Agent running... Press Ctrl+C to stop")
    
    try:
        while running:
            # Agent processing (if auto mode enabled)
            current_time = time.time()
            if vlm and args.agent_auto:
                try:
                    # Get comprehensive state from server
                    state_response = requests.get(f"{server_url}/state", timeout=10)
                    if state_response.status_code == 200:
                        state_data = state_response.json()
                        
                        # Check if there are still actions in the queue before sending new ones
                        action_queue_length = state_data.get("action_queue_length", 0)
                        if action_queue_length > 0:
                            print(f"⏳ Waiting for action queue to empty (queue length: {action_queue_length})")
                            time.sleep(0.5)  # Wait longer to let actions complete
                            continue
                        
                        # Only call agent if enough time has passed AND queue is empty
                        if (current_time - last_agent_time < 2.0):  # Increased to 2 seconds minimum between agent calls
                            time.sleep(0.1)
                            continue
                        
                        # Convert server state format to the format expected by agent modules
                        from PIL import Image
                        import base64, io
                        
                        # Get screenshot from state data
                        screenshot_base64 = state_data.get("visual", {}).get("screenshot_base64", "")
                        if screenshot_base64:
                            img_data = base64.b64decode(screenshot_base64)
                            screenshot = Image.open(io.BytesIO(img_data))
                        else:
                            screenshot = None
                        
                        # Create game state object like single process mode
                        game_state = {
                            "visual": state_data.get("visual", {}),
                            "player": state_data.get("player", {}),
                            "game": state_data.get("game", {}),
                            "map": state_data.get("map", {})
                        }
                        game_state["visual"]["screenshot"] = screenshot
                        
                        # Run agent processing
                        if args and getattr(args, 'simple', False):
                            # Simple mode processing
                            action = simple_mode_processing_multiprocess(vlm, game_state, args)
                        else:
                            # Full agent module processing - call the actual functions
                            action = process_agent_step_multiprocess(vlm, game_state, agent_modules, args)
                        
                        # Send action to server instead of emulator
                        if action and action != "WAIT" and action != ["WAIT"]:
                            # Always send action as a list to the server
                            if isinstance(action, list):
                                # Filter out WAIT actions from the list
                                valid_actions = [a for a in action if a != "WAIT"]
                                if valid_actions:
                                    # Send each action individually for proper step counting
                                    for individual_action in valid_actions:
                                        requests.post(f"{server_url}/action", 
                                                    json={"buttons": [individual_action]}, 
                                                    timeout=5)
                                        agent_step_count += 1
                                        print(f"🤖 Agent action: {individual_action} (step {agent_step_count})")
                                        
                                        # Log each individual action to anti-cheat submission.log
                                        if anticheat_tracker:
                                            try:
                                                # Create state hash for integrity
                                                state_hash = anticheat_tracker.create_state_hash(state_data)
                                                
                                                # Log the action to submission.log
                                                anticheat_tracker.log_submission_data(
                                                    step=agent_step_count,
                                                    state_data=state_data,
                                                    action_taken=individual_action,
                                                    decision_time=0.1,  # Minimal time for multiprocess
                                                    state_hash=state_hash
                                                )
                                            except Exception as e:
                                                print(f"⚠️ Anti-cheat logging error: {e}")
                                        
                                        # Checkpoint saving after each individual action
                                        if agent_step_count - last_checkpoint_step >= CHECKPOINT_INTERVAL:
                                            try:
                                                # First, save state and milestones via server
                                                checkpoint_response = requests.post(f"{server_url}/checkpoint", 
                                                                                  json={"step_count": agent_step_count}, 
                                                                                  timeout=10)
                                                if checkpoint_response.status_code == 200:
                                                    print(f"💾 Server: Checkpoint state saved: checkpoint.state")
                                                    
                                                # Then save LLM history locally on client
                                                if llm_logger:
                                                    llm_logger.save_checkpoint("checkpoint_llm.txt")
                                                    print(f"💾 Client: Checkpoint LLM history saved: checkpoint_llm.txt")
                                                
                                                # SimpleAgent state is preserved via LLM logger history
                                                
                                                print(f"✅ Checkpoint completed at step {agent_step_count}")
                                                last_checkpoint_step = agent_step_count
                                            except Exception as e:
                                                print(f"⚠️ Checkpoint save error: {e}")
                                else:
                                    print(f"🤖 Agent waiting - all actions were WAIT (step {agent_step_count})")
                            else:
                                # Single action string - convert to list
                                requests.post(f"{server_url}/action", 
                                            json={"buttons": [action]}, 
                                            timeout=5)
                                agent_step_count += 1
                                print(f"🤖 Agent action: {action} (step {agent_step_count})")
                                
                                # Log to anti-cheat submission.log
                                if anticheat_tracker:
                                    try:
                                        # Create state hash for integrity
                                        state_hash = anticheat_tracker.create_state_hash(state_data)
                                        
                                        # Log the action to submission.log
                                        anticheat_tracker.log_submission_data(
                                            step=agent_step_count,
                                            state_data=state_data,
                                            action_taken=action,
                                            decision_time=0.1,  # Minimal time for multiprocess
                                            state_hash=state_hash
                                        )
                                    except Exception as e:
                                        print(f"⚠️ Anti-cheat logging error: {e}")
                                
                                # Checkpoint saving after single action
                                if agent_step_count - last_checkpoint_step >= CHECKPOINT_INTERVAL:
                                    try:
                                        # First, save state and milestones via server
                                        checkpoint_response = requests.post(f"{server_url}/checkpoint", 
                                                                          json={"step_count": agent_step_count}, 
                                                                          timeout=10)
                                        if checkpoint_response.status_code == 200:
                                            print(f"💾 Server: Checkpoint state saved: checkpoint.state")
                                            
                                        # Then save LLM history locally on client
                                        if llm_logger:
                                            llm_logger.save_checkpoint("checkpoint_llm.txt")
                                            print(f"💾 Client: Checkpoint LLM history saved: checkpoint_llm.txt")
                                        
                                        # SimpleAgent state is preserved via LLM logger history
                                        
                                        print(f"✅ Checkpoint completed at step {agent_step_count}")
                                        last_checkpoint_step = agent_step_count
                                    except Exception as e:
                                        print(f"⚠️ Checkpoint save error: {e}")
                        else:
                            print(f"🤖 Agent waiting (step {agent_step_count})")
                        
                        # Reset error counter on successful action
                        reset_error_counter()
                        
                        last_agent_time = current_time
                        
                except Exception as e:
                    handle_agent_error(e)
            
            # Small sleep to avoid busy waiting
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")
        running = False
    
    return True

def simple_mode_processing_multiprocess(vlm, game_state, args):
    """Simple mode processing for multiprocess mode using full SimpleAgent with history tracking"""
    try:
        from agent.simple import SimpleAgent
        
        # Get or create a persistent simple agent instance with full history tracking
        if not hasattr(simple_mode_processing_multiprocess, 'simple_agent'):
            simple_mode_processing_multiprocess.simple_agent = SimpleAgent(vlm)
            
            # Try to load history from LLM checkpoint if it exists
            checkpoint_llm_file = "checkpoint_llm.txt"
            if os.path.exists(checkpoint_llm_file):
                print("🔄 Loading SimpleAgent history from checkpoint_llm.txt...")
                if simple_mode_processing_multiprocess.simple_agent.load_history_from_llm_checkpoint(checkpoint_llm_file):
                    stats = simple_mode_processing_multiprocess.simple_agent.get_history_stats()
                    print("✅ SimpleAgent initialized with restored history:")
                    print(f"   - Restored {stats['history_entries']} history entries with (X,Y) positions")
                    print(f"   - Restored {stats['recent_actions']} recent actions")
                    print(f"   - Restored battle/overworld/dialogue contexts")
                    print(f"   - Restored LLM analysis and reasoning for each step")
                    print(f"   - Continuing from step #{stats['step_counter']}")
                else:
                    print("⚠️ Failed to load checkpoint history, starting fresh")
                    print("✅ SimpleAgent initialized with fresh history tracking")
            else:
                print("✅ SimpleAgent initialized with fresh history tracking:")
                print("   - Tracks last 100 states with (X,Y) positions")
                print("   - Tracks battle/overworld/dialogue contexts") 
                print("   - Stores LLM analysis and reasoning for each step")
                print("   - Manages objectives and stuck detection")
        
        frame = game_state["visual"]["screenshot"]
        
        # Use the SimpleAgent's full processing with history, objectives, and structured reasoning
        action = simple_mode_processing_multiprocess.simple_agent.process_step(frame, game_state)
        
        # Get the agent's state for debugging/monitoring
        agent_stats = simple_mode_processing_multiprocess.simple_agent.get_history_stats()
        if agent_stats["step_counter"] % 10 == 0:  # Log every 10 steps
            print(f"📊 SimpleAgent Stats: {agent_stats['history_entries']} history entries, "
                  f"{agent_stats['recent_actions']} recent actions, "
                  f"{agent_stats['objectives_count']} objectives, "
                  f"Step #{agent_stats['step_counter']}")
        
        # The SimpleAgent returns either a single action or a list of actions
        # Return the full action list for the caller to handle properly
        if isinstance(action, list):
            return action if action else ["WAIT"]
        
        # Convert single action to list for consistent handling
        return [action] if action else ["WAIT"]
        
    except Exception as e:
        print(f"Error in simple mode processing: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to basic action if SimpleAgent fails
        return "A"

def process_agent_step_multiprocess(vlm, game_state, agent_modules, args):
    """Process one agent step using the same logic as single process mode"""
    try:
        # Initialize agent state variables like single process mode
        frame = game_state["visual"]["screenshot"]
        
        # Use persistent state (agent_modules is already a parameter)
        
        # Initialize if not exists
        if not hasattr(process_agent_step_multiprocess, 'memory_context'):
            process_agent_step_multiprocess.memory_context = ""
            process_agent_step_multiprocess.observation_buffer = []
            process_agent_step_multiprocess.recent_actions = []
            process_agent_step_multiprocess.current_plan = ""
        
        # Perception step - correct signature: perception_step(frame, state_data, vlm)
        observation, slow_thinking_needed = perception_step(frame, game_state, vlm)
        
        # Update observation buffer (with frame_id for memory module)
        observation_entry = {
            "observation": observation,
            "frame_id": f"multiprocess_{len(process_agent_step_multiprocess.observation_buffer)}"
        }
        process_agent_step_multiprocess.observation_buffer.append(observation_entry)
        if len(process_agent_step_multiprocess.observation_buffer) > 5:
            process_agent_step_multiprocess.observation_buffer.pop(0)
        
        # Planning step - correct signature: planning_step(memory_context, current_plan, slow_thinking_needed, state_data, vlm)
        new_plan = planning_step(
            process_agent_step_multiprocess.memory_context,
            process_agent_step_multiprocess.current_plan,
            slow_thinking_needed,
            game_state,
            vlm
        )
        process_agent_step_multiprocess.current_plan = new_plan
        
        # Memory step - correct signature: memory_step(memory_context, current_plan, recent_actions, observation_buffer, vlm)
        new_memory = memory_step(
            process_agent_step_multiprocess.memory_context,
            process_agent_step_multiprocess.current_plan,
            process_agent_step_multiprocess.recent_actions,
            process_agent_step_multiprocess.observation_buffer,
            vlm
        )
        process_agent_step_multiprocess.memory_context = new_memory
        
        # Action step - correct signature: action_step(memory_context, current_plan, latest_observation, frame, state_data, recent_actions, vlm)
        action = action_step(
            process_agent_step_multiprocess.memory_context,
            process_agent_step_multiprocess.current_plan,
            observation,
            frame,
            game_state,
            process_agent_step_multiprocess.recent_actions,
            vlm
        )
        
        # Update recent actions
        if action and action != "WAIT":
            process_agent_step_multiprocess.recent_actions.append(action)
            if len(process_agent_step_multiprocess.recent_actions) > 20:
                process_agent_step_multiprocess.recent_actions.pop(0)
        
        return action
        
    except Exception as e:
        print(f"Error in agent step processing: {e}")
        import traceback
        traceback.print_exc()
        return "WAIT"

def run_multiprocess_client(server_port=8000, args=None):
    """Run the client component that talks to server in multiprocess mode"""
    try:
        import pygame
        
        # Initialize pygame for display
        pygame.init()
        screen_width, screen_height = 480, 320
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Pokemon Agent - Multiprocess Client")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()
        
        server_url = f"http://127.0.0.1:{server_port}"
        running = True
        frame_counter = 0
        last_good_frame = None
        
        print(f"🎮 Starting client process, connecting to server at {server_url}")
        
        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{server_url}/status", timeout=2)
                if response.status_code == 200:
                    print("✅ Connected to server")
                    break
            except:
                print(f"⏳ Waiting for server... ({i+1}/{max_retries})")
                time.sleep(1)
        else:
            print("❌ Could not connect to server")
            return False
        
        # Initialize agent if auto mode is enabled
        vlm = None
        agent_step_count = 0
        last_agent_time = 0
        
        if args and args.agent_auto:
            # Initialize VLM for agent processing
            try:
                vlm = VLM(backend=args.backend, model_name=args.model_name)
                print(f"✅ Agent initialized with {args.backend}/{args.model_name}")
            except Exception as e:
                print(f"❌ Failed to initialize agent: {e}")
                return False
        
        # Main client loop
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Send manual agent action request
                        try:
                            requests.post(f"{server_url}/action", 
                                        json={"buttons": [], "manual": False}, 
                                        timeout=30)
                        except Exception as e:
                            print(f"Agent action error: {e}")
                    else:
                        # Handle button presses
                        button_map = {
                            pygame.K_z: 'A', pygame.K_x: 'B', 
                            pygame.K_RETURN: 'START', pygame.K_RSHIFT: 'SELECT',
                            pygame.K_UP: 'UP', pygame.K_DOWN: 'DOWN',
                            pygame.K_LEFT: 'LEFT', pygame.K_RIGHT: 'RIGHT'
                        }
                        if event.key in button_map:
                            button = button_map[event.key]
                            try:
                                requests.post(f"{server_url}/action", 
                                            json={"buttons": [button], "manual": True}, 
                                            timeout=5)
                            except Exception as e:
                                print(f"Manual action error: {e}")
            
            # Agent processing (if auto mode enabled)
            current_time = time.time()
            if vlm and args.agent_auto and (current_time - last_agent_time >= 1.0):  # Agent acts every 1 second
                try:
                    # Get comprehensive state from server
                    state_response = requests.get(f"{server_url}/state", timeout=5)
                    if state_response.status_code == 200:
                        state_data = state_response.json()
                        
                        # Convert server state format to the format expected by agent modules
                        from PIL import Image
                        import base64, io
                        
                        # Get screenshot from state data
                        screenshot_base64 = state_data.get("visual", {}).get("screenshot_base64", "")
                        if screenshot_base64:
                            img_data = base64.b64decode(screenshot_base64)
                            screenshot = Image.open(io.BytesIO(img_data))
                        else:
                            screenshot = None
                        
                        # Create game state object like single process mode
                        game_state = {
                            "visual": state_data.get("visual", {}),
                            "player": state_data.get("player", {}),
                            "game": state_data.get("game", {}),
                            "map": state_data.get("map", {})
                        }
                        game_state["visual"]["screenshot"] = screenshot
                        
                        # Run the SAME agent processing as single process mode
                        if args and getattr(args, 'simple', False):
                            # Simple mode processing
                            action = simple_mode_processing_multiprocess(vlm, game_state, args)
                        else:
                            # Full agent module processing - call the actual functions
                            action = process_agent_step_multiprocess(vlm, game_state, agent_modules, args)
                        
                        # Send action to server instead of emulator
                        if action and action != "WAIT":
                            requests.post(f"{server_url}/action", 
                                        json={"buttons": [action]}, 
                                        timeout=5)
                            print(f"🤖 Agent action: {action} (step {agent_step_count})")
                        else:
                            print(f"🤖 Agent waiting (step {agent_step_count})")
                        
                        agent_step_count += 1
                        last_agent_time = current_time
                        
                except Exception as e:
                    print(f"Agent processing error: {e}")
                    time.sleep(1)  # Wait before retrying
            
            # Get current frame from server (only every few frames to maintain performance)
            frame_counter += 1
            if frame_counter % 4 == 0:  # Only fetch every 4th frame for 30 FPS display
                try:
                    response = requests.get(f"{server_url}/screenshot", timeout=0.1)
                    if response.status_code == 200:
                        frame_data = response.json().get("screenshot", "")
                        if frame_data:
                            # Decode and display frame
                            import base64
                            import io
                            img_data = base64.b64decode(frame_data)
                            img = Image.open(io.BytesIO(img_data))
                            frame_array = np.array(img)
                            frame_surface = pygame.surfarray.make_surface(frame_array.swapaxes(0, 1))
                            scaled_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))
                            screen.blit(scaled_surface, (0, 0))
                            last_good_frame = scaled_surface
                except Exception as e:
                    # Use last good frame or fill with black
                    if last_good_frame is not None:
                        screen.blit(last_good_frame, (0, 0))
                    else:
                        screen.fill((0, 0, 0))
            else:
                # Reuse last frame for performance
                if last_good_frame is not None:
                    screen.blit(last_good_frame, (0, 0))
                else:
                    screen.fill((0, 0, 0))
            
            # Add status overlay
            if font:
                status_lines = [
                    "Multiprocess Mode - Client",
                    f"Server: {server_url}",
                    "Controls: WASD=Move, Z=A, X=B, Space=Agent, Esc=Quit"
                ]
                y_offset = screen_height - 80
                for line in status_lines:
                    text_surface = font.render(line, True, (255, 255, 255))
                    bg_rect = pygame.Rect(5, y_offset-2, text_surface.get_width()+4, text_surface.get_height()+4)
                    pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect)
                    screen.blit(text_surface, (5, y_offset))
                    y_offset += 25
            
            pygame.display.flip()
            clock.tick(120)  # 120 FPS for client to match server
        
        pygame.quit()
        return True
        
    except Exception as e:
        print(f"❌ Client process error: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Direct Agent Pokemon Emerald")
    parser.add_argument("--rom", type=str, default="Emerald-GBAdvance/rom.gba", help="Path to ROM file")
    parser.add_argument("--load-state", type=str, help="Load a saved state file on startup")
    parser.add_argument("--load-checkpoint", action="store_true", help="Load from checkpoint files (checkpoint.state, checkpoint_llm.txt, checkpoint_milestones.json)")
    parser.add_argument("--backend", type=str, default="gemini", help="VLM backend (openai, gemini, local)")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash", help="Model name to use")
    parser.add_argument("--port", type=int, default=8000, help="Port for web interface")
    parser.add_argument("--no-display", action="store_true", help="Run without pygame display")
    parser.add_argument("--agent-auto", action="store_true", help="Agent acts automatically")
    parser.add_argument("--manual-mode", action="store_true", help="Start in manual mode instead of agent mode")
    parser.add_argument("--record", action="store_true", help="Record video of the gameplay")
    parser.add_argument("--simple", action="store_true", help="Simple mode: frame + LLM state input only")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR dialogue detection")
    parser.add_argument("--multiprocess", action="store_true", help="Run mGBA/pygame in separate process from agent")
    
    args = parser.parse_args()
    
    # Handle multiprocess mode first 
    if args.multiprocess:
        print("🔀 Multiprocess mode ENABLED")
        
        # Auto-start server if --agent-auto is enabled
        server_process = None
        if args.agent_auto:
            print("⚡ Auto-starting server process...")
            server_cmd = ["python", "-m", "server.app", "--port", str(args.port)]
            
            # Pass through server-relevant arguments
            if args.record:
                server_cmd.append("--record")
            if args.load_state:
                server_cmd.extend(["--load-state", args.load_state])
            if args.manual_mode:
                server_cmd.append("--manual")
            if args.no_ocr:
                server_cmd.append("--no-ocr")
            
            # Start server as subprocess
            import subprocess
            try:
                print(f"📋 Server command: {' '.join(server_cmd)}")
                server_process = subprocess.Popen(
                    server_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                print(f"✅ Server started with PID {server_process.pid}")
                print("⏳ Waiting 3 seconds for server to initialize...")
                time.sleep(3)
                
            except Exception as e:
                print(f"❌ Failed to start server: {e}")
                return 1
        else:
            print("📋 NOTE: You must run the server separately with:")
            server_cmd = "   python -m server.app"
            if args.record:
                server_cmd += " --record"
            print(server_cmd)
            print("     (Use --load-state <file> on server if needed)")
        
        print("\n🤖 Agent flags (handled by client):")
        if args.simple:
            print("   Simple mode enabled")
        if args.no_ocr:
            print("   No-OCR mode enabled")
        print(f"   Backend: {args.backend} / {args.model_name}")
        
        print(f"\n🎮 Starting client process, connecting to server at http://127.0.0.1:{args.port}")
        
        try:
            # Run headless client process (no display needed, server has the display)
            success = run_multiprocess_client_headless(args.port, args)
        except KeyboardInterrupt:
            print("Client interrupted by user")
            success = True
        finally:
            # Clean up server process if we started it
            if server_process:
                print("\n🛑 Stopping server process...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_process.kill()
        
        return 0 if success else 1
    
    # Apply command line flags to global state (single process mode)
    global agent_mode, agent_auto_enabled, simple_mode, no_ocr_mode, multiprocess_mode
    
    if args.manual_mode:
        agent_mode = False
        print("🎮 Starting in MANUAL mode (--manual-mode flag)")
    else:
        print("🤖 Starting in AGENT mode (default)")
        
    if args.agent_auto:
        agent_auto_enabled = True
        print("⚡ Auto agent ENABLED (--agent-auto flag)")
    
    if args.simple:
        simple_mode = True
        print("🚀 Simple mode ENABLED: Frame + LLM state -> Action (skipping perception/planning/memory modules)")
    
    if args.no_ocr:
        no_ocr_mode = True
        print("🚫 No-OCR mode ENABLED: Dialogue detection will use memory-only approach")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Starting Direct Agent Pokemon Emerald")
    print(f"ROM: {args.rom}")
    print(f"Backend: {args.backend} ({args.model_name})")
    print(f"Web interface: http://127.0.0.1:{args.port}")
    
    # Initialize pygame (unless disabled)
    if not args.no_display:
        init_pygame()
    
    # Determine what state to load
    state_to_load = None
    if args.load_checkpoint:
        # Check if checkpoint files exist
        checkpoint_state = "checkpoint.state"
        if os.path.exists(checkpoint_state):
            state_to_load = checkpoint_state
            print(f"🔄 Loading from checkpoint: {checkpoint_state}")
        else:
            print(f"⚠️ Checkpoint file not found: {checkpoint_state}")
            print("   Starting fresh game instead")
    elif args.load_state:
        state_to_load = args.load_state
        print(f"📂 Loading from state: {args.load_state}")
    
    # Initialize emulator
    if not setup_emulator(args.rom, state_to_load):
        print("Failed to initialize emulator")
        return 1
    
    # Initialize video recording if requested
    init_video_recording(args.record)
    
    # Initialize agent
    if not setup_agent(args.backend, args.model_name):
        print("Failed to initialize agent")
        return 1
    
    # Load checkpoint data if requested
    if args.load_checkpoint:
        print("🔄 Loading checkpoint data...")
        if load_checkpoint():
            print("✅ Checkpoint loaded successfully")
        else:
            print("⚠️ Checkpoint loading failed, continuing with fresh state")
    
    # Start agent processing thread
    global agent_processing_thread
    agent_processing_thread = threading.Thread(
        target=agent_processing_worker, 
        daemon=True
    )
    agent_processing_thread.start()
    
    # Start web server in background thread
    server_thread = threading.Thread(
        target=run_fastapi_server, 
        args=(args.port,), 
        daemon=True
    )
    server_thread.start()
    
    try:
        # Run main game loop
        game_loop(
            manual_mode=not args.no_display, 
            agent_auto=args.agent_auto
        )
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        global running, video_writer
        running = False
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_filename}")
        if emulator:
            emulator.stop()
        if not args.no_display:
            pygame.quit()
        print("Direct Agent stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
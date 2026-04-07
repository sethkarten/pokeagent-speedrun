#!/usr/bin/env python3
"""
Fixed Pokemon Emerald server - headless FastAPI server
"""

# Standard library imports
import asyncio
import base64
import datetime
import glob
import io
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

# Third-party imports
import cv2
import numpy as np
import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from urllib.parse import quote_plus

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import hashlib

# Add parent directory to path for local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local application imports — emulator imported conditionally in setup_environment()
from utils.anticheat import AntiCheatTracker
from utils.json_utils import normalize_replan_edits
from utils.llm_provider_ui import infer_llm_provider_family

# Set up logging - reduced verbosity for multiprocess mode
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Game type: "emerald" (GBA, default), "red" (Game Boy), or "browser" (Playwright)
game_type = os.environ.get("GAME_TYPE", "emerald").lower()
game_url = os.environ.get("GAME_URL", "")  # URL for browser games
browser_last_action = None  # Track last browser action for state reporting

# Global state
env = None
anticheat_tracker = None  # AntiCheat tracker for submission logging
step_counter = 0  # Track steps for submission logging
last_action_time = None  # Track time of last action for decision time calculation
running = True
step_count = 0
direct_objectives_sequence = None
direct_objectives_start_index = 0
direct_objectives_battling_start_index = 0
direct_objectives_manager = None
current_run_dir = None  # Timestamped directory for this execution run
agent_step_count = 0  # Track agent steps separately from frame steps
debug_state_enabled = False  # Save game state JSON on each get_game_state call (--debug-state)
_debug_state_counter = 0  # Step index for debug state entries
_debug_state_log = []  # Accumulates all debug state snapshots (written to debug_states.json)
current_obs = None
fps = 80


def _is_simplest_scaffold() -> bool:
    """Return True when the run is using the simplest scaffold."""
    return os.environ.get("EXCLUDE_BUILTIN_SUBAGENTS") == "1"


def _build_story_planning_objective():
    """Create scaffold-aware planning guidance when story objectives run out."""
    from agents.objectives import DirectObjective

    if _is_simplest_scaffold():
        return DirectObjective(
            id="autonomous_01_plan_more_objectives",
            description=(
                "You have reached the end of the story objectives currently available. "
                "Create more objectives to keep progressing through the game."
            ),
            action_type="create_new_objectives",
            category="story",
            target_location=None,
            navigation_hint=(
                "Review your current progress and create new objectives."
            ),
            completion_condition="objectives_created",
            priority=1,
        )

    return DirectObjective(
        id="autonomous_01_plan_more_objectives",
        description=(
            "The current story sequence is complete. Use subagent_plan_objectives "
            "to create more story objectives based on your progress and the walkthrough."
        ),
        action_type="create_new_objectives",
        category="story",
        target_location=None,
        navigation_hint=(
            "Call subagent_plan_objectives with a brief reason explaining that the "
            "current story objectives are exhausted and more objectives are needed."
        ),
        completion_condition="objectives_created",
        priority=1,
    )


def _format_interaction_type_for_ui(interaction_type: str) -> str:
    """Normalize orchestrator labels for UI readability.

    Backends log interaction_type as `<backend>_<module_name>`. For orchestrator
    entries we prefer `<scaffold>_<backend>_orchestrator` in the UI.
    """
    if not interaction_type:
        return "unknown"

    backend_prefixes = ("gemini_", "openai_", "openrouter_", "anthropic_", "vertex_")
    for prefix in backend_prefixes:
        if interaction_type.startswith(prefix) and interaction_type.endswith("_orchestrator"):
            backend = prefix[:-1]
            module_name = interaction_type[len(prefix):]
            if module_name:
                base = module_name[:-13]  # len("_orchestrator") == 13
                return f"{base}_{backend}_orchestrator"
    return interaction_type


def _provider_family_for_llm_log_entry(entry: dict) -> str:
    """UI color bucket from raw log line (handles OpenRouter via model slug)."""
    raw_type = entry.get("interaction_type") or ""
    mi = entry.get("model_info") or {}
    model_name = mi.get("model") or ""
    meta = entry.get("metadata") or {}
    backend = meta.get("backend") or mi.get("backend")
    return infer_llm_provider_family(raw_type, model_name, backend)


# Performance monitoring
last_fps_log = time.time()
frame_count_since_log = 0
action_queue = []  # Queue for multi-action sequences (now stores dicts with timing info)
current_action = None  # Current action being held
action_frames_remaining = 0  # Frames left to hold current action
release_frames_remaining = 0  # Frames left to wait after release
current_action_release_delay = 0  # Release delay for current action

### ACTION TIMING SYSTEM ###
# LLM-controlled speed presets for flexible action timing
SPEED_PRESETS = {
    "fast": {"hold": 6, "release": 3},  # 9 frames total - dialogue, menus
    "normal": {"hold": 10, "release": 8},  # 18 frames total - movement (2x faster than old default!)
    "slow": {"hold": 16, "release": 16},  # 32 frames total - careful inputs
}
DEFAULT_SPEED = "normal"

# Legacy constants for backward compatibility
ACTION_HOLD_FRAMES = SPEED_PRESETS["normal"]["hold"]
ACTION_RELEASE_DELAY = SPEED_PRESETS["normal"]["release"]

# Video recording state
video_writer = None
video_recording = False
video_filename = ""
video_frame_counter = 0
video_frame_skip = 4  # Record every 4th frame (120/4 = 30 FPS)

# Playwright WebUI recording state
playwright_recording = False       # signal flag: set False to stop the recording thread
playwright_thread = None           # background thread running the async recording loop
playwright_video_path = None       # final .mp4 path set when recording finishes

# Frame cache for separate frame server
# Use cache directory instead of /tmp
# Note: CACHE_DIR is now dynamic based on run_id - use get_cache_directory() when needed
# For frame cache, we'll initialize it dynamically
FRAME_CACHE_FILE = None  # Will be set dynamically based on run_id
frame_cache_counter = 0
frame_cache_skip_frames = 30  # Only update cache every 30 frames (4x/sec at 120 FPS)

# Map stitcher performance toggle
# Set to False when using ground truth porymap data to avoid expensive updates
ENABLE_MAP_STITCHER = False  # Disabled - using porymap ground truth instead

# State endpoint cache - cache map data by location to avoid expensive regeneration
_state_cache = {"location": None, "map_data": None, "portal_data": None, "timestamp": 0}
_state_cache_ttl = 5.0  # Cache for 5 seconds per location

# Server runs headless - display handled by client

# Threading locks for thread safety
obs_lock = threading.Lock()
step_lock = threading.Lock()
memory_lock = threading.Lock()  # New lock for memory operations to prevent race conditions


# WebSocket connection manager for frame streaming
class ConnectionManager:
    """Manages WebSocket connections for real-time frame streaming"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = threading.Lock()
        self.latest_frame = None  # Shared frame data (thread-safe)
        self.last_sent_frame_count = {}  # Track last sent frame per connection

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.active_connections.append(websocket)
            self.last_sent_frame_count[id(websocket)] = 0
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                ws_id = id(websocket)
                if ws_id in self.last_sent_frame_count:
                    del self.last_sent_frame_count[ws_id]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_latest_frame(self, websocket: WebSocket):
        """Send latest frame to a specific connection if new frame available"""
        if not self.latest_frame:
            return

        ws_id = id(websocket)
        frame_count = self.latest_frame.get("frame_count", 0)

        # Only send if this is a new frame for this connection
        if ws_id in self.last_sent_frame_count and self.last_sent_frame_count[ws_id] >= frame_count:
            return

        try:
            message = json.dumps(self.latest_frame)
            await websocket.send_text(message)
            self.last_sent_frame_count[ws_id] = frame_count
        except Exception as e:
            logger.debug(f"Failed to send frame: {e}")
            raise


frame_manager = ConnectionManager()

# Background milestone processing
state_update_thread = None
state_update_running = False

# Button mapping removed - handled by client


# Video recording functions
def init_video_recording(record_enabled=False):
    """Initialize video recording if enabled"""
    global video_writer, video_recording, video_filename, fps, video_frame_skip

    if not record_enabled:
        return

    try:
        # Save directly to run_data/end_state/videos/ to avoid copy corruption
        from utils.data_persistence.run_data_manager import get_run_data_manager
        run_manager = get_run_data_manager()
        if run_manager:
            videos_dir = run_manager.run_dir / "end_state" / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_filename = str(videos_dir / f"{run_manager.run_id}.mp4")
        else:
            # Fallback to cwd if no run manager (e.g. early init)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"pokegent_recording_{timestamp}.mp4"

        # Video settings (GBA resolution is 240x160)
        # Record at 30 FPS (skip every 4th frame from 120 FPS emulator)
        recording_fps = fps / video_frame_skip  # 120 / 4 = 30 FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filename, fourcc, float(recording_fps), (240, 160))

        if video_writer.isOpened():
            video_recording = True
            print(
                f"📹 Video recording started: {video_filename} at {recording_fps:.0f} FPS (recording every {video_frame_skip} frames)"
            )
        else:
            print("❌ Failed to initialize video recording")
            video_writer = None

    except Exception as e:
        print(f"❌ Video recording initialization error: {e}")
        video_writer = None


def update_frame_cache(screenshot):
    """Update frame cache for WebSocket streaming - NO FILE I/O!

    Stores frames in memory for WebSocket clients to consume.
    Performance: Updates every 2 frames = 40 FPS stream rate at 80 game FPS.
    """
    global frame_cache_counter

    if screenshot is None:
        return

    # Increment counter on every call
    frame_cache_counter += 1

    # For emulator games (80 FPS), skip every other frame for 40 FPS stream.
    # For browser games, frames are rare (one per agent step) — never skip.
    if game_type != "browser" and frame_cache_counter % 2 != 0:
        return

    try:
        # Convert screenshot to base64
        if hasattr(screenshot, "save"):  # PIL image
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
        elif isinstance(screenshot, np.ndarray):  # Numpy array
            pil_image = Image.fromarray(screenshot)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
        else:
            return

        # Store frame in memory for WebSocket clients (thread-safe)
        frame_manager.latest_frame = {
            "frame_data": img_str,
            "frame_count": frame_cache_counter,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.debug(f"Frame update failed: {e}")


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
        if hasattr(screenshot, "save"):  # PIL image
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
        logger = logging.getLogger(__name__)
        logger.debug(f"Video recording frame error: {e}")


def cleanup_video_recording():
    """Clean up video recording resources"""
    global video_writer, video_recording

    if video_recording and video_writer is not None:
        try:
            video_writer.release()
            print(f"📹 Video recording saved: {video_filename}")
        except Exception as e:
            print(f"❌ Error saving video recording: {e}")
        finally:
            video_writer = None
            video_recording = False


# Milestone tracking is now handled by the emulator

# FastAPI app
app = FastAPI(
    title="PokeAgent Challenge",
    description="Streamer display FastAPI endpoints",
    version="3.0.0-preview",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models for API requests and responses
class ActionRequest(BaseModel):
    buttons: List[str] = Field(
        default_factory=list
    )  # List of button names: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT, WAIT
    speed: Optional[str] = "normal"  # Action speed: "fast", "normal", or "slow"
    hold_frames: Optional[int] = None  # Optional explicit hold duration (overrides speed preset)
    release_frames: Optional[int] = None  # Optional explicit release duration (overrides speed preset)
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GameStateResponse(BaseModel):
    screenshot_base64: str
    step_number: int
    resolution: list  # [width, height]
    status: str


class ComprehensiveStateResponse(BaseModel):
    visual: dict
    player: dict
    game: dict
    map: dict
    milestones: dict = {}
    location_connections: dict = {}  # Add location connections for portal display
    step_number: int
    status: str
    action_queue_length: int = 0


def periodic_milestone_updater():
    """Lightweight background thread that only updates milestones occasionally"""
    global state_update_running

    last_milestone_update = 0

    while state_update_running and running:
        try:
            current_time = time.time()

            # Update milestones only every 5 seconds (much less frequent)
            if current_time - last_milestone_update >= 5.0:
                if env and env.memory_reader:
                    try:
                        # Use lightweight state for milestone updates only
                        party = env.get_party_pokemon() or []
                        location = env.get_location()
                        badges = []
                        if hasattr(env, 'memory_reader') and env.memory_reader and hasattr(env.memory_reader, 'read_badges'):
                            try:
                                badges = env.memory_reader.read_badges() or []
                            except Exception:
                                badges = []
                        basic_state = {
                            "player": {
                                "money": env.get_money(),
                                "party_size": len(env.get_party_pokemon() or []),
                                "position": env.get_coordinates(),
                            },
                            "map": {"location": env.get_location()},
                        }
                        env.check_and_update_milestones(basic_state, agent_step_count=agent_step_count)
                        last_milestone_update = current_time
                        logger.debug("Lightweight milestone update completed")
                    except Exception as e:
                        logger.debug(f"Milestone update failed: {e}")

            # Sleep for 1 second between checks
            time.sleep(1.0)

        except Exception as e:
            logger.error(f"Error in milestone updater: {e}")
            time.sleep(5.0)  # Wait longer on error


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global running, state_update_running, video_recording, video_filename

    # Prevent multiple signal handlers from running simultaneously
    if not running:
        return

    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False
    state_update_running = False

    # IMPORTANT: Finalize run data BEFORE cleanup
    # Check video_recording flag BEFORE cleanup_video_recording() resets it
    was_recording = video_recording

    try:
        from utils.data_persistence.run_data_manager import get_run_data_manager
        from utils.data_persistence.llm_logger import get_llm_logger

        run_manager = get_run_data_manager()
        if run_manager:
            print("📦 Finalizing run data...")

            # Get final metrics from LLM logger
            llm_logger = get_llm_logger()
            final_metrics = llm_logger.get_cumulative_metrics() if llm_logger else None

            # Save end-state snapshot (ensures all data is saved)
            run_manager.save_end_state_snapshot()

            # Copy all data to run_data
            logger.info(f"🔍 [DEBUG] Finalizing run data - run_manager: {run_manager is not None}")
            if llm_logger:
                logger.info(f"🔍 [DEBUG] LLM logger available, log_file: {llm_logger.log_file}")
                # Verify LLM log file exists before copying
                if os.path.exists(llm_logger.log_file):
                    logger.info(f"🔍 [DEBUG] LLM log file exists, copying...")
                    run_manager.copy_llm_traces(llm_logger.log_file)
                    logger.info(f"🔍 [DEBUG] Copied LLM traces from: {llm_logger.log_file}")
                else:
                    logger.warning(f"🔍 [DEBUG] LLM log file not found: {llm_logger.log_file}")
                    logger.warning(f"🔍 [DEBUG] Current working directory: {os.getcwd()}")
                    # Try to find the log file by pattern
                    import glob

                    log_pattern = f"llm_logs/llm_log_{run_manager.run_id.split('_', 1)[1] if '_' in run_manager.run_id else '*'}*.jsonl"
                    logger.info(f"🔍 [DEBUG] Searching for log files with pattern: {log_pattern}")
                    log_files = glob.glob(log_pattern)
                    logger.info(f"🔍 [DEBUG] Found {len(log_files)} log files: {log_files}")
                    if log_files:
                        # Use the most recent one
                        log_file = max(log_files, key=os.path.getmtime)
                        logger.info(f"🔍 [DEBUG] Using most recent log file: {log_file}")
                        run_manager.copy_llm_traces(log_file)
                        logger.info(f"🔍 [DEBUG] Found and copied LLM traces from: {log_file}")
            else:
                logger.warning(f"🔍 [DEBUG] LLM logger is None - cannot copy LLM traces")

            if os.environ.get("POKEAGENT_CLI_MODE") != "1":
                run_manager.copy_objectives()
                run_manager.copy_memory()

            # Sync trajectories from cache to run_data before finalizing
            run_manager.sync_trajectories_to_run_data()

            # Copy frame_cache to end_state
            run_manager.copy_frame_cache()

            # Copy video if recording was enabled (check flag BEFORE cleanup)
            logger.info(f"🔍 [DEBUG] Video recording flag: {was_recording}, video_filename: {video_filename}")
            run_manager.copy_video_recording(record_enabled=was_recording)

            # Finalize with metrics
            run_manager.finalize_run(final_metrics=final_metrics)
            print(f"✅ Run data finalized: {run_manager.get_run_directory()}")
    except Exception as e:
        logger.error(f"❌ Error during run data finalization: {e}", exc_info=True)

    # Cleanup video recording AFTER copying (so file is still available)
    cleanup_video_recording()

    if env:
        env.stop()

    sys.exit(0)


def setup_environment(skip_initial_state=False):
    """Initialize the emulator"""
    global env, current_obs, anticheat_tracker

    try:
        if game_type == "browser":
            from browser_env.browser_emulator import BrowserEnv
            if not game_url:
                raise RuntimeError("GAME_URL environment variable required for browser games")
            browser_headed = os.environ.get("BROWSER_HEADED", "0") == "1"
            env = BrowserEnv(game_url=game_url, headless=not browser_headed)
            env.initialize()

            screenshot = env.get_screenshot()
            if screenshot:
                with obs_lock:
                    current_obs = np.array(screenshot)
            else:
                with obs_lock:
                    current_obs = np.zeros((env.height, env.width, 3), dtype=np.uint8)

            print(f"Browser environment initialized for {game_url}")
            return True

        if game_type == "red":
            from pokemon_red_env.red_emulator import RedEmulator
            rom_path = "PokemonRed-GBC/pokered.gbc"
            if not os.path.exists(rom_path):
                raise RuntimeError(f"ROM not found at {rom_path}")
            env = RedEmulator(rom_path=rom_path)
        else:
            from pokemon_env.emulator import EmeraldEmulator
            rom_path = "Emerald-GBAdvance/rom.gba"
            if not os.path.exists(rom_path):
                raise RuntimeError(f"ROM not found at {rom_path}")
            env = EmeraldEmulator(rom_path=rom_path)
        env.initialize()

        # Initialize AntiCheat tracker for submission logging
        anticheat_tracker = AntiCheatTracker()
        anticheat_tracker.initialize_submission_log("SERVER_MODE")
        print("AntiCheat tracker initialized for submission logging")

        # Mark GAME_RUNNING milestone as completed at startup
        # But defer the expensive state logging - it will happen on first state request
        if not skip_initial_state:
            try:
                env.milestone_tracker.mark_completed("GAME_RUNNING")
                print("GAME_RUNNING milestone marked - initial state logging deferred")
            except Exception as e:
                print(f"Warning: Could not mark initial milestone: {e}")

        screenshot = env.get_screenshot()
        if screenshot:
            with obs_lock:
                current_obs = np.array(screenshot)
        else:
            with obs_lock:
                current_obs = np.zeros((env.height, env.width, 3), dtype=np.uint8)

        print("Emulator initialized successfully!")
        return True

    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return False


def handle_input(manual_mode=False):
    """Handle input - server runs headless, no input handling needed"""
    # Server always runs headless - input handled by client via HTTP API
    return True, []


def step_environment(actions_pressed):
    """Take a step in the environment with optimized locking for better performance"""
    global current_obs

    # Debug: print what actions are being sent to emulator
    # if actions_pressed:
    # print( Stepping emulator with actions: {actions_pressed}")

    # Only use memory_lock for the essential emulator step
    with memory_lock:
        env.run_frame_with_buttons(actions_pressed)

        # Do lightweight area transition detection inside the lock
        if hasattr(env, "memory_reader") and env.memory_reader:
            try:
                transition_detected = env.memory_reader._check_area_transition()
                if transition_detected:
                    logger.info("Area transition detected")
                    env.memory_reader.invalidate_map_cache()
                    if hasattr(env.memory_reader, "_cached_behaviors"):
                        env.memory_reader._cached_behaviors = None
                    if hasattr(env.memory_reader, "_cached_behaviors_map_key"):
                        env.memory_reader._cached_behaviors_map_key = None
                    # Set flag to trigger map stitcher update outside the lock
                    env.memory_reader._area_transition_detected = True
            except Exception as e:
                logger.warning(f"Area transition check failed: {e}")

    # Update screenshot outside the memory lock to reduce contention
    try:
        screenshot = env.get_screenshot()
        if screenshot:
            if not playwright_recording:
                record_frame(screenshot)
            update_frame_cache(screenshot)  # Update frame cache for separate frame server
            with obs_lock:
                current_obs = np.array(screenshot)

            # Update map stitcher on position changes (lightweight approach)
            # This ensures map data stays current as player moves
            # DISABLED: Using porymap ground truth instead for better performance
            if ENABLE_MAP_STITCHER and hasattr(env, "memory_reader") and env.memory_reader:
                try:
                    # Check if player position has changed
                    should_update = False

                    # Get current player coordinates and map info
                    current_coords = env.memory_reader.read_coordinates()
                    current_map_bank = env.memory_reader._read_u8(env.memory_reader.addresses.MAP_BANK)
                    current_map_number = env.memory_reader._read_u8(env.memory_reader.addresses.MAP_NUMBER)
                    current_map_info = (current_map_bank, current_map_number)

                    # Initialize tracking variables if needed
                    if not hasattr(env, "_last_player_coords"):
                        env._last_player_coords = None
                        env._last_map_info = None

                    # Check for position changes
                    if current_coords != env._last_player_coords or current_map_info != env._last_map_info:
                        should_update = True
                        env._last_player_coords = current_coords
                        env._last_map_info = current_map_info
                        logger.debug(f"Position change detected: {current_coords}, map: {current_map_info}")
                        logger.debug(
                            f"Map stitcher update triggered by position change: {current_coords}, map: {current_map_info}"
                        )

                    # Always update on area transitions (already detected above)
                    if (
                        hasattr(env.memory_reader, "_area_transition_detected")
                        and env.memory_reader._area_transition_detected
                    ):
                        should_update = True
                        env.memory_reader._area_transition_detected = False  # Reset flag
                        logger.debug("Map stitcher update triggered by area transition")

                    # Update map stitcher directly when position changes
                    if should_update:
                        # @TODO should do location change warps here too
                        logger.debug("Triggering map stitcher update for position change")
                        # Call map stitcher update directly without full map reading
                        tiles = env.memory_reader.read_map_around_player(radius=7)
                        if tiles:
                            logger.debug(f"Got {len(tiles)} tiles, updating map stitcher")
                            state = {"map": {}}  # Basic state for stitcher
                            env.memory_reader._update_map_stitcher(tiles, state)
                            logger.debug("Map stitcher updated for position change")
                            logger.debug("Map stitcher update completed")
                        else:
                            logger.debug("No tiles found for map stitcher update")

                except Exception as e:
                    logger.error(f"Failed to update map stitcher during movement: {e}")
    except Exception as e:
        logger.warning(f"Error updating screenshot: {e}")


def update_display(manual_mode=False):
    """Update display - server runs headless, no display update needed"""
    # Server runs headless - display handled by client
    pass


def draw_info_overlay():
    """Draw info overlay - server runs headless, no overlay needed"""
    # Server runs headless - overlay handled by client
    pass


def save_screenshot():
    """Save current screenshot"""
    global current_obs

    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None

    if obs_copy is not None:
        timestamp = int(time.time())
        filename = f"simple_test_screenshot_{timestamp}.png"
        img = Image.fromarray(obs_copy)
        img.save(filename)
        print(f"Screenshot saved: {filename}")


def reset_game():
    """Reset the game and all milestones"""
    global env, step_count

    print("Resetting game and milestones...")
    with step_lock:
        env.initialize()
        env.milestone_tracker.reset_all()  # Reset all milestones
        step_count = 0
    print("Game and milestone reset complete")


def game_loop(manual_mode=False):
    """Main game loop - runs in main thread, always headless"""
    global running, step_count

    # Browser games don't need a frame-by-frame loop — the browser runs its own event loop.
    # Just keep the thread alive so the server stays up.
    if game_type == "browser":
        print("Starting browser game loop (idle — browser runs its own event loop)...")
        while running:
            time.sleep(0.5)
        return

    print("Starting headless game loop...")

    while running:
        # Handle input
        should_continue, actions_pressed = handle_input(manual_mode)
        if not should_continue:
            break

        # In server mode, handle action queue with proper button hold timing
        action_completed = False
        if not manual_mode:
            global current_action, action_frames_remaining, release_frames_remaining, current_action_release_delay

            if current_action and action_frames_remaining > 0:
                # Continue holding the current action (WAIT actions press nothing)
                if current_action == "WAIT":
                    actions_pressed = []
                else:
                    actions_pressed = [current_action]
                action_frames_remaining -= 1
                if action_frames_remaining == 0:
                    # Action finished, start release delay
                    # Record ending position for this action (skip if queue is large for performance)
                    global recent_button_presses

                    # Only update position tracking if queue is small (<10 actions)
                    # This prevents expensive state reads during batch action processing
                    if len(action_queue) < 10:
                        current_state = env.get_comprehensive_state()
                        player_data = current_state.get("player", {})
                        position = player_data.get("position", {})
                        location = player_data.get("location", "Unknown")
                        end_pos = (
                            (position.get("x"), position.get("y"), location) if position else (None, None, location)
                        )

                        # Find and update the most recent incomplete action matching current_action
                        for i in range(len(recent_button_presses) - 1, -1, -1):
                            if (
                                recent_button_presses[i]["button"] == current_action
                                and not recent_button_presses[i]["completed"]
                            ):
                                recent_button_presses[i]["end_pos"] = end_pos
                                recent_button_presses[i]["completed"] = True
                                break
                    else:
                        # For large queues, just mark as completed without position update
                        for i in range(len(recent_button_presses) - 1, -1, -1):
                            if (
                                recent_button_presses[i]["button"] == current_action
                                and not recent_button_presses[i]["completed"]
                            ):
                                recent_button_presses[i]["end_pos"] = (None, None, "Unknown")
                                recent_button_presses[i]["completed"] = True
                                break

                    current_action = None
                    release_frames_remaining = current_action_release_delay
                    action_completed = True  # Mark action as completed
                    print(f"✅ Action completed: step_count will increment")
            elif release_frames_remaining > 0:
                # Release delay (no button pressed)
                actions_pressed = []
                release_frames_remaining -= 1
            elif action_queue:
                # Start a new action from the queue
                current_action_data = action_queue.pop(0)

                # Handle both old format (string) and new format (dict) for backward compatibility
                if isinstance(current_action_data, str):
                    current_action = current_action_data
                    timing = SPEED_PRESETS[DEFAULT_SPEED]
                else:
                    current_action = current_action_data["button"]
                    speed = current_action_data.get("speed", DEFAULT_SPEED)
                    timing = SPEED_PRESETS.get(speed, SPEED_PRESETS[DEFAULT_SPEED])

                    # Allow explicit frame overrides
                    if current_action_data.get("hold_frames") is not None:
                        timing = timing.copy()  # Don't modify the preset
                        timing["hold"] = current_action_data["hold_frames"]
                    if current_action_data.get("release_frames") is not None:
                        timing = timing.copy()
                        timing["release"] = current_action_data["release_frames"]

                # Special handling for WAIT action
                if current_action == "WAIT":
                    action_frames_remaining = 0  # Don't actually hold any button
                    current_action_release_delay = timing["release"]  # Wait duration is in release
                else:
                    action_frames_remaining = timing["hold"]
                    current_action_release_delay = timing["release"]

                actions_pressed = [] if current_action == "WAIT" else [current_action]
                queue_len = len(action_queue)

                # Get current FPS for estimation
                current_fps_for_calc = env.get_current_fps(fps) if env else fps
                estimated_time = queue_len * (timing["hold"] + timing["release"]) / current_fps_for_calc
                speed_indicator = f" [{speed}]" if isinstance(current_action_data, dict) else ""
                print(
                    f"🎮 Server processing action: {current_action}{speed_indicator}, Queue remaining: {queue_len} actions (~{estimated_time:.1f}s)"
                )
            else:
                # No action to process
                actions_pressed = []

        # Step environment
        step_environment(actions_pressed)

        # Milestones are now updated in background thread

        # Server runs headless - no display update needed
        update_display(manual_mode)

        # Only increment step count when an action is completed
        if action_completed:
            with step_lock:
                step_count += 1
                print(f"📈 Step count incremented to: {step_count}")

        # Performance monitoring - log actual FPS every 5 seconds
        global last_fps_log, frame_count_since_log
        frame_count_since_log += 1
        current_time = time.time()
        if current_time - last_fps_log >= 5.0:  # Log every 5 seconds
            actual_fps = frame_count_since_log / (current_time - last_fps_log)
            queue_len = len(action_queue)
            print(f"📊 Server FPS: {actual_fps:.1f} (target: {fps}), Queue: {queue_len} actions")
            last_fps_log = current_time
            frame_count_since_log = 0

        # Use dynamic FPS - 2x speed during dialog
        current_fps = env.get_current_fps(fps) if env else fps
        # Simple sleep - more reliable than complex timing
        time.sleep(1.0 / current_fps)


def run_fastapi_server(port):
    """Run FastAPI server in background thread"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="error",
        access_log=False,
        timeout_keep_alive=60,  # Keep connections alive longer
        timeout_graceful_shutdown=30,  # More time for graceful shutdown
    )


# ---------------------------------------------------------------------------
# Continuous frame polling for browser games
# ---------------------------------------------------------------------------
# Browser games are turn-based at the agent level (no internal game loop on
# the server side), so update_frame_cache() only fires when an MCP action
# endpoint runs. That leaves the /ws/frames stream frozen for ~17 s between
# agent steps even though the canvas is animating (Unity particles, NPCs,
# loading bars, etc). This background task polls the BrowserEnv at a fixed
# cadence so the stream stays live even while the agent is mid-VLM-call.
#
# Tunable via BROWSER_STREAM_FPS (default 4, set to 0 to disable). Each
# screenshot dispatches into the Playwright thread via env._call which is
# already serialized, so concurrent polls + agent get_game_state calls are
# safe — they queue.

_browser_poll_task = None


async def _browser_frame_poll_loop():
    """Continuously push fresh screenshots to frame_manager.latest_frame.

    Runs as an asyncio task on the FastAPI event loop. Uses asyncio.to_thread
    to avoid blocking the event loop on the synchronous env.get_screenshot
    call (which dispatches across the Playwright thread).
    """
    import asyncio

    try:
        fps = float(os.environ.get("BROWSER_STREAM_FPS", "4"))
    except (TypeError, ValueError):
        fps = 4.0
    if fps <= 0:
        logger.info("Browser frame poll disabled (BROWSER_STREAM_FPS=0)")
        return
    interval = 1.0 / fps
    logger.info(f"Browser frame poll loop started at {fps:.1f} fps")

    while True:
        try:
            if env is not None and hasattr(env, "get_screenshot"):
                screenshot = await asyncio.to_thread(env.get_screenshot)
                if screenshot is not None:
                    update_frame_cache(screenshot)
        except Exception as e:
            logger.debug(f"Browser frame poll iteration failed: {e}")
        await asyncio.sleep(interval)


@app.on_event("startup")
async def _start_browser_frame_poll():
    """Spawn the frame poller after uvicorn starts the event loop.

    Only fires for browser games. For Pokemon games the existing emulator
    loop already pushes frames at 40 fps via update_frame_cache, so this
    poller would be redundant.
    """
    import asyncio

    global _browser_poll_task
    if game_type != "browser":
        return
    if _browser_poll_task is not None:
        return
    _browser_poll_task = asyncio.create_task(_browser_frame_poll_loop())


# Serve stream.html
@app.get("/stream")
async def get_stream():
    """Serve the stream.html interface"""
    try:
        with open("server/stream.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Stream interface not found")


# FastAPI endpoints
@app.get("/config")
async def get_config():
    """Return server configuration (game type, resolution) for dynamic UI"""
    return {
        "game": game_type,
        "width": env.width if env else 240,
        "height": env.height if env else 160,
    }


@app.get("/health")
async def get_health():
    """Health check endpoint for server monitoring"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/status")
async def get_status():
    """Get server status"""
    with step_lock:
        current_step = step_count

    if game_type == "browser":
        return {
            "status": "running",
            "game_type": "browser",
            "step_count": current_step,
        }

    # Get current FPS (may be 4x during dialog)
    current_fps = env.get_current_fps(fps) if env else fps
    # Use cached dialog state for consistency with FPS calculation
    is_dialog = env._cached_dialog_state if env else False

    return {
        "status": "running",
        "step_count": current_step,
        "base_fps": fps,
        "current_fps": current_fps,
        "is_dialog": is_dialog,
        "fps_multiplier": 2 if is_dialog else 1,
    }


@app.get("/screenshot")
async def get_screenshot():
    """Get current screenshot"""
    global current_obs, step_count

    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")

    if game_type == "browser":
        try:
            screenshot = env.get_screenshot()
            screenshot_b64 = _pil_to_base64(screenshot)
            return {"screenshot": screenshot_b64, "step_count": step_count}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None

    if obs_copy is None:
        raise HTTPException(status_code=500, detail="No screenshot available")

    try:
        # Convert numpy array to PIL image
        pil_image = Image.fromarray(obs_copy)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        with step_lock:
            current_step = step_count

        return GameStateResponse(
            screenshot_base64=img_str,
            step_number=current_step,
            resolution=[obs_copy.shape[1], obs_copy.shape[0]],
            status="running",
        )

    except Exception as e:
        logger.error(f"Error getting screenshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/frame")
async def get_latest_frame():
    """Get latest game frame in same format as single-process mode"""
    global current_obs, env

    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None

    # If current_obs is None (e.g., after server restart), try to get a fresh screenshot
    if obs_copy is None and env:
        try:
            screenshot = env.get_screenshot()
            if screenshot:
                obs_copy = np.array(screenshot)
                # Update current_obs for future requests
                with obs_lock:
                    current_obs = obs_copy.copy()
                logger.debug("Frame endpoint: Retrieved fresh screenshot after restart")
        except Exception as e:
            logger.warning(f"Frame endpoint: Failed to get fresh screenshot: {e}")

    if obs_copy is None:
        return {"frame": ""}

    try:
        # Convert to base64
        pil_image = Image.fromarray(obs_copy)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {"frame": img_str}
    except Exception as e:
        logger.warning(f"Frame endpoint: Error encoding frame: {e}")
        return {"frame": ""}


@app.get("/frame")
async def get_frame_from_cache():
    """DISABLED - Use WebSocket /ws/frames for real-time streaming instead"""
    raise HTTPException(
        status_code=410,
        detail="HTTP polling disabled. Use WebSocket endpoint /ws/frames for real-time frame streaming.",
    )


@app.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket):
    """WebSocket endpoint for real-time frame streaming

    This endpoint continuously sends new frames as they become available.
    The game loop updates frame_manager.latest_frame, and we send it here.
    """
    import asyncio

    await frame_manager.connect(websocket)
    try:
        # Continuously send frames while connection is alive
        while True:
            try:
                # Send latest frame if available (non-blocking check)
                await frame_manager.send_latest_frame(websocket)

                # Small sleep to avoid busy-waiting (check ~60 times per second)
                await asyncio.sleep(0.016)  # ~60 FPS check rate

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"Error in frame streaming: {e}")
                break
    finally:
        frame_manager.disconnect(websocket)


@app.post("/action")
async def take_action(request: ActionRequest):
    """Take an action"""
    global current_obs, step_count, recent_button_presses, action_queue, anticheat_tracker, step_counter, last_action_time

    # print( Action endpoint called with request: {request}")
    # print( Request buttons: {request.buttons}")

    if env is None:
        # print( Emulator not initialized")
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Add all actions to the queue (handle both single actions and lists)
        if request.buttons:
            # Get timing parameters
            speed = request.speed or DEFAULT_SPEED
            hold_frames = request.hold_frames
            release_frames = request.release_frames

            # Validate speed parameter
            if speed not in SPEED_PRESETS:
                speed = DEFAULT_SPEED
                print(f"⚠️ Invalid speed '{request.speed}', using '{DEFAULT_SPEED}'")

            # Add ALL actions to the queue - let the game loop handle execution
            speed_info = f" [speed={speed}]" if speed != DEFAULT_SPEED else ""
            frame_info = ""
            if hold_frames is not None or release_frames is not None:
                frame_info = f" [hold={hold_frames}, release={release_frames}]"

            print(f"📡 Server received actions: {request.buttons}{speed_info}{frame_info}")
            print(f"📋 Action queue before extend: {action_queue}")

            # Create action data with timing info
            for button in request.buttons:
                action_data = {
                    "button": button,
                    "speed": speed,
                    "hold_frames": hold_frames,
                    "release_frames": release_frames,
                }
                action_queue.append(action_data)

            print(f"📋 Action queue after extend: {len(action_queue)} actions")

            # Track button presses for recent actions display with position tracking
            current_time = time.time()

            # Only read position if queue is small (<20 actions) to avoid performance impact
            # For large queues, skip position tracking to maintain FPS
            if len(action_queue) < 20:
                # Get current player position (use cached state - 100ms cache in emulator)
                current_state = env.get_comprehensive_state()  # Already cached internally
                player_data = current_state.get("player", {})
                position = player_data.get("position", {})
                location = player_data.get("location", "Unknown")
                start_pos = (position.get("x"), position.get("y"), location) if position else (None, None, location)
            else:
                # Skip expensive state read for large queues
                start_pos = (None, None, "Unknown")

            source = request.source
            metadata = request.metadata or {}
            sequence_length = len(request.buttons)

            for idx, button in enumerate(request.buttons):
                # Add all buttons to recent actions with starting position
                action_entry = {
                    "button": button,
                    "timestamp": current_time,
                    "start_pos": start_pos,
                    "end_pos": None,  # Will be filled when action completes
                    "completed": False,
                    "sequence_index": idx,
                    "sequence_length": sequence_length,
                }

                if source:
                    action_entry["source"] = source
                if metadata:
                    action_entry["metadata"] = dict(metadata)

                recent_button_presses.append(action_entry)

            # Update total actions count in metrics
            with step_lock:
                latest_metrics["total_actions"] = latest_metrics.get("total_actions", 0) + len(request.buttons)

                # Also update the LLM logger's action count and gameplay time for checkpoint persistence
                try:
                    from utils.data_persistence.llm_logger import get_llm_logger

                    llm_logger = get_llm_logger()
                    if llm_logger:
                        llm_logger.cumulative_metrics["total_actions"] = latest_metrics["total_actions"]

                        # Update gameplay time for button presses too
                        current_time = time.time()
                        time_since_last_update = current_time - llm_logger.cumulative_metrics.get(
                            "last_update_time", current_time
                        )
                        # Only add time if it's reasonable (less than 5 minutes since last interaction)
                        if time_since_last_update < 300:  # 5 minutes
                            llm_logger.cumulative_metrics["total_run_time"] = (
                                llm_logger.cumulative_metrics.get("total_run_time", 0) + time_since_last_update
                            )
                        llm_logger.cumulative_metrics["last_update_time"] = current_time

                        # Save metrics to cache file
                        llm_logger.save_cumulative_metrics()

                        # Sync LLM logger's cumulative metrics back to latest_metrics
                        # This ensures token usage and costs from LLM interactions are displayed
                        cumulative_metrics_to_sync = [
                            "total_tokens",
                            "prompt_tokens",
                            "completion_tokens",
                            "total_cost",
                            "total_llm_calls",
                            "total_run_time",
                        ]
                        for metric_key in cumulative_metrics_to_sync:
                            if metric_key in llm_logger.cumulative_metrics:
                                latest_metrics[metric_key] = llm_logger.cumulative_metrics[metric_key]
                except Exception as e:
                    logger.debug(f"Failed to sync metrics with LLM logger: {e}")

            # Keep only last 50 button presses to avoid memory issues
            if len(recent_button_presses) > 50:
                recent_button_presses = recent_button_presses[-50:]
        else:
            print(f" No buttons in request")

        # DON'T execute action here - let the game loop handle it from the queue
        # This prevents conflicts between the API thread and pygame thread

        # Return immediate success - avoid all locks to prevent deadlocks
        actions_added = len(request.buttons) if request.buttons else 0

        # print( Returning success, actions_added: {actions_added}, queue_length: {len(action_queue)}")

        # Log action to submission.log if anticheat tracker is available
        if anticheat_tracker and request.buttons:
            try:
                # Calculate decision time
                current_time = time.time()
                if last_action_time is not None:
                    decision_time = current_time - last_action_time
                else:
                    decision_time = 0.0  # First action
                last_action_time = current_time

                # Skip expensive state reads and logging when queue is large (>30 actions)
                # This prevents FPS drops during batch action processing
                if len(action_queue) < 30:
                    # Get current game state for logging
                    game_state = env.get_comprehensive_state()
                    action_taken = request.buttons[0] if request.buttons else "NONE"  # Log first action

                    # Create simple state hash
                    import hashlib

                    state_str = str(game_state)
                    state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]

                    # Determine if this is manual mode (from client) or agent mode
                    # For now, assume manual mode if coming through API
                    manual_mode = request.source == "manual" if hasattr(request, "source") else True

                    # Get the latest milestone from the emulator's milestone tracker
                    # First, trigger an immediate milestone check to ensure current state is detected
                    latest_milestone = "NONE"
                    if env and hasattr(env, "milestone_tracker"):
                        try:
                            # Force an immediate milestone check before logging
                            env.check_and_update_milestones(game_state, agent_step_count=agent_step_count)
                        except Exception as e:
                            logger.debug(f"Error during immediate milestone check: {e}")

                        milestone_name, split_time, total_time = env.milestone_tracker.get_latest_milestone_info()
                        latest_milestone = milestone_name if milestone_name != "NONE" else "NONE"

                    # Log the action
                    step_counter += 1
                    anticheat_tracker.log_submission_data(
                        step=step_counter,
                        state_data=game_state,
                        action_taken=action_taken,
                        decision_time=decision_time,
                        state_hash=state_hash,
                        manual_mode=manual_mode,
                        milestone_override=latest_milestone,
                    )
                else:
                    # For large queues, skip detailed logging to maintain FPS
                    logger.debug(f"Skipping submission logging - queue size: {len(action_queue)}")

            except Exception as e:
                logger.warning(f"Error logging to submission.log: {e}")

        # Return lightweight response without any lock acquisition
        return {
            "status": "success",
            "actions_queued": actions_added,
            "queue_length": len(action_queue),  # action_queue access is atomic for lists
            "message": f"Added {actions_added} actions to queue",
        }

    except Exception as e:
        # print( Exception in action endpoint: {e}")
        logger.error(f"Error taking action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue_status")
async def get_queue_status():
    """Get action queue status"""
    global action_queue, current_action, action_frames_remaining, release_frames_remaining

    queue_empty = (
        len(action_queue) == 0
        and current_action is None
        and action_frames_remaining == 0
        and release_frames_remaining == 0
    )

    return {
        "queue_empty": queue_empty,
        "queue_length": len(action_queue),
        "current_action": current_action,
        "action_frames_remaining": action_frames_remaining,
        "release_frames_remaining": release_frames_remaining,
    }


def _build_porymap_visual_map_15x15(state: Dict[str, Any], player_coords: Tuple[int, int]) -> bool:
    """Set ``state['map']['visual_map']`` from elevation-filtered ``porymap.grid`` (15x15 around player).

    Uses whatever grid is already in ``state`` (refreshed while idle, or from /state cache). Safe when
    the action queue is non-empty: recenters the window on *current* player coordinates without
    re-running ``_format_porymap_info``.

    Returns True if ``visual_map`` was written.
    """
    map_section = state.get("map") or {}
    porymap_grid = (map_section.get("porymap") or {}).get("grid")
    if not porymap_grid or not player_coords:
        return False

    px, py = int(player_coords[0]), int(player_coords[1])
    window_size = 15
    half_window = window_size // 2
    height = len(porymap_grid)

    def _slice_row(row: Any, sx: int, ex: int) -> List[str]:
        if isinstance(row, str):
            return list(row[sx:ex])
        return [str(c) for c in row[sx:ex]]

    start_y = max(0, py - half_window)
    end_y = min(height, py + half_window + 1)
    start_x = max(0, px - half_window)

    visual_grid: List[List[str]] = []
    for y in range(start_y, end_y):
        if y < height:
            row = porymap_grid[y]
            end_x = min(len(row), px + half_window + 1)
            window_row = _slice_row(row, start_x, end_x)
            while len(window_row) < window_size:
                window_row.append("#")
            visual_grid.append(window_row[:window_size])
        else:
            visual_grid.append(["#"] * window_size)

    while len(visual_grid) < window_size:
        visual_grid.append(["#"] * window_size)

    player_y_in_window = py - start_y
    player_x_in_window = px - start_x
    if 0 <= player_y_in_window < len(visual_grid) and 0 <= player_x_in_window < len(
        visual_grid[player_y_in_window]
    ):
        visual_grid[player_y_in_window][player_x_in_window] = "P"

    if "map" not in state:
        state["map"] = {}
    state["map"]["visual_map"] = "\n".join(" ".join(r) for r in visual_grid)
    state["map"]["map_source"] = "porymap_with_player_15x15"
    return True


@app.get("/state")
async def get_comprehensive_state():
    """Get comprehensive game state including visual and memory data"""
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")

    if game_type == "browser":
        return {"game": {"game_state": "browser"}, "player": {}, "milestones": {}}

    try:
        # Use the emulator's built-in caching (100ms cache)
        # This avoids expensive operations on rapid requests
        state = env.get_comprehensive_state()

        # Ensure game state is consistent with cached dialog state
        # Use the same cached dialog state as the status endpoint
        is_dialog = env._cached_dialog_state if env else False
        if is_dialog:
            state["game"]["game_state"] = "dialog"
        else:
            # Force overworld if not in dialog (respect 5-second timeout)
            state["game"]["game_state"] = "overworld"

        # Include milestones for storyline objective auto-completion
        if env.milestone_tracker:
            state["milestones"] = env.milestone_tracker.milestones

        # Get map stitcher data for enhanced map display
        # Use the memory_reader's MapStitcher instance which has the accumulated data
        map_stitcher = None
        if env and env.memory_reader and hasattr(env.memory_reader, "_map_stitcher"):
            map_stitcher = env.memory_reader._map_stitcher
            num_areas = len(map_stitcher.map_areas) if map_stitcher and hasattr(map_stitcher, "map_areas") else 0
            logger.debug(f"Using memory_reader's MapStitcher with {num_areas} areas")
        else:
            logger.debug("No MapStitcher available from memory_reader")

        # Get current location name
        current_location = state.get("player", {}).get("location", "Unknown")
        player_pos = state.get("player", {}).get("position")
        if player_pos:
            player_coords = (player_pos.get("x", 0), player_pos.get("y", 0))
        else:
            player_coords = None

        # Add stitched map info to the map section
        if not "map" in state:
            state["map"] = {}

        # Check if we can use cached map data (location-based cache)
        current_time = time.time()
        cache_valid = (
            _state_cache["location"] == current_location and current_time - _state_cache["timestamp"] < _state_cache_ttl
        )

        # Track map source states separately:
        # - slam_map_loaded: a real SLAM map was loaded this request
        # - used_cached_map: state.map came from cache and we should skip heavy generation
        slam_map_loaded = False
        used_cached_map = False

        if cache_valid and _state_cache["map_data"]:
            # Use cached map data - skip expensive map generation!
            state["map"] = _state_cache["map_data"].copy()
            if _state_cache["portal_data"]:
                state.update(_state_cache["portal_data"])
            logger.debug(f"✅ Using cached map data for {current_location}")
            used_cached_map = True
        else:
            # Generate fresh map data
            logger.debug(f"🔄 Generating fresh map data for {current_location}")

            # PRIORITY 1: Check for agent's SLAM map first
            if current_location and current_location != "Unknown":
                try:
                    # Path imported at top of file
                    maps_dir = Path(".pokeagent_cache/maps")
                    # Normalize case to title case for consistent filenames
                    normalized_location = current_location.title()
                    safe_name = "".join(c for c in normalized_location if c.isalnum() or c in (" ", "_", "-")).strip()
                    safe_name = safe_name.replace(" ", "_")
                    slam_map_file = maps_dir / f"{safe_name}.txt"

                    logger.info(
                        f"🗺️ Checking for SLAM map: location='{current_location}' (normalized: '{normalized_location}') → file='{slam_map_file}'"
                    )

                    if slam_map_file.exists():
                        slam_map_data = slam_map_file.read_text()
                        state["map"]["visual_map"] = slam_map_data
                        state["map"]["map_source"] = "agent_slam"
                        logger.info(f"✅ Using SLAM map for {current_location} ({len(slam_map_data)} chars)")
                        slam_map_loaded = True
                    else:
                        logger.info(f"   No SLAM map found at {slam_map_file}")
                except Exception as e:
                    logger.error(f"Error loading SLAM map: {e}")

        # PRIORITY 1.5: Build final porymap map for UI parity with LLM map context.
        # This ensures the "Map & Actions" UI uses the same post-filtered map
        # (reconciliation + flag filtering + object markers) when available.
        if game_type != "red" and current_location and current_location != "Unknown" and state.get("map", {}).get("map_source") != "agent_slam":
            try:
                from utils.mapping.porymap_state import _format_porymap_info

                badges = state.get("game", {}).get("badges", [])
                if isinstance(badges, list):
                    badge_count = len(badges)
                elif isinstance(badges, int):
                    badge_count = badges
                else:
                    badge_count = 0

                porymap_result = _format_porymap_info(
                    location_name=current_location,
                    player_coords=player_coords,
                    badge_count=badge_count,
                    memory_reader=env.memory_reader if env else None,
                    runtime_object_events=state.get("map", {}).get("object_events", []),
                )

                porymap_json = porymap_result.json_map if porymap_result else None
                porymap_ascii = porymap_json.get("ascii") if isinstance(porymap_json, dict) else None
                if isinstance(porymap_ascii, str) and porymap_ascii:
                    state["map"].setdefault("porymap", {})
                    state["map"]["porymap"]["ascii"] = porymap_ascii
                    state["map"]["porymap"]["grid"] = porymap_json.get("grid", [])
                    state["map"]["porymap"]["objects"] = porymap_json.get("objects", [])
                    state["map"]["porymap"]["raw_tiles"] = porymap_json.get("raw_tiles", [])
                    state["map"]["porymap"]["dimensions"] = porymap_json.get("dimensions", {})
                    state["map"]["visual_map"] = porymap_ascii
                    state["map"]["map_source"] = "porymap_final"
            except Exception as e:
                logger.debug(f"Could not build final porymap visual map: {e}")

        # PRIORITY 2: Check if visual_map was already generated by memory_reader
        # If so, preserve it as it has the proper accumulated map data
        if not slam_map_loaded and not used_cached_map:
            visual_map_from_memory_reader = state.get("map", {}).get("visual_map")
            if visual_map_from_memory_reader:
                if state.get("map", {}).get("map_source") != "porymap_final":
                    logger.debug("Using visual_map generated by memory_reader")
                    state["map"]["map_source"] = "memory_reader"
                # Keep the visual_map as-is
            elif not ENABLE_MAP_STITCHER and current_location and current_location != "Unknown":
                # PRIORITY 3: Use porymap ground truth data when map stitcher is disabled
                try:
                    from utils.mapping.map_formatter import format_map_for_llm

                    # Get raw tiles from state
                    raw_tiles = state.get("map", {}).get("tiles")
                    player_pos_dict = state.get("player", {}).get("position", {})
                    player_facing = state.get("player", {}).get("facing", "South")

                    if raw_tiles:
                        # Generate visual map using ground truth tiles
                        visual_map = format_map_for_llm(
                            raw_tiles=raw_tiles,
                            player_facing=player_facing,
                            npcs=None,
                            player_coords=player_pos_dict,
                            location_name=current_location,
                        )

                        if visual_map and visual_map != "No map data available":
                            state["map"]["visual_map"] = visual_map
                            state["map"]["map_source"] = "porymap_ground_truth"
                            logger.debug(f"Generated visual_map from porymap ground truth for {current_location}")
                except Exception as e:
                    logger.error(f"Failed to generate visual_map from porymap: {e}")
            elif map_stitcher:
                # Generate visual map if not already present
                state["map"]["map_source"] = "map_stitcher"
                try:
                    # Get NPCs from state if available
                    npcs = state.get("map", {}).get("object_events", [])

                    # Get connections for this location
                    connections_with_coords = []
                    if current_location and current_location != "Unknown":
                        location_connections = map_stitcher.get_location_connections(current_location)
                        for conn in location_connections:
                            if len(conn) >= 3:
                                other_loc, my_coords, their_coords = conn[0], conn[1], conn[2]
                                connections_with_coords.append(
                                    {
                                        "to": other_loc,
                                        "from_pos": list(my_coords) if my_coords else [],
                                        "to_pos": list(their_coords) if their_coords else [],
                                    }
                                )

                    # Generate the map display
                    map_lines = map_stitcher.generate_location_map_display(
                        location_name=current_location,
                        player_pos=player_coords,
                        npcs=npcs,
                        connections=connections_with_coords,
                    )

                    # Store as formatted text
                    if map_lines:
                        state["map"]["visual_map"] = "\n".join(map_lines)
                        logger.debug(f"Generated visual_map with {len(map_lines)} lines")
                except Exception as e:
                    logger.error(f"Failed to generate visual_map: {e}")

        # Add stitched map info for the client/frontend
        if map_stitcher:
            # Get the location grid and connections
            if current_location and current_location != "Unknown":
                location_grid = map_stitcher.get_location_grid(current_location)
                connections = []

                # Get connections for this location
                for other_loc, my_coords, their_coords in map_stitcher.get_location_connections(current_location):
                    connections.append({"to": other_loc, "from_pos": list(my_coords), "to_pos": list(their_coords)})

                state["map"]["stitched_map_info"] = {
                    "available": True,
                    "current_area": {"name": current_location, "connections": connections, "player_pos": player_coords},
                    "player_local_pos": player_coords,
                }
            else:
                state["map"]["stitched_map_info"] = {"available": False, "reason": "Unknown location"}

            # Also include location connections directly for backward compatibility
            try:
                from utils.data_persistence.run_data_manager import get_cache_path
                cache_file = str(get_cache_path("map_stitcher_data.json"))
                if os.path.exists(cache_file):
                    with open(cache_file, "r") as f:
                        map_data = json.load(f)
                        if "location_connections" in map_data and map_data["location_connections"]:
                            location_connections = map_data["location_connections"]
                            state["location_connections"] = location_connections
                            logger.debug(
                                f"Loaded location connections for {len(location_connections) if location_connections else 0} locations"
                            )
                        elif "warp_connections" in map_data and map_data["warp_connections"]:
                            # Convert warp_connections to portal_connections format for LLM display
                            map_id_connections = {}
                            for conn in map_data["warp_connections"]:
                                from_map = conn["from_map_id"]
                                if from_map not in map_id_connections:
                                    map_id_connections[from_map] = []

                                # Find the location name for the destination map
                                to_map_name = "Unknown Location"
                                if str(conn["to_map_id"]) in map_data.get("map_areas", {}):
                                    to_map_name = map_data["map_areas"][str(conn["to_map_id"])]["location_name"]

                                map_id_connections[from_map].append(
                                    {
                                        "to_name": to_map_name,
                                        "from_pos": conn["from_position"],  # Keep as list for JSON serialization
                                        "to_pos": conn["to_position"],  # Keep as list for JSON serialization
                                    }
                                )

                            state["portal_connections"] = map_id_connections
                            print(f"🗺️ SERVER: Added portal connections to state: {map_id_connections}")
                            print(f"🗺️ SERVER: State now has keys: {list(state.keys())}")
                            logger.debug(
                                f"Loaded portal connections for {len(map_id_connections) if map_id_connections else 0} maps from persistent storage"
                            )
                else:
                    print(f"🗺️ SERVER: Cache file not found at {cache_file}")
                    logger.debug(f"Map stitcher cache file not found: {cache_file}")
            except Exception as e:
                import traceback

                print(f"🗺️ SERVER: Error loading portal connections: {e}")
                print(f"🗺️ SERVER: Full traceback: {traceback.format_exc()}")
                logger.debug(f"Could not load portal connections from persistent storage: {e}")

        # The battle information already contains all necessary data
        # No additional analysis needed - keep it clean

        # Remove MapStitcher instance to avoid serialization issues
        # The instance is only for internal use by state_formatter
        if "_map_stitcher_instance" in state.get("map", {}):
            del state["map"]["_map_stitcher_instance"]

        # Convert screenshot to base64 if available
        if state.get("visual", {}).get("screenshot"):
            buffer = io.BytesIO()
            state["visual"]["screenshot"].save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            state["visual"]["screenshot_base64"] = img_str
            # Remove the PIL image object to avoid serialization issues
            del state["visual"]["screenshot"]

        # Add porymap ground truth data with elevation filtering for frontend display
        # Skip during queued actions to avoid FPS slowdown (Emerald only — Red uses its own map reader)
        current_queue_length = len(action_queue)  # Check queue before expensive porymap operations
        if (
            game_type != "red"
            and current_location
            and current_location != "Unknown"
            and current_location != "TITLE_SEQUENCE"
            and current_queue_length == 0
        ):
            try:
                from utils.state_formatter import _format_porymap_info

                _mr = getattr(env, "memory_reader", None) if env else None
                porymap_result = _format_porymap_info(
                    current_location,
                    player_coords,
                    memory_reader=_mr,
                    runtime_object_events=state.get("map", {}).get("object_events", []),
                )
                porymap_data = getattr(porymap_result, "json_map", None)
                if porymap_data and porymap_data.get("grid"):
                    if "porymap" not in state["map"]:
                        state["map"]["porymap"] = {}
                    state["map"]["porymap"]["grid"] = porymap_data.get("grid")
                    state["map"]["porymap"]["objects"] = porymap_data.get("objects", [])
                    state["map"]["porymap"]["dimensions"] = porymap_data.get("dimensions", {})
                    state["map"]["porymap"]["warps"] = porymap_data.get("warps", [])
                    logger.debug(f"Added elevation-filtered porymap data to /state for {current_location}")
            except Exception as e:
                logger.warning(f"Failed to add porymap data to /state: {e}")

        # Cheap: 15x15 stream UI from cached or freshly refreshed grid (also while actions are queued)
        if game_type != "red" and current_location and current_location not in ("Unknown", "TITLE_SEQUENCE") and player_coords:
            _build_porymap_visual_map_15x15(state, player_coords)
        elif game_type == "red":
            # stream UI map for Red has been built in red_map_reader.py
            visual_map = env.memory_reader.map_reader.format_map_for_llm(radius=7)
            state["map"]["visual_map"] = visual_map
            state["map"]["map_source"] = "red_map_reader"
            whole_map = env.memory_reader.map_reader.get_whole_map_data()
            state["map"]["red_whole_map"] = whole_map

        with step_lock:
            current_step = step_count

        # Include action queue info for multiprocess coordination
        queue_length = len(action_queue)  # Action queue access is atomic for len()

        # Cache map data for this location+position - reduces load on rapid repeated requests
        _state_cache["location"] = current_location
        _state_cache["player_coords"] = player_coords
        _state_cache["map_data"] = state.get("map", {}).copy()
        _state_cache["portal_data"] = {"location_connections": state.get("location_connections", {})}
        _state_cache["timestamp"] = time.time()

        return ComprehensiveStateResponse(
            visual=state["visual"],
            player=state["player"],
            game=state["game"],
            map=state["map"],
            milestones=state.get("milestones", {}),
            location_connections=state.get("location_connections", {}),
            step_number=current_step,
            status="running",
            action_queue_length=queue_length,
        )

    except Exception as e:
        logger.error(f"Error getting comprehensive state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/whole_map")
async def get_whole_map():
    """Get complete map data including full grid, raw tiles, and elevation info for debugging"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    # Red: use RedEmulator.get_whole_map() directly (returns compatible dict)
    if game_type == "red":
        try:
            result = env.get_whole_map()
            if not result or not result.get("grid"):
                raise HTTPException(status_code=400, detail="No valid location loaded")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting whole map (Red): {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Emerald: use porymap ground truth
    try:
        # Get current location
        state = env.get_comprehensive_state()
        location_name = state.get("player", {}).get("location", "Unknown")
        player_pos = state.get("player", {}).get("position", {})
        px, py = player_pos.get("x", 0), player_pos.get("y", 0)

        if not location_name or location_name in ["Unknown", "TITLE_SEQUENCE"]:
            raise HTTPException(status_code=400, detail="No valid location loaded")

        # Load porymap data with raw tiles
        from utils.mapping.porymap_json_builder import build_json_map_for_llm
        from utils.mapping.pokeemerald_parser import PokeemeraldMapLoader
        from utils.state_formatter import ROM_TO_PORYMAP_MAP
        from pathlib import Path

        # Get pokeemerald root (pokemon_env/porymap or POKEEMERALD_ROOT override)
        from pokemon_env.porymap_paths import get_porymap_root
        pokeemerald_root = get_porymap_root()

        if not pokeemerald_root:
            raise HTTPException(status_code=500, detail="Could not find pokeemerald root")

        # Get porymap name
        porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)

        if not porymap_map_name:
            raise HTTPException(status_code=404, detail=f"Could not find porymap for location: {location_name}")

        # Build complete map
        json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root)

        if not json_map:
            raise HTTPException(status_code=500, detail=f"Failed to build map for {porymap_map_name}")

        # Get player elevation
        raw_tiles = json_map.get("raw_tiles", [])
        player_elevation = 0
        if raw_tiles and 0 <= py < len(raw_tiles) and 0 <= px < len(raw_tiles[py]):
            player_tile = raw_tiles[py][px]
            if len(player_tile) >= 4:
                player_elevation = player_tile[3]

        # Build elevation map
        elevation_map = []
        behavior_map = []
        if raw_tiles:
            from pokemon_env.enums import MetatileBehavior

            for row in raw_tiles:
                elev_row = []
                behav_row = []
                for tile in row:
                    if len(tile) >= 4:
                        elev_row.append(tile[3])  # elevation
                        behavior_id = tile[1]
                        try:
                            behavior_name = MetatileBehavior(behavior_id).name
                        except:
                            behavior_name = f"UNKNOWN_{behavior_id}"
                        behav_row.append(behavior_name)
                    else:
                        elev_row.append(0)
                        behav_row.append("UNKNOWN")
                elevation_map.append(elev_row)
                behavior_map.append(behav_row)

        # Count special behaviors
        from pokemon_env.enums import MetatileBehavior

        special_tiles = {}
        for y, row in enumerate(raw_tiles):
            for x, tile in enumerate(row):
                if len(tile) >= 4:
                    behavior_id = tile[1]
                    elevation = tile[3]
                    try:
                        behavior_name = MetatileBehavior(behavior_id).name
                        if any(keyword in behavior_name for keyword in ["LADDER", "STAIRS", "DOOR", "WARP"]):
                            if behavior_name not in special_tiles:
                                special_tiles[behavior_name] = []
                            special_tiles[behavior_name].append(
                                {"x": x, "y": y, "elevation": elevation, "behavior_id": behavior_id}
                            )
                    except:
                        pass

        return {
            "location": location_name,
            "porymap_name": porymap_map_name,
            "player_position": {"x": px, "y": py},
            "player_elevation": player_elevation,
            "dimensions": json_map.get("dimensions", {}),
            "grid": json_map.get("grid", []),
            "raw_tiles": raw_tiles,
            "elevation_map": elevation_map,
            "behavior_map": behavior_map,
            "special_tiles": special_tiles,
            "warps": json_map.get("warps", []),
            "objects": json_map.get("objects", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting whole map: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/memory")
async def debug_memory():
    """Debug memory reading (basic version)"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        if not env.memory_reader:
            return {"error": "Memory reader not initialized"}

        # Test basic memory access
        diagnostics = env.memory_reader.test_memory_access()

        # Try to read some basic data
        try:
            party_size = env.memory_reader.read_party_size()
            coordinates = env.memory_reader.read_coordinates()
            money = env.memory_reader.read_money()

            # Add new debugging info
            is_in_battle = env.memory_reader.is_in_battle()
            game_state = env.memory_reader.get_game_state()
            player_name = env.memory_reader.read_player_name()

            # Add battle detection debugging
            try:
                battle_addr = env.memory_reader.IN_BATTLE_BIT_ADDR
                battle_raw_value = env.memory_reader._read_u8(battle_addr)
                battle_mask = env.memory_reader.IN_BATTLE_BITMASK
                battle_result = (battle_raw_value & battle_mask) != 0
            except Exception as e:
                battle_raw_value = None
                battle_mask = None
                battle_result = None

            diagnostics.update(
                {
                    "party_size": party_size,
                    "coordinates": coordinates,
                    "money": money,
                    "is_in_battle": is_in_battle,
                    "game_state": game_state,
                    "player_name": player_name,
                    "battle_detection": {
                        "address": f"0x{battle_addr:08x}" if "battle_addr" in locals() else "unknown",
                        "raw_value": f"0x{battle_raw_value:02x}" if battle_raw_value is not None else "error",
                        "mask": f"0x{battle_mask:02x}" if battle_mask is not None else "unknown",
                        "result": battle_result,
                    },
                    "working_reads": True,
                }
            )
        except Exception as read_error:
            diagnostics["read_error"] = str(read_error)
            diagnostics["working_reads"] = False

        return diagnostics

    except Exception as e:
        logger.error(f"Error debugging memory: {e}")
        return {"error": str(e)}


@app.get("/debug/memory/comprehensive")
async def debug_memory_comprehensive():
    """Comprehensive memory reading test with detailed diagnostics"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Use the comprehensive memory testing method
        test_results = env.test_memory_reading()
        return test_results

    except Exception as e:
        logger.error(f"Error running comprehensive memory test: {e}")
        return {"error": str(e)}


@app.get("/debug/memory/dump")
async def debug_memory_dump(start: int = 0x02000000, length: int = 0x1000):
    """Dump raw memory from the emulator"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        if not env.memory_reader:
            return {"error": "Memory reader not initialized"}

        # Read raw memory bytes
        memory_bytes = env.memory_reader._read_bytes(start, length)

        # Convert to hex string for easy viewing
        hex_data = memory_bytes.hex()

        # Also try to decode as text using Pokemon Emerald character mapping
        try:
            from pokemon_env.emerald_utils import EmeraldCharmap

            charmap = EmeraldCharmap()
            decoded_text = charmap.decode(memory_bytes)
        except:
            decoded_text = "Could not decode as text"

        return {
            "start_address": f"0x{start:08X}",
            "length": length,
            "hex_data": hex_data,
            "decoded_text": decoded_text,
            "raw_bytes": [b for b in memory_bytes[:100]],  # First 100 bytes as numbers
        }

    except Exception as e:
        logger.error(f"Error dumping memory: {e}")
        return {"error": str(e)}


@app.get("/test_stream")
async def test_stream():
    """Simple test stream to verify SSE works"""
    from fastapi.responses import StreamingResponse
    import asyncio

    async def simple_stream():
        for i in range(5):
            yield f"data: {{'test': {i}, 'timestamp': {time.time()}}}\n\n"
            await asyncio.sleep(1)
        yield f"data: {{'done': true}}\n\n"

    return StreamingResponse(simple_stream(), media_type="text/event-stream")


@app.get("/agent_stream")
async def stream_agent_thinking():
    """Stream agent thinking in real-time using Server-Sent Events"""
    from fastapi.responses import StreamingResponse
    from utils.data_persistence.llm_logger import get_llm_logger
    import asyncio

    async def event_stream():
        """Generate server-sent events for agent thinking"""
        logger.info("SSE: Starting event stream")
        last_timestamp = ""  # Track last seen timestamp instead of count
        sent_timestamps = set()  # Track all sent timestamps to avoid duplicates
        heartbeat_counter = 0
        llm_logger = get_llm_logger()

        try:
            # Send initial connection message
            yield f"data: {json.dumps({'status': 'connected', 'timestamp': time.time()})}\n\n"

            # On startup, mark all existing interactions as "sent" to avoid flooding with old messages
            # We only want to stream NEW interactions from this point forward
            # Use current session's log file only (not glob of all files - avoids cross-execution bleed)
            try:
                log_file = llm_logger.log_file
                if log_file and os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("type") == "interaction":
                                    timestamp = entry.get("timestamp", "")
                                    if timestamp:
                                        sent_timestamps.add(timestamp)
                            except Exception:
                                continue
                logger.info(f"SSE: Marked {len(sent_timestamps)} existing interactions as already sent")
            except Exception as init_e:
                logger.warning(f"SSE: Error initializing sent timestamps: {init_e}")

            while True:
                try:
                    heartbeat_counter += 1

                    # Use simple file reading instead of complex get_agent_thinking()
                    current_step = 0

                    with step_lock:
                        current_step = agent_step_count

                    new_interactions = []
                    try:
                        # Use current session's log file only (not glob - avoids cross-execution bleed)
                        log_file = llm_logger.log_file
                        if log_file and os.path.exists(log_file):
                            with open(log_file, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                                # Only check last 10 lines for new entries (performance optimization)
                                for line in lines[-10:]:
                                    try:
                                        entry = json.loads(line.strip())
                                        if entry.get("type") == "interaction":
                                            timestamp = entry.get("timestamp", "")
                                            # Only add if we haven't sent this timestamp before
                                            if timestamp and timestamp not in sent_timestamps:
                                                model_name = (
                                                    entry.get("model_info", {}) or {}
                                                ).get("model", "")
                                                new_interactions.append(
                                                    {
                                                        "type": _format_interaction_type_for_ui(
                                                            entry.get("interaction_type", "unknown")
                                                        ),
                                                        "model": model_name,
                                                        "provider_family": _provider_family_for_llm_log_entry(
                                                            entry
                                                        ),
                                                        "response": entry.get("response", ""),
                                                        "duration": entry.get("duration", 0),
                                                        "timestamp": timestamp,
                                                        "agent_step": entry.get("agent_step"),
                                                    }
                                                )
                                    except Exception:
                                        continue
                    except Exception as file_e:
                        logger.warning(f"SSE: File reading error: {file_e}")

                    # Sort by timestamp to ensure chronological order
                    new_interactions.sort(key=lambda x: x.get("timestamp", ""))

                    # Check if there are new interactions
                    if new_interactions:
                        logger.info(f"SSE: Found {len(new_interactions)} new interactions to send")
                        # Send new interactions
                        for interaction in new_interactions:
                            # Prefer per-line global step (looping subagents); else server counter
                            line_step = interaction.get("agent_step")
                            if line_step is None:
                                line_step = current_step
                            event_data = {
                                "step": line_step,
                                "type": interaction.get("type", "unknown"),
                                "model": interaction.get("model", ""),
                                "provider_family": interaction.get(
                                    "provider_family", "other"
                                ),
                                "response": interaction.get("response", ""),
                                "duration": interaction.get("duration", 0),
                                "timestamp": interaction.get("timestamp", ""),
                                "is_new": True,
                            }

                            yield f"data: {json.dumps(event_data)}\n\n"
                            # Mark this timestamp as sent
                            sent_timestamps.add(interaction.get("timestamp", ""))

                    # Send periodic heartbeat to keep connection alive (every 10 cycles = 5 seconds)
                    if not new_interactions and heartbeat_counter % 10 == 0:
                        yield f"data: {json.dumps({'heartbeat': True, 'timestamp': time.time(), 'step': current_step})}\n\n"

                    # Wait before checking again (increased from 500ms to 2s for better performance)
                    await asyncio.sleep(2.0)

                except Exception as e:
                    logger.error(f"SSE: Error in stream loop: {e}")
                    yield f"data: {json.dumps({'error': str(e), 'timestamp': time.time()})}\n\n"
                    await asyncio.sleep(2)

        except Exception as outer_e:
            logger.error(f"SSE: Fatal error in event stream: {outer_e}")
            yield f"data: {json.dumps({'fatal_error': str(outer_e), 'timestamp': time.time()})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/agent")
async def get_agent_thinking():
    """Get current agent thinking status and recent LLM interactions"""
    try:
        # Get the most recent LLM log file
        from utils.data_persistence.llm_logger import get_llm_logger

        # Get recent LLM interactions
        llm_logger = get_llm_logger()
        session_summary = llm_logger.get_session_summary()

        # Use current session's log file only (not glob - avoids cross-execution bleed)
        recent_interactions = []
        log_file = llm_logger.log_file
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("type") == "interaction":
                                model_name = (
                                    entry.get("model_info", {}) or {}
                                ).get("model", "")
                                recent_interactions.append(
                                    {
                                        "type": _format_interaction_type_for_ui(
                                            entry.get("interaction_type", "unknown")
                                        ),
                                        "model": model_name,
                                        "provider_family": _provider_family_for_llm_log_entry(
                                            entry
                                        ),
                                        "prompt": entry.get("prompt", ""),
                                        "response": entry.get("response", ""),
                                        "duration": entry.get("duration", 0),
                                        "timestamp": entry.get("timestamp", ""),
                                        "agent_step": entry.get("agent_step"),
                                    }
                                )
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading LLM log {log_file}: {e}")

        # Sort by timestamp and keep only the most recent interaction (current step)
        recent_interactions.sort(key=lambda x: x.get("timestamp", ""))
        recent_interactions = recent_interactions[-1:] if recent_interactions else []
        logger.info(f"Found {len(recent_interactions)} recent interactions (showing current step only)")

        # Format the agent thinking display
        if recent_interactions:
            interaction = recent_interactions[-1]  # Get the most recent interaction
            current_thought = f"Current step LLM output:\n"
            duration = interaction.get("duration", 0)
            current_thought += f"{interaction['type'].upper()} ({duration:.2f}s)\n"
            current_thought += f"Response: {interaction['response']}"
        else:
            current_thought = "No recent LLM interactions. Agent is ready to process game state."

        with step_lock:
            current_step = agent_step_count  # Use agent step count instead of frame step count

        return {
            "status": "thinking",
            "current_thought": current_thought,
            "confidence": 0.85,
            "timestamp": time.time(),
            "llm_session": session_summary,
            "recent_interactions": recent_interactions,
            "current_step": current_step,
        }

    except Exception as e:
        logger.error(f"Error in agent thinking: {e}")
        return {
            "status": "error",
            "current_thought": f"Error getting agent thinking: {str(e)}",
            "confidence": 0.0,
            "timestamp": time.time(),
        }


@app.get("/metrics")
async def get_metrics():
    """Get cumulative metrics for the run"""
    global latest_metrics

    try:
        # Return the latest metrics received from client (with thread safety)
        with step_lock:
            metrics = latest_metrics.copy()
            metrics["agent_step_count"] = agent_step_count

        # If metrics haven't been initialized by client yet, try to load from cumulative_metrics.json
        # BUT only if checkpoint loading is enabled (not for fresh starts with --load-state)
        if metrics.get("total_llm_calls", 0) == 0 and checkpoint_loading_enabled:
            from utils.data_persistence.llm_logger import get_llm_logger
            llm_logger = get_llm_logger()
            if llm_logger and llm_logger.load_cumulative_metrics():
                metrics.update(llm_logger.cumulative_metrics)
            # agent_step_count comes from checkpoint_llm.txt (not cumulative_metrics.json)
            from utils.data_persistence.run_data_manager import get_checkpoint_llm_path
            checkpoint_file = get_checkpoint_llm_path()
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        checkpoint_data = json.load(f)
                        if "agent_step_count" in checkpoint_data:
                            metrics["agent_step_count"] = checkpoint_data["agent_step_count"]
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug(f"Could not read agent_step_count from checkpoint: {e}")

        return metrics

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "total_actions": 0,
            "total_run_time": 0,
            "total_llm_calls": 0,
            "agent_step_count": agent_step_count,
        }


# Store latest metrics from client
latest_metrics = {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_cost": 0.0,
    "total_actions": 0,
    "total_run_time": 0,
    "total_llm_calls": 0,
    "start_time": time.time(),  # Will be overwritten if checkpoint is loaded
}


@app.get("/termination_condition")
async def get_termination_condition(condition_type: str = "gym_badge_count", threshold: int = 1):
    """Check if a termination condition is met based on ground-truth memory data.
    
    This endpoint reads game state directly from ROM memory to provide reliable
    termination conditions for external CLI agents (Claude Code, Codex, etc.).
    
    Args:
        condition_type: Type of condition to check. Supported types:
            - "gym_badge_count": Check number of gym badges obtained
        threshold: Threshold value for the condition (e.g., 1 for first badge)
    
    Returns:
        JSON with condition status:
        {
            "condition_type": str,
            "threshold": int,
            "current_value": int,
            "condition_met": bool,
            "badge_names": list (for gym_badge_count)
        }
    """
    global env
    
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    if not env.memory_reader:
        raise HTTPException(status_code=500, detail="Memory reader not initialized")
    
    try:
        if condition_type == "gym_badge_count":
            # Read badges directly from ROM memory (ground truth)
            badges = env.memory_reader.read_badges()
            badge_count = len(badges) if badges else 0
            
            return {
                "condition_type": condition_type,
                "threshold": threshold,
                "current_value": badge_count,
                "badge_names": badges,
                "condition_met": badge_count >= threshold
            }
        
        # Future condition types can be added here:
        # elif condition_type == "pokemon_count":
        #     party_size = env.memory_reader.read_party_size()
        #     return {...}
        
        else:
            return {
                "error": f"Unknown condition type: {condition_type}",
                "supported_types": ["gym_badge_count"]
            }
    
    except Exception as e:
        logger.error(f"Error checking termination condition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Flag to track whether checkpoint loading should be enabled
checkpoint_loading_enabled = True  # Will be set based on startup args


@app.post("/reset_metrics")
async def reset_metrics():
    """Reset all metrics to zero for fresh start"""
    global latest_metrics, agent_step_count, checkpoint_loading_enabled

    with step_lock:
        latest_metrics.update(
            {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
                "total_actions": 0,
                "total_run_time": 0,
                "total_llm_calls": 0,
                "start_time": time.time(),
            }
        )
        agent_step_count = 0
        # Disable checkpoint loading to prevent loading from checkpoint_llm.txt
        checkpoint_loading_enabled = False

    print("🔄 Server metrics reset for fresh start - checkpoint loading disabled")
    return {"status": "reset", "timestamp": time.time()}


@app.post("/agent_step")
async def update_agent_step(request: Request = None):
    """Update the agent step count and metrics"""
    global agent_step_count, latest_metrics

    skip_default_increment = False

    try:
        # Check if this is a direct set operation or has metrics
        if request:
            try:
                request_data = await request.json()

                # Store agent thinking if provided (same path as VLM: log to LLM logger so SSE has one source)
                if "thinking" in request_data:
                    thinking_text = request_data["thinking"]
                    interaction_type = request_data.get("interaction_type", "thinking")
                    duration = float(request_data.get("duration", 0))
                    thinking_step = request_data.get("agent_step")
                    if thinking_step is None:
                        thinking_step = request_data.get("step")
                    thinking_step_int = None
                    if thinking_step is not None:
                        try:
                            thinking_step_int = int(thinking_step)
                        except (TypeError, ValueError):
                            pass
                    try:
                        from utils.data_persistence.llm_logger import get_llm_logger

                        get_llm_logger().log_thinking(
                            thinking_text,
                            interaction_type,
                            duration,
                            agent_step=thinking_step_int,
                        )
                    except Exception as e:
                        logger.debug(f"Could not log thinking: {e}")

                # Update metrics if provided (with thread safety)
                if "metrics" in request_data and isinstance(request_data["metrics"], dict):
                    metrics_payload = request_data["metrics"]
                    with step_lock:  # Use existing lock for thread safety
                        # Safely update each metric individually to avoid race conditions
                        for key, value in metrics_payload.items():
                            if key in latest_metrics:
                                # Always protect total_actions as it's managed by server
                                if key == "total_actions":
                                    continue
                                else:
                                    latest_metrics[key] = value

                    # Full cumulative sync from client: align server step with max global LLM step
                    if metrics_payload.get("total_llm_calls") is not None:
                        steps = metrics_payload.get("steps") or []
                        max_s = 0
                        for s in steps:
                            try:
                                max_s = max(max_s, int(s.get("step") or 0))
                            except (TypeError, ValueError):
                                continue
                        with step_lock:
                            agent_step_count = max(agent_step_count, max_s)
                        # Only skip legacy +1 when we have step rows; empty steps keeps old increment behavior
                        if steps:
                            skip_default_increment = True

                # Handle set_step for initialization
                if "set_step" in request_data:
                    with step_lock:
                        agent_step_count = request_data["set_step"]
                    return {"status": "set", "agent_step": agent_step_count}
            except Exception as e:
                logger.error(f"Error processing agent_step request: {e}")
                # Continue with default increment behavior
    except Exception as e:
        logger.error(f"Error in agent_step endpoint: {e}")
        # Continue with default increment behavior

    if skip_default_increment:
        return {"status": "updated", "agent_step": agent_step_count}

    # Default increment behavior
    with step_lock:
        agent_step_count += 1

        # Save end-state snapshot every 20 steps
        if agent_step_count % 20 == 0:
            try:
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                if run_manager:
                    run_manager.save_end_state_snapshot()
                    logger.info(f"💾 Saved end-state snapshot at step {agent_step_count}")
            except Exception as e:
                logger.debug(f"Could not save end-state snapshot: {e}")

    return {"status": "updated", "agent_step": agent_step_count}


@app.get("/llm_logs")
async def get_llm_logs():
    """Get recent LLM log entries"""
    try:
        from utils.data_persistence.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        session_summary = llm_logger.get_session_summary()

        # Get recent log entries
        recent_entries = []
        if os.path.exists(llm_logger.log_file):
            try:
                with open(llm_logger.log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Get the last 20 entries
                    for line in lines[-20:]:
                        try:
                            entry = json.loads(line.strip())
                            recent_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading LLM log: {e}")

        return {"session_summary": session_summary, "recent_entries": recent_entries, "log_file": llm_logger.log_file}

    except Exception as e:
        logger.error(f"Error getting LLM logs: {e}")
        return {"error": str(e)}


# Helper function to update objectives cache file (for fast reads by stream.html)
def _update_objectives_cache():
    """Write current objectives state to fast-access cache file"""
    try:
        global direct_objectives_manager
        objectives_data = {
            "mode": "legacy",
            "current": None,
            "recently_completed": [],
            "total_in_sequence": 0,
            "current_index": 0,
        }

        if direct_objectives_manager and direct_objectives_manager.is_sequence_active():
            logger.info(f"📊 Updating objectives cache - mode: {direct_objectives_manager.mode}")
            if direct_objectives_manager.mode == "categorized":
                # Categorized cache: only mode, per-category windows, and status counts.
                # Omit legacy root keys (current, recently_completed, total_in_sequence,
                # current_index); consumers use story/battling/dynamics + categorized_status.

                def get_recent_for_category(category, sequence, index):
                    items = []
                    # Current objective (if exists)
                    if index < len(sequence):
                        current = sequence[index]
                        items.append(
                            {"id": current.id, "description": current.description, "completed": False, "current": True}
                        )

                    # Recently completed (last 5 completed before current, in reverse order - most recent first)
                    for i in range(index - 1, max(-1, index - 6), -1):
                        if i >= 0 and i < len(sequence) and sequence[i].completed:
                            items.append(
                                {
                                    "id": sequence[i].id,
                                    "description": sequence[i].description,
                                    "completed": True,
                                    "current": False,
                                }
                            )

                    return items

                objectives_data = {
                    "mode": "categorized",
                    "story": get_recent_for_category(
                        "story", direct_objectives_manager.story_sequence, direct_objectives_manager.story_index
                    ),
                    "battling": get_recent_for_category(
                        "battling", direct_objectives_manager.battling_sequence, direct_objectives_manager.battling_index
                    ),
                    "dynamics": get_recent_for_category(
                        "dynamics", direct_objectives_manager.dynamics_sequence, direct_objectives_manager.dynamics_index
                    ),
                    "categorized_status": {
                        "story": {
                            "current_index": direct_objectives_manager.story_index,
                            "total": len(direct_objectives_manager.story_sequence),
                        },
                        "battling": {
                            "current_index": direct_objectives_manager.battling_index,
                            "total": len(direct_objectives_manager.battling_sequence),
                        },
                        "dynamics": {
                            "current_index": direct_objectives_manager.dynamics_index,
                            "total": len(direct_objectives_manager.dynamics_sequence),
                        },
                    },
                }

            else:
                # LEGACY: Single objective mode
                current_obj = direct_objectives_manager.get_current_objective()

                # Get recently completed objectives (last 5)
                completed_objectives = []
                if direct_objectives_manager.current_sequence:
                    for i in range(
                        max(0, direct_objectives_manager.current_index - 5), direct_objectives_manager.current_index
                    ):
                        if i < len(direct_objectives_manager.current_sequence):
                            obj = direct_objectives_manager.current_sequence[i]
                            completed_objectives.append(
                                {"id": obj.id, "description": obj.description, "completed": True, "index": i}
                            )

                objectives_data = {
                    "mode": "legacy",
                    "current": {
                        "id": current_obj.id,
                        "description": current_obj.description,
                        "index": direct_objectives_manager.current_index,
                    }
                    if current_obj
                    else None,
                    "recently_completed": completed_objectives,
                    "total_in_sequence": len(direct_objectives_manager.current_sequence),
                    "current_index": direct_objectives_manager.current_index,
                }

        # Write to cache file
        from utils.data_persistence.run_data_manager import get_cache_path
        cache_file = get_cache_path("current_objective.json")
        with open(cache_file, 'w') as f:
            json.dump(objectives_data, f, indent=2)

        logger.info(
            f"✅ Updated objectives cache: mode={objectives_data.get('mode')}, story={len(objectives_data.get('story', []))}, battling={len(objectives_data.get('battling', []))}, dynamics={len(objectives_data.get('dynamics', []))}"
        )

    except Exception as e:
        logger.error(f"Failed to update objectives cache: {e}")
        import traceback

        logger.error(traceback.format_exc())


# Milestone checking is now handled by the emulator


@app.get("/milestones")
async def get_milestones():
    """Get current milestones achieved based on persistent tracking"""
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")

    if game_type == "browser":
        return {"milestones": {}, "agent_step_count": agent_step_count}

    try:
        # Get milestones directly from emulator
        result = env.get_milestones(agent_step_count=agent_step_count)

        # Read objectives from cached file (fast, no manager access needed)
        objectives_data = {
            "current": None,
            "recently_completed": [],
            "total_in_sequence": 0,
            "current_index": 0
        }

        try:
            from utils.data_persistence.run_data_manager import get_cache_path
            objectives_cache_file = get_cache_path("current_objective.json")
            if objectives_cache_file.exists():
                with open(objectives_cache_file, 'r') as f:
                    objectives_data = json.load(f)
        except Exception as e:
            logger.debug(f"Could not read objectives cache: {e}")

        result["objectives"] = objectives_data
        return result

    except Exception as e:
        logger.error(f"Error getting milestones: {e}")
        # Fallback to basic milestones if memory reading fails
        basic_milestones = [
            {"id": 1, "name": "GAME_STARTED", "category": "basic", "completed": True, "timestamp": time.time()},
            {"id": 2, "name": "EMULATOR_RUNNING", "category": "basic", "completed": True, "timestamp": time.time()},
        ]
        return {
            "milestones": basic_milestones,
            "completed": 2,
            "total": 2,
            "progress": 1.0,
            "tracking_system": "fallback",
            "objectives": {"current": None, "recently_completed": [], "total_in_sequence": 0, "current_index": 0},
            "error": str(e),
        }


# Global list to track recent button presses
recent_button_presses = []


@app.get("/recent_actions")
async def get_recent_actions():
    """Get recently pressed buttons"""
    global recent_button_presses
    return {
        "recent_buttons": recent_button_presses[-10:],  # Last 10 button presses
        "timestamp": time.time(),
    }


@app.get("/debug/milestones")
async def debug_milestones():
    """Debug milestone tracking system"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Get current working directory and list milestone files
        import glob

        current_dir = os.getcwd()
        milestone_files = glob.glob("*milestones*.json")

        # Check if default milestone file exists and get its info
        default_file_info = None
        if os.path.exists(env.milestone_tracker.filename):
            try:
                with open(env.milestone_tracker.filename, "r") as f:
                    default_data = json.load(f)
                default_file_info = {
                    "exists": True,
                    "size": os.path.getsize(env.milestone_tracker.filename),
                    "last_modified": time.ctime(os.path.getmtime(env.milestone_tracker.filename)),
                    "milestone_count": len(default_data.get("milestones", {})),
                    "last_updated": default_data.get("last_updated", "unknown"),
                }
            except Exception as e:
                default_file_info = {"exists": True, "error": str(e)}
        else:
            default_file_info = {"exists": False}

        return {
            "tracking_system": "file_based",
            "current_filename": env.milestone_tracker.filename,
            "current_milestones": len(env.milestone_tracker.milestones),
            "completed_milestones": sum(
                1 for m in env.milestone_tracker.milestones.values() if m.get("completed", False)
            ),
            "default_file_info": default_file_info,
            "milestone_files_in_directory": milestone_files,
            "working_directory": current_dir,
            "milestone_details": env.milestone_tracker.milestones,
        }
    except Exception as e:
        logger.error(f"Error in milestone debug: {e}")
        return {"error": str(e)}


@app.post("/debug/reset_milestones")
async def reset_milestones():
    """Reset all milestones (for testing)"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        env.milestone_tracker.reset_all()
        return {
            "status": "reset",
            "milestone_file": env.milestone_tracker.filename,
            "remaining_milestones": len(env.milestone_tracker.milestones),
        }
    except Exception as e:
        logger.error(f"Error resetting milestones: {e}")
        return {"error": str(e)}


@app.post("/debug/test_milestone_operations")
async def test_milestone_operations():
    """Test milestone loading and saving operations"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")

    try:
        # Test data
        test_milestones = {
            "TEST_MILESTONE_1": {"completed": True, "timestamp": time.time(), "first_completed": time.time()},
            "TEST_MILESTONE_2": {"completed": False, "timestamp": None},
        }

        # Save current state
        original_milestones = env.milestone_tracker.milestones.copy()
        original_filename = env.milestone_tracker.filename

        # Test 1: Save milestones with state filename
        test_state_filename = "test_state_123.sav"
        env.milestone_tracker.milestones = test_milestones.copy()
        saved_filename = env.milestone_tracker.save_milestones_for_state(test_state_filename)

        # Test 2: Load milestones for state
        env.milestone_tracker.milestones = {}  # Clear current milestones
        env.milestone_tracker.load_milestones_for_state(test_state_filename)
        loaded_milestones = env.milestone_tracker.milestones.copy()

        # Test 3: Check if file was created
        file_exists = os.path.exists(saved_filename)
        file_size = os.path.getsize(saved_filename) if file_exists else 0

        # Restore original state
        env.milestone_tracker.milestones = original_milestones
        env.milestone_tracker.filename = original_filename

        return {
            "test_results": {
                "save_operation": {
                    "filename": saved_filename,
                    "file_exists": file_exists,
                    "file_size": file_size,
                    "milestones_saved": len(test_milestones),
                },
                "load_operation": {
                    "milestones_loaded": len(loaded_milestones),
                    "milestones_match": loaded_milestones == test_milestones,
                    "loaded_milestones": loaded_milestones,
                },
            },
            "original_state_restored": True,
        }

    except Exception as e:
        logger.error(f"Error testing milestone operations: {e}")
        return {"error": str(e)}


# ============================================================================
# MCP TOOL ENDPOINTS
# ============================================================================


@app.post("/mcp/get_game_state")
async def mcp_get_game_state():
    """MCP Tool: Get current game state"""
    if env is None:
        return {"success": False, "error": "Environment not initialized"}

    # Browser games: return screenshot + page text + game info (no emulator state)
    if game_type == "browser":
        try:
            screenshot = env.get_screenshot()
            screenshot_b64 = _pil_to_base64(screenshot)
            game_info = env.get_game_info()
            page_text = env.get_page_text()

            # Push frame to stream
            update_frame_cache(screenshot)

            # Build a state_text similar to emulator games
            cw = game_info.get('canvas_width', 0)
            ch = game_info.get('canvas_height', 0)
            mx = game_info.get('mouse_x')
            my = game_info.get('mouse_y')
            state_text = f"=== BROWSER GAME STATE ===\n"
            state_text += f"URL: {game_info.get('url', '')}\n"
            state_text += f"Title: {game_info.get('title', '')}\n"
            state_text += f"Canvas: {cw}x{ch} (valid click range: x=0-{cw}, y=0-{ch})\n"
            if mx is not None and my is not None:
                state_text += (
                    f"Cursor position: ({mx}, {my}) — last known canvas-relative "
                    f"coordinates of the mouse cursor. Hover-driven UI elements "
                    f"(tooltips, paddles, mouse-look) react to this position.\n"
                )
            else:
                state_text += "Cursor position: unknown (no mouse_move/click yet this session)\n"
            if browser_last_action:
                la = browser_last_action
                if la["type"] == "mouse_click":
                    state_text += f"Last click: ({la['x']}, {la['y']})\n"
                elif la["type"] == "double_click":
                    state_text += f"Last double-click: ({la['x']}, {la['y']})\n"
                elif la["type"] == "mouse_move":
                    state_text += f"Last mouse move: ({la['x']}, {la['y']})\n"
                elif la["type"] == "mouse_drag":
                    state_text += (
                        f"Last drag: ({la['x1']}, {la['y1']}) -> "
                        f"({la['x2']}, {la['y2']})\n"
                    )
                elif la["type"] == "press_keys":
                    state_text += f"Last keys: {la['keys']}\n"
                elif la["type"] == "hold_key":
                    state_text += f"Last hold: {la['key']} for {la['duration_ms']}ms\n"
            if page_text:
                state_text += f"\n--- Visible Text ---\n{page_text[:2000]}\n"

            return {
                "success": True,
                "state_text": state_text,
                "screenshot_base64": screenshot_b64,
                "game_info": game_info,
                "page_text": page_text[:2000] if page_text else "",
            }
        except Exception as e:
            logger.error(f"Browser get_game_state error: {e}")
            return {"success": False, "error": str(e)}

    try:
        from utils.state_formatter import format_state_for_llm
        from server import game_tools
        from agents.objectives import DirectObjectiveManager

        # Get recent button presses with position history
        global recent_button_presses, current_obs

        # Get latest frame from game loop to ensure sync between memory and visuals
        with obs_lock:
            obs_copy = current_obs.copy() if current_obs is not None else None

        # Use helper function from game_tools with action history and current frame
        result = game_tools.get_game_state_direct(
            env,
            format_state_for_llm,
            action_history=recent_button_presses,
            current_obs=obs_copy,  # Pass latest frame from game loop
        )

        # Add direct objectives information
        if result.get("success", False):
            global direct_objectives_manager, current_run_dir

            # Initialize direct objective manager if needed
            if direct_objectives_manager is None:
                direct_objectives_manager = DirectObjectiveManager()

            # Load direct objectives sequence if specified
            if direct_objectives_sequence:
                # Check if we need to load objectives
                needs_loading = not direct_objectives_manager.is_sequence_active()
                logger.info(f"🔍 Checking if objectives need loading:")
                logger.info(f"   - is_active: {direct_objectives_manager.is_sequence_active()}")
                logger.info(f"   - current mode: {direct_objectives_manager.mode}")
                logger.info(f"   - current sequence_name: '{direct_objectives_manager.sequence_name}'")
                logger.info(f"   - requested sequence: '{direct_objectives_sequence}'")
                logger.info(f"   - needs_loading (initial): {needs_loading}")

                # Force reload if the requested sequence doesn't match the loaded sequence
                if direct_objectives_manager.is_sequence_active():
                    if direct_objectives_manager.sequence_name != direct_objectives_sequence:
                        logger.warning(
                            f"⚠️ Sequence mismatch: loaded='{direct_objectives_manager.sequence_name}', requested='{direct_objectives_sequence}' - forcing reload"
                        )
                        needs_loading = True
                    # Also force reload if requesting categorized but manager is in legacy mode
                    elif (
                        direct_objectives_sequence == "categorized_full_game"
                        and direct_objectives_manager.mode == "legacy"
                    ):
                        logger.warning(f"⚠️ Requesting categorized mode but manager is in legacy mode - forcing reload")
                        needs_loading = True

                logger.info(f"   - needs_loading (final): {needs_loading}")

                if needs_loading and os.environ.get("POKEAGENT_CLI_MODE") != "1":
                    # CLI agents do not use objectives; skip when POKEAGENT_CLI_MODE
                    from utils.data_persistence.run_data_manager import get_run_data_manager, get_cache_path

                    run_manager = get_run_data_manager()
                    objectives_run_dir = str(run_manager.get_scratch_space_dir()) if run_manager else None

                    # Try restoring from persisted objectives.json first
                    _restored_from_file = False
                    try:
                        cache_objectives_path = str(get_cache_path("objectives.json"))
                        if os.path.exists(cache_objectives_path):
                            direct_objectives_manager.restore_from_state(
                                json.load(open(cache_objectives_path, "r", encoding="utf-8"))
                            )
                            _restored_from_file = True
                            logger.info(f"✅ Restored objectives from {cache_objectives_path}")
                    except Exception as e:
                        logger.warning(f"Failed to restore objectives from cache, falling back to sequence load: {e}")

                    if not _restored_from_file:
                        if direct_objectives_sequence == "autonomous_objective_creation":
                            direct_objectives_manager.load_autonomous_objective_creation_sequence(
                                direct_objectives_start_index, run_dir=objectives_run_dir
                            )
                        elif direct_objectives_sequence == "categorized_full_game":
                            direct_objectives_manager.load_categorized_full_game_sequence(
                                start_story_index=direct_objectives_start_index,
                                start_battling_index=direct_objectives_battling_start_index,
                                run_dir=objectives_run_dir,
                            )
                        else:
                            logger.warning(f"Unknown direct objectives sequence: {direct_objectives_sequence}")

                        # Persist the freshly-loaded state
                        direct_objectives_manager.auto_save()

                    # Update objectives cache after loading
                    _update_objectives_cache()

            # Get current objective guidance
            if direct_objectives_manager.is_sequence_active():
                game_state = result.get("raw_state", {})

                # DEBUG: Log manager state
                logger.info(f"📊 Objectives manager state:")
                logger.info(f"   - mode: {direct_objectives_manager.mode}")
                logger.info(f"   - sequence_name: '{direct_objectives_manager.sequence_name}'")
                logger.info(f"   - is_active: {direct_objectives_manager.is_sequence_active()}")

                # Use categorized guidance if in categorized mode
                if direct_objectives_manager.mode == "categorized":
                    result["objectives_mode"] = "categorized"

                    categorized_guidance = direct_objectives_manager.get_categorized_objective_guidance(game_state)
                    if categorized_guidance:
                        result["categorized_objectives"] = categorized_guidance

                    # Add categorized status
                    result["categorized_status"] = {
                        "story": {
                            "current_index": direct_objectives_manager.story_index,
                            "total": len(direct_objectives_manager.story_sequence),
                            "completed": sum(1 for obj in direct_objectives_manager.story_sequence if obj.completed),
                        },
                        "battling": {
                            "current_index": direct_objectives_manager.battling_index,
                            "total": len(direct_objectives_manager.battling_sequence),
                            "completed": sum(1 for obj in direct_objectives_manager.battling_sequence if obj.completed),
                        },
                        "dynamics": {
                            "current_index": direct_objectives_manager.dynamics_index,
                            "total": len(direct_objectives_manager.dynamics_sequence),
                            "completed": sum(1 for obj in direct_objectives_manager.dynamics_sequence if obj.completed),
                        },
                    }
                else:
                    # Legacy mode
                    result["objectives_mode"] = "legacy"

                    current_guidance = direct_objectives_manager.get_current_objective_guidance(game_state)
                    if current_guidance:
                        result["direct_objective"] = current_guidance
                        result["direct_objective_status"] = direct_objectives_manager.get_sequence_status()

                    # Add objective context (previous, current, next) - legacy only
                    objective_context = direct_objectives_manager.get_objective_context(game_state)
                    if objective_context:
                        result["direct_objective_context"] = objective_context

        # Ensure all data is JSON-serializable (objectives data may contain enums/numpy types)
        from utils.json_utils import serialize_for_json
        return serialize_for_json(result)
    except Exception as e:
        logger.error(f"Error in get_game_state: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_map_data")
async def mcp_get_map_data():
    """MCP Tool: Get structured map data for skill code.

    Uses the same preprocessing as get_game_state (state_text with ASCII map)
    but extracts and returns the grid, warps, objects, etc. as structured data.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from utils.state_formatter import format_state_for_llm
        from server import game_tools

        global recent_button_presses, current_obs
        with obs_lock:
            obs_copy = current_obs.copy() if current_obs is not None else None

        state_result = game_tools.get_game_state_direct(
            env, format_state_for_llm,
            action_history=recent_button_presses,
            current_obs=obs_copy,
        )

        if not state_result.get("success"):
            return {"success": False, "error": "Could not get game state"}

        state_text = state_result.get("state_text", "")
        pos = state_result.get("player_position", {})
        location = state_result.get("location", "Unknown")

        # Game-specific player marker and legend
        _gt = os.environ.get("GAME_TYPE", "emerald").lower()
        if _gt == "red":
            _player_marker = "I"
            _grid_legend = (
                "I=player .=walkable #=wall ~=grass W=water D=door "
                "t=cuttable tree G=Card Key gate !=sign ?=hidden item "
                "O=pokeball N=NPC(blocked) ↓/←/→=jump ledge "
                "C=counter *=spinner stop B=bookshelf U=trash "
                "^=display/blueprint P=computer ==bench T=TV/machine"
            )
        else:
            _player_marker = "P"
            _grid_legend = "P=player .=walkable #=blocked ~=grass D=door S=stairs/warp I=item N=NPC(blocked)"

        result = {
            "success": True,
            "location": location,
            "player": {"x": pos.get("x", 0), "y": pos.get("y", 0)},
            "grid_legend": _grid_legend,
        }

        # Get the FULL map grid (not windowed) from the porymap data in raw_state
        # The state_text ASCII map may be cropped, but raw_state has the complete grid
        raw_state = state_result.get("raw_state", {})
        porymap_grid = raw_state.get("map", {}).get("porymap", {}).get("grid")
        grid = None

        if porymap_grid:
            # Use full porymap grid and add player marker
            grid = [list(row) for row in porymap_grid]
            px, py = pos.get("x", 0), pos.get("y", 0)
            if 0 <= py < len(grid) and 0 <= px < len(grid[0]):
                grid[py][px] = _player_marker

            # Mark live NPC positions as 'N' (blocked for pathfinding)
            obj_events = raw_state.get("map", {}).get("object_events", []) # object_events - from runtime memory; objects - from porymap data
            for obj in obj_events:
                ox = obj.get("current_x", -1)
                oy = obj.get("current_y", -1)
                if (ox, oy) != (px, py) and 0 <= oy < len(grid) and 0 <= ox < len(grid[0]):
                    if grid[oy][ox] not in ('#', _player_marker):
                        grid[oy][ox] = 'N'

            grid = ["".join(row) for row in grid]
        elif "ASCII Map:" in state_text:
            # Fallback: extract from state_text if porymap grid unavailable
            map_section = state_text.split("ASCII Map:")[1]
            legend_idx = map_section.find("(Legend:")
            if legend_idx > 0:
                map_section = map_section[:legend_idx]
            grid = [line for line in map_section.strip().split("\n") if line.strip()]

        if grid:
            result["grid"] = grid
            result["dimensions"] = {"width": len(grid[0]) if grid else 0, "height": len(grid)}

        # Extract the JSON map data block if present
        if "Map Data (JSON):" in state_text:
            try:
                import json as _json
                json_section = state_text.split("Map Data (JSON):")[1]
                # Find the JSON object
                start = json_section.find("{")
                if start >= 0:
                    depth = 0
                    end = start
                    for i, c in enumerate(json_section[start:], start):
                        if c == "{": depth += 1
                        elif c == "}": depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                    map_json = _json.loads(json_section[start:end])
                    result["warps"] = map_json.get("warps") or map_json.get("warp_events", [])
                    result["objects"] = map_json.get("objects", [])
                    result["connections"] = map_json.get("connections", [])
                    if not result.get("dimensions"):
                        result["dimensions"] = map_json.get("dimensions", {})
            except Exception:
                pass  # JSON parsing failed, grid is still available

        # Party info
        raw_state = state_result.get("raw_state", {})
        party = raw_state.get("player", {}).get("party") or raw_state.get("game", {}).get("party") or []
        if party:
            result["party"] = [
                {
                    "species": p.get("species_name", "?"),
                    "level": p.get("level", 0),
                    "hp": p.get("current_hp", 0),
                    "max_hp": p.get("max_hp", 0),
                    "moves": p.get("moves", []),
                }
                for p in party
            ]

        return result

    except Exception as e:
        logger.error(f"get_map_data error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.post("/mcp/press_buttons")
async def mcp_press_buttons(request: dict):
    """MCP Tool: Press GBA buttons"""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        buttons = request.get("buttons", [])
        reasoning = request.get("reasoning", "")
        source = request.get("source")
        metadata = request.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        # Normalize buttons to always be a list
        if isinstance(buttons, str):
            buttons = [buttons]
        elif isinstance(buttons, dict):
            # Handle dict like {'U': 'P'} -> reconstruct "UP"
            if len(buttons) == 1:
                key, value = next(iter(buttons.items()))
                if isinstance(value, str) and len(value) == 1:
                    buttons = [key + value]  # Reconstruct "UP" from {'U': 'P'}
                else:
                    buttons = list(buttons.keys())
            else:
                buttons = list(buttons.keys())

        if not buttons:
            return {"success": False, "error": "No buttons specified"}

        # GBC (Red) has no shoulder buttons — reject L/R
        if game_type == "red":
            invalid_shoulder = [b for b in buttons if str(b).upper().strip() in ("L", "R")]
            if invalid_shoulder:
                return {"success": False, "error": f"Game Boy has no shoulder buttons: {invalid_shoulder}"}

        # Valid buttons (including WAIT for no-op)
        if game_type == "red":
            valid_buttons = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
        else:
            valid_buttons = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R", "WAIT"]

        # Validate and normalize buttons with fallback to 'A'
        normalized_buttons = []
        invalid_buttons = []

        for button in buttons:
            # Normalize to uppercase
            button_upper = str(button).upper().strip()

            # Check if valid
            if button_upper in valid_buttons:
                normalized_buttons.append(button_upper)
            else:
                # Invalid button - fallback to A and warn
                invalid_buttons.append(button)
                logger.warning(f"Invalid button '{button}' requested, falling back to 'A'")
                normalized_buttons.append("A")

        # Validate and normalize buttons with fallback to 'A'
        normalized_buttons = []
        invalid_buttons = []

        for button in buttons:
            # Normalize to uppercase
            button_upper = str(button).upper().strip()

            # Check if valid
            if button_upper in valid_buttons:
                normalized_buttons.append(button_upper)
            else:
                # Invalid button - fallback to A and warn
                invalid_buttons.append(button)
                logger.warning(f"Invalid button '{button}' requested, falling back to 'A'")
                normalized_buttons.append("A")

        # Filter out WAIT buttons (they're just for agent decision-making, not actual button presses)
        actual_buttons = [b for b in normalized_buttons if b != "WAIT"]

        # If only WAIT was requested, treat it as a no-op but still complete successfully
        if not actual_buttons:
            logger.info(f"🎮 Agent chose to WAIT (no buttons pressed) - {reasoning}")
            return {"success": True, "buttons_queued": [], "reasoning": reasoning, "action": "WAIT"}

        # Call the existing take_action function to ensure metrics tracking
        action_request = ActionRequest(buttons=actual_buttons, source=source, metadata=metadata_dict)
        await take_action(action_request)
        
        # Track actual button presses in metrics (not text parsing!)
        try:
            from utils.data_persistence.llm_logger import increment_action_count
            increment_action_count(len(actual_buttons))
        except Exception as e:
            logger.debug(f"Could not increment action count: {e}")

        logger.info(f"🎮 Queued buttons via MCP: {actual_buttons} - {reasoning}")
        response_dict = {"success": True, "buttons_queued": actual_buttons, "reasoning": reasoning}

        # Include warning if any buttons were invalid
        if invalid_buttons:
            response_dict["warning"] = f"Invalid buttons replaced with 'A': {invalid_buttons}"

        return response_dict
    except Exception as e:
        logger.error(f"Error pressing buttons: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Browser game MCP endpoints (only active when game_type == "browser")
# ---------------------------------------------------------------------------

@app.post("/mcp/press_keys")
async def mcp_press_keys(request: dict):
    """MCP Tool: Press keyboard keys in a browser game."""
    if game_type != "browser":
        return {"success": False, "error": "press_keys is only available for browser games"}
    if env is None:
        return {"success": False, "error": "Browser environment not initialized"}
    try:
        keys = request.get("keys", [])
        reasoning = request.get("reasoning", "")
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            return {"success": False, "error": "No keys specified"}

        env.press_keys_sequence(keys)

        global browser_last_action
        browser_last_action = {"type": "press_keys", "keys": keys}

        # Return fresh screenshot and push to stream
        screenshot = env.get_screenshot()
        screenshot_b64 = _pil_to_base64(screenshot)
        update_frame_cache(screenshot)

        try:
            from utils.data_persistence.llm_logger import increment_action_count
            increment_action_count(len(keys))
        except Exception:
            pass

        return {
            "success": True,
            "keys_pressed": keys,
            "reasoning": reasoning,
            "screenshot_base64": screenshot_b64,
        }
    except Exception as e:
        logger.error(f"press_keys error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/mouse_click")
async def mcp_mouse_click(request: dict):
    """MCP Tool: Click at (x, y) coordinates in a browser game."""
    if game_type != "browser":
        return {"success": False, "error": "mouse_click is only available for browser games"}
    if env is None:
        return {"success": False, "error": "Browser environment not initialized"}
    try:
        x = int(request.get("x", 0))
        y = int(request.get("y", 0))
        reasoning = request.get("reasoning", "")

        env.click_at(x, y)

        global browser_last_action
        browser_last_action = {"type": "mouse_click", "x": x, "y": y}

        # Return fresh screenshot and push to stream
        screenshot = env.get_screenshot()
        screenshot_b64 = _pil_to_base64(screenshot)
        update_frame_cache(screenshot)

        try:
            from utils.data_persistence.llm_logger import increment_action_count
            increment_action_count(1)
        except Exception:
            pass

        return {
            "success": True,
            "clicked": {"x": x, "y": y},
            "reasoning": reasoning,
            "screenshot_base64": screenshot_b64,
        }
    except Exception as e:
        logger.error(f"mouse_click error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/double_click")
async def mcp_double_click(request: dict):
    """MCP Tool: Double-click at (x, y) coordinates in a browser game."""
    if game_type != "browser":
        return {"success": False, "error": "double_click is only available for browser games"}
    if env is None:
        return {"success": False, "error": "Browser environment not initialized"}
    try:
        x = int(request.get("x", 0))
        y = int(request.get("y", 0))
        reasoning = request.get("reasoning", "")

        env.double_click_at(x, y)

        global browser_last_action
        browser_last_action = {"type": "double_click", "x": x, "y": y}

        # Return fresh screenshot and push to stream
        screenshot = env.get_screenshot()
        screenshot_b64 = _pil_to_base64(screenshot)
        update_frame_cache(screenshot)

        try:
            from utils.data_persistence.llm_logger import increment_action_count
            increment_action_count(1)
        except Exception:
            pass

        return {
            "success": True,
            "double_clicked": {"x": x, "y": y},
            "reasoning": reasoning,
            "screenshot_base64": screenshot_b64,
        }
    except Exception as e:
        logger.error(f"double_click error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/hold_key")
async def mcp_hold_key(request: dict):
    """MCP Tool: Hold a keyboard key for a specified duration in a browser game."""
    if game_type != "browser":
        return {"success": False, "error": "hold_key is only available for browser games"}
    if env is None:
        return {"success": False, "error": "Browser environment not initialized"}
    try:
        key = request.get("key", "")
        duration_ms = int(request.get("duration_ms", 500))
        reasoning = request.get("reasoning", "")
        if not key:
            return {"success": False, "error": "No key specified"}

        env.hold_key(key, duration_ms=duration_ms)

        global browser_last_action
        browser_last_action = {"type": "hold_key", "key": key, "duration_ms": duration_ms}

        # Return fresh screenshot and push to stream
        screenshot = env.get_screenshot()
        screenshot_b64 = _pil_to_base64(screenshot)
        update_frame_cache(screenshot)

        try:
            from utils.data_persistence.llm_logger import increment_action_count
            increment_action_count(1)
        except Exception:
            pass

        return {
            "success": True,
            "key_held": key,
            "duration_ms": duration_ms,
            "reasoning": reasoning,
            "screenshot_base64": screenshot_b64,
        }
    except Exception as e:
        logger.error(f"hold_key error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/mouse_move")
async def mcp_mouse_move(request: dict):
    """MCP Tool: Move the mouse cursor (without clicking) in a browser game.

    Useful for hover-driven UI: tooltips, paddle-follows-cursor, mouse-look,
    or any game that reacts to ``mousemove`` events without a click.
    Coordinates are canvas-relative.
    """
    if game_type != "browser":
        return {"success": False, "error": "mouse_move is only available for browser games"}
    if env is None:
        return {"success": False, "error": "Browser environment not initialized"}
    try:
        x = int(request.get("x", 0))
        y = int(request.get("y", 0))
        steps = int(request.get("steps", 8))
        reasoning = request.get("reasoning", "")

        env.move_to(x, y, steps=steps)

        global browser_last_action
        browser_last_action = {"type": "mouse_move", "x": x, "y": y}

        screenshot = env.get_screenshot()
        screenshot_b64 = _pil_to_base64(screenshot)
        update_frame_cache(screenshot)

        try:
            from utils.data_persistence.llm_logger import increment_action_count
            increment_action_count(1)
        except Exception:
            pass

        return {
            "success": True,
            "moved_to": {"x": x, "y": y},
            "reasoning": reasoning,
            "screenshot_base64": screenshot_b64,
        }
    except Exception as e:
        logger.error(f"mouse_move error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/mouse_drag")
async def mcp_mouse_drag(request: dict):
    """MCP Tool: Press at (x1, y1), drag to (x2, y2), release.

    Coordinates are canvas-relative. Useful for drag-to-aim, dragging
    items, sliders, drawing, etc.
    """
    if game_type != "browser":
        return {"success": False, "error": "mouse_drag is only available for browser games"}
    if env is None:
        return {"success": False, "error": "Browser environment not initialized"}
    try:
        x1 = int(request.get("x1", 0))
        y1 = int(request.get("y1", 0))
        x2 = int(request.get("x2", 0))
        y2 = int(request.get("y2", 0))
        steps = int(request.get("steps", 12))
        hold_ms = int(request.get("hold_ms", 50))
        reasoning = request.get("reasoning", "")

        env.drag_to(x1, y1, x2, y2, steps=steps, hold_ms=hold_ms)

        global browser_last_action
        browser_last_action = {
            "type": "mouse_drag",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        }

        screenshot = env.get_screenshot()
        screenshot_b64 = _pil_to_base64(screenshot)
        update_frame_cache(screenshot)

        try:
            from utils.data_persistence.llm_logger import increment_action_count
            increment_action_count(1)
        except Exception:
            pass

        return {
            "success": True,
            "dragged": {"from": [x1, y1], "to": [x2, y2]},
            "reasoning": reasoning,
            "screenshot_base64": screenshot_b64,
        }
    except Exception as e:
        logger.error(f"mouse_drag error: {e}")
        return {"success": False, "error": str(e)}


def _pil_to_base64(img) -> str:
    """Convert a PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/mcp/complete_direct_objective")
async def mcp_complete_direct_objective(request: dict):
    """MCP Tool: Complete current direct objective"""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from agents.objectives import DirectObjectiveManager

        # Get current game state to check objective completion
        from utils.state_formatter import format_state_for_llm
        from server import game_tools

        game_state_result = game_tools.get_game_state_direct(env, format_state_for_llm)
        if not game_state_result.get("success", False):
            return {"success": False, "error": "Failed to get game state"}

        game_state = game_state_result.get("raw_state", {})

        # Use global direct objective manager and current_run_dir
        global direct_objectives_manager, current_run_dir

        # Initialize direct objective manager if needed
        if direct_objectives_manager is None:
            direct_objectives_manager = DirectObjectiveManager()

        # Load direct objectives sequence if specified
        if direct_objectives_sequence:
            # Check if we need to load objectives
            needs_loading = not direct_objectives_manager.is_sequence_active()

            # Force reload if the requested sequence doesn't match the loaded sequence
            if direct_objectives_manager.is_sequence_active():
                if direct_objectives_manager.sequence_name != direct_objectives_sequence:
                    logger.warning(
                        f"⚠️ Sequence mismatch: loaded='{direct_objectives_manager.sequence_name}', requested='{direct_objectives_sequence}' - forcing reload"
                    )
                    needs_loading = True
                # Also force reload if requesting categorized but manager is in legacy mode
                elif (
                    direct_objectives_sequence == "categorized_full_game" and direct_objectives_manager.mode == "legacy"
                ):
                    logger.warning(f"⚠️ Requesting categorized mode but manager is in legacy mode - forcing reload")
                    needs_loading = True

            if needs_loading and os.environ.get("POKEAGENT_CLI_MODE") != "1":
                # CLI agents do not use objectives; skip when POKEAGENT_CLI_MODE
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                objectives_run_dir = str(run_manager.get_scratch_space_dir()) if run_manager else None

                if direct_objectives_sequence == "autonomous_objective_creation":
                    direct_objectives_manager.load_autonomous_objective_creation_sequence(
                        direct_objectives_start_index, run_dir=objectives_run_dir
                    )
                elif direct_objectives_sequence == "categorized_full_game":
                    direct_objectives_manager.load_categorized_full_game_sequence(
                        start_story_index=direct_objectives_start_index,
                        start_battling_index=direct_objectives_battling_start_index,
                        run_dir=objectives_run_dir,
                    )
                else:
                    logger.warning(f"Unknown direct objectives sequence: {direct_objectives_sequence}")

                # Update objectives cache after loading
                _update_objectives_cache()

        # Check if sequence is active
        if not direct_objectives_manager.is_sequence_active():
            return {"success": False, "error": "No direct objective sequence active"}

        # Extract category parameter if provided
        category = request.get("category")

        # Check if we're in categorized mode
        if direct_objectives_manager.mode == "categorized":
            # Categorized mode - category parameter is required
            if not category:
                return {
                    "success": False,
                    "error": "Category parameter required in categorized mode (story, battling, or dynamics)",
                }

            if category not in ["story", "battling", "dynamics"]:
                return {
                    "success": False,
                    "error": f"Invalid category: {category}. Must be story, battling, or dynamics",
                }

            # Get current objective for the specified category
            current_obj = direct_objectives_manager._get_current_objective_for_category(category)
            if not current_obj:
                return {"success": False, "error": f"No current {category} objective to complete"}

            completed_objective_index = getattr(
                direct_objectives_manager, f"{category}_index"
            )

            # Mark objective as completed
            direct_objectives_manager._mark_objective_completed(current_obj)

            # Advance the appropriate index, and inject guidance when a category ends
            if category == "story":
                if direct_objectives_manager.story_index >= len(direct_objectives_manager.story_sequence) - 1:
                    next_obj = _build_story_planning_objective()
                    direct_objectives_manager.story_sequence.append(next_obj)
                    direct_objectives_manager.story_index = len(direct_objectives_manager.story_sequence) - 1
                    logger.info(
                        f"✅ Completed story objective: {current_obj.id} (appended create_new_objectives guidance at story index {direct_objectives_manager.story_index})"
                    )
                else:
                    direct_objectives_manager.story_index += 1
                    logger.info(
                        f"✅ Completed story objective: {current_obj.id} (advanced to story index {direct_objectives_manager.story_index})"
                    )
            elif category == "battling":
                direct_objectives_manager.battling_index += 1
                logger.info(
                    f"✅ Completed battling objective: {current_obj.id} (advanced to battling index {direct_objectives_manager.battling_index})"
                )
            elif category == "dynamics":
                direct_objectives_manager.dynamics_index += 1
                logger.info(
                    f"✅ Completed dynamics objective: {current_obj.id} (advanced to dynamics index {direct_objectives_manager.dynamics_index})"
                )
        else:
            # Legacy mode - ignore category parameter
            current_obj = direct_objectives_manager.get_current_objective()
            if not current_obj:
                return {"success": False, "error": "No current objective to complete"}

            completed_objective_index = direct_objectives_manager.current_index

            # Mark objective as completed
            direct_objectives_manager._mark_objective_completed(current_obj)

            hold_index = (
                current_obj.action_type == "create_new_objectives"
                and direct_objectives_manager.current_index >= len(direct_objectives_manager.current_sequence) - 1
            )
            if hold_index:
                logger.info(
                    f"✅ Completed objective: {current_obj.id} (holding index {direct_objectives_manager.current_index} for create_new_objectives guidance)"
                )
            else:
                direct_objectives_manager.current_index += 1
                logger.info(
                    f"✅ Completed objective: {current_obj.id} (advanced to index {direct_objectives_manager.current_index})"
                )

        # Persist full objectives state after mutation
        direct_objectives_manager.auto_save()

        # Update objectives cache for stream.html (fast file read)
        _update_objectives_cache()

        # Log objective completion to cumulative_metrics.json (objectives column)
        try:
            from utils.data_persistence.llm_logger import log_objective_completion

            log_objective_completion(
                objective_id=current_obj.id,
                category=category if direct_objectives_manager.mode == "categorized" else "legacy",
                objective_index=completed_objective_index,
                step_number=agent_step_count,
            )
        except Exception as e:
            logger.warning("Failed to log objective completion to metrics: %s", e)

        # Persist completed objectives after each completion (for real time execution... get_progress_summary() relies on this run_data)
        # CLI agents do not use objectives; skip when POKEAGENT_CLI_MODE
        if os.environ.get("POKEAGENT_CLI_MODE") != "1":
            try:
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                if not run_manager:
                    logger.warning("Cannot save completed_objectives: run_data_manager not initialized")
                else:
                    scratch_space_dir = str(run_manager.get_scratch_space_dir())
                    if direct_objectives_manager.mode == "categorized":
                        completed_path = os.path.join(scratch_space_dir, "completed_objectives.json")
                        if os.path.exists(completed_path):
                            with open(completed_path, "r") as f:
                                history = json.load(f)
                        else:
                            history = {
                                "mode": "categorized",
                                "sequence_name": direct_objectives_manager.sequence_name,
                                "categories": {"story": [], "battling": [], "dynamics": []},
                            }

                        category_key = category or "dynamics"
                        history.setdefault("categories", {})
                        history["categories"].setdefault("story", [])
                        history["categories"].setdefault("battling", [])
                        history["categories"].setdefault("dynamics", [])
                        history["categories"][category_key].append(
                            {
                                "id": current_obj.id,
                                "description": current_obj.description,
                                "target_location": current_obj.target_location,
                                "action_type": current_obj.action_type,
                                "completed_at": current_obj.completed_at.isoformat()
                                if hasattr(current_obj, "completed_at") and current_obj.completed_at
                                else None,
                                "category": category_key,
                            }
                        )
                        history["last_updated"] = datetime.datetime.now().isoformat()

                        with open(completed_path, "w") as f:
                            json.dump(history, f, indent=2)
                        logger.info(f"💾 Saved completed objectives to {completed_path}")
                    else:
                        saved_file = direct_objectives_manager.save_completed_objectives(run_dir=scratch_space_dir)
                        logger.info(f"💾 Saved completed objectives to {saved_file}")
            except Exception as e:
                logger.warning(f"Failed to save completed objectives: {e}")

        # Create backup of .pokeagent_cache after completing objective
        try:
            from utils.data_persistence.backup_manager import create_cache_backup

            backup_path = create_cache_backup(
                objective_id=current_obj.id, objective_description=current_obj.description
            )
            if backup_path:
                logger.info(f"📦 Created cache backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create cache backup: {e}")

        # Get next objective if available
        if direct_objectives_manager.mode == "categorized":
            next_guidance = direct_objectives_manager.get_categorized_objective_guidance(game_state)
            categorized_status = {
                "story": {
                    "current_index": direct_objectives_manager.story_index,
                    "total": len(direct_objectives_manager.story_sequence),
                    "completed": sum(1 for obj in direct_objectives_manager.story_sequence if obj.completed),
                },
                "battling": {
                    "current_index": direct_objectives_manager.battling_index,
                    "total": len(direct_objectives_manager.battling_sequence),
                    "completed": sum(1 for obj in direct_objectives_manager.battling_sequence if obj.completed),
                },
                "dynamics": {
                    "current_index": direct_objectives_manager.dynamics_index,
                    "total": len(direct_objectives_manager.dynamics_sequence),
                    "completed": sum(1 for obj in direct_objectives_manager.dynamics_sequence if obj.completed),
                },
            }
            from utils.json_utils import serialize_for_json
            return serialize_for_json({
                "success": True,
                "completed_objective": {"id": current_obj.id, "description": current_obj.description},
                "next_objective": next_guidance,
                "sequence_status": categorized_status,
            })
        else:
            next_obj = direct_objectives_manager.get_current_objective()
        if next_obj:
            next_guidance = direct_objectives_manager.get_current_objective_guidance(game_state)
            from utils.json_utils import serialize_for_json
            return serialize_for_json({
                "success": True,
                "completed_objective": {"id": current_obj.id, "description": current_obj.description},
                "next_objective": next_guidance,
                "sequence_status": direct_objectives_manager.get_sequence_status(),
            })
        else:
            # Sequence complete - save to history in timestamped run directory
            # CLI agents do not use objectives; skip when POKEAGENT_CLI_MODE
            if current_run_dir and os.environ.get("POKEAGENT_CLI_MODE") != "1":
                try:
                    # Save to agent_scratch_space in run_data
                    from utils.data_persistence.run_data_manager import get_run_data_manager

                    run_manager = get_run_data_manager()
                    if not run_manager:
                        logger.error("Cannot save completed_objectives: run_data_manager not initialized")
                        raise RuntimeError("run_data_manager must be initialized to save completed_objectives")
                    scratch_space_dir = str(run_manager.get_scratch_space_dir())
                    saved_file = direct_objectives_manager.save_completed_objectives(run_dir=scratch_space_dir)
                    logger.info(f"💾 Saved completed objectives to {saved_file}")
                except Exception as e:
                    logger.warning(f"Failed to save completed objectives: {e}")

            # Automatically create a new objective to guide the agent through next steps
            try:
                # Get current game state for context
                next_step_obj = _build_story_planning_objective()

                # Add the new objective to the sequence
                direct_objectives_manager.current_sequence.append(next_step_obj)

                # Get guidance for the new objective
                next_guidance = direct_objectives_manager.get_current_objective_guidance(game_state)

                from utils.json_utils import serialize_for_json
                return serialize_for_json({
                    "success": True,
                    "completed_objective": {"id": current_obj.id, "description": current_obj.description},
                    "next_objective": next_guidance,
                    "sequence_status": direct_objectives_manager.get_sequence_status(),
                    "message": "All objectives completed! A new objective has been automatically created to guide you through planning more objectives and continuing the run.",
                    "sequence_complete": False,  # Not complete anymore - we added a new objective
                    "auto_objective_created": True,
                })
            except Exception as e:
                logger.error(f"Failed to create auto objective: {e}")
                # Fallback to original behavior if auto-objective creation fails
                from utils.json_utils import serialize_for_json
                return serialize_for_json({
                    "success": True,
                    "completed_objective": {"id": current_obj.id, "description": current_obj.description},
                    "next_objective": None,
                    "sequence_status": direct_objectives_manager.get_sequence_status(),
                    "message": "All objectives completed! Review your progress and plan more objectives to continue the run.",
                    "sequence_complete": True,
                })

    except Exception as e:
        logger.error(f"Error completing direct objective: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/navigate_to")
async def mcp_navigate_to(request: dict):
    """MCP Tool: Navigate to coordinates using pathfinding"""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from server import game_tools

        x = request.get("x")
        y = request.get("y")
        reason = request.get("reason", "")
        variance = request.get("variance")

        if x is None or y is None:
            return {"success": False, "error": "x and y coordinates required"}

        # Convert to int in case they come as floats from JSON
        try:
            x = int(x)
            y = int(y)
        except (ValueError, TypeError):
            return {"success": False, "error": f"Invalid coordinates: x={x}, y={y}"}

        # Calculate path and get buttons
        result = game_tools.navigate_to_direct(env, x, y, reason=reason, variance=variance)

        if not result.get("success"):
            return result

        # Queue buttons via take_action to ensure metrics tracking
        buttons = result.get("buttons", [])
        if buttons:
            variance_value = result.get("variance")
            if variance_value is None:
                variance_value = variance or "none"
            action_request = ActionRequest(buttons=buttons, source="navigate_to", metadata={"variance": variance_value})
            await take_action(action_request)

        return {
            "success": True,
            "target": result.get("target"),
            "path_length": result.get("path_length"),
            "buttons_queued": len(buttons),
            "reason": reason,
            "variance": result.get("variance", variance or "none"),
        }
    except Exception as e:
        logger.error(f"Error in navigate_to: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/add_memory")
@app.post("/mcp/add_knowledge")  # backward-compat alias
async def mcp_add_memory(request: dict):
    """MCP Tool: Add entry to long-term memory (persistent across runs)"""
    try:
        from server import game_tools

        result = game_tools.add_memory_direct(
            category=request.get("category"),
            title=request.get("title"),
            content=request.get("content"),
            location=request.get("location"),
            coordinates=request.get("coordinates"),
            importance=request.get("importance", 3),
        )
        if os.environ.get("POKEAGENT_CLI_MODE") != "1":
            try:
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                if run_manager:
                    run_manager.copy_memory()
                else:
                    logger.warning("Cannot copy memory: run_data_manager not initialized")
            except Exception as e:
                logger.warning(f"Failed to copy memory: {e}")

        return result
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/search_memory")
@app.post("/mcp/search_knowledge")  # backward-compat alias
async def mcp_search_memory(request: dict):
    """MCP Tool: Search long-term memory (persistent across runs)"""
    try:
        from server import game_tools

        return game_tools.search_memory_direct(
            category=request.get("category"),
            query=request.get("query"),
            location=request.get("location"),
            min_importance=request.get("min_importance", 1),
        )
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_memory_summary")
@app.post("/mcp/get_knowledge_summary")  # backward-compat alias
async def mcp_get_memory_summary(request: dict):
    """MCP Tool: Get long-term memory summary (persistent across runs)"""
    try:
        from server import game_tools

        return game_tools.get_memory_summary_direct(min_importance=request.get("min_importance", 3))
    except Exception as e:
        logger.error(f"Error getting memory summary: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_memory_overview")
async def mcp_get_memory_overview(request: dict):
    """MCP Tool: Get long-term memory tree overview (compact [id] title tree)."""
    try:
        from server import game_tools

        return game_tools.get_memory_overview_direct()
    except Exception as e:
        logger.error(f"Error getting memory overview: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/process_memory")
async def mcp_process_memory(request: dict):
    """MCP Tool: Unified CRUD for long-term memory (read/add/update/delete)."""
    try:
        from server import game_tools

        action = request.get("action", "")
        entries = request.get("entries", [])
        reasoning = request.get("reasoning", "")
        result = game_tools.process_memory_direct(action, entries, reasoning)

        if action in ("add", "update", "delete") and result.get("success"):
            from utils.data_persistence.run_data_manager import get_run_data_manager
            run_manager = get_run_data_manager()
            if run_manager:
                try:
                    run_manager.copy_memory()
                except Exception as sync_err:
                    logger.warning(f"Failed to sync memory to run_data: {sync_err}")

        return result
    except Exception as e:
        logger.error(f"Error in process_memory: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_skill_overview")
async def mcp_get_skill_overview(request: dict):
    """MCP Tool: Get skill library tree overview (compact [id] name tree)."""
    try:
        from server import game_tools

        return game_tools.get_skill_overview_direct()
    except Exception as e:
        logger.error(f"Error getting skill overview: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/process_skill")
async def mcp_process_skill(request: dict):
    """MCP Tool: Unified CRUD for skill library (read/add/update/delete)."""
    try:
        from server import game_tools

        action = request.get("action", "")
        entries = request.get("entries", [])
        reasoning = request.get("reasoning", "")
        result = game_tools.process_skill_direct(action, entries, reasoning)

        if action in ("add", "update", "delete") and result.get("success"):
            from utils.data_persistence.run_data_manager import get_run_data_manager
            run_manager = get_run_data_manager()
            if run_manager:
                try:
                    run_manager.copy_skills()
                except Exception as sync_err:
                    logger.warning(f"Failed to sync skills to run_data: {sync_err}")

        return result
    except Exception as e:
        logger.error(f"Error in process_skill: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_subagent_overview")
async def mcp_get_subagent_overview(request: dict):
    """MCP Tool: Get subagent registry tree overview (compact [id] name tree)."""
    try:
        from server import game_tools

        return game_tools.get_subagent_overview_direct()
    except Exception as e:
        logger.error(f"Error getting subagent overview: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/process_subagent")
async def mcp_process_subagent(request: dict):
    """MCP Tool: Unified CRUD for subagent registry (read/add/update/delete)."""
    try:
        from server import game_tools

        action = request.get("action", "")
        entries = request.get("entries", [])
        reasoning = request.get("reasoning", "")
        result = game_tools.process_subagent_direct(action, entries, reasoning)

        if action in ("add", "update", "delete") and result.get("success"):
            from utils.data_persistence.run_data_manager import get_run_data_manager
            run_manager = get_run_data_manager()
            if run_manager:
                try:
                    run_manager.copy_subagents()
                except Exception as sync_err:
                    logger.warning(f"Failed to sync subagents to run_data: {sync_err}")

        return result
    except Exception as e:
        logger.error(f"Error in process_subagent: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/lookup_pokemon_info")
async def mcp_lookup_pokemon_info(request: dict):
    """MCP Tool: Lookup Pokemon info from wikis"""
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import quote_plus

        topic = request.get("topic")
        source = request.get("source", "bulbapedia")

        if not topic:
            return {"success": False, "error": "topic is required"}

        # Wiki sources configuration
        POKEMON_WIKI_SOURCES = {
            "bulbapedia": {
                "base_url": "https://bulbapedia.bulbagarden.net/wiki/",
                "search_url": "https://bulbapedia.bulbagarden.net/w/index.php?search=",
                "description": "Comprehensive Pokemon encyclopedia",
            },
            "serebii": {
                "base_url": "https://www.serebii.net/",
                "emerald_url": "https://www.serebii.net/emerald/",
                "description": "Detailed Pokemon Emerald guides and data",
            },
            "pokemondb": {"base_url": "https://pokemondb.net/", "description": "Pokemon Database"},
            "marriland": {
                "base_url": "https://marriland.com/",
                "emerald_url": "https://marriland.com/pokemon-emerald/",
                "description": "Marriland guides",
            },
        }

        if source not in POKEMON_WIKI_SOURCES:
            return {
                "success": False,
                "error": f"Unknown source '{source}'. Available: {', '.join(POKEMON_WIKI_SOURCES.keys())}",
            }

        source_info = POKEMON_WIKI_SOURCES[source]

        # Build URL based on source
        if source == "bulbapedia":
            formatted_topic = topic.replace(" ", "_")
            url = f"{source_info['base_url']}{formatted_topic}"
        elif source == "serebii":
            formatted_topic = topic.lower().replace(" ", "")
            url = f"{source_info['emerald_url']}{formatted_topic}.shtml"
        elif source == "pokemondb":
            formatted_topic = topic.lower().replace(" ", "-")
            url = f"{source_info['base_url']}pokedex/{formatted_topic}"
        elif source == "marriland":
            formatted_topic = topic.lower().replace(" ", "-")
            url = f"{source_info['emerald_url']}{formatted_topic}/"
        else:
            url = f"{source_info['base_url']}{topic}"

        logger.info(f"📚 Fetching Pokemon info: {topic} from {source}")

        # Fetch the page
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; PokeAgent/1.0)"})

        # If 404, try search instead
        if response.status_code == 404 and source == "bulbapedia":
            search_url = f"{source_info['search_url']}{quote_plus(topic)}"
            logger.info(f"Page not found, trying search: {search_url}")
            response = requests.get(
                search_url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; PokeAgent/1.0)"}
            )

        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Extract main content based on source
        content = ""
        if source == "bulbapedia":
            main_content = soup.find("div", id="mw-content-text")
            if main_content:
                paragraphs = main_content.find_all("p", limit=5)
                content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        else:
            content = soup.get_text()
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = "\n".join(chunk for chunk in chunks if chunk)

        # Limit content length
        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Content truncated - {len(content)} total characters]"

        if not content or len(content) < 50:
            return {"success": False, "error": f"Could not extract meaningful content from {url}"}

        return {
            "success": True,
            "topic": topic,
            "source": source,
            "url": url,
            "content": content,
            "description": source_info["description"],
        }

    except Exception as e:
        logger.error(f"Error looking up Pokemon info: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/list_wiki_sources")
async def mcp_list_wiki_sources():
    """MCP Tool: List available wiki sources"""
    POKEMON_WIKI_SOURCES = {
        "bulbapedia": {
            "base_url": "https://bulbapedia.bulbagarden.net/wiki/",
            "description": "Comprehensive Pokemon encyclopedia",
        },
        "serebii": {
            "base_url": "https://www.serebii.net/",
            "emerald_url": "https://www.serebii.net/emerald/",
            "description": "Detailed Pokemon Emerald guides and data",
        },
        "pokemondb": {"base_url": "https://pokemondb.net/", "description": "Pokemon Database"},
        "marriland": {
            "base_url": "https://marriland.com/",
            "emerald_url": "https://marriland.com/pokemon-emerald/",
            "description": "Marriland guides",
        },
    }

    sources = []
    for name, info in POKEMON_WIKI_SOURCES.items():
        sources.append(
            {
                "name": name,
                "description": info["description"],
                "base_url": info.get("base_url", ""),
                "emerald_url": info.get("emerald_url", ""),
            }
        )

    return {
        "success": True,
        "sources": sources,
        "count": len(sources),
        "usage": "Use lookup_pokemon_info(topic, source) to fetch information",
    }


@app.post("/mcp/get_walkthrough")
async def mcp_get_walkthrough(request: dict):
    """MCP Tool: Get Pokemon Emerald walkthrough part"""
    try:
        import requests
        from bs4 import BeautifulSoup

        part = request.get("part")

        if part is None:
            return {"success": False, "error": "part is required"}

        # Convert to int in case it comes as float from JSON
        try:
            part = int(part)
        except (ValueError, TypeError):
            return {"success": False, "error": f"Invalid part number: {part}"}

        # Build Bulbapedia walkthrough URL (game-specific)
        if game_type == "red":
            if not 1 <= part <= 17:
                return {"success": False, "error": f"Red walkthrough only has parts 1-17 (got {part})"}
            url = f"https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Red_and_Blue/Part_{part}"
            logger.info(f"📖 Fetching Red walkthrough part {part}")
        else:
            if not 1 <= part <= 21:
                return {"success": False, "error": f"Part must be between 1 and 21 (got {part})"}
            url = f"https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Emerald/Part_{part}"
            logger.info(f"📖 Fetching Emerald walkthrough part {part}")

        # Fetch the page
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; PokeAgent/1.0)"})
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "table"]):
            element.decompose()

        # Extract main content
        main_content = soup.find("div", id="mw-content-text")
        if not main_content:
            return {"success": False, "error": f"Could not find main content for Part {part}"}

        # Get all paragraphs and headings for structured walkthrough
        content_parts = []
        for element in main_content.find_all(["h2", "h3", "h4", "p", "ul"], limit=50):
            if element.name in ["h2", "h3", "h4"]:
                heading_text = element.get_text(strip=True)
                if heading_text and not heading_text.startswith("[edit]"):
                    level = element.name
                    if level == "h2":
                        content_parts.append(f"\n## {heading_text}")
                    elif level == "h3":
                        content_parts.append(f"\n### {heading_text}")
                    else:
                        content_parts.append(f"\n#### {heading_text}")
            elif element.name == "p":
                para_text = element.get_text(strip=True)
                if para_text and len(para_text) > 20:
                    content_parts.append(para_text)
            elif element.name == "ul":
                for li in element.find_all("li"):
                    li_text = li.get_text(strip=True)
                    if li_text:
                        content_parts.append(f"  - {li_text}")

        content = "\n\n".join(content_parts)

        # Limit content length
        max_chars = 8000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Content truncated - {len(content)} total characters]"

        if not content or len(content) < 100:
            return {"success": False, "error": f"Could not extract meaningful content from Part {part}"}

        return {
            "success": True,
            "part": part,
            "url": url,
            "content": content,
            "description": f"Pokemon Emerald Walkthrough - Part {part}",
        }

    except Exception as e:
        logger.error(f"Error getting walkthrough: {e}")
        return {"success": False, "error": str(e)}


@app.post("/mcp/save_memory")
async def mcp_save_memory(request: dict):
    """MCP Tool: Save facts to persistent memory (saved to run directory)"""
    global current_run_dir

    fact = request.get("fact")
    if not fact:
        return {"success": False, "error": "fact is required"}

    if not current_run_dir:
        return {"success": False, "error": "No run directory available"}

    try:
        memory_file = os.path.join(current_run_dir, "AGENT.md")

        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = "# Agent Memory\n\nThis file stores facts and observations from the AI agent.\n"

        if "## Agent Memories" not in content:
            if content and not content.endswith("\n"):
                content += "\n"
            content += "\n## Agent Memories\n"

        content += f"- {fact}\n"

        with open(memory_file, "w", encoding="utf-8") as f:
            f.write(content)

        return {"success": True, "message": f"Memory saved to {memory_file}", "path": memory_file}
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        return {"success": False, "error": str(e)}


def _coerce_replan_edits_to_list(edits: Any) -> List[Dict[str, Any]]:
    """Normalize ``edits`` from JSON / odd client encodings into ``list[dict]``."""
    return normalize_replan_edits(edits)


@app.post("/mcp/replan_objectives")
async def mcp_replan_objectives(request: dict):
    """MCP Tool: Apply index-based edits to a single objective category (used by subagent_plan_objectives)."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        from agents.objectives import DirectObjectiveManager

        global direct_objectives_manager
        if direct_objectives_manager is None:
            return {"success": False, "error": "Objective manager not initialized"}

        if direct_objectives_manager.mode != "categorized":
            return {"success": False, "error": "replan_objectives requires categorized mode"}

        category = request.get("category")
        edits = _coerce_replan_edits_to_list(request.get("edits"))
        return_to_orchestrator = request.get("return_to_orchestrator", False)
        rationale = request.get("rationale", "")

        if not category:
            return {"success": False, "error": "Missing required 'category' parameter"}

        result = direct_objectives_manager.replan_category(category, edits)

        if result.get("success"):
            _update_objectives_cache()
            logger.info(
                f"🔄 replan_objectives({category}): {len(result.get('edits_applied', []))} edits applied. "
                f"return_to_orchestrator={return_to_orchestrator}"
            )

        result["return_to_orchestrator"] = return_to_orchestrator
        result["rationale"] = rationale
        return result

    except Exception as e:
        logger.error(f"Error in replan_objectives: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_full_objective_sequence")
async def mcp_get_full_objective_sequence():
    """MCP Tool: Return the complete objective state across all categories (used by subagent_plan_objectives)."""
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        global direct_objectives_manager
        if direct_objectives_manager is None:
            return {"success": False, "error": "Objective manager not initialized"}

        from utils.json_utils import serialize_for_json
        snapshot = direct_objectives_manager.get_full_sequence_snapshot()
        return serialize_for_json({"success": True, **snapshot})

    except Exception as e:
        logger.error(f"Error in get_full_objective_sequence: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/get_progress_summary")
async def mcp_get_progress_summary(request: dict):
    """MCP Tool: Get comprehensive progress summary.

    Request body may include ``compact: true`` to omit ``memory_overview`` and
    ``completed_sequences_history`` (for subagent prompts that already load memory
    via get_memory_overview). Default / orchestrator tool calls with ``{}`` return
    the full payload.
    """
    if env is None:
        return {"success": False, "error": "Emulator not initialized"}

    try:
        req = request if isinstance(request, dict) else {}
        compact = bool(req.get("compact"))
        # Get milestones
        milestones = env.milestone_tracker.milestones if env.milestone_tracker else {}
        completed_milestones = [mid for mid, data in milestones.items() if data.get("completed", False)]

        # Get current game state
        from utils.state_formatter import format_state_for_llm
        from server import game_tools

        game_state_result = game_tools.get_game_state_direct(env, format_state_for_llm)
        game_state = game_state_result.get("raw_state", {})

        # Get direct objective status
        global direct_objectives_manager
        obj_status = {}
        completed_history = []
        if direct_objectives_manager:
            obj_status = direct_objectives_manager.get_sequence_status()

            # Load completed objectives history (skip when compact — subagents use separate memory overview)
            if not compact:
                global current_run_dir
                completed_obj_file = None
                if current_run_dir and os.environ.get("POKEAGENT_CLI_MODE") != "1":
                    from utils.data_persistence.run_data_manager import get_run_data_manager

                    run_manager = get_run_data_manager()
                    if not run_manager:
                        logger.warning("Cannot load completed_objectives: run_data_manager not initialized")
                    else:
                        completed_obj_file = str(run_manager.get_scratch_space_dir() / "completed_objectives.json")
                if completed_obj_file and os.path.exists(completed_obj_file):
                    import json

                    with open(completed_obj_file, "r") as f:
                        history_data = json.load(f)
                        if "categories" in history_data:
                            completed_history = history_data.get("categories", {})
                        else:
                            completed_history = history_data.get("sequences", [])

        memory_overview = ""
        if not compact:
            mem_result = game_tools.get_memory_overview_direct()
            memory_overview = mem_result.get("overview", "No memories yet") if isinstance(mem_result, dict) else "No memories yet"

        # Get location and coordinates
        location = game_state.get("player", {}).get("location", "Unknown")
        player_pos = game_state_result.get("player_position", {}) if game_state_result.get("success") else {}

        from utils.json_utils import serialize_for_json

        progress = {
            "milestones_completed": completed_milestones,
            "total_milestones_completed": len(completed_milestones),
            "current_location": location,
            "player_coordinates": {"x": player_pos.get("x"), "y": player_pos.get("y")} if player_pos else None,
            "direct_objectives": {
                "current_sequence": obj_status.get("sequence_name"),
                "objectives_completed_in_current_sequence": obj_status.get("completed_count", 0),
                "total_in_current_sequence": obj_status.get("total_objectives", 0),
                "is_sequence_complete": obj_status.get("is_complete", False),
                "current_objective": obj_status.get("current_objective"),
            },
            "run_directory": current_run_dir,
        }
        if not compact:
            progress["completed_sequences_history"] = completed_history
            progress["memory_overview"] = memory_overview

        return serialize_for_json({"success": True, "progress": progress})
    except Exception as e:
        logger.error(f"Error getting progress summary: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/save_map")
async def mcp_save_map(request: dict):
    """MCP Tool: Save agent's mental map of a location to file (SLAM)"""
    try:
        location_name = request.get("location_name", "")
        map_data = request.get("map_data", "")

        if not location_name:
            return {"success": False, "error": "location_name is required"}

        if not map_data:
            return {"success": False, "error": "map_data is required"}

        # Create maps directory (Path imported at top of file)
        from utils.data_persistence.run_data_manager import get_cache_path
        maps_dir = get_cache_path("maps")
        maps_dir.mkdir(parents=True, exist_ok=True)

        # Normalize case to title case for consistent filenames
        normalized_location = location_name.title()
        # Sanitize location name for filename
        safe_name = "".join(c for c in normalized_location if c.isalnum() or c in (" ", "_", "-")).strip()
        safe_name = safe_name.replace(" ", "_")

        map_file = maps_dir / f"{safe_name}.txt"
        map_file.write_text(map_data)

        logger.info(f"🗺️  Saved SLAM map for {location_name} to {map_file}")
        return {
            "success": True,
            "message": f"Map saved to {map_file}",
            "location": location_name,
            "file_path": str(map_file),
        }

    except Exception as e:
        logger.error(f"Error saving map: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/mcp/load_map")
async def mcp_load_map(request: dict):
    """MCP Tool: Load agent's previously saved mental map of a location (SLAM)"""
    try:
        location_name = request.get("location_name", "")

        if not location_name:
            return {"success": False, "error": "location_name is required"}

        # Create maps directory if it doesn't exist (Path imported at top of file)
        from utils.data_persistence.run_data_manager import get_cache_path
        maps_dir = get_cache_path("maps")
        maps_dir.mkdir(parents=True, exist_ok=True)

        # Normalize case to title case for consistent filenames
        normalized_location = location_name.title()
        # Sanitize location name for filename
        safe_name = "".join(c for c in normalized_location if c.isalnum() or c in (" ", "_", "-")).strip()
        safe_name = safe_name.replace(" ", "_")

        map_file = maps_dir / f"{safe_name}.txt"

        if not map_file.exists():
            logger.info(f"🗺️  No existing SLAM map for {location_name}")
            return {
                "success": True,
                "message": f"No existing map for {location_name}",
                "location": location_name,
                "map_data": None,
            }

        map_data = map_file.read_text()
        logger.info(f"🗺️  Loaded SLAM map for {location_name} from {map_file}")
        return {
            "success": True,
            "message": f"Map loaded from {map_file}",
            "location": location_name,
            "map_data": map_data,
            "file_path": str(map_file),
        }

    except Exception as e:
        logger.error(f"Error loading map: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ============================================================================
# SERVER CONTROL
# ============================================================================


@app.post("/stop")
async def stop_server():
    """Stop the server"""
    global running
    running = False
    return {"status": "stopping"}


def _require_state_api_key(request: Request) -> Optional[JSONResponse]:
    """
    When POKEMON_STATE_API_KEY is set (e.g. by run_cli), require X-Internal-API-Key header.
    Prevents the CLI agent from loading/saving state via Bash curl.
    """
    key = os.environ.get("POKEMON_STATE_API_KEY")
    if not key:
        return None
    if request.headers.get("X-Internal-API-Key") != key:
        logger.warning("Rejected state endpoint call: missing or invalid X-Internal-API-Key")
        return JSONResponse(status_code=403, content={"error": "Forbidden: state API protected"})
    return None


@app.post("/save_state")
async def save_state_endpoint(request: Request):
    """Save the current emulator state to a file"""
    if err := _require_state_api_key(request):
        return err
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        from utils.data_persistence.run_data_manager import get_cache_path
        default_filepath = str(get_cache_path("manual_save.state"))
        filepath = body.get("filepath", default_filepath)
        if env:
            env.save_state(filepath)
            logger.info(f"💾 State saved to: {filepath}")
            return {"status": "success", "message": f"State saved to {filepath}"}
        else:
            return JSONResponse(status_code=500, content={"error": "Emulator not initialized"})
    except Exception as e:
        logger.error(f"Error saving state: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/load_state")
async def load_state_endpoint(request: Request):
    """Load an emulator state from a file"""
    if err := _require_state_api_key(request):
        return err
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        from utils.data_persistence.run_data_manager import get_cache_path
        default_filepath = str(get_cache_path("manual_save.state"))
        filepath = body.get("filepath", default_filepath)
        if env:
            if not os.path.exists(filepath):
                return JSONResponse(status_code=404, content={"error": f"State file not found: {filepath}"})
            env.load_state(filepath)
            logger.info(f"📂 State loaded from: {filepath}")
            return {"status": "success", "message": f"State loaded from {filepath}"}
        else:
            return JSONResponse(status_code=500, content={"error": "Emulator not initialized"})
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/checkpoint")
async def save_checkpoint(request_data: dict = None):
    """Save checkpoint - called by client when step count reaches checkpoint interval"""
    try:
        from utils.data_persistence.run_data_manager import get_cache_path
        step_count = request_data.get("step_count", 0) if request_data else 0

        # Save emulator state (not applicable for browser games)
        checkpoint_state = str(get_cache_path("checkpoint.state"))
        if env and hasattr(env, "save_state"):
            env.save_state(checkpoint_state)
            logger.info(f"💾 Server: Saved checkpoint state at step {step_count}")

            # Save milestones
            if hasattr(env, "milestone_tracker") and env.milestone_tracker:
                milestone_file = env.milestone_tracker.save_milestones_for_state(checkpoint_state)
                logger.info(f"💾 Server: Saved checkpoint milestones")

            return {
                "status": "checkpoint_saved",
                "step_count": step_count,
                "files": {
                    "state": checkpoint_state,
                    "milestones": str(get_cache_path("checkpoint_milestones.json")),
                    "map": str(get_cache_path("checkpoint_grids.json"))
                }
            }
        else:
            return {"status": "error", "message": "No emulator available"}

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/sync_llm_metrics")
async def sync_llm_metrics(request: Request):
    """Sync LLM cumulative metrics from client to server"""
    try:
        request_data = await request.json()
        cumulative_metrics = request_data.get("cumulative_metrics", {})

        if not cumulative_metrics:
            return {"status": "error", "message": "No metrics provided"}, 400

        # Update server's LLM logger with client's cumulative metrics
        from utils.data_persistence.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        if llm_logger is not None:
            # Update cumulative metrics (but preserve server-managed metrics like start_time and total_actions)
            server_start_time = llm_logger.cumulative_metrics.get("start_time")
            server_total_actions = llm_logger.cumulative_metrics.get("total_actions")
            server_total_tokens = llm_logger.cumulative_metrics.get("total_tokens", 0)
            server_prompt_tokens = llm_logger.cumulative_metrics.get("prompt_tokens", 0)
            server_completion_tokens = llm_logger.cumulative_metrics.get("completion_tokens", 0)
            server_cached_tokens = llm_logger.cumulative_metrics.get("cached_tokens", 0)
            server_cache_write_tokens = llm_logger.cumulative_metrics.get("cache_write_tokens", 0)
            server_total_cost = llm_logger.cumulative_metrics.get("total_cost", 0.0)
            server_total_llm_calls = llm_logger.cumulative_metrics.get("total_llm_calls", 0)
            server_milestones = llm_logger.cumulative_metrics.get("milestones")
            server_last_milestone_step = llm_logger.cumulative_metrics.get("_last_milestone_step")
            server_last_milestone_tokens = llm_logger.cumulative_metrics.get("_last_milestone_tokens")
            server_last_milestone_time = llm_logger.cumulative_metrics.get("_last_milestone_time")
            # Objectives are appended only on server (complete_direct_objective); client has no copy
            server_objectives = llm_logger.cumulative_metrics.get("objectives")
            server_last_objective_step = llm_logger.cumulative_metrics.get("_last_objective_step")
            server_last_objective_tokens = llm_logger.cumulative_metrics.get("_last_objective_tokens")
            server_last_objective_time = llm_logger.cumulative_metrics.get("_last_objective_time")
            # total_run_time and last_update_time are updated by server on take_action; run_cli never has them
            server_total_run_time = llm_logger.cumulative_metrics.get("total_run_time")
            server_last_update_time = llm_logger.cumulative_metrics.get("last_update_time")

            llm_logger.cumulative_metrics.update(cumulative_metrics)

            # Restore server-managed metrics
            if server_start_time:
                llm_logger.cumulative_metrics["start_time"] = server_start_time
            if server_total_actions is not None:
                llm_logger.cumulative_metrics["total_actions"] = server_total_actions
            # Preserve milestone tracking from server (authoritative source)
            if server_milestones is not None:
                llm_logger.cumulative_metrics["milestones"] = server_milestones
            if server_last_milestone_step is not None:
                llm_logger.cumulative_metrics["_last_milestone_step"] = server_last_milestone_step
            if server_last_milestone_tokens is not None:
                llm_logger.cumulative_metrics["_last_milestone_tokens"] = server_last_milestone_tokens
            if server_last_milestone_time is not None:
                llm_logger.cumulative_metrics["_last_milestone_time"] = server_last_milestone_time
            # Preserve objectives (only server appends on complete_direct_objective)
            if server_objectives is not None:
                llm_logger.cumulative_metrics["objectives"] = server_objectives
            if server_last_objective_step is not None:
                llm_logger.cumulative_metrics["_last_objective_step"] = server_last_objective_step
            if server_last_objective_tokens is not None:
                llm_logger.cumulative_metrics["_last_objective_tokens"] = server_last_objective_tokens
            if server_last_objective_time is not None:
                llm_logger.cumulative_metrics["_last_objective_time"] = server_last_objective_time
            # Preserve gameplay time (server updates on take_action; run_cli sync would overwrite with 0)
            if server_total_run_time is not None:
                llm_logger.cumulative_metrics["total_run_time"] = server_total_run_time
            if server_last_update_time is not None:
                llm_logger.cumulative_metrics["last_update_time"] = server_last_update_time

            # Keep cumulative totals monotonic so restored totals are never downgraded by stale client payloads.
            llm_logger.cumulative_metrics["total_tokens"] = max(
                server_total_tokens, llm_logger.cumulative_metrics.get("total_tokens", 0)
            )
            llm_logger.cumulative_metrics["prompt_tokens"] = max(
                server_prompt_tokens, llm_logger.cumulative_metrics.get("prompt_tokens", 0)
            )
            llm_logger.cumulative_metrics["completion_tokens"] = max(
                server_completion_tokens, llm_logger.cumulative_metrics.get("completion_tokens", 0)
            )
            llm_logger.cumulative_metrics["cached_tokens"] = max(
                server_cached_tokens, llm_logger.cumulative_metrics.get("cached_tokens", 0)
            )
            llm_logger.cumulative_metrics["cache_write_tokens"] = max(
                server_cache_write_tokens, llm_logger.cumulative_metrics.get("cache_write_tokens", 0)
            )
            llm_logger.cumulative_metrics["total_cost"] = max(
                server_total_cost, llm_logger.cumulative_metrics.get("total_cost", 0.0)
            )
            llm_logger.cumulative_metrics["total_llm_calls"] = max(
                server_total_llm_calls, llm_logger.cumulative_metrics.get("total_llm_calls", 0)
            )

            # Also sync to latest_metrics for stream.html display (excluding server-managed metrics)
            global latest_metrics
            with step_lock:
                for key, value in cumulative_metrics.items():
                    if key in latest_metrics and key not in [
                        "total_actions",
                        "start_time",
                        "total_run_time",
                        "total_tokens",
                        "prompt_tokens",
                        "completion_tokens",
                        "cached_tokens",
                        "cache_write_tokens",
                        "total_cost",
                        "total_llm_calls",
                    ]:
                        latest_metrics[key] = value

            with step_lock:
                latest_metrics["total_tokens"] = llm_logger.cumulative_metrics.get("total_tokens", 0)
                latest_metrics["prompt_tokens"] = llm_logger.cumulative_metrics.get("prompt_tokens", 0)
                latest_metrics["completion_tokens"] = llm_logger.cumulative_metrics.get("completion_tokens", 0)
                latest_metrics["cached_tokens"] = llm_logger.cumulative_metrics.get("cached_tokens", 0)
                latest_metrics["cache_write_tokens"] = llm_logger.cumulative_metrics.get("cache_write_tokens", 0)
                latest_metrics["total_cost"] = llm_logger.cumulative_metrics.get("total_cost", 0.0)
                latest_metrics["total_llm_calls"] = llm_logger.cumulative_metrics.get("total_llm_calls", 0)

            # Persist to cache so steps/milestones are saved promptly
            llm_logger.save_cumulative_metrics()

            logger.info(
                f"🔄 Synced LLM metrics: {cumulative_metrics.get('total_llm_calls', 0)} calls, {cumulative_metrics.get('total_tokens', 0)} tokens, ${cumulative_metrics.get('total_cost', 0):.6f}"
            )
            return {"status": "metrics_synced"}
        else:
            logger.error("No LLM logger available for sync")
            return {"status": "error", "message": "No LLM logger available"}, 500
    except Exception as e:
        logger.error(f"Error syncing LLM metrics: {e}")
        return {"status": "error", "message": str(e)}, 500


@app.post("/save_agent_history")
async def save_agent_history():
    """Save agent history to checkpoint_llm.txt (called by client after each step)"""
    try:
        # Use server-side LLM logger to save checkpoint
        from utils.data_persistence.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        if llm_logger is not None:
            # Save checkpoint using current agent step count
            global agent_step_count
            # Save to cache folder (llm_logger handles path internally now)
            llm_logger.save_checkpoint(agent_step_count=agent_step_count)
            logger.info(f"💾 Saved LLM checkpoint at step {agent_step_count}")
            return {"status": "agent_history_saved", "step_count": agent_step_count}
        else:
            return {"status": "no_logger", "message": "No LLM logger available"}

    except Exception as e:
        logger.error(f"Failed to save agent history: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/load_checkpoint")
async def load_checkpoint():
    """Load checkpoint state - called by client on startup if --load-checkpoint flag is used"""
    try:
        from utils.data_persistence.run_data_manager import get_cache_path
        checkpoint_state = str(get_cache_path("checkpoint.state"))
        
        if not os.path.exists(checkpoint_state):
            return {"status": "no_checkpoint", "message": f"No {checkpoint_state} file found"}
        
        if env:
            env.load_state(checkpoint_state)
            logger.info(f"📂 Server: Loaded checkpoint state")

            # Load milestones if available
            if env.milestone_tracker:
                try:
                    env.milestone_tracker.load_milestones_for_state(checkpoint_state)
                    logger.info(f"📂 Server: Loaded checkpoint milestones")
                except:
                    logger.warning(f"Could not load checkpoint milestones")

            return {
                "status": "checkpoint_loaded",
                "files": {
                    "state": checkpoint_state,
                    "milestones": str(get_cache_path("checkpoint_milestones.json")),
                    "map": str(get_cache_path("checkpoint_grids.json"))
                }
            }
        else:
            return {"status": "error", "message": "No emulator available"}

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {"status": "error", "message": str(e)}


def main():
    """Main function"""
    import argparse

    global running, state_update_running, state_update_thread, latest_metrics, agent_step_count

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Game Agent Server")
    parser.add_argument("--game", type=str, default="emerald", choices=["red", "emerald", "browser"],
                       help="Which game to run: 'red', 'emerald', or 'browser' (Playwright)")
    parser.add_argument("--game-url", type=str, default=None,
                       help="URL of browser game (itch.io page). Required when --game=browser")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    parser.add_argument("--manual", action="store_true", help="Enable manual mode with keyboard input and overlay")
    parser.add_argument("--load-state", type=str, help="Load a saved state file on startup")
    parser.add_argument("--record", action="store_true", help="Record video of the gameplay")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR dialogue detection")
    parser.add_argument(
        "--direct-objectives",
        type=str,
        help="Load a specific direct objective sequence ('categorized_full_game' or 'autonomous_objective_creation')",
    )
    parser.add_argument(
        "--direct-objectives-start",
        type=int,
        default=0,
        help="Start index for story objectives in legacy mode, or story objectives in categorized mode (for resuming from checkpoints)",
    )
    parser.add_argument(
        "--direct-objectives-battling-start",
        type=int,
        default=0,
        help="Start index for battling objectives (only used in categorized mode)",
    )
    # Server always runs headless - display handled by client

    args = parser.parse_args()

    # Set game type from args (also sync env var for modules like state_formatter)
    global game_type, game_url
    game_type = args.game
    os.environ["GAME_TYPE"] = game_type
    if args.game_url:
        game_url = args.game_url
        os.environ["GAME_URL"] = game_url
    elif os.environ.get("GAME_URL"):
        game_url = os.environ["GAME_URL"]
    print(f"Game type: {game_type}")
    if game_type == "browser":
        print(f"Game URL: {game_url}")
    
    # Set global direct objectives sequence
    global direct_objectives_sequence, direct_objectives_start_index, direct_objectives_battling_start_index
    if args.direct_objectives:
        direct_objectives_sequence = args.direct_objectives
        direct_objectives_start_index = args.direct_objectives_start
        direct_objectives_battling_start_index = args.direct_objectives_battling_start
        if direct_objectives_battling_start_index > 0:
            print(
                f"🎯 Direct objectives sequence: {direct_objectives_sequence} (story index: {direct_objectives_start_index}, battling index: {direct_objectives_battling_start_index})"
            )
        else:
            print(
                f"🎯 Direct objectives sequence: {direct_objectives_sequence} (starting at index {direct_objectives_start_index})"
            )

    # Check for environment variables from multiprocess mode
    env_load_state = os.environ.get("LOAD_STATE")
    if env_load_state and not args.load_state:
        args.load_state = env_load_state
        print(f"📂 Using load state from environment: {env_load_state}")
        if env_load_state == ".pokeagent_cache/checkpoint.state":
            from utils.data_persistence.run_data_manager import get_cache_path
            checkpoint_state = get_cache_path("checkpoint.state")
            if checkpoint_state.exists():
                print(f"✅ Server startup: {checkpoint_state} file exists")
            else:
                print(f"❌ Server startup: {checkpoint_state} file MISSING!")
    
    # Set checkpoint loading flag based on whether this is a true checkpoint load
    global checkpoint_loading_enabled
    env_load_checkpoint_mode = os.environ.get("LOAD_CHECKPOINT_MODE")

    if env_load_checkpoint_mode == "true":
        checkpoint_loading_enabled = True
        print("🔄 Checkpoint loading enabled - will restore LLM metrics from cumulative_metrics.json")

        # Initialize LLM logger and load checkpoint immediately during server startup
        from utils.data_persistence.llm_logger import get_llm_logger
        from utils.data_persistence.run_data_manager import get_checkpoint_llm_path
        llm_logger = get_llm_logger()
        checkpoint_file = get_checkpoint_llm_path()
        
        # First try to load lightweight cumulative metrics
        metrics_loaded = llm_logger.load_cumulative_metrics() if llm_logger else False

        # Then load full checkpoint for LLM history
        if llm_logger and checkpoint_file.exists():
            restored_step_count = llm_logger.load_checkpoint(str(checkpoint_file))
            if restored_step_count is not None:
                agent_step_count = restored_step_count
                print(f"✅ Server startup: restored LLM checkpoint with step count {restored_step_count}")

                # Sync latest_metrics with loaded cumulative metrics
                latest_metrics.update(llm_logger.cumulative_metrics)
                print(
                    f"✅ Server startup: synced metrics - actions: {latest_metrics.get('total_actions', 0)}, cost: ${latest_metrics.get('total_cost', 0):.4f}, time: {latest_metrics.get('total_run_time', 0):.0f}s"
                )
            else:
                print("❌ Server startup: failed to load LLM checkpoint")
        elif metrics_loaded:
            # Only metrics file loaded (no checkpoint)
            latest_metrics.update(llm_logger.cumulative_metrics)
            print(
                f"✅ Server startup: loaded cumulative metrics - actions: {latest_metrics.get('total_actions', 0)}, cost: ${latest_metrics.get('total_cost', 0):.4f}, time: {latest_metrics.get('total_run_time', 0):.0f}s"
            )
        else:
            print("ℹ️ Server startup: no checkpoint_llm.txt file found")
    elif env_load_checkpoint_mode == "false":
        checkpoint_loading_enabled = False
        print("✨ Fresh start mode - will NOT load LLM metrics from cumulative_metrics.json")
    else:
        # Default behavior: allow checkpoint loading unless explicitly disabled
        checkpoint_loading_enabled = True
        print(
            "🔄 Checkpoint loading enabled by default - will restore LLM metrics from cumulative_metrics.json if available"
        )

    # ALWAYS try to load cumulative metrics, regardless of checkpoint mode
    # This ensures tokens/cost/actions are preserved even if checkpoint loading is off
    if env_load_checkpoint_mode != "true":
        from utils.data_persistence.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        if llm_logger:
            # Only load if we didn't already load above
            metrics_loaded = llm_logger.load_cumulative_metrics()
            if metrics_loaded:
                # latest_metrics is already declared global above, so we can use it here
                latest_metrics.update(llm_logger.cumulative_metrics)
                print(
                    f"✅ Server startup: loaded cumulative metrics - tokens: {latest_metrics.get('total_tokens', 0):,}, cost: ${latest_metrics.get('total_cost', 0):.2f}, actions: {latest_metrics.get('total_actions', 0):,}"
                )

    print("Starting Fixed Simple Pokemon Emerald Server")
    # Initialize run data manager for structured data collection
    # Use run_id from environment if provided (set by client), otherwise create new one
    from utils.data_persistence.run_data_manager import initialize_run_data_manager

    run_id = os.environ.get("RUN_DATA_ID")
    run_name = os.environ.get("RUN_NAME")
    run_manager = initialize_run_data_manager(run_id=run_id, run_name=run_name)
    print(f"📁 Run data directory: {run_manager.get_run_directory()}")

    # Only save metadata if this is a new run (not set by client)
    # If run_id was provided, metadata was already saved by client
    if run_id is None:
        command_args = vars(args)
        run_manager.save_metadata(
            command_args=command_args,
            sys_argv=sys.argv,
            additional_info={"server_mode": True, "checkpoint_loading_enabled": checkpoint_loading_enabled},
        )

    # Initialize run directory for this execution (deprecated, will be moved)
    global current_run_dir
    if current_run_dir is None:
        from utils.data_persistence.run_data_manager import get_cache_directory
        # Use the cache directory directly instead of creating nested subdirectory
        current_run_dir = str(get_cache_directory())
        print(f"📁 Legacy run directory (deprecated): {current_run_dir}")

    # Initialize video recording if requested
    init_video_recording(args.record)
    print("Server mode - headless operation, display handled by client")
    if args.no_ocr:
        print("OCR dialogue detection disabled")
    print("Press Ctrl+C to stop")

    # Initialize environment (emulator or browser)
    if not setup_environment(skip_initial_state=(args.load_state is not None)):
        print("Failed to initialize environment")
        return

    # Disable dialogue detection if --no-ocr flag is set (emulator-only)
    if args.no_ocr and game_type != "browser":
        if env and hasattr(env, 'memory_reader') and env.memory_reader and hasattr(env.memory_reader, '_dialog_detection_enabled'):
            env.memory_reader._dialog_detection_enabled = False
            print("🚫 All dialogue detection disabled (--no-ocr flag)")

    # Load state if specified
    # State loading and milestone tracking (emulator-only)
    if game_type != "browser":
        if args.load_state:
            try:
                env.load_state(args.load_state)
                print(f"Loaded state from: {args.load_state}")

                # Milestones and map data are automatically loaded by env.load_state()
                # Check what was loaded
                state_dir = os.path.dirname(args.load_state)
                base_name = os.path.splitext(os.path.basename(args.load_state))[0]

                milestone_file = os.path.join(state_dir, f"{base_name}_milestones.json")
                if os.path.exists(milestone_file):
                    print(f"📂 Loaded milestones from: {milestone_file}")

                grids_file = os.path.join(state_dir, f"{base_name}_grids.json")
                if os.path.exists(grids_file):
                    print(f"🗺️  Loaded map grids from: {grids_file}")

                # Map buffer should already be found by emulator.load_state() (Emerald only)
                if env.memory_reader and getattr(env.memory_reader, '_map_buffer_addr', None):
                    print(f"Map buffer already initialized at 0x{env.memory_reader._map_buffer_addr:08X}")

                # Mark GAME_RUNNING milestone after state load
                # Defer expensive state logging - it will happen on first state request
                try:
                    env.milestone_tracker.mark_completed("GAME_RUNNING")
                    print("GAME_RUNNING milestone marked after state load - initial logging deferred")
                except Exception as e:
                    print(f"Warning: Could not mark initial milestone: {e}")
            except Exception as e:
                print(f"Failed to load state from {args.load_state}: {e}")
                print("Continuing with fresh game state...")

        # Start lightweight milestone updater thread
        state_update_running = True
        state_update_thread = threading.Thread(target=periodic_milestone_updater, daemon=True)
        state_update_thread.start()

    # Start FastAPI server in background thread
    server_thread = threading.Thread(target=run_fastapi_server, args=(args.port,), daemon=True)
    server_thread.start()

    # Get local IP for network access
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
    except Exception:
        local_ip = "127.0.0.1"

    print(f"🌐 FastAPI server running:")
    print(f"   Local: http://localhost:{args.port}")
    print(f"   Network: http://{local_ip}:{args.port}")
    print(f"📺 Stream interface: http://{local_ip}:{args.port}/stream")

    # Initialize video recording AFTER FastAPI server starts
    # Try Playwright WebUI recording first; fall back to frame-based recording
    if args.record:
        pw_success = init_playwright_recording(args.port, run_id=run_id)
        if not pw_success:
            init_video_recording(True)
    else:
        # No recording requested
        pass
    print("Available endpoints:")
    print("  /status - Server status")
    print("  /screenshot - Current screenshot")
    print("  /action - Take action (POST)")
    print("  /state - Comprehensive game state (visual + memory data)")
    print("  /agent - Agent thinking status")
    print("  /milestones - Current milestones achieved")
    print("  /recent_actions - Recently pressed buttons")
    print("  /debug/memory - Debug memory reading (basic)")
    print("  /debug/memory/comprehensive - Comprehensive memory diagnostics")
    print("  /debug/milestones - Debug milestone tracking system")
    print("  /debug/reset_milestones - Reset all milestones (POST)")
    print("  /debug/test_milestone_operations - Test milestone save/load (POST)")
    print("  /stop - Stop server")

    try:
        # Run headless game loop in main thread
        game_loop(manual_mode=False)  # Server always runs in server mode
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Backup cleanup (in case signal handler didn't run or failed)
        # The signal handler should handle most finalization, but this is a safety net
        # Note: running and state_update_running are already declared as global at start of main()
        was_running = running  # Read value before modifying
        running = False
        state_update_running = False

        # Only run backup if we weren't already shutting down
        # (signal handler should have handled it, but check if it actually did)
        if was_running:
            print("📦 Running backup finalization in finally block...")
            try:
                from utils.data_persistence.run_data_manager import get_run_data_manager

                run_manager = get_run_data_manager()
                if run_manager:
                    # Quick finalization - just save end state snapshot
                    run_manager.save_end_state_snapshot()
                    print("✅ Backup finalization completed")
            except Exception as e:
                logger.error(f"❌ Error in backup finalization: {e}", exc_info=True)

        if env:
            env.stop()
        print("Server stopped")


# Initialize emulator when imported for multiprocess mode
def init_for_multiprocess():
    """Initialize emulator when server is imported for multiprocess mode"""
    global env

    if env is None:  # Only initialize once
        # Check for environment variables set by agent.py multiprocess mode
        rom_path = os.environ.get("ROM_PATH", "Emerald-GBAdvance/rom.gba")
        load_state = os.environ.get("LOAD_STATE")
        record_video = os.environ.get("RECORD_VIDEO") == "1"
        no_ocr = os.environ.get("NO_OCR") == "1"

        print(f"🔧 Initializing server for multiprocess mode...")
        print(f"   ROM: {rom_path}")
        if load_state:
            print(f"   Load state: {load_state}")

        # Initialize emulator
        try:
            if not os.path.exists(rom_path):
                raise RuntimeError(f"ROM not found at {rom_path}")

            if game_type == "red":
                from pokemon_red_env.red_emulator import RedEmulator
                env = RedEmulator(rom_path=rom_path)
            else:
                from pokemon_env.emulator import EmeraldEmulator
                env = EmeraldEmulator(rom_path=rom_path)
            env.initialize()

            # Initialize video recording if requested
            init_video_recording(record_video)

            # Disable OCR if requested
            if no_ocr and env and env.memory_reader and hasattr(env.memory_reader, '_dialog_detection_enabled'):
                env.memory_reader._dialog_detection_enabled = False
                print("🚫 All dialogue detection disabled (--no-ocr flag)")

            # Load state if specified
            if load_state:
                try:
                    print(f"🔄 Attempting to load state from: {load_state}")
                    env.load_state(load_state)
                    print(f"📂 Successfully loaded state from: {load_state}")

                    # Milestones and map data are automatically loaded by env.load_state()
                    # Check what was loaded
                    state_dir = os.path.dirname(load_state)
                    base_name = os.path.splitext(os.path.basename(load_state))[0]

                    milestone_file = os.path.join(state_dir, f"{base_name}_milestones.json")
                    if os.path.exists(milestone_file):
                        print(f"📋 Loaded milestones from: {milestone_file}")

                    grids_file = os.path.join(state_dir, f"{base_name}_grids.json")
                    if os.path.exists(grids_file):
                        print(f"🗺️  Loaded map grids from: {grids_file}")

                    # Map buffer should already be found by emulator.load_state() (Emerald only)
                    if env.memory_reader and getattr(env.memory_reader, '_map_buffer_addr', None):
                        print(f"📍 Map buffer initialized at 0x{env.memory_reader._map_buffer_addr:08X}")

                    print(f"✅ State loading complete for {load_state}")

                except Exception as e:
                    print(f"❌ Failed to load state from {load_state}: {e}")
                    print("   Continuing with fresh game state...")

            # Start lightweight milestone updater thread
            global state_update_running, state_update_thread
            state_update_running = True
            state_update_thread = threading.Thread(target=periodic_milestone_updater, daemon=True)
            state_update_thread.start()

            print("✅ Server initialized successfully for multiprocess mode")

        except Exception as e:
            print(f"❌ Failed to initialize server for multiprocess mode: {e}")
            raise


# Auto-initialize when imported for multiprocess mode (when ROM_PATH env var is set)
if os.environ.get("ROM_PATH") and __name__ != "__main__":
    init_for_multiprocess()

if __name__ == "__main__":
    main()
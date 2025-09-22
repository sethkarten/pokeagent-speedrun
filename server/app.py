#!/usr/bin/env python3
"""
Fixed Simple Pokemon Emerald server - headless FastAPI server
"""

# Standard library imports
import base64
import datetime
import glob
import io
import json
import logging
import os
import signal
import sys
import threading
import time

# Third-party imports
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

# Add parent directory to path for local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local application imports
from pokemon_env.emulator import EmeraldEmulator
from utils.anticheat import AntiCheatTracker

# Set up logging - reduced verbosity for multiprocess mode
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Global state
env = None
anticheat_tracker = None  # AntiCheat tracker for submission logging
step_counter = 0  # Track steps for submission logging
last_action_time = None  # Track time of last action for decision time calculation
running = True
step_count = 0
agent_step_count = 0  # Track agent steps separately from frame steps
current_obs = None
fps = 80

# Performance monitoring
last_fps_log = time.time()
frame_count_since_log = 0
action_queue = []  # Queue for multi-action sequences
current_action = None  # Current action being held
action_frames_remaining = 0  # Frames left to hold current action
release_frames_remaining = 0  # Frames left to wait after release

### IMPORTANT: DO NOT REDUCE THESE OR BUTTONS MAY NOT WORK! ###
ACTION_HOLD_FRAMES = 12   # Hold each action for 12 frames 
ACTION_RELEASE_DELAY = 24   # Delay between actions for processing

# Video recording state
video_writer = None
video_recording = False
video_filename = ""
video_frame_counter = 0
video_frame_skip = 4  # Record every 4th frame (120/4 = 30 FPS)

# Frame cache for separate frame server
# Use cache directory instead of /tmp
CACHE_DIR = ".pokeagent_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
FRAME_CACHE_FILE = os.path.join(CACHE_DIR, "frame_cache.json")
frame_cache_counter = 0

# Server runs headless - display handled by client

# Threading locks for thread safety
obs_lock = threading.Lock()
step_lock = threading.Lock()
memory_lock = threading.Lock()  # New lock for memory operations to prevent race conditions

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
        # Create video filename with timestamp
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

def update_frame_cache(screenshot):
    """Update the frame cache file for the separate frame server"""
    global frame_cache_counter, FRAME_CACHE_FILE
    
    if screenshot is None:
        return
        
    try:
        # Convert screenshot to base64
        if hasattr(screenshot, 'save'):  # PIL image
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
        elif isinstance(screenshot, np.ndarray):  # Numpy array
            pil_image = Image.fromarray(screenshot)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
        else:
            return
            
        frame_cache_counter += 1
        
        # Write to cache file atomically
        cache_data = {
            "frame_data": img_str,
            "frame_counter": frame_cache_counter,
            "timestamp": time.time()
        }
        
        # Write to temporary file first, then move (atomic operation)
        temp_file = FRAME_CACHE_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(cache_data, f)
        os.rename(temp_file, FRAME_CACHE_FILE)
        
    except Exception as e:
        pass  # Silently handle cache write errors

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
        import logging
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
    buttons: list = []  # List of button names: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT

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
                        basic_state = {
                            "player": {
                                "money": env.get_money(),
                                "party_size": len(env.get_party_pokemon() or []),
                                "position": env.get_coordinates()
                            },
                            "map": {
                                "location": env.get_location()
                            }
                        }
                        env.check_and_update_milestones(basic_state)
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
    global running, state_update_running
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False
    state_update_running = False
    cleanup_video_recording()
    if env:
        env.stop()
    sys.exit(0)

def setup_environment(skip_initial_state=False):
    """Initialize the emulator"""
    global env, current_obs, anticheat_tracker
    
    try:
        rom_path = "Emerald-GBAdvance/rom.gba"
        if not os.path.exists(rom_path):
            raise RuntimeError(f"ROM not found at {rom_path}")
        
        env = EmeraldEmulator(rom_path=rom_path)
        env.initialize()
        
        # Initialize AntiCheat tracker for submission logging
        anticheat_tracker = AntiCheatTracker()
        anticheat_tracker.initialize_submission_log("SERVER_MODE")
        print("AntiCheat tracker initialized for submission logging")
        
        # Log initial GAME_RUNNING milestone at startup (STEP=0, time=0)
        # Skip this if we're going to load a state anyway
        if not skip_initial_state:
            try:
                # Mark GAME_RUNNING milestone as completed immediately
                env.milestone_tracker.mark_completed("GAME_RUNNING")
                
                # Get initial game state for logging
                initial_state = env.get_comprehensive_state()
                
                # Create state hash
                import hashlib
                state_str = str(initial_state)
                state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]
                
                # Log initial entry with GAME_RUNNING milestone
                anticheat_tracker.log_submission_data(
                    step=0,
                    state_data=initial_state,
                    action_taken="INIT",
                    decision_time=0.0,
                    state_hash=state_hash,
                    manual_mode=True,
                    milestone_override="GAME_RUNNING"
                )
                print("Initial GAME_RUNNING milestone logged at startup")
                
            except Exception as e:
                print(f"Warning: Could not log initial milestone: {e}")
        
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
        print(f"Failed to initialize emulator: {e}")
        return False

def handle_input(manual_mode=False):
    """Handle input - server runs headless, no input handling needed"""
    # Server always runs headless - input handled by client via HTTP API
    return True, []

def step_environment(actions_pressed):
    """Take a step in the environment with optimized locking for better performance"""
    global current_obs
    
    # Debug: print what actions are being sent to emulator
    if actions_pressed:
        print(f"🎯 DEBUG: Stepping emulator with actions: {actions_pressed}")
    
    
    # Only use memory_lock for the essential emulator step
    with memory_lock:
        env.run_frame_with_buttons(actions_pressed)
        
        # Do lightweight area transition detection inside the lock
        if hasattr(env, 'memory_reader') and env.memory_reader:
            try:
                transition_detected = env.memory_reader._check_area_transition()
                if transition_detected:
                    logger.info("Area transition detected")
                    env.memory_reader.invalidate_map_cache()
                    if hasattr(env.memory_reader, '_cached_behaviors'):
                        env.memory_reader._cached_behaviors = None
                    if hasattr(env.memory_reader, '_cached_behaviors_map_key'):
                        env.memory_reader._cached_behaviors_map_key = None
                    # Set flag to trigger map stitcher update outside the lock
                    env.memory_reader._area_transition_detected = True
            except Exception as e:
                logger.warning(f"Area transition check failed: {e}")
    
    # Update screenshot outside the memory lock to reduce contention
    try:
        screenshot = env.get_screenshot()
        if screenshot:
            record_frame(screenshot)
            update_frame_cache(screenshot)  # Update frame cache for separate frame server
            with obs_lock:
                current_obs = np.array(screenshot)
                
            # Update map stitcher on position changes (lightweight approach)
            # This ensures map data stays current as player moves
            if hasattr(env, 'memory_reader') and env.memory_reader:
                try:
                    # Check if player position has changed
                    should_update = False
                    
                    # Get current player coordinates and map info
                    current_coords = env.memory_reader.read_coordinates()
                    current_map_bank = env.memory_reader._read_u8(env.memory_reader.addresses.MAP_BANK)
                    current_map_number = env.memory_reader._read_u8(env.memory_reader.addresses.MAP_NUMBER)
                    current_map_info = (current_map_bank, current_map_number)
                    
                    # Initialize tracking variables if needed
                    if not hasattr(env, '_last_player_coords'):
                        env._last_player_coords = None
                        env._last_map_info = None
                    
                    # Check for position changes
                    if current_coords != env._last_player_coords or current_map_info != env._last_map_info:
                        should_update = True
                        env._last_player_coords = current_coords
                        env._last_map_info = current_map_info
                        print(f"📍 Position change detected: {current_coords}, map: {current_map_info}")
                        logger.debug(f"Map stitcher update triggered by position change: {current_coords}, map: {current_map_info}")
                    
                    # Always update on area transitions (already detected above)
                    if hasattr(env.memory_reader, '_area_transition_detected') and env.memory_reader._area_transition_detected:
                        should_update = True
                        env.memory_reader._area_transition_detected = False  # Reset flag
                        logger.debug("Map stitcher update triggered by area transition")
                    
                    # Update map stitcher directly when position changes
                    if should_update:
                        # @TODO should do location change warps here too
                        print(f"🗺️ Triggering map stitcher update for position change")
                        # Call map stitcher update directly without full map reading
                        tiles = env.memory_reader.read_map_around_player(radius=7)
                        if tiles:
                            print(f"🗺️ Got {len(tiles)} tiles, updating map stitcher")
                            state = {"map": {}}  # Basic state for stitcher
                            env.memory_reader._update_map_stitcher(tiles, state)
                            logger.debug("Map stitcher updated for position change")
                            print(f"✅ Map stitcher update completed")
                        else:
                            print(f"❌ No tiles found for map stitcher update")
                        
                except Exception as e:
                    logger.error(f"Failed to update map stitcher during movement: {e}")
                    print(f"❌ Map stitcher update failed: {e}")
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
    
    print("Starting headless game loop...")
    
    while running:
        # Handle input
        should_continue, actions_pressed = handle_input(manual_mode)
        if not should_continue:
            break
            
        # In server mode, handle action queue with proper button hold timing
        action_completed = False
        if not manual_mode:
            global current_action, action_frames_remaining, release_frames_remaining
            
            if current_action and action_frames_remaining > 0:
                # Continue holding the current action
                actions_pressed = [current_action]
                action_frames_remaining -= 1
                if action_frames_remaining == 0:
                    # Action finished, start release delay
                    current_action = None
                    release_frames_remaining = ACTION_RELEASE_DELAY
                    action_completed = True  # Mark action as completed
                    print(f"✅ Action completed: step_count will increment")
            elif release_frames_remaining > 0:
                # Release delay (no button pressed)
                actions_pressed = []
                release_frames_remaining -= 1
            elif action_queue:
                # Start a new action from the queue
                current_action = action_queue.pop(0)
                action_frames_remaining = ACTION_HOLD_FRAMES
                actions_pressed = [current_action]
                queue_len = len(action_queue)
                # Get current FPS for estimation
                current_fps_for_calc = env.get_current_fps(fps) if env else fps
                estimated_time = queue_len * (ACTION_HOLD_FRAMES + ACTION_RELEASE_DELAY) / current_fps_for_calc
                print(f"🎮 Server processing action: {current_action}, Queue remaining: {queue_len} actions (~{estimated_time:.1f}s)")
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
        # Server runs headless - always use sleep for timing
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
        timeout_graceful_shutdown=30  # More time for graceful shutdown
    )

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
@app.get("/health")
async def get_health():
    """Health check endpoint for server monitoring"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/status")
async def get_status():
    """Get server status"""
    with step_lock:
        current_step = step_count
    
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
        "fps_multiplier": 2 if is_dialog else 1
    }

@app.get("/screenshot")
async def get_screenshot():
    """Get current screenshot"""
    global current_obs, step_count
    
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    with obs_lock:
        obs_copy = current_obs.copy() if current_obs is not None else None
    
    if obs_copy is None:
        raise HTTPException(status_code=500, detail="No screenshot available")
    
    try:
        # Convert numpy array to PIL image
        pil_image = Image.fromarray(obs_copy)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        with step_lock:
            current_step = step_count
        
        return GameStateResponse(
            screenshot_base64=img_str,
            step_number=current_step,
            resolution=[obs_copy.shape[1], obs_copy.shape[0]],
            status="running"
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
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {"frame": img_str}
    except Exception as e:
        logger.warning(f"Frame endpoint: Error encoding frame: {e}")
        return {"frame": ""}

@app.post("/action")
async def take_action(request: ActionRequest):
    """Take an action"""
    global current_obs, step_count, recent_button_presses, action_queue, anticheat_tracker, step_counter, last_action_time
    
    print(f"🔍 DEBUG: Action endpoint called with request: {request}")
    print(f"🔍 DEBUG: Request buttons: {request.buttons}")
    
    if env is None:
        print(f"❌ DEBUG: Emulator not initialized")
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Add all actions to the queue (handle both single actions and lists)
        if request.buttons:
            # Add ALL actions to the queue - let the game loop handle execution
            print(f"📡 Server received actions: {request.buttons}")
            print(f"📋 Action queue before extend: {action_queue}")
            action_queue.extend(request.buttons)
            print(f"📋 Action queue after extend: {action_queue}")
            
            # Track button presses for recent actions display
            current_time = time.time()
            for button in request.buttons:
                # Add all buttons to recent actions (removed duplicate filtering for debugging)
                recent_button_presses.append({
                    "button": button,
                    "timestamp": current_time
                })
            
            # Update total actions count in metrics
            with step_lock:
                latest_metrics["total_actions"] = latest_metrics.get("total_actions", 0) + len(request.buttons)
                
                # Also update the LLM logger's action count for checkpoint persistence
                try:
                    from utils.llm_logger import get_llm_logger
                    llm_logger = get_llm_logger()
                    if llm_logger:
                        llm_logger.cumulative_metrics["total_actions"] = latest_metrics["total_actions"]
                        
                        # Sync LLM logger's cumulative metrics back to latest_metrics
                        # This ensures token usage and costs from LLM interactions are displayed
                        cumulative_metrics_to_sync = ["total_tokens", "prompt_tokens", "completion_tokens", "total_cost", "total_llm_calls", "total_run_time"]
                        for metric_key in cumulative_metrics_to_sync:
                            if metric_key in llm_logger.cumulative_metrics:
                                latest_metrics[metric_key] = llm_logger.cumulative_metrics[metric_key]
                except Exception as e:
                    logger.debug(f"Failed to sync metrics with LLM logger: {e}")
            
            # Keep only last 50 button presses to avoid memory issues
            if len(recent_button_presses) > 50:
                recent_button_presses = recent_button_presses[-50:]
        else:
            print(f"⚠️ DEBUG: No buttons in request")
        
        # DON'T execute action here - let the game loop handle it from the queue
        # This prevents conflicts between the API thread and pygame thread
        
        # Return immediate success - avoid all locks to prevent deadlocks
        actions_added = len(request.buttons) if request.buttons else 0
        
        print(f"✅ DEBUG: Returning success, actions_added: {actions_added}, queue_length: {len(action_queue)}")
        
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
                
                # Get current game state for logging
                game_state = env.get_comprehensive_state()
                action_taken = request.buttons[0] if request.buttons else "NONE"  # Log first action
                
                # Create simple state hash
                import hashlib
                state_str = str(game_state)
                state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]
                
                # Determine if this is manual mode (from client) or agent mode
                # For now, assume manual mode if coming through API
                manual_mode = request.source == "manual" if hasattr(request, 'source') else True
                
                # Get the latest milestone from the emulator's milestone tracker
                # First, trigger an immediate milestone check to ensure current state is detected
                latest_milestone = "NONE"
                if env and hasattr(env, 'milestone_tracker'):
                    try:
                        # Force an immediate milestone check before logging
                        env.check_and_update_milestones(game_state)
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
                    milestone_override=latest_milestone
                )
                
            except Exception as e:
                logger.warning(f"Error logging to submission.log: {e}")
        
        # Return lightweight response without any lock acquisition
        return {
            "status": "success", 
            "actions_queued": actions_added,
            "queue_length": len(action_queue),  # action_queue access is atomic for lists
            "message": f"Added {actions_added} actions to queue"
        }
            
    except Exception as e:
        print(f"❌ DEBUG: Exception in action endpoint: {e}")
        logger.error(f"Error taking action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue_status")
async def get_queue_status():
    """Get action queue status"""
    global action_queue, current_action, action_frames_remaining, release_frames_remaining
    
    queue_empty = (len(action_queue) == 0 and 
                   current_action is None and 
                   action_frames_remaining == 0 and 
                   release_frames_remaining == 0)
    
    return {
        "queue_empty": queue_empty,
        "queue_length": len(action_queue),
        "current_action": current_action,
        "action_frames_remaining": action_frames_remaining,
        "release_frames_remaining": release_frames_remaining
    }

@app.get("/state")
async def get_comprehensive_state():
    """Get comprehensive game state including visual and memory data"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
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
        if env and env.memory_reader and hasattr(env.memory_reader, '_map_stitcher'):
            map_stitcher = env.memory_reader._map_stitcher
            num_areas = len(map_stitcher.map_areas) if map_stitcher and hasattr(map_stitcher, 'map_areas') else 0
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
        
        # Check if visual_map was already generated by memory_reader
        # If so, preserve it as it has the proper accumulated map data
        visual_map_from_memory_reader = state.get("map", {}).get("visual_map")
        if visual_map_from_memory_reader:
            logger.debug("Using visual_map generated by memory_reader")
            # Keep the visual_map as-is
        elif map_stitcher:
            # Generate visual map if not already present
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
                            connections_with_coords.append({
                                "to": other_loc,
                                "from_pos": list(my_coords) if my_coords else [],
                                "to_pos": list(their_coords) if their_coords else []
                            })
                
                # Generate the map display
                map_lines = map_stitcher.generate_location_map_display(
                    location_name=current_location,
                    player_pos=player_coords,
                    npcs=npcs,
                    connections=connections_with_coords
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
                    connections.append({
                        "to": other_loc,
                        "from_pos": list(my_coords),
                        "to_pos": list(their_coords)
                    })
                
                state["map"]["stitched_map_info"] = {
                    "available": True,
                    "current_area": {
                        "name": current_location,
                        "connections": connections,
                        "player_pos": player_coords
                    },
                    "player_local_pos": player_coords
                }
            else:
                state["map"]["stitched_map_info"] = {
                    "available": False,
                    "reason": "Unknown location"
                }
            
            # Also include location connections directly for backward compatibility
            try:
                cache_file = ".pokeagent_cache/map_stitcher_data.json"
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        map_data = json.load(f)
                        if 'location_connections' in map_data and map_data['location_connections']:
                            location_connections = map_data['location_connections']
                            state["location_connections"] = location_connections
                            logger.debug(f"Loaded location connections for {len(location_connections) if location_connections else 0} locations")
                        elif 'warp_connections' in map_data and map_data['warp_connections']:
                            # Convert warp_connections to portal_connections format for LLM display
                            map_id_connections = {}
                            for conn in map_data['warp_connections']:
                                from_map = conn['from_map_id']
                                if from_map not in map_id_connections:
                                    map_id_connections[from_map] = []
                                
                                # Find the location name for the destination map
                                to_map_name = "Unknown Location"
                                if str(conn['to_map_id']) in map_data.get('map_areas', {}):
                                    to_map_name = map_data['map_areas'][str(conn['to_map_id'])]['location_name']
                                
                                map_id_connections[from_map].append({
                                    'to_name': to_map_name,
                                    'from_pos': conn['from_position'],  # Keep as list for JSON serialization
                                    'to_pos': conn['to_position']       # Keep as list for JSON serialization
                                })
                            
                            state["portal_connections"] = map_id_connections
                            print(f"🗺️ SERVER: Added portal connections to state: {map_id_connections}")
                            print(f"🗺️ SERVER: State now has keys: {list(state.keys())}")
                            logger.debug(f"Loaded portal connections for {len(map_id_connections) if map_id_connections else 0} maps from persistent storage")
                        else:
                            print(f"🗺️ SERVER: No warp connections found in map data")
                            logger.debug("No warp connections found in map stitcher data")
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
        if state["visual"]["screenshot"]:
            buffer = io.BytesIO()
            state["visual"]["screenshot"].save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            state["visual"]["screenshot_base64"] = img_str
            # Remove the PIL image object to avoid serialization issues
            del state["visual"]["screenshot"]
        
        with step_lock:
            current_step = step_count
        
        # Include action queue info for multiprocess coordination
        queue_length = len(action_queue)  # Action queue access is atomic for len()
        
        return ComprehensiveStateResponse(
            visual=state["visual"],
            player=state["player"],
            game=state["game"],
            map=state["map"],
            milestones=state.get("milestones", {}),
            location_connections=state.get("location_connections", {}),
            step_number=current_step,
            status="running",
            action_queue_length=queue_length
        )
        
    except Exception as e:
        logger.error(f"Error getting comprehensive state: {e}")
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
            
            diagnostics.update({
                'party_size': party_size,
                'coordinates': coordinates,
                'money': money,
                'is_in_battle': is_in_battle,
                'game_state': game_state,
                'player_name': player_name,
                'battle_detection': {
                    'address': f'0x{battle_addr:08x}' if 'battle_addr' in locals() else 'unknown',
                    'raw_value': f'0x{battle_raw_value:02x}' if battle_raw_value is not None else 'error',
                    'mask': f'0x{battle_mask:02x}' if battle_mask is not None else 'unknown',
                    'result': battle_result
                },
                'working_reads': True
            })
        except Exception as read_error:
            diagnostics['read_error'] = str(read_error)
            diagnostics['working_reads'] = False
        
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
            "raw_bytes": [b for b in memory_bytes[:100]]  # First 100 bytes as numbers
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
    import asyncio
    
    async def event_stream():
        """Generate server-sent events for agent thinking"""
        logger.info("SSE: Starting event stream")
        last_timestamp = ""  # Track last seen timestamp instead of count
        sent_timestamps = set()  # Track all sent timestamps to avoid duplicates
        heartbeat_counter = 0
        
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'status': 'connected', 'timestamp': time.time()})}\n\n"
            
            # On startup, mark all existing interactions as "sent" to avoid flooding with old messages
            # We only want to stream NEW interactions from this point forward
            try:
                log_files = sorted(glob.glob("llm_logs/llm_log_*.jsonl"))
                for log_file in log_files:
                    if os.path.exists(log_file):
                        with open(log_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    entry = json.loads(line.strip())
                                    if entry.get("type") == "interaction":
                                        timestamp = entry.get("timestamp", "")
                                        if timestamp:
                                            sent_timestamps.add(timestamp)
                                except:
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
                        # Read LLM log files directly (same as working /agent endpoint)
                        log_files = sorted(glob.glob("llm_logs/llm_log_*.jsonl"))
                        
                        # Check all recent log files for new entries
                        for log_file in log_files[-2:]:  # Check last 2 files to catch session changes
                            if os.path.exists(log_file):
                                with open(log_file, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                    # Check all lines, not just last 5
                                    for line in lines:
                                        try:
                                            entry = json.loads(line.strip())
                                            if entry.get("type") == "interaction":
                                                timestamp = entry.get("timestamp", "")
                                                # Only add if we haven't sent this timestamp before
                                                if timestamp and timestamp not in sent_timestamps:
                                                    new_interactions.append({
                                                        "type": entry.get("interaction_type", "unknown"),
                                                        "response": entry.get("response", ""),
                                                        "duration": entry.get("duration", 0),
                                                        "timestamp": timestamp
                                                    })
                                        except:
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
                            
                            event_data = {
                                "step": current_step,
                                "type": interaction.get("type", "unknown"),
                                "response": interaction.get("response", ""),
                                "duration": interaction.get("duration", 0),
                                "timestamp": interaction.get("timestamp", ""),
                                "is_new": True
                            }
                            
                            yield f"data: {json.dumps(event_data)}\n\n"
                            # Mark this timestamp as sent
                            sent_timestamps.add(interaction.get("timestamp", ""))
                    
                    # Send periodic heartbeat to keep connection alive (every 10 cycles = 5 seconds)
                    elif heartbeat_counter % 10 == 0:
                        yield f"data: {json.dumps({'heartbeat': True, 'timestamp': time.time(), 'step': current_step})}\n\n"
                    
                    # Wait before checking again
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"SSE: Error in stream loop: {e}")
                    yield f"data: {json.dumps({'error': str(e), 'timestamp': time.time()})}\n\n"
                    await asyncio.sleep(2)
                    
        except Exception as outer_e:
            logger.error(f"SSE: Fatal error in event stream: {outer_e}")
            yield f"data: {json.dumps({'fatal_error': str(outer_e), 'timestamp': time.time()})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

@app.get("/agent")
async def get_agent_thinking():
    """Get current agent thinking status and recent LLM interactions"""
    try:
        # Get the most recent LLM log file
        from utils.llm_logger import get_llm_logger
        
        # Get recent LLM interactions
        llm_logger = get_llm_logger()
        session_summary = llm_logger.get_session_summary()
        
                # Find all LLM log files and get interactions from all of them
        import glob
        log_files = glob.glob("llm_logs/llm_log_*.jsonl")
        logger.info(f"Found {len(log_files)} log files: {log_files}")
        
        # Get recent interactions from all log files
        recent_interactions = []
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Get interactions from this file
                        for line in lines:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("type") == "interaction":
                                    recent_interactions.append({
                                        "type": entry.get("interaction_type", "unknown"),
                                        "prompt": entry.get("prompt", ""),
                                        "response": entry.get("response", ""),
                                        "duration": entry.get("duration", 0),
                                        "timestamp": entry.get("timestamp", "")
                                    })
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
            current_thought += f"{interaction['type'].upper()} ({interaction['duration']:.2f}s)\n"
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

@app.get("/metrics")
async def get_metrics():
    """Get cumulative metrics for the run"""
    global latest_metrics
    
    try:
        # Return the latest metrics received from client (with thread safety)
        with step_lock:
            metrics = latest_metrics.copy()
            metrics["agent_step_count"] = agent_step_count
        
        # If metrics haven't been initialized by client yet, try to load from checkpoint
        # BUT only if checkpoint loading is enabled (not for fresh starts with --load-state)
        if metrics.get("total_llm_calls", 0) == 0 and checkpoint_loading_enabled:
            # Check cache folder first, then fall back to old location
            cache_dir = ".pokeagent_cache"
            checkpoint_file = os.path.join(cache_dir, "checkpoint_llm.txt") if os.path.exists(cache_dir) else "checkpoint_llm.txt"
            if not os.path.exists(checkpoint_file) and os.path.exists("checkpoint_llm.txt"):
                checkpoint_file = "checkpoint_llm.txt"
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                        if "cumulative_metrics" in checkpoint_data:
                            checkpoint_metrics = checkpoint_data["cumulative_metrics"]
                            metrics.update(checkpoint_metrics)
                            
                            # Recalculate total_run_time based on original start_time
                            if "start_time" in checkpoint_metrics:
                                metrics["total_run_time"] = time.time() - checkpoint_metrics["start_time"]
                            
                            # Update agent step count from checkpoint
                            if "agent_step_count" in checkpoint_data:
                                metrics["agent_step_count"] = checkpoint_data["agent_step_count"]
                except:
                    pass
        
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
            "agent_step_count": agent_step_count
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
    "start_time": time.time()  # Will be overwritten if checkpoint is loaded
}

# Flag to track whether checkpoint loading should be enabled
checkpoint_loading_enabled = True  # Will be set based on startup args

@app.post("/reset_metrics")
async def reset_metrics():
    """Reset all metrics to zero for fresh start"""
    global latest_metrics, agent_step_count, checkpoint_loading_enabled
    
    with step_lock:
        latest_metrics.update({
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "total_actions": 0,
            "total_run_time": 0,
            "total_llm_calls": 0,
            "start_time": time.time()
        })
        agent_step_count = 0
        # Disable checkpoint loading to prevent loading from checkpoint_llm.txt
        checkpoint_loading_enabled = False
    
    print("🔄 Server metrics reset for fresh start - checkpoint loading disabled")
    return {"status": "reset", "timestamp": time.time()}

@app.post("/agent_step")
async def update_agent_step(request: Request = None):
    """Update the agent step count and metrics (called by agent.py)"""
    global agent_step_count, latest_metrics
    
    try:
        # Check if this is a direct set operation or has metrics
        if request:
            try:
                request_data = await request.json()
                
                # Update metrics if provided (with thread safety)
                if "metrics" in request_data and isinstance(request_data["metrics"], dict):
                    with step_lock:  # Use existing lock for thread safety
                        # Safely update each metric individually to avoid race conditions
                        for key, value in request_data["metrics"].items():
                            if key in latest_metrics:
                                # Always protect total_actions as it's managed by server
                                if key == "total_actions":
                                    continue
                                else:
                                    latest_metrics[key] = value
                    
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
    
    # Default increment behavior
    with step_lock:
        agent_step_count += 1
    
    return {"status": "updated", "agent_step": agent_step_count}

@app.get("/llm_logs")
async def get_llm_logs():
    """Get recent LLM log entries"""
    try:
        from utils.llm_logger import get_llm_logger
        
        llm_logger = get_llm_logger()
        session_summary = llm_logger.get_session_summary()
        
        # Get recent log entries
        recent_entries = []
        if os.path.exists(llm_logger.log_file):
            try:
                with open(llm_logger.log_file, 'r', encoding='utf-8') as f:
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
        
        return {
            "session_summary": session_summary,
            "recent_entries": recent_entries,
            "log_file": llm_logger.log_file
        }
        
    except Exception as e:
        logger.error(f"Error getting LLM logs: {e}")
        return {"error": str(e)}

# Milestone checking is now handled by the emulator

@app.get("/milestones")
async def get_milestones():
    """Get current milestones achieved based on persistent tracking"""
    if env is None:
        raise HTTPException(status_code=400, detail="Emulator not initialized")
    
    try:
        # Get milestones directly from emulator
        return env.get_milestones()
        
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
            "error": str(e)
        }

# Global list to track recent button presses
recent_button_presses = []

@app.get("/recent_actions")
async def get_recent_actions():
    """Get recently pressed buttons"""
    global recent_button_presses
    return {
        "recent_buttons": recent_button_presses[-10:],  # Last 10 button presses
        "timestamp": time.time()
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
                with open(env.milestone_tracker.filename, 'r') as f:
                    default_data = json.load(f)
                default_file_info = {
                    "exists": True,
                    "size": os.path.getsize(env.milestone_tracker.filename),
                    "last_modified": time.ctime(os.path.getmtime(env.milestone_tracker.filename)),
                    "milestone_count": len(default_data.get('milestones', {})),
                    "last_updated": default_data.get('last_updated', 'unknown')
                }
            except Exception as e:
                default_file_info = {"exists": True, "error": str(e)}
        else:
            default_file_info = {"exists": False}
        
        return {
            "tracking_system": "file_based",
            "current_filename": env.milestone_tracker.filename,
            "current_milestones": len(env.milestone_tracker.milestones),
            "completed_milestones": sum(1 for m in env.milestone_tracker.milestones.values() if m.get("completed", False)),
            "default_file_info": default_file_info,
            "milestone_files_in_directory": milestone_files,
            "working_directory": current_dir,
            "milestone_details": env.milestone_tracker.milestones
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
            "remaining_milestones": len(env.milestone_tracker.milestones)
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
            "TEST_MILESTONE_1": {
                "completed": True,
                "timestamp": time.time(),
                "first_completed": time.time()
            },
            "TEST_MILESTONE_2": {
                "completed": False,
                "timestamp": None
            }
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
                    "milestones_saved": len(test_milestones)
                },
                "load_operation": {
                    "milestones_loaded": len(loaded_milestones),
                    "milestones_match": loaded_milestones == test_milestones,
                    "loaded_milestones": loaded_milestones
                }
            },
            "original_state_restored": True
        }
        
    except Exception as e:
        logger.error(f"Error testing milestone operations: {e}")
        return {"error": str(e)}

@app.post("/stop")
async def stop_server():
    """Stop the server"""
    global running
    running = False
    return {"status": "stopping"}

@app.post("/save_state")
async def save_state_endpoint(request: dict):
    """Save the current emulator state to a file"""
    try:
        os.makedirs(".pokeagent_cache", exist_ok=True)
        filepath = request.get("filepath", ".pokeagent_cache/manual_save.state")
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
async def load_state_endpoint(request: dict):
    """Load an emulator state from a file"""
    try:
        os.makedirs(".pokeagent_cache", exist_ok=True)
        filepath = request.get("filepath", ".pokeagent_cache/manual_save.state")
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
        step_count = request_data.get("step_count", 0) if request_data else 0
        
        # Save emulator state
        os.makedirs(".pokeagent_cache", exist_ok=True)
        checkpoint_state = ".pokeagent_cache/checkpoint.state"
        if env:
            env.save_state(checkpoint_state)
            logger.info(f"💾 Server: Saved checkpoint state at step {step_count}")
            
            # Save milestones
            if env.milestone_tracker:
                milestone_file = env.milestone_tracker.save_milestones_for_state(checkpoint_state)
                logger.info(f"💾 Server: Saved checkpoint milestones")
            
            return {
                "status": "checkpoint_saved",
                "step_count": step_count,
                "files": {
                    "state": checkpoint_state,
                    "milestones": f".pokeagent_cache/checkpoint_milestones.json",
                    "map": f".pokeagent_cache/checkpoint_grids.json"
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
        from utils.llm_logger import get_llm_logger
        llm_logger = get_llm_logger()
        if llm_logger is not None:
            # Update cumulative metrics (but preserve server-managed metrics like start_time and total_actions)
            server_start_time = llm_logger.cumulative_metrics.get("start_time")
            server_total_actions = llm_logger.cumulative_metrics.get("total_actions")
            
            llm_logger.cumulative_metrics.update(cumulative_metrics)
            
            # Restore server-managed metrics
            if server_start_time:
                llm_logger.cumulative_metrics["start_time"] = server_start_time
            if server_total_actions is not None:
                llm_logger.cumulative_metrics["total_actions"] = server_total_actions
            
            # Also sync to latest_metrics for stream.html display (excluding server-managed metrics)
            global latest_metrics
            with step_lock:
                for key, value in cumulative_metrics.items():
                    if key in latest_metrics and key not in ["total_actions", "start_time"]:
                        latest_metrics[key] = value
            
            logger.info(f"🔄 Synced LLM metrics: {cumulative_metrics.get('total_llm_calls', 0)} calls, {cumulative_metrics.get('total_tokens', 0)} tokens, ${cumulative_metrics.get('total_cost', 0):.6f}")
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
        from utils.llm_logger import get_llm_logger
        
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
        checkpoint_state = ".pokeagent_cache/checkpoint.state"
        
        if not os.path.exists(checkpoint_state):
            return {"status": "no_checkpoint", "message": "No .pokeagent_cache/checkpoint.state file found"}
        
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
                    "milestones": f".pokeagent_cache/checkpoint_milestones.json",
                    "map": f".pokeagent_cache/checkpoint_grids.json"
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
    
    global state_update_running, state_update_thread
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="Simple Pokemon Emerald Server")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    parser.add_argument("--manual", action="store_true", help="Enable manual mode with keyboard input and overlay")
    parser.add_argument("--load-state", type=str, help="Load a saved state file on startup")
    parser.add_argument("--record", action="store_true", help="Record video of the gameplay")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR dialogue detection")
    # Server always runs headless - display handled by client
    
    args = parser.parse_args()
    
    # Check for environment variables from multiprocess mode
    env_load_state = os.environ.get("LOAD_STATE")
    if env_load_state and not args.load_state:
        args.load_state = env_load_state
        print(f"📂 Using load state from environment: {env_load_state}")
        if env_load_state == ".pokeagent_cache/checkpoint.state":
            if os.path.exists(".pokeagent_cache/checkpoint.state"):
                print(f"✅ Server startup: .pokeagent_cache/checkpoint.state file exists")
            else:
                print(f"❌ Server startup: .pokeagent_cache/checkpoint.state file MISSING!")
    
    # Set checkpoint loading flag based on whether this is a true checkpoint load
    global checkpoint_loading_enabled
    env_load_checkpoint_mode = os.environ.get("LOAD_CHECKPOINT_MODE")
    
    if env_load_checkpoint_mode == "true":
        checkpoint_loading_enabled = True
        print("🔄 Checkpoint loading enabled - will restore LLM metrics from checkpoint_llm.txt")
        
        # Initialize LLM logger and load checkpoint immediately during server startup
        from utils.llm_logger import get_llm_logger
        llm_logger = get_llm_logger()
        # Check both cache folder and old location
        cache_dir = ".pokeagent_cache"
        checkpoint_file = os.path.join(cache_dir, "checkpoint_llm.txt") if os.path.exists(cache_dir) else "checkpoint_llm.txt"
        if not os.path.exists(checkpoint_file) and os.path.exists("checkpoint_llm.txt"):
            checkpoint_file = "checkpoint_llm.txt"
        
        if llm_logger and os.path.exists(checkpoint_file):
            restored_step_count = llm_logger.load_checkpoint(checkpoint_file)
            if restored_step_count is not None:
                global agent_step_count
                agent_step_count = restored_step_count
                print(f"✅ Server startup: restored LLM checkpoint with step count {restored_step_count}")
                
                # Sync latest_metrics with loaded cumulative metrics
                global latest_metrics
                latest_metrics.update(llm_logger.cumulative_metrics)
                print(f"✅ Server startup: synced metrics - actions: {latest_metrics.get('total_actions', 0)}, cost: {latest_metrics.get('total_cost', 0)}")
            else:
                print("❌ Server startup: failed to load LLM checkpoint")
        else:
            print("ℹ️ Server startup: no checkpoint_llm.txt file found")
    elif env_load_checkpoint_mode == "false":
        checkpoint_loading_enabled = False
        print("✨ Fresh start mode - will NOT load LLM metrics from checkpoint_llm.txt")
    else:
        # Default behavior: allow checkpoint loading unless explicitly disabled
        checkpoint_loading_enabled = True
        print("🔄 Checkpoint loading enabled by default - will restore LLM metrics from checkpoint_llm.txt if available")
    
    print("Starting Fixed Simple Pokemon Emerald Server")
    # Initialize video recording if requested
    init_video_recording(args.record)
    print("Server mode - headless operation, display handled by client")
    if args.no_ocr:
        print("OCR dialogue detection disabled")
    print("Press Ctrl+C to stop")
    
    # Initialize emulator
    # Skip initial state reading if we're going to load a state
    if not setup_environment(skip_initial_state=(args.load_state is not None)):
        print("Failed to initialize emulator")
        return
    
    # Disable dialogue detection if --no-ocr flag is set
    if args.no_ocr:
        if env and env.memory_reader:
            env.memory_reader._dialog_detection_enabled = False
            print("🚫 All dialogue detection disabled (--no-ocr flag)")
    
    # Load state if specified
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
            
            # Map buffer should already be found by emulator.load_state()
            if env.memory_reader and env.memory_reader._map_buffer_addr:
                print(f"Map buffer already initialized at 0x{env.memory_reader._map_buffer_addr:08X}")
                
            # Now log the initial GAME_RUNNING milestone after state is loaded
            try:
                env.milestone_tracker.mark_completed("GAME_RUNNING")
                initial_state = env.get_comprehensive_state()
                
                import hashlib
                state_str = str(initial_state)
                state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]
                
                anticheat_tracker.log_submission_data(
                    step=0,
                    state_data=initial_state,
                    action_taken="INIT",
                    decision_time=0.0,
                    state_hash=state_hash,
                    manual_mode=True,
                    milestone_override="GAME_RUNNING"
                )
                print("Initial GAME_RUNNING milestone logged after state load")
                
                # Trigger a map stitcher update to ensure visual map is ready
                try:
                    if env.memory_reader and env.memory_reader._map_stitcher:
                        # Check if map stitcher is empty and collect initial map data if needed
                        map_areas = env.memory_reader._map_stitcher.map_areas
                        if not map_areas:
                            print("🗺️ Map stitcher is empty, collecting initial map data...")
                            # Collect initial map data
                            tiles = env.memory_reader.read_map_around_player(radius=7)
                            if tiles:
                                print(f"🗺️ Collected {len(tiles)} tiles, updating map stitcher")
                                # Create minimal state for stitcher update
                                initial_state = {"map": {}}
                                env.memory_reader._update_map_stitcher(tiles, initial_state)
                                print("✅ Initial map data collection completed")
                            else:
                                print("❌ Could not collect initial map data")
                        
                        # Get current state for map stitcher update
                        current_state = env.get_comprehensive_state()
                        # The map stitcher should now have data
                        # This just ensures the visual_map is generated
                        print("Ensuring map stitcher visual data is ready after state load")
                except Exception as e:
                    print(f"Note: Could not update map stitcher after state load: {e}")
            except Exception as e:
                print(f"Warning: Could not log initial milestone: {e}")
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
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.get_local_ip import get_local_ip
    local_ip = get_local_ip()
    
    print(f"🌐 FastAPI server running:")
    print(f"   Local: http://localhost:{args.port}")
    print(f"   Network: http://{local_ip}:{args.port}")
    print(f"📺 Stream interface: http://{local_ip}:{args.port}/stream")
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
        # Cleanup
        global running
        running = False
        state_update_running = False
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
            
            env = EmeraldEmulator(rom_path=rom_path)
            env.initialize()
            
            # Initialize video recording if requested
            init_video_recording(record_video)
            
            # Disable OCR if requested
            if no_ocr and env and env.memory_reader:
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
                    
                    # Map buffer should already be found by emulator.load_state()
                    if env.memory_reader and env.memory_reader._map_buffer_addr:
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
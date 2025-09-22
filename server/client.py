#!/usr/bin/env python3
"""
Simple multiprocess client that connects to the server and runs the agent.
"""

import os
import sys
import time
import base64
import io
import requests
from PIL import Image

# Display-related imports (conditionally used)
try:
    import pygame
    import numpy as np
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from utils.state_formatter import format_state_for_llm


def update_display_with_status(screen, font, mode, step_count, additional_info="", frame_surface=None):
    """
    Update the display with frame (if provided) and status text overlay.
    
    Args:
        screen: Pygame screen surface
        font: Pygame font for rendering text
        mode: Current mode (MANUAL/AGENT/AUTO)
        step_count: Current step count
        additional_info: Additional status information to display
        frame_surface: Optional frame surface to display, if None fills with black
    """
    if frame_surface:
        # Display the frame
        scaled_surface = pygame.transform.scale(frame_surface, (480, 320))
        screen.blit(scaled_surface, (0, 0))
    else:
        # Fill with black if no frame
        screen.fill((0, 0, 0))
    
    # Create status text
    status_text = f"{mode} | Steps: {step_count}"
    if additional_info:
        status_text += f" | {additional_info}"
    
    # Render and display status text
    text_surface = font.render(status_text, True, (255, 255, 255))
    screen.blit(text_surface, (10, 290))
    pygame.display.flip()


def run_multiprocess_client(server_port=8000, args=None):
    """
    Simple client that gets state from server, processes with agent, sends action back.
    Supports manual control when pygame display is enabled.
    
    Args:
        server_port: Port the server is running on
        args: Command line arguments with agent configuration
    
    Returns:
        bool: True if ran successfully
    """
    server_url = f"http://localhost:{server_port}"
    
    # Initialize the agent (it handles VLM, simple vs 4-module, etc internally)
    agent = Agent(args)
    print(f"✅ Agent initialized")
    print(f"🎮 Client connected to server at {server_url}")
    
    # Display setup
    headless = args and args.headless
    screen = None
    clock = None
    font = None
    
    # Control state - three modes: MANUAL, AGENT, AUTO
    if args and args.manual:
        mode = "MANUAL"
    elif args and args.agent_auto:
        mode = "AUTO"
    else:
        mode = "AGENT"
    
    last_agent_time = time.time()
    step_count = 0
    
    # Initialize pygame if not headless
    if not headless and PYGAME_AVAILABLE:
        pygame.init()
        screen = pygame.display.set_mode((480, 320))
        pygame.display.set_caption("Pokemon Emerald")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()
        print("✅ Display initialized")
        print("Controls: Tab=Cycle Mode (MANUAL/AGENT/AUTO), Space=Agent Step, M=Show State, Arrows/WASD=Move, Z=A, X=B")
    elif not headless and not PYGAME_AVAILABLE:
        print("⚠️ Pygame not available, running in headless mode")
        headless = True
    
    # Auto-display comprehensive state in manual mode for debugging
    auto_state_timer = None
    if args and args.manual:
        auto_state_timer = time.time() + 5  # Display state after 5 seconds in manual mode (allow map to initialize)
    
    # Main loop
    running = True
    while running:
        try:
            # Auto-display comprehensive state in manual mode (one time)
            if auto_state_timer and time.time() >= auto_state_timer:
                print("🔍 Auto-displaying comprehensive state in manual mode...")
                try:
                    response = requests.get(f"{server_url}/state", timeout=5)
                    if response.status_code == 200:
                        state_data = response.json()
                        print("=" * 80)
                        print("📊 COMPREHENSIVE STATE (LLM View)")
                        print("=" * 80)
                        from utils.state_formatter import format_state_for_llm
                        formatted_state = format_state_for_llm(state_data)
                        print(formatted_state)
                        print("=" * 80)
                    else:
                        print(f"❌ Failed to get state: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error getting state: {e}")
                auto_state_timer = None  # Only display once
            
            # Handle pygame events and display
            if not headless:
                # Process events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        
                        # Mode cycle
                        elif event.key == pygame.K_TAB:
                            if mode == "MANUAL":
                                mode = "AGENT"
                            elif mode == "AGENT":
                                mode = "AUTO"
                            else:  # AUTO
                                mode = "MANUAL"
                            print(f"🎮 Mode: {mode}")
                        
                        # Manual agent step
                        elif event.key == pygame.K_SPACE and mode in ("AGENT", "AUTO"):
                            # Force an agent step
                            response = requests.get(f"{server_url}/state", timeout=5)
                            if response.status_code == 200:
                                state_data = response.json()
                                screenshot_base64 = state_data.get("visual", {}).get("screenshot_base64", "")
                                if screenshot_base64:
                                    img_data = base64.b64decode(screenshot_base64)
                                    screenshot = Image.open(io.BytesIO(img_data))
                                    game_state = {
                                        'frame': screenshot,
                                        'player': state_data.get('player', {}),
                                        'game': state_data.get('game', {}),
                                        'map': state_data.get('map', {}),
                                        'milestones': state_data.get('milestones', {}),
                                        'visual': state_data.get('visual', {}),
                                        'step_number': state_data.get('step_number', 0),
                                        'status': state_data.get('status', ''),
                                        'action_queue_length': state_data.get('action_queue_length', 0)
                                    }
                                    result = agent.step(game_state)
                                    if result and result.get('action'):
                                        # Convert action to buttons list format expected by server
                                        action = result['action']
                                        if isinstance(action, list):
                                            buttons = action  # Already a list of buttons
                                        else:
                                            # Single action string, convert to list
                                            buttons = action.split(',') if ',' in action else [action]
                                            buttons = [btn.strip() for btn in buttons]
                                        
                                        try:
                                            response = requests.post(
                                                f"{server_url}/action",
                                                json={"buttons": buttons},
                                                timeout=5
                                            )
                                            if response.status_code == 200:
                                                print(f"🎮 Agent: {action} (sent successfully)")
                                            else:
                                                print(f"🎮 Agent: {action} (server error: {response.status_code})")
                                        except requests.exceptions.RequestException as e:
                                            print(f"🎮 Agent: {action} (connection error: {e})")
                                        step_count += 1
                                        print(f"🎮 Step {step_count}: {result['action']}")
                        
                        # Manual controls (only in manual mode)
                        elif mode == "MANUAL":
                            action = None
                            if event.key in (pygame.K_UP, pygame.K_w):
                                action = "UP"
                            elif event.key in (pygame.K_DOWN, pygame.K_s):
                                action = "DOWN"
                            elif event.key in (pygame.K_LEFT, pygame.K_a):
                                action = "LEFT"
                            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                                action = "RIGHT"
                            elif event.key == pygame.K_z:
                                action = "A"
                            elif event.key == pygame.K_x:
                                action = "B"
                            elif event.key == pygame.K_RETURN:
                                action = "START"
                            elif event.key == pygame.K_BACKSPACE:
                                action = "SELECT"
                            elif event.key == pygame.K_LSHIFT:
                                action = "L"
                            elif event.key == pygame.K_RSHIFT:
                                action = "R"
                            elif event.key == pygame.K_1:
                                # Save state
                                print("💾 Saving state...")
                                try:
                                    response = requests.post(f"{server_url}/save_state", 
                                                           json={"filepath": ".pokeagent_cache/manual_save.state"}, 
                                                           timeout=5)
                                    if response.status_code == 200:
                                        print("✅ State saved to .pokeagent_cache/manual_save.state")
                                    else:
                                        print(f"❌ Failed to save state: {response.status_code}")
                                except Exception as e:
                                    print(f"❌ Error saving state: {e}")
                            elif event.key == pygame.K_2:
                                # Load state
                                print("📂 Loading state...")
                                try:
                                    response = requests.post(f"{server_url}/load_state", 
                                                           json={"filepath": ".pokeagent_cache/manual_save.state"}, 
                                                           timeout=5)
                                    if response.status_code == 200:
                                        print("✅ State loaded from .pokeagent_cache/manual_save.state")
                                    else:
                                        print(f"❌ Failed to load state: {response.status_code}")
                                except Exception as e:
                                    print(f"❌ Error loading state: {e}")
                            elif event.key == pygame.K_m:
                                # Display comprehensive state (what LLM sees)
                                print("🔍 Getting comprehensive state...")
                                try:
                                    response = requests.get(f"{server_url}/state", timeout=5)
                                    if response.status_code == 200:
                                        state_data = response.json()
                                        print("=" * 80)
                                        print("📊 COMPREHENSIVE STATE (LLM View)")
                                        print("=" * 80)
                                        
                                        # Format and display state in a readable way (exactly what LLM sees)
                                        formatted_state = format_state_for_llm(state_data)
                                        print(formatted_state)
                                        
                                        print("=" * 80)
                                    else:
                                        print(f"❌ Failed to get state: {response.status_code}")
                                except Exception as e:
                                    print(f"❌ Error getting state: {e}")
                            
                            if action:
                                # Send manual action to server using the same endpoint as agent actions
                                try:
                                    response = requests.post(
                                        f"{server_url}/action",
                                        json={"buttons": [action]},
                                        timeout=2
                                    )
                                    if response.status_code == 200:
                                        print(f"🎮 Manual: {action} (sent successfully)")
                                    else:
                                        print(f"🎮 Manual: {action} (server error: {response.status_code})")
                                except requests.exceptions.RequestException as e:
                                    print(f"🎮 Manual: {action} (connection error: {e})")
                
                # Update display
                try:
                    response = requests.get(f"{server_url}/screenshot", timeout=0.5)
                    if response.status_code == 200:
                        frame_data = response.json().get("screenshot_base64", "")
                        if frame_data:
                            img_data = base64.b64decode(frame_data)
                            img = Image.open(io.BytesIO(img_data))
                            frame_array = np.array(img)
                            frame_surface = pygame.surfarray.make_surface(frame_array.swapaxes(0, 1))
                            update_display_with_status(screen, font, mode, step_count, frame_surface=frame_surface)
                        else:
                            update_display_with_status(screen, font, mode, step_count, "No frame data")
                    else:
                        update_display_with_status(screen, font, mode, step_count, f"Server error: {response.status_code}")
                except Exception as e:
                    update_display_with_status(screen, font, mode, step_count, f"Error: {str(e)[:30]}")
                
                clock.tick(30)  # 30 FPS for display
            
            # Auto agent processing (both headless and display modes)
            if mode == "AUTO":
                current_time = time.time()
                if current_time - last_agent_time > 3.0:  # Every 3 seconds
                    # Check if action queue is ready
                    try:
                        queue_response = requests.get(f"{server_url}/queue_status", timeout=1)
                        if queue_response.status_code == 200:
                            queue_status = queue_response.json()
                            if queue_status.get("queue_empty", False):
                                # Get state and process
                                response = requests.get(f"{server_url}/state", timeout=5)
                                if response.status_code == 200:
                                    state_data = response.json()
                                    screenshot_base64 = state_data.get("visual", {}).get("screenshot_base64", "")
                                    if screenshot_base64:
                                        img_data = base64.b64decode(screenshot_base64)
                                        screenshot = Image.open(io.BytesIO(img_data))
                                        
                                        game_state = {
                                            'frame': screenshot,
                                            'player': state_data.get('player', {}),
                                            'game': state_data.get('game', {}),
                                            'map': state_data.get('map', {}),
                                            'milestones': state_data.get('milestones', {}),
                                            'visual': state_data.get('visual', {}),
                                            'step_number': state_data.get('step_number', 0),
                                            'status': state_data.get('status', ''),
                                            'action_queue_length': state_data.get('action_queue_length', 0)
                                        }
                                        
                                        result = agent.step(game_state)
                                        if result and result.get('action'):
                                            # Convert action to buttons list format expected by server
                                            action = result['action']
                                            if isinstance(action, list):
                                                buttons = action  # Already a list of buttons
                                            else:
                                                # Single action string, convert to list
                                                buttons = action.split(',') if ',' in action else [action]
                                                buttons = [btn.strip() for btn in buttons]
                                            
                                            try:
                                                response = requests.post(
                                                    f"{server_url}/action",
                                                    json={"buttons": buttons},
                                                    timeout=5
                                                )
                                                if response.status_code == 200:
                                                    step_count += 1
                                                    print(f"🎮 Agent: {action} (sent successfully)")
                                                    print(f"🎮 Step {step_count}: {result['action']}")
                                                    last_agent_time = current_time
                                                    
                                                    # Auto-save checkpoint after each step for persistence
                                                    try:
                                                        # Sync client's LLM metrics to server before saving checkpoint
                                                        try:
                                                            from utils.llm_logger import get_llm_logger
                                                            client_llm_logger = get_llm_logger()
                                                            if client_llm_logger:
                                                                sync_response = requests.post(
                                                                    f"{server_url}/sync_llm_metrics",
                                                                    json={"cumulative_metrics": client_llm_logger.cumulative_metrics},
                                                                    timeout=5
                                                                )
                                                                if sync_response.status_code == 200:
                                                                    if step_count % 10 == 0:  # Log every 10 steps to avoid spam
                                                                        print(f"🔄 LLM metrics synced to server")
                                                        except Exception as e:
                                                            print(f"⚠️ LLM metrics sync error: {e}")
                                                        
                                                        # Save game state checkpoint
                                                        checkpoint_response = requests.post(
                                                            f"{server_url}/checkpoint",
                                                            json={"step_count": step_count},
                                                            timeout=10
                                                        )
                                                        
                                                        # Save agent history to checkpoint_llm.txt
                                                        history_response = requests.post(
                                                            f"{server_url}/save_agent_history",
                                                            timeout=5
                                                        )
                                                        
                                                        if checkpoint_response.status_code == 200 and history_response.status_code == 200:
                                                            if step_count % 10 == 0:  # Log every 10 steps to avoid spam
                                                                print(f"💾 Checkpoint and history saved at step {step_count}")
                                                        else:
                                                            print(f"⚠️ Save failed - Checkpoint: {checkpoint_response.status_code}, History: {history_response.status_code}")
                                                    except requests.exceptions.RequestException as e:
                                                        print(f"⚠️ Checkpoint/history save error: {e}")
                                                else:
                                                    print(f"🎮 Agent: {action} (server error: {response.status_code})")
                                            except requests.exceptions.RequestException as e:
                                                print(f"🎮 Agent: {action} (connection error: {e})")
                    except Exception as e:
                        print(f"❌ AUTO mode error: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Small sleep to prevent CPU spinning
            if headless:
                time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("\n👋 Shutdown requested")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(2)
    
    # Cleanup
    if not headless and PYGAME_AVAILABLE:
        pygame.quit()
    
    return True
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
    headless = args and args.no_display
    screen = None
    clock = None
    font = None
    
    # Control state - three modes: MANUAL, AGENT, AUTO
    if args and args.manual_mode:
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
        pygame.display.set_caption("Pokemon Agent - Multiprocess")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()
        print("✅ Display initialized")
        print("Controls: Tab=Cycle Mode (MANUAL/AGENT/AUTO), Space=Agent Step, Arrows/WASD=Move, Z=A, X=B")
    elif not headless and not PYGAME_AVAILABLE:
        print("⚠️ Pygame not available, running in headless mode")
        headless = True
    
    # Main loop
    running = True
    while running:
        try:
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
                                        'screenshot': screenshot,
                                        'game_state': state_data.get('game_state', {}),
                                        'visual': state_data.get('visual', {}),
                                        'audio': state_data.get('audio', {}),
                                        'progress': state_data.get('progress', {})
                                    }
                                    result = agent.step(game_state)
                                    if result and result.get('action'):
                                        requests.post(
                                            f"{server_url}/agent_action",
                                            json={"action": result['action'], "reasoning": result.get('reasoning', '')},
                                            timeout=5
                                        )
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
                            
                            if action:
                                # Send manual action to server
                                requests.post(
                                    f"{server_url}/manual_action",
                                    json={"action": action},
                                    timeout=2
                                )
                                print(f"🎮 Manual: {action}")
                
                # Update display
                try:
                    response = requests.get(f"{server_url}/screenshot", timeout=0.5)
                    if response.status_code == 200:
                        frame_data = response.json().get("screenshot", "")
                        if frame_data:
                            img_data = base64.b64decode(frame_data)
                            img = Image.open(io.BytesIO(img_data))
                            frame_array = np.array(img)
                            frame_surface = pygame.surfarray.make_surface(frame_array.swapaxes(0, 1))
                            scaled_surface = pygame.transform.scale(frame_surface, (480, 320))
                            screen.blit(scaled_surface, (0, 0))
                            
                            # Draw status overlay
                            status_text = f"{mode} | Steps: {step_count}"
                            text_surface = font.render(status_text, True, (255, 255, 255))
                            screen.blit(text_surface, (10, 290))
                            
                            pygame.display.flip()
                except:
                    pass  # Silent fail for display updates
                
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
                                            'screenshot': screenshot,
                                            'game_state': state_data.get('game_state', {}),
                                            'visual': state_data.get('visual', {}),
                                            'audio': state_data.get('audio', {}),
                                            'progress': state_data.get('progress', {})
                                        }
                                        
                                        result = agent.step(game_state)
                                        if result and result.get('action'):
                                            response = requests.post(
                                                f"{server_url}/agent_action",
                                                json={"action": result['action'], "reasoning": result.get('reasoning', '')},
                                                timeout=5
                                            )
                                            if response.status_code == 200:
                                                step_count += 1
                                                print(f"🎮 Step {step_count}: {result['action']}")
                                                last_agent_time = current_time
                    except:
                        pass  # Silent fail for auto mode
            
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
#!/usr/bin/env python3
"""
Manual mode utilities for debugging and testing the Pokemon agent.
"""

from utils.state_formatter import format_state_for_llm
from utils.map_formatter import format_map_for_display
from utils.map_visualizer import visualize_map_state
from pokemon_env.enums import MetatileBehavior


def display_comprehensive_state(emulator):
    """
    Display comprehensive game state exactly as the LLM sees it.
    This is triggered by pressing 'M' in manual mode.
    
    Args:
        emulator: The game emulator instance
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE GAME STATE (What the LLM sees)")
    print("="*80)
    
    try:
        # Get the comprehensive state with screenshot for OCR
        screenshot = emulator.get_screenshot()
        state = emulator.get_comprehensive_state(screenshot)
        
        # Format the state using the same formatter the agent uses
        formatted_state = format_state_for_llm(state)
        
        print(formatted_state)
        
        # Show additional debug info
        print("\n" + "="*80)
        print("ADDITIONAL DEBUG INFO")
        print("="*80)
        
        # Show raw game state
        game_state = state.get('game_state', {})
        print(f"\n📍 Player Position: {game_state.get('player_location', 'Unknown')}")
        print(f"🗺️  Map: {game_state.get('current_map', 'Unknown')}")
        print(f"💰 Money: ${game_state.get('money', 0)}")
        print(f"🎮 Play Time: {game_state.get('play_time', 'Unknown')}")
        
        # Show party info
        party = game_state.get('party', [])
        if party:
            print(f"\n🎭 Party ({len(party)} Pokémon):")
            for i, mon in enumerate(party, 1):
                print(f"   {i}. {mon.get('species', 'Unknown')} "
                      f"Lv.{mon.get('level', '?')} "
                      f"HP: {mon.get('hp', '?')}/{mon.get('max_hp', '?')}")
        
        # Show battle info if in battle
        if game_state.get('in_battle'):
            print(f"\n⚔️  BATTLE MODE:")
            print(f"   Enemy: {game_state.get('enemy_pokemon', 'Unknown')}")
            print(f"   Type: {game_state.get('battle_type', 'Unknown')}")
        
        # Show dialogue if present
        dialogue = state.get('visual', {}).get('dialogue_text', '')
        if dialogue:
            print(f"\n💬 Current Dialogue:")
            print(f"   {dialogue}")
        
        # Show milestone progress
        progress = state.get('progress', {})
        milestones = progress.get('milestones', {})
        if milestones:
            completed = sum(1 for v in milestones.values() if v)
            total = len(milestones)
            print(f"\n🏆 Milestones: {completed}/{total} completed")
            
            # Show next uncompleted milestone
            for name, completed in milestones.items():
                if not completed:
                    print(f"   Next: {name}")
                    break
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"❌ Error displaying state: {e}")
        import traceback
        traceback.print_exc()


def display_map_visualization(emulator):
    """
    Display a visual representation of the current map.
    This is triggered by pressing Shift+M in manual mode.
    
    Args:
        emulator: The game emulator instance
    """
    print("\n" + "="*80)
    print("MAP VISUALIZATION")
    print("="*80)
    
    try:
        # Get current state
        screenshot = emulator.get_screenshot()
        state = emulator.get_comprehensive_state(screenshot)
        game_state = state.get('game_state', {})
        
        # Get map data
        map_grid = game_state.get('map_grid', [[]])
        player_pos = game_state.get('player_position', (0, 0))
        map_name = game_state.get('current_map', 'Unknown')
        
        print(f"\n🗺️  Current Map: {map_name}")
        print(f"📍 Player Position: {player_pos}")
        
        if map_grid:
            # Use the map formatter to display
            formatted_map = format_map_for_display(
                map_grid, 
                player_pos,
                view_radius=7
            )
            print(formatted_map)
            
            # Show tile legend
            print("\n📋 Tile Legend:")
            print("   @ = Player")
            print("   . = Walkable")
            print("   # = Wall/Obstacle")
            print("   ~ = Water")
            print("   ^ = Grass")
            print("   D = Door/Warp")
            print("   N = NPC")
            print("   ! = Ledge")
            print("   ? = Unknown")
        else:
            print("⚠️ No map data available")
        
        # Try to use the visual map if available
        try:
            visualize_map_state(emulator.memory_reader)
        except:
            pass  # Silent fail if visualization not available
        
    except Exception as e:
        print(f"❌ Error displaying map: {e}")
        import traceback
        traceback.print_exc()


def show_debug_menu():
    """Display the debug menu with available commands"""
    print("\n" + "="*50)
    print("DEBUG MENU")
    print("="*50)
    print("M       - Display comprehensive state (LLM view)")
    print("Shift+M - Display map visualization")
    print("S       - Save screenshot")
    print("1       - Save state")
    print("2       - Load state")
    print("Tab     - Toggle Agent/Manual mode")
    print("A       - Toggle auto-agent mode")
    print("Space   - Single agent step")
    print("Esc     - Quit")
    print("="*50)


def handle_debug_command(key, emulator, agent_mode=False):
    """
    Handle debug keyboard commands.
    
    Args:
        key: The key pressed
        emulator: The emulator instance
        agent_mode: Whether in agent mode
    
    Returns:
        dict: Action to take based on the command
    """
    action = {}
    
    if key == 'm':
        display_comprehensive_state(emulator)
    elif key == 'M':  # Shift+M
        display_map_visualization(emulator)
    elif key == 's':
        # Save screenshot
        screenshot = emulator.get_screenshot()
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        screenshot.save(filename)
        print(f"💾 Screenshot saved to {filename}")
    elif key == '1':
        # Save state
        emulator.save_state("manual_save.state")
        print("💾 State saved to manual_save.state")
    elif key == '2':
        # Load state
        try:
            emulator.load_state("manual_save.state")
            print("📂 State loaded from manual_save.state")
        except:
            print("⚠️ No save state found")
    elif key == 'tab':
        action['toggle_mode'] = True
    elif key == 'a':
        action['toggle_auto'] = True
    elif key == ' ' and agent_mode:
        action['agent_step'] = True
    elif key == 'h':
        show_debug_menu()
    
    return action


# Add missing import
from datetime import datetime
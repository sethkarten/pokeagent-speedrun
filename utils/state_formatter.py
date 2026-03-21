#!/usr/bin/env python3
"""
State Formatter Utility

Converts comprehensive game state objects to formatted text for LLM prompts and debugging.
Centralizes all state formatting logic for consistency across agent modules.
"""

import json
import logging
import numpy as np
from PIL import Image
from utils.map_formatter import format_map_grid, format_map_for_llm, generate_dynamic_legend, format_tile_to_symbol, _get_behavior_enum
import base64
import io
import os, sys
from pathlib import Path
from typing import Optional, List, Tuple
from pokemon_env.enums import MetatileBehavior
from utils import state_formatter as sf

logger = logging.getLogger(__name__)

# Global location tracking - MapStitcher handles all persistent storage
CURRENT_LOCATION = None
LAST_LOCATION = None
LAST_TRANSITION = None  # Stores transition coordinates
MAP_STITCHER_SAVE_CALLBACK = None  # Callback to save map stitcher when location connections change
MAP_STITCHER_INSTANCE = None  # Reference to the MapStitcher instance

def _get_location_connections_from_cache():
    """Read location connections from MapStitcher's cache file"""
    try:
        from utils.run_data_manager import get_cache_path
        cache_file = get_cache_path("map_stitcher_data.json")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('location_connections', {})
    except Exception as e:
        print(f"Failed to read location connections from cache: {e}")
    return {}

def detect_dialogue_on_frame(screenshot_base64=None, frame_array=None):
    """
    Detect if dialogue is visible on the game frame by analyzing the lower portion.
    
    Args:
        screenshot_base64: Base64 encoded screenshot string
        frame_array: numpy array of the frame (240x160 for GBA)
        
    Returns:
        dict: {
            'has_dialogue': bool,
            'confidence': float (0-1),
            'reason': str
        }
    """
    try:
        # Convert base64 to image if needed
        if screenshot_base64 and not frame_array:
            image_data = base64.b64decode(screenshot_base64)
            image = Image.open(io.BytesIO(image_data))
            frame_array = np.array(image)
        
        if frame_array is None:
            return {'has_dialogue': False, 'confidence': 0.0, 'reason': 'No frame data'}
        
        # GBA resolution is 240x160
        height, width = frame_array.shape[:2]
        
        # Dialogue typically appears in the bottom 40-50 pixels
        dialogue_region = frame_array[height-50:, :]  # Bottom 50 pixels
        
        # Convert to grayscale for analysis
        if len(dialogue_region.shape) == 3:
            # Convert RGB to grayscale
            gray = np.dot(dialogue_region[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = dialogue_region
        
        # Dialogue boxes in Pokemon are typically:
        # 1. Have a distinct blue/white color scheme
        # 2. Have high contrast text on background
        # 3. Have consistent borders
        
        # Check for dialogue box characteristics
        # 1. Check for blue dialogue box (typical color range)
        if len(dialogue_region.shape) == 3:
            # Blue dialogue box detection (Pokemon dialogue boxes are often blue-ish)
            blue_mask = (
                (dialogue_region[:,:,2] > 100) &  # High blue channel
                (dialogue_region[:,:,2] > dialogue_region[:,:,0] * 1.2) &  # More blue than red
                (dialogue_region[:,:,2] > dialogue_region[:,:,1] * 1.2)    # More blue than green
            )
            blue_percentage = np.sum(blue_mask) / blue_mask.size
            
            # White/light regions (text areas)
            white_mask = (
                (dialogue_region[:,:,0] > 200) &
                (dialogue_region[:,:,1] > 200) &
                (dialogue_region[:,:,2] > 200)
            )
            white_percentage = np.sum(white_mask) / white_mask.size
        else:
            blue_percentage = 0
            white_percentage = 0
        
        # 2. Check for high contrast (text on background)
        std_dev = np.std(gray)
        
        # 3. Check for horizontal lines (dialogue box borders)
        # Detect horizontal edges
        vertical_diff = np.abs(np.diff(gray, axis=0))
        horizontal_edges = np.sum(vertical_diff > 50) / vertical_diff.size
        
        # 4. Check for consistent patterns (not random pixels)
        # Calculate local variance to detect structured content
        local_variance = []
        for i in range(0, gray.shape[0]-5, 5):
            for j in range(0, gray.shape[1]-5, 5):
                patch = gray[i:i+5, j:j+5]
                local_variance.append(np.var(patch))
        
        avg_local_variance = np.mean(local_variance) if local_variance else 0
        
        # Scoring system
        confidence = 0.0
        reasons = []
        
        # Blue/white dialogue box detection
        if blue_percentage > 0.3:
            confidence += 0.3
            reasons.append("blue dialogue box detected")
        
        if white_percentage > 0.1 and white_percentage < 0.5:
            confidence += 0.2
            reasons.append("text area detected")
        
        # High contrast for text
        if std_dev > 30 and std_dev < 100:
            confidence += 0.2
            reasons.append("text contrast detected")
        
        # Horizontal edges (box borders)
        if horizontal_edges > 0.01 and horizontal_edges < 0.1:
            confidence += 0.2
            reasons.append("dialogue box borders detected")
        
        # Structured content (not random)
        if avg_local_variance > 100 and avg_local_variance < 2000:
            confidence += 0.1
            reasons.append("structured content")
        
        # Determine if dialogue is present
        has_dialogue = confidence >= 0.5
        
        return {
            'has_dialogue': has_dialogue,
            'confidence': min(confidence, 1.0),
            'reason': ', '.join(reasons) if reasons else 'no dialogue indicators'
        }
        
    except Exception as e:
        logger.warning(f"Failed to detect dialogue on frame: {e}")
        return {'has_dialogue': False, 'confidence': 0.0, 'reason': f'error: {e}'}

def format_state(state_data, format_type="summary", include_debug_info=False, include_npcs=True, include_movement_preview=True, action_history=None):
    """
    Format comprehensive state data into readable text.
    
    Args:
        state_data (dict): The comprehensive state from /state endpoint
        format_type (str): "summary" for one-line summary, "detailed" for multi-line LLM format
        include_debug_info (bool): Whether to include extra debug information (for detailed format)
        include_npcs (bool): Whether to include NPC information in the state
        include_movement_preview (bool): Whether to include movement preview (for detailed format)
        action_history (list): Optional list of recent actions with start/end positions
    
    Returns:
        str: Formatted state text
    """
    if format_type == "summary":
        return _format_state_summary(state_data)
    elif format_type == "detailed":
        return _format_state_detailed(state_data, include_debug_info, include_npcs, include_movement_preview, action_history)
    else:
        raise ValueError(f"Unknown format_type: {format_type}. Use 'summary' or 'detailed'")

def format_state_for_llm(state_data, include_debug_info=False, include_npcs=True, include_movement_preview=True, action_history=None):
    """
    Format comprehensive state data into a readable context for the VLM.
    
    Args:
        state_data (dict): The comprehensive state from /state endpoint
        include_debug_info (bool): Whether to include extra debug information
        include_npcs (bool): Whether to include NPC information in the state
        include_movement_preview (bool): Whether to include movement preview (deprecated for pathfinding agents)
        action_history (list): Optional list of recent actions with start/end positions
    
    Returns:
        str: Formatted state context for LLM prompts
    """
    return format_state(state_data, format_type="detailed", include_debug_info=include_debug_info, include_npcs=include_npcs, include_movement_preview=include_movement_preview, action_history=action_history)

def format_state_summary(state_data):
    """
    Create a concise one-line summary of the current state for logging.
    
    Args:
        state_data (dict): The comprehensive state from /state endpoint
    
    Returns:
        str: Concise state summary
    """
    return format_state(state_data, format_type="summary")

def _format_state_summary(state_data):
    """
    Internal function to create a concise one-line summary of the current state.
    """
    player_data = state_data.get('player', {})
    game_data = state_data.get('game', {})
    
    summary_parts = []
    
    # Player name (don't show during title sequence)
    player_location = player_data.get('location', '')
    if player_data.get('name') and player_location != 'TITLE_SEQUENCE':
        summary_parts.append(f"Player: {player_data['name']}")
    
    # Location
    location = player_data.get('location')
    if location:
        summary_parts.append(f"Location: {location}")
    
    # Position
    position = player_data.get('position')
    if position and isinstance(position, dict):
        summary_parts.append(f"Pos: ({position.get('x', '?')}, {position.get('y', '?')})")
    
    # Facing direction - removed as it's often unreliable
    # facing = player_data.get('facing')
    # if facing:
    #     summary_parts.append(f"Facing: {facing}")
    
    # Game state
    game_state = game_data.get('game_state')
    if game_state:
        summary_parts.append(f"State: {game_state}")
    
    # Battle status
    if game_data.get('is_in_battle'):
        summary_parts.append("In Battle")
    
    # Money
    money = game_data.get('money')
    if money is not None:
        summary_parts.append(f"Money: ${money}")
    
    # Party information
    party_data = player_data.get('party')
    if party_data:
        party_size = len(party_data)
        if party_size > 0:
            # Get first Pokemon details
            first_pokemon = party_data[0]
            species = first_pokemon.get('species_name', 'Unknown')
            level = first_pokemon.get('level', '?')
            hp = first_pokemon.get('current_hp', '?')
            max_hp = first_pokemon.get('max_hp', '?')
            status = first_pokemon.get('status', 'OK')
            
            summary_parts.append(f"Party: {party_size} pokemon")
            summary_parts.append(f"Lead: {species} Lv{level} HP:{hp}/{max_hp} {status}")
    
    # Pokedex information
    pokedex_seen = game_data.get('pokedex_seen')
    pokedex_caught = game_data.get('pokedex_caught')
    if pokedex_seen is not None:
        summary_parts.append(f"Pokedex: {pokedex_caught or 0} caught, {pokedex_seen} seen")
    
    # Badges
    badges = game_data.get('badges')
    if badges:
        if isinstance(badges, list):
            badge_count = len(badges)
        else:
            badge_count = badges
        summary_parts.append(f"Badges: {badge_count}")
    
    # Items (Gen 1 bag holds max 20 unique item types)
    items = game_data.get('items')
    item_count = game_data.get('item_count')
    if items and len(items) > 0:
        item_names = [f"{it['name']}×{it['quantity']}" for it in items[:20]]
        summary_parts.append(f"Items ({len(items)}): {', '.join(item_names)}")
    elif item_count is not None:
        summary_parts.append(f"Items: {item_count}")
    
    # Game time
    time_data = game_data.get('time')
    if time_data and isinstance(time_data, (list, tuple)) and len(time_data) >= 3:
        hours, minutes, seconds = time_data[:3]
        summary_parts.append(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    # Dialog text (if any)
    dialog_text = game_data.get('dialog_text')
    dialogue_detected = game_data.get('dialogue_detected', {})
    if dialog_text and dialogue_detected.get('has_dialogue', True):
        # Only show dialogue if frame detection confirms it (or if detection wasn't run)
        # Truncate dialog text to first 50 characters
        dialog_preview = dialog_text[:50].replace('\n', ' ').strip()
        if len(dialog_text) > 50:
            dialog_preview += "..."
        summary_parts.append(f"Dialog: {dialog_preview}")
    
    # Progress context (if available)
    progress_context = game_data.get('progress_context')
    if progress_context:
        badges_obtained = progress_context.get('badges_obtained', 0)
        visited_locations = progress_context.get('visited_locations', [])
        if badges_obtained > 0:
            summary_parts.append(f"Progress: {badges_obtained} badges, {len(visited_locations)} locations")
    
    return " | ".join(summary_parts) if summary_parts else "No state data"

def _format_state_detailed(state_data, include_debug_info=False, include_npcs=True, include_movement_preview=True, action_history=None):
    """
    Internal function to create detailed multi-line state format for LLM prompts.
    """
    context_parts = []
    
    # Add action history at the beginning if available
    if action_history:
        action_history_text = format_action_history(action_history)
        context_parts.append(action_history_text)
        context_parts.append("")  # Add blank line for spacing
    
    # Check both player and game sections for data
    player_data = state_data.get('player', {})
    game_data = state_data.get('game', {})
    
    # Check if we're in battle to determine formatting mode
    is_in_battle = game_data.get('is_in_battle', False) or game_data.get('in_battle', False)
    
    if is_in_battle:
        # Remove heavy overworld map data while in battle to reduce prompt size
        map_info = state_data.get('map') if isinstance(state_data, dict) else None
        if isinstance(map_info, dict):
            map_info.pop('porymap', None)
         # BATTLE MODE: Focus on battle-relevant information
        context_parts.append("=== BATTLE MODE ===")
        context_parts.append("Currently in battle - map and dialogue information hidden")
        
        # Battle information first
        if 'battle_info' in game_data and game_data['battle_info']:
            battle = game_data['battle_info']
            context_parts.append("\n=== BATTLE STATUS ===")
            
            # Battle type and context
            battle_type = battle.get('battle_type', 'unknown')
            context_parts.append(f"Battle Type: {battle_type.title()}")
            if battle.get('is_capturable'):
                context_parts.append("🟢 Wild Pokémon - CAN BE CAPTURED")
            if battle.get('can_escape'):
                context_parts.append("🟡 Can escape from battle")
            
            # Player's active Pokémon
            if 'player_pokemon' in battle and battle['player_pokemon']:
                player_pkmn = battle['player_pokemon']
                context_parts.append(f"\n--- YOUR POKÉMON ---")
                context_parts.append(f"{player_pkmn.get('nickname', player_pkmn.get('species', 'Unknown'))} (Lv.{player_pkmn.get('level', '?')})")
                
                # Health display with percentage
                current_hp = player_pkmn.get('current_hp', 0)
                max_hp = player_pkmn.get('max_hp', 1)
                hp_pct = player_pkmn.get('hp_percentage', 0)
                health_bar = "🟢" if hp_pct > 50 else "🟡" if hp_pct > 25 else "🔴"
                context_parts.append(f"  HP: {current_hp}/{max_hp} ({hp_pct}%) {health_bar}")
                
                # Status condition
                status = player_pkmn.get('status', 'Normal')
                if status != 'Normal':
                    context_parts.append(f"  Status: {status}")
                
                # Types
                types = player_pkmn.get('types', [])
                if types:
                    context_parts.append(f"  Type: {'/'.join(types)}")
                
                # Available moves with PP
                moves = player_pkmn.get('moves', [])
                move_pp = player_pkmn.get('move_pp', [])
                if moves:
                    context_parts.append(f"  Moves:")
                    for i, move in enumerate(moves):
                        if move and move.strip():
                            pp = move_pp[i] if i < len(move_pp) else '?'
                            context_parts.append(f"    {i+1}. {move} (PP: {pp})")
                
            # Opponent Pokémon
            if 'opponent_pokemon' in battle:
                if battle['opponent_pokemon']:
                    opp_pkmn = battle['opponent_pokemon']
                    context_parts.append(f"\n--- OPPONENT POKÉMON ---")
                    context_parts.append(f"{opp_pkmn.get('species', 'Unknown')} (Lv.{opp_pkmn.get('level', '?')})")
                    
                    # Health display with percentage
                    current_hp = opp_pkmn.get('current_hp', 0)
                    max_hp = opp_pkmn.get('max_hp', 1)
                    hp_pct = opp_pkmn.get('hp_percentage', 0)
                    health_bar = "🟢" if hp_pct > 50 else "🟡" if hp_pct > 25 else "🔴"
                    context_parts.append(f"  HP: {current_hp}/{max_hp} ({hp_pct}%) {health_bar}")
                    
                    # Status condition
                    status = opp_pkmn.get('status', 'Normal')
                    if status != 'Normal':
                        context_parts.append(f"  Status: {status}")
                    
                    # Types
                    types = opp_pkmn.get('types', [])
                    if types:
                        context_parts.append(f"  Type: {'/'.join(types)}")
                    
                    # Moves (for wild Pokémon, showing moves can help with strategy)
                    moves = opp_pkmn.get('moves', [])
                    if moves and any(move.strip() for move in moves):
                        context_parts.append(f"  Known Moves:")
                        for i, move in enumerate(moves):
                            if move and move.strip():
                                context_parts.append(f"    • {move}")
                    
                    # Stats (helpful for battle strategy)
                    stats = opp_pkmn.get('stats', {})
                    if stats:
                        context_parts.append(f"  Battle Stats: ATK:{stats.get('attack', '?')} DEF:{stats.get('defense', '?')} SPD:{stats.get('speed', '?')}")
                    
                    # Special indicators
                    if opp_pkmn.get('is_shiny'):
                        context_parts.append(f"  ✨ SHINY POKÉMON!")
                else:
                    # Opponent data not ready
                    context_parts.append(f"\n--- OPPONENT POKÉMON ---")
                    opponent_status = battle.get('opponent_status', 'Opponent data not available')
                    context_parts.append(f"⏳ {opponent_status}")
                    context_parts.append("  (Battle may be in initialization phase)")
                    
            # Battle interface info
            interface = battle.get('battle_interface', {})
            available_actions = interface.get('available_actions', [])
            if available_actions:
                context_parts.append(f"\n--- AVAILABLE ACTIONS ---")
                context_parts.append(f"Options: {', '.join(available_actions)}")
                
            # Trainer battle specific info
            if battle.get('is_trainer_battle'):
                remaining = battle.get('opponent_team_remaining', 1)
                if remaining > 1:
                    context_parts.append(f"\nTrainer has {remaining} Pokémon remaining")
                    
            # Battle phase info
            battle_phase = battle.get('battle_phase_name')
            if battle_phase:
                context_parts.append(f"\nBattle Phase: {battle_phase}")
        
        # Party information (important for switching decisions)
        context_parts.append("\n=== PARTY STATUS ===")
        party_context = _format_party_info(player_data, game_data)
        context_parts.extend(party_context)
        
        # Trainer info if available
        if 'name' in player_data and player_data['name']:
            context_parts.append(f"\nTrainer: {player_data['name']}")
        
        # Money/badges might be relevant
        money = player_data.get('money') or game_data.get('money')
        if money is not None:
            context_parts.append(f"Money: ${money}")
            
    else:
        # NORMAL MODE: Full state information
        context_parts.append("=== PLAYER INFO ===")
        
        # Player name and basic info (don't show during title sequence as it hasn't been set yet)
        player_location = player_data.get('location', '')
        if 'name' in player_data and player_data['name'] and player_location != 'TITLE_SEQUENCE':
            context_parts.append(f"Player Name: {player_data['name']}")
        
        # Position information
        position = _get_player_position(player_data)
        if position:
            context_parts.append(f"Position: X={position.get('x', 'unknown')}, Y={position.get('y', 'unknown')}")
        
        # Facing direction - removed as it's often unreliable
        # if 'facing' in player_data and player_data['facing']:
        #     context_parts.append(f"Facing: {player_data['facing']}")
        
        # Money (check both player and game sections)
        money = player_data.get('money') or game_data.get('money')
        if money is not None:
            context_parts.append(f"Money: ${money}")
        
        # Pokemon Party (check both player and game sections)
        party_context = _format_party_info(player_data, game_data)
        context_parts.extend(party_context)

        # Map/Location information with traversability (NOT shown in battle)
        map_context = _format_map_info(state_data.get('map', {}), player_data, include_debug_info, include_npcs, state_data)
        context_parts.extend(map_context)

        # Game state information (including dialogue if not in battle)
        game_context = _format_game_state(game_data, state_data, include_movement_preview)
        context_parts.extend(game_context)
    
    # Debug information if requested (shown in both modes)
    if include_debug_info:
        debug_context = _format_debug_info(state_data)
        context_parts.extend(debug_context)
    
    return "\n".join(context_parts)

def format_state_for_debug(state_data):
    """
    Format state data for detailed debugging output.
    
    Args:
        state_data (dict): The comprehensive state from /state endpoint
    
    Returns:
        str: Detailed debug information
    """
    debug_parts = []
    debug_parts.append("=" * 60)
    debug_parts.append("COMPREHENSIVE STATE DEBUG")
    debug_parts.append("=" * 60)
    
    # Raw structure overview
    debug_parts.append("\n--- STRUCTURE OVERVIEW ---")
    for key, value in state_data.items():
        if isinstance(value, dict):
            debug_parts.append(f"{key}: dict with {len(value)} keys")
        elif isinstance(value, list):
            debug_parts.append(f"{key}: list with {len(value)} items")
        else:
            debug_parts.append(f"{key}: {type(value).__name__} = {value}")
    
    # Detailed formatted state
    debug_parts.append("\n--- FORMATTED STATE ---")
    debug_parts.append(format_state_for_llm(state_data, include_debug_info=True))
    
    # Raw JSON (truncated if too long)
    debug_parts.append("\n--- RAW JSON (truncated) ---")
    raw_json = json.dumps(state_data, indent=2)
    if len(raw_json) > 2000:
        debug_parts.append(raw_json[:2000] + "\n... (truncated)")
    else:
        debug_parts.append(raw_json)
    
    debug_parts.append("=" * 60)
    return "\n".join(debug_parts)

# Helper functions for state formatting

def _get_player_position(player_data):
    """Extract player position from various possible locations in player data."""
    if 'coordinates' in player_data:
        return player_data['coordinates']
    elif 'position' in player_data:
        pos = player_data['position']
            # print( _get_player_position found position: {pos}, type: {type(pos)}")
        # Check if it's a valid position dict with x,y keys
        if pos and isinstance(pos, dict) and 'x' in pos and 'y' in pos:
            return pos
            # print( position invalid - missing x,y or not dict")
    return None

def _get_party_size(party_data):
    """Get party size from party data regardless of format."""
    if isinstance(party_data, dict):
        return party_data.get('size', len(party_data.get('pokemon', [])))
    elif isinstance(party_data, list):
        return len(party_data)
    return 0

def _format_party_info(player_data, game_data):
    """Format pokemon party information."""
    context_parts = []
    
    # Pokemon Party (check both player and game sections)
    party_data = player_data.get('party') or game_data.get('party')
    if party_data:
        pokemon_list = []
        if isinstance(party_data, dict) and party_data.get('pokemon'):
            # Format: {"size": X, "pokemon": [...]}
            pokemon_list = party_data.get('pokemon', [])
            party_size = party_data.get('size', len(pokemon_list))
        elif isinstance(party_data, list):
            # Format: [pokemon1, pokemon2, ...]
            pokemon_list = party_data
            party_size = len(pokemon_list)
        else:
            party_size = 0
        
        if party_size > 0:
            context_parts.append(f"Pokemon Party ({party_size} pokemon):")
            for i, pokemon in enumerate(pokemon_list[:6]):
                if pokemon:
                    # Check if it's an egg
                    is_egg = pokemon.get('is_egg', False)
                    if is_egg:
                        species = "Egg"
                        level = "?"
                        status = "Egg"
                    else:
                        species = pokemon.get('species_name', pokemon.get('species', 'Unknown'))
                        level = pokemon.get('level', '?')
                        status = pokemon.get('status', 'Normal')
                    hp = pokemon.get('current_hp', '?')
                    max_hp = pokemon.get('max_hp', '?')
                    context_parts.append(f"  {i+1}. {species} (Lv.{level}) HP: {hp}/{max_hp} Status: {status}")
        else:
            context_parts.append("No Pokemon in party")
    else:
        context_parts.append("No Pokemon in party")
    
    return context_parts

def _format_map_info(map_info, player_data=None, include_debug_info=False, include_npcs=True, full_state_data=None):
    """Format map and traversability information using MapStitcher."""
    context_parts = []
    
    if not map_info:
        return context_parts
    
    # Ensure map_info is actually part of state (not a copy)
    # This ensures porymap data we store persists in the state
    if full_state_data and 'map' not in full_state_data:
        full_state_data['map'] = map_info
    elif full_state_data and full_state_data.get('map') is not map_info:
        # If map_info is a different object, merge our changes back
        full_state_data['map'].update(map_info)
        map_info = full_state_data['map']
    
    # Get location name from player data
    location_name = None
    if player_data and 'location' in player_data and player_data['location']:
        location = player_data['location']
        if isinstance(location, dict):
            location_name = location.get('map_name', str(location))
        else:
            location_name = str(location)
    
    # Special handling for title sequence - don't show map
    if location_name == 'TITLE_SEQUENCE':
        context_parts.append("\n=== LOCATION INFO ===")
        context_parts.append(f"Current Location: {location_name}")
        context_parts.append("No map available during title sequence")
        return context_parts
    
    context_parts.append("\n=== LOCATION & MAP INFO ===")
    if location_name:
        context_parts.append(f"Current Location: {location_name}")
    
    # Get player coordinates from ROM (read via memory_reader.read_coordinates())
    # This is the actual player position from the game, not from MapStitcher
    player_coords = None
    rom_player_coords = None
    if player_data and 'position' in player_data:
        pos = player_data['position']
        if pos:
            rom_player_coords = (pos.get('x', 0), pos.get('y', 0))
            player_coords = rom_player_coords  # May be adjusted below if using override map
    
    # Get badge count for game-state-aware map selection (e.g., Petalburg Gym lobby)
    badge_count = 0
    if full_state_data:
        game_data = full_state_data.get('game', {})
        badges = game_data.get('badges', [])
        if isinstance(badges, list):
            badge_count = len(badges)
        elif isinstance(badges, int):
            badge_count = badges
    
    # Game-type detection for map data source
    _game_type = os.environ.get("GAME_TYPE", "emerald")

    # Coordinate offset (Emerald porymap overrides only — Red doesn't use porymap)
    if _game_type != "red":
        from utils.ascii_map_loader import get_effective_map_name, get_override
        porymap_map_name = _get_porymap_map_name(location_name)
        coord_offset = None
        if porymap_map_name:
            effective_map_name = get_effective_map_name(porymap_map_name, badge_count=badge_count)
            override = get_override(effective_map_name)
            if override and ('offset_x' in override or 'offset_y' in override):
                offset_x = override.get('offset_x', 0)
                offset_y = override.get('offset_y', 0)
                coord_offset = (offset_x, offset_y)
                if rom_player_coords:
                    # Translate ROM coordinates to local map coordinates
                    player_coords = (rom_player_coords[0] - offset_x, rom_player_coords[1] - offset_y)

    # Display player position (translated if using override map)
    if player_coords:
        context_parts.append(f"Player Position: ({player_coords[0]}, {player_coords[1]})")

    # Add map data (game-specific)
    if _game_type == "red":
        porymap_result = _format_red_map_info(location_name, player_coords, map_info)
    else:
        porymap_result = _format_porymap_info(location_name, player_coords, badge_count=badge_count)
    if isinstance(porymap_result, tuple):
        porymap_info, porymap_data = porymap_result
        if porymap_info:
            context_parts.extend(porymap_info)
            # Store porymap data in map_info for pathfinding (don't add to context text)
            if 'porymap' not in map_info:
                map_info['porymap'] = {}
            map_info['porymap']['grid'] = porymap_data.get('grid')
            map_info['porymap']['objects'] = porymap_data.get('objects', [])
            map_info['porymap']['dimensions'] = porymap_data.get('dimensions', {})
            map_info['porymap']['raw_tiles'] = porymap_data.get('raw_tiles')  # Include raw tiles with elevation

            # Debug: Verify the grid was stored
            stored_grid = map_info['porymap'].get('grid')
            if stored_grid:
                logger.debug(f"Stored elevation-filtered porymap grid: {len(stored_grid)}x{len(stored_grid[0]) if stored_grid else 0}")
    elif porymap_result:
        context_parts.extend(porymap_result)

    # Fallback: display existing visual_map (e.g. from Red's map reader) if no porymap map was added
    has_map_content = any("MAP" in p.upper() or "Grid" in p for p in context_parts[2:] if isinstance(p, str))
    if not has_map_content:
        visual_map = map_info.get("visual_map")
        if visual_map:
            context_parts.append(f"\n--- MAP (viewport) ---")
            context_parts.append(visual_map)
            source = map_info.get("map_source", "memory")
            context_parts.append(f"(Source: {source})")

    return context_parts

def _add_local_map_fallback(context_parts, map_info, include_npcs, location_name=None):
    """Helper function to add local map display as fallback"""
    if 'tiles' in map_info and map_info['tiles']:
        raw_tiles = map_info['tiles']
        # Use default facing direction since memory-based facing is unreliable
        facing = "South"  # default
        
        # Get player coordinates
        player_coords = map_info.get('player_coords')
        
        # Get NPCs if available and include_npcs is True
        npcs = []
        if include_npcs and 'object_events' in map_info:
            npcs = map_info.get('object_events', [])
        
        # Use unified LLM formatter for consistency with NPCs if available
        map_display = format_map_for_llm(raw_tiles, facing, npcs, player_coords, location_name)
        context_parts.append(map_display)
        
        # Add dynamic legend based on symbols in the map
        grid = format_map_grid(raw_tiles, facing, npcs, player_coords, location_name=location_name)
        legend = generate_dynamic_legend(grid)
        context_parts.append(f"\n{legend}")

def _format_world_map_display(stitched_data, full_state_data=None):
    """Format location-specific map display"""
    try:
        # Build separate map for each location
        return _build_stitched_world_map(stitched_data, full_state_data)
        
    except Exception as e:
        logger.warning(f"World map generation failed: {e}")
        return []

def _get_map_stitcher_instance():
    """Get the MapStitcher instance - always reload from cache for multiprocess compatibility"""
    from utils.map_stitcher import MapStitcher
    # Always create fresh instance to read latest cache
    # This is needed because server and client run in different processes
    return MapStitcher()

def save_persistent_world_map(file_path=None):
    """Deprecated - MapStitcher handles all persistence now"""
    # MapStitcher auto-saves, nothing to do here
    pass

def load_persistent_world_map(file_path=None):
    """Deprecated - MapStitcher handles all persistence now"""
    # MapStitcher auto-loads, nothing to do here
    pass

def clear_persistent_world_map():
    """Clear the MapStitcher's data for testing"""
    global CURRENT_LOCATION, LAST_LOCATION, LAST_TRANSITION
    CURRENT_LOCATION = None
    LAST_LOCATION = None
    LAST_TRANSITION = None
    # Clear MapStitcher data if instance exists
    if MAP_STITCHER_INSTANCE:
        MAP_STITCHER_INSTANCE.map_areas.clear()
        MAP_STITCHER_INSTANCE.warp_connections.clear()
        MAP_STITCHER_INSTANCE.save_to_file()
            # print( Cleared map stitcher data")

# Helper function removed - now handled by MapStitcher._is_explorable_edge()


def _build_stitched_world_map(stitched_data, full_state_data=None):
    """Build map display using MapStitcher"""
    
    # Get the current area and location info
    current_area = stitched_data.get('current_area', {})
    location_name = current_area.get('name')
    
    # Get NPCs if available
    npcs = stitched_data.get('object_events', [])
    
    # Use fallback location name if current_area has 'Unknown' or empty name  
    fallback_name = stitched_data.get('location_name_fallback')
    if not location_name or location_name == 'Unknown':
        if fallback_name and fallback_name != 'Unknown':
            location_name = fallback_name
        else:
            location_name = 'Unknown'
    elif fallback_name and fallback_name != location_name and fallback_name != 'Unknown':
        location_name = fallback_name
    
    # Track location changes and transition coordinates
    global CURRENT_LOCATION, LAST_LOCATION, LAST_TRANSITION
    LAST_TRANSITION = None  # Will store transition info
    
    # Get MapStitcher instance - prefer the one from memory_reader if available
    # This ensures we use the instance that has the actual map data
    map_stitcher = map_info.get('_map_stitcher_instance') if map_info else None
    if not map_stitcher:
        map_stitcher = _get_map_stitcher_instance()
    
    # Get player position for transition tracking
    player_local_pos = stitched_data.get('player_local_pos', (0, 0))
    
    # Check if we've changed locations
    if CURRENT_LOCATION != location_name:
        # Store transition information
        if CURRENT_LOCATION is not None and CURRENT_LOCATION != location_name:
            LAST_TRANSITION = {
                'from_location': CURRENT_LOCATION,
                'from_coords': player_local_pos,  # Use current position as it's the exit point
                'to_location': location_name,
                'to_coords': player_local_pos
            }
            
            # Trigger MapStitcher save if callback is available
            if MAP_STITCHER_SAVE_CALLBACK:
                try:
                    # print( Location transition detected, triggering MapStitcher save...")
                    MAP_STITCHER_SAVE_CALLBACK()
                    # print( MapStitcher save completed")
                except Exception as e:
                    print("Failed to save via MapStitcher callback: {e}")
            
            # print( Location transition: {CURRENT_LOCATION} → {location_name} at position {player_local_pos}")
        
        LAST_LOCATION = CURRENT_LOCATION
        CURRENT_LOCATION = location_name
    
    # Get player position
    player_local_pos = stitched_data.get('player_local_pos')
    if not player_local_pos:
        # Fallback to extracting from current area
        player_local_pos = current_area.get('player_pos', (5, 5))  # Default to center
    
    # Get connection info from current_area
    connections = current_area.get('connections', [])
    
    # Build the display for this location
    lines = []
    
    # Add location transition indicator if we just changed locations
    if LAST_LOCATION and LAST_LOCATION != location_name:
        lines.append(f"\n⚡ LOCATION TRANSITION: {LAST_LOCATION} → {location_name}")
        # Add transition coordinates if available
        if LAST_TRANSITION and LAST_TRANSITION['to_location'] == location_name:
            from_coords = LAST_TRANSITION['from_coords']
            to_coords = LAST_TRANSITION['to_coords']
            from_location = LAST_TRANSITION['from_location']
            lines.append(f"📍 Exited {from_location} at ({from_coords[0]}, {from_coords[1]})")
            lines.append(f"📍 Entered {location_name} at ({to_coords[0]}, {to_coords[1]})")
        # Clear the LAST_LOCATION after showing transition
        LAST_LOCATION = None
        lines.append("")
    
    # Use MapStitcher to generate the map display
    map_stitcher = full_state_data.get('map_stitcher') if full_state_data else None
    if not map_stitcher and MAP_STITCHER_INSTANCE:
        map_stitcher = MAP_STITCHER_INSTANCE
    
    if map_stitcher:
        # Generate map display using MapStitcher
        map_lines = map_stitcher.generate_location_map_display(
            location_name=location_name,
            player_pos=player_local_pos,
            npcs=npcs,
            connections=connections
        )
        lines.extend(map_lines)
        
        # Add exploration statistics
        location_grid = map_stitcher.get_location_grid(location_name)
        if location_grid:
            total_tiles = len(location_grid)
            lines.append("")
            lines.append(f"Total explored in {location_name}: {total_tiles} tiles")
    else:
        # Fallback if no MapStitcher available
        lines.append(f"\n--- MAP: {location_name.upper()} ---")
        lines.append("Map data not available")
    
    # Add discovered connection points from our transition tracking
            # print( Checking portal coordinates for location: {location_name}")
    portal_connections_found = False
    
    # Check location connections from MapStitcher cache first
    location_connections = _get_location_connections_from_cache()
    if location_connections and location_name in location_connections:
        if not portal_connections_found:
            lines.append("")
            lines.append("Known Portal Coordinates:")
            portal_connections_found = True
        for other_loc, my_coords, their_coords in location_connections[location_name]:
            lines.append(f"  At ({my_coords[0]}, {my_coords[1]}) → {other_loc} ({their_coords[0]}, {their_coords[1]})")
    
    # Also check MAP_ID_CONNECTIONS if available (from loaded MapStitcher data or HTTP response)
            # print( About to check MAP_ID_CONNECTIONS...")
            # print( full_state_data is: {type(full_state_data)} with keys: {list(full_state_data.keys()) if full_state_data else 'None'}")
    try:
        # Try multiple ways to find MAP_ID_CONNECTIONS
        map_id_connections = None
        
        # Method 0: Check if passed via HTTP response (NEW - preferred method)
        # First try location_connections (more accurate), then fall back to portal_connections
        if full_state_data and 'location_connections' in full_state_data:
            location_connections = full_state_data['location_connections']
            # print( Found location_connections in HTTP response: {location_connections}")
            
            # Use the location connections display logic
            if location_name in location_connections:
                if not portal_connections_found:
                    lines.append("")
                    lines.append("Known Portal Coordinates:")
                    portal_connections_found = True
                for other_loc, my_coords, their_coords in location_connections[location_name]:
                    lines.append(f"  At ({my_coords[0]}, {my_coords[1]}) → {other_loc} ({their_coords[0]}, {their_coords[1]})")
            # print( Added location connection: At ({my_coords[0]}, {my_coords[1]}) → {other_loc} ({their_coords[0]}, {their_coords[1]})")
        
        elif full_state_data and 'portal_connections' in full_state_data:
            map_id_connections = full_state_data['portal_connections']
            # print( Found MAP_ID_CONNECTIONS in HTTP response: {map_id_connections}")
            
            # Get current map ID to find relevant portals
            current_map_id = None
            if stitched_data and 'current_area' in stitched_data:
                area_id = stitched_data['current_area'].get('id', '')
                if area_id:
                    try:
                        current_map_id = int(area_id, 16) if isinstance(area_id, str) else area_id
                        # print( Current map ID: {current_map_id}")
                    except ValueError:
                        print("Could not parse map ID from: {area_id}")
            
            # Try to find current map ID by matching current location name
            if not current_map_id:
                current_location = location_name  # Use the location_name parameter
                # print( Trying to find map ID for location: {current_location}")
                
                # Look through server's map data to find current map ID
                if 'map' in full_state_data:
                    map_info = full_state_data.get('map', {})
                    map_location = map_info.get('location_name', '')
                    if map_location:
                        current_location = map_location
                        # print( Using map location: {current_location}")
                
                # Match location with portal connections - try all map IDs
                for map_id in map_id_connections.keys():
                    # Convert string keys to int if needed
                    try:
                        test_map_id = int(map_id) if isinstance(map_id, str) else map_id
            # print( Testing map ID {test_map_id} for location '{current_location}'")
                        
                        # For LITTLEROOT TOWN MAYS HOUSE 2F, map_id should be 259
                        if current_location == "LITTLEROOT TOWN MAYS HOUSE 2F" and test_map_id == 259:
                            current_map_id = test_map_id
            # print( Found matching map ID: {current_map_id}")
                            break
                        elif current_location == "LITTLEROOT TOWN MAYS HOUSE 1F" and test_map_id == 258:
                            current_map_id = test_map_id
            # print( Found matching map ID: {current_map_id}")
                            break
                    except (ValueError, TypeError):
                        continue
                
            # Display portal coordinates if we found them
            if current_map_id and current_map_id in map_id_connections:
                if not portal_connections_found:
                    lines.append("")
                    lines.append("Known Portal Coordinates:")
                    portal_connections_found = True
                    
                for conn in map_id_connections[current_map_id]:
                    to_name = conn.get('to_name', 'Unknown Location')
                    from_pos = conn.get('from_pos', [0, 0])
                    to_pos = conn.get('to_pos', [0, 0])
                    lines.append(f"  At ({from_pos[0]}, {from_pos[1]}) → {to_name} ({to_pos[0]}, {to_pos[1]})")
            # print( Added portal: At ({from_pos[0]}, {from_pos[1]}) → {to_name} ({to_pos[0]}, {to_pos[1]})")
                    
        elif stitched_data and 'portal_connections' in stitched_data:
            map_id_connections = stitched_data['portal_connections']
            # print( Found MAP_ID_CONNECTIONS in stitched_data: {map_id_connections}")
        
        # Method 1: Check current module (fallback)
        if not map_id_connections:
            current_module = sys.modules[__name__]
            # print( Checking current module for MAP_ID_CONNECTIONS attribute...")
            # print( hasattr(current_module, 'MAP_ID_CONNECTIONS'): {hasattr(current_module, 'MAP_ID_CONNECTIONS')}")
            if hasattr(current_module, 'MAP_ID_CONNECTIONS') and current_module.MAP_ID_CONNECTIONS:
                map_id_connections = current_module.MAP_ID_CONNECTIONS
            # print( Found MAP_ID_CONNECTIONS in current module: {map_id_connections}")
        
        # Method 2: Check global variable (fallback)
        if not map_id_connections:
            # print( Checking globals for MAP_ID_CONNECTIONS...")
            # print( 'MAP_ID_CONNECTIONS' in globals(): {'MAP_ID_CONNECTIONS' in globals()}")
            try:
                if 'MAP_ID_CONNECTIONS' in globals():
                    # print( globals()['MAP_ID_CONNECTIONS']: {globals()['MAP_ID_CONNECTIONS']}")
                    if 'MAP_ID_CONNECTIONS' in globals() and globals()['MAP_ID_CONNECTIONS']:
                        map_id_connections = globals()['MAP_ID_CONNECTIONS']
                        # print( Found MAP_ID_CONNECTIONS in globals: {map_id_connections}")
            except Exception as e:
                print("Error checking globals: {e}")
        
        # Method 3: Re-import state_formatter and check (fallback)
        if not map_id_connections:
            try:
                # print( Attempting to re-import state_formatter...")
                # print( hasattr(sf, 'MAP_ID_CONNECTIONS'): {hasattr(sf, 'MAP_ID_CONNECTIONS')}")
                if hasattr(sf, 'MAP_ID_CONNECTIONS'):
                    # print( sf.MAP_ID_CONNECTIONS: {sf.MAP_ID_CONNECTIONS}")
                    if hasattr(sf, 'MAP_ID_CONNECTIONS') and sf.MAP_ID_CONNECTIONS:
                        map_id_connections = sf.MAP_ID_CONNECTIONS
                        # print( Found MAP_ID_CONNECTIONS in imported module: {map_id_connections}")
            except Exception as e:
                print(f" Error re-importing state_formatter: {e}")
        
        if map_id_connections:
            # print( MAP_ID_CONNECTIONS available with {len(map_id_connections)} maps")
            # print( Available map IDs: {list(map_id_connections.keys())}")
            
            # Get current map ID from stitched data
            current_map_id = None
            # print( stitched_data structure: {stitched_data}")
            if stitched_data and stitched_data.get('available'):
                # Try to get map ID from current area
                current_area = stitched_data.get('current_area', {})
                map_id_str = current_area.get('id')
            # print( Current area ID from stitched_data: {map_id_str}")
            # print( Full current_area: {current_area}")
                if map_id_str:
                    try:
                        current_map_id = int(map_id_str, 16)  # Convert from hex string
                        print(f" Converted to map ID: {current_map_id}")
                    except ValueError:
                        print(f" Failed to convert map ID: {map_id_str}")
                else:
                    print("No map ID found in current_area: {current_area}")
            # else:
            # print( No stitched_data available or not available")
            # print( stitched_data type: {type(stitched_data)}")
                # if stitched_data:
            # print( stitched_data keys: {list(stitched_data.keys()) if isinstance(stitched_data, dict) else 'not a dict'}")
            # print( stitched_data.get('available'): {stitched_data.get('available') if isinstance(stitched_data, dict) else 'N/A'}")
            
            if current_map_id and current_map_id in map_id_connections:
                # print( Found connections for map ID {current_map_id}")
                if not portal_connections_found:
                    lines.append("")
                    lines.append("Known Portal Coordinates:")
                    portal_connections_found = True
                for conn in map_id_connections[current_map_id]:
                    to_name = conn['to_name']
                    from_pos = conn['from_pos']
                    to_pos = conn['to_pos']
                    lines.append(f"  At ({from_pos[0]}, {from_pos[1]}) → {to_name} ({to_pos[0]}, {to_pos[1]})")
            else:
                print("No connections found for map ID {current_map_id} (available: {list(map_id_connections.keys()) if map_id_connections else 'None'})")
        else:
            print("MAP_ID_CONNECTIONS not found in any location")
    except Exception as e:
        print(f" Error checking MAP_ID_CONNECTIONS: {e}")
    
    return lines


def _format_stitched_map_info(map_info):
    """Format stitched map information for the agent"""
    context_parts = []
    
    # Check if stitched map info is available
    stitched_data = map_info.get('stitched_map_info')
    if not stitched_data or not stitched_data.get('available'):
        return context_parts
    
    # Check if world map display with terrain was already shown
    # Old world map knowledge system removed - replaced by location-based maps with portal coordinates
    return context_parts

def _get_pokeemerald_root() -> Optional[Path]:
    """Get the pokeemerald root directory path."""
    # Try environment variable first
    root = os.environ.get('POKEEMERALD_ROOT')
    if root:
        root_path = Path(root).resolve()
        if (root_path / "data" / "maps").exists():
            logger.info(f"Found pokeemerald root from env var: {root_path}")
            return root_path
    
    # Try porymap_data directory first (under pokeagent-speedrun)
    current_dir = Path(__file__).parent.parent
    porymap_path = current_dir / "porymap_data"
    if (porymap_path / "data" / "maps").exists():
        logger.info(f"Found pokeemerald root: {porymap_path}")
        return porymap_path.resolve()
    
    # Try common relative paths
    possible_paths = [
        current_dir / "pokeemerald",
        current_dir / "../pokeemerald",
        current_dir / "../../pokeemerald",
    ]
    
    for path in possible_paths:
        resolved = path.resolve()
        if (resolved / "data" / "maps").exists():
            logger.info(f"Found pokeemerald root: {resolved}")
            return resolved
    
    logger.warning("Could not find pokeemerald root directory. Checked:")
    logger.warning(f"  - POKEEMERALD_ROOT env var: {os.environ.get('POKEEMERALD_ROOT', 'not set')}")
    logger.warning(f"  - porymap_data: {porymap_path}")
    logger.warning(f"  - Common paths: {possible_paths}")
    return None

# ROM location name to Porymap map name mapping
ROM_TO_PORYMAP_MAP = {
    # Intro/Cutscene locations
    "MOVING_VAN": "InsideOfTruck",  # Intro cutscene (moving van)
    
    # Towns
    "LITTLEROOT TOWN": "LittlerootTown",
    "OLDALE TOWN": "OldaleTown",
    "DEWFORD TOWN": "DewfordTown",
    "LAVARIDGE TOWN": "LavaridgeTown",
    "FALLARBOR TOWN": "FallarborTown",
    "VERDANTURF TOWN": "VerdanturfTown",
    "PACIFIDLOG TOWN": "PacifidlogTown",
    
    # Cities
    "PETALBURG CITY": "PetalburgCity",
    "SLATEPORT CITY": "SlateportCity",
    "MAUVILLE CITY": "MauvilleCity",
    "RUSTBORO CITY": "RustboroCity",
    "FORTREE CITY": "FortreeCity",
    "LILYCOVE CITY": "LilycoveCity",
    "MOSSDEEP CITY": "MossdeepCity",
    "SOOTOPOLIS CITY": "SootopolisCity",
    "EVER GRANDE CITY": "EverGrandeCity",
    
    # Routes
    "ROUTE 101": "Route101",
    "ROUTE 102": "Route102",
    "ROUTE 103": "Route103",
    "ROUTE 104": "Route104",
    "ROUTE 105": "Route105",
    "ROUTE 106": "Route106",
    "ROUTE 107": "Route107",
    "ROUTE 108": "Route108",
    "ROUTE 109": "Route109",
    "ROUTE 110": "Route110",
    "ROUTE 111": "Route111",
    "ROUTE 112": "Route112",
    "ROUTE 113": "Route113",
    "ROUTE 114": "Route114",
    "ROUTE 115": "Route115",
    "ROUTE 116": "Route116",
    "ROUTE 117": "Route117",
    "ROUTE 118": "Route118",
    "ROUTE 119": "Route119",
    "ROUTE 120": "Route120",
    "ROUTE 121": "Route121",
    "ROUTE 122": "Route122",
    "ROUTE 123": "Route123",
    "ROUTE 124": "Route124",
    "ROUTE 125": "Route125",
    "ROUTE 126": "Route126",
    "ROUTE 127": "Route127",
    "ROUTE 128": "Route128",
    "ROUTE 129": "Route129",
    "ROUTE 130": "Route130",
    "ROUTE 131": "Route131",
    "ROUTE 132": "Route132",
    "ROUTE 133": "Route133",
    "ROUTE 134": "Route134",
    
    # Buildings (common patterns)
    "PETALBURG WOODS": "PetalburgWoods",
    "RUSTURF TUNNEL": "RusturfTunnel",
    "RUSTURF TUNNEL ALT": "RusturfTunnel",  # Alternative map ID 0x1804

    # Route 110 Trick House (Group 29 = 0x1D)
    "ROUTE 110 TRICK HOUSE ENTRANCE ALT": "Route110_TrickHouseEntrance",
    "ROUTE 110 TRICK HOUSE END ALT": "Route110_TrickHouseEnd",
    "ROUTE 110 TRICK HOUSE CORRIDOR ALT": "Route110_TrickHouseCorridor",
    "ROUTE 110 TRICK HOUSE PUZZLE1 ALT": "Route110_TrickHousePuzzle1",
    "ROUTE 110 TRICK HOUSE PUZZLE2 ALT": "Route110_TrickHousePuzzle2",
    "ROUTE 110 TRICK HOUSE PUZZLE3 ALT": "Route110_TrickHousePuzzle3",
    "ROUTE 110 TRICK HOUSE PUZZLE4 ALT": "Route110_TrickHousePuzzle4",
    "ROUTE 110 TRICK HOUSE PUZZLE5 ALT": "Route110_TrickHousePuzzle5",
    "ROUTE 110 TRICK HOUSE PUZZLE6 ALT": "Route110_TrickHousePuzzle6",
    "ROUTE 110 TRICK HOUSE PUZZLE7 ALT": "Route110_TrickHousePuzzle7",
    "ROUTE 110 TRICK HOUSE PUZZLE8 ALT": "Route110_TrickHousePuzzle8",
    "ROUTE 110 SEASIDE CYCLING ROAD SOUTH ENTRANCE ALT": "Route110_SeasideCyclingRoadSouthEntrance",
    "ROUTE 110 SEASIDE CYCLING ROAD NORTH ENTRANCE ALT": "Route110_SeasideCyclingRoadNorthEntrance",

    # Professor Birch's Lab
    "LITTLEROOT TOWN PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",
    "PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",

    # Raw map IDs (fallback when memory reader can't resolve location name)
    "Map_18_0B": "PetalburgWoods",  # Group 0x18 (Dungeons), Map 0x0B
    "Map_18_04": "RusturfTunnel",  # Group 0x18 (Indoor Route 104), Map 0x04
    "MAP_18_04": "RusturfTunnel",  # Alternate capitalization

        # The folloing mappings were generated via cursor and verified
    "ABANDONED SHIP CAPTAINS OFFICE": "AbandonedShip_CaptainsOffice",  # 100.0% match, verified
    "ABANDONED SHIP CORRIDORS 1F": "AbandonedShip_Corridors_1F",  # 100.0% match, verified
    "ABANDONED SHIP CORRIDORS B1F": "AbandonedShip_Corridors_B1F",  # 100.0% match, verified
    "ABANDONED SHIP DECK": "AbandonedShip_Deck",  # 100.0% match, verified
    "ABANDONED SHIP HIDDEN FLOOR CORRIDORS": "AbandonedShip_HiddenFloorCorridors",  # 100.0% match, verified
    "ABANDONED SHIP HIDDEN FLOOR ROOMS": "AbandonedShip_HiddenFloorRooms",  # 100.0% match, verified
    "ABANDONED SHIP ROOM B1F": "AbandonedShip_Room_B1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS 1F": "AbandonedShip_Rooms_1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS B1F": "AbandonedShip_Rooms_B1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS2 1F": "AbandonedShip_Rooms2_1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS2 B1F": "AbandonedShip_Rooms2_B1F",  # 100.0% match, verified
    "ABANDONED SHIP UNDERWATER1": "AbandonedShip_Underwater1",  # 100.0% match, verified
    "ABANDONED SHIP UNDERWATER2": "AbandonedShip_Underwater2",  # 100.0% match, verified
    "ALTERING CAVE": "AlteringCave",  # 100.0% match, verified
    "ANCIENT TOMB": "AncientTomb",  # 100.0% match, verified
    "AQUA HIDEOUT 1F": "AquaHideout_1F",  # 100.0% match, verified
    "AQUA HIDEOUT B1F": "AquaHideout_B1F",  # 100.0% match, verified
    "AQUA HIDEOUT B2F": "AquaHideout_B2F",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP1": "AquaHideout_UnusedRubyMap1",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP2": "AquaHideout_UnusedRubyMap2",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP3": "AquaHideout_UnusedRubyMap3",  # 100.0% match, verified
    "ARTISAN CAVE 1F": "ArtisanCave_1F",  # 100.0% match, verified
    "ARTISAN CAVE B1F": "ArtisanCave_B1F",  # 100.0% match, verified
    "CAVE OF ORIGIN 1F": "CaveOfOrigin_1F",  # 100.0% match, verified
    "CAVE OF ORIGIN B1F": "CaveOfOrigin_B1F",  # 100.0% match, verified
    "CAVE OF ORIGIN ENTRANCE": "CaveOfOrigin_Entrance",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP1": "CaveOfOrigin_UnusedRubySapphireMap1",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP2": "CaveOfOrigin_UnusedRubySapphireMap2",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP3": "CaveOfOrigin_UnusedRubySapphireMap3",  # 100.0% match, verified
    "DESERT RUINS": "DesertRuins",  # 100.0% match, verified
    "DESERT UNDERPASS": "DesertUnderpass",  # 100.0% match, verified
    "DEWFORD TOWN GYM": "DewfordTown_Gym",  # 100.0% match, verified
    "DEWFORD TOWN HALL": "DewfordTown_Hall",  # 100.0% match, verified
    "DEWFORD TOWN HOUSE1": "DewfordTown_House1",  # 100.0% match, verified
    "DEWFORD TOWN HOUSE2": "DewfordTown_House2",  # 100.0% match, verified
    "DEWFORD TOWN POKEMON CENTER 1F": "DewfordTown_PokemonCenter_1F",  # 100.0% match, verified
    "DEWFORD TOWN POKEMON CENTER 2F": "DewfordTown_PokemonCenter_2F",  # 100.0% match, verified
    "EVER GRANDE CITY CHAMPIONS ROOM": "EverGrandeCity_ChampionsRoom",  # 100.0% match, verified
    "EVER GRANDE CITY DRAKES ROOM": "EverGrandeCity_DrakesRoom",  # 100.0% match, verified
    "EVER GRANDE CITY GLACIAS ROOM": "EverGrandeCity_GlaciasRoom",  # 100.0% match, verified
    "EVER GRANDE CITY HALL OF FAME": "EverGrandeCity_HallOfFame",  # 100.0% match, verified
    "EVER GRANDE CITY HALL1": "EverGrandeCity_Hall1",  # 100.0% match, verified
    "EVER GRANDE CITY HALL2": "EverGrandeCity_Hall2",  # 100.0% match, verified
    "EVER GRANDE CITY HALL3": "EverGrandeCity_Hall3",  # 100.0% match, verified
    "EVER GRANDE CITY HALL4": "EverGrandeCity_Hall4",  # 100.0% match, verified
    "EVER GRANDE CITY HALL5": "EverGrandeCity_Hall5",  # 100.0% match, verified
    "EVER GRANDE CITY PHOEBES ROOM": "EverGrandeCity_PhoebesRoom",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON CENTER 1F": "EverGrandeCity_PokemonCenter_1F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON CENTER 2F": "EverGrandeCity_PokemonCenter_2F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON LEAGUE 1F": "EverGrandeCity_PokemonLeague_1F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON LEAGUE 2F": "EverGrandeCity_PokemonLeague_2F",  # 100.0% match, verified
    "EVER GRANDE CITY SIDNEYS ROOM": "EverGrandeCity_SidneysRoom",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT BATTLE ROOM": "FallarborTown_BattleTentBattleRoom",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT CORRIDOR": "FallarborTown_BattleTentCorridor",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT LOBBY": "FallarborTown_BattleTentLobby",  # 100.0% match, verified
    "FALLARBOR TOWN COZMOS HOUSE": "FallarborTown_CozmosHouse",  # 100.0% match, verified
    "FALLARBOR TOWN MART": "FallarborTown_Mart",  # 100.0% match, verified
    "FALLARBOR TOWN MOVE RELEARNERS HOUSE": "FallarborTown_MoveRelearnersHouse",  # 100.0% match, verified
    "FALLARBOR TOWN POKEMON CENTER 1F": "FallarborTown_PokemonCenter_1F",  # 100.0% match, verified
    "FALLARBOR TOWN POKEMON CENTER 2F": "FallarborTown_PokemonCenter_2F",  # 100.0% match, verified
    "FIERY PATH": "FieryPath",  # 100.0% match, verified
    "FIERY PATH INTERIOR": "FieryPath",  # Same location, different name variant
    "FORTREE CITY DECORATION SHOP": "FortreeCity_DecorationShop",  # 100.0% match, verified
    "FORTREE CITY GYM": "FortreeCity_Gym",  # 100.0% match, verified
    "FORTREE CITY HOUSE1": "FortreeCity_House1",  # 100.0% match, verified
    "FORTREE CITY HOUSE2": "FortreeCity_House2",  # 100.0% match, verified
    "FORTREE CITY HOUSE3": "FortreeCity_House3",  # 100.0% match, verified
    "FORTREE CITY HOUSE4": "FortreeCity_House4",  # 100.0% match, verified
    "FORTREE CITY HOUSE5": "FortreeCity_House5",  # 100.0% match, verified
    "FORTREE CITY MART": "FortreeCity_Mart",  # 100.0% match, verified
    "FORTREE CITY POKEMON CENTER 1F": "FortreeCity_PokemonCenter_1F",  # 100.0% match, verified
    "FORTREE CITY POKEMON CENTER 2F": "FortreeCity_PokemonCenter_2F",  # 100.0% match, verified
    "GRANITE CAVE 1F": "GraniteCave_1F",  # 100.0% match, verified
    "GRANITE CAVE 1F ALT": "GraniteCave_1F",  # Map ID 0x1807 - same layout as 0x1907
    "GRANITE CAVE B1F": "GraniteCave_B1F",  # 100.0% match, verified
    "GRANITE CAVE B1F ALT": "GraniteCave_B1F",  # Map ID 0x1808 - same layout as 0x1908
    "GRANITE CAVE B2F": "GraniteCave_B2F",  # 100.0% match, verified
    "GRANITE CAVE B2F ALT": "GraniteCave_B2F",  # Map ID 0x1809 - same layout as 0x1909
    "GRANITE CAVE STEVENS ROOM": "GraniteCave_StevensRoom",  # 100.0% match, verified
    "ISLAND CAVE": "IslandCave",  # 100.0% match, verified
    "JAGGED PASS": "JaggedPass",  # 100.0% match, verified
    "LAVARIDGE TOWN GYM 1F": "LavaridgeTown_Gym_1F",  # 100.0% match, verified
    "LAVARIDGE TOWN GYM B1F": "LavaridgeTown_Gym_B1F",  # 100.0% match, verified
    "LAVARIDGE TOWN HERB SHOP": "LavaridgeTown_HerbShop",  # 100.0% match, verified
    "LAVARIDGE TOWN HOUSE": "LavaridgeTown_House",  # 100.0% match, verified
    "LAVARIDGE TOWN MART": "LavaridgeTown_Mart",  # 100.0% match, verified
    "LAVARIDGE TOWN POKEMON CENTER 1F": "LavaridgeTown_PokemonCenter_1F",  # 100.0% match, verified
    "LAVARIDGE TOWN POKEMON CENTER 2F": "LavaridgeTown_PokemonCenter_2F",  # 100.0% match, verified
    "LILYCOVE CITY CONTEST HALL": "LilycoveCity_ContestHall",  # 100.0% match, verified
    "LILYCOVE CITY CONTEST LOBBY": "LilycoveCity_ContestLobby",  # 100.0% match, verified
    "LILYCOVE CITY COVE LILY MOTEL 1F": "LilycoveCity_CoveLilyMotel_1F",  # 100.0% match, verified
    "LILYCOVE CITY COVE LILY MOTEL 2F": "LilycoveCity_CoveLilyMotel_2F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 1F": "LilycoveCity_DepartmentStore_1F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 2F": "LilycoveCity_DepartmentStore_2F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 3F": "LilycoveCity_DepartmentStore_3F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 4F": "LilycoveCity_DepartmentStore_4F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 5F": "LilycoveCity_DepartmentStore_5F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE ELEVATOR": "LilycoveCity_DepartmentStoreElevator",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE ROOFTOP": "LilycoveCity_DepartmentStoreRooftop",  # 100.0% match, verified
    "LILYCOVE CITY HARBOR": "LilycoveCity_Harbor",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE1": "LilycoveCity_House1",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE2": "LilycoveCity_House2",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE3": "LilycoveCity_House3",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE4": "LilycoveCity_House4",  # 100.0% match, verified
    "LILYCOVE CITY LILYCOVE MUSEUM 1F": "LilycoveCity_LilycoveMuseum_1F",  # 100.0% match, verified
    "LILYCOVE CITY LILYCOVE MUSEUM 2F": "LilycoveCity_LilycoveMuseum_2F",  # 100.0% match, verified
    "LILYCOVE CITY MOVE DELETERS HOUSE": "LilycoveCity_MoveDeletersHouse",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON CENTER 1F": "LilycoveCity_PokemonCenter_1F",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON CENTER 2F": "LilycoveCity_PokemonCenter_2F",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON TRAINER FAN CLUB": "LilycoveCity_PokemonTrainerFanClub",  # 100.0% match, verified
    "LILYCOVE CITY UNUSED MART": "LilycoveCity_UnusedMart",  # 100.0% match, verified
    "LITTLEROOT TOWN BRENDANS HOUSE 1F": "LittlerootTown_BrendansHouse_1F",  # 100.0% match, verified
    "LITTLEROOT TOWN BRENDANS HOUSE 2F": "LittlerootTown_BrendansHouse_2F",  # 100.0% match, verified
    "LITTLEROOT TOWN MAYS HOUSE 1F": "LittlerootTown_MaysHouse_1F",  # 100.0% match, verified
    "LITTLEROOT TOWN MAYS HOUSE 2F": "LittlerootTown_MaysHouse_2F",  # 100.0% match, verified
    "MAGMA HIDEOUT 1F": "MagmaHideout_1F",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 1R": "MagmaHideout_2F_1R",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 2R": "MagmaHideout_2F_2R",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 3R": "MagmaHideout_2F_3R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 1R": "MagmaHideout_3F_1R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 2R": "MagmaHideout_3F_2R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 3R": "MagmaHideout_3F_3R",  # 100.0% match, verified
    "MAGMA HIDEOUT 4F": "MagmaHideout_4F",  # 100.0% match, verified
    "MAP RUSTURF TUNNEL": "RusturfTunnel",  # 89.7% match, verified
    "MARINE CAVE END": "MarineCave_End",  # 100.0% match, verified
    "MARINE CAVE ENTRANCE": "MarineCave_Entrance",  # 100.0% match, verified
    "MAUVILLE CITY BIKE SHOP": "MauvilleCity_BikeShop",  # 100.0% match, verified
    "MAUVILLE CITY GAME CORNER": "MauvilleCity_GameCorner",  # 100.0% match, verified
    "MAUVILLE CITY GYM": "MauvilleCity_Gym",  # 100.0% match, verified
    "MAUVILLE CITY HOUSE1": "MauvilleCity_House1",  # 100.0% match, verified
    "MAUVILLE CITY HOUSE2": "MauvilleCity_House2",  # 100.0% match, verified
    "MAUVILLE CITY MART": "MauvilleCity_Mart",  # 100.0% match, verified
    "MAUVILLE CITY POKEMON CENTER 1F": "MauvilleCity_PokemonCenter_1F",  # 100.0% match, verified
    "MAUVILLE CITY POKEMON CENTER 2F": "MauvilleCity_PokemonCenter_2F",  # 100.0% match, verified
    "METEOR FALLS 1F 1R": "MeteorFalls_1F_1R",  # 100.0% match, verified
    "METEOR FALLS 1F 2R": "MeteorFalls_1F_2R",  # 100.0% match, verified
    "METEOR FALLS B1F 1R": "MeteorFalls_B1F_1R",  # 100.0% match, verified
    "METEOR FALLS B1F 2R": "MeteorFalls_B1F_2R",  # 100.0% match, verified
    "METEOR FALLS STEVENS CAVE": "MeteorFalls_StevensCave",  # 100.0% match, verified
    "MIRAGE TOWER 1F": "MirageTower_1F",  # 100.0% match, verified
    "MIRAGE TOWER 2F": "MirageTower_2F",  # 100.0% match, verified
    "MIRAGE TOWER 3F": "MirageTower_3F",  # 100.0% match, verified
    "MIRAGE TOWER 4F": "MirageTower_4F",  # 100.0% match, verified
    "MOSSDEEP CITY GAME CORNER 1F": "MossdeepCity_GameCorner_1F",  # 100.0% match, verified
    "MOSSDEEP CITY GAME CORNER B1F": "MossdeepCity_GameCorner_B1F",  # 100.0% match, verified
    "MOSSDEEP CITY GYM": "MossdeepCity_Gym",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE1": "MossdeepCity_House1",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE2": "MossdeepCity_House2",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE3": "MossdeepCity_House3",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE4": "MossdeepCity_House4",  # 100.0% match, verified
    "MOSSDEEP CITY MART": "MossdeepCity_Mart",  # 100.0% match, verified
    "MOSSDEEP CITY POKEMON CENTER 1F": "MossdeepCity_PokemonCenter_1F",  # 100.0% match, verified
    "MOSSDEEP CITY POKEMON CENTER 2F": "MossdeepCity_PokemonCenter_2F",  # 100.0% match, verified
    "MOSSDEEP CITY SPACE CENTER 1F": "MossdeepCity_SpaceCenter_1F",  # 100.0% match, verified
    "MOSSDEEP CITY SPACE CENTER 2F": "MossdeepCity_SpaceCenter_2F",  # 100.0% match, verified
    "MOSSDEEP CITY STEVENS HOUSE": "MossdeepCity_StevensHouse",  # 100.0% match, verified
    "MT CHIMNEY": "MtChimney",  # 100.0% match, verified
    "MT CHIMNEY CABLE CAR STATION": "MtChimney_CableCarStation",  # 100.0% match, verified
    "MT PYRE 1F": "MtPyre_1F",  # 100.0% match, verified
    "MT PYRE 2F": "MtPyre_2F",  # 100.0% match, verified
    "MT PYRE 3F": "MtPyre_3F",  # 100.0% match, verified
    "MT PYRE 4F": "MtPyre_4F",  # 100.0% match, verified
    "MT PYRE 5F": "MtPyre_5F",  # 100.0% match, verified
    "MT PYRE 6F": "MtPyre_6F",  # 100.0% match, verified
    "MT PYRE EXTERIOR": "MtPyre_Exterior",  # 100.0% match, verified
    "MT PYRE SUMMIT": "MtPyre_Summit",  # 100.0% match, verified
    "NEW MAUVILLE ENTRANCE": "NewMauville_Entrance",  # 100.0% match, verified
    "NEW MAUVILLE INSIDE": "NewMauville_Inside",  # 100.0% match, verified
    "OLDALE TOWN HOUSE1": "OldaleTown_House1",  # 100.0% match, verified
    "OLDALE TOWN HOUSE2": "OldaleTown_House2",  # 100.0% match, verified
    "OLDALE TOWN MART": "OldaleTown_Mart",  # 100.0% match, verified
    "OLDALE TOWN POKEMON CENTER 1F": "OldaleTown_PokemonCenter_1F",  # 100.0% match, verified
    "OLDALE TOWN POKEMON CENTER 2F": "OldaleTown_PokemonCenter_2F",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE1": "PacifidlogTown_House1",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE2": "PacifidlogTown_House2",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE3": "PacifidlogTown_House3",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE4": "PacifidlogTown_House4",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE5": "PacifidlogTown_House5",  # 100.0% match, verified
    "PACIFIDLOG TOWN POKEMON CENTER 1F": "PacifidlogTown_PokemonCenter_1F",  # 100.0% match, verified
    "PACIFIDLOG TOWN POKEMON CENTER 2F": "PacifidlogTown_PokemonCenter_2F",  # 100.0% match, verified
    "PETALBURG CITY GYM": "PetalburgCity_Gym",  # 100.0% match, verified
    "PETALBURG CITY HOUSE1": "PetalburgCity_House1",  # 100.0% match, verified
    "PETALBURG CITY HOUSE2": "PetalburgCity_House2",  # 100.0% match, verified
    "PETALBURG CITY MART": "PetalburgCity_Mart",  # 100.0% match, verified
    "PETALBURG CITY POKEMON CENTER 1F": "PetalburgCity_PokemonCenter_1F",  # 100.0% match, verified
    "PETALBURG CITY POKEMON CENTER 2F": "PetalburgCity_PokemonCenter_2F",  # 100.0% match, verified
    "PETALBURG CITY WALLYS HOUSE": "PetalburgCity_WallysHouse",  # 100.0% match, verified
    "ROUTE 104 MR BRINEYS HOUSE": "Route104_MrBrineysHouse",  # 100.0% match, verified
    "ROUTE 104 MR BRINEYS HOUSE ALT": "Route104_MrBrineysHouse",  # 91.9% match, verified
    "ROUTE 104 PRETTY PETAL FLOWER SHOP": "Route104_PrettyPetalFlowerShop",  # 100.0% match, verified
    "ROUTE 109 SEASHORE HOUSE": "Route109_SeashoreHouse",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE CORRIDOR": "Route110_TrickHouseCorridor",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE END": "Route110_TrickHouseEnd",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE ENTRANCE": "Route110_TrickHouseEntrance",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE1": "Route110_TrickHousePuzzle1",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE2": "Route110_TrickHousePuzzle2",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE3": "Route110_TrickHousePuzzle3",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE4": "Route110_TrickHousePuzzle4",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE5": "Route110_TrickHousePuzzle5",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE6": "Route110_TrickHousePuzzle6",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE7": "Route110_TrickHousePuzzle7",  # 100.0% match, verified
    "ROUTE 111 OLD LADYS REST STOP": "Route111_OldLadysRestStop",  # 100.0% match, verified
    "ROUTE 111 WINSTRATE FAMILYS HOUSE": "Route111_WinstrateFamilysHouse",  # 100.0% match, verified
    "ROUTE 112 CABLE CAR STATION": "Route112_CableCarStation",  # 100.0% match, verified
    "ROUTE 113 GLASS WORKSHOP": "Route113_GlassWorkshop",  # 100.0% match, verified
    "ROUTE_113_GLASS_WORKSHOP": "Route113_GlassWorkshop",  # Enum name variant
    "ROUTE 114 FOSSIL MANIACS HOUSE": "Route114_FossilManiacsHouse",  # 100.0% match, verified
    "ROUTE 114 FOSSIL MANIACS TUNNEL": "Route114_FossilManiacsTunnel",  # 100.0% match, verified
    "ROUTE 114 LANETTES HOUSE": "Route114_LanettesHouse",  # 100.0% match, verified
    "ROUTE 116 TUNNELERS REST HOUSE": "Route116_TunnelersRestHouse",  # 100.0% match, verified
    "ROUTE 117 POKEMON DAY CARE": "Route117_PokemonDayCare",  # 100.0% match, verified
    "ROUTE 119 HOUSE": "Route119_House",  # 100.0% match, verified
    "ROUTE 119 WEATHER INSTITUTE 1F": "Route119_WeatherInstitute_1F",  # 100.0% match, verified
    "ROUTE 119 WEATHER INSTITUTE 2F": "Route119_WeatherInstitute_2F",  # 100.0% match, verified
    "ROUTE 121 SAFARI ZONE ENTRANCE": "Route121_SafariZoneEntrance",  # 100.0% match, verified
    "ROUTE 123 BERRY MASTERS HOUSE": "Route123_BerryMastersHouse",  # 100.0% match, verified
    "ROUTE 124 DIVING TREASURE HUNTERS HOUSE": "Route124_DivingTreasureHuntersHouse",  # 100.0% match, verified
    "RUSTBORO CITY CUTTERS HOUSE": "RustboroCity_CuttersHouse",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 1F": "RustboroCity_DevonCorp_1F",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 2F": "RustboroCity_DevonCorp_2F",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 3F": "RustboroCity_DevonCorp_3F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT1 1F": "RustboroCity_Flat1_1F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT1 2F": "RustboroCity_Flat1_2F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 1F": "RustboroCity_Flat2_1F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 2F": "RustboroCity_Flat2_2F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 3F": "RustboroCity_Flat2_3F",  # 100.0% match, verified
    "RUSTBORO CITY GYM": "RustboroCity_Gym",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE1": "RustboroCity_House1",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE2": "RustboroCity_House2",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE3": "RustboroCity_House3",  # 100.0% match, verified
    "RUSTBORO CITY MART": "RustboroCity_Mart",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON CENTER 1F": "RustboroCity_PokemonCenter_1F",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON CENTER 2F": "RustboroCity_PokemonCenter_2F",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON SCHOOL": "RustboroCity_PokemonSchool",  # 100.0% match, verified
    "SCORCHED SLAB": "ScorchedSlab",  # 100.0% match, verified
    "SEAFLOOR CAVERN ENTRANCE": "SeafloorCavern_Entrance",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM1": "SeafloorCavern_Room1",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM2": "SeafloorCavern_Room2",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM3": "SeafloorCavern_Room3",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM4": "SeafloorCavern_Room4",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM5": "SeafloorCavern_Room5",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM6": "SeafloorCavern_Room6",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM7": "SeafloorCavern_Room7",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM8": "SeafloorCavern_Room8",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM9": "SeafloorCavern_Room9",  # 100.0% match, verified
    "SEALED CHAMBER INNER ROOM": "SealedChamber_InnerRoom",  # 100.0% match, verified
    "SEALED CHAMBER OUTER ROOM": "SealedChamber_OuterRoom",  # 100.0% match, verified
    "SHOAL CAVE HIGH TIDE ENTRANCE ROOM": "ShoalCave_HighTideEntranceRoom",  # 100.0% match, verified
    "SHOAL CAVE HIGH TIDE INNER ROOM": "ShoalCave_HighTideInnerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE ENTRANCE ROOM": "ShoalCave_LowTideEntranceRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE ICE ROOM": "ShoalCave_LowTideIceRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE INNER ROOM": "ShoalCave_LowTideInnerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE LOWER ROOM": "ShoalCave_LowTideLowerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE STAIRS ROOM": "ShoalCave_LowTideStairsRoom",  # 100.0% match, verified
    "SKY PILLAR 1F": "SkyPillar_1F",  # 100.0% match, verified
    "SKY PILLAR 2F": "SkyPillar_2F",  # 100.0% match, verified
    "SKY PILLAR 3F": "SkyPillar_3F",  # 100.0% match, verified
    "SKY PILLAR 4F": "SkyPillar_4F",  # 100.0% match, verified
    "SKY PILLAR 5F": "SkyPillar_5F",  # 100.0% match, verified
    "SKY PILLAR ENTRANCE": "SkyPillar_Entrance",  # 100.0% match, verified
    "SKY PILLAR OUTSIDE": "SkyPillar_Outside",  # 100.0% match, verified
    "SKY PILLAR TOP": "SkyPillar_Top",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT BATTLE ROOM": "SlateportCity_BattleTentBattleRoom",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT CORRIDOR": "SlateportCity_BattleTentCorridor",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT LOBBY": "SlateportCity_BattleTentLobby",  # 100.0% match, verified
    "SLATEPORT CITY HARBOR": "SlateportCity_Harbor",  # 100.0% match, verified
    "SLATEPORT CITY HOUSE": "SlateportCity_House",  # 100.0% match, verified
    "SLATEPORT CITY MART": "SlateportCity_Mart",  # 100.0% match, verified
    "SLATEPORT CITY NAME RATERS HOUSE": "SlateportCity_NameRatersHouse",  # 100.0% match, verified
    "SLATEPORT CITY OCEANIC MUSEUM 1F": "SlateportCity_OceanicMuseum_1F",  # 100.0% match, verified
    "SLATEPORT CITY OCEANIC MUSEUM 2F": "SlateportCity_OceanicMuseum_2F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON CENTER 1F": "SlateportCity_PokemonCenter_1F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON CENTER 2F": "SlateportCity_PokemonCenter_2F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON FAN CLUB": "SlateportCity_PokemonFanClub",  # 100.0% match, verified
    "SLATEPORT CITY STERNS SHIPYARD 1F": "SlateportCity_SternsShipyard_1F",  # 100.0% match, verified
    "SLATEPORT CITY STERNS SHIPYARD 2F": "SlateportCity_SternsShipyard_2F",  # 100.0% match, verified
    "SOOTOPOLIS CITY GYM 1F": "SootopolisCity_Gym_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY GYM B1F": "SootopolisCity_Gym_B1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE1": "SootopolisCity_House1",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE2": "SootopolisCity_House2",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE3": "SootopolisCity_House3",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE4": "SootopolisCity_House4",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE5": "SootopolisCity_House5",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE6": "SootopolisCity_House6",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE7": "SootopolisCity_House7",  # 100.0% match, verified
    "SOOTOPOLIS CITY LOTAD AND SEEDOT HOUSE": "SootopolisCity_LotadAndSeedotHouse",  # 100.0% match, verified
    "SOOTOPOLIS CITY MART": "SootopolisCity_Mart",  # 100.0% match, verified
    "SOOTOPOLIS CITY MYSTERY EVENTS HOUSE 1F": "SootopolisCity_MysteryEventsHouse_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY MYSTERY EVENTS HOUSE B1F": "SootopolisCity_MysteryEventsHouse_B1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY POKEMON CENTER 1F": "SootopolisCity_PokemonCenter_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY POKEMON CENTER 2F": "SootopolisCity_PokemonCenter_2F",  # 100.0% match, verified
    "TERRA CAVE END": "TerraCave_End",  # 100.0% match, verified
    "TERRA CAVE ENTRANCE": "TerraCave_Entrance",  # 100.0% match, verified
    "UNDERWATER MARINE CAVE": "Underwater_MarineCave",  # 100.0% match, verified
    "UNDERWATER ROUTE 105": "Underwater_Route105",  # 100.0% match, verified
    "UNDERWATER ROUTE 124": "Underwater_Route124",  # 100.0% match, verified
    "UNDERWATER ROUTE 125": "Underwater_Route125",  # 100.0% match, verified
    "UNDERWATER ROUTE 126": "Underwater_Route126",  # 100.0% match, verified
    "UNDERWATER ROUTE 127": "Underwater_Route127",  # 100.0% match, verified
    "UNDERWATER ROUTE 128": "Underwater_Route128",  # 100.0% match, verified
    "UNDERWATER ROUTE 129": "Underwater_Route129",  # 100.0% match, verified
    "UNDERWATER ROUTE134": "Underwater_Route134",  # 100.0% match, verified
    "UNDERWATER SEAFLOOR CAVERN": "Underwater_SeafloorCavern",  # 100.0% match, verified
    "UNDERWATER SEALED CHAMBER": "Underwater_SealedChamber",  # 100.0% match, verified
    "UNDERWATER SOOTOPOLIS CITY": "Underwater_SootopolisCity",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT BATTLE ROOM": "VerdanturfTown_BattleTentBattleRoom",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT CORRIDOR": "VerdanturfTown_BattleTentCorridor",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT LOBBY": "VerdanturfTown_BattleTentLobby",  # 100.0% match, verified
    "VERDANTURF TOWN FRIENDSHIP RATERS HOUSE": "VerdanturfTown_FriendshipRatersHouse",  # 100.0% match, verified
    "VERDANTURF TOWN HOUSE": "VerdanturfTown_House",  # 100.0% match, verified
    "VERDANTURF TOWN MART": "VerdanturfTown_Mart",  # 100.0% match, verified
    "VERDANTURF TOWN POKEMON CENTER 1F": "VerdanturfTown_PokemonCenter_1F",  # 100.0% match, verified
    "VERDANTURF TOWN POKEMON CENTER 2F": "VerdanturfTown_PokemonCenter_2F",  # 100.0% match, verified
    "VERDANTURF TOWN WANDAS HOUSE": "VerdanturfTown_WandasHouse",  # 100.0% match, verified
    "VICTORY ROAD 1F": "VictoryRoad_1F",  # 100.0% match, verified
    "VICTORY ROAD B1F": "VictoryRoad_B1F",  # 100.0% match, verified
    "VICTORY ROAD B2F": "VictoryRoad_B2F",  # 100.0% match, verified
}

def _get_porymap_map_name(location_name: Optional[str]) -> Optional[str]:
    """Convert ROM location name to porymap map name."""
    if not location_name:
        return None
    return ROM_TO_PORYMAP_MAP.get(location_name)


def _format_red_map_info(location_name: Optional[str], player_coords: Optional[Tuple[int, int]], map_info: dict):
    """Format Red's native map data for the agent (equivalent to _format_porymap_info for Emerald).

    Returns same shape as _format_porymap_info: either (context_parts, map_data) tuple or plain list.
    """
    context_parts = []

    if not location_name or location_name in ('TITLE_SEQUENCE', 'Unknown'):
        return context_parts

    whole_map = map_info.get('red_whole_map')
    if not whole_map or not whole_map.get('grid'):
        return context_parts

    grid = whole_map['grid']
    dims = whole_map.get('dimensions', {})
    w, h = dims.get('width', 0), dims.get('height', 0)

    context_parts.append("\n=== MAP (FULL) ===")
    context_parts.append(f"Location: {location_name}")
    context_parts.append(f"Dimensions: {w}x{h}")

    # Build ASCII map string: Poké Balls as 'O', other NPCs as 'N', player as 'I' (wins ties)
    objects = whole_map.get('objects', [])
    # Map (y, x) → display symbol; Poké Balls get 'O', everything else gets 'N'
    obj_symbol: dict = {}
    for obj in objects:
        ny, nx = obj.get('y'), obj.get('x')
        if ny is None or nx is None:
            continue
        is_pokeball = "POKE_BALL" in obj.get('sprite_name', '').upper()
        obj_symbol[(int(ny), int(nx))] = 'O' if is_pokeball else 'N'

    ascii_lines = []
    for y, row in enumerate(grid):
        line = list(row)
        # Overlay objects first (Poké Balls as 'O', NPCs as 'N')
        for (oy, ox), sym in obj_symbol.items():
            if oy == y and 0 <= ox < len(line):
                line[ox] = sym
        # Player always on top
        if player_coords and y == player_coords[1]:
            px = player_coords[0]
            if 0 <= px < len(line):
                line[px] = 'I'
        ascii_lines.append(''.join(line))
    ascii_map = '\n'.join(ascii_lines)

    context_parts.append("\nASCII Map:")
    context_parts.append(ascii_map)
    context_parts.append("(Legend: I=Player .=walkable #=wall t=cuttable tree (use HM01 Cut) ~=grass W=water D=door G=Card Key gate (blocked, walk into and press A) !=sign ?=hidden item O=pokéball ↓/←/→=ledge C=counter B=bookshelf U=trash ^=display/blueprint P=computer '='=bench T=TV/machine N=NPC *=spinner stop)")

    # Compact JSON summary (warp_events, bg_events, objects)
    compact_json = {
        "name": location_name,
        "dimensions": dims,
        "warp_events": whole_map.get('warp_events', []),
        "bg_events":   whole_map.get('bg_events', []),
        "objects":     whole_map.get('objects', []),
    }
    context_parts.append("\nMap Data (JSON):")
    context_parts.append(json.dumps(compact_json, indent=2))

    # Return same tuple shape as _format_porymap_info
    # map_data['warps'] is populated from warp_events for pathfinder compat
    map_data = {
        'grid':       grid,
        'raw_tiles':  whole_map.get('raw_tiles'),
        'dimensions': dims,
        'objects':    whole_map.get('objects', []),
        'warps':      whole_map.get('warp_events', []),
    }

    logger.info(f"Red map: formatted '{location_name}' ({w}x{h}) with {len(whole_map.get('warp_events', []))} warps, {len(whole_map.get('objects', []))} objects")
    return context_parts, map_data


def _format_porymap_info(location_name: Optional[str], player_coords: Optional[Tuple[int, int]] = None, badge_count: int = 0) -> List[str]:
    """
    Format porymap ground truth data (JSON and ASCII map) for the agent.
    
    Args:
        location_name: Current location name from ROM
        player_coords: Player's (x, y) coordinates
        badge_count: Number of badges player has (for game-state-aware map selection)
    
    Returns list of formatted strings to add to context.
    """
    context_parts = []
    
    if not location_name or location_name == 'TITLE_SEQUENCE' or location_name == 'Unknown':
        return context_parts
    
    try:
        # Import here to avoid circular dependencies and allow graceful failure
        from utils.porymap_json_builder import build_json_map_for_llm
        from utils.pokeemerald_parser import PokeemeraldMapLoader
        
        # Get pokeemerald root
        pokeemerald_root = _get_pokeemerald_root()
        if not pokeemerald_root:
            logger.warning(f"Porymap: Could not find pokeemerald root for location '{location_name}'")
            return context_parts
        
        # Convert ROM location name to porymap map name using mapping
        porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)

        # Debug log for Glass Workshop issue
        if "GLASS" in location_name.upper() or "WORKSHOP" in location_name.upper():
            logger.warning(f"DEBUG: Glass Workshop mapping - ROM location: '{location_name}' -> porymap: '{porymap_map_name}'")

        # If not in direct mapping, try fuzzy matching
        if not porymap_map_name:
            map_loader = PokeemeraldMapLoader(pokeemerald_root)
            
            def normalize_for_matching(name: str) -> str:
                """Normalize location name for fuzzy matching."""
                # Normalize: lowercase, remove spaces/underscores, remove common suffixes
                normalized = str(name).lower().replace(" ", "").replace("_", "").replace("town", "").replace("city", "").replace("route", "")
                # Also try removing "professor", "birchs", "lab" for building matching
                if "professor" in normalized or "birch" in normalized or "lab" in normalized:
                    normalized = normalized.replace("professor", "").replace("birchs", "").replace("birch", "").replace("lab", "")
                return normalized
            
            rom_normalized = normalize_for_matching(location_name)
            maps_dir = pokeemerald_root / "data" / "maps"

            # Debug log for Glass Workshop fuzzy matching
            if "GLASS" in location_name.upper() or "WORKSHOP" in location_name.upper():
                logger.warning(f"DEBUG: Fuzzy matching Glass Workshop - normalized: '{rom_normalized}'")

            if maps_dir.exists():
                # Try direct directory name match
                best_match = None
                best_match_score = 0

                for map_dir in maps_dir.iterdir():
                    if not map_dir.is_dir() or map_dir.name == "map_groups.json":
                        continue
                    
                    map_name = map_dir.name
                    map_normalized = normalize_for_matching(map_name)
                    
                    # Exact match
                    if rom_normalized == map_normalized:
                        porymap_map_name = map_name
                        logger.info(f"Porymap: Matched '{location_name}' to '{porymap_map_name}' via fuzzy match")
                        break
                    
                    # Partial match scoring (for cases like "LITTLEROOT TOWN PROFESSOR BIRCHS LAB" -> "LittlerootTown_ProfessorBirchsLab")
                    if rom_normalized in map_normalized or map_normalized in rom_normalized:
                        match_length = min(len(rom_normalized), len(map_normalized))
                        if match_length > best_match_score and match_length > 5:  # Require at least 5 chars match
                            best_match = map_name
                            best_match_score = match_length
                
                # Use best partial match if no exact match found
                if not porymap_map_name and best_match:
                    porymap_map_name = best_match
                    logger.info(f"Porymap: Matched '{location_name}' to '{porymap_map_name}' via partial match (score: {best_match_score})")
        
        if not porymap_map_name:
            logger.warning(f"Porymap: Could not map ROM location '{location_name}' to porymap map name")
            return context_parts
        
        logger.info(f"Porymap: Building map for '{porymap_map_name}' (ROM location: '{location_name}', badges: {badge_count})")
        
        # Build JSON map (with grid included for pathfinding, even though we don't show it in prompt)
        # Pass badge_count for game-state-aware map selection (e.g., Petalburg Gym lobby)
        try:
            json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root, badge_count=badge_count)
        except ValueError as e:
            logger.error(f"Porymap: Failed to build map for '{porymap_map_name}' due to corrupted tileset data: {e}")
            logger.error("This likely indicates missing or corrupted tileset files in the porymap_data directory.")
            logger.error("Pathfinding will not be available for this location.")
            return context_parts
        
        # Ensure grid is built (even if we don't include it in the text output)
        if not json_map.get('grid'):
            # Rebuild with grid if needed
            from utils.porymap_json_builder import build_json_map
            try:
                json_map_with_grid = build_json_map(porymap_map_name, pokeemerald_root, include_grid=True, include_ascii=True)
                if json_map_with_grid and json_map_with_grid.get('grid'):
                    json_map['grid'] = json_map_with_grid['grid']
            except ValueError as e:
                logger.error(f"Porymap: Failed to rebuild grid for '{porymap_map_name}': {e}")

        if not json_map:
            logger.warning(f"Porymap: Failed to build JSON map for '{porymap_map_name}'")
            return context_parts

        # Filter grid based on player elevation to handle multi-level maps
        # For caves/dungeons with multiple connected levels, be more permissive
        # For buildings/bridges with truly separate floors, be strict
        if player_coords and json_map.get('raw_tiles') and json_map.get('grid'):
            try:
                px, py = player_coords[0], player_coords[1]
                raw_tiles = json_map['raw_tiles']

                # Get player's current elevation from the tile they're standing on
                if 0 <= py < len(raw_tiles) and 0 <= px < len(raw_tiles[py]):
                    player_tile = raw_tiles[py][px]
                    if len(player_tile) >= 4:
                        player_elevation = player_tile[3]  # elevation is 4th element

                        # Check if this is a cave/dungeon (has elevation variety but connected paths)
                        # vs a multi-floor building (strict separation)
                        elevations_in_map = set()
                        for row in raw_tiles:
                            for tile in row:
                                if len(tile) >= 4:
                                    elevations_in_map.add(tile[3])

                        # NO tolerance - only allow exact elevation matches
                        # Elevation changes ONLY through stairs/doors/ledges (handled separately)
                        elevation_tolerance = 0  # Must be exact same elevation

                        # First pass: Find all stair and warp positions (S, D, arrow tiles, and ladders)
                        grid = json_map['grid']
                        warp_positions = set()
                        arrow_positions = set()  # Non-warp stairs (directional arrows - ledges)
                        ladder_positions = set()  # Ladder tiles that connect elevations
                        for y in range(len(grid)):
                            for x in range(len(grid[y])):
                                if grid[y][x] in ['S', 'D']:  # Stairs and Doors are both warps
                                    warp_positions.add((x, y))
                                elif grid[y][x] in ['←', '→', '↑', '↓']:  # Arrow tiles (ledges)
                                    arrow_positions.add((x, y))
                                elif grid[y][x] == '&':  # Ladder/bridge tiles
                                    ladder_positions.add((x, y))

                        # Combine all types of elevation connectors
                        # Ladders (&) connect different elevations vertically
                        all_stair_positions = warp_positions | arrow_positions | ladder_positions

                        # Build elevation connectivity graph from ladders AND adjacent walkable tiles
                        # Ladders (&) connect elevations, but also regular tiles at adjacent different elevations (slopes)
                        connected_elevations = set([player_elevation])  # Start with player's elevation

                        # Iteratively find all connected elevations (BFS)
                        prev_size = 0
                        max_iterations = 10  # Prevent infinite loops
                        iteration = 0
                        while len(connected_elevations) != prev_size and iteration < max_iterations:
                            prev_size = len(connected_elevations)
                            iteration += 1

                            # Detect direct walkable connections (slopes between elevations)
                            # Check for adjacent walkable tiles at different elevations
                            for y in range(len(grid)):
                                for x in range(len(grid[y])):
                                    if y < len(raw_tiles) and x < len(raw_tiles[y]):
                                        tile = raw_tiles[y][x]
                                        if len(tile) >= 4 and grid[y][x] in ['.', '~']:
                                            tile_elev = tile[3]
                                            if tile_elev in connected_elevations:
                                                # Check adjacent tiles in all 4 directions
                                                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                                                    nx, ny = x + dx, y + dy
                                                    if 0 <= ny < len(raw_tiles) and 0 <= nx < len(raw_tiles[ny]):
                                                        neighbor_tile = raw_tiles[ny][nx]
                                                        if len(neighbor_tile) >= 4:
                                                            neighbor_elev = neighbor_tile[3]
                                                            if ny < len(grid) and nx < len(grid[ny]):
                                                                neighbor_char = grid[ny][nx]
                                                                # If adjacent tile is walkable and at different elevation, connect them
                                                                if neighbor_char in ['.', '~', 'S', 'D'] and neighbor_elev != tile_elev:
                                                                    connected_elevations.add(neighbor_elev)

                            # Also check ladder tiles for connections
                            for lx, ly in ladder_positions:
                                if ly < len(raw_tiles) and lx < len(raw_tiles[ly]):
                                    ladder_tile = raw_tiles[ly][lx]
                                    if len(ladder_tile) >= 4:
                                        ladder_elev = ladder_tile[3]
                                        if ladder_elev in connected_elevations:
                                            # Check tiles in all 4 directions (ladders can connect up/down/left/right)
                                            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                                                nx, ny = lx + dx, ly + dy
                                                if 0 <= ny < len(raw_tiles) and 0 <= nx < len(raw_tiles[ny]):
                                                    neighbor_tile = raw_tiles[ny][nx]
                                                    if len(neighbor_tile) >= 4:
                                                        neighbor_elev = neighbor_tile[3]
                                                        # Add elevation from walkable tiles OR other ladders
                                                        if ny < len(grid) and nx < len(grid[ny]):
                                                            neighbor_char = grid[ny][nx]
                                                            if neighbor_char in ['.', '~', 'S', 'D', '&']:
                                                                connected_elevations.add(neighbor_elev)

                        # Filter the grid based on elevation
                        filtered_grid = []

                        for y in range(len(grid)):
                            filtered_row = []
                            for x in range(len(grid[y])):
                                original_char = grid[y][x]

                                # Always preserve special markers (warps, doors, NPCs, items, PC, TV/notebook)
                                if original_char in ['D', 'S', 'T', 'K', 'N', 'I', 'P', 'V', '←', '→', '↑', '↓']:
                                    filtered_row.append(original_char)
                                elif y < len(raw_tiles) and x < len(raw_tiles[y]):
                                    tile = raw_tiles[y][x]
                                    if len(tile) >= 4:
                                        tile_elevation = tile[3]
                                        elevation_diff = abs(tile_elevation - player_elevation)

                                        # Check if tile is adjacent to stairs (immediate neighbors only)
                                        # This includes both warps (S/D) and arrow tiles (←/→/↑/↓)
                                        is_adjacent_to_stairs = False
                                        for stair_x, stair_y in all_stair_positions:
                                            if abs(x - stair_x) + abs(y - stair_y) == 1:
                                                is_adjacent_to_stairs = True
                                                break

                                        # Adjacent to stairs: keep WALKABLE tiles accessible regardless of elevation
                                        # But still block walls/cliffs!
                                        if is_adjacent_to_stairs and original_char in ['.', '~', '←', '→', '↑', '↓']:
                                            filtered_row.append(original_char)  # Walkable tiles near stairs stay walkable
                                        # Handle bridge tiles (&) based on whether there's a path underneath
                                        elif original_char == '&':
                                            # Check if bridge has adjacent walkable tiles (. & & & . pattern)
                                            # This indicates a ground path underneath the bridge
                                            # Need to search through consecutive & tiles to find ground at both ends
                                            has_ground_path = False
                                            if 0 <= y < len(grid):
                                                # Search left through consecutive bridge tiles to find ground
                                                left_walkable = False
                                                search_x = x - 1
                                                while search_x >= 0 and search_x < len(grid[y]):
                                                    search_char = grid[y][search_x]
                                                    if search_char == '&':
                                                        # Continue searching left through bridge
                                                        search_x -= 1
                                                    elif search_char in ['.', '~']:
                                                        # Found walkable ground - check elevation
                                                        if search_x < len(raw_tiles[y]):
                                                            search_tile = raw_tiles[y][search_x]
                                                            if len(search_tile) >= 4:
                                                                search_elev = search_tile[3]
                                                                if abs(search_elev - player_elevation) <= elevation_tolerance:
                                                                    left_walkable = True
                                                        break
                                                    else:
                                                        # Hit a non-ground, non-bridge tile
                                                        break

                                                # Search right through consecutive bridge tiles to find ground
                                                right_walkable = False
                                                search_x = x + 1
                                                while search_x < len(grid[y]):
                                                    search_char = grid[y][search_x]
                                                    if search_char == '&':
                                                        # Continue searching right through bridge
                                                        search_x += 1
                                                    elif search_char in ['.', '~']:
                                                        # Found walkable ground - check elevation
                                                        if search_x < len(raw_tiles[y]):
                                                            search_tile = raw_tiles[y][search_x]
                                                            if len(search_tile) >= 4:
                                                                search_elev = search_tile[3]
                                                                if abs(search_elev - player_elevation) <= elevation_tolerance:
                                                                    right_walkable = True
                                                        break
                                                    else:
                                                        # Hit a non-ground, non-bridge tile
                                                        break

                                                # Ground path exists if BOTH left and right ends are walkable at player elevation
                                                has_ground_path = left_walkable and right_walkable

                                            # If there's a ground path underneath, show as walkable
                                            if has_ground_path and tile_elevation > player_elevation:
                                                filtered_row.append('.')  # Can walk under bridge
                                            else:
                                                filtered_row.append('&')  # Keep bridge visible for pathfinding
                                        # Block tiles beyond elevation tolerance
                                        elif elevation_diff > elevation_tolerance:
                                            # Check if tile's elevation is connected via ladders
                                            if tile_elevation in connected_elevations:
                                                filtered_row.append(original_char)  # Allow tiles at connected elevations
                                            # Special case: If player is in water, allow ground/grass tiles to show
                                            # (player can surf to shore, but shore players can't access water - handled by pathfinding)
                                            elif 0 <= py < len(grid) and 0 <= px < len(grid[py]) and grid[py][px] == 'W' and original_char in ['.', '~']:
                                                filtered_row.append(original_char)  # Allow ground tiles from water
                                            else:
                                                filtered_row.append('#')  # Block tiles at very different elevations
                                        else:
                                            filtered_row.append(original_char)  # Keep original tile
                                    else:
                                        filtered_row.append(original_char)
                                else:
                                    filtered_row.append(original_char)
                            filtered_grid.append(filtered_row)

                        # Update the grid and ASCII map
                        json_map['grid'] = filtered_grid

                        # Regenerate ASCII from filtered grid unless this map uses override ASCII
                        # (override ASCII must be kept verbatim so P/V/K/S/I/D match the override)
                        if not json_map.get('ascii_from_override'):
                            ascii_lines = [''.join(row) for row in filtered_grid]
                            json_map['ascii'] = '\n'.join(ascii_lines)

                        # Count how many tiles were blocked by elevation filtering
                        blocked_count = sum(1 for row in filtered_grid for cell in row if cell == '#')
                        original_blocked_count = sum(1 for row in grid for cell in row if cell == '#')
                        newly_blocked = blocked_count - original_blocked_count

                        logger.info(f"Elevation filtering: player at ({px}, {py}) elevation {player_elevation} - map has elevations {sorted(elevations_in_map)} - tolerance: {elevation_tolerance} - blocked {newly_blocked} additional tiles")
            except Exception as e:
                logger.warning(f"Failed to filter map by elevation: {e}")
        
        # Format for LLM
        context_parts.append("\n=== PORYMAP GROUND TRUTH MAP ===")
        context_parts.append(f"Location: {json_map.get('name', porymap_map_name)}")
        context_parts.append(f"Dimensions: {json_map['dimensions']['width']}x{json_map['dimensions']['height']}")
        
        # Add ASCII map with player position marked
        if json_map.get('ascii'):
            ascii_map = json_map['ascii']

            # Insert player position 'P' if provided
            if player_coords and json_map.get('grid'):
                px, py = player_coords[0], player_coords[1]
                grid = json_map['grid']

                # Check if player position is within map bounds
                if 0 <= py < len(grid) and 0 <= px < len(grid[0]) if grid else False:
                    # Split ASCII map into lines
                    ascii_lines = ascii_map.split('\n')

                    # Find the line corresponding to player's Y coordinate
                    if py < len(ascii_lines):
                        line = list(ascii_lines[py])
                        # Replace character at player's X position with 'P'
                        if px < len(line):
                            original_char = line[px]
                            line[px] = 'P'
                            ascii_lines[py] = ''.join(line)
                            ascii_map = '\n'.join(ascii_lines)
            
            context_parts.append("\nASCII Map:")
            context_parts.append(ascii_map)
            context_parts.append("(Legend: 'P' = Player, '.' = walkable, '#' = blocked, 'I' = item, '~' = tall grass, 'X' = out of bounds, 'T' = TV, 'K' = Clock, 'S' = Stairs/Warp, 'D' = Door)")
        
        # NOTE: Warps, Objects/NPCs, and Connections lists are DEPRECATED
        # This data is already included in the Map Data (JSON) section below.
        # Removed to reduce redundancy and potential agent confusion.
        
        # Add compact JSON map data (simplified format to save tokens)
        context_parts.append("\nMap Data (JSON):")
        
        # Include full object details (retain additional fields for precision)
        objects_for_json = json_map.get('objects', [])
        
        # Simplified warps
        simplified_warps = []
        for warp in json_map.get('warps', []):
            simplified_warps.append({
                "x": warp.get('x', 0),
                "y": warp.get('y', 0),
                "elevation": warp.get('elevation', 0),
                "dest_map": warp.get('dest_map', '?'),
                "dest_warp_id": warp.get('dest_warp_id', 0)
            })
        
        # Simplified connections
        simplified_connections = []
        for conn in json_map.get('connections', []):
            simplified_connections.append({
                "direction": conn.get('direction', '?'),
                "offset": conn.get('offset', 0),
                "map": conn.get('map', '?')
            })
        
        # BG events (PC, clock, TV, notebook, etc.)
        bg_events_for_json = json_map.get('bg_events', [])
        
        # Build compact JSON map (matching example format)
        compact_json_map = {
            "name": json_map.get('name'),
            "id": json_map.get('id'),
            "dimensions": json_map.get('dimensions'),
            "warps": simplified_warps,
            "objects": objects_for_json,
            "bg_events": bg_events_for_json,
            "connections": simplified_connections
        }
        context_parts.append(json.dumps(compact_json_map, indent=2))
        
        logger.info(f"Porymap: Successfully added map data for '{porymap_map_name}' ({json_map['dimensions']['width']}x{json_map['dimensions']['height']})")
        
        # Store porymap data in context_parts for later extraction (hidden from LLM but accessible for pathfinding)
        # We'll store it as a special marker that can be extracted from state
        return context_parts, json_map  # Return both formatted text and raw data
        
    except ImportError as e:
        # Log import errors as warnings
        logger.warning(f"Porymap modules not available: {e}")
    except Exception as e:
        # Log errors but don't break state formatting
        logger.warning(f"Error adding porymap info for location '{location_name}': {e}", exc_info=True)
    
    # Return formatted text only if json_map not available
    return context_parts

def _format_game_state(game_data, state_data=None, include_movement_preview=True):
    """Format game state information (for non-battle mode)."""
    context_parts = []
    
    if not game_data:
        return context_parts
    
    context_parts.append("\n=== GAME STATE ===")
    
    # Note: Battle info is handled separately in battle mode
    # This is for showing game state when NOT in battle
    
    # Dialogue detection and validation (only show when not in battle)
    is_in_battle = game_data.get('is_in_battle', False) or game_data.get('in_battle', False)
    
    if not is_in_battle:
        dialog_text = game_data.get('dialog_text')
        dialogue_detected = game_data.get('dialogue_detected', {})
        
        if dialog_text and dialogue_detected.get('has_dialogue', False):
            # Only show dialogue if it's actually visible and active
            context_parts.append(f"\n--- DIALOGUE ---")
            if dialogue_detected.get('confidence') is not None:
                context_parts.append(f"Detection confidence: {dialogue_detected['confidence']:.1%}")
            context_parts.append(f"Text: {dialog_text}")
            # Note: Residual/invisible dialogue text is completely hidden from agent
    
    # Check if we're in title sequence and override game state
    player_location = state_data.get('player', {}).get('location', '') if state_data else ''
    if player_location == 'TITLE_SEQUENCE':
        context_parts.append(f"Game State: title")
    elif 'game_state' in game_data:
        context_parts.append(f"Game State: {game_data['game_state']}")
    
    # Get player data from state_data if provided
    player_data = state_data.get('player', {}) if state_data else {}
    
    # Add helpful prompt for title sequence
    player_location = player_data.get('location', '')
    if player_location == 'TITLE_SEQUENCE':
        context_parts.append("")
        context_parts.append("💡 TIP: Make sure to choose a fun name for your character!")
        context_parts.append("Be creative and have fun with the naming!")
    
    # Add movement preview for overworld navigation (but not during title sequence)
    # Can be disabled for agents using pathfinding utility
    if (include_movement_preview and state_data and not is_in_battle and 
        game_data.get('game_state') == 'overworld' and 
        player_location != 'TITLE_SEQUENCE'):
        movement_preview = format_movement_preview_for_llm(state_data)
        if movement_preview:
            context_parts.append("")
            context_parts.append(movement_preview)
    
    return context_parts

def _format_debug_info(state_data):
    """Format additional debug information."""
    context_parts = []
    
    context_parts.append("\n=== DEBUG INFO ===")
    
    # Step information
    if 'step_number' in state_data:
        context_parts.append(f"Step Number: {state_data['step_number']}")
    
    if 'status' in state_data:
        context_parts.append(f"Status: {state_data['status']}")
    
    # Visual data info
    if 'visual' in state_data:
        visual = state_data['visual']
        if 'resolution' in visual:
            context_parts.append(f"Resolution: {visual['resolution']}")
        if 'screenshot_base64' in visual:
            context_parts.append(f"Screenshot: Available ({len(visual['screenshot_base64'])} chars)")
    
    return context_parts

# Convenience functions for specific use cases

def get_movement_options(state_data):
    """
    Extract movement options from traversability data.
    
    Returns:
        dict: Direction -> description mapping
    """
    map_info = state_data.get('map', {})
    if 'traversability' not in map_info or not map_info['traversability']:
        return {}
    
    traversability = map_info['traversability']
    center_y = len(traversability) // 2
    center_x = len(traversability[0]) // 2
    
    directions = {
        'UP': (0, -1), 'DOWN': (0, 1), 
        'LEFT': (-1, 0), 'RIGHT': (1, 0)
    }
    
    movement_options = {}
    for direction, (dx, dy) in directions.items():
        new_x, new_y = center_x + dx, center_y + dy
        if 0 <= new_y < len(traversability) and 0 <= new_x < len(traversability[new_y]):
            cell = str(traversability[new_y][new_x])
            if cell == "0":
                movement_options[direction] = "BLOCKED"
            elif cell == ".":
                movement_options[direction] = "Normal path"
            elif "TALL" in cell:
                movement_options[direction] = "Tall grass (wild encounters)"
            elif "WATER" in cell:
                movement_options[direction] = "Water (need Surf)"
            else:
                movement_options[direction] = cell
        else:
            movement_options[direction] = "Out of bounds"
    
    return movement_options


def get_movement_preview(state_data):
    """
    Get detailed preview of what happens with each directional movement.
    Shows new coordinates and tile information for each direction.
    
    Args:
        state_data: Complete game state data
        
    Returns:
        dict: Direction -> preview info mapping
    """
    # Get current player position
    player_data = state_data.get('player', {})
    player_position = _get_player_position(player_data)
    
    if not player_position or 'x' not in player_position or 'y' not in player_position:
        # print( Movement preview - No player position. player_position={player_position}")
        return {}
    
    current_x = int(player_position['x'])
    current_y = int(player_position['y'])
    
    # Get map and tile data
    # IMPORTANT: Use the elevation-filtered porymap grid if available, not raw memory tiles
    map_info = state_data.get('map', {})

    # Try to use porymap grid first (elevation-filtered and more accurate)
    porymap = map_info.get('porymap', {})
    porymap_grid = porymap.get('grid')
    raw_tiles_for_elevation = porymap.get('raw_tiles') if porymap else None

    # Fallback to memory-read tiles if porymap not available
    raw_tiles = map_info.get('tiles', [])

    if not raw_tiles and not porymap_grid:
        # print( Movement preview - No tiles. map_info keys: {list(map_info.keys()) if map_info else 'None'}")
        return {}
    
    # Get NPCs from map info.
    # Emerald stores NPCs in map_info['object_events']; Red stores them in
    # porymap['objects'] (set by game_tools.py).  Fall back to porymap objects
    # so Red NPC blocking works correctly.
    npcs = map_info.get('object_events', []) or porymap.get('objects', [])
    
    directions = {
        'UP': (0, -1),
        'DOWN': (0, 1), 
        'LEFT': (-1, 0),
        'RIGHT': (1, 0)
    }
    
    movement_preview = {}
    
    # Player is at center of the 15x15 grid
    center_x = len(raw_tiles[0]) // 2 if raw_tiles and raw_tiles[0] else 7
    center_y = len(raw_tiles) // 2 if raw_tiles else 7
    
    # Get the tile the player is currently standing on
    current_tile_symbol = None
    if (0 <= center_y < len(raw_tiles) and 
        0 <= center_x < len(raw_tiles[center_y]) and
        raw_tiles[center_y] and raw_tiles[center_y][center_x]):
        current_tile = raw_tiles[center_y][center_x]
        current_tile_symbol = format_tile_to_symbol(current_tile)
    
    for direction, (dx, dy) in directions.items():
        # Calculate new world coordinates
        new_world_x = current_x + dx
        new_world_y = current_y + dy
        
        # Calculate grid position in the tile array
        grid_x = center_x + dx
        grid_y = center_y + dy
        
        preview_info = {
            'new_coords': (new_world_x, new_world_y),
            'blocked': True,
            'tile_symbol': '#',
            'tile_description': 'BLOCKED - Out of bounds',
            'npc_at_position': None,
            'npc_info': None
        }
        
        # Check if there's an NPC at the target position
        npc_at_position = None
        for npc in npcs:
            npc_x = npc.get('current_x', npc.get('x', 0))
            npc_y = npc.get('current_y', npc.get('y', 0))
            if npc_x == new_world_x and npc_y == new_world_y:
                npc_at_position = npc
                break
        
        # Check if the target position is within the grid bounds
        # Use porymap grid if available (elevation-aware), otherwise fall back to raw tiles
        if porymap_grid and 0 <= new_world_y < len(porymap_grid) and 0 <= new_world_x < len(porymap_grid[new_world_y] if porymap_grid[new_world_y] else []):
            # Use porymap grid (already filtered by elevation)
            try:
                tile_symbol = porymap_grid[new_world_y][new_world_x]
                target_tile = None  # Don't have the raw tile data when using porymap grid

                # Determine if movement is blocked by terrain (using porymap symbols).
                # Whitelist of all symbols that are passable in either Emerald or Red.
                # Everything else — furniture, interactables, walls — is blocked.
                # '?' = hidden item in Red (walkable); '!' = sign (blocked); 'O' = Poké Ball (blocked).
                _PASSABLE_SYMBOLS = {
                    ".", "~", "D", "S",
                    "↓", "←", "→", "↑", "↗", "↖", "↘", "↙",
                    "&", "?",
                }
                is_blocked_by_terrain = tile_symbol not in _PASSABLE_SYMBOLS

                # Check if movement is blocked by NPC
                is_blocked_by_npc = False
                if npc_at_position and npc_at_position.get('is_blocking', True):
                    is_blocked_by_npc = True

                # Overall blocking status
                is_blocked = is_blocked_by_terrain or is_blocked_by_npc

                # Set tile description based on symbol
                if tile_symbol == 'D':
                    tile_description = 'Door/Exit'
                elif tile_symbol == 'S':
                    tile_description = 'Stairs/Warp'
                elif tile_symbol == '.':
                    tile_description = 'Walkable'
                elif tile_symbol == '~':
                    tile_description = 'Grass'
                elif tile_symbol == '#':
                    tile_description = 'Wall'
                elif tile_symbol == 'W':
                    tile_description = 'Water'
                elif tile_symbol == '!':
                    tile_description = 'Sign/Signpost (cannot walk through)'
                elif tile_symbol == '?':
                    tile_description = 'Hidden item (walkable)'
                elif tile_symbol == 'O':
                    tile_description = 'Poké Ball (interact from adjacent tile)'
                elif tile_symbol == 't':
                    tile_description = 'Cuttable tree (use HM Cut)'
                else:
                    tile_description = f'Tile ({tile_symbol})'

                # Update preview info
                preview_info['blocked'] = is_blocked
                preview_info['tile_symbol'] = tile_symbol
                preview_info['tile_description'] = tile_description
                preview_info['npc_at_position'] = npc_at_position is not None
                if npc_at_position:
                    preview_info['npc_info'] = {
                        'graphics_id': npc_at_position.get('graphics_id', 'Unknown'),
                        'x': npc_at_position.get('x', 0),
                        'y': npc_at_position.get('y', 0)
                    }

            except (IndexError, TypeError):
                tile_symbol = '#'
                target_tile = None
                # Keep defaults (blocked=True)

        elif (0 <= grid_y < len(raw_tiles) and
            0 <= grid_x < len(raw_tiles[grid_y]) and
            raw_tiles[grid_y]):

            try:
                # Get the tile at the target position from memory-read tiles
                target_tile = raw_tiles[grid_y][grid_x]

                # Get tile symbol and check if walkable
                tile_symbol = format_tile_to_symbol(target_tile)
                
                # Determine if movement is blocked by terrain
                is_blocked_by_terrain = tile_symbol in ['#', 'W', 'N', 't']  # Walls, water, NPCs, and cuttable trees block movement
                
                # Check if movement is blocked by NPC
                is_blocked_by_npc = False
                if npc_at_position and npc_at_position.get('is_blocking', True):
                    is_blocked_by_npc = True
                
                # Overall blocking status
                is_blocked = is_blocked_by_terrain or is_blocked_by_npc
                
                # SPECIAL CASE: If player is standing on stairs/door, don't block the warp direction
                # Stairs and doors often require moving in a specific direction to activate
                if current_tile_symbol in ['S', 'D']:
                    # When on stairs/doors, typically you need to move forward to activate them
                    # Don't block any direction when on these tiles to allow proper navigation
                    # This ensures the agent can properly use warps/doors even if the destination
                    # tile might normally be considered blocked
                    if tile_symbol in ['#', 'W']:
                        # Override the blocking for navigation tiles but KEEP original symbol
                        is_blocked = is_blocked_by_npc  # Only block if NPC is present
                        # DO NOT change tile_symbol - preserve S, D, #, W, etc.
                
                # Special handling for jump ledges - they're only walkable in their direction
                if tile_symbol in ['↓', '↑', '←', '→', '↗', '↖', '↘', '↙']:
                    # Map directions to tile symbols
                    ledge_direction_map = {
                        'UP': '↑',
                        'DOWN': '↓', 
                        'LEFT': '←',
                        'RIGHT': '→'
                    }
                    
                    # Only allow movement if we're going in the direction the ledge points
                    if direction in ledge_direction_map:
                        allowed_symbol = ledge_direction_map[direction]
                        if tile_symbol != allowed_symbol:
                            is_blocked = True  # Block movement in wrong direction
                    else:
                        is_blocked = True  # Block diagonal movements for basic directional ledges
                
                # Get tile description
                if len(target_tile) >= 2:
                    tile_id, behavior = target_tile[:2]
                    
                    # Convert behavior to readable description
                    if hasattr(behavior, 'name'):
                        behavior_name = behavior.name
                    elif isinstance(behavior, int):
                        try:
                            # Use Red's enum for Red maps so that RedMetatileBehavior
                            # integer codes resolve to the correct names.
                            behavior_enum = _get_behavior_enum()(behavior)
                            behavior_name = behavior_enum.name
                        except (ValueError, ImportError):
                            behavior_name = f"BEHAVIOR_{behavior}"
                    else:
                        behavior_name = str(behavior)
                    
                    # Create human-readable description
                    # Check if we're overriding blocking due to being on stairs/door
                    is_override = current_tile_symbol in ['S', 'D'] and not is_blocked and tile_symbol in ['#', 'W']
                    
                    if is_override:
                        # We're on stairs/door and this normally blocked tile is walkable
                        if tile_symbol == '#':
                            tile_description = f"Walkable - Warp/Door exit (normally blocked) (ID: {tile_id})"
                        elif tile_symbol == 'W':
                            tile_description = f"Walkable - Warp/Door exit over water (ID: {tile_id})"
                    elif tile_symbol == '.':
                        tile_description = f"Walkable path (ID: {tile_id})"
                    elif tile_symbol == '#':
                        tile_description = f"BLOCKED - Wall/Obstacle (ID: {tile_id}, {behavior_name})"
                    elif tile_symbol == 't':
                        tile_description = f"BLOCKED - Cuttable tree (use HM Cut) (ID: {tile_id})"
                    elif tile_symbol == 'W':
                        tile_description = f"BLOCKED - Water (need Surf) (ID: {tile_id})"
                    elif tile_symbol == '~':
                        tile_description = f"Walkable - Tall grass (wild encounters) (ID: {tile_id})"
                    elif tile_symbol == 'D':
                        tile_description = f"Walkable - Door/Entrance (ID: {tile_id})"
                    elif tile_symbol == 'S':
                        tile_description = f"Walkable - Stairs/Warp (ID: {tile_id})"
                    elif tile_symbol in ['↓', '↑', '←', '→', '↗', '↖', '↘', '↙']:
                        # Ledge description based on whether movement is allowed
                        if is_blocked:
                            tile_description = f"BLOCKED - Jump ledge {tile_symbol} (wrong direction) (ID: {tile_id})"
                        else:
                            tile_description = f"Walkable - Jump ledge {tile_symbol} (correct direction) (ID: {tile_id})"
                    else:
                        tile_description = f"Walkable - {behavior_name} (ID: {tile_id})"
                else:
                    tile_description = "Unknown tile"
                
                # Add NPC information to description if present
                npc_info = None
                if npc_at_position:
                    npc_name = npc_at_position.get('name', 'Unknown')
                    npc_type = npc_at_position.get('npc_type', 'npc')
                    npc_description = npc_at_position.get('description', '')
                    
                    # Create NPC info string
                    npc_info = f"{npc_name} ({npc_type})"
                    if npc_description:
                        npc_info += f" - {npc_description}"
                    
                    # Update tile description to include NPC info
                    if is_blocked_by_npc:
                        if is_blocked_by_terrain:
                            tile_description += f" + BLOCKED by {npc_info}"
                        else:
                            tile_description = f"BLOCKED - {npc_info}"
                    else:
                        tile_description += f" + {npc_info} (non-blocking)"
                
                preview_info.update({
                    'blocked': is_blocked,
                    'tile_symbol': tile_symbol,
                    'tile_description': tile_description,
                    'npc_at_position': npc_at_position,
                    'npc_info': npc_info
                })
                
            except (IndexError, TypeError) as e:
                logger.warning(f"Error analyzing tile at {grid_x}, {grid_y}: {e}")
                # Keep default blocked values
                pass
        
        movement_preview[direction] = preview_info
    
    return movement_preview


def format_movement_preview_for_llm(state_data):
    """
    Format movement preview in a concise format suitable for LLM prompts.
    
    Args:
        state_data: Complete game state data
        
    Returns:
        str: Formatted movement preview text
    """
    preview = get_movement_preview(state_data)
    
    if not preview:
        return "Movement preview: Not available"
    
    lines = ["MOVEMENT PREVIEW:"]
    
    for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
        if direction in preview:
            info = preview[direction]
            new_x, new_y = info['new_coords']
            symbol = info['tile_symbol']
            status = "BLOCKED" if info['blocked'] else "WALKABLE"
            
            # Special override: if description confirms Stairs/Warp, show 'S' symbol.
            # ('W' must NOT be used here — it means Water in both Emerald and Red grids.)
            desc = info.get('tile_description', '')
            if not info['blocked'] and ('Stairs' in desc or 'Warp' in desc):
                symbol = 'S'
            
            lines.append(f"  {direction:5}: ({new_x:3},{new_y:3}) [{symbol}] {status}")
            
            # Add NPC information if present
            if info.get('npc_info'):
                npc_info = info['npc_info']
                if info['blocked'] and 'BLOCKED by' in info['tile_description']:
                    lines[-1] += f" - {npc_info}"
                elif not info['blocked'] and 'non-blocking' in info['tile_description']:
                    lines[-1] += f" - {npc_info}"
                else:
                    lines[-1] += f" - {npc_info}"
            
            # Add brief description for tiles
            desc = info['tile_description']
            if info['blocked']:
                # Special messages for blocked tiles
                if 'Jump ledge' in desc and 'wrong direction' in desc:
                    lines[-1] += " - Can only jump in arrow direction"
                elif 'Water' in desc:
                    lines[-1] += " - Need Surf to cross"
                elif 'Wall' in desc or 'Obstacle' in desc:
                    lines[-1] += " - Impassable"
                elif 'trainer' in desc.lower() or 'npc' in desc.lower():
                    lines[-1] += " - NPC blocks movement"
            else:
                # Add brief description for walkable tiles
                if 'Tall grass' in desc:
                    lines[-1] += " - Tall grass (wild encounters)"
                elif 'Stairs' in desc or 'Warp' in desc:
                    lines[-1] += " - Stairs/Warp"
                elif 'Door' in desc or 'Entrance' in desc:
                    lines[-1] += " - Door/Entrance"
                elif 'Jump ledge' in desc and 'correct direction' in desc:
                    lines[-1] += " - Jump ledge (can jump this way)"
                elif 'trainer' in desc.lower() or 'npc' in desc.lower():
                    lines[-1] += " - NPC present (interact with A)"
    
    return "\n".join(lines)


def format_action_history(action_history, max_actions=10):
    """
    Format action history with starting and ending positions.
    
    Args:
        action_history: List of action dicts with button, start_pos, end_pos
        max_actions: Maximum number of recent actions to show
    
    Returns:
        str: Formatted action history text
    """
    if not action_history:
        return "No recent actions"
    
    lines = []
    
    # Get the most recent completed actions
    completed_actions = [a for a in action_history if a.get('completed', False)]
    recent_actions = completed_actions[-max_actions:]
    
    if not recent_actions:
        return "No completed actions yet"
    
    lines.append("RECENT ACTION HISTORY:")
    lines.append("(Shows last {} actions with start → end positions)".format(len(recent_actions)))
    
    for i, action in enumerate(recent_actions, 1):
        button = action.get('button', 'UNKNOWN')
        start_pos = action.get('start_pos', (None, None, 'Unknown'))
        end_pos = action.get('end_pos', (None, None, 'Unknown'))
        
        start_x, start_y, start_loc = start_pos
        end_x, end_y, end_loc = end_pos
        sequence_index = action.get('sequence_index', 0)
        metadata = action.get('metadata') or {}
        source = action.get('source')

        # Build base line text
        if button in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if start_x is not None and end_x is not None:
                if start_x == end_x and start_y == end_y and start_loc == end_loc:
                    line = f"  {i}. {button:5} @ ({start_x:3},{start_y:3}) → BLOCKED (stayed at same position)"
                else:
                    if start_loc == end_loc:
                        line = f"  {i}. {button:5} @ ({start_x:3},{start_y:3}) → ({end_x:3},{end_y:3})"
                    else:
                        line = f"  {i}. {button:5} @ ({start_x:3},{start_y:3}) [{start_loc}] → ({end_x:3},{end_y:3}) [{end_loc}]"
            else:
                line = f"  {i}. {button:5} (position unavailable)"
        else:
            if start_x is not None:
                line = f"  {i}. {button:5} @ ({start_x:3},{start_y:3})"
            else:
                line = f"  {i}. {button:5}"

        # Append contextual metadata (only on first button of a sequence)
        context_parts = []
        if sequence_index == 0:
            if source == 'navigate_to':
                variance = metadata.get('variance')
                if variance is not None:
                    context_parts.append(f"navigate_to variance={variance}")
            elif source:
                context_parts.append(str(source))
            elif metadata:
                for key, value in metadata.items():
                    context_parts.append(f"{key}={value}")

        if context_parts:
            line += " [" + "; ".join(str(part) for part in context_parts) + "]"

        lines.append(line)
        
    return "\n".join(lines)


def get_party_health_summary(state_data):
    """
    Get a summary of party health status.
    
    Returns:
        dict: Summary with healthy_count, total_count, critical_pokemon
    """
    player_data = state_data.get('player', {})
    game_data = state_data.get('game', {})
    party_data = player_data.get('party') or game_data.get('party')
    
    if not party_data:
        return {"healthy_count": 0, "total_count": 0, "critical_pokemon": []}
    
    pokemon_list = []
    if isinstance(party_data, dict) and party_data.get('pokemon'):
        pokemon_list = party_data.get('pokemon', [])
    elif isinstance(party_data, list):
        pokemon_list = party_data
    
    healthy_count = 0
    critical_pokemon = []
    
    for i, pokemon in enumerate(pokemon_list[:6]):
        if pokemon:
            hp = pokemon.get('current_hp', 0)
            max_hp = pokemon.get('max_hp', 1)
            status = pokemon.get('status', 'OK')
            species = pokemon.get('species_name', pokemon.get('species', 'Unknown Pokemon'))
            
            # Check if healthy: has HP and no negative status (OK or Normal are both healthy)
            if hp > 0 and status in ['OK', 'Normal']:
                healthy_count += 1
            
            hp_percent = (hp / max_hp * 100) if max_hp > 0 else 0
            # Mark as critical if low HP or has a status condition
            if hp_percent < 25 or status not in ['OK', 'Normal']:
                critical_pokemon.append(f"{species} ({hp_percent:.0f}% HP, {status})")
    
    return {
        "healthy_count": healthy_count,
        "total_count": len(pokemon_list),
        "critical_pokemon": critical_pokemon
    } 
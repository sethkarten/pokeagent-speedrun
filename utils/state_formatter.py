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
from utils.map_formatter import format_map_grid, format_map_for_llm, generate_dynamic_legend, format_tile_to_symbol
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
        cache_file = '.pokeagent_cache/map_stitcher_data.json'
        if os.path.exists(cache_file):
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
    
    # Items
    item_count = game_data.get('item_count')
    if item_count is not None:
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
                    species = pokemon.get('species_name', pokemon.get('species', 'Unknown'))
                    level = pokemon.get('level', '?')
                    hp = pokemon.get('current_hp', '?')
                    max_hp = pokemon.get('max_hp', '?')
                    status = pokemon.get('status', 'Normal')
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
    if player_data and 'position' in player_data:
        pos = player_data['position']
        if pos:
            player_coords = (pos.get('x', 0), pos.get('y', 0))
            context_parts.append(f"Player Position (ROM): ({player_coords[0]}, {player_coords[1]})")
    
    # MapStitcher map display removed - using porymap ground truth instead
    
    # Add porymap ground truth data (JSON and ASCII map)
    porymap_result = _format_porymap_info(location_name, player_coords)
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
    elif porymap_result:
        context_parts.extend(porymap_result)
    
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
    
    # Professor Birch's Lab
    "LITTLEROOT TOWN PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",
    "PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",
}

def _format_porymap_info(location_name: Optional[str], player_coords: Optional[Tuple[int, int]] = None) -> List[str]:
    """
    Format porymap ground truth data (JSON and ASCII map) for the agent.
    
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
        
        logger.info(f"Porymap: Building map for '{porymap_map_name}' (ROM location: '{location_name}')")
        
        # Build JSON map (with grid included for pathfinding, even though we don't show it in prompt)
        json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root)
        
        # Ensure grid is built (even if we don't include it in the text output)
        if not json_map.get('grid'):
            # Rebuild with grid if needed
            from utils.porymap_json_builder import build_json_map
            json_map_with_grid = build_json_map(porymap_map_name, pokeemerald_root, include_grid=True, include_ascii=True)
            if json_map_with_grid and json_map_with_grid.get('grid'):
                json_map['grid'] = json_map_with_grid['grid']
        
        if not json_map:
            logger.warning(f"Porymap: Failed to build JSON map for '{porymap_map_name}'")
            return context_parts
        
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
            context_parts.append("(Legend: 'P' = Player, '.' = walkable, '#' = blocked, 'X' = out of bounds)")
        
        # Add warps
        warps = json_map.get('warps', [])
        if warps:
            context_parts.append(f"\nWarps ({len(warps)}):")
            for warp in warps[:10]:  # Limit to first 10
                dest = warp.get('dest_map', '?')
                context_parts.append(f"  At ({warp.get('x', 0)}, {warp.get('y', 0)}) → {dest}")
            if len(warps) > 10:
                context_parts.append(f"  ... and {len(warps) - 10} more warps")
        
        # Add objects (NPCs, items, etc.)
        objects = json_map.get('objects', [])
        if objects:
            context_parts.append(f"\nObjects/NPCs ({len(objects)}):")
            for obj in objects[:10]:  # Limit to first 10
                gfx_id = obj.get('graphics_id', '?')
                context_parts.append(f"  {gfx_id} at ({obj.get('x', 0)}, {obj.get('y', 0)})")
            if len(objects) > 10:
                context_parts.append(f"  ... and {len(objects) - 10} more objects")
        
        # Add connections
        connections = json_map.get('connections', [])
        if connections:
            context_parts.append(f"\nMap Connections ({len(connections)}):")
            for conn in connections[:5]:  # Limit to first 5
                direction = conn.get('direction', '?')
                target = conn.get('map', '?')
                context_parts.append(f"  {direction} → {target}")
            if len(connections) > 5:
                context_parts.append(f"  ... and {len(connections) - 5} more connections")
        
        # Add compact JSON map data (without grid to save tokens - ASCII map is sufficient)
        context_parts.append("\nMap Data (JSON):")
        # Exclude grid and ASCII to save tokens - ASCII is already shown above
        compact_json_map = {
            "name": json_map.get('name'),
            "id": json_map.get('id'),
            "dimensions": json_map.get('dimensions'),
            "warps": json_map.get('warps'),
            "objects": json_map.get('objects'),
            "connections": json_map.get('connections'),
            "metadata": json_map.get('metadata')
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
    map_info = state_data.get('map', {})
    raw_tiles = map_info.get('tiles', [])
    
    if not raw_tiles:
        # print( Movement preview - No tiles. map_info keys: {list(map_info.keys()) if map_info else 'None'}")
        return {}
    
    # Get NPCs from map info
    npcs = map_info.get('object_events', [])
    
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
        if (0 <= grid_y < len(raw_tiles) and 
            0 <= grid_x < len(raw_tiles[grid_y]) and
            raw_tiles[grid_y]):
            
            try:
                # Get the tile at the target position
                target_tile = raw_tiles[grid_y][grid_x]
                
                # Get tile symbol and check if walkable
                tile_symbol = format_tile_to_symbol(target_tile)
                
                # Determine if movement is blocked by terrain
                is_blocked_by_terrain = tile_symbol in ['#', 'W', 'N']  # Walls, water, and NPCs block movement
                
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
                            behavior_enum = MetatileBehavior(behavior)
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
            
            # Special override: if description contains "Stairs" or "Warp", show 'W' instead of any other symbol
            desc = info.get('tile_description', '')
            if not info['blocked'] and ('Stairs' in desc or 'Warp' in desc):
                symbol = 'W'
            
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
        
        # Check if movement actually occurred
        if button in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if start_x is not None and end_x is not None:
                if start_x == end_x and start_y == end_y and start_loc == end_loc:
                    # Movement blocked
                    lines.append(f"  {i}. {button:5} @ ({start_x:3},{start_y:3}) → BLOCKED (stayed at same position)")
                else:
                    # Movement succeeded
                    if start_loc == end_loc:
                        lines.append(f"  {i}. {button:5} @ ({start_x:3},{start_y:3}) → ({end_x:3},{end_y:3})")
                    else:
                        # Changed location (went through door/warp)
                        lines.append(f"  {i}. {button:5} @ ({start_x:3},{start_y:3}) [{start_loc}] → ({end_x:3},{end_y:3}) [{end_loc}]")
            else:
                lines.append(f"  {i}. {button:5} (position unavailable)")
        else:
            # Non-movement action (A, B, START, SELECT)
            if start_x is not None:
                lines.append(f"  {i}. {button:5} @ ({start_x:3},{start_y:3})")
            else:
                lines.append(f"  {i}. {button:5}")
    
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
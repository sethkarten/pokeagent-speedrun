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
        print(f"🗺️ DEBUG: Failed to read location connections from cache: {e}")
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

def format_state(state_data, format_type="summary", include_debug_info=False, include_npcs=True):
    """
    Format comprehensive state data into readable text.
    
    Args:
        state_data (dict): The comprehensive state from /state endpoint
        format_type (str): "summary" for one-line summary, "detailed" for multi-line LLM format
        include_debug_info (bool): Whether to include extra debug information (for detailed format)
        include_npcs (bool): Whether to include NPC information in the state
    
    Returns:
        str: Formatted state text
    """
    if format_type == "summary":
        return _format_state_summary(state_data)
    elif format_type == "detailed":
        return _format_state_detailed(state_data, include_debug_info, include_npcs)
    else:
        raise ValueError(f"Unknown format_type: {format_type}. Use 'summary' or 'detailed'")

def format_state_for_llm(state_data, include_debug_info=False, include_npcs=True):
    """
    Format comprehensive state data into a readable context for the VLM.
    
    Args:
        state_data (dict): The comprehensive state from /state endpoint
        include_debug_info (bool): Whether to include extra debug information
        include_npcs (bool): Whether to include NPC information in the state
    
    Returns:
        str: Formatted state context for LLM prompts
    """
    print(f"🗺️ DEBUG: format_state_for_llm received keys: {list(state_data.keys()) if state_data else 'None'}")
    if state_data and 'location_connections' in state_data:
        print(f"🗺️ DEBUG: location_connections found in state_data: {state_data['location_connections']}")
    else:
        print(f"🗺️ DEBUG: location_connections NOT found in state_data")
    if state_data and 'portal_connections' in state_data:
        print(f"🗺️ DEBUG: Portal connections found in format_state_for_llm: {state_data['portal_connections']}")
    return format_state(state_data, format_type="detailed", include_debug_info=include_debug_info, include_npcs=include_npcs)

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
    
    # Player name
    if player_data.get('name'):
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

def _format_state_detailed(state_data, include_debug_info=False, include_npcs=True):
    """
    Internal function to create detailed multi-line state format for LLM prompts.
    """
    context_parts = []
    
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
        
        # Player name and basic info
        if 'name' in player_data and player_data['name']:
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
        game_context = _format_game_state(game_data)
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
    elif 'position' in player_data and player_data['position']:
        return player_data['position']
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
    
    context_parts.append("\n=== LOCATION & MAP INFO ===")
    
    # Get location name from player data
    location_name = None
    if player_data and 'location' in player_data and player_data['location']:
        location = player_data['location']
        if isinstance(location, dict):
            location_name = location.get('map_name', str(location))
        else:
            location_name = str(location)
        context_parts.append(f"Current Location: {location_name}")
    
    # Also add current map if different
    if 'current_map' in map_info and map_info['current_map'] != location_name:
        context_parts.append(f"Current Map: {map_info['current_map']}")
    
    # Get player coordinates
    player_coords = None
    player_coords_dict = map_info.get('player_coords', {})
    if isinstance(player_coords_dict, dict) and player_coords_dict.get('x') is not None:
        player_coords = (player_coords_dict.get('x', 0), player_coords_dict.get('y', 0))
    elif player_coords_dict and not isinstance(player_coords_dict, dict):
        player_coords = player_coords_dict
    elif player_data and 'position' in player_data:
        pos = player_data['position']
        player_coords = (pos.get('x', 0), pos.get('y', 0))
    
    # Get MapStitcher instance
    map_stitcher = _get_map_stitcher_instance()
    
    # Get NPCs if available
    npcs = []
    if include_npcs and 'object_events' in map_info:
        npcs = map_info.get('object_events', [])
    
    # Get connections from current area
    connections = []
    if map_info.get('stitched_map_info'):
        current_area = map_info['stitched_map_info'].get('current_area', {})
        connections = current_area.get('connections', [])
    
    # Use MapStitcher to generate the map display
    if location_name:
        map_lines = map_stitcher.generate_location_map_display(
            location_name=location_name,
            player_pos=player_coords,
            npcs=npcs,
            connections=connections
        )
        
        if map_lines:
            context_parts.extend(map_lines)
        else:
            # Fallback if MapStitcher doesn't have data for this location
            context_parts.append("\n--- LOCAL MAP (No stitched data) ---")
            if 'tiles' in map_info and map_info['tiles']:
                _add_local_map_fallback(context_parts, map_info, include_npcs)
    else:
        # No location name - use local map fallback
        context_parts.append("\n--- LOCAL MAP (Location unknown) ---")
        if 'tiles' in map_info and map_info['tiles']:
            _add_local_map_fallback(context_parts, map_info, include_npcs)
    
    # NPC information removed - unreliable detection with incorrect positions
    
    # Add stitched map information if available
    stitched_info = _format_stitched_map_info(map_info)
    if stitched_info:
        context_parts.extend(stitched_info)
    
    return context_parts

def _add_local_map_fallback(context_parts, map_info, include_npcs):
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
        map_display = format_map_for_llm(raw_tiles, facing, npcs, player_coords)
        context_parts.append(map_display)
        
        # Add dynamic legend based on symbols in the map
        grid = format_map_grid(raw_tiles, facing, npcs, player_coords)
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
    """Get or create the MapStitcher instance"""
    global MAP_STITCHER_INSTANCE
    if MAP_STITCHER_INSTANCE is None:
        from utils.map_stitcher import MapStitcher
        MAP_STITCHER_INSTANCE = MapStitcher()
    return MAP_STITCHER_INSTANCE

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
        print("🗺️ DEBUG: Cleared map stitcher data")

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
    
    # Get MapStitcher instance
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
                    print(f"🗺️ DEBUG: Location transition detected, triggering MapStitcher save...")
                    MAP_STITCHER_SAVE_CALLBACK()
                    print(f"🗺️ DEBUG: MapStitcher save completed")
                except Exception as e:
                    print(f"🗺️ DEBUG: Failed to save via MapStitcher callback: {e}")
            
            print(f"🗺️ DEBUG: Location transition: {CURRENT_LOCATION} → {location_name} at position {player_local_pos}")
        
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
    print(f"🗺️ DEBUG: Checking portal coordinates for location: {location_name}")
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
    print(f"🗺️ DEBUG: About to check MAP_ID_CONNECTIONS...")
    print(f"🗺️ DEBUG: full_state_data is: {type(full_state_data)} with keys: {list(full_state_data.keys()) if full_state_data else 'None'}")
    try:
        # Try multiple ways to find MAP_ID_CONNECTIONS
        map_id_connections = None
        
        # Method 0: Check if passed via HTTP response (NEW - preferred method)
        # First try location_connections (more accurate), then fall back to portal_connections
        if full_state_data and 'location_connections' in full_state_data:
            location_connections = full_state_data['location_connections']
            print(f"🗺️ DEBUG: Found location_connections in HTTP response: {location_connections}")
            
            # Use the location connections display logic
            if location_name in location_connections:
                if not portal_connections_found:
                    lines.append("")
                    lines.append("Known Portal Coordinates:")
                    portal_connections_found = True
                for other_loc, my_coords, their_coords in location_connections[location_name]:
                    lines.append(f"  At ({my_coords[0]}, {my_coords[1]}) → {other_loc} ({their_coords[0]}, {their_coords[1]})")
                    print(f"🗺️ DEBUG: Added location connection: At ({my_coords[0]}, {my_coords[1]}) → {other_loc} ({their_coords[0]}, {their_coords[1]})")
        
        elif full_state_data and 'portal_connections' in full_state_data:
            map_id_connections = full_state_data['portal_connections']
            print(f"🗺️ DEBUG: Found MAP_ID_CONNECTIONS in HTTP response: {map_id_connections}")
            
            # Get current map ID to find relevant portals
            current_map_id = None
            if stitched_data and 'current_area' in stitched_data:
                area_id = stitched_data['current_area'].get('id', '')
                if area_id:
                    try:
                        current_map_id = int(area_id, 16) if isinstance(area_id, str) else area_id
                        print(f"🗺️ DEBUG: Current map ID: {current_map_id}")
                    except ValueError:
                        print(f"🗺️ DEBUG: Could not parse map ID from: {area_id}")
            
            # Try to find current map ID by matching current location name
            if not current_map_id:
                current_location = location_name  # Use the location_name parameter
                print(f"🗺️ DEBUG: Trying to find map ID for location: {current_location}")
                
                # Look through server's map data to find current map ID
                if 'map' in full_state_data:
                    map_info = full_state_data.get('map', {})
                    map_location = map_info.get('location_name', '')
                    if map_location:
                        current_location = map_location
                        print(f"🗺️ DEBUG: Using map location: {current_location}")
                
                # Match location with portal connections - try all map IDs
                for map_id in map_id_connections.keys():
                    # Convert string keys to int if needed
                    try:
                        test_map_id = int(map_id) if isinstance(map_id, str) else map_id
                        print(f"🗺️ DEBUG: Testing map ID {test_map_id} for location '{current_location}'")
                        
                        # For LITTLEROOT TOWN MAYS HOUSE 2F, map_id should be 259
                        if current_location == "LITTLEROOT TOWN MAYS HOUSE 2F" and test_map_id == 259:
                            current_map_id = test_map_id
                            print(f"🗺️ DEBUG: Found matching map ID: {current_map_id}")
                            break
                        elif current_location == "LITTLEROOT TOWN MAYS HOUSE 1F" and test_map_id == 258:
                            current_map_id = test_map_id
                            print(f"🗺️ DEBUG: Found matching map ID: {current_map_id}")
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
                    print(f"🗺️ DEBUG: Added portal: At ({from_pos[0]}, {from_pos[1]}) → {to_name} ({to_pos[0]}, {to_pos[1]})")
                    
        elif stitched_data and 'portal_connections' in stitched_data:
            map_id_connections = stitched_data['portal_connections']
            print(f"🗺️ DEBUG: Found MAP_ID_CONNECTIONS in stitched_data: {map_id_connections}")
        
        # Method 1: Check current module (fallback)
        if not map_id_connections:
            current_module = sys.modules[__name__]
            print(f"🗺️ DEBUG: Checking current module for MAP_ID_CONNECTIONS attribute...")
            print(f"🗺️ DEBUG: hasattr(current_module, 'MAP_ID_CONNECTIONS'): {hasattr(current_module, 'MAP_ID_CONNECTIONS')}")
            if hasattr(current_module, 'MAP_ID_CONNECTIONS') and current_module.MAP_ID_CONNECTIONS:
                map_id_connections = current_module.MAP_ID_CONNECTIONS
                print(f"🗺️ DEBUG: Found MAP_ID_CONNECTIONS in current module: {map_id_connections}")
        
        # Method 2: Check global variable (fallback)
        if not map_id_connections:
            try:
                print(f"🗺️ DEBUG: Checking globals for MAP_ID_CONNECTIONS...")
                print(f"🗺️ DEBUG: 'MAP_ID_CONNECTIONS' in globals(): {'MAP_ID_CONNECTIONS' in globals()}")
                if 'MAP_ID_CONNECTIONS' in globals():
                    print(f"🗺️ DEBUG: globals()['MAP_ID_CONNECTIONS']: {globals()['MAP_ID_CONNECTIONS']}")
                if 'MAP_ID_CONNECTIONS' in globals() and globals()['MAP_ID_CONNECTIONS']:
                    map_id_connections = globals()['MAP_ID_CONNECTIONS']
                    print(f"🗺️ DEBUG: Found MAP_ID_CONNECTIONS in globals: {map_id_connections}")
            except Exception as e:
                print(f"🗺️ DEBUG: Error checking globals: {e}")
        
        # Method 3: Re-import state_formatter and check (fallback)
        if not map_id_connections:
            try:
                print(f"🗺️ DEBUG: Attempting to re-import state_formatter...")
                print(f"🗺️ DEBUG: hasattr(sf, 'MAP_ID_CONNECTIONS'): {hasattr(sf, 'MAP_ID_CONNECTIONS')}")
                if hasattr(sf, 'MAP_ID_CONNECTIONS'):
                    print(f"🗺️ DEBUG: sf.MAP_ID_CONNECTIONS: {sf.MAP_ID_CONNECTIONS}")
                if hasattr(sf, 'MAP_ID_CONNECTIONS') and sf.MAP_ID_CONNECTIONS:
                    map_id_connections = sf.MAP_ID_CONNECTIONS
                    print(f"🗺️ DEBUG: Found MAP_ID_CONNECTIONS in imported module: {map_id_connections}")
            except Exception as e:
                print(f"🗺️ DEBUG: Error re-importing state_formatter: {e}")
        
        if map_id_connections:
            print(f"🗺️ DEBUG: MAP_ID_CONNECTIONS available with {len(map_id_connections)} maps")
            print(f"🗺️ DEBUG: Available map IDs: {list(map_id_connections.keys())}")
            
            # Get current map ID from stitched data
            current_map_id = None
            print(f"🗺️ DEBUG: stitched_data structure: {stitched_data}")
            if stitched_data and stitched_data.get('available'):
                # Try to get map ID from current area
                current_area = stitched_data.get('current_area', {})
                map_id_str = current_area.get('id')
                print(f"🗺️ DEBUG: Current area ID from stitched_data: {map_id_str}")
                print(f"🗺️ DEBUG: Full current_area: {current_area}")
                if map_id_str:
                    try:
                        current_map_id = int(map_id_str, 16)  # Convert from hex string
                        print(f"🗺️ DEBUG: Converted to map ID: {current_map_id}")
                    except ValueError:
                        print(f"🗺️ DEBUG: Failed to convert map ID: {map_id_str}")
                else:
                    print(f"🗺️ DEBUG: No map ID found in current_area: {current_area}")
            else:
                print(f"🗺️ DEBUG: No stitched_data available or not available")
                print(f"🗺️ DEBUG: stitched_data type: {type(stitched_data)}")
                if stitched_data:
                    print(f"🗺️ DEBUG: stitched_data keys: {list(stitched_data.keys()) if isinstance(stitched_data, dict) else 'not a dict'}")
                    print(f"🗺️ DEBUG: stitched_data.get('available'): {stitched_data.get('available') if isinstance(stitched_data, dict) else 'N/A'}")
            
            if current_map_id and current_map_id in map_id_connections:
                print(f"🗺️ DEBUG: Found connections for map ID {current_map_id}")
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
                print(f"🗺️ DEBUG: No connections found for map ID {current_map_id} (available: {list(map_id_connections.keys()) if map_id_connections else 'None'})")
        else:
            print(f"🗺️ DEBUG: MAP_ID_CONNECTIONS not found in any location")
    except Exception as e:
        print(f"🗺️ DEBUG: Error checking MAP_ID_CONNECTIONS: {e}")
    
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

def _format_game_state(game_data):
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
    
    if 'game_state' in game_data:
        context_parts.append(f"Game State: {game_data['game_state']}")
    
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
        return {}
    
    current_x = int(player_position['x'])
    current_y = int(player_position['y'])
    
    # Get map and tile data
    map_info = state_data.get('map', {})
    raw_tiles = map_info.get('tiles', [])
    
    if not raw_tiles:
        return {}
    
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
            'tile_description': 'BLOCKED - Out of bounds'
        }
        
        # Check if the target position is within the grid bounds
        if (0 <= grid_y < len(raw_tiles) and 
            0 <= grid_x < len(raw_tiles[grid_y]) and
            raw_tiles[grid_y]):
            
            try:
                # Get the tile at the target position
                target_tile = raw_tiles[grid_y][grid_x]
                
                # Get tile symbol and check if walkable
                tile_symbol = format_tile_to_symbol(target_tile)
                
                # Determine if movement is blocked
                is_blocked = tile_symbol in ['#', 'W']  # Walls and water block movement
                
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
                    if tile_symbol == '.':
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
                
                preview_info.update({
                    'blocked': is_blocked,
                    'tile_symbol': tile_symbol,
                    'tile_description': tile_description
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
            
            lines.append(f"  {direction:5}: ({new_x:3},{new_y:3}) [{symbol}] {status}")
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
            else:
                # Add brief description for walkable tiles
                if 'Tall grass' in desc:
                    lines[-1] += " - Tall grass (wild encounters)"
                elif 'Door' in desc:
                    lines[-1] += " - Door/Entrance"
                elif 'Stairs' in desc or 'Warp' in desc:
                    lines[-1] += " - Stairs/Warp"
                elif 'Jump ledge' in desc and 'correct direction' in desc:
                    lines[-1] += " - Jump ledge (can jump this way)"
    
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
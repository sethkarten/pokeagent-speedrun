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

logger = logging.getLogger(__name__)

# Global persistent location maps storage - separate map for each location
PERSISTENT_LOCATION_GRIDS = {}  # {location_name: {(x,y): symbol}}
PERSISTENT_MAP_FILE = "/tmp/pokemon_location_maps.json"
CURRENT_LOCATION = None
LAST_LOCATION = None
LAST_TRANSITION = None  # Stores transition coordinates
LAST_PLAYER_POSITION = {}  # {location_name: (x, y)} - tracks last position in each location
LOCATION_CONNECTIONS = {}  # {location_name: [(other_location, my_coords, their_coords)]} - bidirectional connections

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

def _analyze_npc_terrain(npc, raw_tiles, player_coords):
    """
    Analyze what terrain is underneath an NPC position.
    
    Args:
        npc: NPC object with current_x, current_y coordinates
        raw_tiles: 2D array of tile data
        player_coords: Player coordinates for grid positioning
        
    Returns:
        str: Description of terrain under NPC, or None if no notable terrain
    """
    if not raw_tiles or not player_coords:
        return None
    
    try:
        # Handle both tuple and dict formats for player_coords
        if isinstance(player_coords, dict):
            player_abs_x = player_coords.get('x', 0)
            player_abs_y = player_coords.get('y', 0)
        else:
            player_abs_x, player_abs_y = player_coords
        
        # Ensure coordinates are integers
        player_abs_x = int(player_abs_x) if player_abs_x is not None else 0
        player_abs_y = int(player_abs_y) if player_abs_y is not None else 0
        
        # Get NPC absolute coordinates
        npc_abs_x = int(npc.get('current_x', 0))
        npc_abs_y = int(npc.get('current_y', 0))
        
        # Calculate offset from player position
        center_y = len(raw_tiles) // 2
        center_x = len(raw_tiles[0]) // 2
        
        # Calculate grid position
        offset_x = npc_abs_x - player_abs_x
        offset_y = npc_abs_y - player_abs_y
        grid_x = center_x + offset_x
        grid_y = center_y + offset_y
        
        # Check if NPC is within grid bounds
        if 0 <= grid_y < len(raw_tiles) and 0 <= grid_x < len(raw_tiles[grid_y]):
            tile = raw_tiles[grid_y][grid_x]
            symbol = format_tile_to_symbol(tile)
            
            # Check for important terrain types
            if symbol == "D":
                return "BLOCKING DOOR"
            elif symbol == "S":
                return "blocking stairs/warp"
            elif symbol == "#":
                return "on wall/blocked tile"
            elif symbol in ["P", "T", "B", "C", "=", "t"]:
                return f"on furniture ({symbol})"
            elif symbol == "~":
                return "in tall grass"
            elif symbol == "W":
                return "on water"
            
    except (ValueError, IndexError, TypeError) as e:
        logger.debug(f"Failed to analyze NPC terrain: {e}")
    
    return None

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
        map_context = _format_map_info(state_data.get('map', {}), player_data, include_debug_info, include_npcs)
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

def _format_map_info(map_info, player_data=None, include_debug_info=False, include_npcs=True):
    """Format map and traversability information using unified formatter."""
    context_parts = []
    
    if not map_info:
        return context_parts
    
    context_parts.append("\n=== LOCATION & MAP INFO ===")
    
    # Add current location from player data
    if player_data and 'location' in player_data and player_data['location']:
        location = player_data['location']
        if isinstance(location, dict):
            # If location is a dict, try to get map_name or use str representation
            location_name = location.get('map_name', str(location))
        else:
            # If location is a string, use it directly
            location_name = str(location)
        context_parts.append(f"Current Location: {location_name}")
    
    if 'current_map' in map_info:
        context_parts.append(f"Current Map: {map_info['current_map']}")
    
    # For now, use the local map approach but prepare for stitching integration
    # We need to modify this to use actual current local map data and stitch it
    
    # Try to build stitched map from current local map data
    if 'tiles' in map_info and map_info['tiles']:
        # Get current local map (this is the 11x11 around player)
        current_local_tiles = map_info['tiles']
        
        # Extract player coordinates properly - try map first, then player data as fallback
        player_coords_dict = map_info.get('player_coords', {})
        if isinstance(player_coords_dict, dict) and player_coords_dict.get('x') is not None:
            player_coords = (player_coords_dict.get('x', 0), player_coords_dict.get('y', 0))
        elif player_coords_dict and not isinstance(player_coords_dict, dict):
            # Fallback if it's already a tuple
            player_coords = player_coords_dict
        else:
            # Use player position from player data as fallback
            if player_data and 'position' in player_data:
                pos = player_data['position']
                player_coords = (pos.get('x', 0), pos.get('y', 0))
                print(f"🗺️ DEBUG: Using player.position coordinates: {player_coords}")
            else:
                player_coords = (0, 0)
                print(f"🗺️ DEBUG: No coordinates found, defaulting to (0, 0)")
        
        # Get location name from player data or stitched map info
        location_name = None
        if player_data and 'location' in player_data and player_data['location']:
            location = player_data['location']
            if isinstance(location, dict):
                location_name = location.get('map_name', str(location))
            else:
                location_name = str(location)
        
        # Build current area info with location name
        current_area = map_info.get('stitched_map_info', {}).get('current_area', {}) if map_info.get('stitched_map_info') else {}
        if location_name and not current_area.get('name'):
            current_area['name'] = location_name
        
        # Build stitched data structure from current local map
        stitched_data = {
            'current_local_map': current_local_tiles,
            'player_local_pos': player_coords,  # Player's position in local map coordinates
            'terrain_areas': map_info.get('stitched_map_info', {}).get('terrain_areas', []) if map_info.get('stitched_map_info') else [],
            'current_area': current_area,
            'location_name_fallback': location_name  # Direct fallback for location name
        }
        
        # Try the new stitching approach
        world_map_display = _format_world_map_display(stitched_data)
        if world_map_display:
            context_parts.extend(world_map_display)
        else:
            # Fall back to local map
            context_parts.append("\n--- LOCAL MAP (Stitching failed) ---")
            _add_local_map_fallback(context_parts, map_info, include_npcs)
    else:
        # No local map data - show whatever we have
        context_parts.append("\n--- LOCAL MAP (No local data) ---")
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
        
        # Use unified LLM formatter for consistency (no NPCs)
        map_display = format_map_for_llm(raw_tiles, facing, [], player_coords)
        context_parts.append(map_display)
        
        # Add dynamic legend based on symbols in the map
        grid = format_map_grid(raw_tiles, facing, [], player_coords)
        legend = generate_dynamic_legend(grid)
        context_parts.append(f"\n{legend}")

def _format_world_map_display(stitched_data):
    """Format location-specific map display"""
    try:
        # Build separate map for each location
        return _build_stitched_world_map(stitched_data)
        
    except Exception as e:
        logger.warning(f"World map generation failed: {e}")
        return []

def _load_persistent_world_map():
    """Load persistent location maps from file"""
    global PERSISTENT_LOCATION_GRIDS
    try:
        import os
        if os.path.exists(PERSISTENT_MAP_FILE):
            with open(PERSISTENT_MAP_FILE, 'r') as f:
                data = json.load(f)
                # Convert string keys back to tuples for coordinates
                PERSISTENT_LOCATION_GRIDS = {}
                for location, tiles in data.items():
                    PERSISTENT_LOCATION_GRIDS[location] = {eval(k): v for k, v in tiles.items()}
                print(f"🗺️ DEBUG: Loaded {len(PERSISTENT_LOCATION_GRIDS)} location maps from persistent storage")
        else:
            PERSISTENT_LOCATION_GRIDS = {}
            print("🗺️ DEBUG: No persistent map file found, starting fresh")
    except Exception as e:
        print(f"🗺️ DEBUG: Failed to load persistent map: {e}")
        PERSISTENT_LOCATION_GRIDS = {}

def _save_persistent_world_map():
    """Save persistent location maps to file"""
    try:
        # Convert tuple keys to strings for JSON serialization
        data = {}
        for location, tiles in PERSISTENT_LOCATION_GRIDS.items():
            data[location] = {str(k): v for k, v in tiles.items()}
        with open(PERSISTENT_MAP_FILE, 'w') as f:
            json.dump(data, f)
        print(f"🗺️ DEBUG: Saved {len(PERSISTENT_LOCATION_GRIDS)} location maps to persistent storage")
    except Exception as e:
        print(f"🗺️ DEBUG: Failed to save persistent map: {e}")

def save_persistent_world_map(file_path=None):
    """Save persistent location maps and connections to specified file (public function for checkpoint system)"""
    if file_path is None:
        file_path = PERSISTENT_MAP_FILE
    
    try:
        # Convert tuple keys to strings for JSON serialization
        map_data = {}
        for location, tiles in PERSISTENT_LOCATION_GRIDS.items():
            map_data[location] = {str(k): v for k, v in tiles.items()}
        
        # Save all persistent data including connections
        data = {
            'location_grids': map_data,
            'location_connections': LOCATION_CONNECTIONS,
            'current_location': CURRENT_LOCATION,
            'last_location': LAST_LOCATION,
            'last_player_position': LAST_PLAYER_POSITION
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f"🗺️ DEBUG: Saved {len(PERSISTENT_LOCATION_GRIDS)} location maps and connections to {file_path}")
    except Exception as e:
        print(f"🗺️ DEBUG: Failed to save persistent map to {file_path}: {e}")

def load_persistent_world_map(file_path=None):
    """Load persistent location maps and connections from specified file (public function for checkpoint system)"""
    global PERSISTENT_LOCATION_GRIDS, LOCATION_CONNECTIONS, CURRENT_LOCATION, LAST_LOCATION, LAST_PLAYER_POSITION
    if file_path is None:
        file_path = PERSISTENT_MAP_FILE
    
    try:
        import os
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both old format (just grids) and new format (with connections)
            if 'location_grids' in data:
                # New format with connections
                map_data = data['location_grids']
                LOCATION_CONNECTIONS = data.get('location_connections', {})
                CURRENT_LOCATION = data.get('current_location')
                LAST_LOCATION = data.get('last_location')
                LAST_PLAYER_POSITION = data.get('last_player_position', {})
            else:
                # Old format - just grids
                map_data = data
                LOCATION_CONNECTIONS = {}
                CURRENT_LOCATION = None
                LAST_LOCATION = None
                LAST_PLAYER_POSITION = {}
            
            # Convert string keys back to tuples for coordinates
            PERSISTENT_LOCATION_GRIDS = {}
            for location, tiles in map_data.items():
                PERSISTENT_LOCATION_GRIDS[location] = {eval(k): v for k, v in tiles.items()}
            
            print(f"🗺️ DEBUG: Loaded {len(PERSISTENT_LOCATION_GRIDS)} location maps and {len(LOCATION_CONNECTIONS)} connections from {file_path}")
        else:
            PERSISTENT_LOCATION_GRIDS = {}
            LOCATION_CONNECTIONS = {}
            CURRENT_LOCATION = None
            LAST_LOCATION = None
            LAST_PLAYER_POSITION = {}
            print(f"🗺️ DEBUG: No map file found at {file_path}, starting fresh")
    except Exception as e:
        print(f"🗺️ DEBUG: Failed to load persistent map from {file_path}: {e}")
        PERSISTENT_LOCATION_GRIDS = {}
        LOCATION_CONNECTIONS = {}
        CURRENT_LOCATION = None
        LAST_LOCATION = None
        LAST_PLAYER_POSITION = {}

def clear_persistent_world_map():
    """Clear the persistent location maps for testing"""
    global PERSISTENT_LOCATION_GRIDS, CURRENT_LOCATION, LAST_LOCATION, LAST_TRANSITION, LAST_PLAYER_POSITION
    PERSISTENT_LOCATION_GRIDS.clear()
    LAST_PLAYER_POSITION.clear()
    CURRENT_LOCATION = None
    LAST_LOCATION = None
    LAST_TRANSITION = None
    try:
        import os
        if os.path.exists(PERSISTENT_MAP_FILE):
            os.remove(PERSISTENT_MAP_FILE)
        print("🗺️ DEBUG: Cleared persistent location maps and file")
    except Exception as e:
        print(f"🗺️ DEBUG: Failed to clear persistent map file: {e}")

def _build_stitched_world_map(stitched_data):
    """Build separate persistent maps for each location"""
    
    # Get the current local map data (should be 11x11 around player)
    current_map = stitched_data.get('current_local_map')  # This should be the 11x11 map
    current_area = stitched_data.get('current_area', {})
    location_name = current_area.get('name')
    
    # Use fallback location name if current_area has 'Unknown' or empty name
    if not location_name or location_name == 'Unknown':
        fallback_name = stitched_data.get('location_name_fallback')
        if fallback_name and fallback_name != 'Unknown':
            location_name = fallback_name
            print(f"🗺️ DEBUG: Using fallback location name: {location_name}")
        else:
            location_name = 'Unknown'
    
    # Debug if we're still getting Unknown location
    if location_name == 'Unknown':
        print(f"⚠️ WARNING: Location name is 'Unknown', current_area data: {current_area}")
        print(f"⚠️ WARNING: Fallback location name: {stitched_data.get('location_name_fallback')}")
    
    # Track location changes and transition coordinates
    global CURRENT_LOCATION, LAST_LOCATION, PERSISTENT_LOCATION_GRIDS, LAST_TRANSITION, LAST_PLAYER_POSITION, LOCATION_CONNECTIONS
    LAST_TRANSITION = None  # Will store transition info
    
    # Load persistent maps if empty
    if not PERSISTENT_LOCATION_GRIDS:
        _load_persistent_world_map()
    
    # Get player position for transition tracking
    player_local_pos = stitched_data.get('player_local_pos', (0, 0))
    
    # Update current location's player position
    if location_name:
        LAST_PLAYER_POSITION[location_name] = player_local_pos
    
    # Check if we've changed locations
    if CURRENT_LOCATION != location_name:
        # Store transition information
        if CURRENT_LOCATION is not None and CURRENT_LOCATION != location_name:
            # Get the exit coordinates from the previous location
            from_coords = LAST_PLAYER_POSITION.get(CURRENT_LOCATION, (0, 0))
            
            LAST_TRANSITION = {
                'from_location': CURRENT_LOCATION,
                'from_coords': from_coords,
                'to_location': location_name,
                'to_coords': player_local_pos
            }
            
            # Record bidirectional connection
            if CURRENT_LOCATION not in LOCATION_CONNECTIONS:
                LOCATION_CONNECTIONS[CURRENT_LOCATION] = []
            if location_name not in LOCATION_CONNECTIONS:
                LOCATION_CONNECTIONS[location_name] = []
            
            # Add connection from current to new location
            connection_exists = False
            for other_loc, my_coords, their_coords in LOCATION_CONNECTIONS[CURRENT_LOCATION]:
                if other_loc == location_name:
                    connection_exists = True
                    break
            
            if not connection_exists:
                LOCATION_CONNECTIONS[CURRENT_LOCATION].append((location_name, from_coords, player_local_pos))
                LOCATION_CONNECTIONS[location_name].append((CURRENT_LOCATION, player_local_pos, from_coords))
                print(f"🗺️ DEBUG: Recorded connection: {CURRENT_LOCATION} ({from_coords}) ↔ {location_name} ({player_local_pos})")
            
            print(f"🗺️ DEBUG: Location transition: {CURRENT_LOCATION} ({from_coords}) → {location_name} ({player_local_pos})")
        
        LAST_LOCATION = CURRENT_LOCATION
        CURRENT_LOCATION = location_name
    
    # Initialize this location's grid if it doesn't exist
    if location_name not in PERSISTENT_LOCATION_GRIDS:
        PERSISTENT_LOCATION_GRIDS[location_name] = {}
        print(f"🗺️ DEBUG: Created new map for location: {location_name}")
    
    current_location_grid = PERSISTENT_LOCATION_GRIDS[location_name]
    
    # Debug: show current location's grid size
    print(f"🗺️ DEBUG: Location '{location_name}' currently has {len(current_location_grid)} tiles")
    
    # Get player position - use local coordinates for this location
    player_local_pos = stitched_data.get('player_local_pos')
    if not player_local_pos:
        # Fallback to extracting from current area
        player_local_pos = current_area.get('player_pos', (5, 5))  # Default to center
    
    # If we have current local map data, add it to this location's grid
    if current_map:
        player_x, player_y = player_local_pos
        
        # Current map from memory reader is typically 11x11 centered on player
        map_size = len(current_map)
        center = map_size // 2  # For 11x11: center at 5
        
        # For 11x11 input, process 10x10 (skip outer border)
        # For 11x11 with center at 5: process indices 0-9 (center-5 to center+4)
        
        print(f"🗺️ DEBUG: Full observation is {map_size}x{map_size}, processing 10x10 (skipping outer border), player at local ({player_x}, {player_y})")
        
        tiles_added = 0
        # Process 10x10 area: skip the outer border of the 11x11
        if map_size == 11:
            start_y = 0  # Skip top border
            end_y = 10   # Skip bottom border  
            start_x = 0  # Skip left border
            end_x = 10   # Skip right border
        else:
            # For other sizes, use radius approach
            radius = 4  # For 10x10 from center
            start_y = max(0, center - radius)
            end_y = min(map_size, center + radius + 1)
            start_x = max(0, center - radius)  
            end_x = min(map_size, center + radius + 1)
        
        for local_y in range(start_y, end_y):
            row = current_map[local_y]
            for local_x in range(start_x, end_x):
                if local_x < len(row):
                    tile_data = row[local_x]
                    
                    # Calculate position in this location's coordinate system
                    offset_x = local_x - center
                    offset_y = local_y - center
                    location_x = player_x + offset_x
                    location_y = player_y + offset_y
                    
                    # Convert tile to symbol
                    symbol = format_tile_to_symbol(tile_data)
                    
                    # Only add tiles with real data
                    if symbol and symbol != '?' and symbol != ' ':
                        if (location_x, location_y) not in current_location_grid:
                            tiles_added += 1
                        current_location_grid[(location_x, location_y)] = symbol
        
        print(f"🗺️ DEBUG: Added {tiles_added} new tiles to {location_name}, total now {len(current_location_grid)}")
        
        # Save to file after adding new tiles
        if tiles_added > 0:
            _save_persistent_world_map()
    
    if not current_location_grid:
        return []
    
    # Use player local position for display
    player_display_pos = player_local_pos
    if not player_display_pos:
        return []
    
    # Find the bounds of the full explored area in this location
    all_positions = list(current_location_grid.keys())
    if player_display_pos:
        all_positions.append(player_display_pos)
    
    if not all_positions:
        return []
    
    min_x = min(pos[0] for pos in all_positions)
    max_x = max(pos[0] for pos in all_positions)
    min_y = min(pos[1] for pos in all_positions)
    max_y = max(pos[1] for pos in all_positions)
    
    print(f"🗺️ DEBUG: {location_name} - showing full explored area: x={min_x} to {max_x}, y={min_y} to {max_y}")
    
    # Get connection info from current_area for portal markers
    connections = current_area.get('connections', [])
    portal_positions = {}  # Track where portals lead
    
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
    
    lines.append(f"\n--- MAP: {location_name.upper()} ---")
    
    # Create the map display showing full explored area
    for y in range(min_y, max_y + 1):
        row = ""
        for x in range(min_x, max_x + 1):
            # Check if this is an edge position for potential portals
            is_edge = (x == min_x or x == max_x or y == min_y or y == max_y)
            
            if (x, y) == player_display_pos:
                row += "P"
            elif (x, y) in current_location_grid:
                tile = current_location_grid[(x, y)]
                # Check if this is an edge tile that could be a portal
                if is_edge and tile == '.':
                    # Mark portals based on position and connections
                    for conn in connections:
                        direction = conn.get('direction', '').lower()
                        if direction == 'east' and x == max_x:
                            row += "→"
                            portal_positions[(x, y)] = conn.get('name', 'Unknown')
                            break
                        elif direction == 'west' and x == min_x:
                            row += "←"
                            portal_positions[(x, y)] = conn.get('name', 'Unknown')
                            break
                        elif direction == 'north' and y == min_y:
                            row += "↑"
                            portal_positions[(x, y)] = conn.get('name', 'Unknown')
                            break
                        elif direction == 'south' and y == max_y:
                            row += "↓"
                            portal_positions[(x, y)] = conn.get('name', 'Unknown')
                            break
                    else:
                        row += tile
                else:
                    row += tile
            else:
                row += "?"  # Unexplored area in this location
        
        # Add spacing for readability
        spaced_row = " ".join(row)
        lines.append(spaced_row)
    
    # Add legend
    legend_lines = ["", "Legend:"]
    legend_lines.append("  Movement: P=Player")
    
    # Check what terrain symbols we have visible in the full explored area
    visible_symbols = set(current_location_grid.values())
    
    terrain_items = []
    for symbol in [".", "#", "~", "W", "D", "S"]:
        if symbol in visible_symbols:
            if symbol == ".":
                terrain_items.append(".=Walkable path")
            elif symbol == "#":
                terrain_items.append("#=Wall/Blocked")
            elif symbol == "~":
                terrain_items.append("~=Tall grass")
            elif symbol == "W":
                terrain_items.append("W=Water")
            elif symbol == "D":
                terrain_items.append("D=Door")
            elif symbol == "S":
                terrain_items.append("S=Stairs/Warp")
    
    if terrain_items:
        legend_lines.append(f"  Terrain: {', '.join(terrain_items)}")
    
    # Add portal/exit markers to legend if any are visible
    portal_items = []
    if portal_positions:
        unique_portals = {}
        for pos, dest in portal_positions.items():
            x, y = pos
            # Determine direction symbol
            if x == min_x:
                unique_portals["←"] = dest
            elif x == max_x:
                unique_portals["→"] = dest
            elif y == min_y:
                unique_portals["↑"] = dest
            elif y == max_y:
                unique_portals["↓"] = dest
        
        for symbol, dest in unique_portals.items():
            portal_items.append(f"{symbol}=To {dest}")
    
    if portal_items:
        legend_lines.append(f"  Portals: {', '.join(portal_items)}")
    
    lines.extend(legend_lines)
    
    # Add exploration statistics for this location
    total_tiles = len(current_location_grid)
    lines.append("")
    lines.append(f"Total explored in {location_name}: {total_tiles} tiles")
    
    # Add discovered connection points from our transition tracking
    if location_name in LOCATION_CONNECTIONS:
        lines.append("")
        lines.append("Known Portal Coordinates:")
        for other_loc, my_coords, their_coords in LOCATION_CONNECTIONS[location_name]:
            lines.append(f"  At ({my_coords[0]}, {my_coords[1]}) → {other_loc} ({their_coords[0]}, {their_coords[1]})")
    
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
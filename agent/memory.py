import logging
from collections import deque
from agent.system_prompt import system_prompt
from utils.state_formatter import format_state_summary, get_party_health_summary
from utils.vlm import VLM

# Set up module logging
logger = logging.getLogger(__name__)

def extract_key_state_info(state_data):
    """Extract key information from comprehensive state for memory storage using the utility functions"""
    # Use the state formatter utilities for consistency
    state_summary = format_state_summary(state_data)
    party_health = get_party_health_summary(state_data)
    
    # Extract additional info
    player_data = state_data.get('player', {})
    game_data = state_data.get('game', {})
    map_info = state_data.get('map', {})
    
    key_info = {
        'state_summary': state_summary,
        'player_name': player_data.get('name', 'Player'),
        'money': player_data.get('money') or game_data.get('money', 0),
        'current_map': player_data.get('location', 'Unknown Location'),
        'in_battle': game_data.get('in_battle', False),
        'party_health': f"{party_health['healthy_count']}/{party_health['total_count']}",
        'critical_pokemon': party_health['critical_pokemon']
    }
    
    # Position info
    if 'coordinates' in player_data:
        key_info['position'] = player_data['coordinates']
    elif 'position' in player_data:
        key_info['position'] = player_data['position']
    else:
        key_info['position'] = {}
    
    # Battle opponent
    if game_data.get('battle_info'):
        battle = game_data['battle_info']
        opponent_pokemon = battle.get('opponent_pokemon', {})
        key_info['battle_opponent'] = opponent_pokemon.get('species_name', opponent_pokemon.get('species', 'Unknown Pokemon'))
    
    # Traversability summary
    if 'traversability' in map_info and map_info['traversability']:
        traversability = map_info['traversability']
        total_tiles = sum(len(row) for row in traversability)
        blocked_count = sum(1 for row in traversability for cell in row if str(cell) in ['0', '0'])
        passable_tiles = total_tiles - blocked_count
        key_info['traversability_summary'] = f"{passable_tiles}/{total_tiles} passable"
    else:
        key_info['traversability_summary'] = "No data"
    
    return key_info

def memory_step(memory_context, current_plan, recent_actions, observation_buffer, vlm):
    """
    Maintain a rolling buffer of the previous 50 actions and observations with state information.
    Returns updated memory_context with the most recent 50 entries and key insights.
    """
    # Initialize memory buffer if it doesn't exist
    if not hasattr(memory_step, 'memory_buffer'):
        memory_step.memory_buffer = deque(maxlen=50)
    
    logger.info(f"[MEMORY] Processing {len(observation_buffer)} new observations")
    
    # Add new observations with state info to the buffer
    for obs in observation_buffer:
        state_info = extract_key_state_info(obs.get('state', {}))
        memory_step.memory_buffer.append({
            "type": "observation",
            "frame_id": obs["frame_id"],
            "content": obs["observation"],
            "state": state_info
        })
        logger.info(f"[MEMORY] Added observation frame {obs['frame_id']}: {state_info['state_summary']}")
    
    # Add recent actions to the buffer
    for action in recent_actions:
        memory_step.memory_buffer.append({
            "type": "action",
            "content": action
        })
    
    # Create a formatted memory context from the buffer with state insights
    memory_entries = []
    key_events = []
    
    # Track significant state changes
    previous_map = None
    previous_battle_state = None
    
    for i, entry in enumerate(memory_step.memory_buffer):
        if entry["type"] == "observation":
            frame_id = entry['frame_id']
            description = entry['content']
            state = entry.get('state', {})
            
            # Use the consistent state summary
            state_summary = state.get('state_summary', '')
            
            # Check for significant events
            current_map = state.get('current_map', 'Unknown Location')
            current_battle = state.get('in_battle', False)
            
            if current_map != previous_map and previous_map is not None:
                key_events.append(f"Moved from {previous_map} to {current_map}")
                logger.info(f"[MEMORY] Key event: Map change from {previous_map} to {current_map}")
            
            if current_battle != previous_battle_state:
                if current_battle:
                    opponent = state.get('battle_opponent', 'Unknown Pokemon')
                    key_events.append(f"Entered battle vs {opponent}")
                    logger.info(f"[MEMORY] Key event: Entered battle vs {opponent}")
                else:
                    key_events.append("Exited battle")
                    logger.info("[MEMORY] Key event: Exited battle")
            
            previous_map = current_map
            previous_battle_state = current_battle
            
            # Format observation entry
            if isinstance(description, dict):
                desc_text = description.get('description', str(description))
            else:
                desc_text = str(description)
            
            memory_entries.append(f"Frame {frame_id}: {desc_text} [{state_summary}]")
        else:
            memory_entries.append(f"Action: {entry['content']}")
    
    # Get current state summary from the latest observation
    current_state_summary = ""
    if observation_buffer:
        latest_state = extract_key_state_info(observation_buffer[-1].get('state', {}))
        current_state_summary = latest_state.get('state_summary', 'No state data')
    
    # Combine into comprehensive memory context
    memory_context = f"""★★★ COMPREHENSIVE MEMORY CONTEXT ★★★

CURRENT STATE: {current_state_summary}

CURRENT PLAN: {current_plan if current_plan else 'No plan yet'}

KEY EVENTS: {' -> '.join(key_events[-5:]) if key_events else 'None recently'}

RECENT MEMORY (last 50 entries):
{chr(10).join(memory_entries[-30:])}"""  # Show last 30 entries to avoid too much text
    
    logger.info(f"[MEMORY] Memory context updated with {len(memory_entries)} total entries")
    logger.info(f"[MEMORY] Current state: {current_state_summary}")
    logger.info(f"[MEMORY] Key events: {len(key_events)} tracked")
    
    return memory_context 
#!/usr/bin/env python3
"""
MCP Server for Pokemon Emerald Game
Exposes game state and actions as MCP tools for use with gemini-cli or other MCP clients.

This server provides:
- get_game_state: Retrieve current game state (manual request)
- press_buttons: Execute button presses on the emulator
- navigate_to: Pathfind to coordinates automatically
- add_knowledge: Store discoveries in knowledge base
- search_knowledge: Query knowledge base
- get_knowledge_summary: View important discoveries
- lookup_pokemon_info: Fetch info from Pokemon wikis (Bulbapedia, Serebii, etc.)
- list_wiki_sources: List available wiki sources
- get_walkthrough: Get official Emerald walkthrough (Parts 1-21)
"""

import sys
import logging
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
from urllib.parse import quote_plus

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
import requests
from bs4 import BeautifulSoup

from utils.pathfinding import Pathfinder
from utils.knowledge_base import get_knowledge_base
from utils.state_formatter import format_state_for_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server configuration
SERVER_URL = "http://localhost:8000"

# Initialize MCP server
mcp = FastMCP(
    name="pokemon-emerald"
)

# Initialize game tools
pathfinder = Pathfinder()
knowledge_base = get_knowledge_base()


# ============================================================================
# HELPER FUNCTIONS FOR SERVER ENDPOINTS (NO HTTP CALLS)
# ============================================================================

def get_game_state_direct(env, state_formatter, action_history=None) -> dict:
    """
    Get game state without HTTP calls - for use by server endpoints.

    Args:
        env: EmeraldEmulator instance
        state_formatter: format_state_for_llm function
        action_history: Optional list of recent actions with start/end positions

    Returns:
        Dictionary with success status and state data including screenshot
    """
    try:
        import base64
        import io
        from PIL import Image

        state = env.get_comprehensive_state()
        # Pass action history to state formatter
        state_text = state_formatter(state, action_history=action_history)

        # Get screenshot as base64 for vision models
        screenshot_b64 = None
        if hasattr(env, 'get_screenshot'):
            screenshot = env.get_screenshot()
            if screenshot is not None:
                # Convert numpy array to PIL Image if needed
                if hasattr(screenshot, 'shape'):  # numpy array
                    img = Image.fromarray(screenshot)
                else:
                    img = screenshot

                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                screenshot_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "success": True,
            "state_text": state_text,
            "screenshot_base64": screenshot_b64,
            "player_position": state.get('player', {}).get('position', {}),
            "location": state.get('player', {}).get('location', 'Unknown')
        }
    except Exception as e:
        logger.error(f"Failed to get game state: {e}")
        return {"success": False, "error": str(e)}


def press_buttons_direct(buttons, action_queue, reasoning="", source=None, metadata=None) -> dict:
    """
    Queue buttons without HTTP calls - for use by server endpoints.

    Args:
        buttons: List of button strings
        action_queue: Global action queue to extend
        reasoning: Reason for pressing buttons

    Returns:
        Dictionary with success status
    """
    if not buttons:
        return {"success": False, "error": "No buttons provided"}

    try:
        # Validate and normalize buttons
        VALID_BUTTONS = {'A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R'}
        normalized_buttons = []
        
        for button in buttons:
            # Normalize to uppercase
            button_upper = str(button).upper().strip()
            
            # Check if valid
            if button_upper in VALID_BUTTONS:
                normalized_buttons.append(button_upper)
            else:
                # Invalid button - fallback to A and warn
                logger.warning(f"Invalid button '{button}' requested, falling back to 'A'")
                normalized_buttons.append('A')
        
        # Add to action queue
        action_queue.extend(normalized_buttons)
        logger.info(f"🎮 Queued buttons: {normalized_buttons} - {reasoning}")

        return {
            "success": True,
            "buttons_queued": normalized_buttons,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.error(f"Failed to queue buttons: {e}")
        return {"success": False, "error": str(e)}


def navigate_to_direct(env, x, y, reason: str = "", variance: Optional[str] = None) -> dict:
    """
    Calculate path to coordinates without HTTP calls - for use by server endpoints.
    Returns buttons to be queued via take_action.

    Args:
        env: EmeraldEmulator instance
        x: Target X coordinate (will be converted to int)
        y: Target Y coordinate (will be converted to int)
        reason: Reason for navigation

    Returns:
        Dictionary with success status, buttons, and path info
    """
    try:
        # Ensure x and y are integers
        x = int(x)
        y = int(y)

        variance_level = None
        nav_reason = "" if reason is None else str(reason)

        if variance is not None:
            variance_str = str(variance).strip()
            variance_lower = variance_str.lower()
            if variance_lower in {"low", "medium", "high"}:
                variance_level = variance_lower
            elif variance_lower in {"", "none"}:
                variance_level = None
            elif not reason:
                nav_reason = variance_str
        elif nav_reason.lower() in {"low", "medium", "high"}:
            variance_level = nav_reason.lower()
            nav_reason = ""

        # Get current state for pathfinding
        state = env.get_comprehensive_state()

        # CRITICAL: Load porymap data into state for pathfinding
        # The pathfinder needs map['porymap']['grid'] which is normally added by format_state_for_llm
        location_name = state.get('player', {}).get('location', 'Unknown')
        if location_name and location_name != 'Unknown' and location_name != 'TITLE_SEQUENCE':
            try:
                from utils.porymap_json_builder import build_json_map_for_llm
                from utils.pokeemerald_parser import PokeemeraldMapLoader
                from pathlib import Path
                import os
                
                # Get pokeemerald root (use same logic as state_formatter)
                pokeemerald_root = None
                # Try environment variable first (same as state_formatter)
                root = os.environ.get('POKEEMERALD_ROOT')
                if root:
                    root_path = Path(root).resolve()
                    if (root_path / "data" / "maps").exists():
                        pokeemerald_root = root_path
                
                # Try porymap_data directory (same as state_formatter)
                if not pokeemerald_root:
                    # Get the server/cli directory's parent's parent (pokeagent-speedrun root)
                    current_dir = Path(__file__).parent.parent.parent
                    porymap_path = current_dir / "porymap_data"
                    if (porymap_path / "data" / "maps").exists():
                        pokeemerald_root = porymap_path.resolve()
                
                # Try common relative paths (same as state_formatter)
                if not pokeemerald_root:
                    current_dir = Path(__file__).parent.parent.parent
                    possible_paths = [
                        current_dir / "pokeemerald",
                        current_dir / "../pokeemerald",
                        current_dir / "../../pokeemerald",
                    ]
                    for path in possible_paths:
                        resolved = path.resolve()
                        if (resolved / "data" / "maps").exists():
                            pokeemerald_root = resolved
                            break
                
                if pokeemerald_root:
                    # Import comprehensive ROM to Porymap mapping from state_formatter
                    from utils.state_formatter import ROM_TO_PORYMAP_MAP
                    
                    porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)
                    
                    # Try fuzzy matching if not in direct mapping (use fuzzywuzzy like state_formatter)
                    if not porymap_map_name:
                        try:
                            from fuzzywuzzy import process
                            map_loader = PokeemeraldMapLoader(pokeemerald_root)
                            maps_dir = pokeemerald_root / "data" / "maps"
                            if maps_dir.exists():
                                available_maps = [d.name for d in maps_dir.iterdir() if d.is_dir() and d.name != "map_groups.json"]
                                best_match, best_match_score = process.extractOne(location_name, available_maps)
                                # Only accept matches with score >= 70 to avoid bad matches
                                if best_match_score >= 70:
                                    porymap_map_name = best_match
                                    logger.debug(f"Fuzzy matched '{location_name}' to '{porymap_map_name}' (score: {best_match_score})")
                        except ImportError:
                            # Fallback: use simple substring matching but prefer exact substring matches
                            # and prefer longer matches (more specific)
                            map_loader = PokeemeraldMapLoader(pokeemerald_root)
                            maps_dir = pokeemerald_root / "data" / "maps"
                            if maps_dir.exists():
                                location_normalized = location_name.lower().replace(" ", "").replace("_", "")
                                best_candidate = None
                                best_match_len = 0
                                for map_dir in maps_dir.iterdir():
                                    if map_dir.is_dir() and map_dir.name != "map_groups.json":
                                        map_normalized = map_dir.name.lower().replace(" ", "").replace("_", "")
                                        # Check if location is a substring of map name (prefer longer matches)
                                        if location_normalized in map_normalized:
                                            match_len = len(map_normalized)
                                            if match_len > best_match_len:
                                                best_candidate = map_dir.name
                                                best_match_len = match_len
                                if best_candidate:
                                    porymap_map_name = best_candidate
                    
                    if porymap_map_name:
                        # Build JSON map with grid for pathfinding
                        try:
                            json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root)
                        except ValueError as e:
                            logger.error(f"Failed to build porymap for '{porymap_map_name}' due to corrupted tileset: {e}")
                            json_map = None
                        
                        if json_map and 'grid' in json_map:
                            # Ensure map dict exists
                            if 'map' not in state:
                                state['map'] = {}
                            if 'porymap' not in state['map']:
                                state['map']['porymap'] = {}
                            
                            # Add porymap data for pathfinding
                            state['map']['porymap']['grid'] = json_map['grid']
                            state['map']['porymap']['objects'] = json_map.get('objects', [])
                            state['map']['porymap']['dimensions'] = json_map.get('dimensions', {})
                            state['map']['porymap']['warps'] = json_map.get('warps', [])
                            logger.debug(f"Loaded porymap data for '{porymap_map_name}' (ROM: '{location_name}')")
                        else:
                            logger.warning(f"Porymap data for '{porymap_map_name}' missing grid data")
                    else:
                        logger.warning(f"Could not map ROM location '{location_name}' to porymap map name")
                else:
                    logger.warning(f"Could not find pokeemerald root for porymap data")
            except Exception as porymap_err:
                logger.warning(f"Failed to load porymap data for pathfinding: {porymap_err}")

        # Get player position
        player_pos = state.get('player', {}).get('position', {})
        start_x = player_pos.get('x', 0)
        start_y = player_pos.get('y', 0)
        start = (start_x, start_y)
        goal = (x, y)

        # Check if requested goal is blocked (for agent notification)
        goal_was_blocked = False
        map_data = state.get('map', {}).get('porymap', {})
        if map_data.get('grid'):
            grid = map_data['grid']
            if 0 <= y < len(grid):
                row = grid[y]
                if isinstance(row, (list, str)) and 0 <= x < len(row):
                    cell = row[x] if isinstance(row, str) else row[x]
                    if cell == '#':
                        goal_was_blocked = True

        # Calculate path buttons using Pathfinder
        buttons = pathfinder.find_path(start, goal, state, variance=variance_level)

        if not buttons:
            return {
                "success": False,
                "error": "No path found to target location",
                "target": f"({x}, {y})"
            }

        nav_reason_text = f"Navigating from ({start_x}, {start_y}) to ({x}, {y})"
        if nav_reason:
            nav_reason_text += f": {nav_reason}"
        if variance_level:
            nav_reason_text += f" (variance={variance_level})"

        logger.info(f"🗺️ {nav_reason_text} - Buttons calculated: {len(buttons)}")

        result = {
            "success": True,
            "buttons": buttons,
            "target": f"({x}, {y})",
            "path_length": len(buttons),
            "reason": nav_reason,
            "variance": variance_level or "none"
        }
        
        # Inform agent if goal was blocked and adjusted
        if goal_was_blocked:
            result["note"] = f"Requested target ({x}, {y}) was blocked; navigated to nearest reachable tile"

        return result
    except Exception as e:
        logger.error(f"Failed to navigate: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


def add_knowledge_direct(category, title, content, location=None, coordinates=None, importance=3) -> dict:
    """Add knowledge to knowledge base - for use by server endpoints."""
    try:
        entry_id = knowledge_base.add(
            category=category,
            title=title,
            content=content,
            location=location,
            coordinates=coordinates,
            importance=importance
        )
        logger.info(f"📝 Added knowledge: {title} ({category})")
        return {
            "success": True,
            "entry_id": entry_id,
            "message": f"Stored knowledge: {title}"
        }
    except Exception as e:
        logger.error(f"Failed to add knowledge: {e}")
        return {"success": False, "error": str(e)}


def search_knowledge_direct(category=None, query="", location="", min_importance=1) -> dict:
    """Search knowledge base - for use by server endpoints."""
    try:
        results = knowledge_base.search(
            category=category,
            location=location or None,
            query=query or None,
            min_importance=min_importance
        )
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Failed to search knowledge: {e}")
        return {"success": False, "error": str(e)}


def get_knowledge_summary_direct(min_importance=3) -> dict:
    """Get knowledge summary - for use by server endpoints."""
    try:
        summary = knowledge_base.get_summary(min_importance=min_importance)
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Failed to get knowledge summary: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_game_state() -> dict:
    """
    Get the current game state including player position, party, map, items, and screenshot.
    Use this to manually inspect the game state when needed.

    Returns:
        Dictionary containing formatted state text and raw state data
    """
    try:
        response = requests.get(f"{SERVER_URL}/state", timeout=5)
        response.raise_for_status()
        state = response.json()

        # Format state for LLM
        state_text = format_state_for_llm(state)

        return {
            "success": True,
            "state_text": state_text,
            "player_position": state.get('player', {}).get('position', {}),
            "location": state.get('map', {}).get('current_map', 'Unknown')
        }
    except Exception as e:
        logger.error(f"Failed to get game state: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def press_buttons(
    buttons: List[str],
    reasoning: str = "",
    source: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Press buttons on the Game Boy Advance emulator. Buttons are executed sequentially.
    You will automatically receive the updated game state after buttons are executed.

    Args:
        buttons: List of buttons to press in sequence (e.g., ['A', 'A', 'B'])
                 Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R
        reasoning: Brief explanation of why you're pressing these buttons
        source: Optional label identifying where this action originated (e.g., 'navigate_to')
        metadata: Optional dictionary of metadata to attach to the action (e.g., {'variance': 'high'})

    Returns:
        Dictionary with success status, buttons pressed, and updated game state
    """
    if not buttons:
        return {"success": False, "error": "No buttons provided"}

    try:
        # Validate and normalize buttons
        VALID_BUTTONS = {'A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R'}
        normalized_buttons = []
        invalid_buttons = []
        
        for button in buttons:
            # Normalize to uppercase
            button_upper = str(button).upper().strip()
            
            # Check if valid
            if button_upper in VALID_BUTTONS:
                normalized_buttons.append(button_upper)
            else:
                # Invalid button - fallback to A and warn
                invalid_buttons.append(button)
                logger.warning(f"Invalid button '{button}' requested, falling back to 'A'")
                normalized_buttons.append('A')
        
        # Send normalized buttons to server
        payload = {"buttons": normalized_buttons}
        if source:
            payload["source"] = source
        if metadata is not None:
            payload["metadata"] = metadata

        response = requests.post(
            f"{SERVER_URL}/action",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        # Get updated state after action
        state_response = get_game_state()

        response_dict = {
            "success": True,
            "buttons_pressed": normalized_buttons,
            "reasoning": reasoning,
            "result": result,
            "new_state": state_response.get("state_text", "State unavailable")
        }
        
        # Include warning if any buttons were invalid
        if invalid_buttons:
            response_dict["warning"] = f"Invalid buttons replaced with 'A': {invalid_buttons}"
        
        return response_dict
    except Exception as e:
        logger.error(f"Failed to press buttons: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def navigate_to(x: int, y: int, variance: str = "none", reason: str = "") -> dict:
    """
    Automatically pathfind and move to a specific coordinate on the current map using A* algorithm.
    Handles collision detection and finds the optimal path. The pathfinding will be executed
    and you will receive state updates as you move.

    Args:
        x: Target X coordinate
        y: Target Y coordinate
        variance: Path variance level ('low', 'medium', 'high', or 'none')
        reason: Why you're navigating to this location (optional context)

    Returns:
        Dictionary with success status, path information, and navigation result
    """
    # Get current state
    state_response = get_game_state()
    if not state_response.get("success"):
        return {"success": False, "error": "Failed to get current game state"}

    # Get raw state for pathfinding
    try:
        response = requests.get(f"{SERVER_URL}/state", timeout=5)
        response.raise_for_status()
        state = response.json()
    except Exception as e:
        return {"success": False, "error": f"Failed to get state for pathfinding: {e}"}

    try:
        variance_level = None
        nav_reason = "" if reason is None else str(reason)

        if variance is not None:
            variance_str = str(variance).strip()
            variance_lower = variance_str.lower()
            if variance_lower in {"low", "medium", "high"}:
                variance_level = variance_lower
            elif variance_lower in {"", "none"}:
                variance_level = None
            elif not reason:
                nav_reason = variance_str
        elif nav_reason.lower() in {"low", "medium", "high"}:
            variance_level = nav_reason.lower()
            nav_reason = ""

        # Get player position from state
        player_pos = state.get('player', {}).get('position', {})
        start_x = player_pos.get('x', 0)
        start_y = player_pos.get('y', 0)
        start = (start_x, start_y)
        goal = (x, y)
        
        # Calculate path using Pathfinder
        buttons = pathfinder.find_path(start, goal, state, variance=variance_level)

        if not buttons:
            return {
                "success": False,
                "error": "No path found to target location",
                "target": f"({x}, {y})"
            }

        # Execute navigation
        nav_reason_text = f"Navigating to ({x}, {y})"
        if nav_reason:
            nav_reason_text += f": {nav_reason}"
        if variance_level:
            nav_reason_text += f" (variance={variance_level})"

        variance_metadata = variance_level or "none"
        result = press_buttons(
            buttons,
            nav_reason_text,
            source="navigate_to",
            metadata={"variance": variance_metadata}
        )

        return {
            "success": True,
            "target": f"({x}, {y})",
            "path_length": len(buttons),
            "buttons_executed": len(buttons),
            "reason": nav_reason,
            "variance": variance_level or "none",
            "navigation_result": result
        }
    except Exception as e:
        logger.error(f"Navigation failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def add_knowledge(
    category: str,
    title: str,
    content: str,
    location: str = "",
    coordinates: str = "",
    importance: int = 3
) -> dict:
    """
    Store important information in your persistent knowledge base.
    Use this to remember locations, NPCs, items, strategies, or any other useful information you discover.

    Args:
        category: Category of knowledge (location, npc, item, pokemon, strategy, custom)
        title: Brief title for this knowledge
        content: Detailed description or notes
        location: Map name where this applies (optional)
        coordinates: Coordinates like 'X:10,Y:20' (optional)
        importance: Importance level 1-5 (5 = critical, 3 = normal, 1 = minor)

    Returns:
        Dictionary with success status and entry ID
    """
    valid_categories = ["location", "npc", "item", "pokemon", "strategy", "custom"]
    if category not in valid_categories:
        return {
            "success": False,
            "error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"
        }

    if not 1 <= importance <= 5:
        return {"success": False, "error": "Importance must be between 1 and 5"}

    try:
        entry_id = knowledge_base.add(
            category=category,
            title=title,
            content=content,
            location=location or None,
            coordinates=coordinates or None,
            importance=importance
        )

        return {
            "success": True,
            "entry_id": entry_id,
            "message": f"Stored knowledge: {title}"
        }
    except Exception as e:
        logger.error(f"Failed to add knowledge: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def search_knowledge(
    category: str = "all",
    query: str = "",
    location: str = "",
    min_importance: int = 1
) -> dict:
    """
    Search your knowledge base for stored information.
    Use this to recall what you've learned about locations, NPCs, items, or strategies.

    Args:
        category: Category to search (location, npc, item, pokemon, strategy, custom, all)
        query: Text to search for in titles and content (optional)
        location: Filter by map name (optional)
        min_importance: Minimum importance level (1-5, default 1)

    Returns:
        Dictionary with success status, count, and search results
    """
    try:
        search_category = None if category == "all" else category

        results = knowledge_base.search(
            category=search_category,
            location=location or None,
            query=query or None,
            min_importance=min_importance
        )

        return {
            "success": True,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Failed to search knowledge: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_knowledge_summary(min_importance: int = 3) -> dict:
    """
    Get a summary of the most important things you've learned.
    Shows your top discoveries and notes.

    Args:
        min_importance: Minimum importance level to include (1-5, default 3)

    Returns:
        Dictionary with success status and formatted summary
    """
    try:
        summary = knowledge_base.get_summary(min_importance=min_importance)

        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Failed to get knowledge summary: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# POKEMON WIKI RESOURCES
# ============================================================================

# Curated list of Pokemon Emerald information sources
POKEMON_WIKI_SOURCES = {
    "bulbapedia": {
        "base_url": "https://bulbapedia.bulbagarden.net/wiki/",
        "search_url": "https://bulbapedia.bulbagarden.net/w/index.php?search=",
        "description": "Comprehensive Pokemon encyclopedia"
    },
    "serebii": {
        "base_url": "https://www.serebii.net/",
        "emerald_url": "https://www.serebii.net/emerald/",
        "description": "Detailed Pokemon Emerald guides and data"
    },
    "pokemondb": {
        "base_url": "https://pokemondb.net/",
        "emerald_url": "https://pokemondb.net/pokedex/game/emerald",
        "description": "Pokemon database with stats and locations"
    },
    "marriland": {
        "base_url": "https://marriland.com/",
        "emerald_url": "https://marriland.com/pokemon-emerald/",
        "description": "Strategy guides and walkthroughs"
    }
}


@mcp.tool()
def lookup_pokemon_info(topic: str, source: str = "bulbapedia") -> dict:
    """
    Look up information about Pokemon Emerald from trusted wiki sources.
    Use this to get details about Pokemon, moves, locations, items, NPCs, gym leaders, etc.

    Args:
        topic: What to search for (e.g., "Mudkip", "Route 101", "May", "Rare Candy")
        source: Which wiki to use (bulbapedia, serebii, pokemondb, marriland)

    Returns:
        Dictionary with success status and extracted wiki content

    Examples:
        lookup_pokemon_info("Mudkip")  # Get info about Mudkip
        lookup_pokemon_info("Route 103", "serebii")  # Route 103 details from Serebii
        lookup_pokemon_info("Norman", "bulbapedia")  # Info about Gym Leader Norman
    """
    try:
        if source not in POKEMON_WIKI_SOURCES:
            return {
                "success": False,
                "error": f"Unknown source '{source}'. Available: {', '.join(POKEMON_WIKI_SOURCES.keys())}"
            }

        source_info = POKEMON_WIKI_SOURCES[source]

        # Build URL based on source
        if source == "bulbapedia":
            # Bulbapedia uses wiki-style URLs
            formatted_topic = topic.replace(" ", "_")
            url = f"{source_info['base_url']}{formatted_topic}"
        elif source == "serebii":
            # Try emerald-specific section first
            formatted_topic = topic.lower().replace(" ", "")
            url = f"{source_info['emerald_url']}{formatted_topic}.shtml"
        elif source == "pokemondb":
            # PokemonDB uses lowercase with hyphens
            formatted_topic = topic.lower().replace(" ", "-")
            url = f"{source_info['base_url']}pokedex/{formatted_topic}"
        elif source == "marriland":
            formatted_topic = topic.lower().replace(" ", "-")
            url = f"{source_info['emerald_url']}{formatted_topic}/"
        else:
            url = f"{source_info['base_url']}{topic}"

        logger.info(f"Fetching Pokemon info: {topic} from {source} ({url})")

        # Fetch the page
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; PokeAgent/1.0)'
        })

        # If 404, try search instead
        if response.status_code == 404 and source == "bulbapedia":
            search_url = f"{source_info['search_url']}{quote_plus(topic)}"
            logger.info(f"Page not found, trying search: {search_url}")
            response = requests.get(search_url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; PokeAgent/1.0)'
            })

        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract main content based on source
        content = ""
        if source == "bulbapedia":
            # Bulbapedia has content in #mw-content-text
            main_content = soup.find('div', id='mw-content-text')
            if main_content:
                # Get first few paragraphs
                paragraphs = main_content.find_all('p', limit=5)
                content = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        else:
            # For other sources, get text from body
            content = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = '\n'.join(chunk for chunk in chunks if chunk)

        # Limit content length
        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Content truncated - {len(content)} total characters]"

        if not content or len(content) < 50:
            return {
                "success": False,
                "error": f"Could not extract meaningful content from {url}. Page may not exist or format changed."
            }

        return {
            "success": True,
            "topic": topic,
            "source": source,
            "url": url,
            "content": content,
            "description": source_info['description']
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {topic} from {source}: {e}")
        return {
            "success": False,
            "error": f"Failed to fetch from {source}: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error looking up {topic}: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def list_wiki_sources() -> dict:
    """
    List available Pokemon wiki sources and what they're good for.
    Use this to see which sources you can query with lookup_pokemon_info().

    Returns:
        Dictionary with available sources and their descriptions
    """
    sources = []
    for name, info in POKEMON_WIKI_SOURCES.items():
        sources.append({
            "name": name,
            "description": info['description'],
            "base_url": info.get('base_url', ''),
            "emerald_url": info.get('emerald_url', '')
        })

    return {
        "success": True,
        "sources": sources,
        "count": len(sources),
        "usage": "Use lookup_pokemon_info(topic, source) to fetch information"
    }


@mcp.tool()
def get_walkthrough(part: int) -> dict:
    """
    Get the official Bulbapedia walkthrough for Pokemon Emerald.
    Use this to understand the intended game progression and what to do next.

    Args:
        part: Walkthrough part number (1-21)
            Part 1: Introduction, Littleroot Town
            Part 2: Route 101, Oldale Town
            Part 3: Route 103, Rival Battle
            Part 4: Route 102, Petalburg City
            Part 5: Petalburg Woods, Rustboro City
            Part 6: Roxanne (Gym 1), Route 116
            Part 7: Rusturf Tunnel, Dewford Town
            Part 8: Brawly (Gym 2), Route 106-109
            Part 9: Slateport City, Route 110
            Part 10: Mauville City, Wattson (Gym 3)
            Part 11: Route 117-111, Desert, Lavaridge Town
            Part 12: Flannery (Gym 4), Route 112-113
            Part 13: Fallarbor Town, Meteor Falls
            Part 14: Mt. Chimney, Team Magma
            Part 15: Petalburg Gym, Norman (Gym 5)
            Part 16: Route 118-123, Lilycove City
            Part 17: Team Aqua/Magma, Mt. Pyre
            Part 18: Mossdeep City, Tate & Liza (Gym 7)
            Part 19: Seafloor Cavern, Sootopolis City
            Part 20: Wallace (Gym 8), Victory Road
            Part 21: Pokemon League, Elite Four, Champion

    Returns:
        Dictionary with success status and walkthrough content

    Example:
        get_walkthrough(1)  # Get Part 1: Introduction and Littleroot Town
        get_walkthrough(6)  # Get Part 6: Roxanne battle and Route 116
    """
    try:
        if not 1 <= part <= 21:
            return {
                "success": False,
                "error": f"Part must be between 1 and 21 (got {part})"
            }

        # Build Bulbapedia walkthrough URL
        url = f"https://bulbapedia.bulbagarden.net/wiki/Walkthrough:Pok%C3%A9mon_Emerald/Part_{part}"

        logger.info(f"Fetching Emerald walkthrough part {part}: {url}")

        # Fetch the page
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; PokeAgent/1.0)'
        })
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'table']):
            element.decompose()

        # Extract main content
        main_content = soup.find('div', id='mw-content-text')
        if not main_content:
            return {
                "success": False,
                "error": f"Could not find main content for Part {part}"
            }

        # Get all paragraphs and headings for structured walkthrough
        content_parts = []
        for element in main_content.find_all(['h2', 'h3', 'h4', 'p', 'ul'], limit=50):
            if element.name in ['h2', 'h3', 'h4']:
                # Add headings with formatting
                heading_text = element.get_text(strip=True)
                if heading_text and not heading_text.startswith('[edit]'):
                    level = element.name
                    if level == 'h2':
                        content_parts.append(f"\n## {heading_text}")
                    elif level == 'h3':
                        content_parts.append(f"\n### {heading_text}")
                    else:
                        content_parts.append(f"\n#### {heading_text}")
            elif element.name == 'p':
                # Add paragraphs
                para_text = element.get_text(strip=True)
                if para_text and len(para_text) > 20:  # Skip short fragments
                    content_parts.append(para_text)
            elif element.name == 'ul':
                # Add lists
                for li in element.find_all('li'):
                    li_text = li.get_text(strip=True)
                    if li_text:
                        content_parts.append(f"  - {li_text}")

        content = '\n\n'.join(content_parts)

        # Limit content length
        max_chars = 8000  # Larger for walkthrough
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Content truncated - {len(content)} total characters]"

        if not content or len(content) < 100:
            return {
                "success": False,
                "error": f"Could not extract meaningful content from Part {part}"
            }

        return {
            "success": True,
            "part": part,
            "url": url,
            "content": content,
            "description": f"Pokemon Emerald Walkthrough - Part {part}"
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch walkthrough part {part}: {e}")
        return {
            "success": False,
            "error": f"Failed to fetch Part {part}: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error getting walkthrough part {part}: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    logger.info("Pokemon MCP Server starting...")
    logger.info("Server name: pokemon-emerald")
    logger.info("Available tools:")
    logger.info("  Game: get_game_state, press_buttons, navigate_to")
    logger.info("  Knowledge: add_knowledge, search_knowledge, get_knowledge_summary")
    logger.info("  Wiki: lookup_pokemon_info, list_wiki_sources, get_walkthrough")
    logger.info("  Walkthrough: 21 parts available (Part 1-21)")

    # Run the MCP server with stdio transport
    mcp.run(transport='stdio')

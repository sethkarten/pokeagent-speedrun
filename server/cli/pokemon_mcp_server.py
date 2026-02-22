#!/usr/bin/env python3
"""
Thin MCP proxy for the Pokemon Emerald game server.

Every @mcp.tool() function forwards its arguments as an HTTP request to the
corresponding /mcp/* endpoint on the game server (server/app.py).  No game
logic lives here — this module is purely a transport adapter between the
MCP protocol (used by Claude Code / Codex CLI / Gemini CLI) and the game server's
REST API.

Available tools (13 total):
  Game:        get_game_state, press_buttons, navigate_to
  Knowledge:   add_knowledge, search_knowledge, get_knowledge_summary
  Wiki:        lookup_pokemon_info, list_wiki_sources, get_walkthrough
  Objectives:  complete_direct_objective, create_direct_objectives, get_progress_summary
  Reflection:  reflect
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SERVER_URL = os.environ.get("POKEMON_SERVER_URL", "http://localhost:8000")

mcp = FastMCP(name="pokemon-emerald")

_TIMEOUT_SHORT = 10   # seconds — lightweight reads
_TIMEOUT_MEDIUM = 30  # seconds — actions/pathfinding


def _post(path: str, body: dict | None = None, timeout: int = _TIMEOUT_MEDIUM) -> dict:
    """POST to the game server and return the JSON response (or an error dict)."""
    try:
        resp = requests.post(f"{SERVER_URL}{path}", json=body or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error calling {path}: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Game tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_game_state() -> dict:
    """
    Get the current game state including player position, party, map, items, and screenshot.
    Use this to manually inspect the game state when needed.

    Returns:
        Dictionary containing formatted state text and raw state data
    """
    return _post("/mcp/get_game_state", timeout=_TIMEOUT_SHORT)


@mcp.tool()
def press_buttons(
    buttons: List[str],
    speed: str = "normal",
    hold_frames: Optional[int] = None,
    release_frames: Optional[int] = None,
    reasoning: str = "",
    source: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Press buttons on the Game Boy Advance emulator with optional speed control.
    Buttons are executed sequentially. You control action timing for optimal gameplay.

    Args:
        buttons: List of buttons to press in sequence (e.g., ['A', 'A', 'B'])
                 Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT, L, R, WAIT
        speed: Action speed preset - "fast" (dialogue/menus, 9 frames), "normal" (movement, 18 frames),
               or "slow" (careful inputs, 32 frames). Default is "normal".
        hold_frames: Optional explicit hold duration in frames (overrides speed preset)
        release_frames: Optional explicit release duration in frames (overrides speed preset)
        reasoning: Brief explanation of why you're pressing these buttons
        source: Optional label identifying where this action originated (e.g., 'navigate_to')
        metadata: Optional dictionary of metadata to attach to the action (e.g., {'variance': 'high'})

    Speed Guide:
        - "fast": Use for dialogue advancement, menu spam, rapid button presses (9 frames = ~0.09s at 100 FPS)
        - "normal": Use for movement, pathfinding, general gameplay (18 frames = ~0.18s at 100 FPS) [DEFAULT]
        - "slow": Use for critical inputs, careful timing (32 frames = ~0.32s at 100 FPS)

    WAIT Action:
        - Use WAIT to pause without pressing buttons (e.g., waiting for NPCs, animations)
        - Example: press_buttons(["WAIT"], speed="slow") for a long wait
        - Example: press_buttons(["WAIT"], release_frames=60) for a custom 60-frame wait

    Returns:
        Dictionary with success status, buttons pressed, and updated game state
    """
    body: Dict[str, Any] = {"buttons": buttons, "reasoning": reasoning}
    if speed:
        body["speed"] = speed
    if hold_frames is not None:
        body["hold_frames"] = hold_frames
    if release_frames is not None:
        body["release_frames"] = release_frames
    if source:
        body["source"] = source
    if metadata is not None:
        body["metadata"] = metadata
    return _post("/mcp/press_buttons", body)


@mcp.tool()
def navigate_to(
    x: int,
    y: int,
    variance: str = "none",
    reason: str = "",
    consider_npcs: bool = True,
    blocked_coords: Optional[List[List[int]]] = None,
) -> dict:
    """
    Automatically pathfind and move to a specific coordinate on the current map using A* algorithm.
    Handles collision detection and finds the optimal path. The pathfinding will be executed
    and you will receive state updates as you move.

    Args:
        x: Target X coordinate
        y: Target Y coordinate
        variance: Path variance level ('low', 'medium', 'high', or 'none')
        reason: Why you're navigating to this location (optional context)
        consider_npcs: Whether to avoid NPC positions during pathfinding (default True, NPCs avoided)
        blocked_coords: Optional list of additional coordinates to treat as blocked (e.g. [ [10, 11], [10, 12] ])

    Returns:
        Dictionary with success status, path information, and navigation result
    """
    body: Dict[str, Any] = {"x": x, "y": y, "reason": reason}
    if variance and variance != "none":
        body["variance"] = variance
    if not consider_npcs:
        body["consider_npcs"] = False
    if blocked_coords:
        body["blocked_coords"] = blocked_coords
    return _post("/mcp/navigate_to", body)


# ---------------------------------------------------------------------------
# Knowledge tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add_knowledge(
    category: str, title: str, content: str, location: str = "", coordinates: str = "", importance: int = 3
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
    return _post("/mcp/add_knowledge", {
        "category": category,
        "title": title,
        "content": content,
        "location": location,
        "coordinates": coordinates,
        "importance": importance,
    })


@mcp.tool()
def search_knowledge(category: str = "all", query: str = "", location: str = "", min_importance: int = 1) -> dict:
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
    return _post("/mcp/search_knowledge", {
        "category": category,
        "query": query,
        "location": location,
        "min_importance": min_importance,
    })


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
    return _post("/mcp/get_knowledge_summary", {"min_importance": min_importance})


# ---------------------------------------------------------------------------
# Wiki / walkthrough tools
# ---------------------------------------------------------------------------

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
    return _post("/mcp/lookup_pokemon_info", {"topic": topic, "source": source}, timeout=_TIMEOUT_MEDIUM)


@mcp.tool()
def list_wiki_sources() -> dict:
    """
    List available Pokemon wiki sources and what they're good for.
    Use this to see which sources you can query with lookup_pokemon_info().

    Returns:
        Dictionary with available sources and their descriptions
    """
    return _post("/mcp/list_wiki_sources", timeout=_TIMEOUT_SHORT)


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
    return _post("/mcp/get_walkthrough", {"part": part}, timeout=_TIMEOUT_MEDIUM)


# ---------------------------------------------------------------------------
# Objective / progress tools
# ---------------------------------------------------------------------------

@mcp.tool()
def complete_direct_objective(
    objective_id: str,
    completion_notes: str = "",
    category: str = "",
) -> dict:
    """
    Mark the current direct objective as completed.
    Call this when you've successfully accomplished the current objective.

    Args:
        objective_id: The ID of the objective to complete (must match the current objective)
        completion_notes: Optional notes about how the objective was completed
        category: Category of the objective (story, battling, dynamics) — required in categorized mode

    Returns:
        Dictionary with success status, completed objective info, and next objective
    """
    body: Dict[str, Any] = {"objective_id": objective_id}
    if completion_notes:
        body["completion_notes"] = completion_notes
    if category:
        body["category"] = category
    return _post("/mcp/complete_direct_objective", body)


@mcp.tool()
def create_direct_objectives(
    objectives: List[Dict[str, Any]],
    reasoning: str = "",
    category: str = "",
) -> dict:
    """
    Create the next 3 direct objectives dynamically.
    Use this after completing all objectives in a sequence to plan your next steps.

    Args:
        objectives: List of exactly 3 objective dicts, each with:
            - id (str): Unique objective identifier
            - description (str): What needs to be done
            - action_type (str): Type of action (navigate, interact, battle, create_new_objectives)
            - target_location (str, optional): Where this takes place
            - navigation_hint (str, optional): How to get there
            - completion_condition (str, optional): How to know it's done
            - priority (int, optional): Priority level
        reasoning: Why these objectives were chosen
        category: Category for categorized mode (story, battling, dynamics)

    Returns:
        Dictionary with success status and next objective guidance
    """
    body: Dict[str, Any] = {"objectives": objectives}
    if reasoning:
        body["reasoning"] = reasoning
    if category:
        body["category"] = category
    return _post("/mcp/create_direct_objectives", body)


@mcp.tool()
def get_progress_summary() -> dict:
    """
    Get comprehensive progress summary including milestones, completed objectives, and knowledge.
    Use this to review your overall progress and plan next steps.

    Returns:
        Dictionary with success status and progress data including:
        - Milestones completed
        - Direct objectives status
        - Knowledge base summary
        - Current location and coordinates
    """
    return _post("/mcp/get_progress_summary", timeout=_TIMEOUT_SHORT)


@mcp.tool()
def reflect(situation: str = "Agent requested reflection") -> dict:
    """
    Return context data for self-reflection on current progress.
    Use this to analyze what you've done recently, assess whether you're stuck,
    and plan your next moves based on game state and objective progress.

    Args:
        situation: Description of the current situation or why you're reflecting

    Returns:
        Dictionary with context data including:
        - Current game state summary
        - Current objective and progress
        - Recent action history
        - Porymap ground truth data
    """
    return _post("/mcp/reflect", {"situation": situation})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Pokemon MCP Server starting (thin proxy mode)...")
    logger.info(f"Proxying to game server at: {SERVER_URL}")
    logger.info("Server name: pokemon-emerald")
    logger.info("Available tools (13):")
    logger.info("  Game:       get_game_state, press_buttons, navigate_to")
    logger.info("  Knowledge:  add_knowledge, search_knowledge, get_knowledge_summary")
    logger.info("  Wiki:       lookup_pokemon_info, list_wiki_sources, get_walkthrough")
    logger.info("  Objectives: complete_direct_objective, create_direct_objectives, get_progress_summary")
    logger.info("  Reflection: reflect")

    mcp.run(transport="stdio")

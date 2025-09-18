#!/usr/bin/env python3
"""
Centralized Map Formatting Utility

Single source of truth for all map formatting across the codebase.
"""

from pokemon_env.enums import MetatileBehavior


def format_tile_to_symbol(tile):
    """
    Convert a single tile to its display symbol.
    
    Args:
        tile: Tuple of (tile_id, behavior, collision, elevation)
        
    Returns:
        str: Single character symbol representing the tile
    """
    if len(tile) >= 4:
        tile_id, behavior, collision, _ = tile  # elevation not used
    elif len(tile) >= 2:
        tile_id, behavior = tile[:2]
        collision = 0
    else:
        tile_id = tile[0] if tile else 0
        behavior = MetatileBehavior.NORMAL
        collision = 0
    
    # Convert behavior to symbol using unified logic
    if hasattr(behavior, 'name'):
        behavior_name = behavior.name
    elif isinstance(behavior, int):
        try:
            behavior_enum = MetatileBehavior(behavior)
            behavior_name = behavior_enum.name
        except ValueError:
            behavior_name = "UNKNOWN"
    else:
        behavior_name = "UNKNOWN"
    
    # Map to symbol - SINGLE SOURCE OF TRUTH
    if tile_id == 1023:  # Unknown/unloaded tile
        return "#"  # Mark as blocked
    elif behavior_name == "NORMAL":
        return "." if collision == 0 else "#"
    elif "DOOR" in behavior_name:
        return "D"
    elif "STAIRS" in behavior_name or "WARP" in behavior_name:
        return "S"
    elif "WATER" in behavior_name:
        return "W"
    elif "TALL_GRASS" in behavior_name:
        return "~"
    elif "COMPUTER" in behavior_name or "PC" in behavior_name:
        return "PC"  # PC/Computer
    elif "TELEVISION" in behavior_name or "TV" in behavior_name:
        return "T"  # Television
    elif "BOOKSHELF" in behavior_name or "SHELF" in behavior_name:
        return "B"  # Bookshelf
    elif "SIGN" in behavior_name or "SIGNPOST" in behavior_name:
        return "?"  # Sign/Information
    elif "FLOWER" in behavior_name or "PLANT" in behavior_name:
        return "F"  # Flowers/Plants
    elif "COUNTER" in behavior_name or "DESK" in behavior_name:
        return "C"  # Counter/Desk
    elif "BED" in behavior_name or "SLEEP" in behavior_name:
        return "="  # Bed
    elif "TABLE" in behavior_name or "CHAIR" in behavior_name:
        return "t"  # Table/Chair
    elif "JUMP" in behavior_name:
        if "SOUTH" in behavior_name:
            return "↓"
        elif "EAST" in behavior_name:
            return "→"
        elif "WEST" in behavior_name:
            return "←"
        elif "NORTH" in behavior_name:
            return "↑"
        elif "NORTHEAST" in behavior_name:
            return "↗"
        elif "NORTHWEST" in behavior_name:
            return "↖"
        elif "SOUTHEAST" in behavior_name:
            return "↘"
        elif "SOUTHWEST" in behavior_name:
            return "↙"
        else:
            return "J"
    elif "IMPASSABLE" in behavior_name or "SEALED" in behavior_name:
        return "#"  # Blocked
    elif "INDOOR" in behavior_name:
        return "."  # Indoor tiles are walkable
    elif "DECORATION" in behavior_name or "HOLDS" in behavior_name:
        return "."  # Decorations are walkable
    else:
        # For unknown behavior, mark as blocked for safety
        return "#"


def format_map_grid(raw_tiles, player_facing="South", npcs=None, player_coords=None):
    """
    Format raw tile data into a traversability grid with NPCs.
    
    Args:
        raw_tiles: 2D list of tile tuples
        player_facing: Player facing direction for center marker
        npcs: List of NPC/object events with positions
        
    Returns:
        list: 2D list of symbol strings
    """
    if not raw_tiles or len(raw_tiles) == 0:
        return []
    
    grid = []
    center_y = len(raw_tiles) // 2
    center_x = len(raw_tiles[0]) // 2
    
    # Player is always at the center of the 15x15 grid view
    # but we need the actual player coordinates for NPC positioning
    player_map_x = center_x  # Grid position (always 7,7 in 15x15)
    player_map_y = center_y
    
    # Always use P for player instead of direction arrows
    player_symbol = "P"
    
    # Create NPC position lookup (convert to relative grid coordinates)
    npc_positions = {}
    if npcs and player_coords:
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
            
            for npc in npcs:
                # NPCs have absolute world coordinates, convert to relative grid position
                npc_abs_x = npc.get('current_x', 0)
                npc_abs_y = npc.get('current_y', 0)
                
                # Ensure NPC coordinates are integers
                npc_abs_x = int(npc_abs_x) if npc_abs_x is not None else 0
                npc_abs_y = int(npc_abs_y) if npc_abs_y is not None else 0
                
                # Calculate offset from player in absolute coordinates
                offset_x = npc_abs_x - player_abs_x
                offset_y = npc_abs_y - player_abs_y
                
                # Convert offset to grid position (player is at center)
                grid_x = center_x + offset_x
                grid_y = center_y + offset_y
                
                # Check if NPC is within our grid view
                if 0 <= grid_x < len(raw_tiles[0]) and 0 <= grid_y < len(raw_tiles):
                    npc_positions[(grid_y, grid_x)] = npc
                    
        except (ValueError, TypeError) as e:
            # If coordinate conversion fails, skip NPC positioning
            print(f"Warning: Failed to convert coordinates for NPC positioning: {e}")
            print(f"  player_coords: {player_coords}")
            if npcs:
                print(f"  npc coords: {[(npc.get('current_x'), npc.get('current_y')) for npc in npcs]}")
            npc_positions = {}
    
    for y, row in enumerate(raw_tiles):
        grid_row = []
        for x, tile in enumerate(row):
            if y == center_y and x == center_x:
                # Player position
                grid_row.append(player_symbol)
            elif (y, x) in npc_positions:
                # NPC position - use NPC symbol
                npc = npc_positions[(y, x)]
                # Use different symbols for different NPC types
                if npc.get('trainer_type', 0) > 0:
                    grid_row.append("@")  # Trainer
                else:
                    grid_row.append("N")  # Regular NPC
            else:
                # Regular tile
                symbol = format_tile_to_symbol(tile)
                grid_row.append(symbol)
        grid.append(grid_row)
    
    return grid


def format_map_for_display(raw_tiles, player_facing="South", title="Map", npcs=None, player_coords=None):
    """
    Format raw tiles into a complete display string with headers and legend.
    
    Args:
        raw_tiles: 2D list of tile tuples
        player_facing: Player facing direction
        title: Title for the map display
        npcs: List of NPC/object events with positions
        player_coords: Dict with player absolute coordinates {'x': x, 'y': y}
        
    Returns:
        str: Formatted map display
    """
    if not raw_tiles:
        return f"{title}: No map data available"
    
    # Convert player_coords to tuple if it's a dict
    if player_coords and isinstance(player_coords, dict):
        player_coords_tuple = (player_coords['x'], player_coords['y'])
    else:
        player_coords_tuple = player_coords
    
    grid = format_map_grid(raw_tiles, player_facing, npcs, player_coords_tuple)
    
    lines = [f"{title} ({len(grid)}x{len(grid[0])}):", ""]
    
    # Add column headers
    header = "      "
    for i in range(len(grid[0])):
        header += f"{i:2} "
    lines.append(header)
    lines.append("     " + "--" * len(grid[0]))
    
    # Add grid with row numbers
    for y, row in enumerate(grid):
        row_str = f"  {y:2}: " + " ".join(f"{cell:2}" for cell in row)
        lines.append(row_str)
    
    # Add dynamic legend based on symbols that appear
    lines.append("")
    lines.append(generate_dynamic_legend(grid))
    
    return "\n".join(lines)


def get_symbol_legend():
    """
    Get the complete symbol legend for map displays.
    
    Returns:
        dict: Symbol -> description mapping
    """
    return {
        "P": "Player",
        ".": "Walkable path",
        "#": "Wall/Blocked/Unknown",
        "D": "Door",
        "S": "Stairs/Warp",
        "W": "Water",
        "~": "Tall grass",
        "PC": "PC/Computer",
        "T": "Television",
        "B": "Bookshelf", 
        "?": "Unexplored area",
        "F": "Flowers/Plants",
        "C": "Counter/Desk",
        "=": "Bed",
        "t": "Table/Chair",
        "J": "Jump ledge",
        "↗": "Jump Northeast",
        "↖": "Jump Northwest", 
        "↘": "Jump Southeast",
        "↙": "Jump Southwest",
        "N": "NPC",
        "@": "Trainer"
    }


def generate_dynamic_legend(grid):
    """
    Generate a legend based on symbols that actually appear in the grid.
    
    Args:
        grid: 2D list of symbol strings
        
    Returns:
        str: Formatted legend string
    """
    if not grid:
        return ""
    
    symbol_legend = get_symbol_legend()
    symbols_used = set()
    
    # Collect all unique symbols in the grid
    for row in grid:
        for symbol in row:
            symbols_used.add(symbol)
    
    # Build legend for used symbols
    legend_lines = ["Legend:"]
    
    # Group symbols by category for better organization
    player_symbols = ["P"]
    terrain_symbols = [".", "#", "W", "~", "?"] 
    structure_symbols = ["D", "S"]
    jump_symbols = ["J", "↗", "↖", "↘", "↙"]
    furniture_symbols = ["PC", "T", "B", "F", "C", "=", "t"]
    npc_symbols = ["N", "@"]
    
    categories = [
        ("Movement", player_symbols),
        ("Terrain", terrain_symbols),
        ("Structures", structure_symbols), 
        ("Jump ledges", jump_symbols),
        ("Furniture", furniture_symbols),
        ("NPCs", npc_symbols)
    ]
    
    for category_name, symbol_list in categories:
        category_items = []
        for symbol in symbol_list:
            if symbol in symbols_used and symbol in symbol_legend:
                category_items.append(f"{symbol}={symbol_legend[symbol]}")
        
        if category_items:
            legend_lines.append(f"  {category_name}: {', '.join(category_items)}")
    
    return "\n".join(legend_lines)


def format_map_for_llm(raw_tiles, player_facing="South", npcs=None, player_coords=None):
    """
    Format raw tiles into LLM-friendly grid format (no headers/legends).
    
    Args:
        raw_tiles: 2D list of tile tuples  
        player_facing: Player facing direction
        npcs: List of NPC/object events with positions
        player_coords: Tuple of (player_x, player_y) in absolute world coordinates
        
    Returns:
        str: Grid format suitable for LLM
    """
    if not raw_tiles:
        return "No map data available"
    
    grid = format_map_grid(raw_tiles, player_facing, npcs, player_coords)
    
    # Simple grid format for LLM
    lines = []
    for row in grid:
        lines.append(" ".join(row))
    
    return "\n".join(lines)
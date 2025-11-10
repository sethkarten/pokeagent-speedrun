#!/usr/bin/env python3
"""
Centralized Map Formatting Utility

Single source of truth for all map formatting across the codebase.
"""

try:
    from pokemon_env.enums import MetatileBehavior
except ImportError:  # Allow usage in builder scripts without mgba
    class _FallbackEnum(int):
        name = "NORMAL"

        def __new__(cls, value=0):
            obj = int.__new__(cls, value)
            obj._name = "NORMAL"
            return obj

        @property
        def name(self):
            return self._name

    MetatileBehavior = type("MetatileBehavior", (), {"NORMAL": _FallbackEnum(0)})


def format_tile_to_symbol(tile, x=None, y=None, location_name=None, player_pos=None, stairs_pos=None):
    """
    Convert a single tile to its display symbol.
    
    Args:
        tile: Tuple of (tile_id, behavior, collision, elevation)
        x: Optional x coordinate for context-specific symbols
        y: Optional y coordinate for context-specific symbols
        location_name: Optional location name for context-specific symbols
        player_pos: Optional player position tuple (px, py) for relative positioning
        stairs_pos: Optional stairs position tuple (sx, sy) for relative positioning
        
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
    
    # Special handling for Brendan's House 2F wall clock
    # The clock is a wall tile with no special behavior, just tile ID 1023
    # Position it relative to the stairs dynamically
    if location_name and "BRENDAN" in location_name.upper() and "2F" in location_name.upper():
        if x is not None and y is not None and stairs_pos:
            sx, sy = stairs_pos
            # Clock is 2 tiles west of stairs on the same row
            if (x, y) == (sx - 2, sy):
                return "K"  # K for Klock (C is taken by Computer)
    
    # Map to symbol - SINGLE SOURCE OF TRUTH
    # tile_id 1023 (0x3FF) is usually invalid/out-of-bounds
    if tile_id == 1023:
        return "#"  # Always show as blocked/wall
    elif behavior_name == "NORMAL":
        return "." if collision == 0 else "#"
    # Fix for reversed door/stairs mapping in Brendan's house
    # NON_ANIMATED_DOOR (96) appears at top and should show as 'S' 
    # SOUTH_ARROW_WARP (101) appears at bottom and should show as 'D'
    elif behavior == 96 or "NON_ANIMATED_DOOR" in behavior_name:
        return "S"  # This is actually stairs going upstairs
    elif behavior == 101 or "SOUTH_ARROW_WARP" in behavior_name:
        return "D"  # This is actually the exit door
    elif "DOOR" in behavior_name:
        return "D"  # Other doors remain as doors
    elif "STAIRS" in behavior_name or "WARP" in behavior_name:
        return "S"  # Other stairs/warps remain as stairs
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
    elif "CLOCK" in behavior_name:
        return "O"  # Clock (O for clock face)
    elif "PICTURE" in behavior_name or "PAINTING" in behavior_name:
        return "^"  # Picture/Painting on wall
    elif "TRASH" in behavior_name or "BIN" in behavior_name:
        return "U"  # Trash can/bin
    elif "POT" in behavior_name or "VASE" in behavior_name:
        return "V"  # Pot/Vase
    elif "MACHINE" in behavior_name or "DEVICE" in behavior_name:
        return "M"  # Machine/Device
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
    elif "BRIDGE" in behavior_name:
        return "&"  # Bridge tiles are walkable
    elif "IMPASSABLE" in behavior_name or "SEALED" in behavior_name:
        return "#"  # Blocked
    elif "INDOOR" in behavior_name:
        return "."  # Indoor tiles are walkable
    elif "DECORATION" in behavior_name or "HOLDS" in behavior_name:
        return "."  # Decorations are walkable
    elif behavior == 999:
        return "N"  # NPC marker (visually detected)
    else:
        # For unknown behavior, mark as blocked for safety
        return "#"


def format_map_grid(raw_tiles, player_facing="South", npcs=None, player_coords=None, trim_padding=True, location_name=None):
    """
    Format raw tile data into a traversability grid with NPCs.
    
    Args:
        raw_tiles: 2D list of tile tuples
        player_facing: Player facing direction for center marker
        npcs: List of NPC/object events with positions
        player_coords: Player coordinates for relative positioning
        trim_padding: If True, remove padding rows/columns that are all walls
        location_name: Optional location name for context-specific symbols
        
    Returns:
        list: 2D list of symbol strings
    """
    if not raw_tiles or len(raw_tiles) == 0:
        return []
    
    # First pass: find the stairs position if in Brendan's house 2F
    stairs_pos = None
    if location_name and "BRENDAN" in location_name.upper() and "2F" in location_name.upper():
        for y, row in enumerate(raw_tiles):
            for x, tile in enumerate(row):
                if len(tile) >= 2:
                    _, behavior = tile[:2]
                    # Stairs have behavior 96 (NON_ANIMATED_DOOR which we mapped to 'S')
                    if behavior == 96:
                        stairs_pos = (x, y)
                        break
            if stairs_pos:
                break
    
    grid = []
    center_y = len(raw_tiles) // 2
    center_x = len(raw_tiles[0]) // 2
    
    # Player is always at the center of the 15x15 grid view
    # but we need the actual player coordinates for NPC positioning
    player_map_x = center_x  # Grid position (always 7,7 in 15x15)
    player_map_y = center_y
    
    # Always use P for player instead of direction arrows
    player_symbol = "P"
    
    # NPCs are not displayed on the map grid - they appear in movement preview only
    
    for y, row in enumerate(raw_tiles):
        grid_row = []
        for x, tile in enumerate(row):
            if y == center_y and x == center_x:
                # Player position
                grid_row.append(player_symbol)
            else:
                # Regular tile - pass coordinates and context for special handling
                symbol = format_tile_to_symbol(tile, x=x, y=y, location_name=location_name, 
                                               player_pos=(center_x, center_y), stairs_pos=stairs_pos)
                grid_row.append(symbol)
        grid.append(grid_row)
    
    # Trim padding if requested - but keep room boundaries!
    if trim_padding and len(grid) > 0:
        # First pass: Remove obvious padding (rows/columns that are ALL walls with no variation)
        # But we need to be careful to keep actual room walls
        
        # Check if we have any content in the middle
        has_walkable = False
        for row in grid:
            if any(cell in ['.', 'P', 'D', 'N', 'T', 'S'] for cell in row):
                has_walkable = True
                break
        
        if has_walkable:
            # Only trim extra padding beyond the first wall layer
            # Count consecutive wall rows from top
            top_wall_rows = 0
            for row in grid:
                if all(cell == '#' for cell in row):
                    top_wall_rows += 1
                else:
                    break
            
            # Remove extra top padding but keep one wall row
            while top_wall_rows > 1 and len(grid) > 1:
                grid.pop(0)
                top_wall_rows -= 1
            
            # Count consecutive wall rows from bottom
            bottom_wall_rows = 0
            for row in reversed(grid):
                if all(cell == '#' for cell in row):
                    bottom_wall_rows += 1
                else:
                    break
            
            # Remove extra bottom padding but keep one wall row
            while bottom_wall_rows > 1 and len(grid) > 1:
                grid.pop()
                bottom_wall_rows -= 1
            
            # Similar for left/right but be more conservative
            # Don't trim sides if we have doors or other features in the walls
    
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
        "&": "Bridge (walkable)",
        "t": "Table/Chair",
        "K": "Clock (Wall)",
        "O": "Clock",
        "^": "Picture/Painting",
        "U": "Trash can",
        "V": "Pot/Vase",
        "M": "Machine/Device",
        "J": "Jump ledge",
        "↓": "Jump South",
        "↑": "Jump North",
        "←": "Jump West",
        "→": "Jump East",
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
    jump_symbols = ["J", "↓", "↑", "←", "→", "↗", "↖", "↘", "↙"]
    furniture_symbols = ["PC", "T", "B", "F", "C", "=", "t", "K", "O", "^", "U", "V", "M"]
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


def format_map_for_llm(raw_tiles, player_facing="South", npcs=None, player_coords=None, location_name=None):
    """
    Format raw tiles into LLM-friendly grid format with detailed NPC information.
    
    Args:
        raw_tiles: 2D list of tile tuples  
        player_facing: Direction player is facing
        npcs: List of NPC/object events
        player_coords: Player position for relative positioning
        location_name: Location name for context-specific symbols  
        player_facing: Player facing direction
        npcs: List of NPC/object events with positions
        player_coords: Tuple of (player_x, player_y) in absolute world coordinates
        
    Returns:
        str: Grid format suitable for LLM with NPC details
    """
    if not raw_tiles:
        return "No map data available"
    
    grid = format_map_grid(raw_tiles, player_facing, npcs, player_coords, location_name=location_name)
    
    # Simple grid format for LLM
    lines = []
    for row in grid:
        lines.append(" ".join(row))
    
    # NPCs are shown in movement preview, not on the map
    
    return "\n".join(lines)

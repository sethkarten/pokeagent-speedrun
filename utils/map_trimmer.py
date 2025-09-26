#!/usr/bin/env python3
"""
Map Trimming Utility

Removes unnecessary padding from maps - rows/columns that are all walls (#)
with no meaningful content should be trimmed.
"""

def trim_map_padding(grid_dict):
    """
    Trim unnecessary padding from a map grid.
    
    Args:
        grid_dict: Dictionary mapping (x, y) to symbols
        
    Returns:
        Trimmed grid dictionary
    """
    if not grid_dict:
        return grid_dict
    
    # Get bounds
    all_coords = list(grid_dict.keys())
    if not all_coords:
        return grid_dict
        
    min_x = min(x for x, y in all_coords)
    max_x = max(x for x, y in all_coords)
    min_y = min(y for x, y in all_coords)
    max_y = max(y for x, y in all_coords)
    
    # Check each edge to see if it can be trimmed
    # A row/column can be trimmed if it's all # or empty
    
    # Check top rows
    trim_top = 0
    for y in range(min_y, max_y + 1):
        row_values = [grid_dict.get((x, y), ' ') for x in range(min_x, max_x + 1)]
        # Skip if all walls or empty
        if all(v in ['#', ' ', None] for v in row_values):
            trim_top += 1
        else:
            break
    
    # Check bottom rows
    trim_bottom = 0
    for y in range(max_y, min_y - 1, -1):
        row_values = [grid_dict.get((x, y), ' ') for x in range(min_x, max_x + 1)]
        if all(v in ['#', ' ', None] for v in row_values):
            trim_bottom += 1
        else:
            break
    
    # Check left columns
    trim_left = 0
    for x in range(min_x, max_x + 1):
        col_values = [grid_dict.get((x, y), ' ') for y in range(min_y, max_y + 1)]
        if all(v in ['#', ' ', None] for v in col_values):
            trim_left += 1
        else:
            break
    
    # Check right columns
    trim_right = 0
    for x in range(max_x, min_x - 1, -1):
        col_values = [grid_dict.get((x, y), ' ') for y in range(min_y, max_y + 1)]
        if all(v in ['#', ' ', None] for v in col_values):
            trim_right += 1
        else:
            break
    
    # But keep at least one row of walls around the actual content
    # Don't trim if it would remove actual room walls
    if trim_top > 1:
        trim_top -= 1  # Keep one row of walls
    if trim_bottom > 1:
        trim_bottom -= 1
    if trim_left > 1:
        trim_left -= 1
    if trim_right > 1:
        trim_right -= 1
    
    # Create trimmed grid
    trimmed = {}
    new_min_x = min_x + trim_left
    new_max_x = max_x - trim_right
    new_min_y = min_y + trim_top
    new_max_y = max_y - trim_bottom
    
    for y in range(new_min_y, new_max_y + 1):
        for x in range(new_min_x, new_max_x + 1):
            if (x, y) in grid_dict:
                # Adjust coordinates to start from 0
                new_x = x - new_min_x
                new_y = y - new_min_y
                trimmed[(new_x, new_y)] = grid_dict[(x, y)]
    
    return trimmed


def is_padding_row(row_values):
    """Check if a row is just padding (all walls with no content)"""
    # A padding row is one that's all # or empty spaces
    # But not if it contains doors, NPCs, items, etc.
    meaningful_symbols = ['.', 'D', 'N', 'T', 'P', 'S', 'W', '?', '~']
    return not any(v in meaningful_symbols for v in row_values)
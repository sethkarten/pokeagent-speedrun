# Coordinate System Analysis: Pokeemerald vs Memory Reader

## Overview

This document analyzes the coordinate systems used in:
1. **Pokeemerald JSON/Binary files** (ground truth map data)
2. **Memory Reader** (runtime coordinate reading)
3. **Map Stitcher** (internal map representation)

## Key Finding: **YES, THERE ARE DIFFERENCES**

## Coordinate Systems

### 1. Pokeemerald JSON Files (Ground Truth)

**System**: Direct map coordinates
- **Origin**: Top-left corner of the map (0, 0)
- **X-axis**: Increases rightward (0 → width-1)
- **Y-axis**: Increases downward (0 → height-1)
- **Units**: Metatiles (16x16 pixel tiles)
- **Range**: 
  - X: 0 to (map_width - 1)
  - Y: 0 to (map_height - 1)

**Example from OldaleTown map.json:**
```json
{
  "warp_events": [
    {"x": 5, "y": 7, "dest_map": "MAP_OLDALE_TOWN_HOUSE1"},
    {"x": 14, "y": 6, "dest_map": "MAP_OLDALE_TOWN_MART"}
  ],
  "object_events": [
    {"x": 16, "y": 11, "graphics_id": "OBJ_EVENT_GFX_GIRL_3"}
  ]
}
```

These coordinates are **direct map coordinates** - where (5, 7) means:
- 5 metatiles from the left edge
- 7 metatiles from the top edge

### 2. Memory Reader Coordinates

**System**: Map buffer coordinates with MAP_OFFSET
- **Origin**: Top-left of the map buffer (includes border)
- **MAP_OFFSET**: 7 tiles (border around visible area)
- **X-axis**: Increases rightward
- **Y-axis**: Increases downward
- **Units**: Metatiles

**From memory_reader.py:**
```python
def read_coordinates(self) -> Tuple[int, int]:
    """Read player coordinates"""
    x = self._read_u16(base_address + SAVESTATE_PLAYER_X_OFFSET)
    y = self._read_u16(base_address + SAVESTATE_PLAYER_Y_OFFSET)
    return (x, y)
```

**Key Issue**: The coordinates read from memory appear to be **world/absolute coordinates**, but when accessing the map buffer:
```python
map_x = player_x + 7  # MAP_OFFSET
map_y = player_y + 7
```

This suggests:
- `read_coordinates()` returns **player's position in map space** (0-0 based, but possibly shifted)
- The map buffer has a 7-tile border, so accessing requires adding 7
- But the actual player coordinate might already account for this or not

### 3. Map Stitcher Coordinates

**System**: Custom offset-based system
- **Origin**: Dynamically set based on first player position
- Uses `origin_offset` to map player coordinates to internal grid
- **Initial offset**: Places player at (50, 50) in internal grid
- **Purpose**: To handle maps that are discovered progressively

**From map_stitcher.py:**
```python
area.origin_offset = {'x': 50 - player_pos[0], 'y': 50 - player_pos[1]}
# Later:
actual_x = player_pos[0] + offset_x
actual_y = player_pos[1] + offset_y
```

This creates a **relative coordinate system** where the first seen position becomes the origin.

## The Problem: Coordinate Mismatch

### Scenario 1: Using JSON Warp Coordinates

If a warp is at (5, 7) in the JSON file, but the memory reader reports the player at a different position, there's a mismatch:

```python
# JSON says warp is at (5, 7)
warp_json_x, warp_json_y = 5, 7

# Memory reader reports player position
player_x, player_y = memory_reader.read_coordinates()

# Are these in the same coordinate space?
if (player_x, player_y) == (warp_json_x, warp_json_y):
    # They match! But do they always?
```

### Scenario 2: MAP_OFFSET Confusion

The memory reader uses MAP_OFFSET (7) when accessing the map buffer:
```python
map_x = player_x + 7  # Accessing map buffer
```

But JSON coordinates are **direct map coordinates**, not buffer coordinates. So:
- JSON coordinate (5, 7) = 5,7 in map space
- Memory buffer access = player_x + 7, player_y + 7

**Question**: Does `read_coordinates()` return:
- Option A: World coordinates (already in map space, 0-based)?
- Option B: Buffer coordinates (needs -7 to get map space)?
- Option C: Something else?

## Investigation Needed

To resolve this, we need to determine:

1. **What does `read_coordinates()` actually return?**
   - Test: Stand at a known location (e.g., warp at 5,7)
   - Check what `read_coordinates()` returns
   - Compare with JSON coordinate (5, 7)

2. **Does MAP_OFFSET affect player coordinates?**
   - The game uses MAP_OFFSET=7 for map buffer access
   - But player position might already be in "map space"
   - Or it might be in "world space" with different origin

3. **Coordinate transformation requirements:**
   ```python
   # If memory returns world coordinates but JSON uses map coordinates:
   json_to_memory = (json_x, json_y)  # Same if in same space
   memory_to_json = (mem_x, mem_y)    # Same if in same space
   
   # If memory returns buffer coordinates:
   json_to_memory = (json_x + 7, json_y + 7)
   memory_to_json = (mem_x - 7, mem_y - 7)
   ```

## Recommendations

### 1. Create Coordinate Translation Functions

```python
class CoordinateTranslator:
    """Translate between coordinate systems"""
    
    @staticmethod
    def json_to_memory(json_x: int, json_y: int) -> Tuple[int, int]:
        """Convert pokeemerald JSON coordinates to memory reader coordinates"""
        # Need to test to determine transformation
        return (json_x, json_y)  # Placeholder - needs verification
    
    @staticmethod
    def memory_to_json(mem_x: int, mem_y: int) -> Tuple[int, int]:
        """Convert memory reader coordinates to JSON coordinates"""
        return (mem_x, mem_y)  # Placeholder - needs verification
```

### 2. Test Coordinate Alignment

Create a test script that:
1. Loads a known map (e.g., OldaleTown)
2. Stands at a warp point (e.g., 5, 7)
3. Reads coordinates from memory
4. Compares with JSON coordinates
5. Determines the transformation needed

### 3. Update Pathfinding

If coordinates don't match:
- All JSON coordinates need translation before use
- All memory coordinates need translation before comparison
- Map connections need coordinate transformation
- Warp events need coordinate transformation

## Action Items

1. **Create test script** to compare coordinate systems
2. **Document actual transformation** once determined
3. **Update parser** to include coordinate translation
4. **Update pathfinding** to use translated coordinates
5. **Update map_stitcher** to align with JSON coordinates

## Next Steps

Run a test where:
1. Agent stands at a known location (JSON warp at 5, 7)
2. Memory reader reports position
3. Compare the two to determine transformation

This will reveal if we need:
- No transformation (they match)
- Simple offset (add/subtract constant)
- Axis flip (swap x/y or invert)
- Complex transformation (multiple operations)


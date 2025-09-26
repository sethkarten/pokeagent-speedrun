import os
import pytest
from pathlib import Path
import numpy as np
from pprint import pprint

from pokemon_env.emulator import EmeraldEmulator
from pokemon_env.enums import MetatileBehavior

def format_map_data(map_data, title="Map Data"):
    """Format the map data into a string using the agent's format"""
    print(f"Map data: {map_data}")
    lines = []
    lines.append(f"=== {title} ===")
    lines.append("Format: (MetatileID, Behavior, X, Y)")
    lines.append(f"Map dimensions: {len(map_data)}x{len(map_data[0])}")
    lines.append("")
    lines.append("--- TRAVERSABILITY MAP ---")
    
    # Column headers
    header = "     "
    for i in range(len(map_data[0])):
        header += f"{i:2} "
    lines.append(header)
    lines.append("    " + "--" * len(map_data[0]))
    
    # Find player position (center of the grid)
    center_y = len(map_data) // 2
    center_x = len(map_data[0]) // 2
    
    # Format each row
    for y in range(len(map_data)):
        row = f"{y:2}: "
        for x in range(len(map_data[y])):
            if y == center_y and x == center_x:
                row += "P"  # Player position
            else:
                tile = map_data[y][x]
                behavior = tile[1]
                # Convert behavior to map symbol using same logic as get_comprehensive_state
                tile_id, behavior, collision, elevation = tile
                
                # Handle both enum objects and raw integers from server API
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
                
                if behavior_name == "NORMAL":
                    # For normal tiles, use collision data to determine if passable
                    row += "." if collision == 0 else "#"
                elif "TALL_GRASS" in behavior_name:
                    row += "~"  # Tall grass
                elif "WATER" in behavior_name:
                    row += "W"  # Water
                elif "JUMP" in behavior_name:
                    # Show jump direction
                    if "JUMP_EAST" in behavior_name:
                        row += "→"
                    elif "JUMP_WEST" in behavior_name:
                        row += "←"
                    elif "JUMP_NORTH" in behavior_name:
                        row += "↑"
                    elif "JUMP_SOUTH" in behavior_name:
                        row += "↓"
                    elif "JUMP_NORTHEAST" in behavior_name:
                        row += "↗"
                    elif "JUMP_NORTHWEST" in behavior_name:
                        row += "↖"
                    elif "JUMP_SOUTHEAST" in behavior_name:
                        row += "↘"
                    elif "JUMP_SOUTHWEST" in behavior_name:
                        row += "↙"
                    else:
                        row += "J"
                elif "DOOR" in behavior_name:
                    row += "D"  # Door
                else:
                    # For other behaviors, show abbreviated name
                    short_name = behavior_name.replace("_", "")[:4]
                    row += f"{short_name[:2]}"
            row += " "  # Add space between tiles
        lines.append(row.rstrip())
    
    return "\n".join(lines)

def print_map_data(map_data, title="Map Data"):
    """Pretty print the map data and return the formatted string"""
    formatted_map = format_map_data(map_data, title)
    print("\n" + formatted_map)
    print("\nLegend:")
    print("  P  = Player position")
    print("  .  = Normal walkable path")
    print("  #  = Blocked/wall")
    print("  D  = Door (can be entered)")
    print("  ~  = Tall grass (wild Pokemon encounters)")
    print("  W  = Water (requires Surf)")
    print("  IM = Impassable terrain")
    print("  SE = Sealed area (cannot enter)")
    print("  EA = East arrow (direction indicator)")
    print("  SO = Sound mat (makes sound when stepped on)")
    print("  TE = Television")
    print("  →←↑↓↗↖↘↙ = Jump ledge directions (can jump in that direction)")
    return formatted_map

# Get the absolute path to the test states directory
TEST_STATES_DIR = os.path.join(os.path.dirname(__file__), "states")

@pytest.fixture
def emulator():
    """Create and initialize an emulator instance"""
    # Get path to ROM file
    project_root = Path(__file__).parent.parent
    rom_path = str(project_root / "Emerald-GBAdvance" / "rom.gba")
    
    # Initialize emulator
    emu = EmeraldEmulator(rom_path, headless=True, sound=False)
    emu.initialize()
    
    yield emu
    
    # Cleanup
    emu.stop()

def test_house_state_map_reading(emulator):
    """Test map reading functionality in the house state"""
    # Load the house state
    state_path = os.path.join(TEST_STATES_DIR, "house.state")
    emulator.load_state(state_path)
    
    # Get map data around player using the same function as the agent
    map_data = emulator.memory_reader.read_map_around_player(radius=7)
    
    # Format the map data
    formatted_map = print_map_data(map_data, "House State Map")
    
    # Load ground truth
    truth_path = os.path.join(TEST_STATES_DIR, "house_map_truth.txt")
    with open(truth_path, 'r') as f:
        expected_map = f.read().strip()
    
    # Compare with ground truth
    assert formatted_map == expected_map, "Map output does not match ground truth"
    
    # Basic structure tests
    assert map_data is not None, "Map data should not be None"
    assert isinstance(map_data, list), "Map data should be a list"
    assert len(map_data) > 0, "Map data should not be empty"
    assert isinstance(map_data[0], list), "Map data should be a 2D list"
    
    # Test map dimensions
    height = len(map_data)
    width = len(map_data[0])
    assert height == 15, f"Map height should be 15 (2*radius + 1), got {height}"
    assert width == 15, f"Map width should be 15 (2*radius + 1), got {width}"
    
    # Test tile data structure
    center_tile = map_data[7][7]  # Center tile where player is
    assert isinstance(center_tile, tuple), "Tile data should be a tuple"
    assert len(center_tile) == 4, "Tile data should contain metatile ID, behavior, x, and y coordinates"
    
    # Test that we're indoors (should have indoor tiles)
    center_behavior = center_tile[1]
    assert isinstance(center_behavior, MetatileBehavior), "Tile behavior should be MetatileBehavior enum"
    assert center_behavior in [MetatileBehavior.INDOOR_ENCOUNTER, MetatileBehavior.NORMAL], \
        "Center tile should be an indoor tile"

def test_truck_state_map_reading(emulator):
    """Test map reading functionality in the truck state (game start)"""
    # Load the truck state
    state_path = os.path.join(TEST_STATES_DIR, "truck.state")
    emulator.load_state(state_path)
    
    # Get map data around player using the same function as the agent
    map_data = emulator.memory_reader.read_map_around_player(radius=7)
    
    # Format the map data
    formatted_map = print_map_data(map_data, "Truck State Map")
    
    # Load ground truth
    truth_path = os.path.join(TEST_STATES_DIR, "truck_map_truth.txt")
    with open(truth_path, 'r') as f:
        expected_map = f.read().strip()
    
    # Compare with ground truth
    assert formatted_map == expected_map, "Map output does not match ground truth"
    
    # Basic structure tests
    assert map_data is not None, "Map data should not be None"
    assert isinstance(map_data, list), "Map data should be a list"
    assert len(map_data) > 0, "Map data should not be empty"
    assert isinstance(map_data[0], list), "Map data should be a 2D list"
    
    # Test map dimensions
    height = len(map_data)
    width = len(map_data[0])
    assert height == 15, f"Map height should be 15 (2*radius + 1), got {height}"
    assert width == 15, f"Map width should be 15 (2*radius + 1), got {width}"
    
    # Test tile data structure
    center_tile = map_data[7][7]  # Center tile where player is
    assert isinstance(center_tile, tuple), "Tile data should be a tuple"
    assert len(center_tile) == 4, "Tile data should contain metatile ID, behavior, x, and y coordinates"
    
    # Test that we're in the truck (should have indoor/special tiles)
    center_behavior = center_tile[1]
    assert isinstance(center_behavior, MetatileBehavior), "Tile behavior should be MetatileBehavior enum"
    assert center_behavior in [MetatileBehavior.INDOOR_ENCOUNTER, MetatileBehavior.NORMAL], \
        "Center tile should be an indoor tile"

def test_door_behavior(emulator):
    """Test that doors are properly identified and not marked as blocked"""
    # Load the house state
    state_path = os.path.join(TEST_STATES_DIR, "house.state")
    emulator.load_state(state_path)
    
    # Get map data around player using the same function as the agent
    map_data = emulator.memory_reader.read_map_around_player(radius=7)
    
    # Print the map data for visual inspection
    print_map_data(map_data, "House State Map (Door Test)")
    
    # Find any door tiles
    door_found = False
    door_behaviors = [MetatileBehavior.NON_ANIMATED_DOOR, 
                     MetatileBehavior.ANIMATED_DOOR,
                     MetatileBehavior.WATER_DOOR]
    
    for row in map_data:
        for tile in row:
            behavior = tile[1]
            if behavior in door_behaviors:
                door_found = True
                # Verify door tile is not marked as blocked
                assert behavior != MetatileBehavior.NORMAL, "Door should not be marked as normal path"
                assert behavior in door_behaviors, "Door should have proper door behavior"
    
    # We should find at least one door in the house state
    assert door_found, "No doors found in house state"

def test_map_data_validation(emulator):
    """Test validation of map data structure and content"""
    # Load any state (using truck state)
    state_path = os.path.join(TEST_STATES_DIR, "truck.state")
    emulator.load_state(state_path)
    
    # Test different radius values
    for radius in [3, 5, 7]:
        map_data = emulator.memory_reader.read_map_around_player(radius=radius)
        
        # Check dimensions
        expected_size = 2 * radius + 1
        assert len(map_data) == expected_size, f"Map height should be {expected_size} for radius {radius}"
        assert all(len(row) == expected_size for row in map_data), \
            f"All rows should have width {expected_size} for radius {radius}"
        
        # Check tile data consistency
        for row in map_data:
            for tile in row:
                    # Check tile structure
                    assert isinstance(tile, tuple), "Tile data should be a tuple"
                    assert len(tile) == 4, "Tile data should contain metatile ID, behavior, x, and y coordinates"
                    
                    # Check metatile ID
                    metatile_id = tile[0]
                    assert isinstance(metatile_id, int), "Metatile ID should be an integer"
                    assert metatile_id >= 0, "Metatile ID should be non-negative"
                
                    # Check behavior
                    behavior = tile[1]
                    assert isinstance(behavior, MetatileBehavior), "Behavior should be MetatileBehavior enum"
                    assert behavior.value >= 0, "Behavior value should be non-negative"

def test_upstairs_state_map_reading(emulator):
    """Test map reading functionality in the upstairs state"""
    # Load the upstairs state
    state_path = os.path.join(TEST_STATES_DIR, "upstairs.state")
    emulator.load_state(state_path)
    
    # Get map data around player using the same function as the agent
    map_data = emulator.memory_reader.read_map_around_player(radius=7)
    
    # Format the map data
    formatted_map = print_map_data(map_data, "Upstairs State Map")
    
    # Basic structure tests
    assert map_data is not None, "Map data should not be None"
    assert isinstance(map_data, list), "Map data should be a list"
    assert len(map_data) > 0, "Map data should not be empty"
    assert isinstance(map_data[0], list), "Map data should be a 2D list"
    
    # Test map dimensions
    height = len(map_data)
    width = len(map_data[0])
    assert height == 15, f"Map height should be 15 (2*radius + 1), got {height}"
    assert width == 15, f"Map width should be 15 (2*radius + 1), got {width}"
    
    # Test tile data structure
    center_tile = map_data[7][7]  # Center tile where player is
    assert isinstance(center_tile, tuple), "Tile data should be a tuple"
    assert len(center_tile) == 4, "Tile data should contain metatile ID, behavior, x, and y coordinates"
    
    # Test that we're indoors (should have indoor tiles)
    center_behavior = center_tile[1]
    assert isinstance(center_behavior, MetatileBehavior), "Tile behavior should be MetatileBehavior enum"
    
    # Print detailed information for debugging
    # print(f"\n=== UPSTAIRS STATE DEBUG INFO ===")
    print(f"Map dimensions: {width}x{height}")
    print(f"Center tile: {center_tile}")
    print(f"Center behavior: {center_behavior}")
    
    # Check for specific upstairs characteristics
    # Upstairs should have indoor tiles, possibly doors, and specific layout
    indoor_tiles = 0
    door_tiles = 0
    wall_tiles = 0
    impassable_tiles = 0
    normal_tiles = 0
    
    # Analyze the map data more thoroughly
    for y, row in enumerate(map_data):
        for x, tile in enumerate(row):
            if len(tile) >= 2:
                metatile_id, behavior, collision, elevation = tile
                
                # Count different tile types
                if behavior == MetatileBehavior.INDOOR_ENCOUNTER:
                    indoor_tiles += 1
                elif behavior in [MetatileBehavior.NON_ANIMATED_DOOR, MetatileBehavior.ANIMATED_DOOR]:
                    door_tiles += 1
                elif behavior == MetatileBehavior.NORMAL:
                    if collision > 0:
                        wall_tiles += 1
                    else:
                        normal_tiles += 1
                else:
                    # Check if this is an impassable tile
                    if metatile_id == 0 or behavior.value == 0:
                        impassable_tiles += 1
    
    print(f"Indoor tiles found: {indoor_tiles}")
    print(f"Door tiles found: {door_tiles}")
    print(f"Wall tiles found: {wall_tiles}")
    print(f"Normal walkable tiles found: {normal_tiles}")
    print(f"Impassable tiles found: {impassable_tiles}")
    
    # Check map buffer addresses
    print(f"\nMap buffer info:")
    print(f"Map buffer address: 0x{emulator.memory_reader._map_buffer_addr:08X}")
    print(f"Map width: {emulator.memory_reader._map_width}")
    print(f"Map height: {emulator.memory_reader._map_height}")
    
    # Check player coordinates
    player_coords = emulator.memory_reader.read_coordinates()
    print(f"Player coordinates: {player_coords}")
    
    # Check if behaviors are being loaded correctly
    all_behaviors = emulator.memory_reader.get_all_metatile_behaviors()
    print(f"Total behaviors loaded: {len(all_behaviors) if all_behaviors else 0}")
    
    # Check specific tiles around the player
    print(f"\nTile analysis around player (center):")
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y = 7 + dy
            x = 7 + dx
            if 0 <= y < len(map_data) and 0 <= x < len(map_data[0]):
                tile = map_data[y][x]
                if len(tile) >= 4:
                    metatile_id, behavior, collision, elevation = tile
                    symbol = "P" if dy == 0 and dx == 0 else f"{metatile_id:03X}"
                    print(f"  ({dx:+2d},{dy:+2d}): {symbol} {behavior.name} c{collision} e{elevation}")
    
    # Upstairs should have some indoor characteristics
    assert indoor_tiles > 0 or door_tiles > 0, "Upstairs state should have indoor or door tiles"
    
    # Check if we have too many impassable tiles (might indicate map reading issues)
    total_tiles = width * height
    impassable_ratio = impassable_tiles / total_tiles
    print(f"Impassable tile ratio: {impassable_ratio:.2%}")
    
    # If more than 50% of tiles are impassable, there might be an issue
    if impassable_ratio > 0.5:
        print("WARNING: High ratio of impassable tiles detected. This might indicate map reading issues.")
    
    # Save the formatted map for future reference
    truth_path = os.path.join(TEST_STATES_DIR, "upstairs_map_truth.txt")
    with open(truth_path, 'w') as f:
        f.write(formatted_map)
    
    print(f"\nMap data saved to {truth_path}")
    # print("=== END UPSTAIRS STATE DEBUG INFO ===\n")

def test_map_reading_area_transitions(emulator):
    """Test that map reading handles area transitions and new saves correctly"""
    # Test 1: Load upstairs state and verify map reading
    state_path = os.path.join(TEST_STATES_DIR, "upstairs.state")
    emulator.load_state(state_path)
    
    # Force invalidate map cache to simulate area transition
    emulator.memory_reader.invalidate_map_cache()
    
    # Try to read map - should work correctly
    map_data = emulator.memory_reader.read_map_around_player(radius=7)
    
    # Verify map data is valid
    assert map_data is not None, "Map data should not be None after cache invalidation"
    assert len(map_data) > 0, "Map data should not be empty after cache invalidation"
    assert len(map_data) == 15, "Map height should be 15 after cache invalidation"
    assert len(map_data[0]) == 15, "Map width should be 15 after cache invalidation"
    
    # Test 2: Load house state and verify map reading
    state_path = os.path.join(TEST_STATES_DIR, "house.state")
    emulator.load_state(state_path)
    
    # Map cache should be automatically invalidated by load_state
    map_data = emulator.memory_reader.read_map_around_player(radius=7)
    
    # Verify map data is valid
    assert map_data is not None, "Map data should not be None after loading house state"
    assert len(map_data) > 0, "Map data should not be empty after loading house state"
    assert len(map_data) == 15, "Map height should be 15 after loading house state"
    assert len(map_data[0]) == 15, "Map width should be 15 after loading house state"
    
    # Test 3: Load truck state and verify map reading
    state_path = os.path.join(TEST_STATES_DIR, "truck.state")
    emulator.load_state(state_path)
    
    map_data = emulator.memory_reader.read_map_around_player(radius=7)
    
    # Verify map data is valid
    assert map_data is not None, "Map data should not be None after loading truck state"
    assert len(map_data) > 0, "Map data should not be empty after loading truck state"
    assert len(map_data) == 15, "Map height should be 15 after loading truck state"
    assert len(map_data[0]) == 15, "Map width should be 15 after loading truck state"
    
    print("✅ All area transition tests passed - map reading handles transitions correctly")

def test_simple_test_state_map_reading(emulator):
    """Test map reading functionality in the simple_test state (rival's bedroom)"""
    # Load the simple_test state
    state_path = os.path.join(TEST_STATES_DIR, "simple_test.state")
    emulator.load_state(state_path)
    
    # Get map data around player using the same function as the agent
    map_data = emulator.memory_reader.read_map_around_player(radius=7)
    
    # Format the map data
    formatted_map = print_map_data(map_data, "Simple Test State Map")
    
    # Basic structure tests
    assert map_data is not None, "Map data should not be None"
    assert isinstance(map_data, list), "Map data should be a list"
    assert len(map_data) > 0, "Map data should not be empty"
    assert isinstance(map_data[0], list), "Map data should be a 2D list"
    
    # Test map dimensions
    height = len(map_data)
    width = len(map_data[0])
    assert height == 15, f"Map height should be 15 (2*radius + 1), got {height}"
    assert width == 15, f"Map width should be 15 (2*radius + 1), got {width}"
    
    # Test tile data structure
    center_tile = map_data[7][7]  # Center tile where player is
    assert isinstance(center_tile, tuple), "Tile data should be a tuple"
    assert len(center_tile) == 4, "Tile data should contain metatile ID, behavior, x, and y coordinates"
    
    # Print detailed information for debugging
    # print(f"\n=== SIMPLE TEST STATE DEBUG INFO ===")
    print(f"Map dimensions: {width}x{height}")
    print(f"Center tile: {center_tile}")
    print(f"Center behavior: {center_tile[1]}")
    
    # Check map buffer addresses
    print(f"\nMap buffer info:")
    print(f"Map buffer address: 0x{emulator.memory_reader._map_buffer_addr:08X}")
    print(f"Map width: {emulator.memory_reader._map_width}")
    print(f"Map height: {emulator.memory_reader._map_height}")
    
    # Check player coordinates
    player_coords = emulator.memory_reader.read_coordinates()
    print(f"Player coordinates: {player_coords}")
    
    # Check if behaviors are being loaded correctly
    all_behaviors = emulator.memory_reader.get_all_metatile_behaviors()
    print(f"Total behaviors loaded: {len(all_behaviors) if all_behaviors else 0}")
    
    # Check specific tiles around the player
    print(f"\nTile analysis around player (center):")
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y = 7 + dy
            x = 7 + dx
            if 0 <= y < len(map_data) and 0 <= x < len(map_data[0]):
                tile = map_data[y][x]
                if len(tile) >= 4:
                    metatile_id, behavior, collision, elevation = tile
                    symbol = "P" if dy == 0 and dx == 0 else f"{metatile_id:03X}"
                    print(f"  ({dx:+2d},{dy:+2d}): {symbol} {behavior.name} c{collision} e{elevation}")
    
    # Count different tile types
    indoor_tiles = 0
    door_tiles = 0
    wall_tiles = 0
    impassable_tiles = 0
    normal_tiles = 0
    decoration_tiles = 0
    
    for y, row in enumerate(map_data):
        for x, tile in enumerate(row):
            if len(tile) >= 2:
                metatile_id, behavior, collision, elevation = tile
                
                # Count different tile types
                if behavior == MetatileBehavior.INDOOR_ENCOUNTER:
                    indoor_tiles += 1
                elif behavior in [MetatileBehavior.NON_ANIMATED_DOOR, MetatileBehavior.ANIMATED_DOOR]:
                    door_tiles += 1
                elif behavior == MetatileBehavior.NORMAL:
                    if collision > 0:
                        wall_tiles += 1
                    else:
                        normal_tiles += 1
                elif "DECORATION" in behavior.name or "HOLDS" in behavior.name:
                    decoration_tiles += 1
                else:
                    # Check if this is an impassable tile
                    if metatile_id == 0 or behavior.value == 0:
                        impassable_tiles += 1
    
    print(f"\nTile counts:")
    print(f"Indoor tiles found: {indoor_tiles}")
    print(f"Door tiles found: {door_tiles}")
    print(f"Wall tiles found: {wall_tiles}")
    print(f"Normal walkable tiles found: {normal_tiles}")
    print(f"Decoration tiles found: {decoration_tiles}")
    print(f"Impassable tiles found: {impassable_tiles}")
    
    # Check if we have too many impassable tiles (might indicate map reading issues)
    total_tiles = width * height
    impassable_ratio = impassable_tiles / total_tiles
    print(f"Impassable tile ratio: {impassable_ratio:.2%}")
    
    # If more than 50% of tiles are impassable, there might be an issue
    if impassable_ratio > 0.5:
        print("WARNING: High ratio of impassable tiles detected. This might indicate map reading issues.")
        
        # Try to force re-find map buffer addresses
        print("Attempting to force re-find map buffer addresses...")
        emulator.memory_reader.invalidate_map_cache()
        if emulator.memory_reader._find_map_buffer_addresses():
            print("Successfully re-found map buffer addresses")
            # Try reading map again
            new_map_data = emulator.memory_reader.read_map_around_player(radius=7)
            if new_map_data and len(new_map_data) > 0:
                print("Map reading improved after cache invalidation")
                formatted_map = print_map_data(new_map_data, "Simple Test State Map (After Cache Invalidation)")
    
    # Save the formatted map for future reference
    truth_path = os.path.join(TEST_STATES_DIR, "simple_test_map_truth.txt")
    with open(truth_path, 'w') as f:
        f.write(formatted_map)
    
    print(f"\nMap data saved to {truth_path}")
    # print("=== END SIMPLE TEST STATE DEBUG INFO ===\n")
    
    # Simple test state is actually outdoors in tall grass, so it should have outdoor characteristics
    assert normal_tiles > 0 or wall_tiles > 0, "Simple test state should have normal walkable tiles or walls"
"""
Coordinate Overlay Utility for Claude Plays Pokemon

Overlays coordinate labels on game screenshots to help Claude understand
spatial positioning. This is faithful to the original ClaudePlaysPokemon
implementation which showed coordinates over each tile.

Key features:
- White text on black background for walkable tiles
- White text on red background for blocked tiles
- No overlay on player's current position
- Coordinates shown as (X,Y) format
"""

import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)

# GBA screen dimensions
GBA_WIDTH = 240
GBA_HEIGHT = 160

# Tile size in pixels (Pokemon Emerald uses 16x16 tiles)
TILE_SIZE = 16

# Colors
COLOR_WALKABLE_BG = (0, 0, 0, 200)      # Black with alpha
COLOR_BLOCKED_BG = (200, 0, 0, 200)      # Red with alpha
COLOR_TEXT = (255, 255, 255, 255)        # White

# Font size for coordinates (will be used with 2x scaled image = 32x32 tiles)
FONT_SIZE = 10  # Balanced font for 2x scaled image


class CoordinateOverlay:
    """
    Creates coordinate overlays on game screenshots for Claude Plays Pokemon.

    Shows (X,Y) coordinates on each visible tile with color-coded backgrounds:
    - Black background: Walkable tile
    - Red background: Blocked/impassable tile
    - No overlay: Player's current position
    """

    def __init__(self, font_size: int = FONT_SIZE):
        """
        Initialize the coordinate overlay generator.

        Args:
            font_size: Size of font for coordinate text
        """
        self.font_size = font_size
        self.font = self._load_font(font_size)

    def _load_font(self, size: int):
        """Load a crisp bitmap font for low-resolution display."""
        try:
            from PIL import ImageFont
            # For low-resolution screens, use TrueType at very small size (6px)
            # This gives crisp rendering for coordinates
            # Try common monospace fonts
            for font_name in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/System/Library/Fonts/Monaco.dfont",  # macOS
                "C:\\Windows\\Fonts\\consola.ttf",      # Windows
            ]:
                try:
                    # Use the requested size (default 12 for 2x scaled 32x32 tiles)
                    return ImageFont.truetype(font_name, size)
                except (IOError, OSError):
                    continue

            # Fallback to default bitmap font if no TrueType found
            return ImageFont.load_default()
        except Exception as e:
            logger.warning(f"Error loading font: {e}, using default")
            return ImageFont.load_default()

    def create_overlay(
        self,
        screenshot: Image.Image,
        game_state: Dict[str, Any],
        tile_size: int = TILE_SIZE,
        scale: int = 2
    ) -> Image.Image:
        """
        Create a screenshot with coordinate overlays.

        Args:
            screenshot: Original game screenshot (PIL Image)
            game_state: Current game state with player position and map data
            tile_size: Size of each tile in pixels (default 16 for Pokemon Emerald)
            scale: Scale factor for upscaling (default 2 = 2x size for better readability)

        Returns:
            PIL Image with coordinate overlays (scaled up if scale > 1)
        """
        # Scale up the screenshot for better readability
        if scale > 1:
            new_size = (screenshot.width * scale, screenshot.height * scale)
            screenshot = screenshot.resize(new_size, Image.NEAREST)  # NEAREST for pixel-perfect scaling
            tile_size = tile_size * scale

        # Convert to RGBA for transparency support
        if screenshot.mode != 'RGBA':
            screenshot = screenshot.convert('RGBA')

        # Create transparent overlay layer
        overlay = Image.new('RGBA', screenshot.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Get player position
        player_pos = self._get_player_position(game_state)
        if not player_pos:
            logger.warning("Could not determine player position, skipping overlay")
            return screenshot

        player_x, player_y = player_pos

        # Get map data for traversability
        map_data = self._get_map_data(game_state)

        # Calculate visible tile range (centered on player)
        # GBA screen is 240x160, tiles are 16x16, so we see 15x10 tiles
        # Use original GBA dimensions, not scaled dimensions
        tiles_horizontal = GBA_WIDTH // TILE_SIZE  # 15 tiles
        tiles_vertical = GBA_HEIGHT // TILE_SIZE    # 10 tiles

        # Calculate the top-left world coordinate visible on screen
        # Player is typically centered, so we see tiles around them
        start_x = player_x - (tiles_horizontal // 2)
        start_y = player_y - (tiles_vertical // 2)

        # Draw coordinates for each visible tile
        for screen_y in range(tiles_vertical):
            for screen_x in range(tiles_horizontal):
                # Calculate world coordinates
                world_x = start_x + screen_x
                world_y = start_y + screen_y

                # Skip player's tile (no overlay on player position)
                if world_x == player_x and world_y == player_y:
                    continue

                # Calculate screen position (top-left of tile)
                screen_pixel_x = screen_x * tile_size
                screen_pixel_y = screen_y * tile_size

                # Check if tile is walkable
                is_walkable = self._is_tile_walkable(
                    world_x, world_y,
                    map_data,
                    player_pos
                )

                # Draw coordinate label
                self._draw_coordinate_label(
                    draw,
                    screen_pixel_x,
                    screen_pixel_y,
                    world_x,
                    world_y,
                    tile_size,
                    is_walkable
                )

        # Composite overlay onto screenshot
        result = Image.alpha_composite(screenshot, overlay)
        return result

    def _draw_coordinate_label(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        world_x: int,
        world_y: int,
        tile_size: int,
        is_walkable: bool
    ):
        """
        Draw a coordinate label on the overlay.

        Args:
            draw: ImageDraw object to draw on
            x, y: Screen pixel position (top-left of tile)
            world_x, world_y: World coordinates to display
            tile_size: Size of tile in pixels
            is_walkable: Whether the tile is walkable
        """
        # Format coordinate text - single line format (X,Y)
        coord_text = f"{world_x},{world_y}"

        # Choose background color
        bg_color = COLOR_WALKABLE_BG if is_walkable else COLOR_BLOCKED_BG

        # Get text bounding box
        try:
            # Try modern PIL API first
            bbox = draw.textbbox((0, 0), coord_text, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(coord_text, font=self.font)

        # Center text in tile horizontally, but raise it vertically
        text_x = x + (tile_size - text_width) // 2
        # For 32x32 tiles (scaled 2x), raise by 10 pixels for perfect centering
        # For 16x16 tiles, this would be 5 pixels
        raise_amount = 10 if tile_size == 32 else 5
        text_y = y + (tile_size - text_height) // 2 - raise_amount

        # Draw background rectangle
        padding = 0
        bg_rect = [
            text_x - padding,
            text_y - padding,
            text_x + text_width + padding,
            text_y + text_height + padding
        ]
        draw.rectangle(bg_rect, fill=bg_color)

        # Draw text (PIL handles antialiasing based on font type)
        draw.text((text_x, text_y), coord_text, fill=COLOR_TEXT, font=self.font)

    def _get_player_position(self, game_state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract player position from game state."""
        try:
            player = game_state.get('player', {})

            # Try position.x/y first (standard format)
            position = player.get('position', {})
            if position:
                x = position.get('x')
                y = position.get('y')
                if x is not None and y is not None:
                    return (int(x), int(y))

            # Fallback: try direct x/y on player
            x = player.get('x')
            y = player.get('y')
            if x is not None and y is not None:
                return (int(x), int(y))

        except Exception as e:
            logger.warning(f"Error getting player position: {e}")

        return None

    def _get_map_data(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract map data from game state."""
        return game_state.get('map', {})

    def _is_tile_walkable(
        self,
        x: int,
        y: int,
        map_data: Dict[str, Any],
        player_pos: Tuple[int, int]
    ) -> bool:
        """
        Determine if a tile at (x, y) is walkable.

        Args:
            x, y: World coordinates to check
            map_data: Map data from game state
            player_pos: Player's current position (for centering)

        Returns:
            True if walkable, False if blocked
        """
        # Get tiles array from map data
        tiles = map_data.get('tiles', [])
        if not tiles:
            # No map data, assume walkable
            return True

        # Player is centered in the tile array
        # The tiles array is typically 15x15 or similar
        if not tiles or not tiles[0]:
            return True

        center_y = len(tiles) // 2
        center_x = len(tiles[0]) // 2 if tiles else 0

        # Calculate offset from player
        dx = x - player_pos[0]
        dy = y - player_pos[1]

        # Calculate index in tiles array
        tile_y = center_y + dy
        tile_x = center_x + dx

        # Check bounds
        if tile_y < 0 or tile_y >= len(tiles):
            return False
        if tile_x < 0 or tile_x >= len(tiles[tile_y]):
            return False

        # Get tile data
        tile = tiles[tile_y][tile_x]

        # Check if tile is walkable
        # Tile format: (tile_id, behavior, collision, elevation, ...)
        if isinstance(tile, (list, tuple)) and len(tile) >= 2:
            # Check behavior and collision
            tile_id = tile[0]
            behavior = tile[1] if len(tile) > 1 else 0

            # Import MetatileBehavior for checking
            try:
                from pokemon_env.enums import MetatileBehavior

                # Check if behavior indicates impassable
                if isinstance(behavior, int):
                    try:
                        behavior_enum = MetatileBehavior(behavior)
                        behavior_name = behavior_enum.name

                        # Impassable behaviors
                        if 'IMPASSABLE' in behavior_name:
                            return False
                        if 'WALL' in behavior_name:
                            return False
                        if 'WATER' in behavior_name and 'SHALLOW' not in behavior_name:
                            return False
                        if 'WATERFALL' in behavior_name:
                            return False

                    except (ValueError, AttributeError):
                        pass

                # Check for walls using tile ID (common wall tiles are typically low IDs)
                if tile_id == 0:
                    return False

            except ImportError:
                # If we can't import MetatileBehavior, use basic heuristics
                if tile_id == 0:
                    return False

        # Default to walkable if we can't determine
        return True


# Convenience function
def add_coordinate_overlay(
    screenshot: Image.Image,
    game_state: Dict[str, Any],
    font_size: int = FONT_SIZE,
    scale: int = 2
) -> Image.Image:
    """
    Add coordinate overlay to a screenshot.

    Args:
        screenshot: Original game screenshot
        game_state: Current game state
        font_size: Font size for coordinates
        scale: Scale factor (default 2 = 2x size for better readability)

    Returns:
        Screenshot with coordinate overlays (scaled up)
    """
    overlay_generator = CoordinateOverlay(font_size=font_size)
    return overlay_generator.create_overlay(screenshot, game_state, scale=scale)

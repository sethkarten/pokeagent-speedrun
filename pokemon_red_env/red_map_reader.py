"""Pokemon Red map reader — hybrid RAM + processed_map approach.

Reads NPC positions from live RAM (wSpriteStateData1 at 0xC100) and uses
pre-computed processed_map/{map_name}.json files for tile classification.
Viewport is clamped to map bounds (Emerald-style) — never padded with walls.

Returns (tile_id, type_str, collision_int, elevation) tuples compatible with
map_formatter.is_tile_walkable() and format_map_grid().
"""

import json
import logging
import os
import re
import sys

# Allow import of sibling utils whether used as package or run from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from red_metatile_behavior import RedMetatileBehavior  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid symbol walkability (matching map_formatter.py conventions)
# ---------------------------------------------------------------------------
_WALKABLE_SYMBOLS = {".", "~", "D", "S", "↓", "←", "→", "↑", "&", "?"}
# "?" = hidden item (walkable); "O" = Poké Ball (non-walkable overlay, NOT in static grid)

# Compact single-char aliases for multi-char grid symbols (for format_map_for_llm)
_COMPACT_SYMBOL = {
    "PC": "P",  # Computer (note: player 'P' overrides at player position)
    "T":  "T",  # Television (already 1 char)
    "B":  "B",  # Bookshelf
    "^":  "^",  # Picture/Painting
    "U":  "U",  # Trash can
    "=":  "=",  # Bed/bench
}

# Sprite facing direction byte → string
_FACING_MAP = {0: "down", 4: "up", 8: "left", 12: "right"}

# Regex for parsing object_event sprite names from .asm files (fallback)
_OBJECT_EVENT_RE = re.compile(r"object_event\s+\d+,\s*\d+,\s*([A-Z0-9_]+)")


class RedMapReader:
    """Provides map collision + NPC data for Pokemon Red."""

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self._map_names: dict = {}   # str(map_id) → map_name
        self._map_cache: dict = {}   # map_name → loaded JSON dict
        self._sprite_name_cache: dict = {}  # map_name → list[str]
        self._load_map_names()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_map_names(self):
        """Load map_names.json mapping map_id string → map_name."""
        path = os.path.join(self.DATA_DIR, "map_names.json")
        try:
            with open(path) as f:
                self._map_names = json.load(f)
            logger.debug(f"Loaded {len(self._map_names)} map names from {path}")
        except Exception as e:
            logger.warning(f"Could not load map_names.json: {e}")

    @staticmethod
    def _resolve_path_ci(directory: str, filename: str) -> str:
        """Return the real path for *filename* inside *directory*, matching
        case-insensitively on Linux.  Returns the exact-case path whether or
        not it exists (caller checks existence)."""
        exact = os.path.join(directory, filename)
        if os.path.exists(exact):
            return exact
        lower = filename.lower()
        try:
            for entry in os.listdir(directory):
                if entry.lower() == lower:
                    return os.path.join(directory, entry)
        except OSError:
            pass
        return exact  # not found; return original so caller gets a clear miss

    def _load_map_data(self, map_name: str) -> dict:
        """Load and cache processed_map/{map_name}.json.

        Returns {} if the file does not exist.
        """
        if map_name in self._map_cache:
            return self._map_cache[map_name]

        pm_dir = os.path.join(self.DATA_DIR, "processed_map")
        path = self._resolve_path_ci(pm_dir, f"{map_name}.json")
        if not os.path.exists(path):
            logger.debug(f"No processed_map file for '{map_name}'")
            self._map_cache[map_name] = {}
            return {}

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._map_cache[map_name] = data
            grid = data.get("grid", [])
            logger.debug(f"Loaded map data for '{map_name}': {len(grid)} rows")
            return data
        except Exception as e:
            logger.warning(f"Error loading processed_map/{map_name}.json: {e}")
            self._map_cache[map_name] = {}
            return {}

    def _load_sprite_names(self, map_name: str) -> list:
        """Get sprite names for this map.

        Prefers npc_data from the preprocessed JSON. Falls back to parsing
        the .asm file if npc_data is unavailable.
        """
        if map_name in self._sprite_name_cache:
            return self._sprite_name_cache[map_name]

        # Try npc_data from JSON first
        data = self._load_map_data(map_name)
        npc_data = data.get("npc_data", [])
        if npc_data:
            names = [npc["sprite"] for npc in npc_data]
            self._sprite_name_cache[map_name] = names
            return names

        # Fallback: parse .asm file
        asm_dir = os.path.join(self.DATA_DIR, "pokered", "data", "maps", "objects")
        asm_path = self._resolve_path_ci(asm_dir, f"{map_name}.asm")
        names = []
        if os.path.exists(asm_path):
            try:
                with open(asm_path, encoding="utf-8") as f:
                    for line in f:
                        match = _OBJECT_EVENT_RE.search(line)
                        if match:
                            names.append(match.group(1))
            except Exception as e:
                logger.debug(f"Could not parse sprite names from {asm_path}: {e}")

        self._sprite_name_cache[map_name] = names
        return names

    # ------------------------------------------------------------------
    # Grid symbol classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_grid_symbol(symbol: str) -> tuple:
        """Grid symbol → (type_str, collision_int).

        The type_str IS the grid symbol itself (e.g. '.', '#', 'PC', '~').
        collision_int: 0 = walkable, 1 = blocked.
        """
        collision_int = 0 if symbol in _WALKABLE_SYMBOLS else 1
        return (symbol, collision_int)

    # ------------------------------------------------------------------
    # RAM helpers
    # ------------------------------------------------------------------

    def _read_u8(self, addr: int) -> int:
        try:
            return int(self.pyboy.memory[addr])
        except Exception:
            return 0

    def read_map_id(self) -> int:
        """Current map ID from wCurMap (0xD35E)."""
        return self._read_u8(0xD35E)

    def read_map_dimensions(self) -> tuple:
        """Return (width_blocks, height_blocks) from wCurMapWidth/Height."""
        return self._read_u8(0xD369), self._read_u8(0xD368)

    def read_player_coords(self) -> tuple:
        """Return (player_x, player_y) block coords from wXCoord/wYCoord."""
        return self._read_u8(0xD362), self._read_u8(0xD361)

    def read_map_name(self) -> str:
        """Return the current map name string."""
        map_id = self.read_map_id()
        return self._map_names.get(str(map_id), f"UNKNOWN_{map_id}")

    # ------------------------------------------------------------------
    # NPC / sprite reading from RAM
    # ------------------------------------------------------------------

    def read_sprites(self) -> list:
        """Read NPC positions from wSpriteStateData1 (0xC100).

        Uses screen pixel deltas relative to player sprite (slot 0) to
        compute absolute map coordinates.  Proven approach from
        pyboy_runner.py:384-408.

        Returns list of dicts:
          {
            'slot': int,           # sprite slot 1-15
            'picture_id': int,     # sprite picture ID from RAM
            'map_x': int,          # absolute coll_map X
            'map_y': int,          # absolute coll_map Y
            'facing': str,         # "down"/"up"/"left"/"right"
            'sprite_name': str,    # from .asm or "NPC_{slot}"
          }
        """
        player_x, player_y = self.read_player_coords()
        map_name = self.read_map_name()
        sprite_names = self._load_sprite_names(map_name)

        # Player sprite screen position (slot 0)
        player_screen_y = self._read_u8(0xC100 + 4)
        player_screen_x = self._read_u8(0xC100 + 6)

        sprites = []
        for i in range(1, 16):
            base = 0xC100 + i * 0x10

            # Skip inactive / off-screen sprites
            if self._read_u8(base + 2) == 0xFF:
                continue

            picture_id = self._read_u8(base)
            if picture_id == 0:
                continue

            # Screen pixel positions
            sprite_y = self._read_u8(base + 4)
            sprite_x = self._read_u8(base + 6)

            # Convert screen deltas to tile offsets (16 px per tile)
            y_delta = ((sprite_y + 4) % 256 - (player_screen_y + 4)) // 16
            x_delta = (sprite_x - player_screen_x) // 16

            map_x = player_x + x_delta
            map_y = player_y + y_delta

            facing_byte = self._read_u8(base + 9)
            facing = _FACING_MAP.get(facing_byte, "down")

            name = sprite_names[i - 1] if i - 1 < len(sprite_names) else f"NPC_{i}"

            sprites.append({
                'slot': i,
                'picture_id': picture_id,
                'map_x': map_x,
                'map_y': map_y,
                'facing': facing,
                'sprite_name': name,
            })

        return sprites

    # ------------------------------------------------------------------
    # Viewport clamping helper
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_viewport(player_coord, radius, map_size):
        """Return (view_start, view_end) clamped to [0, map_size).

        Shifts viewport inward at map edges (Emerald-style) rather than
        padding with walls.  For maps smaller than the viewport, returns
        the full map range.
        """
        target = 2 * radius + 1
        if map_size <= target:
            return 0, map_size
        start = max(0, min(player_coord - radius, map_size - target))
        return start, start + target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_map_around_player(self, radius: int = 7) -> list:
        """Return 2D list of (0, type_str, collision_int, 0) tuples.

        Grid is clamped to map bounds (Emerald-style).  NPC positions
        from RAM are overlaid on the tile grid.

        Format is compatible with map_formatter.is_tile_walkable() which
        checks tile[2] == 0.
        """
        map_name = self.read_map_name()
        data = self._load_map_data(map_name)
        grid = data.get("grid", [])

        if not grid:
            return []

        player_x, player_y = self.read_player_coords()
        map_h = len(grid)
        map_w = len(grid[0]) if map_h > 0 else 0

        vx_start, vx_end = self._clamp_viewport(player_x, radius, map_w)
        vy_start, vy_end = self._clamp_viewport(player_y, radius, map_h)

        # Build NPC position set from RAM; distinguish Poké Ball sprites from regular NPCs
        npc_positions = set()
        pokeball_positions: set = set()
        try:
            npc_data_json = data.get("npc_data", [])
            pokeball_map_positions = {
                (n["x"], n["y"]) for n in npc_data_json
                if "POKE_BALL" in n.get("sprite", "").upper()
            }
            live_sprites = self.read_sprites()
            live_sprite_positions = {(s['map_x'], s['map_y']) for s in live_sprites}
            npc_positions = live_sprite_positions
            # Active Poké Balls = those whose sprite slot is still in RAM
            pokeball_positions = pokeball_map_positions & live_sprite_positions
        except Exception as e:
            logger.debug(f"Could not read sprites: {e}")

        rows = []
        for sy in range(vy_start, vy_end):
            row = []
            for sx in range(vx_start, vx_end):
                symbol = grid[sy][sx]
                type_str, collision_int = self._classify_grid_symbol(symbol)

                # Overlay sprites from RAM (Poké Balls as 'POKE_BALL', NPCs as 'NPC')
                if (sx, sy) in pokeball_positions:
                    type_str = 'POKE_BALL'
                    collision_int = 1
                elif (sx, sy) in npc_positions:
                    type_str = 'NPC'
                    collision_int = 1

                row.append((0, type_str, collision_int, 0))
            rows.append(row)
        return rows

    def format_map_for_llm(self, radius: int = 4) -> str:
        """Return compact ASCII map of the area around the player (for /stream).

        Symbols match map_formatter.py conventions:
          I  player      .  walkable    #  wall / blocked
          ~  tall grass  W  water       D  door / warp
          N  NPC         O  Poké Ball   C  counter
          !  sign        ?  hidden item
          ↓  ledge down  ←  ledge left  →  ledge right
          P  PC          T  TV/Machine  B  Bookshelf
          ^  display     S  Stair/warp
        """
        map_name = self.read_map_name()
        data = self._load_map_data(map_name)
        grid = data.get("grid", [])

        if not grid:
            return ""

        player_x, player_y = self.read_player_coords()
        map_h = len(grid)
        map_w = len(grid[0]) if map_h > 0 else 0

        vx_start, vx_end = self._clamp_viewport(player_x, radius, map_w)
        vy_start, vy_end = self._clamp_viewport(player_y, radius, map_h)

        # NPC and Poké Ball positions from RAM
        npc_positions = set()
        pokeball_positions: set = set()
        try:
            npc_data_json = data.get("npc_data", [])
            pokeball_map_positions = {
                (n["x"], n["y"]) for n in npc_data_json
                if "POKE_BALL" in n.get("sprite", "").upper()
            }
            live_sprites = self.read_sprites()
            live_sprite_positions = {(s['map_x'], s['map_y']) for s in live_sprites}
            npc_positions = live_sprite_positions
            pokeball_positions = pokeball_map_positions & live_sprite_positions
        except Exception:
            pass

        lines = []
        for sy in range(vy_start, vy_end):
            line = ""
            for sx in range(vx_start, vx_end):
                if sx == player_x and sy == player_y:
                    line += 'I'
                elif (sx, sy) in pokeball_positions:
                    line += 'O'
                elif (sx, sy) in npc_positions:
                    line += 'N'
                else:
                    symbol = grid[sy][sx]
                    # Use first char for multi-char symbols in compact view
                    if len(symbol) == 1:
                        line += symbol
                    else:
                        line += _COMPACT_SYMBOL.get(symbol, symbol[0])
            lines.append(line)
        return "\n".join(lines)

    def get_full_coll_map(self) -> dict:
        """Return the complete map data for the current location (for test/debug).

        Returns a dict with:
          map_name   : str
          player_x   : int   (grid column)
          player_y   : int   (grid row)
          map_width  : int   (grid columns)
          map_height : int   (grid rows)
          grid       : list[list[str]]  — full grid, row-major
        """
        map_name = self.read_map_name()
        data = self._load_map_data(map_name)
        grid = data.get("grid", [])
        player_x, player_y = self.read_player_coords()
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        return {
            "map_name":   map_name,
            "player_x":   player_x,
            "player_y":   player_y,
            "map_width":  w,
            "map_height": h,
            "grid":       grid,
        }

    def get_whole_map_data(self) -> dict:
        """Return complete map data for /whole_map endpoint.

        Returns a dict aligned with Emerald's porymap event structure:
          location, player_position, player_elevation, dimensions,
          grid (list-of-strings), raw_tiles (4-tuples),
          elevation_map, behavior_map,
          warp_events, bg_events, objects
        """
        map_name = self.read_map_name()
        data = self._load_map_data(map_name)
        grid = data.get("grid", [])
        player_x, player_y = self.read_player_coords()

        if not grid:
            return {
                "location": map_name,
                "player_position": {"x": player_x, "y": player_y},
                "dimensions": {"width": 0, "height": 0},
                "grid": [], "raw_tiles": [], "elevation_map": [],
                "behavior_map": [],
                "warp_events": [], "bg_events": [], "objects": [],
            }

        map_h = len(grid)
        map_w = len(grid[0]) if map_h > 0 else 0

        # Build raw_tiles from JSON raw_tile field (RedMetatileBehavior ints)
        raw_tile_json = data.get("raw_tile", [])
        raw_tiles = []
        if raw_tile_json:
            for row in raw_tile_json:
                tile_row = []
                for cell in row:
                    tid, beh_int, col, elev = cell
                    tile_row.append((tid, RedMetatileBehavior(beh_int), col, elev))
                raw_tiles.append(tile_row)
        else:
            # Fallback for old JSONs without raw_tile (builds from grid symbol strings)
            for row_s in grid:
                tile_row = []
                for symbol in row_s:
                    type_str, collision_int = self._classify_grid_symbol(symbol)
                    tile_row.append((0, type_str, collision_int, 0))
                raw_tiles.append(tile_row)

        # Derive behavior/elevation maps from raw_tiles
        behavior_map = [[t[1] for t in tile_row] for tile_row in raw_tiles]
        elevation_map = [[t[3] for t in tile_row] for tile_row in raw_tiles]

        # Structured data from JSON
        warps_data  = data.get("warps", [])
        signs_data  = data.get("signs", [])
        hidden_data = data.get("hidden_objects", [])

        # warp_events — matches Emerald's warps list shape (explicit elevation field)
        warp_events = [
            {"x": w["x"], "y": w["y"], "elevation": 0,
             "dest_map": w.get("dest_map", ""), "dest_warp_id": w.get("dest_warp_id", 0)}
            for w in warps_data
        ]

        # bg_events — signs + hidden objects combined, typed (matches Emerald bg_events)
        # "sign" entries carry text_id in script; "hidden_item" entries carry func name.
        # Both carry a Gen-1-specific symbol field (P/T/B/^/U/?) from classify_sign/hidden.
        bg_events = []
        for s in signs_data:
            bg_events.append({
                "type": "sign",
                "x": s["x"], "y": s["y"], "elevation": 0,
                "player_facing_dir": 0,
                "script": s.get("text_id", ""),
                "symbol": s.get("symbol", "?"),
            })
        for h in hidden_data:
            bg_events.append({
                "type": "hidden_item",
                "x": h["x"], "y": h["y"], "elevation": 0,
                "player_facing_dir": 0,
                "script": h.get("script", ""),
                "symbol": h.get("symbol", "#"),
            })

        # objects — NPC list from processed map data, filtered by live RAM.
        # Only item sprites (Poké Balls, Fossils) are filtered via live RAM:
        # they are permanently removed when the player picks them up.
        # Regular NPCs are always included from static data — Gen 1 only loads
        # on-screen sprites, so NPCs off-screen won't be in sprite slots even
        # though they still exist on the map.
        objects = []
        npc_data = data.get("npc_data", [])
        try:
            live_sprites = self.read_sprites()
            live_sprite_positions = {(s["map_x"], s["map_y"]) for s in live_sprites}
        except Exception as e:
            logger.warning(f"Failed to read NPC data from memory; Using processed map data only. {e}")
            live_sprites = None
            live_sprite_positions = None

        for s in npc_data:
            sprite_upper = s.get("sprite", "").upper()
            is_item_sprite = "POKE_BALL" in sprite_upper or "FOSSIL" in sprite_upper

            # Filter item sprites: only include if still present in live sprite RAM
            if is_item_sprite and live_sprite_positions is not None:
                if (s["x"], s["y"]) not in live_sprite_positions:
                    continue

            obj_tmp = {
                "x": s["x"], "y": s["y"], "elevation": 0,
                "sprite_name": s["sprite"],
                "movement_type": s["movement"],
                "facing": s["direction"],
                "graphics_id": s["text_id"]
            }
            objects.append(obj_tmp)

        return {
            "location": map_name,
            "player_position": {"x": player_x, "y": player_y},
            "player_elevation": 0,
            "dimensions": {"width": map_w, "height": map_h},
            "grid": grid,
            "raw_tiles": raw_tiles,
            "elevation_map": elevation_map,
            "behavior_map": behavior_map,
            "warp_events": warp_events,
            "bg_events": bg_events,
            "objects": objects,
        }

    def get_traversability_grid(self, radius: int = 4) -> list:
        """Return 2D bool grid.  True = walkable.  Clamped to map bounds."""
        map_name = self.read_map_name()
        data = self._load_map_data(map_name)
        grid = data.get("grid", [])

        if not grid:
            return []

        player_x, player_y = self.read_player_coords()
        map_h = len(grid)
        map_w = len(grid[0]) if map_h > 0 else 0

        vx_start, vx_end = self._clamp_viewport(player_x, radius, map_w)
        vy_start, vy_end = self._clamp_viewport(player_y, radius, map_h)

        result = []
        for sy in range(vy_start, vy_end):
            row = []
            for sx in range(vx_start, vx_end):
                symbol = grid[sy][sx]
                _, collision_int = self._classify_grid_symbol(symbol)
                row.append(collision_int == 0)
            result.append(row)
        return result

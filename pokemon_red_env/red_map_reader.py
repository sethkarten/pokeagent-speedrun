"""Pokemon Red map reader — hybrid RAM + processed_map approach.

Reads NPC positions from live RAM (wSpriteStateData1 at 0xC100) and uses
pre-computed processed_map/{map_name}.json files for tile classification.
Viewport is clamped to map bounds (Emerald-style) — never padded with walls.

Returns (tile_id, type_str, collision_int, elevation) tuples compatible with
map_formatter.is_tile_walkable() and format_map_grid().
"""

import copy
import json
import logging
import os
import re
import sys

from pokemon_red_env.utils.red_metatile_behavior import RedMetatileBehavior

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
        self._last_map_name: str = ""           # Track map changes for cache invalidation
        self._npc_position_cache: list = []     # Deep-copied npc_data with live corrections
        self._hidden_sprites: set = set()       # Indices of removed/hidden NPCs (picture_id=0)
        self._picked_up_items: set = set()      # Indices of picked-up items (POKE_BALL/FOSSIL)
        self._cleared_obstacles: set = set()    # {(cx, cy)} cut trees / opened gates this map visit
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

    def _ensure_npc_cache(self, map_name: str, npc_data: list) -> list:
        """Get or initialize NPC position cache for the current map.

        Returns the mutable cache list.  On map change, resets to static
        npc_data (Gen 1 resets NPC positions when player re-enters a map).
        """
        if map_name != self._last_map_name:
            self._last_map_name = map_name
            self._npc_position_cache = copy.deepcopy(npc_data)
            self._hidden_sprites = set()
            self._picked_up_items = set()
            self._cleared_obstacles = set()
            logger.debug(f"NPC cache reset for '{map_name}' ({len(npc_data)} entries)")

        # Guard: re-init if npc_data length changed (data reload edge case)
        if len(self._npc_position_cache) != len(npc_data):
            self._npc_position_cache = copy.deepcopy(npc_data)
            self._hidden_sprites = set()
            self._picked_up_items = set()
            self._cleared_obstacles = set()

        return self._npc_position_cache

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
    # Dynamic obstacle detection (cut trees / Card Key gates)
    # ------------------------------------------------------------------

    # Symbols that represent removable obstacles
    _OBSTACLE_SYMBOLS = frozenset({"t", "G"})

    def _detect_cleared_obstacles(self, grid: list, raw_tile_data: list) -> set:
        """Detect cut trees ('t') and opened gates ('G') via wTileMap comparison.

        Only checks the 3×3 area around the player (Cut and Card Key both
        require adjacency).  Before comparing, validates that the player's
        own tile matches between wTileMap and raw_tile to avoid false
        positives during screen transitions, menus, or battles.

        Results are accumulated in ``self._cleared_obstacles`` so off-screen
        removals persist until map re-entry (when Gen 1 respawns them).

        Returns the full set of cleared (cx, cy) positions.
        """
        if not grid or not raw_tile_data:
            return self._cleared_obstacles

        map_h = len(grid)
        map_w = len(grid[0]) if map_h else 0

        # Read wTileMap (360 bytes = 20 cols × 18 rows)
        try:
            tile_map_bytes = bytes(
                self._read_u8(0xC3A0 + i) for i in range(360)
            )
        except Exception:
            return self._cleared_obstacles

        # Player screen pixel position (sprite slot 0)
        player_screen_y = self._read_u8(0xC100 + 4)
        player_screen_x = self._read_u8(0xC100 + 6)
        player_x, player_y = self.read_player_coords()

        # Player collision cell → screen tile (top-left of 2×2 block).
        # +4 on Y adjusts from sprite top to collision cell top (Gen 1 offset).
        player_tile_row = (player_screen_y + 4) // 8
        player_tile_col = player_screen_x // 8

        # Consistency check: player's own cell must match raw_tile.
        # If it doesn't, the screen is in a transition/menu/battle — skip.
        if player_y < len(raw_tile_data) and player_x < len(raw_tile_data[player_y]):
            expected_player_tile = raw_tile_data[player_y][player_x][0]
            p_bl_row = player_tile_row + 1
            p_bl_col = player_tile_col
            if 0 <= p_bl_row < 18 and 0 <= p_bl_col < 20:
                actual_player_tile = tile_map_bytes[p_bl_row * 20 + p_bl_col]
                if actual_player_tile != expected_player_tile:
                    return self._cleared_obstacles  # screen not in sync
            else:
                return self._cleared_obstacles  # player off-screen (edge case)
        else:
            return self._cleared_obstacles

        # Only scan the 3×3 area around the player (adjacency requirement)
        for cy in range(max(0, player_y - 1), min(map_h, player_y + 2)):
            row_str = grid[cy]
            for cx in range(max(0, player_x - 1), min(map_w, player_x + 2)):
                if (cx, cy) in self._cleared_obstacles:
                    continue  # already known cleared
                sym = row_str[cx] if cx < len(row_str) else "#"
                # Only cuttable trees ('t') and Card Key gates ('G')
                if sym not in self._OBSTACLE_SYMBOLS:
                    continue

                # Screen tile position (top-left of 2×2 collision cell block)
                cell_tile_row = player_tile_row + (cy - player_y) * 2
                cell_tile_col = player_tile_col + (cx - player_x) * 2

                # Use bottom-left tile to match raw_tile convention
                # (raw_tile stores tile_id_map[2*cy+1][2*cx])
                bl_row = cell_tile_row + 1
                bl_col = cell_tile_col

                # Skip if off-screen
                if bl_row < 0 or bl_row >= 18 or bl_col < 0 or bl_col >= 20:
                    continue

                actual_tile = tile_map_bytes[bl_row * 20 + bl_col]

                # Expected tile from preprocessed data
                if cy < len(raw_tile_data) and cx < len(raw_tile_data[cy]):
                    expected_tile = raw_tile_data[cy][cx][0]
                else:
                    continue

                if actual_tile != expected_tile:
                    self._cleared_obstacles.add((cx, cy))
                    logger.debug(
                        f"Obstacle cleared at ({cx},{cy}): "
                        f"expected tile 0x{expected_tile:02X}, "
                        f"got 0x{actual_tile:02X} (symbol='{sym}')"
                    )

        return self._cleared_obstacles

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

        # Detect cleared obstacles (cut trees / opened gates) via screen tiles
        raw_tile_json = data.get("raw_tile", [])
        cleared = self._detect_cleared_obstacles(grid, raw_tile_json)

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
                # Replace cleared obstacles with walkable
                if (sx, sy) in cleared:
                    symbol = "."
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
          ^  display     S  Stair/warp  t  cuttable tree
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

        # Detect cleared obstacles (cut trees / opened gates) via screen tiles
        raw_tile_json = data.get("raw_tile", [])
        cleared = self._detect_cleared_obstacles(grid, raw_tile_json)

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
                    # Replace cleared obstacles with walkable
                    if (sx, sy) in cleared:
                        symbol = "."
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
        grid_static = data.get("grid", [])
        player_x, player_y = self.read_player_coords()

        if not grid_static:
            return {
                "location": map_name,
                "player_position": {"x": player_x, "y": player_y},
                "dimensions": {"width": 0, "height": 0},
                "grid": [], "raw_tiles": [], "elevation_map": [],
                "behavior_map": [],
                "warp_events": [], "bg_events": [], "objects": [],
            }

        map_h = len(grid_static)
        map_w = len(grid_static[0]) if map_h > 0 else 0

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
            for row_s in grid_static:
                tile_row = []
                for symbol in row_s:
                    type_str, collision_int = self._classify_grid_symbol(symbol)
                    tile_row.append((0, type_str, collision_int, 0))
                raw_tiles.append(tile_row)

        # Detect and apply cleared obstacles (cut trees / opened gates)
        cleared = self._detect_cleared_obstacles(grid_static, raw_tile_json)
        if cleared:
            # Copy grid (list of strings) to mutable form, apply clears
            grid = [list(row) for row in grid_static]
            for cx, cy in cleared:
                if 0 <= cy < map_h and 0 <= cx < map_w:
                    grid[cy][cx] = "."
                    # Also update raw_tiles collision to walkable
                    if cy < len(raw_tiles) and cx < len(raw_tiles[cy]):
                        tid, beh, _col, elev = raw_tiles[cy][cx]
                        raw_tiles[cy][cx] = (tid, RedMetatileBehavior.NORMAL, 0, elev)
            grid = ["".join(row) for row in grid]
        else:
            grid = grid_static

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

        # objects — NPC list with live RAM position correction.
        # Slot i in RAM maps to npc_data[i-1] (Gen 1 stable ordering,
        # verified from pokered/home/overworld.asm LoadMapHeader).
        npc_data = data.get("npc_data", [])
        npc_cache = self._ensure_npc_cache(map_name, npc_data)

        try:
            live_sprites = self.read_sprites()
        except Exception as e:
            logger.warning(f"Failed to read sprites from RAM; using cached positions. {e}")
            live_sprites = []

        # Slot → live sprite for O(1) lookup
        live_by_slot: dict = {s["slot"]: s for s in live_sprites}

        # Phase 1: Update cached positions from live RAM
        for idx, cached_npc in enumerate(npc_cache):
            slot = idx + 1  # slot is 1-indexed
            if slot > 15:
                break  # Gen 1 hardware limit: 15 NPC sprite slots

            if slot in live_by_slot:
                # Sprite active in RAM — update position + facing
                live = live_by_slot[slot]
                old_x, old_y = cached_npc["x"], cached_npc["y"]
                cached_npc["x"] = live["map_x"]
                cached_npc["y"] = live["map_y"]
                cached_npc["_live_facing"] = live["facing"]
                self._hidden_sprites.discard(idx)  # re-appeared (e.g. ShowObject)

                if old_x != cached_npc["x"] or old_y != cached_npc["y"]:
                    logger.debug(
                        f"NPC {cached_npc['sprite']} (slot {slot}): "
                        f"({old_x},{old_y})->({cached_npc['x']},{cached_npc['y']})"
                    )
            elif idx not in self._hidden_sprites:
                # Sprite not in read_sprites(). Check picture_id to distinguish
                # off-screen (picture_id>0, byte+2=0xFF) from removed/hidden
                # (picture_id=0). Applies to ALL NPCs: items picked up, NPCs
                # hidden via toggleable objects (e.g. Viridian Old Man).
                picture_id = self._read_u8(0xC100 + slot * 0x10)
                if picture_id == 0:
                    self._hidden_sprites.add(idx)
                    logger.debug(
                        f"Sprite {cached_npc['sprite']} (slot {slot}) hidden "
                        f"(picture_id=0)"
                    )

        # Phase 1.5: Detect disappeared sprites via proximity.
        # Gen 1 hides sprites (picked-up items, defeated trainers) by
        # setting byte+2=0xFF rather than picture_id=0, so _hidden_sprites
        # won't catch them. If ANY sprite is not in live RAM AND the player
        # is within Manhattan distance 4, mark it as removed. Items go into
        # _picked_up_items; other NPCs go into _hidden_sprites.
        for idx, cached_npc in enumerate(npc_cache):
            slot = idx + 1
            if slot > 15:
                break
            if idx in self._hidden_sprites or idx in self._picked_up_items:
                continue
            if slot not in live_by_slot:
                dx = abs(cached_npc["x"] - player_x)
                dy = abs(cached_npc["y"] - player_y)
                if dx + dy <= 4:
                    sprite_upper = cached_npc.get("sprite", "").upper()
                    is_item = "POKE_BALL" in sprite_upper or "FOSSIL" in sprite_upper
                    if is_item:
                        self._picked_up_items.add(idx)
                        logger.debug(
                            f"Item {cached_npc['sprite']} (slot {slot}) picked up "
                            f"(player dist={dx+dy}, not in live RAM)"
                        )
                    else:
                        self._hidden_sprites.add(idx)
                        logger.debug(
                            f"NPC {cached_npc['sprite']} (slot {slot}) disappeared "
                            f"(player dist={dx+dy}, not in live RAM)"
                        )

        # Phase 2: Build objects list from cache (skip hidden/removed sprites)
        objects = []
        for idx, cached_npc in enumerate(npc_cache):
            if idx in self._hidden_sprites:
                continue
            if idx in self._picked_up_items:
                continue
            facing = cached_npc.get("_live_facing", cached_npc["direction"])
            objects.append({
                "x": cached_npc["x"],
                "y": cached_npc["y"],
                "elevation": 0,
                "sprite_name": cached_npc["sprite"],
                "movement_type": cached_npc["movement"],
                "facing": facing,
                "graphics_id": cached_npc["text_id"],
            })

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

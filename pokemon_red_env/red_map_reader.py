"""Pokemon Red map reader — hybrid RAM + processed_map approach.

Reads NPC positions from live RAM (wSpriteStateData1 at 0xC100) and uses
pre-computed processed_map/{map_name}.py files for tile classification.
Viewport is clamped to map bounds (Emerald-style) — never padded with walls.

Returns (tile_id, type_str, collision_int, elevation) tuples compatible with
map_formatter.is_tile_walkable() and format_map_grid().
"""

import importlib.util
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# coll_map character → (type_str, collision_int)
#
# In Gen 1, the "collision tile IDs" list contains PASSABLE tiles.
# map_preprocess.py: is_collision = (tid in coll_set) → 'O' means passable,
# 'X' means obstacle.  (The naming is counterintuitive but verified against
# pokered decomp: overworld.asm:1263 "wTilesetCollisionPtr ; pointer to list
# of passable tiles".)
# ---------------------------------------------------------------------------
COLL_CHAR_MAP = {
    'O':   ('WALKABLE', 0),  # passable ground (in collision/passable set)
    'X':   ('WALL', 1),      # blocked / obstacle (NOT in passable set)
    'G':   ('GRASS', 0),     # tall grass (walkable, triggers encounters)
    '~':   ('WATER', 1),     # water (blocked without Surf)
    'D':   ('LEDGE_D', 0),   # jump-down ledge
    'L':   ('LEDGE_L', 0),   # jump-left ledge
    'R':   ('LEDGE_R', 0),   # jump-right ledge
    '|':   ('WALL', 1),      # pair-collision directional wall
    '-':   ('WALL', 1),      # pair-collision directional wall
    'C':   ('COUNTER', 1),   # shop/PC counter
    'Cut': ('CUT', 1),       # HM Cut tree (blocked without Cut)
    '?':   ('UNKNOWN', 0),   # unknown → assume walkable
} # TODO: change ascii symbol mapping to match the Emerald settings

# String prefixes handled separately in _classify_cell:
#   'WarpPoint' → ('WARP',    0)
#   'SIGN_'     → ('SIGN',    1)
#   'TalkTo'    → ('SIGN',    1)

# Sprite facing direction byte → string
_FACING_MAP = {0: "down", 4: "up", 8: "left", 12: "right"}

# Regex for parsing object_event sprite names from .asm files
_OBJECT_EVENT_RE = re.compile(r"object_event\s+\d+,\s*\d+,\s*([A-Z0-9_]+)")


def _type_str_to_symbol(type_str: str, symbols: dict) -> str:
    """Map a (possibly rich) type_str to an ASCII symbol for visual display."""
    if type_str in symbols:
        return symbols[type_str]
    if type_str.startswith('WarpPoint'):
        return 'W'
    if type_str.startswith('SIGN_') or type_str.startswith('TalkTo'):
        return 's'
    if type_str.startswith('SPRITE_'):
        return 'N'
    return '?'


class RedMapReader:
    """Provides map collision + NPC data for Pokemon Red."""

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self._map_names: dict = {}   # str(map_id) → map_name
        self._map_cache: dict = {}   # map_name → coll_map (list[list[str]])
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

    def _load_coll_map(self, map_name: str) -> list:
        """Load and cache coll_map from processed_map/{map_name}.py.

        Returns [] if the file does not exist.
        """
        if map_name in self._map_cache:
            return self._map_cache[map_name]

        pm_dir = os.path.join(self.DATA_DIR, "processed_map")
        path = self._resolve_path_ci(pm_dir, f"{map_name}.py")
        if not os.path.exists(path):
            logger.debug(f"No processed_map file for '{map_name}'")
            self._map_cache[map_name] = []
            return []

        try:
            spec = importlib.util.spec_from_file_location(map_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            coll_map = getattr(mod, "coll_map", [])
            self._map_cache[map_name] = coll_map
            logger.debug(f"Loaded coll_map for '{map_name}': {len(coll_map)} rows")
            return coll_map
        except Exception as e:
            logger.warning(f"Error loading processed_map/{map_name}.py: {e}")
            self._map_cache[map_name] = []
            return []

    def _load_sprite_names(self, map_name: str) -> list:
        """Parse object_event entries from the map's .asm file.

        Returns ordered list of sprite constant names (e.g. SPRITE_OAK).
        Slot i (1-based) corresponds to sprite_names[i-1].
        """
        if map_name in self._sprite_name_cache:
            return self._sprite_name_cache[map_name]

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
    # Cell classification
    # ------------------------------------------------------------------

    def _classify_cell(self, cell: str) -> tuple:
        """Convert a coll_map string to (type_str, collision_int).

        collision_int: 0 = walkable, 1 = blocked.
        type_str preserves the original cell string for rich types
        (e.g. 'WarpPoint', 'SIGN_ROUTE_1', 'TalkToOAK') so downstream
        consumers get full context.  Only simple chars like 'O'/'X' are
        mapped to generic labels ('WALKABLE'/'WALL').
        """
        if cell in COLL_CHAR_MAP:
            return COLL_CHAR_MAP[cell]
        if cell.startswith('WarpPoint'):
            return (cell, 0)
        if cell.startswith('SIGN_') or cell.startswith('TalkTo'):
            return (cell, 1)
        if cell.startswith('SPRITE_'):
            return (cell, 1)
        return (cell, 0)

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
        coll_map = self._load_coll_map(map_name)

        if not coll_map:
            return []

        player_x, player_y = self.read_player_coords()
        map_h = len(coll_map)
        map_w = len(coll_map[0]) if map_h > 0 else 0

        vx_start, vx_end = self._clamp_viewport(player_x, radius, map_w)
        vy_start, vy_end = self._clamp_viewport(player_y, radius, map_h)

        # Build NPC position set from RAM
        npc_positions = set()
        try:
            for s in self.read_sprites():
                npc_positions.add((s['map_x'], s['map_y']))
        except Exception as e:
            logger.debug(f"Could not read sprites: {e}")

        rows = []
        for sy in range(vy_start, vy_end):
            row = []
            for sx in range(vx_start, vx_end):
                cell = coll_map[sy][sx]
                type_str, collision_int = self._classify_cell(cell)

                # Overlay NPC from RAM
                if (sx, sy) in npc_positions:
                    type_str = 'NPC'
                    collision_int = 1

                row.append((0, type_str, collision_int, 0))
            rows.append(row)
        return rows

    def format_map_for_llm(self, radius: int = 4) -> str:
        """Return compact ASCII map of the area around the player (for test only)

        Symbols:
          P  player      .  walkable    #  wall / blocked
          G  tall grass  ~  water       W  warp / door
          N  NPC         s  sign        T  Cut tree
          c  counter     v  ledge down  <  ledge left   >  ledge right
          ?  unknown
        """
        CELL_SYMBOLS = {
            'WALKABLE': '.', 'WALL': '#', 'GRASS': 'G', 'WATER': '~',
            'WARP':     'W', 'NPC':  'N', 'SIGN':  's', 'CUT':   'T',
            'COUNTER':  'c', 'LEDGE_D': 'v', 'LEDGE_L': '<', 'LEDGE_R': '>',
            'UNKNOWN':  '?',
        }
        map_name = self.read_map_name()
        coll_map = self._load_coll_map(map_name)

        if not coll_map:
            return ""

        player_x, player_y = self.read_player_coords()
        map_h = len(coll_map)
        map_w = len(coll_map[0]) if map_h > 0 else 0

        vx_start, vx_end = self._clamp_viewport(player_x, radius, map_w)
        vy_start, vy_end = self._clamp_viewport(player_y, radius, map_h)

        # NPC positions from RAM
        npc_positions = set()
        try:
            for s in self.read_sprites():
                npc_positions.add((s['map_x'], s['map_y']))
        except Exception:
            pass

        lines = []
        for sy in range(vy_start, vy_end):
            line = ""
            for sx in range(vx_start, vx_end):
                if sx == player_x and sy == player_y:
                    line += 'P'
                elif (sx, sy) in npc_positions:
                    line += 'N'
                else:
                    cell = coll_map[sy][sx]
                    type_str, _ = self._classify_cell(cell)
                    line += _type_str_to_symbol(type_str, CELL_SYMBOLS)
            lines.append(line)
        return "\n".join(lines)

    def get_full_coll_map(self) -> dict:
        """Return the complete coll_map for the current location (for test/debug only).

        Returns a dict with:
          map_name   : str
          player_x   : int   (coll_map column)
          player_y   : int   (coll_map row)
          map_width  : int   (coll_map columns = .blk blocks × 2)
          map_height : int   (coll_map rows    = .blk blocks × 2)
          coll_map   : list[list[str]]  — full map, row-major
        """
        map_name = self.read_map_name()
        coll_map = self._load_coll_map(map_name)
        player_x, player_y = self.read_player_coords()
        h = len(coll_map)
        w = len(coll_map[0]) if h > 0 else 0
        return {
            "map_name":   map_name,
            "player_x":   player_x,
            "player_y":   player_y,
            "map_width":  w,
            "map_height": h,
            "coll_map":   coll_map,
        }

    def get_whole_map_data(self) -> dict:
        """Return complete map data for /whole_map endpoint.

        Returns a dict matching the shape of Emerald's /whole_map response:
          location, player_position, dimensions, grid (ASCII),
          raw_tiles (4-tuples), elevation_map, behavior_map,
          special_tiles, warps, objects
        """
        CELL_SYMBOLS = {
            'WALKABLE': '.', 'WALL': '#', 'GRASS': 'G', 'WATER': '~',
            'WARP':     'W', 'NPC':  'N', 'SIGN':  's', 'CUT':   'T',
            'COUNTER':  'c', 'LEDGE_D': 'v', 'LEDGE_L': '<', 'LEDGE_R': '>',
            'UNKNOWN':  '?',
        }

        map_name = self.read_map_name()
        coll_map = self._load_coll_map(map_name)
        player_x, player_y = self.read_player_coords()

        if not coll_map:
            return {
                "location": map_name,
                "player_position": {"x": player_x, "y": player_y},
                "dimensions": {"width": 0, "height": 0},
                "grid": [], "raw_tiles": [], "elevation_map": [],
                "behavior_map": [], "special_tiles": {},
                "warps": [], "objects": [],
            }

        map_h = len(coll_map)
        map_w = len(coll_map[0]) if map_h > 0 else 0

        grid = []
        raw_tiles = []
        behavior_map = []
        elevation_map = []
        special_tiles = {}
        warps = []

        for y, row in enumerate(coll_map):
            grid_row = []
            tile_row = []
            behav_row = []
            elev_row = []
            for x, cell in enumerate(row):
                type_str, collision_int = self._classify_cell(cell)

                # ASCII grid
                grid_row.append(_type_str_to_symbol(type_str, CELL_SYMBOLS))

                # Raw tile 4-tuple
                tile_row.append((0, type_str, collision_int, 0))

                # Behavior and elevation
                behav_row.append(type_str)
                elev_row.append(0)  # Gen 1 has no elevation

                # Collect special tiles
                if cell.startswith('WarpPoint'):
                    warps.append({"x": x, "y": y, "name": cell})
                    special_tiles.setdefault("WARP", []).append(
                        {"x": x, "y": y, "elevation": 0, "behavior_id": cell}
                    )
                elif cell.startswith('SIGN_') or cell.startswith('TalkTo'):
                    special_tiles.setdefault("SIGN", []).append(
                        {"x": x, "y": y, "elevation": 0, "behavior_id": cell}
                    )

            grid.append(grid_row)
            raw_tiles.append(tile_row)
            behavior_map.append(behav_row)
            elevation_map.append(elev_row)

        # NPC/object data from RAM
        objects = []
        try:
            for s in self.read_sprites():
                objects.append({
                    "x": s["map_x"], "y": s["map_y"],
                    "sprite_name": s["sprite_name"],
                    "facing": s["facing"],
                    "graphics_id": s["picture_id"],
                })
        except Exception:
            pass

        return {
            "location": map_name,
            "player_position": {"x": player_x, "y": player_y},
            "player_elevation": 0,
            "dimensions": {"width": map_w, "height": map_h},
            "grid": grid,
            "raw_tiles": raw_tiles,
            "elevation_map": elevation_map,
            "behavior_map": behavior_map,
            "special_tiles": special_tiles,
            "warps": warps,
            "objects": objects,
        }

    def get_traversability_grid(self, radius: int = 4) -> list:
        """Return 2D bool grid.  True = walkable.  Clamped to map bounds."""
        map_name = self.read_map_name()
        coll_map = self._load_coll_map(map_name)

        if not coll_map:
            return []

        player_x, player_y = self.read_player_coords()
        map_h = len(coll_map)
        map_w = len(coll_map[0]) if map_h > 0 else 0

        vx_start, vx_end = self._clamp_viewport(player_x, radius, map_w)
        vy_start, vy_end = self._clamp_viewport(player_y, radius, map_h)

        grid = []
        for sy in range(vy_start, vy_end):
            row = []
            for sx in range(vx_start, vx_end):
                cell = coll_map[sy][sx]
                _, collision_int = self._classify_cell(cell)
                row.append(collision_int == 0)
            grid.append(row)
        return grid

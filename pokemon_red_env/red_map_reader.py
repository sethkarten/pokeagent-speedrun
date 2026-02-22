"""Pokemon Red map reader using pre-computed processed_map data.

Reads map collision data from processed_map/{map_name}.py files (derived from
the pokered decompilation) and uses live RAM coordinates to extract the viewport
around the player.  Returns (tile_id, type_str, collision_int, elevation) tuples
compatible with map_formatter.is_tile_walkable() and format_map_grid().

Algorithm mirrors self-evolving-game-agent/evaluation_utils/…/pyboy_runner.py
get_map_info() (lines 419-467).
"""

import importlib.util
import json
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# coll_map character → (type_str, collision_int)
# ---------------------------------------------------------------------------
COLL_CHAR_MAP = {
    'X':   ('WALKABLE', 0),  # passable ground
    'O':   ('WALL', 1),      # blocked / wall
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
}
# String prefixes handled separately in _classify_cell:
#   'WarpPoint' → ('WARP',    0)
#   'SIGN_'     → ('SIGN',    1)
#   'TalkTo'    → ('SIGN',    1)
#   'SPRITE_'   → ('NPC',     1)


class RedMapReader:
    """Provides map collision data for Pokemon Red using pre-computed processed_map files."""

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self._map_names: dict = {}   # str(map_id) → map_name
        self._map_cache: dict = {}   # map_name → coll_map (list[list[str]])
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

    def _load_coll_map(self, map_name: str) -> list:
        """Load and cache coll_map from processed_map/{map_name}.py.

        Returns [] if the file does not exist.
        """
        if map_name in self._map_cache:
            return self._map_cache[map_name]

        path = os.path.join(self.DATA_DIR, "processed_map", f"{map_name}.py")
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
    # Cell classification
    # ------------------------------------------------------------------

    def _classify_cell(self, cell: str) -> tuple:
        """Convert a coll_map string to (type_str, collision_int)."""
        if cell in COLL_CHAR_MAP:
            return COLL_CHAR_MAP[cell]
        if cell.startswith('WarpPoint'):
            return ('WARP', 0)
        if cell.startswith('SIGN_') or cell.startswith('TalkTo'):
            return ('SIGN', 1)
        if cell.startswith('SPRITE_'):
            return ('NPC', 1)
        return ('UNKNOWN', 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_map_around_player(self, radius: int = 7) -> list:
        """Return 2D list of (0, type_str, collision_int, 0) tuples.

        Grid dimensions: (2*radius+1) rows × (2*radius+1) columns in block coords,
        centred on the player.  Out-of-bounds cells are treated as walls.

        Format is compatible with map_formatter.is_tile_walkable() which checks
        tile[2] == 0.
        """
        map_name = self.read_map_name()
        coll_map = self._load_coll_map(map_name)
        player_x, player_y = self.read_player_coords()

        rows = []
        for dy in range(-radius, radius + 1):
            sy = player_y + dy
            row = []
            for dx in range(-radius, radius + 1):
                sx = player_x + dx
                if coll_map and 0 <= sy < len(coll_map) and 0 <= sx < len(coll_map[sy]):
                    cell = coll_map[sy][sx]
                else:
                    cell = 'O'  # out-of-bounds → wall
                type_str, collision_int = self._classify_cell(cell)
                row.append((0, type_str, collision_int, 0))
            rows.append(row)
        return rows

    def get_traversability_grid(self, radius: int = 4) -> list:
        """Return 2D bool grid (default radius 4 = 9×9).  True = walkable."""
        map_name = self.read_map_name()
        coll_map = self._load_coll_map(map_name)
        player_x, player_y = self.read_player_coords()

        grid = []
        for dy in range(-radius, radius + 1):
            sy = player_y + dy
            row = []
            for dx in range(-radius, radius + 1):
                sx = player_x + dx
                if coll_map and 0 <= sy < len(coll_map) and 0 <= sx < len(coll_map[sy]):
                    cell = coll_map[sy][sx]
                    _, collision_int = self._classify_cell(cell)
                    row.append(collision_int == 0)
                else:
                    row.append(False)
            grid.append(row)
        return grid

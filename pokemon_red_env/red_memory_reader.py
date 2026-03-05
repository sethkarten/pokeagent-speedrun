"""Pokemon Red memory reader using PyBoy backend.

Mirrors the public interface of PokemonEmeraldReader so RedEmulator can be used
as a drop-in replacement for EmeraldEmulator by server/app.py and agent scaffolds.

All data sourced from the pokered decompilation (https://github.com/pret/pokered)
and validated against docs/pokemon_red_proposal.md.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_json_mapping(filename: str) -> Dict[str, str]:
    """Load a JSON file from the data directory; return {} on failure."""
    path = os.path.join(_DATA_DIR, filename)
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {filename}: {e}")
        return {}


# Internal species ID (stored in RAM) → species name
# Gen 1 uses internal ordering that differs from Pokedex order
_SPECIES_NAMES: Dict[str, str] = _load_json_mapping("species_names.json")

# Move ID → move name
_MOVE_NAMES: Dict[str, str] = _load_json_mapping("move_names.json")

# Byte value (str) → display character; JSON null means string terminator
_CHARMAP: Dict[str, Any] = _load_json_mapping("charmap.json")

# Type ID → type name
_TYPE_NAMES: Dict[str, str] = _load_json_mapping("type_names.json")

# Item ID → item name
_ITEM_NAMES: Dict[str, str] = _load_json_mapping("item_names.json")

# Map ID → map name (248 entries, full coverage)
_MAP_NAMES: Dict[str, str] = _load_json_mapping("map_names.json")

# RAM address table (from pokered decompilation)
# TODO: verify with https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Rival
RED_ADDR: Dict[str, int] = {
    # Player
    "player_name":      0xD158,  # 11 bytes, Gen1 charset, 0x50 = terminator
    "rival_name":       0xD34A,  # 11 bytes
    "player_y":         0xD361,  # 1 byte tile coord
    "player_x":         0xD362,  # 1 byte tile coord
    "player_dir":       0xC109,  # 1 byte: 0=down, 4=up, 8=left, 12=right
    # Map
    "map_id":           0xD35E,  # 1 byte
    "map_height":       0xD368,  # wCurMapHeight (blocks)
    "map_width":        0xD369,  # wCurMapWidth  (blocks)
    # Economy
    "money":            0xD347,  # 3 bytes BCD big-endian
    # Inventory (wNumBagItems / wBagItems from pokered decomp)
    "inventory_count":  0xD31D,  # 1 byte  (wNumBagItems)
    "inventory_items":  0xD31E,  # 2*count bytes: alternating (item_id, quantity) (wBagItems)
    # Party
    "party_count":      0xD163,  # 1 byte (0–6)
    "party_species":    0xD164,  # 6 bytes, species IDs
    "party_data_base":  0xD16B,  # 6 × 0x2C bytes (44 bytes per slot)
    "party_nicknames":  0xD2B5,  # 6 × 11 bytes
    "party_ot_names":   0xD273,  # 6 × 11 bytes
    # Progress
    "badges":           0xD356,  # 1 byte bitmask (bit 0 = Boulder Badge)
    "pokedex_owned":    0xD2F7,  # 19 bytes bitfield (151 entries)
    "pokedex_seen":     0xD30A,  # 19 bytes bitfield
    # Battle
    "in_battle":        0xD057,  # 0=none, 1=wild, 2=trainer
    "link_state":       0xD72E,  # 0x05 = link battle
    # wEnemyMon struct @ 0xCFE5 (pokered wram.asm layout):
    #   species(1), hp(2), blank(1), status(1), type1(1), type2(1), catch_rate(1),
    #   moves(4), OTID(2), level(1), max_hp(2), attack(2), defense(2), speed(2), special(2), pp(4)
    "enemy_species":    0xCFE5,  # 1 byte
    "enemy_hp":         0xCFE6,  # 2 bytes big-endian
    "enemy_status":     0xCFE9,  # 1 byte status condition
    "enemy_type1":      0xCFEA,  # 1 byte
    "enemy_type2":      0xCFEB,  # 1 byte
    "enemy_moves":      0xCFED,  # 4 bytes (move IDs)
    "enemy_level":      0xCFF3,  # 1 byte
    "enemy_max_hp":     0xCFF4,  # 2 bytes big-endian
    "enemy_attack":     0xCFF6,  # 2 bytes big-endian
    "enemy_defense":    0xCFF8,  # 2 bytes big-endian
    "enemy_speed":      0xCFFA,  # 2 bytes big-endian
    "enemy_special":    0xCFFC,  # 2 bytes big-endian (Gen 1: unified special stat)
    "enemy_pp":         0xCFFE,  # 4 bytes (PP for moves 1-4)
    # Active player battle mon name (confirmed: pyboy_runner.py::get_active_pokemon_name)
    "active_pokemon_name": 0xD009,  # 11 bytes, Gen1 charset
    # Title screen detection (wTitleCheckDigit / wCurMap at title)
    "title_check":      0xC0EF,  # 0x1F on title screen
    "title_map_id":     0xD35C,  # 0x00 on title screen
    # Dialog / text
    "text_progress":    0xC6AC,  # 1 byte, non-zero = text printing active
    "vram_tilemap":     0xC3A0,  # 20×18 = 360 screen tiles
}

# Each party slot is 44 bytes.  Confirmed offsets (pokered decomp wPartyMon1):
#   +0x00: species_id (internal ID, NOT Pokedex number)
#   +0x01–0x02: current_hp (big-endian)
#   +0x04: status condition  (0=OK, 0x08=PSN, 0x10=BRN, 0x20=FRZ, 0x40=PAR, 0–7=SLP turns)
#   +0x05: type1
#   +0x06: type2
#   +0x08–0x0B: move IDs (4 × 1 byte)
#   +0x1D–0x20: PP for moves 1–4 (1 byte each)
#   +0x21: level
#   +0x22–0x23: max_hp (big-endian)
#   +0x24–0x25: attack (big-endian)
#   +0x26–0x27: defense (big-endian)
#   +0x28–0x29: speed (big-endian)
#   +0x2A–0x2B: special (big-endian)
PARTY_SLOT_SIZE = 0x2C  # 44 bytes

# ---------------------------------------------------------------------------
# Badge names (bit 0 = Boulder, bit 7 = Earth) — no JSON equivalent needed
# ---------------------------------------------------------------------------
RED_BADGE_NAMES: List[str] = [
    "Boulder", "Cascade", "Thunder", "Rainbow",
    "Soul", "Marsh", "Volcano", "Earth",
]


# ---------------------------------------------------------------------------
# Main reader class
# ---------------------------------------------------------------------------

class RedMemoryReader:
    """Memory reader for Pokemon Red using PyBoy backend.

    Mirrors the public interface of PokemonEmeraldReader so that RedEmulator
    can serve as a drop-in replacement for EmeraldEmulator.
    """

    def __init__(self, pyboy):
        self.pyboy = pyboy
        # Dialog state cache (updated by _update_dialog_state_cache in emulator)
        self._dialog_cache: bool = False
        self._dialog_cache_time: float = 0.0
        # Map reader — set by RedEmulator.initialize() after construction
        self.map_reader: Optional[object] = None
        # Area transition detection (used by server/app.py step_environment)
        self._last_map_id: Optional[int] = None
        self._area_transition_detected: bool = False
        # Dialog detection flag — set to False by --no-ocr to suppress VRAM dialog text
        self._dialog_detection_enabled: bool = True

    # ------------------------------------------------------------------
    # Low-level read primitives
    # ------------------------------------------------------------------

    def _read_u8(self, addr: int) -> int:
        try:
            return int(self.pyboy.memory[addr])
        except Exception:
            return 0

    def _read_u16_be(self, addr: int) -> int:
        """Read 2-byte big-endian unsigned integer."""
        hi = self._read_u8(addr)
        lo = self._read_u8(addr + 1)
        return (hi << 8) | lo

    def _read_u16_le(self, addr: int) -> int:
        """Read 2-byte little-endian unsigned integer."""
        lo = self._read_u8(addr)
        hi = self._read_u8(addr + 1)
        return lo | (hi << 8)

    def _read_bytes(self, addr: int, n: int) -> bytes:
        try:
            return bytes([self.pyboy.memory[addr + i] for i in range(n)])
        except Exception:
            return b"\x00" * n

    @staticmethod
    def _bcd_to_int(raw: bytes) -> int:
        """Decode big-endian BCD bytes to integer (e.g. b'\\x01\\x23\\x45' → 12345)."""
        result = 0
        for byte in raw:
            result = result * 100 + ((byte >> 4) * 10) + (byte & 0x0F)
        return result

    @staticmethod
    def _decode_gen1_text(byte_array: bytes) -> str:
        """Decode Gen 1 custom character encoding to a Python string."""
        out = []
        for b in byte_array:
            if b == 0x50:  # Gen 1 string terminator (not in charmap.json)
                break
            key = str(b)
            if key not in _CHARMAP:
                continue  # unknown control byte, skip
            ch = _CHARMAP[key]
            if ch is None or ch == "<NULL>":
                break  # JSON-null or explicit null marker = terminator
            if ch:  # skip empty-string control codes
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _status_name(byte: int) -> str:
        """Convert status condition byte to human-readable string."""
        if byte == 0:
            return "OK"
        if byte & 0x40:
            return "PAR"
        if byte & 0x20:
            return "FRZ"
        if byte & 0x10:
            return "BRN"
        if byte & 0x08:
            return "PSN"
        if byte & 0x07:
            return f"SLP({byte & 0x07})"
        return f"STATUS(0x{byte:02X})"

    # ------------------------------------------------------------------
    # Core public interface (mirrors PokemonEmeraldReader)
    # ------------------------------------------------------------------

    def read_player_name(self) -> str:
        raw = self._read_bytes(RED_ADDR["player_name"], 11)
        return self._decode_gen1_text(raw)

    def read_money(self) -> int:
        raw = self._read_bytes(RED_ADDR["money"], 3)
        return self._bcd_to_int(raw)

    def read_coordinates(self) -> Tuple[int, int]:
        """Return (x, y) tile coordinates."""
        x = self._read_u8(RED_ADDR["player_x"])
        y = self._read_u8(RED_ADDR["player_y"])
        return (x, y)

    def read_location(self) -> str:
        """Return the map name string for the current map ID."""
        map_id = self._read_u8(RED_ADDR["map_id"])
        return _MAP_NAMES.get(str(map_id), f"MAP_{map_id}")

    def read_party_size(self) -> int:
        return min(self._read_u8(RED_ADDR["party_count"]), 6)

    def read_party_pokemon(self) -> List[Dict[str, Any]]:
        """Parse all party Pokemon from RAM. Returns a list of dicts.

        Slot layout (pokered decompilation wPartyMon1, 0x2C bytes per slot):
          +0x00: species_id (internal ID, NOT Pokedex number)
          +0x01-0x02: current HP (big-endian)
          +0x04: status condition
          +0x05: type1
          +0x06: type2
          +0x08-0x0B: move IDs (4 bytes)
          +0x1D-0x20: PP for moves 1-4
          +0x21: level
          +0x22-0x23: max HP (big-endian)
          +0x24-0x25: attack
          +0x26-0x27: defense
          +0x28-0x29: speed
          +0x2A-0x2B: special
        """
        count = self.read_party_size()
        party = []
        for i in range(count):
            base = RED_ADDR["party_data_base"] + i * PARTY_SLOT_SIZE
            slot = self._read_bytes(base, PARTY_SLOT_SIZE)

            species_id  = slot[0x00]
            current_hp  = (slot[0x01] << 8) | slot[0x02]
            status      = slot[0x04]
            type1_id    = slot[0x05]
            type2_id    = slot[0x06]
            move_ids    = [slot[0x08], slot[0x09], slot[0x0A], slot[0x0B]]
            move_pp     = [slot[0x1D], slot[0x1E], slot[0x1F], slot[0x20]]
            level       = slot[0x21]
            max_hp      = (slot[0x22] << 8) | slot[0x23]
            attack      = (slot[0x24] << 8) | slot[0x25]
            defense     = (slot[0x26] << 8) | slot[0x27]
            speed       = (slot[0x28] << 8) | slot[0x29]
            special     = (slot[0x2A] << 8) | slot[0x2B]

            nick_raw  = self._read_bytes(RED_ADDR["party_nicknames"] + i * 11, 11)
            nickname  = self._decode_gen1_text(nick_raw)

            # Use internal-ID → name mapping (internal ID ≠ Pokedex number in Gen 1)
            species_name = _SPECIES_NAMES.get(str(species_id), f"Species_{species_id}")
            type1_name   = _TYPE_NAMES.get(str(type1_id), f"Type_{type1_id}")
            type2_name   = _TYPE_NAMES.get(str(type2_id), f"Type_{type2_id}")
            # Monotype Pokemon store the same type in both slots; deduplicate
            types        = [type1_name] if type1_name == type2_name else [type1_name, type2_name]
            # Always return 4 move slots; use "NONE" for empty slots (matches move_pp length)
            move_names   = [
                _MOVE_NAMES.get(str(mid), "NONE") if mid != 0 else "NONE"
                for mid in move_ids
            ]

            party.append({
                "species_id":   species_id,
                "species_name": species_name,
                "nickname":     nickname,
                "level":        level,
                "current_hp":   current_hp,
                "max_hp":       max_hp,
                "status":       self._status_name(status),
                "types":        types,
                "attack":       attack,
                "defense":      defense,
                "speed":        speed,
                "special":      special,
                "moves":        move_names,
                "move_pp":      move_pp,
            })
        return party

    def read_badges(self) -> List[str]:
        """Return list of earned badge names (bit 0 = Boulder Badge)."""
        byte = self._read_u8(RED_ADDR["badges"])
        return [name for i, name in enumerate(RED_BADGE_NAMES) if byte & (1 << i)]

    def is_in_battle(self) -> bool:
        return self._read_u8(RED_ADDR["in_battle"]) != 0

    def is_in_dialog(self) -> bool:  # may DOUBLE CHECK
        """Two-signal dialog detection: text_progress register + VRAM border check."""
        if self.is_in_battle():
            return False
        # Signal 1: text progress register
        if self._read_u8(RED_ADDR["text_progress"]) == 0:
            return False
        # Signal 2: confirm via VRAM tilemap border tiles
        return self._has_dialog_box_borders()

    def _has_dialog_box_borders(self) -> bool:
        """Scan the bottom 5 rows of the VRAM tilemap for dialog box borders.

        The Gen 1 dialog box uses specific tile IDs for its borders.
        As a conservative initial implementation, return True whenever
        text_progress is non-zero (the calling method already checks that).
        Refine tile IDs against a live ROM dump if false positives appear.
        """
        try:
            tiles = self._read_bytes(RED_ADDR["vram_tilemap"], 360)
            # Dialog box occupies the bottom rows of the 20x18 screen.
            # Border tiles include 0x79 (─), 0x7A (│), 0x7B (┐), 0x7C (└) etc.
            # Check rows 13-17 (bottom 5 rows × 20 columns = bytes 260-359)
            border_tile_ids = {0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F}
            bottom_rows = tiles[260:]
            return any(t in border_tile_ids for t in bottom_rows)
        except Exception:
            # If VRAM read fails, trust text_progress alone
            return True

    def is_on_title_screen(self) -> bool:
        """Detect the title screen (title_check == 0x1F and title_map_id == 0x00)."""
        return (self._read_u8(RED_ADDR["title_check"]) == 0x1F and
                self._read_u8(RED_ADDR["title_map_id"]) == 0x00)

    def get_game_state(self) -> str:
        """Return game state string matching pyboy_runner: 'title', 'battle', 'dialog', or 'overworld'."""
        if self.is_on_title_screen():
            return "title"
        if self.is_in_battle():
            return "battle"
        if self.is_in_dialog():
            return "dialog"
        return "overworld"

    def read_direction(self) -> str:
        """Return player facing direction as a cardinal string."""
        byte = self._read_u8(RED_ADDR["player_dir"])
        return {0: "South", 4: "North", 8: "West", 12: "East"}.get(byte, "South")

    def read_item_count(self) -> int:
        return self._read_u8(RED_ADDR["inventory_count"])

    def read_items(self) -> List[Dict[str, Any]]:
        """Return list of {item_id, name, quantity} dicts from player's bag."""
        count = self.read_item_count()
        base = RED_ADDR["inventory_items"]
        items = []
        for i in range(min(count, 20)):  # Gen 1 max 20 item slots
            item_id = self._read_u8(base + i * 2)
            qty = self._read_u8(base + i * 2 + 1)
            if item_id == 0xFF:  # inventory terminator
                break
            items.append({
                "item_id":  item_id,
                "name":     _ITEM_NAMES.get(str(item_id), f"Item_{item_id}"),
                "quantity": qty,
            })
        return items

    def read_pokedex_caught_count(self) -> int:
        """Count bits set in the 19-byte 'owned' Pokedex bitfield."""
        raw = self._read_bytes(RED_ADDR["pokedex_owned"], 19)
        return sum(bin(b).count("1") for b in raw)

    def read_pokedex_seen_count(self) -> int:
        """Count bits set in the 19-byte 'seen' Pokedex bitfield."""
        raw = self._read_bytes(RED_ADDR["pokedex_seen"], 19)
        return sum(bin(b).count("1") for b in raw)

    # Box-drawing characters that form the Gen 1 dialog box border
    _BOX_BORDER_CHARS = frozenset("┌─┐│└┘")

    def read_screen_text(self) -> str:
        """Decode VRAM tilemap to readable dialog text.

        Strips the leading empty (overworld) rows and the box-drawing border
        that surrounds the Gen 1 dialog window, returning only the text lines.
        Input rows look like:
          (empty)×10, ┌──────────────────┐, │                  │,
          │Some NPC text here│, │                  │, └──────────────────┘
        Output: "Some NPC text here"
        """
        try:
            tiles = self._read_bytes(RED_ADDR["vram_tilemap"], 360)
            lines = []
            for row in range(18):
                row_bytes = tiles[row * 20: row * 20 + 20]
                decoded = self._decode_gen1_text(row_bytes)
                # Strip │ frame characters and surrounding whitespace
                stripped = decoded.strip()
                if stripped.startswith("│"):
                    stripped = stripped[1:]
                if stripped.endswith("│"):
                    stripped = stripped[:-1]
                stripped = stripped.strip()
                # Drop empty rows and pure border rows (only box-drawing chars)
                if stripped and not all(c in self._BOX_BORDER_CHARS for c in stripped):
                    lines.append(stripped)
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"read_screen_text failed: {e}")
            return ""

    def read_battle_details(self) -> Optional[Dict[str, Any]]:
        """Return battle info dict aligned with Emerald's structure, or None if not in battle."""
        battle_type = self._read_u8(RED_ADDR["in_battle"])
        if battle_type == 0:
            return None

        is_wild = (battle_type == 1)

        # Opponent (enemy) Pokemon from RAM (wEnemyMon struct @ 0xCFE5)
        enemy_species = self._read_u8(RED_ADDR["enemy_species"])
        enemy_hp      = self._read_u16_be(RED_ADDR["enemy_hp"])
        enemy_status  = self._read_u8(RED_ADDR["enemy_status"])
        enemy_level   = self._read_u8(RED_ADDR["enemy_level"])
        enemy_max_hp  = self._read_u16_be(RED_ADDR["enemy_max_hp"])
        enemy_type1   = self._read_u8(RED_ADDR["enemy_type1"])
        enemy_type2   = self._read_u8(RED_ADDR["enemy_type2"])
        enemy_attack  = self._read_u16_be(RED_ADDR["enemy_attack"])
        enemy_defense = self._read_u16_be(RED_ADDR["enemy_defense"])
        enemy_speed   = self._read_u16_be(RED_ADDR["enemy_speed"])
        enemy_special = self._read_u16_be(RED_ADDR["enemy_special"])
        enemy_move_ids = self._read_bytes(RED_ADDR["enemy_moves"], 4)
        enemy_pp_raw   = self._read_bytes(RED_ADDR["enemy_pp"], 4)

        etype1_name = _TYPE_NAMES.get(str(enemy_type1), f"Type_{enemy_type1}")
        etype2_name = _TYPE_NAMES.get(str(enemy_type2), f"Type_{enemy_type2}")
        enemy_types = [etype1_name] if etype1_name == etype2_name else [etype1_name, etype2_name]
        enemy_moves = [
            _MOVE_NAMES.get(str(mid), "NONE") if mid != 0 else "NONE"
            for mid in enemy_move_ids
        ]
        enemy_move_pp = list(enemy_pp_raw)

        # Player's active Pokemon: match name at 0xD009 against party nicknames/species names.
        # (same approach as pyboy_runner.py::get_active_pokemon_name)
        active_name_raw = self._read_bytes(RED_ADDR["active_pokemon_name"], 11)
        active_name = self._decode_gen1_text(active_name_raw)
        party = self.read_party_pokemon()
        player_mon = None
        if party:
            for slot in party:
                if slot["nickname"] == active_name or slot["species_name"] == active_name:
                    player_mon = slot
                    break
        # removed fallback for player_mon for test
        # if player_mon == None: player_mon = party[0]

        return {
            "in_battle":         True,
            "battle_type":       "wild" if is_wild else "trainer",
            "is_trainer_battle": not is_wild,
            "is_capturable":     is_wild,
            "can_escape":        is_wild,
            "player_pokemon":    player_mon,
            "opponent_pokemon": {
                "species_id":    enemy_species,
                "species":       _SPECIES_NAMES.get(str(enemy_species), f"Species_{enemy_species}"),
                "level":         enemy_level,
                "current_hp":    enemy_hp,
                "max_hp":        enemy_max_hp,
                "hp_percentage": round(enemy_hp / max(enemy_max_hp, 1) * 100, 1),
                "status":        self._status_name(enemy_status),
                "types":         enemy_types,
                "moves":         enemy_moves,
                "move_pp":       enemy_move_pp,
                "is_fainted":    enemy_hp == 0,
                "stats": {
                    "attack":    enemy_attack,
                    "defense":   enemy_defense,
                    "speed":     enemy_speed,
                    "special":   enemy_special,  # Gen 1: unified special (sp_atk = sp_def)
                },
            },
            "battle_interface": {
                "available_actions": ["FIGHT", "ITEM", "PKMN", "RUN"],
            },
        }

    def _build_progress_context(self, party: List[Dict], badges: List[str]) -> Dict[str, Any]:
        """Build progress_context dict mirroring EmeraldEmulator's shape.

        visited_locations is seeded with the current map location and then
        augmented by badge-based city inference (earned badge → city visited).
        """
        badge_byte = self._read_u8(RED_ADDR["badges"])
        # Numbered badge flags (badge_01..badge_08) matching Emerald's pattern
        badge_flags = {
            f"badge_{i + 1:02d}": bool(badge_byte & (1 << i))
            for i in range(8)
        }

        # Infer visited cities from earned badges
        _BADGE_LOCATIONS = [
            ("Boulder", "pewter_city"),
            ("Cascade", "cerulean_city"),
            ("Thunder", "vermilion_city"),
            ("Rainbow", "celadon_city"),
            ("Soul",    "fuchsia_city"),
            ("Marsh",   "saffron_city"),
            ("Volcano", "cinnabar_island"),
            ("Earth",   "viridian_city"),
        ]
        current_loc = self.read_location().lower().replace(" ", "_")
        visited_set: set = {f"visited_{current_loc}"}
        for badge_name, city in _BADGE_LOCATIONS:
            if badge_name in badges:
                visited_set.add(f"visited_{city}")
        visited_locations = sorted(visited_set)

        has_pokedex = (self.read_pokedex_caught_count() > 0
                       or self.read_pokedex_seen_count() > 0)
        is_champion = len(badges) == 8

        flags: Dict[str, Any] = {**badge_flags, "has_pokedex": has_pokedex, "is_champion": is_champion}
        for v in visited_locations:
            flags[v] = True

        return {
            "badges_obtained":   len(badges),
            "badge_names":       badges,
            "party_size":        len(party),
            "has_pokedex":       has_pokedex,
            "is_champion":       is_champion,
            "visited_locations": visited_locations,
            "flags":             flags,
            "party_levels":      [p["level"] for p in party],
            "party_species":     [p["species_name"] for p in party],
        }

    # ------------------------------------------------------------------
    # Comprehensive state aggregator (matches EmeraldEmulator dict shape)
    # ------------------------------------------------------------------

    def get_comprehensive_state(self, screenshot=None) -> Dict[str, Any]:
        """Return a state dict with the same four-key shape as EmeraldEmulator."""
        state: Dict[str, Any] = {
            "visual": {
                "screenshot": screenshot,
                "resolution": [160, 144],
            },
            "player": {
                "position": None,
                "location": None,
                "name":     None,
                "party":    None,
            },
            "game": {
                "money":        None,
                "party":        None,
                "game_state":   None,
                "is_in_battle": None,
                "time":         None,   # Gen 1 has no RTC
                "badges":       None,
                "items":        None,
                "item_count":   None,
                "pokedex_caught": None,
                "pokedex_seen":   None,
                "battle_info":    None,
                "dialog_text":    None,
                "progress_context": None,
                "dialogue_detected": None,
            },
            "map": {
                "tiles":              None,
                "tile_names":         None,
                "metatile_behaviors": None,
                "metatile_info":      None,
                "traversability":     None,
                "object_events":      [],
            },
        }

        try:
            coords = self.read_coordinates()
            state["player"]["position"] = {"x": coords[0], "y": coords[1]}
            state["player"]["location"] = self.read_location()
            state["player"]["name"]     = self.read_player_name()
            party = self.read_party_pokemon()
            state["player"]["party"]    = party

            in_battle   = self.is_in_battle()
            game_state  = self.get_game_state()

            if self._dialog_detection_enabled:
                in_dialog   = self.is_in_dialog() if not in_battle else False
                dialog_text = self.read_screen_text() if in_dialog else None
            else:
                in_dialog   = False
                dialog_text = None

            badges = self.read_badges()
            state["game"].update({
                "money":        self.read_money(),
                "party":        party,
                "game_state":   game_state,
                "is_in_battle": in_battle,
                "badges":       badges,
                "items":        self.read_items(),
                "item_count":   self.read_item_count(),
                "pokedex_caught": self.read_pokedex_caught_count(),
                "pokedex_seen":   self.read_pokedex_seen_count(),
                "battle_info":  self.read_battle_details() if in_battle else None,
                "dialog_text":  dialog_text,
                "progress_context": self._build_progress_context(party, badges),
                "dialogue_detected": {
                    "has_dialogue": in_dialog,
                    "confidence":   1.0 if in_dialog else 0.0,
                    "reason":       "gen1_text_progress + vram_border_check",
                },
            })
        except Exception as e:
            logger.warning(f"RedMemoryReader.get_comprehensive_state error: {e}")

        try:
            if self.map_reader is not None:
                tiles = self.read_map_around_player()
                if tiles:
                    tile_names = []
                    metatile_behaviors = []
                    metatile_info = []
                    for row in tiles:
                        row_names, row_behaviors, row_info = [], [], []
                        for tile_id, type_str, collision, elevation in row:
                            row_names.append(type_str)
                            row_behaviors.append(type_str)
                            row_info.append({
                                "id":                 tile_id,
                                "behavior":           type_str,
                                "collision":          collision,
                                "elevation":          elevation,
                                "passable":           collision == 0,
                                "encounter_possible": type_str in ("GRASS", "WATER"),
                                "surfable":           type_str == "WATER",
                            })
                        tile_names.append(row_names)
                        metatile_behaviors.append(row_behaviors)
                        metatile_info.append(row_info)

                    state["map"].update({
                        "tiles":              tiles,
                        "tile_names":         tile_names,
                        "metatile_behaviors": metatile_behaviors,
                        "metatile_info":      metatile_info,
                        "traversability":     self.map_reader.get_traversability_grid(),
                    })

                # Populate object_events from live RAM sprite data
                try:
                    sprites = self.map_reader.read_sprites()
                    player_x, player_y = self.map_reader.read_player_coords()
                    object_events = []
                    for s in sprites:
                        distance = abs(s['map_x'] - player_x) + abs(s['map_y'] - player_y)
                        object_events.append({
                            'id': s['slot'],
                            'local_id': s['slot'],
                            'current_x': s['map_x'],
                            'current_y': s['map_y'],
                            'facing': s['facing'],
                            'graphics_id': s['picture_id'],
                            'sprite_name': s['sprite_name'],
                            'distance': distance,
                            'source': f"sprite_slot_{s['slot']}_ram",
                        })
                    state["map"]["object_events"] = object_events
                except Exception as e:
                    logger.debug(f"Could not read sprites for object_events: {e}")
        except Exception as e:
            logger.warning(f"RedMemoryReader map state error: {e}")

        return state

    # ------------------------------------------------------------------
    # Map reader wiring
    # ------------------------------------------------------------------

    def set_map_reader(self, map_reader) -> None:
        """Attach a RedMapReader instance (called by RedEmulator.initialize())."""
        self.map_reader = map_reader

    def read_map_around_player(self, radius: int = 7) -> list:
        """Delegate to map_reader.read_map_around_player(radius), or return []."""
        if self.map_reader is not None:
            return self.map_reader.read_map_around_player(radius)
        return []

    # ------------------------------------------------------------------
    # Interface compatibility stubs
    # ------------------------------------------------------------------

    def _check_area_transition(self) -> bool:
        """Detect if player moved to a different map.

        Called every frame by server/app.py step_environment().
        Returns True on the frame where wCurMap changes.
        """
        current_map_id = self._read_u8(RED_ADDR["map_id"])
        if self._last_map_id is None:
            self._last_map_id = current_map_id
            return False
        if current_map_id != self._last_map_id:
            self._last_map_id = current_map_id
            return True
        return False

    def invalidate_map_cache(self, **kwargs) -> None:
        """No-op — processed_map data is static; no cache to invalidate."""
        pass

    def reset_dialog_tracking(self) -> None:
        """Reset any cached dialog state."""
        self._dialog_cache = False
        self._dialog_cache_time = 0.0

    def clear_dialogue_cache_on_button_press(self) -> None:
        """Called when 'A' is pressed to dismiss dialog; resets cache."""
        self._dialog_cache = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def test_memory_reading(self) -> Dict[str, Any]:
        """Read a handful of known Red addresses and return diagnostic info."""
        results: Dict[str, Any] = {}
        try:
            results["player_x"]    = self._read_u8(RED_ADDR["player_x"])
            results["player_y"]    = self._read_u8(RED_ADDR["player_y"])
            results["map_id"]      = self._read_u8(RED_ADDR["map_id"])
            results["map_name"]    = self.read_location()
            results["money"]       = self.read_money()
            results["party_count"] = self._read_u8(RED_ADDR["party_count"])
            results["badges_raw"]  = self._read_u8(RED_ADDR["badges"])
            results["badges"]      = self.read_badges()
            results["in_battle"]   = self.is_in_battle()
            results["is_in_dialog"]= self.is_in_dialog()
            results["player_name"] = self.read_player_name()
            results["status"]      = "ok"
        except Exception as e:
            results["status"] = "error"
            results["error"]  = str(e)
        return results

    def test_memory_access(self) -> Dict[str, Any]:
        """Alias for server /debug/memory endpoint compatibility."""
        return self.test_memory_reading()

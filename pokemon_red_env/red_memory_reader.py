"""Pokemon Red memory reader using PyBoy backend.

Mirrors the public interface of PokemonEmeraldReader so RedEmulator can be used
as a drop-in replacement for EmeraldEmulator by server/app.py and agent scaffolds.

All data sourced from the pokered decompilation (https://github.com/pret/pokered)
and validated against docs/pokemon_red_proposal.md.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAM address table (from pokered decompilation)
# ---------------------------------------------------------------------------
RED_ADDR: Dict[str, int] = {
    # Player
    "player_name":      0xD158,  # 11 bytes, Gen1 charset, 0x50 = terminator
    "rival_name":       0xD34A,  # 11 bytes
    "player_y":         0xD361,  # 1 byte tile coord
    "player_x":         0xD362,  # 1 byte tile coord
    "player_dir":       0xC109,  # 1 byte: 0=down, 4=up, 8=left, 12=right
    # Map
    "map_id":           0xD35E,  # 1 byte
    # Economy
    "money":            0xD347,  # 3 bytes BCD big-endian
    # Inventory
    "inventory_count":  0xD31C,  # 1 byte
    "inventory_items":  0xD31D,  # 2*count bytes: alternating (item_id, quantity)
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
    "enemy_species":    0xCFE5,  # 1 byte
    "enemy_hp":         0xCFE6,  # 2 bytes big-endian
    "enemy_level":      0xCFF3,  # 1 byte
    "enemy_max_hp":     0xCFF4,  # 2 bytes big-endian
    # Dialog / text
    "text_progress":    0xC6AC,  # 1 byte, non-zero = text printing active
    "vram_tilemap":     0xC3A0,  # 20×18 = 360 screen tiles
}

# Each party slot is 44 bytes.  Confirmed offsets (pokered decomp):
#   +0x00: species_id
#   +0x01–0x02: current_hp (big-endian)
#   +0x03: level (battle copy)
#   +0x04: status condition  (0=OK, 0x08=PSN, 0x10=BRN, 0x20=FRZ, 0x40=PAR, 0–7=SLP turns)
#   +0x05: type1
#   +0x06: type2
#   +0x08–0x0B: move IDs (4 × 1 byte)
#   +0x0D–0x0E: max_hp (big-endian)
#   +0x0F–0x10: attack (big-endian)
#   +0x11–0x12: defense (big-endian)
#   +0x13–0x14: speed (big-endian)
#   +0x15–0x16: special (big-endian)
#   +0x1D: level (actual / display)
#   +0x1E–0x21: PP for moves 1–4 (1 byte each)
PARTY_SLOT_SIZE = 0x2C  # 44 bytes

# ---------------------------------------------------------------------------
# Gen 1 character map
# ---------------------------------------------------------------------------
GEN1_CHARMAP: Dict[int, str] = {
    0x50: "",    # string terminator (handled by stopping loop)
    0x7F: " ",
    **{0x80 + i: chr(ord("A") + i) for i in range(26)},   # A–Z
    **{0xA0 + i: chr(ord("a") + i) for i in range(26)},   # a–z
    # Digits (0xF6 = ♂ conflicts; digits appear in item/move names but rarely in
    # player names.  Map them conservatively; override ♂/♀ below.)
    **{0xF6 + i: str(i) for i in range(10)},
    0xF5: "♀",
    0xF6: "♂",  # takes priority over digit '0' above — fine for names
    0xE0: "'",
    0xE8: ".",
    0xE9: ",",
    0xEA: "!",
    0xEB: "?",
    0xED: "-",
    0xF3: "×",
    0xF4: ".",
}

# ---------------------------------------------------------------------------
# Badge names (bit 0 = Boulder, bit 7 = Earth)
# ---------------------------------------------------------------------------
RED_BADGE_NAMES: List[str] = [
    "Boulder", "Cascade", "Thunder", "Rainbow",
    "Soul", "Marsh", "Volcano", "Earth",
]

# ---------------------------------------------------------------------------
# Gen 1 type names
# ---------------------------------------------------------------------------
GEN1_TYPES: Dict[int, str] = {
    0: "Normal", 1: "Fighting", 2: "Flying", 3: "Poison", 4: "Ground",
    5: "Rock", 7: "Bug", 8: "Ghost", 20: "Fire", 21: "Water",
    22: "Grass", 23: "Electric", 24: "Psychic", 25: "Ice",
    26: "Dragon",
}

# ---------------------------------------------------------------------------
# Gen 1 species names (151 Pokemon)
# ---------------------------------------------------------------------------
GEN1_SPECIES: Dict[int, str] = {
    1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur",
    4: "Charmander", 5: "Charmeleon", 6: "Charizard",
    7: "Squirtle", 8: "Wartortle", 9: "Blastoise",
    10: "Caterpie", 11: "Metapod", 12: "Butterfree",
    13: "Weedle", 14: "Kakuna", 15: "Beedrill",
    16: "Pidgey", 17: "Pidgeotto", 18: "Pidgeot",
    19: "Rattata", 20: "Raticate",
    21: "Spearow", 22: "Fearow",
    23: "Ekans", 24: "Arbok",
    25: "Pikachu", 26: "Raichu",
    27: "Sandshrew", 28: "Sandslash",
    29: "Nidoran♀", 30: "Nidorina", 31: "Nidoqueen",
    32: "Nidoran♂", 33: "Nidorino", 34: "Nidoking",
    35: "Clefairy", 36: "Clefable",
    37: "Vulpix", 38: "Ninetales",
    39: "Jigglypuff", 40: "Wigglytuff",
    41: "Zubat", 42: "Golbat",
    43: "Oddish", 44: "Gloom", 45: "Vileplume",
    46: "Paras", 47: "Parasect",
    48: "Venonat", 49: "Venomoth",
    50: "Diglett", 51: "Dugtrio",
    52: "Meowth", 53: "Persian",
    54: "Psyduck", 55: "Golduck",
    56: "Mankey", 57: "Primeape",
    58: "Growlithe", 59: "Arcanine",
    60: "Poliwag", 61: "Poliwhirl", 62: "Poliwrath",
    63: "Abra", 64: "Kadabra", 65: "Alakazam",
    66: "Machop", 67: "Machoke", 68: "Machamp",
    69: "Bellsprout", 70: "Weepinbell", 71: "Victreebel",
    72: "Tentacool", 73: "Tentacruel",
    74: "Geodude", 75: "Graveler", 76: "Golem",
    77: "Ponyta", 78: "Rapidash",
    79: "Slowpoke", 80: "Slowbro",
    81: "Magnemite", 82: "Magneton",
    83: "Farfetch'd",
    84: "Doduo", 85: "Dodrio",
    86: "Seel", 87: "Dewgong",
    88: "Grimer", 89: "Muk",
    90: "Shellder", 91: "Cloyster",
    92: "Gastly", 93: "Haunter", 94: "Gengar",
    95: "Onix",
    96: "Drowzee", 97: "Hypno",
    98: "Krabby", 99: "Kingler",
    100: "Voltorb", 101: "Electrode",
    102: "Exeggcute", 103: "Exeggutor",
    104: "Cubone", 105: "Marowak",
    106: "Hitmonlee", 107: "Hitmonchan",
    108: "Lickitung",
    109: "Koffing", 110: "Weezing",
    111: "Rhyhorn", 112: "Rhydon",
    113: "Chansey",
    114: "Tangela",
    115: "Kangaskhan",
    116: "Horsea", 117: "Seadra",
    118: "Goldeen", 119: "Seaking",
    120: "Staryu", 121: "Starmie",
    122: "Mr. Mime",
    123: "Scyther",
    124: "Jynx",
    125: "Electabuzz",
    126: "Magmar",
    127: "Pinsir",
    128: "Tauros",
    129: "Magikarp", 130: "Gyarados",
    131: "Lapras",
    132: "Ditto",
    133: "Eevee", 134: "Vaporeon", 135: "Jolteon", 136: "Flareon",
    137: "Porygon",
    138: "Omanyte", 139: "Omastar",
    140: "Kabuto", 141: "Kabutops",
    142: "Aerodactyl",
    143: "Snorlax",
    144: "Articuno", 145: "Zapdos", 146: "Moltres",
    147: "Dratini", 148: "Dragonair", 149: "Dragonite",
    150: "Mewtwo",
    151: "Mew",
}

# ---------------------------------------------------------------------------
# Map ID → name (partial; extend from pokered/data/maps/mapHeaders.asm)
# ---------------------------------------------------------------------------
RED_MAP_NAMES: Dict[int, str] = {
    # Towns / cities
    0x00: "PALLET_TOWN",
    0x01: "VIRIDIAN_CITY",
    0x02: "PEWTER_CITY",
    0x03: "CERULEAN_CITY",
    0x04: "LAVENDER_TOWN",
    0x05: "VERMILION_CITY",
    0x06: "CELADON_CITY",
    0x07: "FUCHSIA_CITY",
    0x08: "CINNABAR_ISLAND",
    0x09: "INDIGO_PLATEAU",
    0x0A: "SAFFRON_CITY",
    # Routes
    0x0C: "ROUTE_1",  0x0D: "ROUTE_2",  0x0E: "ROUTE_3",  0x0F: "ROUTE_4",
    0x10: "ROUTE_5",  0x11: "ROUTE_6",  0x12: "ROUTE_7",  0x13: "ROUTE_8",
    0x14: "ROUTE_9",  0x15: "ROUTE_10", 0x16: "ROUTE_11", 0x17: "ROUTE_12",
    0x18: "ROUTE_13", 0x19: "ROUTE_14", 0x1A: "ROUTE_15", 0x1B: "ROUTE_16",
    0x1C: "ROUTE_17", 0x1D: "ROUTE_18", 0x1E: "ROUTE_19", 0x1F: "ROUTE_20",
    0x20: "ROUTE_21", 0x21: "ROUTE_22", 0x22: "ROUTE_23",
    0x23: "ROUTE_24", 0x24: "ROUTE_25",
    # Buildings
    0x25: "PALLET_TOWN_PLAYER_HOUSE_1F",
    0x26: "PALLET_TOWN_RIVAL_HOUSE",
    0x27: "OAKS_LAB",
    0x28: "VIRIDIAN_CITY_GYM",
    0x29: "PEWTER_CITY_GYM",
    0x2C: "CERULEAN_CITY_GYM",
    0x2E: "VERMILION_CITY_GYM",
    0x33: "CELADON_CITY_GYM",
    0x34: "FUCHSIA_CITY_GYM",
    0x35: "SAFFRON_CITY_GYM",
    0x38: "CINNABAR_ISLAND_GYM",
    0x39: "SILPH_CO_1F",
    # Dungeons
    0xE2: "SS_ANNE_1F",
    0xE9: "VIRIDIAN_FOREST",
    0xED: "ROCK_TUNNEL_1F",
    0xEE: "ROCK_TUNNEL_2F",
    0xF3: "SAFARI_ZONE_CENTER",
    0xF4: "SAFARI_ZONE_EAST",
    0xF5: "MT_MOON_1F",
    0xF6: "MT_MOON_2F",
    0xF7: "MT_MOON_3F",
    # Elite Four
    0xDE: "LORELEIS_ROOM",
    0xDF: "BRUNOS_ROOM",
    0xE0: "AGATHAS_ROOM",
    0xE1: "LANCES_ROOM",
    0xE3: "CHAMPIONS_ROOM",
    # TODO: extend from pokered/data/maps/mapHeaders.asm (~250 total entries)
}


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
            if b == 0x50:
                break
            ch = GEN1_CHARMAP.get(b)
            if ch is None:
                ch = "?"
                logger.debug(f"Unknown Gen1 char byte: 0x{b:02X}")
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
        return self._decode_gen1_text(raw) or "RED"

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
        return RED_MAP_NAMES.get(map_id, f"MAP_{map_id}")

    def read_party_size(self) -> int:
        return min(self._read_u8(RED_ADDR["party_count"]), 6)

    def read_party_pokemon(self) -> List[Dict[str, Any]]:
        """Parse all party Pokemon from RAM. Returns a list of dicts."""
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
            max_hp      = (slot[0x0D] << 8) | slot[0x0E]
            attack      = (slot[0x0F] << 8) | slot[0x10]
            defense     = (slot[0x11] << 8) | slot[0x12]
            speed       = (slot[0x13] << 8) | slot[0x14]
            special     = (slot[0x15] << 8) | slot[0x16]
            level       = slot[0x1D]
            move_pp     = [slot[0x1E], slot[0x1F], slot[0x20], slot[0x21]]

            nick_raw  = self._read_bytes(RED_ADDR["party_nicknames"] + i * 11, 11)
            nickname  = self._decode_gen1_text(nick_raw)

            party.append({
                "species_id":   species_id,
                "species":      GEN1_SPECIES.get(species_id, f"Species_{species_id}"),
                "species_name": GEN1_SPECIES.get(species_id, f"Species_{species_id}"),
                "nickname":     nickname,
                "level":        level,
                "current_hp":   current_hp,
                "max_hp":       max_hp,
                "status":       self._status_name(status),
                "type1":        GEN1_TYPES.get(type1_id, f"Type_{type1_id}"),
                "type2":        GEN1_TYPES.get(type2_id, f"Type_{type2_id}"),
                "attack":       attack,
                "defense":      defense,
                "speed":        speed,
                "special":      special,
                "moves":        [f"Move_{mid}" for mid in move_ids if mid != 0],
                "move_ids":     move_ids,
                "move_pp":      move_pp,
            })
        return party

    def read_badges(self) -> List[str]:
        """Return list of earned badge names (bit 0 = Boulder Badge)."""
        byte = self._read_u8(RED_ADDR["badges"])
        return [name for i, name in enumerate(RED_BADGE_NAMES) if byte & (1 << i)]

    def is_in_battle(self) -> bool:
        return self._read_u8(RED_ADDR["in_battle"]) != 0

    def is_in_dialog(self) -> bool:
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

    def get_game_state(self) -> str:
        """Return game state string: 'overworld', 'battle', or 'dialog'."""
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

    def read_items(self) -> List[Tuple[int, int]]:
        """Return list of (item_id, quantity) tuples from player's bag."""
        count = self.read_item_count()
        base = RED_ADDR["inventory_items"]
        items = []
        for i in range(min(count, 20)):  # Gen 1 max 20 item slots
            item_id = self._read_u8(base + i * 2)
            qty = self._read_u8(base + i * 2 + 1)
            if item_id == 0xFF:  # inventory terminator
                break
            items.append((item_id, qty))
        return items

    def read_pokedex_caught_count(self) -> int:
        """Count bits set in the 19-byte 'owned' Pokedex bitfield."""
        raw = self._read_bytes(RED_ADDR["pokedex_owned"], 19)
        return sum(bin(b).count("1") for b in raw)

    def read_pokedex_seen_count(self) -> int:
        """Count bits set in the 19-byte 'seen' Pokedex bitfield."""
        raw = self._read_bytes(RED_ADDR["pokedex_seen"], 19)
        return sum(bin(b).count("1") for b in raw)

    def read_screen_text(self) -> str:
        """Decode VRAM tilemap to a string (useful for reading dialog/menus)."""
        try:
            tiles = self._read_bytes(RED_ADDR["vram_tilemap"], 360)
            # Each row is 20 tiles wide
            rows = []
            for row in range(18):
                row_bytes = tiles[row * 20: row * 20 + 20]
                rows.append(self._decode_gen1_text(row_bytes))
            return "\n".join(rows)
        except Exception as e:
            logger.warning(f"read_screen_text failed: {e}")
            return ""

    def read_battle_details(self) -> Optional[Dict[str, Any]]:
        """Return enemy Pokemon battle info, or None if not in battle."""
        battle_type = self._read_u8(RED_ADDR["in_battle"])
        if battle_type == 0:
            return None
        enemy_species = self._read_u8(RED_ADDR["enemy_species"])
        enemy_hp      = self._read_u16_be(RED_ADDR["enemy_hp"])
        enemy_level   = self._read_u8(RED_ADDR["enemy_level"])
        enemy_max_hp  = self._read_u16_be(RED_ADDR["enemy_max_hp"])
        return {
            "in_battle":    True,
            "battle_type":  "wild" if battle_type == 1 else "trainer",
            "opponent": {
                "species_id":   enemy_species,
                "species":      GEN1_SPECIES.get(enemy_species, f"Species_{enemy_species}"),
                "level":        enemy_level,
                "current_hp":   enemy_hp,
                "max_hp":       enemy_max_hp,
                "hp_percentage": round(enemy_hp / max(enemy_max_hp, 1) * 100, 1),
            },
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
                "dialogue_detected": None,
            },
            "map": {
                "tiles":              None,
                "tile_names":         None,
                "metatile_behaviors": None,
                "metatile_info":      None,
                "traversability":     None,
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
            in_dialog   = self.is_in_dialog() if not in_battle else False
            game_state  = self.get_game_state()

            state["game"].update({
                "money":        self.read_money(),
                "party":        party,
                "game_state":   game_state,
                "is_in_battle": in_battle,
                "badges":       self.read_badges(),
                "items":        self.read_items(),
                "item_count":   self.read_item_count(),
                "pokedex_caught": self.read_pokedex_caught_count(),
                "pokedex_seen":   self.read_pokedex_seen_count(),
                "battle_info":  self.read_battle_details() if in_battle else None,
                "dialogue_detected": {
                    "has_dialogue": in_dialog,
                    "confidence":   1.0 if in_dialog else 0.0,
                    "reason":       "gen1_text_progress + vram_border_check",
                },
            })
        except Exception as e:
            logger.warning(f"RedMemoryReader.get_comprehensive_state error: {e}")

        return state

    # ------------------------------------------------------------------
    # Interface compatibility stubs
    # ------------------------------------------------------------------

    def invalidate_map_cache(self, **kwargs) -> None:
        """No-op — Gen 1 map system not yet implemented."""
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

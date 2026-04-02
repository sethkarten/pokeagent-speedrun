# This file is taken from https://github.com/dvruette/pygba/blob/main/pygba/utils.py
# and modified to work with the Pokemon Emerald game.

import functools
import struct
from collections import namedtuple

from pokemon_env.enums import Move, PokemonType, StatusCondition
from pokemon_env.types import PokemonData


class BaseCharmap:
    charmap: list[str]
    terminator: int

    def decode(self, chars: bytes) -> str:
        string = ""
        for i in range(len(chars)):
            if chars[i] == self.terminator:
                break
            string += self.charmap[chars[i]]
        return string

class AsciiCharmap(BaseCharmap):
    charmap = [
        "", "", "", "", "", "", "", "", "\r", "\t", " ", " ", " ", "\n", "", "",
        "", "", "", "", "", "", "", "", "", "", "SUB", "ESC", "", "", "", "",
        " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1",
        "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A",
        "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
        "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a",
        "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
        "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL", "Ç", "ü",
        "é", "â", "ä", "à", "å", "ç", "ê", "ë", "è", "ï", "î", "ì", "Ä", "Å", "É", "æ",
        "Æ", "ô", "ö", "ò", "û", "ù", "ÿ", "Ö", "Ü", "ø", "£", "Ø", "×", "ƒ", "á", "í",
        "ó", "ú", "ñ", "Ñ", "ª", "º", "¿", "®", "¬", "½", "¼", "¡", "«", "»", "░", "▒",
        "▓", "│", "┤", "Á", "Â", "À", "©", "╣", "║", "╗", "╝", "¢", "¥", "┐", "└", "┴",
        "┬", "├", "─", "┼", "ã", "Ã", "╚", "╔", "╩", "╦", "╠", "═", "╬", "¤", "ð", "Ð",
        "Ê", "Ë", "È", "ı", "Í", "Î", "Ï", "┘", "┌", "█", "▄", "¦", "Ì", "▀", "Ó", "ß",
        "Ô", "Ò", "õ", "Õ", "µ", "þ", "Þ", "Ú", "Û", "Ù", "ý", "Ý", "¯", "´", "¬", "±",
        "‗", "¾", "¶", "§", "÷", "¸", "°", "¨", "•", "¹", "³", "²", "■", "\u00a0",
    ]
    terminator = 0x00



# Pokemon Emerald Sym Addresses
# https://raw.githubusercontent.com/pret/pokeemerald/symbols/pokeemerald.sym

ADDRESSES = {
    "gPlayerPartyCount":            0x020244e9,
    "gPlayerParty":                 0x020244ec,
    "gSaveBlock1Ptr":               0x03005d8c,
    "gSaveBlock2Ptr":               0x03005d90,
    "gPokemonStoragePtr":           0x03005d94,
    "gSpeciesNames":                0x083185c8,
    "sSpeciesToHoennPokedexNum":    0x0831d94c,
    "sSpeciesToNationalPokedexNum": 0x0831dc82,
    "sHoennToNationalOrder":        0x0831dfb8,
    "gExperienceTables":            0x0831f72c,
    "gSpeciesInfo":                 0x083203cc,
    "gItems":                       0x085839a0,
}


# Struct layouts and constants taken from pret/pokeemerald:
# https://github.com/pret/pokeemerald/blob/master/include/pokemon.h
# https://github.com/pret/pokeemerald/blob/master/include/global.h


## Constants

POKEMON_NAME_LENGTH = 10
PLAYER_NAME_LENGTH = 7
PC_ITEMS_COUNT = 50
BAG_ITEMS_COUNT = 30
BAG_KEYITEMS_COUNT = 30
BAG_POKEBALLS_COUNT = 16
BAG_TMHM_COUNT = 64
BAG_BERRIES_COUNT = 46

NUM_SPECIES = 412
NUM_DEX_FLAG_BYTES = (NUM_SPECIES + 7) // 8

TOTAL_BOXES_COUNT = 14

# ROM species_id -> species name (pokeemerald include/constants/species.h numbering).
# Used to derive canonical species name from ROM data; nickname is stored separately.
ROM_SPECIES_ID_TO_NAME = {
    277: "Treecko", 278: "Grovyle", 279: "Sceptile",
    280: "Torchic", 281: "Combusken", 282: "Blaziken",
    283: "Mudkip", 284: "Marshtomp", 285: "Swampert",
    286: "Poochyena", 287: "Mightyena", 288: "Zigzagoon", 289: "Linoone",
    290: "Wurmple", 291: "Silcoon", 292: "Beautifly", 293: "Cascoon", 294: "Dustox",
    295: "Lotad", 296: "Lombre", 297: "Ludicolo",
    298: "Seedot", 299: "Nuzleaf", 300: "Shiftry",
    301: "Nincada", 302: "Ninjask", 303: "Shedinja",
    304: "Taillow", 305: "Swellow", 306: "Shroomish", 307: "Breloom", 308: "Spinda",
    309: "Wingull", 310: "Pelipper", 311: "Surskit", 312: "Masquerain",
    313: "Wailmer", 314: "Wailord", 315: "Skitty", 316: "Delcatty", 317: "Kecleon",
    318: "Baltoy", 319: "Claydol", 320: "Nosepass", 321: "Torkoal",
    322: "Sableye", 323: "Barboach", 324: "Whiscash", 325: "Luvdisc",
    326: "Corphish", 327: "Crawdaunt", 328: "Feebas", 329: "Milotic",
    330: "Carvanha", 331: "Sharpedo", 332: "Trapinch", 333: "Vibrava", 334: "Flygon",
    335: "Makuhita", 336: "Hariyama", 337: "Electrike", 338: "Manectric",
    339: "Numel", 340: "Camerupt", 341: "Spheal", 342: "Sealeo", 343: "Walrein",
    344: "Cacnea", 345: "Cacturne", 346: "Snorunt", 347: "Glalie",
    348: "Lunatone", 349: "Solrock", 350: "Azurill", 351: "Spoink", 352: "Grumpig",
    353: "Plusle", 354: "Minun", 355: "Mawile", 356: "Meditite", 357: "Medicham",
    358: "Swablu", 359: "Altaria", 360: "Wynaut", 361: "Duskull", 362: "Dusclops",
    363: "Roselia", 364: "Slakoth", 365: "Vigoroth", 366: "Slaking",
    367: "Gulpin", 368: "Swalot", 369: "Tropius", 370: "Whismur", 371: "Loudred",
    372: "Exploud", 373: "Clamperl", 374: "Huntail", 375: "Gorebyss",
    376: "Absol", 377: "Shuppet", 378: "Banette", 379: "Seviper", 380: "Zangoose",
    381: "Relicanth", 382: "Aron", 383: "Lairon", 384: "Aggron",
    385: "Castform", 386: "Volbeat", 387: "Illumise", 388: "Lileep", 389: "Cradily",
    390: "Anorith", 391: "Armaldo", 392: "Ralts", 393: "Kirlia", 394: "Gardevoir",
    395: "Bagon", 396: "Shelgon", 397: "Salamence", 398: "Beldum", 399: "Metang",
    400: "Metagross", 401: "Regirock", 402: "Regice", 403: "Registeel",
    404: "Kyogre", 405: "Groudon", 406: "Rayquaza", 407: "Latias", 408: "Latios",
    409: "Jirachi", 410: "Deoxys", 411: "Chimecho",
}


def _species_id_to_name(species_id: int) -> str:
    """Derive canonical species name from ROM species_id (pokeemerald numbering)."""
    return ROM_SPECIES_ID_TO_NAME.get(species_id, f"Species_{species_id}")


# ROM item_id -> human-readable item name (pokeemerald include/constants/items.h numbering).
# Keep this lookup lightweight and deterministic for hot-path game state reads.
ITEM_ID_TO_NAME = {
    # Balls
    1: "Master Ball", 2: "Ultra Ball", 3: "Great Ball", 4: "Poke Ball",
    5: "Safari Ball", 6: "Net Ball", 7: "Dive Ball", 8: "Nest Ball",
    9: "Repeat Ball", 10: "Timer Ball", 11: "Luxury Ball", 12: "Premier Ball",
    # Medicine and utility
    13: "Potion", 14: "Antidote", 15: "Burn Heal", 16: "Ice Heal",
    17: "Awakening", 18: "Paralyze Heal", 19: "Full Restore", 20: "Max Potion",
    21: "Hyper Potion", 22: "Super Potion", 23: "Full Heal", 24: "Revive",
    25: "Max Revive", 26: "Fresh Water", 27: "Soda Pop", 28: "Lemonade",
    29: "Moomoo Milk", 30: "Energy Powder", 31: "Energy Root", 32: "Heal Powder",
    33: "Revival Herb", 34: "Ether", 35: "Max Ether", 36: "Elixir",
    37: "Max Elixir", 38: "Lava Cookie", 39: "Blue Flute", 40: "Yellow Flute",
    41: "Red Flute", 42: "Black Flute", 43: "White Flute", 44: "Berry Juice",
    45: "Sacred Ash", 46: "Shoal Salt", 47: "Shoal Shell", 48: "Red Shard",
    49: "Blue Shard", 50: "Yellow Shard", 51: "Green Shard",
    63: "HP Up", 64: "Protein", 65: "Iron", 66: "Carbos",
    67: "Calcium", 68: "Rare Candy", 69: "PP Up", 70: "Zinc",
    71: "PP Max", 73: "Guard Spec.", 74: "Dire Hit", 75: "X Attack",
    76: "X Defend", 77: "X Speed", 78: "X Accuracy", 79: "X Special",
    80: "Poke Doll", 81: "Fluffy Tail", 83: "Super Repel", 84: "Max Repel",
    85: "Escape Rope", 86: "Repel", 93: "Sun Stone", 94: "Moon Stone",
    95: "Fire Stone", 96: "Thunder Stone", 97: "Water Stone", 98: "Leaf Stone",
    103: "Tiny Mushroom", 104: "Big Mushroom", 106: "Pearl", 107: "Big Pearl",
    108: "Stardust", 109: "Star Piece", 110: "Nugget", 111: "Heart Scale",
    # Mail
    121: "Orange Mail", 122: "Harbor Mail", 123: "Glitter Mail", 124: "Mech Mail",
    125: "Wood Mail", 126: "Wave Mail", 127: "Bead Mail", 128: "Shadow Mail",
    129: "Tropic Mail", 130: "Dream Mail", 131: "Fab Mail", 132: "Retro Mail",
    # Berries
    133: "Cheri Berry", 134: "Chesto Berry", 135: "Pecha Berry", 136: "Rawst Berry",
    137: "Aspear Berry", 138: "Leppa Berry", 139: "Oran Berry", 140: "Persim Berry",
    141: "Lum Berry", 142: "Sitrus Berry", 143: "Figy Berry", 144: "Wiki Berry",
    145: "Mago Berry", 146: "Aguav Berry", 147: "Iapapa Berry", 148: "Razz Berry",
    149: "Bluk Berry", 150: "Nanab Berry", 151: "Wepear Berry", 152: "Pinap Berry",
    153: "Pomeg Berry", 154: "Kelpsy Berry", 155: "Qualot Berry", 156: "Hondew Berry",
    157: "Grepa Berry", 158: "Tamato Berry", 159: "Cornn Berry", 160: "Magost Berry",
    161: "Rabuta Berry", 162: "Nomel Berry", 163: "Spelon Berry", 164: "Pamtre Berry",
    165: "Watmel Berry", 166: "Durin Berry", 167: "Belue Berry", 168: "Liechi Berry",
    169: "Ganlon Berry", 170: "Salac Berry", 171: "Petaya Berry", 172: "Apicot Berry",
    173: "Lansat Berry", 174: "Starf Berry", 175: "Enigma Berry",
    # Held items
    179: "Bright Powder", 180: "White Herb", 181: "Macho Brace", 182: "Exp. Share",
    183: "Quick Claw", 184: "Soothe Bell", 185: "Mental Herb", 186: "Choice Band",
    187: "King's Rock", 188: "SilverPowder", 189: "Amulet Coin", 190: "Cleanse Tag",
    191: "Soul Dew", 192: "Deep Sea Tooth", 193: "Deep Sea Scale", 194: "Smoke Ball",
    195: "Everstone", 196: "Focus Band", 197: "Lucky Egg", 198: "Scope Lens",
    199: "Metal Coat", 200: "Leftovers", 201: "Dragon Scale", 202: "Light Ball",
    203: "Soft Sand", 204: "Hard Stone", 205: "Miracle Seed", 206: "Black Glasses",
    207: "Black Belt", 208: "Magnet", 209: "Mystic Water", 210: "Sharp Beak",
    211: "Poison Barb", 212: "NeverMeltIce", 213: "Spell Tag", 214: "TwistedSpoon",
    215: "Charcoal", 216: "Dragon Fang", 217: "Silk Scarf", 218: "Up-Grade",
    219: "Shell Bell", 220: "Sea Incense", 221: "Lax Incense", 222: "Lucky Punch",
    223: "Metal Powder", 224: "Thick Club", 225: "Stick",
    # Contest scarves
    254: "Red Scarf", 255: "Blue Scarf", 256: "Pink Scarf", 257: "Green Scarf",
    258: "Yellow Scarf",
    # Key items
    259: "Mach Bike", 260: "Coin Case", 261: "Itemfinder", 262: "Old Rod",
    263: "Good Rod", 264: "Super Rod", 265: "S.S. Ticket", 266: "Contest Pass",
    268: "Wailmer Pail", 269: "Devon Goods", 270: "Soot Sack", 271: "Basement Key",
    272: "Acro Bike", 273: "Pokeblock Case", 274: "Letter", 275: "Eon Ticket",
    276: "Red Orb", 277: "Blue Orb", 278: "Scanner", 279: "Go-Goggles",
    280: "Meteorite", 281: "Room 1 Key", 282: "Room 2 Key", 283: "Room 4 Key",
    284: "Room 6 Key", 285: "Storage Key", 286: "Root Fossil", 287: "Claw Fossil",
    288: "Devon Scope",
    # Emerald exclusives
    375: "Magma Emblem", 376: "Old Sea Map",
}


def _item_id_to_name(item_id: int) -> str:
    """Derive item display name from ROM item_id (pokeemerald numbering)."""
    if 289 <= item_id <= 338:
        return f"TM{item_id - 288:02d}"
    if 339 <= item_id <= 346:
        return f"HM{item_id - 338:02d}"
    return ITEM_ID_TO_NAME.get(item_id, f"Item_{item_id}")


IN_BOX_COUNT = 30
BOX_NAME_LENGTH = 8


## Flag IDs

SCRIPT_FLAGS_START =                0x50
TRAINER_FLAGS_START =               0x500
SYSTEM_FLAGS_START =                0x860
DAILY_FLAGS_START =                 0x920

FLAG_DEFEATED_RUSTBORO_GYM =        0x4F0
FLAG_DEFEATED_DEWFORD_GYM =         0x4F1
FLAG_DEFEATED_MAUVILLE_GYM =        0x4F2
FLAG_DEFEATED_LAVARIDGE_GYM =       0x4F3
FLAG_DEFEATED_PETALBURG_GYM =       0x4F4
FLAG_DEFEATED_FORTREE_GYM =         0x4F5
FLAG_DEFEATED_MOSSDEEP_GYM =        0x4F6
FLAG_DEFEATED_SOOTOPOLIS_GYM =      0x4F7
FLAG_DEFEATED_METEOR_FALLS_STEVEN = 0x4F8

FLAG_DEFEATED_ELITE_4_SIDNEY =      0x4FB
FLAG_DEFEATED_ELITE_4_PHOEBE =      0x4FC
FLAG_DEFEATED_ELITE_4_GLACIA =      0x4FD
FLAG_DEFEATED_ELITE_4_DRAKE =       0x4FE


FLAG_SYS_POKEMON_GET =              SYSTEM_FLAGS_START + 0x0
FLAG_SYS_POKEDEX_GET =              SYSTEM_FLAGS_START + 0x1
FLAG_SYS_POKENAV_GET =              SYSTEM_FLAGS_START + 0x2
FLAG_RECEIVED_POKEDEX_FROM_BIRCH =  SYSTEM_FLAGS_START + 0x84

FLAG_BADGE01_GET =                  SYSTEM_FLAGS_START + 0x7
FLAG_BADGE02_GET =                  SYSTEM_FLAGS_START + 0x8
FLAG_BADGE03_GET =                  SYSTEM_FLAGS_START + 0x9
FLAG_BADGE04_GET =                  SYSTEM_FLAGS_START + 0xa
FLAG_BADGE05_GET =                  SYSTEM_FLAGS_START + 0xb
FLAG_BADGE06_GET =                  SYSTEM_FLAGS_START + 0xc
FLAG_BADGE07_GET =                  SYSTEM_FLAGS_START + 0xd
FLAG_BADGE08_GET =                  SYSTEM_FLAGS_START + 0xe

FLAG_VISITED_LITTLEROOT_TOWN =      SYSTEM_FLAGS_START + 0xF
FLAG_VISITED_OLDALE_TOWN =          SYSTEM_FLAGS_START + 0x10
FLAG_VISITED_DEWFORD_TOWN =         SYSTEM_FLAGS_START + 0x11
FLAG_VISITED_LAVARIDGE_TOWN =       SYSTEM_FLAGS_START + 0x12
FLAG_VISITED_FALLARBOR_TOWN =       SYSTEM_FLAGS_START + 0x13
FLAG_VISITED_VERDANTURF_TOWN =      SYSTEM_FLAGS_START + 0x14
FLAG_VISITED_PACIFIDLOG_TOWN =      SYSTEM_FLAGS_START + 0x15
FLAG_VISITED_PETALBURG_CITY =       SYSTEM_FLAGS_START + 0x16
FLAG_VISITED_SLATEPORT_CITY =       SYSTEM_FLAGS_START + 0x17
FLAG_VISITED_MAUVILLE_CITY =        SYSTEM_FLAGS_START + 0x18
FLAG_VISITED_RUSTBORO_CITY =        SYSTEM_FLAGS_START + 0x19
FLAG_VISITED_FORTREE_CITY =         SYSTEM_FLAGS_START + 0x1A
FLAG_VISITED_LILYCOVE_CITY =        SYSTEM_FLAGS_START + 0x1B
FLAG_VISITED_MOSSDEEP_CITY =        SYSTEM_FLAGS_START + 0x1C
FLAG_VISITED_SOOTOPOLIS_CITY =      SYSTEM_FLAGS_START + 0x1D
FLAG_VISITED_EVER_GRANDE_CITY =     SYSTEM_FLAGS_START + 0x1E

FLAG_IS_CHAMPION =                  SYSTEM_FLAGS_START + 0x1F




class EmeraldCharmap(BaseCharmap):
    charmap = [
        " ", "À", "Á", "Â", "Ç", "È", "É", "Ê", "Ë", "Ì", "こ", "Î", "Ï", "Ò", "Ó", "Ô",
        "Œ", "Ù", "Ú", "Û", "Ñ", "ß", "à", "á", "ね", "ç", "è", "é", "ê", "ë", "ì", "ま",
        "î", "ï", "ò", "ó", "ô", "œ", "ù", "ú", "û", "ñ", "º", "ª", "�", "&", "+", "あ",
        "ぃ", "ぅ", "ぇ", "ぉ", "Lv", "=", ";", "が", "ぎ", "ぐ", "げ", "ご", "ざ", "じ", "ず", "ぜ",
        "ぞ", "だ", "ぢ", "づ", "で", "ど", "ば", "び", "ぶ", "べ", "ぼ", "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",
        "っ", "¿", "¡", "P\u200dk", "M\u200dn", "P\u200do", "K\u200dé", "B\u200dL", "O\u200dC", "\u200dK", "Í", "%", "(", ")", "セ", "ソ",
        "タ", "チ", "ツ", "テ", "ト", "ナ", "ニ", "ヌ", "â", "ノ", "ハ", "ヒ", "フ", "ヘ", "ホ", "í",
        "ミ", "ム", "メ", "モ", "ヤ", "ユ", "ヨ", "ラ", "リ", "⬆", "⬇", "⬅", "➡", "ヲ", "ン", "ァ",
        "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ガ", "ギ", "グ", "ゲ", "ゴ", "ザ", "ジ", "ズ", "ゼ",
        "ゾ", "ダ", "ヂ", "ヅ", "デ", "ド", "バ", "ビ", "ブ", "ベ", "ボ", "パ", "ピ", "プ", "ペ", "ポ",
        "ッ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "?", ".", "-", "・",
        "…", "“", "”", "‘", "’", "♂", "♀", "$", ",", "×", "/", "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
        "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "▶",
        ":", "Ä", "Ö", "Ü", "ä", "ö", "ü", "⬆", "⬇", "⬅", "�", "�", "�", "�", "�", "",
    ]
    terminator = 0xFF


PokemonSubstruct0_spec = (
    ("species", "H"),
    ("heldItem", "H"),
    ("experience", "I"),
    ("ppBonuses", "B"),
    ("friendship", "B"),
    ("unknown", "H"),
)
PokemonSubstruct0 = namedtuple("PokemonSubstruct0", [x[0] for x in PokemonSubstruct0_spec])
PokemonSubstruct0_format = "".join([x[1] for x in PokemonSubstruct0_spec])

PokemonSubstruct1_spec = None
PokemonSubstruct1 = namedtuple("PokemonSubstruct1", ("moves", "pp"))
PokemonSubstruct1_format = "4H4B"

PokemonSubstruct2_spec = (
    ("hpEV", "B"),
    ("attackEV", "B"),
    ("defenseEV", "B"),
    ("speedEV", "B"),
    ("spAttackEV", "B"),
    ("spDefenseEV", "B"),
    ("cool", "B"),
    ("beauty", "B"),
    ("cute", "B"),
    ("smart", "B"),
    ("tough", "B"),
    ("sheen", "B"),
)
PokemonSubstruct2 = namedtuple("PokemonSubstruct2", [x[0] for x in PokemonSubstruct2_spec])
PokemonSubstruct2_format = "".join([x[1] for x in PokemonSubstruct2_spec])

PokemonSubstruct3_spec = None
PokemonSubstruct3_format = "III"
PokemonSubstruct3 = namedtuple("PokemonSubstruct3", (
    "pokerus",
    "metLocation",
    "metLevel",
    "metGame",
    "pokeball",
    "otGender",
    "hpIV",
    "attackIV",
    "defenseIV",
    "speedIV",
    "spAttackIV",
    "spDefenseIV",
    "isEgg",
    "abilityNum",
    "ribbons",
))


BoxPokemon_spec = (
    ("personality", "I"),
    ("otId", "I"),
    ("nickname", f"{POKEMON_NAME_LENGTH}s"),
    ("language", "B"),
    ("flags", "B"),
    ("otName", f"{PLAYER_NAME_LENGTH}s"),
    ("markings", "B"),
    ("checksum", "H"),
    ("unknown", "H"),
    ("substructs", f"{48}s"),
)
BoxPokemon = namedtuple("BoxPokemon", [x[0] for x in BoxPokemon_spec])
BoxPokemon_format = "".join([x[1] for x in BoxPokemon_spec])


Pokemon_spec = (
    ("box", f"{struct.calcsize(BoxPokemon_format)}s"),
    ("status", "I"),
    ("level", "B"),
    ("mail", "B"),
    ("hp", "H"),
    ("maxHp", "H"),
    ("attack", "H"),
    ("defense", "H"),
    ("speed", "H"),
    ("spAttack", "H"),
    ("spDefense", "H"),
)
Pokemon = namedtuple("Pokemon", [x[0] for x in Pokemon_spec])
Pokemon_format = "".join([x[1] for x in Pokemon_spec])


Pokedex_spec = (
    ("order", "B"),
    ("mode", "B"),
    ("nationalMagic", "B"),
    ("padding1", "B"),
    ("unownPersonality", "I"),
    ("spindaPersonality", "I"),
    ("padding2", "4s"),
    ("owned", f"{NUM_DEX_FLAG_BYTES}s"),
    ("seen", f"{NUM_DEX_FLAG_BYTES}s"),
)
Pokedex = namedtuple("Pokedex", [x[0] for x in Pokedex_spec])
Pokedex_format = "".join([x[1] for x in Pokedex_spec])


SpeciesInfo_spec = (
    ("baseHP", "B"),
    ("baseAttack", "B"),
    ("baseDefense", "B"),
    ("baseSpeed", "B"),
    ("baseSpAttack", "B"),
    ("baseSpDefense", "B"),
    ("type1", "B"),
    ("type2", "B"),
    ("catchRate", "B"),
    ("expYield", "B"),
    ("evYield", "H"),
    ("itemCommon", "H"),
    ("itemRare", "H"),
    ("genderRatio", "B"),
    ("eggCycles", "B"),
    ("friendship", "B"),
    ("growthRate", "B"),
    ("eggGroup1", "B"),
    ("eggGroup2", "B"),
    ("ability1", "B"),
    ("ability2", "B"),
    ("safariZoneFleeRate", "B"),
    ("bodyColor", "Bxx"),
)
SpeciesInfo = namedtuple("SpeciesInfo", [x[0] for x in SpeciesInfo_spec])
SpeciesInfo_format = "".join([x[1] for x in SpeciesInfo_spec])

Coords16_spec = (
    ("x", "H"),
    ("y", "H"),
)
Coords16 = namedtuple("Coords16", [x[0] for x in Coords16_spec])
Coords16_format = "".join([x[1] for x in Coords16_spec])

WarpData_spec = (
    ("mapGroup", "b"),
    ("mapNum", "b"),
    ("warpId", "bx"),
    ("x", "H"),
    ("y", "H"),
)
WarpData = namedtuple("WarpData", [x[0] for x in WarpData_spec])
WarpData_format = "".join([x[1] for x in WarpData_spec])

ItemSlot_spec = (
    ("itemId", "H"),
    ("quantity", "H"),
)
ItemSlot = namedtuple("ItemSlot", [x[0] for x in ItemSlot_spec])
ItemSlot_format = "".join([x[1] for x in ItemSlot_spec])

SaveBlock2_spec = (
    ("playerName", f"{PLAYER_NAME_LENGTH + 1}s"),
    ("playerGender", "B"),
    ("specialSaveWarpFlags", "B"),
    ("playerTrainerId", "4s"),
    ("playTimeHours", "H"),
    ("playTimeMinutes", "B"),
    ("playTimeSeconds", "B"),
    ("playTimeVBlanks", "B"),
    ("optionsButtonMode", "B"),
    ("options", "H"),
    ("padding1", "2s"),
    ("pokedex", f"{struct.calcsize(Pokedex_format)}s"),
    ("filler_90", "8s"),
    ("localTimeOffset", "8s"),
    ("lastBerryTreeUpdate", "8s"),
    ("gcnLinkFlags", "I"),
    ("encryptionKey", "I"),
    ("rest", f"{0xe7c}s"),
)
SaveBlock2 = namedtuple("SaveBlock2", [x[0] for x in SaveBlock2_spec])
SaveBlock2_format = "".join([x[1] for x in SaveBlock2_spec])

SaveBlock1_spec = (
    ("pos", f"{struct.calcsize(Coords16_format)}s"),
    ("location", f"{struct.calcsize(WarpData_format)}s"),
    ("continueGameWarp", f"{struct.calcsize(WarpData_format)}s"),
    ("dynamicWarp", f"{struct.calcsize(WarpData_format)}s"),
    ("lastHealLocation", f"{struct.calcsize(WarpData_format)}s"),
    ("escapeWarp", f"{struct.calcsize(WarpData_format)}s"),
    ("savedMusic", "H"),
    ("weather", "B"),
    ("weatherCycleStage", "B"),
    ("flashLevel", "B"),
    ("padding1", "B"),
    ("mapLayoutId", "H"),
    ("mapView", f"{0x200}s"),
    ("playerPartyCount", "B"),
    ("padding2", "3s"),
    ("playerParty", f"{600}s"),
    ("money", "I"),
    ("coins", "H"),
    ("registeredItem", "H"),
    ("pcItems", f"{struct.calcsize(ItemSlot_format) * PC_ITEMS_COUNT}s"),
    ("bagPocket_Items", f"{struct.calcsize(ItemSlot_format) * BAG_ITEMS_COUNT}s"),
    ("bagPocket_KeyItems", f"{struct.calcsize(ItemSlot_format) * BAG_KEYITEMS_COUNT}s"),
    ("bagPocket_PokeBalls", f"{struct.calcsize(ItemSlot_format) * BAG_POKEBALLS_COUNT}s"),
    ("bagPocket_TMHM", f"{struct.calcsize(ItemSlot_format) * BAG_TMHM_COUNT}s"),
    ("bagPocket_Berries", f"{struct.calcsize(ItemSlot_format) * BAG_BERRIES_COUNT}s"),
    ("pokeblocks", f"{320}s"),
    ("seen1", f"{NUM_DEX_FLAG_BYTES}s"),
    ("berryBlenderRecords", "6s"),
    ("unused", "6s"),
    ("trainerRematchStepCounter", "H"),
    ("trainedRematches", "100s"),
    ("padding3", "2s"),
    ("objectEvents", f"{576}s"),
    ("objectEventTemplates", f"{1536}s"),
    ("flags", f"{300}s"),
    ("rest", f"{0x29ec}s"),
)
SaveBlock1 = namedtuple("SaveBlock1", [x[0] for x in SaveBlock1_spec])
SaveBlock1_format = "".join([x[1] for x in SaveBlock1_spec])


def _compute_save_block_1_offsets():
    offsets = {}
    offset = 0
    for name, fmt in SaveBlock1_spec:
        offsets[name] = offset
        offset += struct.calcsize(fmt)
    return offsets


SAVE_BLOCK_1_OFFSETS = _compute_save_block_1_offsets()


PokemonStorage_spec = (
    ("currentBox", "B"),
    ("padding", "3s"),  # 3 bytes padding
    ("boxes", f"{struct.calcsize(BoxPokemon_format) * TOTAL_BOXES_COUNT * IN_BOX_COUNT}s"),
    ("boxNames", f"{TOTAL_BOXES_COUNT * (BOX_NAME_LENGTH + 1)}s"),
    ("boxWallpapers", f"{TOTAL_BOXES_COUNT}s"),
)
PokemonStorage = namedtuple("PokemonStorage", [x[0] for x in PokemonStorage_spec])
PokemonStorage_format = "".join([x[1] for x in PokemonStorage_spec])


def parse_box_pokemon(data):
    if int.from_bytes(data[:4], "little") == 0:
        return None

    box = BoxPokemon._make(struct.unpack("<" + BoxPokemon_format, data))
    
    key = box.otId ^ box.personality
    substructs_raw = struct.unpack("<" + "I" * 12, box.substructs)
    substructs = [x ^ key for x in substructs_raw]

    substructSelector = [
		[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 3, 1, 2],
		[0, 2, 3, 1], [0, 3, 2, 1], [1, 0, 2, 3], [1, 0, 3, 2],
		[2, 0, 1, 3], [3, 0, 1, 2], [2, 0, 3, 1], [3, 0, 2, 1],
		[1, 2, 0, 3], [1, 3, 0, 2], [2, 1, 0, 3], [3, 1, 0, 2],
		[2, 3, 0, 1], [3, 2, 0, 1], [1, 2, 3, 0], [1, 3, 2, 0],
		[2, 1, 3, 0], [3, 1, 2, 0], [2, 3, 1, 0], [3, 2, 1, 0],
    ]
    # get substruct permutation by personality mod 24
    perm = substructSelector[box.personality % 24]
    substruct0 = substructs[3 * perm[0] : 3 * (perm[0] + 1)]
    substruct1 = substructs[3 * perm[1] : 3 * (perm[1] + 1)]
    substruct2 = substructs[3 * perm[2] : 3 * (perm[2] + 1)]
    substruct3 = substructs[3 * perm[3] : 3 * (perm[3] + 1)]

    substruct0 = PokemonSubstruct0._make(struct.unpack("<" + PokemonSubstruct0_format, struct.pack("<" + "I" * 3, *substruct0)))
    substruct2 = PokemonSubstruct2._make(struct.unpack("<" + PokemonSubstruct2_format, struct.pack("<" + "I" * 3, *substruct2)))

    x1, x2, x3 = substruct1
    substruct1 = PokemonSubstruct1(
        [
            (x1 >> 0)  & 0xFFFF,
            (x1 >> 16) & 0xFFFF,
            (x2 >> 0) & 0xFFFF,
            (x2 >> 16) & 0xFFFF,
        ],
        [
            (x3 >> 0)  & 0xFF,
            (x3 >> 8)  & 0xFF,
            (x3 >> 16) & 0xFF,
            (x3 >> 24) & 0xFF,
        ]
    )
    
    x1, x2, x3 = substruct3
    substruct3 = PokemonSubstruct3(
        (x1 >> 0)  & 0xFF,
        (x1 >> 8)  & 0xFFFF,
        (x1 >> 16) & 0b01111111,
        (x1 >> 23) & 0xF,
        (x1 >> 27) & 0xF,
        (x1 >> 31) & 0b1,
        (x2 >> 0)  & 0b00011111,
        (x2 >> 5)  & 0b00011111,
        (x2 >> 10) & 0b00011111,
        (x2 >> 15) & 0b00011111,
        (x2 >> 20) & 0b00011111,
        (x2 >> 25) & 0b00011111,
        (x2 >> 30) & 0b1,
        (x2 >> 31) & 0b1,
        x3,
    )

    box = box._replace(
        nickname=EmeraldCharmap().decode(box.nickname),
        otName=EmeraldCharmap().decode(box.otName),
        substructs=(
            substruct0._asdict(),
            substruct1._asdict(),
            substruct2._asdict(),
            substruct3._asdict(),
        ),
    )

    box = box._asdict()
    del box["unknown"]
    del box["substructs"][0]["unknown"]
    return box

def parse_pokemon(data):
    pokemon = Pokemon._make(struct.unpack("<" + Pokemon_format, data))
    box = parse_box_pokemon(pokemon.box)
    pokemon = pokemon._replace(box=box)
    species_id = box['substructs'][0]['species']
    # Construct a PokemonData object
    return PokemonData(
        species_id=species_id,
        species_name=_species_id_to_name(species_id),
        current_hp=pokemon.hp,
        max_hp=pokemon.maxHp,
        level=pokemon.level,
        status=StatusCondition(pokemon.status),
        type1=PokemonType(box['type1']) if 'type1' in box.keys() else None,
        type2=PokemonType(box['type2']) if 'type2' in box.keys() else None,
        moves=[Move(move).name for move in box['substructs'][1]['moves']],
        move_pp=box['substructs'][1]['pp'],
        trainer_id=box['otId'],
        nickname=box['nickname'],
        experience=box['substructs'][0]['experience'],
        is_egg=bool(box['substructs'][3]['isEgg'])
    )


def read_save_block_2(gba):
    save_block_2_ptr = gba.read_u32(ADDRESSES["gSaveBlock2Ptr"])
    if save_block_2_ptr == 0:
        return None

    save_block_2_data = gba.read_memory(save_block_2_ptr, struct.calcsize(SaveBlock2_format))
    save_block_2 = SaveBlock2._make(struct.unpack("<" + SaveBlock2_format, save_block_2_data))
    save_block_2 = save_block_2._replace(pokedex=Pokedex._make(struct.unpack("<" + Pokedex_format, save_block_2.pokedex))._asdict())
    return save_block_2._asdict()

def read_save_block_1(gba, parse_items: bool = False):
    save_block_1_ptr = gba.read_u32(ADDRESSES["gSaveBlock1Ptr"])
    if save_block_1_ptr == 0:
        return None

    save_block_1_data = gba.read_memory(save_block_1_ptr, struct.calcsize(SaveBlock1_format))
    save_block_1 = SaveBlock1._make(struct.unpack("<" + SaveBlock1_format, save_block_1_data))
    
    player_party_count = gba.read_u8(ADDRESSES["gPlayerPartyCount"])
    player_party_data = gba.read_memory(ADDRESSES["gPlayerParty"], player_party_count * struct.calcsize(Pokemon_format))

    # parse nested structs
    save_block_1 = save_block_1._replace(
        pos=Coords16._make(struct.unpack("<" + Coords16_format, save_block_1.pos))._asdict(),
        location=WarpData._make(struct.unpack("<" + WarpData_format, save_block_1.location))._asdict(),
        continueGameWarp=WarpData._make(struct.unpack("<" + WarpData_format, save_block_1.continueGameWarp))._asdict(),
        dynamicWarp=WarpData._make(struct.unpack("<" + WarpData_format, save_block_1.dynamicWarp))._asdict(),
        lastHealLocation=WarpData._make(struct.unpack("<" + WarpData_format, save_block_1.lastHealLocation))._asdict(),
        escapeWarp=WarpData._make(struct.unpack("<" + WarpData_format, save_block_1.escapeWarp))._asdict(),
        playerParty=[
            parse_pokemon(player_party_data[i:i+struct.calcsize(Pokemon_format)])
            for i in range(0, player_party_count * struct.calcsize(Pokemon_format), struct.calcsize(Pokemon_format))
        ],
    )
    if parse_items:
        save_block_1 = save_block_1._replace(
            pcItems=[
                ItemSlot._make(struct.unpack("<" + ItemSlot_format, save_block_1.pcItems[i:i+struct.calcsize(ItemSlot_format)]))._asdict()
                for i in range(0, len(save_block_1.pcItems), struct.calcsize(ItemSlot_format))
            ],
            bagPocket_Items=[
                ItemSlot._make(struct.unpack("<" + ItemSlot_format, save_block_1.bagPocket_Items[i:i+struct.calcsize(ItemSlot_format)]))._asdict()
                for i in range(0, len(save_block_1.bagPocket_Items), struct.calcsize(ItemSlot_format))
            ],
            bagPocket_KeyItems=[
                ItemSlot._make(struct.unpack("<" + ItemSlot_format, save_block_1.bagPocket_KeyItems[i:i+struct.calcsize(ItemSlot_format)]))._asdict()
                for i in range(0, len(save_block_1.bagPocket_KeyItems), struct.calcsize(ItemSlot_format))
            ],
            bagPocket_PokeBalls=[
                ItemSlot._make(struct.unpack("<" + ItemSlot_format, save_block_1.bagPocket_PokeBalls[i:i+struct.calcsize(ItemSlot_format)]))._asdict()
                for i in range(0, len(save_block_1.bagPocket_PokeBalls), struct.calcsize(ItemSlot_format))
            ],
            bagPocket_TMHM=[
                ItemSlot._make(struct.unpack("<" + ItemSlot_format, save_block_1.bagPocket_TMHM[i:i+struct.calcsize(ItemSlot_format)]))._asdict()
                for i in range(0, len(save_block_1.bagPocket_TMHM), struct.calcsize(ItemSlot_format))
            ],
            bagPocket_Berries=[
                ItemSlot._make(struct.unpack("<" + ItemSlot_format, save_block_1.bagPocket_Berries[i:i+struct.calcsize(ItemSlot_format)]))._asdict()
                for i in range(0, len(save_block_1.bagPocket_Berries), struct.calcsize(ItemSlot_format))
            ]
        )

    return save_block_1._asdict()


def read_pokemon_storage(gba):
    pokemon_storage_ptr = gba.read_u32(ADDRESSES["gPokemonStoragePtr"])
    if pokemon_storage_ptr == 0:
        return None

    pokemon_storage_data = gba.read_memory(pokemon_storage_ptr, struct.calcsize(PokemonStorage_format))
    pokemon_storage = PokemonStorage._make(struct.unpack("<" + PokemonStorage_format, pokemon_storage_data))
    
    box_mon_size = struct.calcsize(BoxPokemon_format)
    box_size = box_mon_size * IN_BOX_COUNT
    parsed_boxes = []
    for j in range(TOTAL_BOXES_COUNT):
        parsed_boxes.append([
            parse_box_pokemon(pokemon_storage.boxes[i:i+box_mon_size])
            for i in range(j * box_size, (j + 1) * box_size, box_mon_size)
        ])
    pokemon_storage = pokemon_storage._replace(
        boxes=parsed_boxes,
        boxNames=[
            pokemon_storage.boxNames[i:i+BOX_NAME_LENGTH]
            for i in range(0, len(pokemon_storage.boxNames), BOX_NAME_LENGTH + 1)
        ]
    )
    return pokemon_storage._asdict()

@functools.lru_cache(maxsize=1)
def read_species_names(gba):
    species_names_ptr = ADDRESSES["gSpeciesNames"]
    if species_names_ptr == 0:
        return None

    species_names_data = gba.read_memory(species_names_ptr, NUM_SPECIES * (POKEMON_NAME_LENGTH +1))
    species_names = [
        EmeraldCharmap().decode(species_names_data[i:i+POKEMON_NAME_LENGTH+1])
        for i in range(0, len(species_names_data), POKEMON_NAME_LENGTH+1)
    ]
    return species_names

@functools.lru_cache(maxsize=1)
def read_species_info(gba):
    species_info_ptr = ADDRESSES["gSpeciesInfo"]
    if species_info_ptr == 0:
        return None

    species_info_data = gba.read_memory(species_info_ptr, NUM_SPECIES * struct.calcsize(SpeciesInfo_format))
    species_info = [
        SpeciesInfo._make(struct.unpack("<" + SpeciesInfo_format, species_info_data[i:i+struct.calcsize(SpeciesInfo_format)]))
        for i in range(0, len(species_info_data), struct.calcsize(SpeciesInfo_format))
    ]
    return species_info

@functools.lru_cache(maxsize=1)
def read_experience_tables(gba):
    exp_table_ptr = ADDRESSES["gExperienceTables"]
    if exp_table_ptr == 0:
        return None
    
    # there's 6 different growth rates and 101 different levels, each being a 4-byte int
    num_ints = 6 * 101
    exp_table_data = gba.read_memory(exp_table_ptr, num_ints * 4)
    exp_table_format = "<" + "I" * num_ints
    exp_table_flat = struct.unpack(exp_table_format, exp_table_data)
    exp_tables = []
    for i in range(0, len(exp_table_flat), 101):
        exp_tables.append(exp_table_flat[i:i+101])
    return exp_tables
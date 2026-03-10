"""
red_metatile_behavior.py

Gen 1 (pokered) tile behavior enum analogous to Emerald's MetatileBehavior.

Integer values are chosen to be valid members of Emerald's MetatileBehavior enum
(pokemon_env/enums.py) so pathfinding.py's MetatileBehavior(int).name substring
checks produce correct walk/block/connector classifications without modification.

Sources consulted (all under pokemon_red_env/data/pokered/data/tilesets/):
  - collision_tile_ids.asm      → NORMAL / IMPASSABLE
  - ledge_tiles.asm             → JUMP_EAST / JUMP_WEST / JUMP_SOUTH
  - pair_collision_tile_ids.asm → movement constraints (no new behavior needed)
  - door_tile_ids.asm           → NON_ANIMATED_DOOR
  - warp_tile_ids.asm           → LADDER
  - warp_pad_hole_tile_ids.asm  → WARP_PAD (value=1) / CRACKED_FLOOR_HOLE (value=2)
  - warp_carpet_tile_ids.asm    → WARP_CARPET
  - spinner_tiles.asm           → SPINNER
  - bookshelf_tile_ids.asm      → BOOKSHELF
  - tileset_headers.asm         → TALL_GRASS (byte 10), COUNTER (bytes 7-9)
  - engine/overworld/           → water tile $14 confirmed across all tilesets
  - data/maps/hidden_events.asm → PC, TELEVISION, QUESTIONNAIRE, BLUEPRINT, TRASH_CAN
"""

from enum import IntEnum


class RedMetatileBehavior(IntEnum):
    """
    Gen 1 (pokered) tile behavior codes for processed_map JSON raw_tile field.

    Integer values are chosen to be valid in Emerald's MetatileBehavior enum
    (pokemon_env/enums.py) so pathfinding.py's MetatileBehavior(int).name
    substring checks produce correct walk/block/connector classifications
    without any modification to the pathfinder.

    Naming convention:
      - Exact Emerald match: same name AND value as Emerald
      - Approximate match: descriptive Gen-1 name, value from nearest Emerald analog
    """
    # --- Direct Emerald matches (same name + same value + same concept) ---
    NORMAL            = 0    # Walkable floor/path tile           [Emerald NORMAL=0]
    TALL_GRASS        = 2    # Tall grass; wild encounters trigger [Emerald TALL_GRASS=2]
    DEEP_WATER        = 18   # Water tile; Surf HM required        [Emerald DEEP_WATER=18]
    JUMP_EAST         = 56   # One-way ledge: jump eastward        [Emerald JUMP_EAST=56]
    JUMP_WEST         = 57   # One-way ledge: jump westward        [Emerald JUMP_WEST=57]
    JUMP_SOUTH        = 59   # One-way ledge: jump southward       [Emerald JUMP_SOUTH=59]
    NON_ANIMATED_DOOR = 96   # Building door; door_tile_ids.asm    [Emerald NON_ANIMATED_DOOR=96]
    LADDER            = 97   # Interior warp/stair; warp_tile_ids.asm [Emerald LADDER=97]
    CRACKED_FLOOR_HOLE = 102 # Fall-through hole; warp_pad_hole=2  [Emerald CRACKED_FLOOR_HOLE=102]
    COUNTER           = 128  # Counter/desk; extends NPC talk range [Emerald COUNTER=128]
    PC                = 131  # Hidden PC event (OpenRedsPC, etc.)  [Emerald PC=131]
    TELEVISION        = 134  # Hidden TV event (PrintRedSNESText)  [Emerald TELEVISION=134]
    QUESTIONNAIRE     = 143  # Hidden statue/binoculars/quiz event [Emerald QUESTIONNAIRE=143]
    BOOKSHELF         = 225  # Bookshelf tile; bookshelf_tile_ids.asm [Emerald BOOKSHELF=225]
    TRASH_CAN         = 228  # Hidden trash can event (TrashCan*)  [Emerald TRASH_CAN=228]
    BLUEPRINT         = 230  # Hidden poster/painting event        [Emerald BLUEPRINT=230]

    # --- Approximate Emerald matches (similar concept, valid Emerald value) ---
    WARP_CARPET = 98   # Directional entrance mat; warp_carpet_tile_ids.asm
                       # [Emerald EAST_ARROW_WARP=98 — directional warp]
    WARP_PAD    = 103  # Teleport pad; warp_pad_hole value=1 (Facility/Silph Co.)
                       # [Emerald AQUA_HIDEOUT_WARP=103 — hidden facility warp pad]
    SPINNER     = 68   # Spinner tile; spinner_tiles.asm (Facility/Gym)
                       # [Emerald SLIDE_EAST=68 — forced movement tile]
    POKE_BALL   = 160  # Visible item sprite on the ground (walkable, interact to pick up)
                       # [Emerald BERRY_TREE_SOIL=160 — walkable interactable ground tile]

    # --- Gen-1 specific (repurposed unused/least-mismatched Emerald values) ---
    IMPASSABLE = 1    # Wall/blocked; any tile not in collision passable set;
                      # also bench ("=") and uncategorised hidden objects ("#")
                      # [Emerald SECRET_BASE_WALL=1 — non-walkable, no connector keywords]
    CUT_TREE   = 4    # Cuttable tree; Cut HM required
                      # [Emerald UNUSED_04=4 — repurposed; no Gen-3 equivalent]


# ---------------------------------------------------------------------------
# Pathfinder-compatible collision value per behavior (0=walkable, 1=blocked)
# ---------------------------------------------------------------------------
BEHAVIOR_COLLISION: dict[RedMetatileBehavior, int] = {
    RedMetatileBehavior.NORMAL:             0,
    RedMetatileBehavior.IMPASSABLE:         1,
    RedMetatileBehavior.TALL_GRASS:         0,
    RedMetatileBehavior.DEEP_WATER:         1,  # blocked without Surf
    RedMetatileBehavior.CUT_TREE:           1,  # blocked without Cut
    RedMetatileBehavior.NON_ANIMATED_DOOR:  0,  # door tile is walkable (triggers exit anim)
    RedMetatileBehavior.LADDER:             0,  # staircase/warp tile is walkable
    RedMetatileBehavior.WARP_PAD:           0,  # warp pad is walkable (triggers warp)
    RedMetatileBehavior.WARP_CARPET:        0,  # entrance mat is walkable (triggers warp)
    RedMetatileBehavior.CRACKED_FLOOR_HOLE: 0,  # hole is walkable (fall-through)
    RedMetatileBehavior.COUNTER:            1,  # counter tile blocks player
    RedMetatileBehavior.BOOKSHELF:          1,  # bookshelf blocks player
    RedMetatileBehavior.JUMP_EAST:          0,  # ledge: entered, then forced east
    RedMetatileBehavior.JUMP_WEST:          0,  # ledge: entered, then forced west
    RedMetatileBehavior.JUMP_SOUTH:         0,  # ledge: entered, then forced south
    RedMetatileBehavior.SPINNER:            0,  # walkable, spins player facing
    RedMetatileBehavior.POKE_BALL:          0,  # pokéball sprite tile is walkable
    # Hidden interactive objects — all collision=1 (player cannot walk through them)
    RedMetatileBehavior.PC:            1,
    RedMetatileBehavior.TELEVISION:    1,
    RedMetatileBehavior.QUESTIONNAIRE: 1,
    RedMetatileBehavior.BLUEPRINT:     1,
    RedMetatileBehavior.TRASH_CAN:     1,
}


# ---------------------------------------------------------------------------
# Maps classify_hidden_object() return symbol → RedMetatileBehavior
# Keys are the single-character symbols produced by map_preprocess.py's
# classify_hidden_object() function.
# ---------------------------------------------------------------------------
HIDDEN_SYMBOL_TO_BEHAVIOR: dict[str, RedMetatileBehavior] = {
    "P": RedMetatileBehavior.PC,             # OpenRedsPC, BillsHousePC, etc.
    "T": RedMetatileBehavior.TELEVISION,     # PrintRedSNESText, etc.
    "B": RedMetatileBehavior.BOOKSHELF,      # Bookcase, Notebook, Magazine events
    "^": RedMetatileBehavior.BLUEPRINT,      # Poster, Email display events
    "U": RedMetatileBehavior.TRASH_CAN,      # TrashCan events
    "?": RedMetatileBehavior.QUESTIONNAIRE,  # Statue, Quiz, Binoculars, Dojo kiosk events
    "=": RedMetatileBehavior.IMPASSABLE,     # Bench events (no Emerald equivalent)
    "#": RedMetatileBehavior.IMPASSABLE,     # HiddenItems, HiddenCoins, Mansion switches, etc.
    "O": RedMetatileBehavior.POKE_BALL,      # Visible pokéball item on the ground (walkable)
}

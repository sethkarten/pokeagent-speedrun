# Pokemon Red Integration Proposal (Revised)

## Current State

`pokemon_env/red_emulator.py` provides a `RedEmulator` class backed by [PyBoy](https://github.com/Baekalfen/PyBoy).
It already implements the same public interface as `EmeraldEmulator` and can run any ROM.
The stubs that remain are memory-reader-dependent (party HP/moves, dialog, battle state, map tiles, milestones).

Reference implementation studied: [feng-rrRay/self-evolving-game-agent](https://github.com/feng-rrRay/self-evolving-game-agent).

---

## 1. Emulator (`pokemon_env/red_emulator.py`)

### Technical Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Native resolution | **160 × 144 px** | Game Boy — confirmed by `pyboy.screen.image` and `assert img.size == (160, 144)` in existing tests |
| Server FPS | **80 FPS** | Match `fps = 80` in `server/app.py`; PyBoy is tick-driven so the app loop controls pace |
| Screenshot format | **PIL Image, RGB mode** | `pyboy.screen.image.copy()` — already returns RGBA; call `.convert("RGB")` before encoding |
| Headless window | `window="null"` | Confirmed in `initialize()` |
| Sound | Off by default | `sound=False`; same as Emerald |
| State format | PyBoy binary | `pyboy.save_state(buf)` / `pyboy.load_state(buf)` with `io.BytesIO` — already implemented |

### Action Timing

Align with the existing `SPEED_PRESETS` in `server/app.py`. At 80 FPS:

| Preset | Hold frames | Release frames | Total time |
|--------|-------------|----------------|------------|
| `fast` | 6 | 3 | 112 ms |
| `normal` (default) | 10 | 8 | 225 ms |
| `slow` | 16 | 16 | 400 ms |
| `deliberate` | 20 | 10 | 375 ms |

During active dialog, `get_current_fps()` returns `base_fps × 4 = 320` FPS, same as Emerald.
This requires `PokemonRedReader.is_in_dialog()` to be functional (see §2).

Reference implementation used: hold 10 frames + 2-frame buffer + 100-frame post-action delay at
100 FPS (≈ 1 second per action). Our SPEED_PRESETS are much tighter; we can absorb any
"settling" time in the release phase rather than adding a fixed 100-frame stall.

### Key Method Gap-Fills

| Method | Status | Action |
|--------|--------|--------|
| `get_screenshot()` | Working | Ensure `.convert("RGB")` before base64 encoding |
| `run_frame_with_buttons()` | Working | Unchanged |
| `press_buttons()` | Working | Unchanged |
| `tick()` | Working | Unchanged |
| `get_current_fps()` | Stub | Needs `self._cached_dialog_state` from `PokemonRedReader` |
| `get_comprehensive_state()` | Partial | Fills out as `PokemonRedReader` is added |
| `check_and_update_milestones()` | Stub | Implement once milestones defined (§6) |
| `get_map_tiles()` | Stub | Implement with map system (§3) |

### Button Set

Game Boy has no L/R triggers. Red's `KEY_MAP` is a strict subset of Emerald's:
`a, b, start, select, up, down, left, right`.
Any caller that passes `l` or `r` gets a warning and is skipped (already handled in `press_buttons()`).

---

## 2. Memory Reader (`pokemon_env/red_memory_reader.py`)

Create a `PokemonRedReader` class mirroring `PokemonEmeraldReader`'s interface.
Source addresses from the [pokered decompilation](https://github.com/pret/pokered)
and validated against the reference implementation.

### RAM Address Table

| Field | Address | Size | Encoding |
|-------|---------|------|----------|
| Player name | `0xD158` | 11 bytes | Custom charset (A=0x80) |
| Rival name | `0xD34A` | 11 bytes | Same charset |
| Map ID | `0xD35E` | 1 byte | See map name table |
| Player Y | `0xD361` | 1 byte | Tile coord |
| Player X | `0xD362` | 1 byte | Tile coord |
| Player direction | `0xC109` | 1 byte | 0=down, 4=up, 8=left, 12=right |
| Party count | `0xD163` | 1 byte | 0–6 |
| Party species IDs | `0xD164` | 6 bytes | Species ID array |
| Party data base | `0xD16B` | 6 × 0x2C bytes | Full party struct (see below) |
| Badges | `0xD356` | 1 byte | Bitmask (bit 0 = Boulder Badge) |
| Money | `0xD347` | 3 bytes | BCD (e.g. `0x01 0x23 0x45` → ¥12345) |
| Inventory count | `0xD31C` | 1 byte | Number of item slots |
| Inventory items | `0xD31D` | 2 × count bytes | Alternating (item_id, quantity) |
| Pokedex seen | `0xD30A` | 19 bytes | Bit array (151 entries) |
| Pokedex owned | `0xD2F7` | 19 bytes | Bit array |
| Battle type | `0xD057` | 1 byte | 0=none, 1=wild, 2=trainer |
| Link state | `0xD72E` | 1 byte | 0x05 = link battle |
| Enemy species | `0xCFE5` | 1 byte | Species ID |
| Enemy level | `0xCFF3` | 1 byte | |
| Enemy HP | `0xCFE6` | 2 bytes | Big-endian |
| Enemy max HP | `0xCFF4` | 2 bytes | Big-endian |
| Active Pokémon name | `0xD009` | 11 bytes | |
| Text progress | `0xC6AC` | 1 byte | Dialog advancement flag |
| Tilemap (VRAM) | `0xC3A0` | 20 × 18 bytes | ASCII-encoded screen tiles |
| Game state flags | `0xD72E`, `0xCC26` | 1 byte each | Menu/overworld detection |

#### Party Struct (`0xD16B + i × 0x2C`)

Each of the 6 slots is 44 bytes:

| Offset | Field | Size |
|--------|-------|------|
| +0x00 | Species ID | 1 |
| +0x01 | Current HP | 2 (big-endian) |
| +0x03 | Level (in battle) | 1 |
| +0x04 | Status condition | 1 |
| +0x05 | Type 1 | 1 |
| +0x06 | Type 2 | 1 |
| +0x08 | Move 1–4 IDs | 4 × 1 |
| +0x0D–0x0E | Max HP | 2 |
| +0x0F–0x10 | Attack | 2 |
| +0x11–0x12 | Defense | 2 |
| +0x13–0x14 | Speed | 2 |
| +0x15–0x16 | Special | 2 |
| +0x1D | Level (actual) | 1 |
| +0x1E–0x25 | Move 1–4 PP | 1 each |

### Dialog Detection Strategy

1. Read tilemap from VRAM `0xC3A0` (20 × 18 = 360 characters).
2. Detect dialog box by scanning for border tiles (`─`, `│`, `┐`, `└`, etc.) in the bottom 5 rows.
3. Confirm with `0xC6AC` (text progress byte, non-zero = active text print).
4. Return `bool` — sets `_cached_dialog_state` in `RedEmulator` to activate 4× FPS multiplier.

### Game State Classification

```
0xD057 = 0  AND  link_state != 0x05  →  "overworld" or "menu"
0xD057 = 1                            →  "wild_battle"
0xD057 = 2                            →  "trainer_battle"
dialog_detected = True                →  "dialog" (overlaps with above)
```

### Interface (mirrors `PokemonEmeraldReader`)

```python
class PokemonRedReader:
    def __init__(self, pyboy: PyBoy)
    def read_coordinates(self) -> tuple[int, int]
    def read_location(self) -> str               # Map name from ID
    def read_money(self) -> int                  # BCD decode
    def read_party_pokemon(self) -> list[dict]   # Full struct parse
    def read_badges(self) -> list[str]           # Named badges from bitmask
    def read_game_state(self) -> str             # "overworld"/"dialog"/"battle"
    def is_in_dialog(self) -> bool
    def is_in_battle(self) -> bool
    def get_battle_info(self) -> dict
    def read_items(self) -> dict                 # {item_name: count}
    def read_pokedex(self) -> dict               # {"seen": int, "caught": int}
    def read_direction(self) -> str              # "North"/"South"/"East"/"West"
    def read_screen_text(self) -> str            # Tilemap decode for dialog/menus
    def get_comprehensive_state(self, screenshot) -> dict
```

---

## 3. Map System (`pokemon_env/red_map_reader.py`)

### Approach A — VRAM Tilemap (Recommended First Step)

Read the 20 × 18 background tile map live from VRAM (`0xC3A0`).
Map tile IDs to walkability/terrain categories using a hardcoded lookup table
(grass/water/wall, derived from the pokered ROM's collision data).

Pros: no offline preprocessing; always reflects current map state.
Cons: 20 × 18 viewport only; no ground-truth collision outside the visible window.

Output: same 7-radius ASCII grid format as Emerald (`utils/state_formatter.py`):
```
##########
#....@...#     @ = player
#.G.G....#     . = walkable, G = grass, # = wall, W = water
##########
```

### Approach B — Ported pokered Map Data (Full Ground Truth)

Extract map dimensions, collision arrays, warp coordinates, and NPC positions from the
[pokered decompilation](https://github.com/pret/pokered) into JSON files that mirror
`porymap_data/` (the Emerald ground-truth data already in the repo).

Structure:
```
pokered_map_data/
  maps/
    PALLET_TOWN.json
    VIRIDIAN_CITY.json
    PEWTER_CITY.json
    MT_MOON_1F.json
    ...
  connections.json   # warp/connection graph
  map_names.json     # map_id → name
```

Each map JSON follows the same schema as `porymap_data/`:
```json
{
  "name": "PALLET_TOWN",
  "width": 10,
  "height": 9,
  "collision": [[0,0,...], ...],
  "warps": [{"x":3,"y":7,"dest_map":"PLAYER_HOUSE_1F","dest_x":3,"dest_y":1}],
  "objects": [{"x":5,"y":4,"type":"npc","name":"Mom"}]
}
```

Pros: A* pathfinding, NPC avoidance, portal tracking — same features Emerald has today.
Cons: one-time but significant extraction effort (~100 maps).

**Recommendation**: ship Approach A first for a quick working demo; port Approach B iteratively
map-by-map as the agent actually visits them.

### Shared Infrastructure

`utils/state_formatter.py` already handles ASCII grid rendering and NPC overlays.
The only extension needed is a `get_red_map_data(map_id)` loader that sources from
`pokered_map_data/` instead of `porymap_data/`.

---

## 4. MCP Tools (`server/cli/pokemon_mcp_server.py`)

The existing MCP server exposes game tools that are game-agnostic.
Red needs these additions or adaptations:

### Existing Tools (work without change)
- `get_game_state_direct` — reads from `env.get_comprehensive_state()`
- `press_buttons_direct` — posts to action queue
- `add_knowledge` / `search_knowledge` — game-independent

### Tools Requiring Red Extensions

| Tool | Change Needed |
|------|--------------|
| `navigate_to(x, y)` | A* needs Red collision grid (Approach B) or VRAM fallback (Approach A) |
| `get_walkthrough(part)` | Point to Red/Blue Bulbapedia walkthrough (currently Emerald) |
| `lookup_pokemon_info(pokemon)` | Works already (external API call) |

### New Red-Specific Tools

```python
@mcp.tool()
def get_text_on_screen() -> str:
    """Return decoded dialog/menu text from VRAM tilemap."""

@mcp.tool()
def get_battle_menu_state() -> dict:
    """Return current battle menu (FIGHT/PKMN/ITEM/RUN) and move options."""

@mcp.tool()
def solve_puzzle(puzzle_name: str) -> list[str]:
    """Return canned button sequence for known Red puzzles:
    'surge_trashcans', 'silph_elevators', 'sabrina_teleporters',
    'rocket_hideout_spinners', 'victory_road_boulders'"""
```

### Tool Routing

When `--game red` is active, `server/app.py` passes the `RedEmulator` instance to the MCP
server at startup. All existing tool handlers that call `env.*` will work automatically
because `RedEmulator` shares the same interface.

---

## 5. Server Integration (`server/app.py`)

### Minimal Change (Option A — Recommended)

Add a `--game red` CLI flag to `run.py` and pass it through to `server/app.py`.
`setup_environment()` becomes:

```python
def setup_environment(args):
    game = getattr(args, "game", "emerald")
    if game == "red":
        from pokemon_env.red_emulator import RedEmulator
        global env
        env = RedEmulator(rom_path=args.rom_path or "PokemonRed-GBC/pokered.gbc")
    else:
        from pokemon_env.emulator import EmeraldEmulator
        env = EmeraldEmulator(rom_path=args.rom_path or "Emerald-GBAdvance/emerald.gba")
    env.initialize()
```

No other changes to the 800-line server are needed — every endpoint already calls
`env.get_screenshot()`, `env.run_frame_with_buttons()`, `env.get_comprehensive_state()`.

### Difference Checklist

| Area | Emerald | Red | Action |
|------|---------|-----|--------|
| Resolution | 240 × 160 (GBA) | 160 × 144 (GB) | `state["visual"]["resolution"]` already reads from `[self.width, self.height]` — no change |
| Screenshot encoding | PNG base64 | PNG base64 | Same code path |
| FPS (server loop) | 80 | **80** (keep identical) | PyBoy ticks unconstrained; server loop throttles to 80 FPS |
| Dialog 4× boost | Yes | Yes (after reader built) | Same `get_current_fps()` contract |
| WebSocket streaming | every 30 frames → ~2.7 FPS at 80 | Same | No change |
| Recording | every 4 frames → 20 FPS at 80 | Same | No change |
| State `["map"]["porymap"]` | porymap_data JSON | pokered_map_data JSON | Loader abstraction |
| Buttons | 10 (incl. L/R) | 8 (no L/R) | Warning on L/R, otherwise same |
| Anti-cheat | Hashes game state | Must hash Red state | Same `anticheat_tracker` — works if state dict shape matches |

### Streaming Page

The HTML stream page (`/`) served by the server embeds a live JPEG WebSocket feed.
Resolution is purely cosmetic there — the `<img>` tag stretches to fit.
No change needed; the 160 × 144 frames will simply appear with a different aspect ratio.

If visual parity matters, add a CSS `image-rendering: pixelated` rule and a fixed pixel
width (e.g. `320 × 288` upscale at 2×) in the stream page template.

---

## 6. Milestone Tracker

### Milestone Sequence

```python
RED_MILESTONES = [
    # Story milestones (badge gates)
    {"id": "PALLET_TOWN_START",    "desc": "Game started in Pallet Town",    "check": lambda s: s["player"]["location"] == "PALLET_TOWN"},
    {"id": "OAK_ENCOUNTER",        "desc": "Met Professor Oak",               "check": lambda s: has_starter(s)},
    {"id": "VIRIDIAN_CITY",        "desc": "Reached Viridian City",           "check": lambda s: s["player"]["location"] == "VIRIDIAN_CITY"},
    {"id": "PEWTER_CITY",          "desc": "Reached Pewter City",             "check": lambda s: s["player"]["location"] == "PEWTER_CITY"},
    {"id": "BROCK_DEFEATED",       "desc": "Earned Boulder Badge",            "check": lambda s: badge_earned(s, 0)},
    {"id": "MT_MOON_CROSSED",      "desc": "Exited Mt. Moon",                 "check": lambda s: visited_any(s, ["CERULEAN_CITY"])},
    {"id": "CERULEAN_CITY",        "desc": "Reached Cerulean City",           "check": lambda s: s["player"]["location"] == "CERULEAN_CITY"},
    {"id": "MISTY_DEFEATED",       "desc": "Earned Cascade Badge",            "check": lambda s: badge_earned(s, 1)},
    {"id": "SS_ANNE",              "desc": "Boarded S.S. Anne",               "check": lambda s: visited_any(s, ["SS_ANNE_1F"])},
    {"id": "SURGE_DEFEATED",       "desc": "Earned Thunder Badge",            "check": lambda s: badge_earned(s, 2)},
    {"id": "ROCK_TUNNEL",          "desc": "Crossed Rock Tunnel",             "check": lambda s: visited_any(s, ["LAVENDER_TOWN"])},
    {"id": "LAVENDER_TOWN",        "desc": "Reached Lavender Town",           "check": lambda s: s["player"]["location"] == "LAVENDER_TOWN"},
    {"id": "CELADON_CITY",         "desc": "Reached Celadon City",            "check": lambda s: s["player"]["location"] == "CELADON_CITY"},
    {"id": "ERIKA_DEFEATED",       "desc": "Earned Rainbow Badge",            "check": lambda s: badge_earned(s, 3)},
    {"id": "ROCKET_HIDEOUT",       "desc": "Cleared Team Rocket Hideout",     "check": lambda s: rocket_defeated(s)},
    {"id": "SILPH_CO",             "desc": "Cleared Silph Co.",               "check": lambda s: silph_cleared(s)},
    {"id": "KOGA_DEFEATED",        "desc": "Earned Soul Badge",               "check": lambda s: badge_earned(s, 4)},
    {"id": "SABRINA_DEFEATED",     "desc": "Earned Marsh Badge",              "check": lambda s: badge_earned(s, 5)},
    {"id": "BLAINE_DEFEATED",      "desc": "Earned Volcano Badge",            "check": lambda s: badge_earned(s, 6)},
    {"id": "GIOVANNI_DEFEATED",    "desc": "Earned Earth Badge",              "check": lambda s: badge_earned(s, 7)},
    {"id": "VICTORY_ROAD",         "desc": "Entered Victory Road",            "check": lambda s: visited_any(s, ["VICTORY_ROAD_1F"])},
    {"id": "ELITE_FOUR_START",     "desc": "Entered Elite Four",              "check": lambda s: visited_any(s, ["INDIGO_PLATEAU_LORELEIS_ROOM"])},
    {"id": "CHAMPION",             "desc": "Became Champion",                 "check": lambda s: champion_defeated(s)},
]
```

### Badge Bitmask (`0xD356`)

```
Bit 0: Boulder Badge (Brock)
Bit 1: Cascade Badge (Misty)
Bit 2: Thunder Badge (Lt. Surge)
Bit 3: Rainbow Badge (Erika)
Bit 4: Soul Badge (Koga)
Bit 5: Marsh Badge (Sabrina)
Bit 6: Volcano Badge (Blaine)
Bit 7: Earth Badge (Giovanni)
```

### Milestone Update Loop

`check_and_update_milestones()` is called from `server/app.py`'s background milestone thread
(every 5 seconds). For Red, it reads `PokemonRedReader.read_badges()` and
`PokemonRedReader.read_location()` and evaluates each uncompleted milestone's `check` function.

### Map ID → Name Table

Partial list; full table from pokered `data/maps/mapHeaders.asm`:

```python
RED_MAP_NAMES = {
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
    # ... (full table ~250 entries)
    0xF5: "MT_MOON_1F",
    0xF6: "MT_MOON_2F",
    # ...
}
```

---

## 7. Additional Files

| File | Purpose |
|------|---------|
| `agent/red_direct_objectives.py` | Story progression checklist (mirrors `direct_objectives.py`) |
| `agent/red_puzzle_solver.py` | Canned sequences: Surge trash cans, Silph elevators, Sabrina teleporters |
| `agent/system_prompt_red.md` | Red-specific system instructions (replace Emerald references) |
| `agent/base_prompt_red.md` | Red base prompt |
| `PokemonRed-GBC/` | ROM directory (add `.gitkeep`; ROM not included) |
| `pokered_map_data/` | Ground truth map JSON (generated offline from pokered decomp) |

---

## 8. Open Questions for Discussion

### Q1 — Single game vs. two games simultaneously

**Option A (single game per server process):**
Run one `python run.py --game red` instance and one `python run.py --game emerald` instance
on separate ports (e.g. 8000 and 8001). Evaluation scripts submit a log for each.

*Pros:* zero changes to server/client architecture; clean process isolation; crash in one game
does not affect the other.

*Cons:* two separate agent processes; no shared VLM cache; users must manage two terminal windows.

**Option B (one server, two emulator instances):**
`server/app.py` runs both emulators concurrently. A new `/game/{game}/*` URL namespace
routes requests to the correct emulator.

*Pros:* single process to manage; shared resources.
*Cons:* significant server refactor; mGBA and PyBoy use different threading models;
double the memory and GPU (screenshot encoding) load; more failure modes.

**Recommendation for discussion:** start with Option A. Add a convenience wrapper script
(`run_both.py`) that launches both processes with coordinated ports and a unified log view.

---

### Q2 — FPS parity: 80 vs. 100

The reference implementation ticks PyBoy at 100 FPS (frame_time = 0.01 s).
Our server loop targets 80 FPS (matching mGBA).

At 80 FPS:
- Game speed is 80/59.73 ≈ 1.34× real-time for GB (GBC runs at 59.73 Hz).
- At 100 FPS it would be 1.67× real-time.

Should we unify at 80 FPS (simpler server code), 60 FPS (closer to real-time, more comparable
to Emerald's GBA 59.73 Hz target), or allow game-specific FPS?

---

### Q3 — Map system priority

Full porymap-style ground truth for Red is powerful but requires extracting ~250 maps from the
pokered decompilation. Should we:

a) Ship VRAM-only map (Approach A) in the first PR, deferring ground truth,
b) Extract the 20 most critical maps (Pallet → Cerulean routing + key dungeons) first, or
c) Run a semi-automated extraction script over all maps before merging?

---

### Q4 — State formatter for Red

`utils/state_formatter.py` is Emerald-specific in several places (type names, move descriptions,
ability references, Emerald map sources). Options:

a) Add `if game == "red":` branches inside the existing formatter,
b) Subclass or instantiate `StateFormatter(game="red")` with pluggable map/species data,
c) Create a thin `red_state_formatter.py` that adapts the Red state dict to the shape
   `state_formatter.py` already expects.

Option c is least invasive and easiest to test in isolation.

---

### Q5 — Anti-cheat compatibility

`utils/anticheat.py` hashes the game state dict on every step. As long as `RedEmulator`
returns the same top-level keys (`visual`, `player`, `game`, `map`, `milestones`),
the anti-cheat system works without modification. Confirm this before submission.

---

### Q6 — Evaluation scope

For the PokéAgent Challenge, should Red submissions be evaluated on the same milestone
completion + time metric as Emerald? Red has 8 gyms + Elite Four; Emerald has 8 gyms + Elite Four.
The difficulty and optimal speedrun time differ. Should each game have an independent leaderboard,
or a combined normalized score?

---

## 9. Implementation Order (Suggested)

1. **`PokemonRedReader` basic** — coordinates, location name, badges, money, game_state, dialog
2. **Wire into server** — `--game red` flag, `setup_environment()` branch
3. **Milestone tracker** — badge-gate milestones using the basic reader
4. **Full party stats** — HP, moves, types from party struct
5. **Battle info** — enemy species/HP, player active Pokémon
6. **Map system Approach A** — VRAM tilemap + collision lookup
7. **Red direct objectives + prompts**
8. **Map system Approach B** — pokered ground truth JSON (iterative)
9. **MCP tool extensions** — puzzle solver, screen text tool
10. **Red-specific agent scaffold tuning**

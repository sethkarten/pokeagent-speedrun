# Pokemon Red Integration — Summary of Changes

## Overview

Pokemon Red (Gen 1, Game Boy) has been integrated into the pokeagent-speedrun framework as a drop-in alternative to Pokemon Emerald. The agent can now play Red by running `python run.py --game red`. All existing Emerald functionality is preserved — the `--game` flag (default: `emerald`) gates all game-specific behavior.

## Architecture

The integration follows the same client-server model as Emerald. When `--game red` is passed:

1. **Server** (`server/app.py`) instantiates `RedEmulator` instead of `EmeraldEmulator`
2. **Client** (`agent/my_cli_agent.py`) loads Red-specific prompts and skips Emerald-specific logic
3. **GAME_TYPE** env var is set in both server and client processes for game-type gating throughout the codebase

## New Files

### Core Emulator (`pokemon_red_env/`)

| File | Lines | Purpose |
|------|-------|---------|
| `red_emulator.py` | 718 | PyBoy-backed emulator with same interface as `EmeraldEmulator` |
| `red_memory_reader.py` | 718 | RAM reader for Gen 1 data structures (party, badges, battle, dialog) |
| `red_map_reader.py` | 532 | Hybrid map system: RAM sprites + pre-processed collision maps |
| `red_milestone_tracker.py` | 308 | Badge-gate milestone tracking (8 gyms + Elite Four) |

### Data Files (`pokemon_red_env/data/`)

| File | Purpose |
|------|---------|
| `map_names.json` | Map ID (str) to map name (248 entries) |
| `species_names.json` | Internal species ID to species name (uses Gen 1 internal IDs, not Pokedex numbers) |
| `move_names.json` | Move ID to move name |
| `type_names.json` | Type ID to type name |
| `item_names.json` | Item ID to item name |
| `charmap.json` | Gen 1 character encoding table |
| `processed_map/` | 224 pre-processed collision map files (`.py` with `coll_map` arrays) |
| `pokered/` | Sprite/object data from pokered decompilation |

### Agent Prompts (`agent/prompts/`)

| File | Purpose |
|------|---------|
| `POKEAGENT_RED.md` | Red-specific system prompt (Gen 1 storyline, 9 Game Boy buttons, Red walkthrough parts 1-17) |
| `base_prompt_red.md` | Red-specific strategic guidance (generic decision framework without Emerald gym puzzles) |

### Tests (`pokemon_red_env/test/`)

| File | Purpose |
|------|---------|
| `test_red_states.py` | 14 test states: comprehensive state extraction validation |
| `test_red_state_formatter.py` | 73 assertions: state formatting, map rendering, movement preview |
| `test_red_map.py` | 14 test states: collision map loading, visual map generation |
| `test_red_server.py` | Server integration smoke tests |

## Modified Files

### Server Layer

**`server/app.py`** — Game-type conditional for:
- Emulator instantiation (`RedEmulator` vs `EmeraldEmulator`)
- Map data injection (`red_whole_map` from Red's map reader vs porymap from Emerald's ground truth)
- Visual map generation (Red uses `map_reader.format_map_for_llm()`)
- Dialog state caching (same pattern, different RAM addresses)
- Button validation (reject L/R for Red — Game Boy has no shoulder buttons)

**`server/game_tools.py`** — Two changes:
- `navigate_to_direct()`: loads Red map data as porymap-compatible format for A* pathfinding
- `get_game_state_direct()`: injects `red_whole_map` + porymap grid so movement preview uses world coordinates (fixes coordinate bug on small maps)

### State Formatting

**`utils/state_formatter.py`** — Added:
- `_format_red_map_info()`: formats Red's collision-map-based map data into ASCII map + JSON metadata (same shape as `_format_porymap_info` for Emerald)
- Game-type gating at the top of `format_state_for_llm()` to route Red vs Emerald map formatting
- Porymap grid population from `red_whole_map` for movement preview

**`utils/map_formatter.py`** — Fixed `format_tile_to_symbol()` to handle Red's string-based tile types (e.g., `"GRASS"`, `"WATER"`, `"WarpPoint"`) alongside Emerald's tuple-based tiles.

### Agent Scaffold

**`agent/my_cli_agent.py`** — Game-type conditional for:
- Prompt file selection (`POKEAGENT_RED.md` vs `POKEAGENT.md`)
- Base prompt loading (`base_prompt_red.md` vs `base_prompt.md`)
- Turnstile states (empty for Red — no Fortree gym puzzle)
- Walkthrough tool description (parts 1-17 for Red, 1-21 for Emerald)
- Progress metrics (skip Winona distance for Red)
- Fallback system instruction text
- VLM query timeout (120s via `concurrent.futures`)

**`run.py`** — Added:
- `GAME_TYPE` env var propagation to both server subprocess and client process
- Game-aware banner text
- ROM default auto-switch for Red (`PokemonRed-GBC/pokered.gbc`)

## Key Technical Decisions

### Map System: Hybrid Approach

Rather than VRAM-only (limited to 20x18 viewport) or full pokered JSON extraction, Red uses a **hybrid** approach:

- **Collision data**: Pre-processed from pokered decompilation into 224 `.py` files with `coll_map` arrays. Each cell is a string: `"O"` (walkable), `"X"` (wall), `"GRASS"`, `"WATER"`, `"WarpPoint"`, `"SIGN_*"`, etc.
- **Sprite/NPC data**: Read live from RAM (`wSpriteStateData1` at `0xC100`, `wSpriteStateData2` at `0xC200`). Sprite names resolved from pokered ASM object files.
- **Viewport**: Clamped to map bounds (Emerald-style), never padded with walls. Small maps return actual size.

This gives the agent a full-map ASCII view with NPC positions — equivalent to what Emerald gets from porymap ground truth.

### Gen 1 Specifics

- **Internal species IDs**: Gen 1 uses internal IDs (Bulbasaur = 153/0x99), not Pokedex numbers. Lookups go through `species_names.json`.
- **Party struct**: 44 bytes per slot at `0xD16B + i*0x2C`. Key offsets differ from initial assumptions — level at `+0x21`, max HP at `+0x22` (big-endian), stats at `+0x24`-`+0x2B`.
- **Collision semantics**: Gen 1 lists PASSABLE tiles (opposite of Emerald's IMPASSABLE convention). The map reader inverts this.
- **Dialog detection**: Two-signal approach — `wTextProgressFlags` (0xC6AC) non-zero + VRAM tilemap border check for dialog box tiles.
- **No running shoes or abilities**: Gen 1 movement is slower; no abilities affect gameplay.

### Agent Prompt Adaptations

- **Dialog handling**: Red prompts instruct 1 A-press per step (not A-spam) to prevent infinite NPC dialog loops caused by queued presses re-triggering NPCs after dialog ends.
- **Warp mechanics**: Red warps don't auto-trigger — player must press a direction after arriving on the warp tile. Documented in Navigation Quick Reference.
- **Battle strategy**: Same structure as Emerald but with Gen 1 type chart and no abilities.

## Verification

```bash
# All Red tests
python pokemon_red_env/test/test_red_state_formatter.py  # 73 assertions, 14 states
python pokemon_red_env/test/test_red_states.py            # 14 states
python pokemon_red_env/test/test_red_map.py               # 14 states

# Run the agent
python run.py --game red

# Run the agent with walkthrough objectives
python run.py --game red --direct-objectives categorized_full_game
```

## What Remains

- **Red-specific objectives**: Can be created via `--direct-objectives` but no default objective sequence yet
- **Puzzle solver**: Surge trash cans, Sabrina teleporters, Victory Road boulders — not implemented
- **Battle-specific Red tools**: `get_battle_menu_state`, `get_text_on_screen` from the proposal — not yet needed (agent uses screenshot + state)
- **Outdoor maps**: Large outdoor maps (Route 1, Viridian City, etc.) tested but agent routing across multi-map areas needs more runtime validation

# Pokemon Red Integration

## Overview

Pokemon Red (Gen 1, Game Boy) is a drop-in alternative to Pokemon Emerald. Run with `python run.py --game red`. The `--game` flag (default: `emerald`) gates all game-specific behavior via the `GAME_TYPE` env var.

## Architecture

Same client-server model as Emerald. When `--game red`:
1. `run.py` sets `GAME_TYPE=red` before any agent imports
2. `server/app.py` instantiates `RedEmulator` instead of `EmeraldEmulator`
3. `agents/prompts/paths.py` reads `GAME_TYPE` at import time, resolves Red-specific prompt paths
4. Agent loads Red-specific prompts and skips Emerald-specific logic

## New Files

### Core Emulator (`pokemon_red_env/`)

| File | Lines | Purpose |
|------|-------|---------|
| `red_emulator.py` | 827 | PyBoy-backed emulator; same interface as `EmeraldEmulator`. 48 milestones (GAME_RUNNING → CHAMPION). Dialog state caching with 50ms interval for FPS adjustment (4x during dialog). |
| `red_memory_reader.py` | 790 | RAM reader for Gen 1 structures. 23 hardcoded memory addresses. Party (6x44 bytes at 0xD16B), badges (0xD356 bitmask), battle (0xD057), dialog (two-signal: text_progress 0xC6AC + VRAM border check). |
| `red_map_reader.py` | 804 | Hybrid map system: live NPC positions from RAM (wSpriteStateData1 at 0xC100, 16 sprite slots) + pre-processed collision maps from JSON. 20+ grid symbols. Obstacle detection for cut trees and Card Key gates. |
| `red_milestone_tracker.py` | 308 | Persistent file-based milestone tracking with split times. 48 milestones across 12 phases. |

### Map Utilities (`pokemon_red_env/utils/`)

| File | Lines | Purpose |
|------|-------|---------|
| `red_metatile_behavior.py` | 127 | 21-value behavior enum with integer values matching Emerald's `MetatileBehavior` for pathfinding compatibility |
| `map_overrides.py` | 549 | Manual grid symbol overrides for complex maps (Mt. Moon, Rock Tunnel, Rocket Hideout, Silph Co., etc.) |
| `map_preprocess.py` | 1087 | Preprocesses pokered ASM decompilation into 224 portable JSON map files |

### Data Files (`pokemon_red_env/data/`)

| File | Purpose |
|------|---------|
| `map_names.json` | Map ID → name (248 entries) |
| `species_names.json` | Internal species ID → name (Gen 1 internal IDs, not Pokedex numbers) |
| `move_names.json`, `type_names.json`, `item_names.json` | Standard lookup tables |
| `charmap.json` | Gen 1 character encoding (byte → display char, 0x50 = terminator) |
| `processed_map/` | 224 pre-processed JSON map files with grid, raw_tiles, warps, signs, hidden_objects, npc_data |
| `pokered/` | ASM source data from pokered decompilation |

### Agent Prompts (`agents/prompts/`)

| File | Purpose |
|------|---------|
| `POKEAGENT_RED.md` | Red system prompt for `pokeagent` scaffold (Gen 1 storyline, 9 Game Boy buttons, walkthrough parts 1-17) |
| `SIMPLE_RED.md` | Red system prompt for `simple`/`simplest` scaffolds |
| `system_prompt_red.md` | Red system prompt with prompt optimization enabled |
| `auto-evolve/SYSTEM_PROMPT_RED.md` | Red system prompt for `autoevolve` scaffold |

### Agent Subagents (`agents/subagents/`)

| File | Lines | Purpose |
|------|-------|---------|
| `red_puzzle.py` | 238 | Red puzzle solver with hardcoded knowledge for Rocket Hideout spinner mazes (B2F: 9 phases, B3F: 6 phases) |

### Objectives (`agents/objectives/`)

| File | Lines | Purpose |
|------|-------|---------|
| `all_obj_categorized_red.py` | 1269 | ~78 story objectives + ~22 battling objectives (Pallet Town → Elite Four) |

### Tests (`pokemon_red_env/test/`)

| File | Purpose |
|------|---------|
| `test_red_states.py` | State extraction validation across test save states |
| `test_red_state_formatter.py` | State formatting, map rendering, movement preview |
| `test_red_map.py` | Collision map loading, visual map generation |
| `test_red_server.py` | Server integration smoke tests |
| `test_red_game_state.py` | Game state reading validation |

## Modified Files

### Server Layer

**`server/app.py`** (5014 lines) — Game-type conditionals for:
- Emulator instantiation (`RedEmulator` vs `EmeraldEmulator`)
- Map data: Red uses `memory_reader.map_reader.format_map_for_llm()` and `get_whole_map_data()`; Emerald uses porymap ground truth
- Button validation: rejects L/R for Red (Game Boy has no shoulder buttons)
- Walkthrough URLs: Red has parts 1-17, Emerald 1-21
- Whole-map endpoint: Red returns from `env.get_whole_map()` directly

**`server/game_tools.py`** (854 lines) — Two Red paths:
- `get_game_state_direct()`: injects `red_whole_map` + populates `porymap` field for movement preview compatibility
- `navigate_to_direct()`: loads Red map data from `map_reader.get_whole_map_data()` as porymap-compatible format for A* pathfinding. Key: maps `warp_events` to `warps` key.

### State Formatting

**`utils/state_formatter.py`** (1949 lines):
- `_format_red_map_info()`: formats Red's collision-map data into ASCII map + JSON metadata (same shape as `_format_porymap_info` for Emerald)
- Game-type routing in `_format_map_info()`: dispatches to Red or Emerald formatter
- Facing direction: only enabled for Red (Emerald's is unreliable)
- Coordinate offsets: skipped for Red (only Emerald uses porymap overrides)

**`utils/mapping/map_formatter.py`** (606 lines):
- `_get_behavior_enum()`: returns `RedMetatileBehavior` or `MetatileBehavior` based on `GAME_TYPE`
- Both enums share integer values, so `is_tile_walkable()` and `format_tile_to_symbol()` work without branching

### Agent Scaffold

**`agents/PokeAgent.py`** (3567 lines):
- Button description: 9 Game Boy buttons (no L/R) vs 11 GBA buttons
- Red puzzle subagent: `red_puzzle_agent` tool registered instead of Emerald's `subagent_gym_puzzle`
- `_execute_subagent_red_puzzle()`: dedicated Red puzzle solver method

**`agents/prompts/paths.py`** (71 lines):
- Reads `GAME_TYPE` at import time
- Resolves all prompt paths per game via `_default_system_prompts`, `_simple_prompts`, `_autoevolve_system_prompts`, `_optimization_enabled_prompts` dicts
- `render_prompt()`: substitutes `{game_name}` → "Pokemon Red" or "Pokemon Emerald"

**`agents/subagents/utils/registry.py`**:
- `_is_red()`: checks `GAME_TYPE` env var
- Dynamically registers `_RED_PUZZLE_SPEC` (tool: `red_puzzle_agent`) or `_EMERALD_PUZZLE_SPEC` (tool: `subagent_gym_puzzle`)
- Planner's allowed tools updated via `_puzzle_tool_name()`

**`agents/objectives/direct_objectives.py`** (3924 lines):
- Conditional import: `all_obj_categorized_red.STORY_OBJECTIVES` when `GAME_TYPE=red`

**`run.py`** (554 lines):
- Sets `GAME_TYPE` env var before agent imports (critical: `paths.py` reads at module level)
- Propagates to server subprocess env
- Auto-switches ROM default for Red

## Key Technical Decisions

### Map System: Hybrid Approach

Red uses pre-processed static grids + live RAM sprite positions (not VRAM-only or full pokered extraction):

- **Collision data**: 224 JSON files preprocessed from pokered ASM. Each cell is a single-char symbol (`.`=walkable, `#`=wall, `~`=grass, `W`=water, `D`=door, etc.). Manual overrides for complex maps (spinners, Silph Co. gates).
- **Sprite/NPC data**: Live from RAM (`wSpriteStateData1` at 0xC100). 16 slots (0=player, 1-15=NPCs). Screen pixel deltas converted to map tile coordinates. Names resolved from preprocessed npc_data or fallback "NPC_{slot}".
- **Obstacle tracking**: Cut trees and Card Key gates detected via wTileMap VRAM comparison. Persists cleared state until map re-entry.
- **Viewport**: Clamped to map bounds (Emerald-style), never padded with walls.

### Behavior Enum Compatibility

`RedMetatileBehavior` reuses Emerald's `MetatileBehavior` integer values for shared behaviors (LADDER=97, DOOR=96, WARP_PAD=103, etc.). This means `map_formatter.py` and `pathfinding.py` work without game-specific branching — behavior name substring checks ("DOOR", "WARP", "LADDER") resolve identically.

### Gen 1 Specifics

- **Internal species IDs**: Gen 1 uses internal IDs (Bulbasaur=153/0x99), not Pokedex numbers
- **Party struct**: 44 bytes per slot at 0xD16B. Level at +0x21, max HP at +0x22 (big-endian), stats at +0x24-0x2B. Gen 1 has unified Special (no Sp.Atk/Sp.Def split)
- **Collision semantics**: Gen 1 lists PASSABLE tiles (opposite of Emerald's IMPASSABLE). The preprocessor inverts this
- **Dialog detection**: Two-signal — `wTextProgressFlags` (0xC6AC) non-zero + VRAM tilemap border check for dialog box tiles (0x79-0x7F)
- **Money**: 3-byte BCD big-endian at 0xD347
- **No running shoes, no abilities**: Gen 1 movement is slower; no abilities affect gameplay
- **Badges**: 8-bit bitmask at 0xD356 (Boulder, Cascade, Thunder, Rainbow, Soul, Marsh, Volcano, Earth)

### Agent Prompt Adaptations

- **Dialog handling**: Red prompts instruct 1 A-press per step (not A-spam) to prevent infinite NPC dialog loops from queued presses re-triggering NPCs
- **Warp mechanics**: Red warps don't auto-trigger — player must press a direction after arriving on the warp tile
- **Puzzle knowledge**: Hardcoded spinner maze strategies for Rocket Hideout (B2F/B3F) in `red_puzzle.py`

## Verification

```bash
# Red tests
uv run pytest pokemon_red_env/test/ -v

# Run the agent
uv run python run.py --game red --agent-auto

# Run with categorized objectives
uv run python run.py --game red --agent-auto --direct-objectives categorized_full_game
```

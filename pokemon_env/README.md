# Pokemon environment

Low-level infrastructure for the Pokémon Emerald game: emulator control, memory reading, and map data.

## Components

- **`emulator.py`** — `EmeraldEmulator`: loads the ROM (mGBA), runs the game loop, and executes button input (`press_key`, `press_buttons`). Handles frame advancement and adaptive speed (e.g. faster when in dialog).
- **`memory_reader.py`** — `PokemonEmeraldReader`: reads structured game state from emulator RAM (party, position, map, etc.). Uses cached memory regions and frame-based cache invalidation. **Do not modify for competitive submissions.**
- **`emerald_utils.py`**, **`enums.py`**, **`types.py`**, **`utils.py`** — Data structures and helpers for memory layout and parsing.
- **`porymap_paths.py`** — Path resolution for Porymap data (maps, tilesets, layouts).
- **`porymap/`** — Pokeemerald decompilation data (e.g. `data/maps`, `data/tilesets`) used for ground-truth map info.

## Map and state

Map data is loaded from the Porymap layout; the server and `utils/mapping/` (e.g. `porymap_json_builder`, `state_formatter`) combine it with runtime state to produce the ASCII-style grid and JSON state (item/npc locations) sent to the agent.

# Pokemon Infrastructure Architecture

This document describes the low-level infrastructure for interacting with the Pokemon Emerald game state, including memory reading, map data integration, and emulator control.

## Overview

The system uses a layered architecture to bridge the gap between the raw Game Boy Advance (GBA) memory and high-level agent understanding.
- **Emulator Layer**: `mgba` (via `libmgba`) provides the core emulation.
- **Memory Layer**: `PokemonEmeraldReader` extracts structured game data from memory.
- **Map Layer**: `Porymap` integration provides ground-truth map layouts and metadata.
- **Control Layer**: `EmeraldEmulator` manages input execution and frame advancement.

## 1. Emulator Control (`pokemon_env/emulator.py`)

### `EmeraldEmulator` Class
- **Initialization**: Loads the ROM via `mgba.core.load_path()`, creating a temporary copy to avoid save file conflicts.
- **Input Execution**:
  - `press_key(key, frames)`: Holds a button for N frames.
  - `press_buttons(buttons, hold_frames, release_frames)`: Executes a sequence of button presses.
  - **Button Mapping**: Translates string keys ("A", "B", "UP") to `lib.GBA_KEY_*` constants.
- **Frame Management**:
  - `tick(frames)`: Advances the emulator by N frames.
  - **Adaptive Speed**: Detects dialog states (`is_in_dialog()`) and speeds up emulation (4x) to skip text faster.

## 2. Memory Reading (`pokemon_env/memory_reader.py`)

### `PokemonEmeraldReader` Class
- **Purpose**: Reads structured game state directly from emulator RAM.
- **Mechanism**:
  - Uses `cffi` to access raw memory via `mgba._pylib`.
  - **Region Caching**: Caches 64KB memory blocks (`_get_memory_region`) to minimize expensive FFI calls.
  - **Cache Invalidation**: Clears the cache on every frame advance callback.

### Data Structures (`emerald_utils.py`)
- **MemoryAddresses**: A dataclass collecting critical memory pointers (e.g., `gPlayerParty`, `gSaveBlock1Ptr`).
- **PokemonDataStructure**: Defines the byte layout for encrypted Pokemon data.
- **Parsing Logic**:
  - `parse_pokemon()`: Decrypts Pokemon data using the XOR key (`otId ^ personality`).
  - Handles the 24 permutations of substruct order based on personality value.

## 3. Porymap Integration (Map Data)

### Purpose
To provide the agent with ground-truth map data (walls, warps, ledges) that cannot be easily inferred from the visual screen alone.

### Data Location
- **In-repo only**: `pokemon_env/porymap/` (pokeemerald layout: `data/maps`, `data/tilesets`, `data/layouts`)
- **Path resolution**: `pokemon_env/porymap_paths.py` → `get_porymap_root()`

### Components
- **`utils/pokeemerald_parser.py`**:
  - `PokeemeraldMapLoader`: Reads JSON map definitions from the `pokeemerald` decompilation project.
  - `PokeemeraldLayoutParser`: Parses binary layout files (`map.bin`) and tileset attributes (`metatile_attributes.bin`).
- **`utils/porymap_json_builder.py`**:
  - Combines static map data with runtime state to produce a comprehensive JSON representation of the current area.
  - Handles dynamic map selection (e.g., "Gym Lobby" vs. "Gym Puzzle").

### Data Flow
1. **Agent Request**: Calls `get_comprehensive_state()`.
2. **Memory Read**: Identifying the current map bank and ID from RAM.
3. **Lookup**: Mapping (Bank, ID) to a Porymap file path via `ROM_TO_PORYMAP_MAP`.
4. **Load & Format**: Loading the Porymap JSON, overlaying dynamic objects (NPCs), and returning an ASCII-style grid to the agent.

## 4. Software Engineering Principles Deviation

**Tight Coupling (High Dependency)**
- **Issue**: `EmeraldEmulator` is tightly coupled to `PokemonEmeraldReader`, which is tightly coupled to specific memory addresses.
- **Principle**: *Loose Coupling*.
- **Impact**: Changes to the memory layout (e.g., a different ROM version) would break the entire chain. Abstraction layers are thin.

**Magic Numbers (Maintainability)**
- **Issue**: The codebase is littered with hardcoded memory addresses (`0x020244e9`), bit masks (`0x03FF`), and frame counts (`hold_frames=16`).
- **Principle**: *No Magic Numbers* / *Self-Documenting Code*.
- **Impact**: Makes the code difficult to understand and modify. These should be defined as named constants in a central configuration file.

**Silent Failures (Error Handling)**
- **Issue**: Memory reading and map loading often use broad `try/except` blocks that return `None` or empty lists on failure.
- **Principle**: *Fail Fast*.
- **Impact**: Debugging "why is the map empty?" is difficult because the underlying error (e.g., missing file) was swallowed.

**Performance Bottlenecks (Efficiency)**
- **Issue**: Scanning memory for map buffers or reloading large JSON files on every step can be slow.
- **Principle**: *Efficiency* / *Premature Optimization (Avoidance)* - though here it might be actual performance debt.
- **Impact**: Slows down the agent's step rate. Better caching strategies or async loading could help.

**Code Duplication (DRY)**
- **Issue**: Address definitions appear in both `emerald_utils.py` and `memory_reader.py`. Map reading logic is duplicated between `memory_reader.py` and legacy `enums.py`.
- **Principle**: *Don't Repeat Yourself (DRY)*.
- **Impact**: Divergent definitions can lead to subtle bugs where one part of the system reads from the wrong address.

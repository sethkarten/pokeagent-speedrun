# Pokemon Red Integration Proposal

## What Exists Today

`pokemon_env/red_emulator.py` provides a `RedEmulator` class using [PyBoy](https://github.com/Baekalfen/PyBoy) that implements the same public interface as `EmeraldEmulator`:

| Capability | Status | Notes |
|-----------|--------|-------|
| ROM loading & init | Working | `PyBoy(rom_path, window="null")` |
| Screenshot (PIL) | Working | 160x144 RGB via `pyboy.screen.image` |
| Button input | Working | `press_buttons`, `run_frame_with_buttons`, `press_key` |
| Save/load state | Working | PyBoy native state format |
| Raw memory read | Working | `pyboy.memory[addr]` — flat 64KB Game Boy address space |
| Player coords | Working | Reads `0xD361`/`0xD362` directly |
| Map ID | Working | Reads `0xD35E`, returns `MAP_{id}` |
| Money (BCD) | Working | Reads 3-byte BCD at `0xD347` |
| Party species IDs | Working | Reads `0xD163` count + `0xD164` species array |
| Badges | Working | Reads bitfield at `0xD356` |
| Frame capture thread | Working | Background thread for streaming |
| Comprehensive state | Partial | Returns coords/money/badges; map/battle/dialog stubs |
| Milestones | Stub | No Red milestones defined yet |
| Memory reader | Missing | No `PokemonRedReader` — needed for dialog, battle, items, full party stats |
| Map/pathfinding | Missing | No pokered map data, no ASCII grid, no A* |

## How to Use RedEmulator

```python
from pokemon_env.red_emulator import RedEmulator

emu = RedEmulator("PokemonRed-GBC/pokered.gbc", headless=True)
emu.initialize()

img = emu.get_screenshot()          # PIL Image (160x144)
emu.press_buttons(["a", "up"])      # Button input
state = emu.get_comprehensive_state()  # Game state dict
diag = emu.test_memory_reading()    # Memory diagnostics

emu.save_state("checkpoint.state")
emu.load_state("checkpoint.state")
emu.stop()
```

Run inline tests: `python -m pokemon_env.red_emulator`

## What's Needed for Full Agent Support

### 1. PokemonRedReader (memory_reader equivalent)

Create `pokemon_env/red_memory_reader.py` sourcing addresses from [pokered decompilation](https://github.com/pret/pokered):

- **Dialog detection** — text progress at `0xC6AC`, window state
- **Battle state** — `0xD057` (battle type), enemy party data
- **Full party stats** — HP, level, moves, types from party struct at `0xD16B`+
- **Items** — bag items at `0xD31D`, item count at `0xD31C`
- **Game state** — overworld/battle/menu/dialog detection
- **Pokedex** — owned/seen bitfields at `0xD2F7`/`0xD30A`

### 2. Map System

Options (pick one):
- **Minimal**: Read the 2D tile map from VRAM + collision data from ROM. Enough for basic ASCII grid + pathfinding.
- **Full**: Port the pokered map data (from [pokered](https://github.com/pret/pokered) `data/maps/`) into the same porymap JSON format. Enables ground-truth collision grids and A* pathfinding.

### 3. Milestones

Define Red-specific milestones in `RedEmulator.check_and_update_milestones()`:

```
PALLET_TOWN → VIRIDIAN_CITY → PEWTER_CITY → BROCK_DEFEATED →
MT_MOON → CERULEAN_CITY → MISTY_DEFEATED → SS_ANNE →
LT_SURGE_DEFEATED → ROCK_TUNNEL → LAVENDER_TOWER →
CELADON_CITY → ERIKA_DEFEATED → ROCKET_HIDEOUT →
SILPH_CO → SABRINA_DEFEATED → KOGA_DEFEATED →
BLAINE_DEFEATED → GIOVANNI_DEFEATED → VICTORY_ROAD →
ELITE_FOUR → CHAMPION
```

### 4. Direct Objectives & Walkthrough

- Create `agent/red_direct_objectives.py` with Red story progression
- Point `get_walkthrough()` to Red/Blue Bulbapedia walkthrough

### 5. Prompts

- Create `POKEAGENT_RED.md` system instructions
- Create `base_prompt_red.md` base prompt
- Create `agent/red_puzzle_solver.py` for Red gym puzzles (Surge's trash cans, Sabrina's teleporters, etc.)

## Integration Path

### Option A: CLI Flag (minimal change)

Add `--game red` flag to `run.py` and `server/app.py`:

```python
# server/app.py setup_environment()
if game == "red":
    from pokemon_env.red_emulator import RedEmulator
    env = RedEmulator(rom_path=rom_path)
else:
    from pokemon_env.emulator import EmeraldEmulator
    env = EmeraldEmulator(rom_path=rom_path)
env.initialize()
```

The rest of the server works unchanged — all endpoints use the same `env.get_screenshot()`, `env.run_frame_with_buttons()`, `env.get_comprehensive_state()` interface.

### Option B: Game Config Abstraction (scalable)

For supporting many games, introduce a config layer:

```
game_config/
  base.py         # Abstract: get_addresses(), get_milestones(), get_species_map()
  emerald.py      # Current Emerald constants extracted here
  red.py          # Red constants
```

Then parametrize: `MemoryReader(game_config)`, `MilestoneTracker(game_config)`, `StateFormatter(game_config)`.

## Effort Estimate by Component

| Component | Complexity | Depends On |
|-----------|-----------|------------|
| Wire into server/app.py (Option A) | Low | Nothing |
| PokemonRedReader (basic) | Medium | pokered address research |
| PokemonRedReader (full) | High | pokered decompilation deep-dive |
| Map system (minimal ASCII) | Medium | PokemonRedReader |
| Map system (full porymap) | High | pokered map data extraction |
| Milestones | Low | PokemonRedReader (location detection) |
| Direct objectives | Medium | Red walkthrough knowledge |
| Prompts & puzzle solver | Low | Game knowledge |

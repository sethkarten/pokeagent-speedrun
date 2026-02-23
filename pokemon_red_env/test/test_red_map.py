"""Map test for Pokemon Red — visual map + coll_map ground truth.

For each .state file in PokemonRed-GBC/test_states/:
  1. Load ROM + state, tick 1 frame
  2. Save screenshot          → output/<stem>/<stem>.png
  3. Save ASCII visual map    → output/<stem>/<stem>_visual_map.txt
  4. Save full coll_map JSON  → output/<stem>/<stem>_coll_map.json

The visual map uses format_map_for_llm() (radius=7 for a 15×15 view).
The coll_map JSON contains the *complete* map grid for the current location
so it can be used as ground truth when checking agent navigation logic.

Run:
    python pokemon_red_env/test/test_red_map.py
"""

import json
import pathlib
import sys
import traceback

# ---------------------------------------------------------------------------
THIS_DIR   = pathlib.Path(__file__).parent          # pokemon_red_env/test/
REPO_ROOT  = THIS_DIR.parent.parent                 # pokeagent-speedrun/
ROM_PATH   = REPO_ROOT / "PokemonRed-GBC" / "pokered.gbc"
STATES_DIR = REPO_ROOT / "PokemonRed-GBC" / "test_states"
OUTPUT_DIR = THIS_DIR / "output"

sys.path.insert(0, str(REPO_ROOT))
# ---------------------------------------------------------------------------

VISUAL_MAP_RADIUS = 7   # → 15×15 grid centred on player


def run_state(state_path: pathlib.Path, out_dir: pathlib.Path):
    from pokemon_red_env.red_emulator import RedEmulator

    stem = state_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  State : {state_path.name}")
    print(f"  Output: {out_dir.relative_to(REPO_ROOT)}")
    print(f"{'='*60}")

    emu = RedEmulator(str(ROM_PATH), headless=True)
    try:
        emu.initialize()
        emu.load_state(path=str(state_path))
        emu.tick(1)

        mr = emu.memory_reader           # RedMemoryReader
        map_reader = mr.map_reader       # RedMapReader (attached in initialize())

        # ---- screenshot ------------------------------------------------
        img = emu.get_screenshot()
        if img is not None:
            img_path = out_dir / f"{stem}.png"
            img.save(img_path)
            print(f"  [OK] screenshot        → {img_path.name}")
        else:
            print("  [WARN] screenshot returned None")

        # ---- visual map (ASCII) ----------------------------------------
        map_name  = map_reader.read_map_name()
        player_x, player_y = map_reader.read_player_coords()
        visual    = map_reader.format_map_for_llm(radius=VISUAL_MAP_RADIUS)

        vis_path  = out_dir / f"{stem}_visual_map.txt"
        header    = (
            f"Map    : {map_name}\n"
            f"Player : ({player_x}, {player_y})  [coll_map col, row]\n"
            f"Radius : {VISUAL_MAP_RADIUS}  ({2*VISUAL_MAP_RADIUS+1}×{2*VISUAL_MAP_RADIUS+1} view)\n"
            f"Legend : P=player  .=walkable  #=wall  G=grass  ~=water\n"
            f"         W=warp    N=NPC       s=sign   T=Cut    v=ledge-down\n"
            f"         <=ledge-left  >=ledge-right  c=counter  ?=unknown\n"
            f"\n"
        )
        vis_path.write_text(header + visual + "\n", encoding="utf-8")
        print(f"  [OK] visual map        → {vis_path.name}")

        # ---- full coll_map (ground truth) --------------------------------
        full = map_reader.get_full_coll_map()
        coll_path = out_dir / f"{stem}_coll_map.json"
        with open(coll_path, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2)
        print(f"  [OK] coll_map ground truth → {coll_path.name}")

        # ---- console summary -------------------------------------------
        coll_map  = full["coll_map"]
        h         = full["map_height"]
        w         = full["map_width"]
        player_cell = coll_map[player_y][player_x] if (
            coll_map and 0 <= player_y < h and 0 <= player_x < w
        ) else "OUT_OF_BOUNDS"

        print(f"       map_name    : {map_name}")
        print(f"       coll_map    : {w} cols × {h} rows")
        print(f"       player pos  : ({player_x}, {player_y})  cell='{player_cell}'")
        print()
        print("  Visual map:")
        for line in visual.splitlines():
            print(f"    {line}")

    finally:
        emu.stop()


def main():
    if not ROM_PATH.exists():
        print(f"ERROR: ROM not found at {ROM_PATH}")
        sys.exit(1)

    state_files = sorted(STATES_DIR.glob("*.state"))
    if not state_files:
        print(f"ERROR: No .state files found in {STATES_DIR}")
        sys.exit(1)

    print(f"ROM    : {ROM_PATH}")
    print(f"States : {len(state_files)} files in {STATES_DIR.relative_to(REPO_ROOT)}")
    print(f"Output : {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    passed = 0
    failed = []

    for state_path in state_files:
        stem    = state_path.stem
        out_dir = OUTPUT_DIR / stem
        try:
            run_state(state_path, out_dir)
            passed += 1
        except Exception as exc:
            print(f"\n  [FAIL] {stem}: {exc}")
            traceback.print_exc()
            failed.append(stem)

    print(f"\n{'='*60}")
    print(f"Results : {passed}/{len(state_files)} passed")
    if failed:
        print(f"Failed  : {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All states OK.")


if __name__ == "__main__":
    main()

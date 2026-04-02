"""Test RedEmulator against each save state in PokemonRed-GBC/test_states/.

For each .state file:
  1. Load ROM + state
  2. Tick 1 frame
  3. Save screenshot  → <stem>.png
  4. Save comprehensive state → <stem>.json
  5. Save milestones  → <stem>_milestones.json

All output files are written next to this script in
  pokemon_red_env/test/output/<stem>/
"""

import json
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the script works from any CWD
# ---------------------------------------------------------------------------
THIS_DIR   = pathlib.Path(__file__).parent          # pokemon_red_env/test/
REPO_ROOT  = THIS_DIR.parent.parent                 # pokeagent-speedrun/
ROM_PATH   = REPO_ROOT / "PokemonRed-GBC" / "pokered.gbc"
STATES_DIR = REPO_ROOT / "PokemonRed-GBC" / "test_states"
OUTPUT_DIR = THIS_DIR / "output"

sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert non-serialisable objects to JSON-safe equivalents."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    # PIL Image → skip (replaced with a placeholder string)
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            return f"<PIL.Image {obj.size} {obj.mode}>"
    except ImportError:
        pass
    # numpy arrays
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    except ImportError:
        pass
    # bytes
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes len={len(obj)}>"
    return obj


def run_state(state_path: pathlib.Path, rom_path: pathlib.Path, out_dir: pathlib.Path):
    from pokemon_red_env.red_emulator import RedEmulator

    stem = state_path.stem                     # e.g. "pokered_overworld_redhouse1f"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  State : {state_path.name}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    emu = RedEmulator(str(rom_path), headless=True)
    try:
        emu.initialize()
        emu.load_state(path=str(state_path))
        emu.tick(1)

        # ---- screenshot ------------------------------------------------
        img = emu.get_screenshot()
        if img is not None:
            img_path = out_dir / f"{stem}.png"
            img.save(img_path)
            print(f"  [OK] screenshot → {img_path.name}")
        else:
            print("  [WARN] screenshot returned None")

        # ---- comprehensive state ----------------------------------------
        state = emu.get_comprehensive_state()

        # Swap PIL image out before serialising
        if "visual" in state and "screenshot" in state["visual"]:
            state["visual"]["screenshot"] = (
                f"<PIL.Image {img.size} {img.mode}>" if img else None
            )

        safe_state = _json_safe(state)
        json_path = out_dir / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump(safe_state, f, indent=2)
        print(f"  [OK] comprehensive state → {json_path.name}")

        # ---- milestones ------------------------------------------------
        milestone_path = out_dir / f"{stem}_milestones.json"
        emu.milestone_tracker.save_milestones_for_state(str(milestone_path))
        print(f"  [OK] milestones → {milestone_path.name}")

        # ---- quick sanity prints ----------------------------------------
        player  = state.get("player", {})
        game    = state.get("game", {})
        map_s   = state.get("map", {})
        tiles   = map_s.get("tiles")

        print(f"       location  : {player.get('location')}")
        print(f"       position  : {player.get('position')}")
        print(f"       money     : {game.get('money')}")
        print(f"       badges    : {game.get('badges')}")
        print(f"       party_len : {len(player.get('party') or [])}")
        print(f"       tiles grid: {len(tiles)}×{len(tiles[0])} "
              f"({tiles[len(tiles)//2][len(tiles[0])//2][1]} at centre)"
              if tiles else "       tiles: None")
        trav = map_s.get("traversability")
        print(f"       trav  grid: {len(trav)}×{len(trav[0])}"
              if trav else "       traversability: None")

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

    print(f"ROM   : {ROM_PATH}")
    print(f"States: {len(state_files)} files in {STATES_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    passed = 0
    failed = []

    for state_path in state_files:
        stem    = state_path.stem
        out_dir = OUTPUT_DIR / stem
        try:
            run_state(state_path, ROM_PATH, out_dir)
            passed += 1
        except Exception as exc:
            import traceback
            print(f"\n  [FAIL] {stem}: {exc}")
            traceback.print_exc()
            failed.append(stem)

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(state_files)} passed")
    if failed:
        print(f"Failed : {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All states OK.")


if __name__ == "__main__":
    main()

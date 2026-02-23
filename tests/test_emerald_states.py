"""Test EmeraldEmulator against save states.

Scans two directories for .state files:
  - Emerald-GBAdvance/          (production/checkpoint states)
  - tests/states/               (unit-test states)

For each .state file:
  1. Load ROM + state
  2. Tick 1 frame
  3. Save screenshot            → output/<stem>/<stem>.png
  4. Save comprehensive state   → output/<stem>/<stem>.json
  5. Save milestones            → output/<stem>/<stem>_milestones.json

Output lands in tests/output/<stem>/.
"""

import json
import os
import pathlib
import sys
import traceback

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
THIS_DIR   = pathlib.Path(__file__).parent          # tests/
REPO_ROOT  = THIS_DIR.parent                        # pokeagent-speedrun/
ROM_PATH   = REPO_ROOT / "Emerald-GBAdvance" / "rom.gba"
OUTPUT_DIR = THIS_DIR / "output"

STATE_DIRS = [
    REPO_ROOT / "Emerald-GBAdvance",
    THIS_DIR / "states",
]

sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert non-serialisable objects to JSON-safe equivalents."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            return f"<PIL.Image {obj.size} {obj.mode}>"
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    except ImportError:
        pass
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes len={len(obj)}>"
    # Enums (MetatileBehavior etc.)
    if hasattr(obj, "name"):
        try:
            return obj.name
        except Exception:
            pass
    return obj


def run_state(state_path: pathlib.Path, rom_path: pathlib.Path, out_dir: pathlib.Path):
    from pokemon_env.emulator import EmeraldEmulator

    stem = state_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  State : {state_path.relative_to(REPO_ROOT)}")
    print(f"  Output: {out_dir.relative_to(REPO_ROOT)}")
    print(f"{'='*60}")

    emu = EmeraldEmulator(str(rom_path), headless=True)
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

        # Replace PIL image with placeholder before serialising
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
        player = state.get("player", {})
        game   = state.get("game", {})
        map_s  = state.get("map", {})
        tiles  = map_s.get("tiles")

        print(f"       location  : {player.get('location')}")
        print(f"       position  : {player.get('position')}")
        print(f"       money     : {game.get('money')}")
        print(f"       badges    : {game.get('badges')}")
        print(f"       party_len : {len(player.get('party') or [])}")
        if tiles:
            centre = tiles[len(tiles) // 2][len(tiles[0]) // 2]
            print(f"       tiles grid: {len(tiles)}×{len(tiles[0])} "
                  f"({centre[1] if len(centre) > 1 else centre} at centre)")
        else:
            print("       tiles     : None")
        trav = map_s.get("traversability")
        if trav:
            print(f"       trav  grid: {len(trav)}×{len(trav[0])}")
        else:
            print("       traversability: None")

    finally:
        emu.stop()


def collect_states() -> list:
    """Return sorted list of pathlib.Path for all .state files in STATE_DIRS."""
    states = []
    for d in STATE_DIRS:
        if d.is_dir():
            states.extend(sorted(d.glob("*.state")))
    return states


def main():
    if not ROM_PATH.exists():
        print(f"ERROR: ROM not found at {ROM_PATH}")
        sys.exit(1)

    state_files = collect_states()
    if not state_files:
        print(f"ERROR: No .state files found in {STATE_DIRS}")
        sys.exit(1)

    print(f"ROM   : {ROM_PATH}")
    print(f"States: {len(state_files)} files")
    for d in STATE_DIRS:
        count = len(list(d.glob("*.state"))) if d.is_dir() else 0
        print(f"         {count:3d}  {d.relative_to(REPO_ROOT)}")
    print(f"Output: {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    passed = 0
    failed = []

    for state_path in state_files:
        stem    = state_path.stem
        out_dir = OUTPUT_DIR / stem
        try:
            run_state(state_path, ROM_PATH, out_dir)
            passed += 1
        except Exception as exc:
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

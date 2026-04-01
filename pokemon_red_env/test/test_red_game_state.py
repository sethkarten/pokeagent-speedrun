#!/usr/bin/env python3
"""
Comprehensive tests for mcp_get_game_state pipeline on Pokemon Red.

Loads real emulator states (.state files and .zip checkpoint archives),
runs the full get_game_state_direct() pipeline, and validates state_text,
location, and raw_state across all game states (overworld, battle, dialog,
menu) to verify what the LLM actually sees.

Usage:
    python pokemon_red_env/test/test_red_game_state.py
"""

import json
import os
import pathlib
import sys
import zipfile

# Set GAME_TYPE before any project imports
os.environ["GAME_TYPE"] = "red"

THIS_DIR = pathlib.Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent
ROM_PATH = REPO_ROOT / "PokemonRed-GBC" / "pokered.gbc"
STATES_DIR = REPO_ROOT / "PokemonRed-GBC" / "test_states"
INIT_ZIP = REPO_ROOT / "PokemonRed-GBC" / "red_init.zip"
OUTPUT_DIR = THIS_DIR / "output_game_state"

sys.path.insert(0, str(REPO_ROOT))

from pokemon_red_env.red_emulator import RedEmulator
from server.game_tools import get_game_state_direct
from utils.state_formatter import format_state_for_llm

PASSED = 0
FAILED = 0
SEP = "=" * 70


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  {detail}")


def print_state_text(state_text, max_lines=120, indent="    "):
    if not state_text or not state_text.strip():
        print(f"{indent}(empty)")
        return
    lines = state_text.split("\n")
    for line in lines[:max_lines]:
        print(f"{indent}{line}")
    if len(lines) > max_lines:
        print(f"{indent}... ({len(lines) - max_lines} more lines)")


# ---------------------------------------------------------------------------
# Emulator helpers
# ---------------------------------------------------------------------------

def load_emu(state_path):
    """Create RedEmulator, load a .state file, tick 1 frame."""
    emu = RedEmulator(str(ROM_PATH), headless=True)
    emu.initialize()
    emu.load_state(path=str(state_path))
    emu.tick(1)
    return emu


def load_emu_from_zip(zip_path):
    """Extract checkpoint.state from a backup zip and load it."""
    with zipfile.ZipFile(zip_path) as z:
        candidates = [n for n in z.namelist() if n.endswith("checkpoint.state")]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint.state in {zip_path}")
        state_bytes = z.read(candidates[0])
    emu = RedEmulator(str(ROM_PATH), headless=True)
    emu.initialize()
    emu.load_state(state_bytes=state_bytes)
    emu.tick(1)
    return emu


def run_pipeline(emu):
    """Run the full mcp_get_game_state pipeline."""
    return get_game_state_direct(emu, format_state_for_llm)


def save_output(name, result):
    """Save state_text and raw_state JSON for inspection."""
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    # state_text
    with open(out_dir / "state_text.txt", "w") as f:
        f.write(result.get("state_text", ""))
    # raw_state (skip non-serializable)
    try:
        with open(out_dir / "raw_state.json", "w") as f:
            json.dump(result.get("raw_state", {}), f, indent=2, default=str)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Common validation
# ---------------------------------------------------------------------------

PARTY_REQUIRED_KEYS = {
    "species_name", "level", "current_hp", "max_hp", "status",
    "moves", "move_pp", "types", "attack", "defense", "speed", "special",
}


def validate_common(result, label, min_party=1, allow_title=False):
    """Structural assertions that apply to every state."""
    check(f"{label}: success", result.get("success") is True)

    state_text = result.get("state_text", "")
    check(f"{label}: state_text non-empty", isinstance(state_text, str) and len(state_text) > 50,
          f"len={len(state_text)}")

    location = result.get("location", "")
    if not allow_title:
        check(f"{label}: location valid", location and location != "Unknown",
              f"got '{location}'")

    raw = result.get("raw_state", {})
    for key in ("visual", "player", "game", "map"):
        check(f"{label}: raw_state has '{key}'", key in raw)

    pos = raw.get("player", {}).get("position", {})
    check(f"{label}: position has x/y",
          isinstance(pos.get("x"), (int, float)) and isinstance(pos.get("y"), (int, float)),
          f"got {pos}")

    party = raw.get("player", {}).get("party", [])
    check(f"{label}: party has >= {min_party} pokemon", len(party) >= min_party,
          f"got {len(party)}")

    # Validate party pokemon fields (Gen 1 stats)
    for i, pkmn in enumerate(party):
        missing = PARTY_REQUIRED_KEYS - set(pkmn.keys())
        check(f"{label}: party[{i}] has all required fields",
              len(missing) == 0,
              f"missing: {missing}")

    # JSON-serializable check (no PIL Images or numpy arrays should leak)
    try:
        json.dumps(raw, default=str)
        check(f"{label}: raw_state is JSON-serializable", True)
    except (TypeError, ValueError) as e:
        check(f"{label}: raw_state is JSON-serializable", False, str(e))


# ---------------------------------------------------------------------------
# Test: Overworld states
# ---------------------------------------------------------------------------

def test_overworld_states():
    print(f"\n{SEP}")
    print("=== TEST: Overworld States ===")
    print(SEP)

    state_path = STATES_DIR / "pokered_overworld_redhouse1f.state"
    if not state_path.exists():
        print(f"  [SKIP] {state_path.name} not found")
        return

    emu = load_emu(state_path)
    try:
        result = run_pipeline(emu)
        label = "overworld_redhouse1f"
        # This state is before getting a starter, so party may be empty
        validate_common(result, label, min_party=0)
        save_output(label, result)

        st = result["state_text"]
        raw = result["raw_state"]

        # State text should have full map
        check(f"{label}: state_text has MAP (FULL)", "MAP (FULL)" in st,
              f"first 500 chars: {st[:500]}")
        check(f"{label}: state_text has ASCII Map", "ASCII Map" in st)
        check(f"{label}: state_text has location name",
              result["location"] in st,
              f"location={result['location']}")
        check(f"{label}: state_text has Legend", "Legend" in st)
        check(f"{label}: state_text has player marker 'I'", "I" in st)
        check(f"{label}: state_text has party info",
              "Pokemon Party" in st or "pokemon)" in st or "No Pokemon in party" in st,
              f"first 800: {st[:800]}")

        # Raw state checks
        game = raw.get("game", {})
        check(f"{label}: game_state is overworld",
              game.get("game_state") == "overworld",
              f"got '{game.get('game_state')}'")
        check(f"{label}: is_in_battle is False",
              game.get("is_in_battle") is False)

        # Map data
        map_data = raw.get("map", {})
        whole_map = map_data.get("red_whole_map", {})
        check(f"{label}: red_whole_map has grid",
              isinstance(whole_map.get("grid"), list) and len(whole_map["grid"]) > 0)
        check(f"{label}: red_whole_map has dimensions",
              "width" in whole_map.get("dimensions", {}) and "height" in whole_map.get("dimensions", {}))
        check(f"{label}: red_whole_map has warp_events",
              "warp_events" in whole_map)
        check(f"{label}: red_whole_map has objects",
              "objects" in whole_map)

        print(f"\n  --- state_text ({len(st)} chars) ---")
        print_state_text(st)
    finally:
        emu.stop()


# ---------------------------------------------------------------------------
# Test: Battle states
# ---------------------------------------------------------------------------

def test_battle_states():
    print(f"\n{SEP}")
    print("=== TEST: Battle States ===")
    print(SEP)

    battle_files = [
        "pokered_battle_1.state",
        "pokered_battle_2.state",
        "pokered_battle_3_pre.state",
    ]

    for fname in battle_files:
        state_path = STATES_DIR / fname
        if not state_path.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        stem = state_path.stem
        print(f"\n  --- {fname} ---")
        emu = load_emu(state_path)
        try:
            result = run_pipeline(emu)
            validate_common(result, stem)
            save_output(stem, result)

            st = result["state_text"]
            raw = result["raw_state"]
            game = raw.get("game", {})
            battle_info = game.get("battle_info")

            is_battle = game.get("is_in_battle", False)

            if is_battle:
                # State text battle content
                check(f"{stem}: state_text has BATTLE MODE",
                      "BATTLE MODE" in st or "BATTLE" in st.upper())
                check(f"{stem}: state_text has player pokemon section",
                      "YOUR" in st.upper() and "MON" in st.upper(),
                      f"searching for YOUR POKEMON in state_text")
                check(f"{stem}: state_text has opponent pokemon section",
                      "OPPONENT" in st.upper(),
                      f"searching for OPPONENT in state_text")
                check(f"{stem}: state_text has HP info", "HP:" in st)
                check(f"{stem}: state_text has moves", "Moves:" in st or "PP:" in st)
                check(f"{stem}: state_text has party status",
                      "PARTY STATUS" in st.upper() or "PARTY" in st.upper())
                check(f"{stem}: state_text has available actions",
                      "FIGHT" in st or "AVAILABLE ACTIONS" in st.upper())

                # Raw state battle checks
                check(f"{stem}: is_in_battle is True", is_battle is True)
                check(f"{stem}: battle_info is not None", battle_info is not None)

                if battle_info:
                    # Player pokemon
                    player_pkmn = battle_info.get("player_pokemon")
                    check(f"{stem}: battle_info has player_pokemon",
                          player_pkmn is not None)
                    if player_pkmn:
                        for key in ("species_name", "level", "current_hp", "max_hp",
                                    "status", "moves", "move_pp"):
                            check(f"{stem}: player_pokemon has '{key}'",
                                  key in player_pkmn,
                                  f"keys: {list(player_pkmn.keys())}")

                    # Opponent pokemon
                    opp_pkmn = battle_info.get("opponent_pokemon")
                    check(f"{stem}: battle_info has opponent_pokemon",
                          opp_pkmn is not None)
                    if opp_pkmn:
                        for key in ("species", "level", "current_hp", "max_hp",
                                    "status", "types", "moves"):
                            check(f"{stem}: opponent_pokemon has '{key}'",
                                  key in opp_pkmn,
                                  f"keys: {list(opp_pkmn.keys())}")
                        # Opponent stats
                        check(f"{stem}: opponent has stats dict",
                              isinstance(opp_pkmn.get("stats"), dict),
                              f"got {type(opp_pkmn.get('stats'))}")

                    # Battle interface
                    bi = battle_info.get("battle_interface", {})
                    actions = bi.get("available_actions", [])
                    check(f"{stem}: available_actions has FIGHT/ITEM/PKMN/RUN",
                          set(actions) == {"FIGHT", "ITEM", "PKMN", "RUN"},
                          f"got {actions}")

                    # Battle type
                    check(f"{stem}: battle_type is wild or trainer",
                          battle_info.get("battle_type") in ("wild", "trainer"),
                          f"got '{battle_info.get('battle_type')}'")
            else:
                # battle_3_pre might be a pre-battle transition
                print(f"  [NOTE] {stem}: is_in_battle=False (pre-battle transition?)")
                check(f"{stem}: pre-battle state still succeeds", result["success"] is True)

            print(f"\n  state_text ({len(st)} chars):")
            print_state_text(st)
        finally:
            emu.stop()


# ---------------------------------------------------------------------------
# Test: Dialog states
# ---------------------------------------------------------------------------

def test_dialog_states():
    print(f"\n{SEP}")
    print("=== TEST: Dialog States ===")
    print(SEP)

    dialog_files = [
        "pokered_dialog_mom.state",
        "pokered_dialog_npc.state",
        "pokered_dialog_oak.state",
        "pokered_dialog_tv.state",
    ]

    for fname in dialog_files:
        state_path = STATES_DIR / fname
        if not state_path.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        stem = state_path.stem
        print(f"\n  --- {fname} ---")
        emu = load_emu(state_path)
        try:
            result = run_pipeline(emu)
            # Dialog states may be before getting a starter (e.g., mom, oak, tv)
            validate_common(result, stem, min_party=0)
            save_output(stem, result)

            st = result["state_text"]
            raw = result["raw_state"]
            game = raw.get("game", {})

            # Should not be in battle
            check(f"{stem}: is_in_battle is False",
                  game.get("is_in_battle") is False)

            # Game state should be dialog (or overworld if detection fails)
            gs = game.get("game_state", "")
            check(f"{stem}: game_state is dialog or overworld",
                  gs in ("dialog", "overworld"),
                  f"got '{gs}'")

            # Dialogue detection
            dd = game.get("dialogue_detected", {})
            check(f"{stem}: dialogue_detected is dict",
                  isinstance(dd, dict))
            if dd.get("has_dialogue"):
                check(f"{stem}: dialogue detected (has_dialogue=True)", True)
            else:
                print(f"  [NOTE] {stem}: dialogue not detected by RAM signals "
                      f"(confidence={dd.get('confidence')})")

            # If dialog text is present, state_text should show it
            dialog_text = game.get("dialog_text")
            if dialog_text:
                check(f"{stem}: dialog_text present in raw_state", True)
                check(f"{stem}: state_text has DIALOGUE section",
                      "DIALOGUE" in st.upper() or "Text:" in st,
                      f"dialog_text={dialog_text[:80]}")

            # Non-battle state should still have player info and map
            check(f"{stem}: state_text has PLAYER INFO",
                  "PLAYER INFO" in st.upper() or "Player" in st)
            check(f"{stem}: state_text has map or location info",
                  "MAP" in st.upper() or "LOCATION" in st.upper())

            print(f"\n  state_text ({len(st)} chars):")
            print_state_text(st)
        finally:
            emu.stop()


# ---------------------------------------------------------------------------
# Test: Menu / special states
# ---------------------------------------------------------------------------

def test_menu_states():
    print(f"\n{SEP}")
    print("=== TEST: Menu / Special States ===")
    print(SEP)

    menu_files = [
        "pokered_choose_pokemon_1.state",
        "pokered_choose_pokemon_2.state",
        "pokered_menu.state",
        "pokered_name_pokemon.state",
        "pokered_oak_encounter.state",
        "pokered_warp.state",
    ]

    for fname in menu_files:
        state_path = STATES_DIR / fname
        if not state_path.exists():
            print(f"  [SKIP] {fname} not found")
            continue

        stem = state_path.stem
        print(f"\n  --- {fname} ---")
        emu = load_emu(state_path)
        try:
            result = run_pipeline(emu)
            # Relaxed validation: some menu/special states may have unusual conditions
            check(f"{stem}: success", result.get("success") is True)

            st = result.get("state_text", "")
            check(f"{stem}: state_text non-empty",
                  isinstance(st, str) and len(st) > 20,
                  f"len={len(st)}")

            raw = result.get("raw_state", {})
            for key in ("visual", "player", "game", "map"):
                check(f"{stem}: raw_state has '{key}'", key in raw)

            # Party validation (may have 0 pokemon in choose_pokemon/oak_encounter)
            party = raw.get("player", {}).get("party", [])
            check(f"{stem}: party is a list", isinstance(party, list),
                  f"type={type(party)}")
            if party:
                missing = PARTY_REQUIRED_KEYS - set(party[0].keys())
                check(f"{stem}: party[0] has Gen1 fields",
                      len(missing) == 0,
                      f"missing: {missing}")

            save_output(stem, result)

            print(f"\n  state_text ({len(st)} chars):")
            print_state_text(st)
        except Exception as e:
            check(f"{stem}: no crash", False, str(e))
            import traceback
            traceback.print_exc()
        finally:
            emu.stop()


# ---------------------------------------------------------------------------
# Test: Zip checkpoint states (mid-game progression)
# ---------------------------------------------------------------------------

def test_zip_states():
    print(f"\n{SEP}")
    print("=== TEST: Zip Checkpoint States (mid-game) ===")
    print(SEP)

    zip_files = sorted(STATES_DIR.glob("*.zip"))
    if not zip_files:
        print("  [SKIP] No .zip files found in test_states/")
        return

    for zip_path in zip_files:
        stem = zip_path.stem[:60]  # Truncate long names
        print(f"\n  --- {zip_path.name[:80]} ---")
        try:
            emu = load_emu_from_zip(zip_path)
        except Exception as e:
            check(f"{stem}: load from zip", False, str(e))
            continue

        try:
            result = run_pipeline(emu)
            validate_common(result, stem)
            save_output(stem, result)

            st = result["state_text"]
            raw = result["raw_state"]
            game = raw.get("game", {})
            player = raw.get("player", {})

            # Mid-game progression: should have multiple pokemon
            party = player.get("party", [])
            check(f"{stem}: party has > 1 pokemon (mid-game)",
                  len(party) > 1,
                  f"got {len(party)}")

            # Should have badges
            badges = game.get("badges", [])
            check(f"{stem}: has badges (mid-game)",
                  isinstance(badges, list) and len(badges) > 0,
                  f"got {badges}")

            # Should have money
            money = game.get("money")
            check(f"{stem}: has money > 0",
                  money is not None and money > 0,
                  f"got {money}")

            # Location should not be early-game
            loc = player.get("location", "")
            check(f"{stem}: location is not early-game start",
                  loc not in ("PalletTown", "OaksLab", "Unknown", ""),
                  f"got '{loc}'")

            # defeat_misty should have >= 2 badges
            if "defeat_misty" in zip_path.name.lower():
                check(f"{stem}: defeat_misty has >= 2 badges",
                      len(badges) >= 2,
                      f"got {len(badges)}: {badges}")

            # Items check
            items = game.get("items", [])
            check(f"{stem}: items is a list", isinstance(items, list))

            # Progress context
            progress = game.get("progress_context", {})
            check(f"{stem}: progress_context exists",
                  isinstance(progress, dict) and len(progress) > 0)

            # Pokedex
            caught = game.get("pokedex_caught", 0)
            seen = game.get("pokedex_seen", 0)
            check(f"{stem}: pokedex_caught > 0", caught > 0, f"got {caught}")
            check(f"{stem}: pokedex_seen >= caught", seen >= caught,
                  f"seen={seen}, caught={caught}")

            # State text content depends on game state
            if game.get("is_in_battle"):
                check(f"{stem}: battle state_text has BATTLE",
                      "BATTLE" in st.upper())
            else:
                check(f"{stem}: non-battle state_text has PLAYER INFO or MAP",
                      "PLAYER INFO" in st.upper() or "MAP" in st.upper())

            print(f"\n  state_text ({len(st)} chars):")
            print_state_text(st, max_lines=80)
        except Exception as e:
            check(f"{stem}: no crash", False, str(e))
            import traceback
            traceback.print_exc()
        finally:
            emu.stop()


# ---------------------------------------------------------------------------
# Test: Init zip (game start)
# ---------------------------------------------------------------------------

def test_init_zip():
    print(f"\n{SEP}")
    print("=== TEST: Init Zip (red_init.zip) ===")
    print(SEP)

    if not INIT_ZIP.exists():
        print(f"  [SKIP] {INIT_ZIP} not found")
        return

    try:
        emu = load_emu_from_zip(INIT_ZIP)
    except Exception as e:
        check("init_zip: load from zip", False, str(e))
        return

    try:
        result = run_pipeline(emu)
        label = "init_zip"
        check(f"{label}: success", result.get("success") is True)

        st = result.get("state_text", "")
        check(f"{label}: state_text non-empty",
              isinstance(st, str) and len(st) > 20,
              f"len={len(st)}")

        raw = result.get("raw_state", {})
        for key in ("visual", "player", "game", "map"):
            check(f"{label}: raw_state has '{key}'", key in raw)

        # Init state: may have 0 or 1 pokemon
        party = raw.get("player", {}).get("party", [])
        check(f"{label}: party is a list", isinstance(party, list))
        print(f"  [INFO] init_zip party size: {len(party)}")

        # Location
        loc = result.get("location", "")
        print(f"  [INFO] init_zip location: {loc}")

        save_output(label, result)

        print(f"\n  state_text ({len(st)} chars):")
        print_state_text(st)
    finally:
        emu.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not ROM_PATH.exists():
        print(f"ERROR: ROM not found at {ROM_PATH}")
        sys.exit(1)

    print(f"ROM:    {ROM_PATH}")
    print(f"States: {STATES_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    test_overworld_states()
    test_battle_states()
    test_dialog_states()
    test_menu_states()
    test_zip_states()
    test_init_zip()

    print(f"\n{SEP}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    if FAILED == 0:
        print("All tests passed!")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

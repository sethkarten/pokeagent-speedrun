#!/usr/bin/env python3
"""
Unit tests for utils/state_formatter.py compatibility with Pokemon Red state data.

Tests that _format_map_info and format_state_for_llm work correctly with Red's
state structure (no porymap data, visual_map from red_map_reader, GBC-specific fields).

Usage:
    python pokemon_red_env/test/test_red_state_formatter.py
"""

import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from utils.state_formatter import format_state_for_llm, _format_map_info

PASSED = 0
FAILED = 0

# Width of the separator / header lines
SEP = "-" * 70


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  {detail}")


def print_formatted_map(result, indent="    "):
    """Pretty-print the list returned by _format_map_info."""
    text = "\n".join(str(p) for p in result)
    if not text.strip():
        print(f"{indent}(empty)")
        return text
    for line in text.split("\n"):
        print(f"{indent}{line}")
    return text


def print_formatted_state(result, indent="    ", max_lines=80):
    """Pretty-print the string returned by format_state_for_llm."""
    if not result or not result.strip():
        print(f"{indent}(empty)")
        return
    lines = result.split("\n")
    for line in lines[:max_lines]:
        print(f"{indent}{line}")
    if len(lines) > max_lines:
        print(f"{indent}... ({len(lines) - max_lines} more lines)")


# ---------------------------------------------------------------
# Sample Red state data (matches real emulator output)
# ---------------------------------------------------------------

def make_red_map_info(visual_map=None, map_source=None):
    """Create a Red-style map dict (no porymap, tiles are 4-tuples)."""
    tiles = [
        [(0, "WALL", 1, 0)] * 10,
        [(0, "WALL", 1, 0)] * 4 + [(0, "WALKABLE", 0, 0)] * 2 + [(0, "WALL", 1, 0)] * 4,
        [(0, "WALKABLE", 0, 0)] * 5 + [(0, "NPC", 1, 0)] + [(0, "WALKABLE", 0, 0)] * 4,
        [(0, "WALKABLE", 0, 0)] * 10,
        [(0, "WALKABLE", 0, 0)] * 3 + [(0, "WALL", 1, 0)] * 2 + [(0, "WALKABLE", 0, 0)] * 5,
        [(0, "WALKABLE", 0, 0)] * 3 + [(0, "WALL", 1, 0)] * 2 + [(0, "WALKABLE", 0, 0)] * 5,
        [(0, "WALKABLE", 0, 0)] * 10,
        [(0, "WALKABLE", 0, 0)] * 10,
        [(0, "WALKABLE", 0, 0)] * 10,
        [(0, "WALKABLE", 0, 0)] * 2 + [(0, "WARP", 0, 0)] * 2 + [(0, "WALKABLE", 0, 0)] * 6,
        [(0, "WALKABLE", 0, 0)] * 10,
        [(0, "WALL", 1, 0)] * 10,
    ]
    # Convert to lists for JSON compatibility
    tiles = [[list(t) for t in row] for row in tiles]

    m = {
        "tiles": tiles,
        "tile_names": [["WALL" if t[1] == "WALL" else "WALKABLE" for t in row] for row in tiles],
        "metatile_behaviors": [[t[1] for t in row] for row in tiles],
        "metatile_info": [[{"type": t[1], "collision": t[2]} for t in row] for row in tiles],
        "traversability": [[True] * 9 for _ in range(9)],
        "object_events": [
            {"id": 1, "local_id": 1, "current_x": 5, "current_y": 2,
             "facing": "down", "graphics_id": 3, "sprite_name": "SPRITE_OAK",
             "distance": 3, "source": "sprite_slot_1_ram"},
        ],
    }
    if visual_map is not None:
        m["visual_map"] = visual_map
    if map_source is not None:
        m["map_source"] = map_source
    return m


SAMPLE_VISUAL_MAP = """\
##########
####..####
.....N....
......#N#.
..........
...##.....
...##.....
..........
..........
..WW......
..........
##########"""


def make_red_player_data(location="OaksLab", x=5, y=6):
    return {
        "position": {"x": x, "y": y},
        "location": location,
        "name": "RED",
        "party": [{
            "species_id": 153, "species_name": "Bulbasaur", "nickname": "BB",
            "level": 5, "current_hp": 19, "max_hp": 19, "status": "OK",
            "types": ["Grass", "Poison"],
            "attack": 9, "defense": 10, "speed": 10, "special": 11,
            "moves": ["TACKLE", "GROWL", "NONE", "NONE"],
            "move_pp": [35, 40, 0, 0],
        }],
    }


def make_red_full_state(visual_map=None, map_source=None, location="OaksLab",
                        game_state="overworld"):
    return {
        "visual": {"screenshot": None, "resolution": [160, 144]},
        "player": make_red_player_data(location=location),
        "game": {
            "money": 3000,
            "party": make_red_player_data(location)["party"],
            "game_state": game_state,
            "is_in_battle": game_state == "battle",
            "time": None,
            "badges": [],
            "items": [],
            "item_count": 0,
            "pokedex_caught": 1,
            "pokedex_seen": 2,
            "battle_info": None,
            "dialog_text": None,
            "progress_context": {
                "badges_obtained": 0, "badge_names": [],
                "party_size": 1, "has_pokedex": True, "is_champion": False,
                "visited_locations": ["visited_oakslab"],
                "flags": {}, "party_levels": [5], "party_species": ["Bulbasaur"],
            },
            "dialogue_detected": {
                "has_dialogue": False, "confidence": 0.0,
                "reason": "gen1_text_progress + vram_border_check",
            },
        },
        "map": make_red_map_info(visual_map=visual_map, map_source=map_source),
    }


# ---------------------------------------------------------------
# Tests for _format_map_info
# ---------------------------------------------------------------

def test_format_map_info_no_crash():
    """_format_map_info must not crash on Red state data (no porymap)."""
    print(f"\n{SEP}")
    print("--- _format_map_info: no crash with Red data ---")
    map_info = make_red_map_info()
    player_data = make_red_player_data()
    full_state = make_red_full_state()

    try:
        result = _format_map_info(map_info, player_data=player_data,
                                  full_state_data=full_state)
        check("no exception raised", True)
        check("returns list", isinstance(result, list))
        print("\n  Formatted map output:")
        print_formatted_map(result)
    except Exception as e:
        check("no exception raised", False, str(e))
        return


def test_format_map_info_has_location():
    """Output includes location name from player_data."""
    print(f"\n{SEP}")
    print("--- _format_map_info: location extracted ---")
    map_info = make_red_map_info()
    player_data = make_red_player_data(location="PalletTown")
    full_state = make_red_full_state(location="PalletTown")

    result = _format_map_info(map_info, player_data=player_data,
                              full_state_data=full_state)
    text = "\n".join(str(p) for p in result)
    check("location in output", "PalletTown" in text, f"output: {text[:200]}")

    print("\n  Formatted map output:")
    print_formatted_map(result)


def test_format_map_info_has_player_position():
    """Output includes player coordinates."""
    print(f"\n{SEP}")
    print("--- _format_map_info: player position ---")
    map_info = make_red_map_info()
    player_data = make_red_player_data(x=5, y=6)
    full_state = make_red_full_state()

    result = _format_map_info(map_info, player_data=player_data,
                              full_state_data=full_state)
    text = "\n".join(str(p) for p in result)
    check("player position in output", "5" in text and "6" in text,
          f"output: {text[:200]}")


def test_format_map_info_visual_map_fallback():
    """When visual_map is present and porymap is absent, the fallback renders it."""
    print(f"\n{SEP}")
    print("--- _format_map_info: visual_map fallback ---")
    map_info = make_red_map_info(visual_map=SAMPLE_VISUAL_MAP,
                                 map_source="red_map_reader")
    player_data = make_red_player_data()
    full_state = make_red_full_state(visual_map=SAMPLE_VISUAL_MAP,
                                     map_source="red_map_reader")

    result = _format_map_info(map_info, player_data=player_data,
                              full_state_data=full_state)
    text = "\n".join(str(p) for p in result)

    check("visual_map content present", "##########" in text,
          f"output: {text[:300]}")
    check("MAP header present", "MAP" in text.upper(), f"output: {text[:300]}")
    check("source attribution present", "red_map_reader" in text,
          f"output: {text[:300]}")

    print("\n  Formatted map output (with visual_map fallback):")
    print_formatted_map(result)


def test_format_map_info_no_visual_map():
    """When neither porymap nor visual_map is present, output has location but no map."""
    print(f"\n{SEP}")
    print("--- _format_map_info: no visual_map, no porymap ---")
    map_info = make_red_map_info()  # no visual_map
    player_data = make_red_player_data()
    full_state = make_red_full_state()

    result = _format_map_info(map_info, player_data=player_data,
                              full_state_data=full_state)
    text = "\n".join(str(p) for p in result)

    check("location still present", "OaksLab" in text)
    # Should NOT have a map section
    check("no MAP section without data", "--- MAP" not in text,
          f"output: {text[:300]}")

    print("\n  Formatted map output (no visual_map):")
    print_formatted_map(result)


def test_format_map_info_title_sequence():
    """TITLE_SEQUENCE location returns early with message."""
    print(f"\n{SEP}")
    print("--- _format_map_info: TITLE_SEQUENCE ---")
    map_info = make_red_map_info()
    player_data = make_red_player_data(location="TITLE_SEQUENCE")

    result = _format_map_info(map_info, player_data=player_data)
    text = "\n".join(str(p) for p in result)

    check("title sequence message", "title sequence" in text.lower(),
          f"output: {text[:200]}")
    check("no map content", "MAP (viewport)" not in text)

    print("\n  Formatted map output (title sequence):")
    print_formatted_map(result)


def test_format_map_info_empty_map():
    """Empty/None map_info returns empty list."""
    print(f"\n{SEP}")
    print("--- _format_map_info: empty map ---")
    result = _format_map_info(None)
    check("returns empty list for None", result == [])
    result = _format_map_info({})
    check("returns empty list for {}", result == [])


# ---------------------------------------------------------------
# Tests for format_state_for_llm (end-to-end)
# ---------------------------------------------------------------

def test_format_state_for_llm_no_crash():
    """format_state_for_llm must not crash on full Red state."""
    print(f"\n{SEP}")
    print("--- format_state_for_llm: no crash ---")
    state = make_red_full_state(visual_map=SAMPLE_VISUAL_MAP,
                                map_source="red_map_reader")
    try:
        result = format_state_for_llm(state)
        check("no exception raised", True)
        check("returns string", isinstance(result, str))
        check("non-empty output", len(result) > 50, f"len={len(result)}")

        print(f"\n  Full formatted state for LLM ({len(result)} chars):")
        print_formatted_state(result)
    except Exception as e:
        check("no exception raised", False, str(e))


def test_format_state_for_llm_contains_key_info():
    """Output text includes player name, location, party info."""
    print(f"\n{SEP}")
    print("--- format_state_for_llm: key info present ---")
    state = make_red_full_state(visual_map=SAMPLE_VISUAL_MAP,
                                map_source="red_map_reader")
    result = format_state_for_llm(state)

    check("contains player name 'RED'", "RED" in result)
    check("contains location 'OaksLab'", "OaksLab" in result)
    check("contains 'Bulbasaur'", "Bulbasaur" in result)
    check("contains money '3000'", "3000" in result)


def test_format_state_for_llm_contains_visual_map():
    """When visual_map is present, it shows up in the formatted output."""
    print(f"\n{SEP}")
    print("--- format_state_for_llm: visual_map in output ---")
    state = make_red_full_state(visual_map=SAMPLE_VISUAL_MAP,
                                map_source="red_map_reader")
    result = format_state_for_llm(state)

    check("visual_map content in output", "##########" in result,
          f"output length: {len(result)}")


def test_format_state_for_llm_battle_state():
    """Works with battle state (has battle_info)."""
    print(f"\n{SEP}")
    print("--- format_state_for_llm: battle state ---")
    state = make_red_full_state(game_state="battle")
    state["game"]["is_in_battle"] = True
    state["game"]["battle_info"] = {
        "in_battle": True, "battle_type": "trainer",
        "opponent": {
            "species_id": 176, "species": "Charmander", "level": 5,
            "current_hp": 19, "max_hp": 19, "hp_percentage": 100.0,
            "status": "OK",
        },
    }
    try:
        result = format_state_for_llm(state)
        check("no exception in battle state", True)
        check("returns string", isinstance(result, str))
        # Battle info should appear somewhere
        check("battle info present", "battle" in result.lower() or "Charmander" in result,
              f"output: {result[:200]}")

        print(f"\n  Battle state for LLM ({len(result)} chars):")
        print_formatted_state(result)
    except Exception as e:
        check("no exception in battle state", False, str(e))


def test_format_state_for_llm_dialog_state():
    """Works when dialog_text is present."""
    print(f"\n{SEP}")
    print("--- format_state_for_llm: dialog state ---")
    state = make_red_full_state()
    state["game"]["dialog_text"] = "OAK: Now is not the time to use that!"
    state["game"]["dialogue_detected"] = {
        "has_dialogue": True, "confidence": 0.9,
        "reason": "gen1_text_progress + vram_border_check",
    }
    try:
        result = format_state_for_llm(state)
        check("no exception with dialog", True)
        check("returns string", isinstance(result, str))

        print(f"\n  Dialog state for LLM ({len(result)} chars):")
        print_formatted_state(result)
    except Exception as e:
        check("no exception with dialog", False, str(e))


def test_format_state_for_llm_gen1_party_fields():
    """Gen 1 party fields (attack, defense, speed, special) don't cause errors."""
    print(f"\n{SEP}")
    print("--- format_state_for_llm: Gen 1 party fields ---")
    state = make_red_full_state()
    # Verify the party has Gen 1 stats
    party = state["player"]["party"][0]
    check("party has 'attack'", "attack" in party)
    check("party has 'special'", "special" in party)
    check("party has no 'sp_attack'", "sp_attack" not in party)

    try:
        result = format_state_for_llm(state)
        check("no exception with Gen 1 stats", True)
    except Exception as e:
        check("no exception with Gen 1 stats", False, str(e))


def test_format_state_for_llm_no_map_at_all():
    """Works when map section has only tiles and no visual_map."""
    print(f"\n{SEP}")
    print("--- format_state_for_llm: no visual_map ---")
    state = make_red_full_state()  # no visual_map
    try:
        result = format_state_for_llm(state)
        check("no exception without visual_map", True)
        check("still has location", "OaksLab" in result)

        print(f"\n  State without visual_map ({len(result)} chars):")
        print_formatted_state(result)
    except Exception as e:
        check("no exception without visual_map", False, str(e))


# ---------------------------------------------------------------
# Tests with real saved states (if available)
# ---------------------------------------------------------------

def test_with_real_states():
    """Load real Red state JSONs and run format_state_for_llm on them."""
    print(f"\n{SEP}")
    print("--- format_state_for_llm: real saved states ---")
    output_dir = REPO / "pokemon_red_env" / "test" / "output"
    if not output_dir.exists():
        print("  (skipping -- no output directory)")
        return

    tested = 0
    for state_dir in sorted(output_dir.iterdir()):
        if not state_dir.is_dir():
            continue
        json_files = list(state_dir.glob("*_battle_*.json")) + \
                     list(state_dir.glob("*_overworld_*.json")) + \
                     list(state_dir.glob("*_choose_*.json")) + \
                     list(state_dir.glob("*_warp*.json"))
        # Also catch any .json that isn't milestones/coll_map
        if not json_files:
            json_files = [f for f in state_dir.glob("*.json")
                          if "milestones" not in f.name and "coll_map" not in f.name]
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                # Skip if not a valid state (must have player + game + map)
                if not all(k in data for k in ("player", "game", "map")):
                    continue
                # Add a fake visual_map to simulate server behavior
                if "visual_map" not in data["map"]:
                    data["map"]["visual_map"] = "P . . .\n. . . .\n. . # #"
                    data["map"]["map_source"] = "test_fallback"

                loc = data["player"].get("location", "?")

                # --- _format_map_info ---
                map_result = _format_map_info(
                    data["map"],
                    player_data=data["player"],
                    full_state_data=data,
                )
                check(f"{jf.stem} ({loc}): _format_map_info OK", True)

                print(f"\n  _format_map_info for {jf.stem} ({loc}):")
                print_formatted_map(map_result)

                # --- format_state_for_llm ---
                state_result = format_state_for_llm(data)
                check(f"{jf.stem} ({loc}): format_state_for_llm OK", True)

                print(f"\n  format_state_for_llm for {jf.stem} ({loc}, {len(state_result)} chars):")
                print_formatted_state(state_result)
                print()

                tested += 1
            except Exception as e:
                check(f"{jf.stem}: no crash", False, str(e))
                tested += 1

    if tested == 0:
        print("  (no real state JSONs found)")
    else:
        print(f"  Tested {tested} real state files")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    print("Testing state_formatter.py compatibility with Pokemon Red...\n")

    # _format_map_info tests
    test_format_map_info_no_crash()
    test_format_map_info_has_location()
    test_format_map_info_has_player_position()
    test_format_map_info_visual_map_fallback()
    test_format_map_info_no_visual_map()
    test_format_map_info_title_sequence()
    test_format_map_info_empty_map()

    # format_state_for_llm tests
    test_format_state_for_llm_no_crash()
    test_format_state_for_llm_contains_key_info()
    test_format_state_for_llm_contains_visual_map()
    test_format_state_for_llm_battle_state()
    test_format_state_for_llm_dialog_state()
    test_format_state_for_llm_gen1_party_fields()
    test_format_state_for_llm_no_map_at_all()

    # Real state tests
    test_with_real_states()

    print(f"\n{'='*60}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    if FAILED == 0:
        print("All tests passed!")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()

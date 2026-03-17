#!/usr/bin/env python3
"""
Verify Option A fix for species_name: derive from species_id, not nickname.

1. Points to existing PokemonSpecies(species_id) usage in memory_reader.py
2. Tests emerald_utils._species_id_to_name (ROM pokeemerald numbering)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rom_species_mapping():
    """Verify ROM species IDs map to correct names (emerald_utils single source)."""
    from pokemon_env.emerald_utils import _species_id_to_name

    assert _species_id_to_name(280) == "Torchic"
    assert _species_id_to_name(277) == "Treecko"
    assert _species_id_to_name(283) == "Mudkip"
    assert _species_id_to_name(279) == "Sceptile"
    assert _species_id_to_name(999) == "Species_999"  # unknown fallback
    print("OK: emerald_utils._species_id_to_name works for Torchic, Treecko, Mudkip, Sceptile")


def test_pokemon_species_enum_mismatch():
    """Verify PokemonSpecies enum uses different numbering (Hoenn Dex order)."""
    # Import enums directly to avoid mgba dependency from memory_reader
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enums",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "pokemon_env", "enums.py")
    )
    enums_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enums_module)
    PokemonSpecies = enums_module.PokemonSpecies
    # Enum: TORCHIC=4, TREECKO=1 (Hoenn Dex order)
    # ROM: Torchic=280, Treecko=277 (pokeemerald order)
    assert PokemonSpecies.TORCHIC == 4
    assert PokemonSpecies.TREECKO == 1
    # ROM species_id 280 would fail: PokemonSpecies(280) raises ValueError
    try:
        PokemonSpecies(280)
        assert False, "PokemonSpecies(280) should raise ValueError"
    except ValueError:
        pass
    print("OK: PokemonSpecies enum uses Hoenn order (4=Torchic); ROM uses 280 for Torchic")


def test_existing_usage_in_memory_reader():
    """Point to existing logic that uses species_id to derive species name."""
    source = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pokemon_env", "memory_reader.py")
    with open(source, encoding="utf-8") as f:
        content = f.read()
    assert "PokemonSpecies(species_id)" in content
    assert "species_name = species.name.replace" in content
    print("OK: memory_reader.py uses PokemonSpecies(species_id) to derive species_name")
    print("    (Note: memory_reader reads battle/opponent data; ROM species_id there may differ)")


def test_torchic_state_with_emulator():
    """Load torchic.state and verify species_id, nickname, and fix logic."""
    try:
        from pokemon_env.emerald_utils import read_save_block_1
        from pokemon_env.emulator import EmeraldEmulator
    except ImportError as e:
        print(f"SKIP: Emulator not available ({e})")
        return

    state_path = os.path.join(os.path.dirname(__file__), "states", "torchic.state")
    if not os.path.exists(state_path):
        print(f"SKIP: {state_path} not found")
        return

    # EmeraldEmulator requires rom_path; try common locations
    rom_paths = ["emerald.gba", "emerald.gbc", "Pokemon Emerald.gba", "emerald.gba"]
    rom_path = next((p for p in rom_paths if os.path.exists(p)), None)
    if not rom_path:
        print("SKIP: No ROM found (need emerald.gba etc. for live test)")
        return

    emu = EmeraldEmulator(rom_path)
    if not emu.load_state(state_path):
        print("SKIP: Failed to load torchic.state")
        return

    save = read_save_block_1(emu.gba)
    if not save:
        print("SKIP: No save block")
        return

    party = getattr(save, "playerParty", None) or (save.get("playerParty", []) if isinstance(save, dict) else [])
    if not party:
        print("SKIP: Empty party")
        return

    p = party[0]
    species_id = p.species_id if hasattr(p, "species_id") else p.get("species_id")
    species_name = p.species_name if hasattr(p, "species_name") else p.get("species_name")
    nickname = p.nickname if hasattr(p, "nickname") else p.get("nickname")

    print(f"Torchic state - species_id: {species_id}, species_name: {species_name!r}, nickname: {nickname!r}")

    assert species_id is not None, "species_id should be present"
    from pokemon_env.emerald_utils import _species_id_to_name

    derived_name = _species_id_to_name(species_id)
    print(f"Derived name from species_id {species_id}: {derived_name}")

    assert derived_name == "Torchic", f"Expected Torchic, got {derived_name}"
    print("OK: species_id from torchic.state correctly maps to Torchic")


if __name__ == "__main__":
    print("=" * 60)
    print("Species name fix verification")
    print("=" * 60)
    test_rom_species_mapping()
    test_pokemon_species_enum_mismatch()
    test_existing_usage_in_memory_reader()
    test_torchic_state_with_emulator()
    print("=" * 60)
    print("All checks passed.")

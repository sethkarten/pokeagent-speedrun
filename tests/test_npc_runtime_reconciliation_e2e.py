#!/usr/bin/env python3
"""E2E validation for runtime NPC reconciliation on auto-evolve init state."""

from pathlib import Path
import zipfile

import pytest

from pokemon_env.emulator import EmeraldEmulator
from utils.mapping.porymap_state import _format_porymap_info


def test_autoevolve_init_birch_lab_runtime_reconciliation(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    rom_path = project_root / "Emerald-GBAdvance" / "rom.gba"
    init_zip = project_root / "Emerald-GBAdvance" / "auto-evolve_init.zip"

    if not rom_path.exists():
        pytest.skip("ROM not available for e2e test")
    if not init_zip.exists():
        pytest.skip("auto-evolve init zip not available for e2e test")

    with zipfile.ZipFile(init_zip, "r") as archive:
        checkpoint_members = [name for name in archive.namelist() if name.endswith("checkpoint.state")]
        assert checkpoint_members, "Expected checkpoint.state in auto-evolve_init.zip"
        checkpoint_name = checkpoint_members[0]
        extracted_state = archive.extract(checkpoint_name, path=tmp_path)

    emu = EmeraldEmulator(str(rom_path), headless=True, sound=False)
    emu.initialize()
    try:
        emu.load_state(extracted_state)
        state = emu.get_comprehensive_state()
        runtime_npcs = emu.memory_reader.read_object_events()

        assert runtime_npcs, "Expected runtime NPCs in Birch's Lab auto-evolve init state"

        location = state.get("player", {}).get("location")
        position = state.get("player", {}).get("position", {})
        player_coords = (position.get("x", 0), position.get("y", 0))

        result = _format_porymap_info(
            location_name=location,
            player_coords=player_coords,
            runtime_object_events=runtime_npcs,
        )
        assert result.json_map is not None

        reconciled_objects = result.json_map.get("objects", [])
        assert reconciled_objects, "Expected reconciled objects in Birch's Lab"

        graphics_ids = {obj.get("graphics_id") for obj in reconciled_objects}
        # Auto-evolve init should at least keep live lab NPCs and remove starter balls.
        assert "OBJ_EVENT_GFX_SCIENTIST_1" in graphics_ids
        assert "OBJ_EVENT_GFX_ITEM_BALL" not in graphics_ids
    finally:
        emu.stop()

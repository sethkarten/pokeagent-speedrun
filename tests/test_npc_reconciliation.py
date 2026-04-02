#!/usr/bin/env python3
"""Unit and contract tests for runtime NPC reconciliation."""

from utils.mapping.pathfinding import Pathfinder
from utils.mapping.porymap_state import PorymapResult, _format_porymap_info


OCEANIC_LOCATION = "SLATEPORT CITY OCEANIC MUSEUM 2F"


def test_porymap_result_contract_success():
    """_format_porymap_info always returns a PorymapResult on success."""
    result = _format_porymap_info(OCEANIC_LOCATION, player_coords=(13, 6))
    assert isinstance(result, PorymapResult)
    assert isinstance(result.context_parts, list)
    assert result.json_map is not None
    assert "objects" in result.json_map


def test_porymap_result_contract_failure():
    """_format_porymap_info always returns a PorymapResult on failure."""
    result = _format_porymap_info("UNKNOWN_LOCATION_FOR_TEST")
    assert isinstance(result, PorymapResult)
    assert isinstance(result.context_parts, list)
    assert result.json_map is None


def test_reconciliation_local_id_priority():
    """local_id matching should win even if coordinates are far away."""
    runtime = [
        {
            "local_id": "LOCALID_OCEANIC_MUSEUM_2F_CAPT_STERN",
            "graphics_id": "OBJ_EVENT_GFX_SCIENTIST_1",
            "current_x": 99,
            "current_y": 99,
        }
    ]
    result = _format_porymap_info(
        OCEANIC_LOCATION,
        player_coords=(13, 6),
        runtime_object_events=runtime,
    )
    objects = result.json_map["objects"]
    assert len(objects) == 1
    assert objects[0]["local_id"] == "LOCALID_OCEANIC_MUSEUM_2F_CAPT_STERN"


def test_reconciliation_coordinate_proximity_fallback():
    """Coordinate proximity + graphics_id should match nearby runtime NPCs."""
    runtime = [{"local_id": "", "graphics_id": "OBJ_EVENT_GFX_SCIENTIST_1", "current_x": 12, "current_y": 6}]
    result = _format_porymap_info(
        OCEANIC_LOCATION,
        player_coords=(13, 6),
        runtime_object_events=runtime,
    )
    objects = result.json_map["objects"]
    assert len(objects) == 1
    assert objects[0]["graphics_id"] == "OBJ_EVENT_GFX_SCIENTIST_1"


def test_reconciliation_static_reference_when_no_runtime():
    """No runtime NPC list should keep static porymap objects."""
    result = _format_porymap_info(OCEANIC_LOCATION, player_coords=(13, 6), runtime_object_events=[])
    assert result.json_map is not None
    assert len(result.json_map["objects"]) >= 7


def test_pathfinding_blocks_runtime_objects_not_static_when_available():
    """Pathfinding should prioritize runtime object_events for NPC blocking."""
    game_state = {
        "map": {
            "porymap": {
                "grid": [[".", ".", "."], [".", ".", "."], [".", ".", "."]],
                "dimensions": {"width": 3, "height": 3},
                "objects": [{"x": 0, "y": 0, "movement_type": "MOVEMENT_TYPE_FACE_DOWN"}],
            },
            "object_events": [{"current_x": 1, "current_y": 1, "is_blocking": True}],
        }
    }
    pf = Pathfinder()
    map_data = pf._extract_map_data(game_state)
    blocked = pf._get_blocked_positions(
        game_state=game_state,
        map_data=map_data,
        start_pos=(2, 2),
        goal_pos=None,
        consider_npcs=True,
    )

    assert (1, 1) in blocked
    assert (0, 0) not in blocked

#!/usr/bin/env python3
"""
Focused tests for supported direct objective modes.
"""

import json
import os
import sys
import tempfile

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.objectives import DirectObjective, DirectObjectiveManager, get_first_objective_info


class TestDirectObjective:
    def test_direct_objective_creation(self):
        obj = DirectObjective(
            id="test_01",
            description="Test objective",
            action_type="navigate",
            target_location="Test Location",
            target_coords=(5, 10),
            navigation_hint="Go north",
            completion_condition="location_reached",
            priority=1,
            completed=False,
        )

        assert obj.id == "test_01"
        assert obj.description == "Test objective"
        assert obj.action_type == "navigate"
        assert obj.target_location == "Test Location"
        assert obj.target_coords == (5, 10)
        assert obj.navigation_hint == "Go north"
        assert obj.completion_condition == "location_reached"
        assert obj.priority == 1
        assert obj.completed is False

    def test_direct_objective_defaults(self):
        obj = DirectObjective(
            id="test_02",
            description="Minimal objective",
            action_type="interact",
        )

        assert obj.target_location is None
        assert obj.target_coords is None
        assert obj.navigation_hint is None
        assert obj.completion_condition is None
        assert obj.priority == 1
        assert obj.completed is False


class TestDirectObjectiveManager:
    @pytest.fixture
    def manager(self):
        return DirectObjectiveManager()

    def test_manager_initialization(self, manager):
        assert manager.current_sequence == []
        assert manager.current_index == 0
        assert manager.sequence_name == ""
        assert manager.mode == "legacy"

    def test_load_categorized_full_game_sequence(self, manager):
        manager.load_categorized_full_game_sequence()

        assert manager.mode == "categorized"
        assert manager.sequence_name == "categorized_full_game"
        assert len(manager.story_sequence) > 0
        assert len(manager.battling_sequence) > 0
        assert manager.story_index == 0
        assert manager.battling_index == 0
        assert manager.dynamics_sequence == []

    def test_load_categorized_full_game_with_start_indexes(self, manager):
        manager.load_categorized_full_game_sequence(start_story_index=3, start_battling_index=2)

        assert manager.story_index == 3
        assert manager.battling_index == 2
        assert all(obj.completed for obj in manager.story_sequence[:3])
        assert all(obj.completed for obj in manager.battling_sequence[:2])
        assert manager.story_sequence[3].completed is False
        assert manager.battling_sequence[2].completed is False

    def test_load_autonomous_objective_creation_sequence(self, manager):
        manager.load_autonomous_objective_creation_sequence()

        assert manager.mode == "categorized"
        assert manager.sequence_name == "autonomous_objective_creation"
        assert len(manager.story_sequence) == 1
        assert manager.story_sequence[0].id == "autonomous_01_plan_objectives"
        assert manager.battling_sequence == []
        assert manager.dynamics_sequence == []

    def test_load_autonomous_sequence_mentions_replan_when_simplest_scaffold(self, manager, monkeypatch):
        monkeypatch.setenv("EXCLUDE_BUILTIN_SUBAGENTS", "1")
        manager.load_autonomous_objective_creation_sequence()
        desc = manager.story_sequence[0].description
        hint = manager.story_sequence[0].navigation_hint or ""
        assert "replan_objectives" in desc or "replan_objectives" in hint
        assert "get_progress_summary" in hint
        assert "subagent_plan_objectives" not in desc

    def test_get_first_objective_info_for_supported_sequences(self):
        autonomous = get_first_objective_info("autonomous_objective_creation")
        categorized = get_first_objective_info("categorized_full_game")
        unsupported = get_first_objective_info("tutorial_to_rival")

        assert autonomous == ("autonomous", "autonomous_objective_creation")
        assert categorized[0] is not None
        assert categorized[1] is not None
        assert unsupported == (None, None)

    def test_get_categorized_objective_guidance(self, manager):
        manager.load_categorized_full_game_sequence()

        guidance = manager.get_categorized_objective_guidance({})

        assert guidance is not None
        assert guidance["story"]["id"] == manager.story_sequence[0].id
        assert "recommended_battling_objectives" in guidance
        assert "dynamics" in guidance

    def test_add_dynamic_objectives_legacy_mode(self, manager):
        manager.add_dynamic_objectives(
            [
                {
                    "id": "dynamic_01",
                    "description": "Navigate somewhere",
                    "action_type": "navigate",
                    "target_location": "Test Area",
                },
                {
                    "id": "dynamic_02",
                    "description": "Interact with NPC",
                    "action_type": "interact",
                },
            ]
        )

        assert len(manager.current_sequence) == 2
        assert manager.current_index == 0
        assert manager.get_current_objective().id == "dynamic_01"

    def test_add_objectives_to_category_updates_dynamics(self, manager):
        manager.load_categorized_full_game_sequence()
        manager.add_objectives_to_category(
            "dynamics",
            [
                {
                    "id": "dynamic_01",
                    "description": "Temporary cleanup objective",
                    "action_type": "navigate",
                    "target_location": "Somewhere",
                }
            ],
        )

        current = manager.get_current_objectives_by_category()
        assert len(manager.dynamics_sequence) == 1
        assert current["dynamics"].id == "dynamic_01"

    def test_reset_sequence(self, manager):
        manager.load_categorized_full_game_sequence()
        manager.reset_sequence()

        assert manager.current_sequence == []
        assert manager.current_index == 0
        assert manager.sequence_name == ""
        assert manager.story_sequence == []
        assert manager.battling_sequence == []
        assert manager.dynamics_sequence == []

    def test_save_completed_objectives_delegates_to_auto_save(self, manager):
        """save_completed_objectives is deprecated and should delegate to auto_save."""
        import warnings
        manager.add_dynamic_objectives(
            [
                {
                    "id": "dynamic_01",
                    "description": "Navigate somewhere",
                    "action_type": "navigate",
                }
            ]
        )
        manager._mark_objective_completed(manager.current_sequence[0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.save_completed_objectives()
            assert any(issubclass(x.category, DeprecationWarning) for x in w)


class TestDirectObjectiveSerde:
    """Tests for DirectObjective.to_dict / from_dict round-trip."""

    def test_round_trip_full(self):
        from datetime import datetime
        obj = DirectObjective(
            id="story_007",
            description="Enter Petalburg City",
            action_type="navigate",
            category="story",
            target_location="Petalburg City",
            target_coords=(12, 34),
            navigation_hint="Go west on Route 102",
            completion_condition="location_reached",
            priority=1,
            completed=True,
            completed_at=datetime(2026, 1, 15, 10, 30, 0),
            optional=False,
            recommended_battling_objectives=["battling_01", "battling_02"],
            prerequisite_story_objective="story_006",
        )
        d = obj.to_dict()
        restored = DirectObjective.from_dict(d)

        assert restored.id == obj.id
        assert restored.description == obj.description
        assert restored.action_type == obj.action_type
        assert restored.category == obj.category
        assert restored.target_location == obj.target_location
        assert restored.target_coords == obj.target_coords
        assert restored.navigation_hint == obj.navigation_hint
        assert restored.completion_condition == obj.completion_condition
        assert restored.priority == obj.priority
        assert restored.completed == obj.completed
        assert restored.completed_at == obj.completed_at
        assert restored.optional == obj.optional
        assert restored.recommended_battling_objectives == obj.recommended_battling_objectives
        assert restored.prerequisite_story_objective == obj.prerequisite_story_objective

    def test_round_trip_minimal(self):
        obj = DirectObjective(id="dyn_01", description="Quick task", action_type="interact")
        d = obj.to_dict()
        restored = DirectObjective.from_dict(d)

        assert restored.id == "dyn_01"
        assert restored.completed is False
        assert restored.completed_at is None
        assert restored.target_coords is None
        assert restored.recommended_battling_objectives == []

    def test_to_dict_coords_are_list(self):
        obj = DirectObjective(
            id="t", description="t", action_type="move", target_coords=(5, 10)
        )
        d = obj.to_dict()
        assert isinstance(d["target_coords"], list)

    def test_from_dict_coords_become_tuple(self):
        d = {"id": "t", "description": "t", "action_type": "move", "target_coords": [5, 10]}
        obj = DirectObjective.from_dict(d)
        assert isinstance(obj.target_coords, tuple)
        assert obj.target_coords == (5, 10)


class TestDirectObjectiveManagerSerde:
    """Tests for serialize_full_state / restore_from_state round-trip."""

    @pytest.fixture
    def populated_manager(self):
        mgr = DirectObjectiveManager()
        mgr.load_categorized_full_game_sequence(start_story_index=3, start_battling_index=2)
        mgr.add_objectives_to_category(
            "dynamics",
            [{"id": "dyn_01", "description": "Temp task", "action_type": "navigate"}],
        )
        return mgr

    def test_serialize_restore_round_trip(self, populated_manager):
        state = populated_manager.serialize_full_state()
        restored = DirectObjectiveManager()
        restored.restore_from_state(state)

        assert restored.mode == populated_manager.mode
        assert restored.sequence_name == populated_manager.sequence_name
        assert restored.story_index == populated_manager.story_index
        assert restored.battling_index == populated_manager.battling_index
        assert restored.dynamics_index == populated_manager.dynamics_index
        assert len(restored.story_sequence) == len(populated_manager.story_sequence)
        assert len(restored.battling_sequence) == len(populated_manager.battling_sequence)
        assert len(restored.dynamics_sequence) == len(populated_manager.dynamics_sequence)

    def test_serialize_restore_preserves_completion(self, populated_manager):
        state = populated_manager.serialize_full_state()
        restored = DirectObjectiveManager()
        restored.restore_from_state(state)

        for orig, rest in zip(populated_manager.story_sequence, restored.story_sequence):
            assert orig.completed == rest.completed

    def test_save_load_file_round_trip(self, populated_manager):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "objectives.json")
            populated_manager.save_to_file(path)

            assert os.path.exists(path)

            loaded = DirectObjectiveManager.load_from_file(path)
            assert loaded.mode == populated_manager.mode
            assert loaded.story_index == populated_manager.story_index
            assert loaded.battling_index == populated_manager.battling_index
            assert len(loaded.dynamics_sequence) == len(populated_manager.dynamics_sequence)

    def test_current_objective_matches_after_restore(self, populated_manager):
        orig_story = populated_manager._get_current_objective_for_category("story")
        state = populated_manager.serialize_full_state()

        restored = DirectObjectiveManager()
        restored.restore_from_state(state)
        rest_story = restored._get_current_objective_for_category("story")

        assert rest_story is not None
        assert rest_story.id == orig_story.id
        assert rest_story.description == orig_story.description

    def test_json_schema_version(self, populated_manager):
        state = populated_manager.serialize_full_state()
        assert state["version"] == 1
        assert "saved_at" in state
        assert "story" in state and "battling" in state and "dynamics" in state


class TestDirectObjectiveIntegration:
    @pytest.fixture
    def manager(self):
        return DirectObjectiveManager()

    def test_categorized_progression_workflow(self, manager):
        manager.load_categorized_full_game_sequence()

        current = manager.get_current_objectives_by_category()
        first_story = current["story"]
        assert first_story is not None

        manager._mark_objective_completed(first_story)
        manager.story_index += 1

        next_current = manager.get_current_objectives_by_category()
        assert next_current["story"] is not None
        assert next_current["story"].id != first_story.id

    def test_autonomous_sequence_can_seed_new_objectives(self, manager):
        manager.load_autonomous_objective_creation_sequence()
        manager.add_objectives_to_category(
            "story",
            [
                {
                    "id": "story_new_01",
                    "description": "Move to Route 102",
                    "action_type": "navigate",
                    "target_location": "Route 102",
                },
                {
                    "id": "story_new_02",
                    "description": "Enter Petalburg City",
                    "action_type": "navigate",
                    "target_location": "Petalburg City",
                },
            ],
        )

        assert len(manager.story_sequence) >= 3
        assert manager.story_sequence[-2].id == "story_new_01"
        assert manager.story_sequence[-1].id == "story_new_02"

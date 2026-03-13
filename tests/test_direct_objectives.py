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
        assert manager.story_sequence[0].id == "autonomous_01_create_next_story_objectives"
        assert manager.battling_sequence == []
        assert manager.dynamics_sequence == []

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

    def test_save_completed_objectives(self, manager):
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

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = manager.save_completed_objectives(run_dir=tmpdir)

            assert os.path.exists(filename)

            with open(filename, "r") as f:
                data = json.load(f)

            assert data["sequences"][0]["total_objectives_completed"] == 1


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

#!/usr/bin/env python3
"""
Tests for DirectObjectives Module
==================================

Tests the DirectObjectiveManager and DirectObjective classes with:
- Objective sequence loading
- Objective guidance retrieval
- Objective completion tracking
- Dynamic objective creation
- Sequence status tracking
- Objective context generation
"""

import pytest
import json
import sys
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.direct_objectives import DirectObjective, DirectObjectiveManager


class TestDirectObjective:
    """Test suite for DirectObjective dataclass"""
    
    def test_direct_objective_creation(self):
        """Test creating a DirectObjective with all fields"""
        obj = DirectObjective(
            id="test_01",
            description="Test objective",
            action_type="navigate",
            target_location="Test Location",
            target_coords=(5, 10),
            navigation_hint="Go north",
            completion_condition="location_reached",
            priority=1,
            completed=False
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
        """Test DirectObjective with minimal required fields"""
        obj = DirectObjective(
            id="test_02",
            description="Minimal objective",
            action_type="interact"
        )
        
        assert obj.id == "test_02"
        assert obj.description == "Minimal objective"
        assert obj.action_type == "interact"
        assert obj.target_location is None
        assert obj.target_coords is None
        assert obj.navigation_hint is None
        assert obj.completion_condition is None
        assert obj.priority == 1
        assert obj.completed is False


class TestDirectObjectiveManager:
    """Test suite for DirectObjectiveManager"""
    
    @pytest.fixture
    def manager(self):
        """Create a DirectObjectiveManager instance for testing"""
        return DirectObjectiveManager()
    
    def test_manager_initialization(self, manager):
        """Test that manager initializes with empty state"""
        assert manager.current_sequence == []
        assert manager.current_index == 0
        assert manager.sequence_name == ""
    
    def test_load_birch_to_rival_sequence(self, manager):
        """Test loading the birch_to_rival sequence"""
        manager.load_birch_to_rival_sequence()
        
        assert manager.sequence_name == "birch_to_rival"
        assert len(manager.current_sequence) > 0  # Sequence has objectives
        assert manager.current_index == 0
        
        # Check first objective
        first_obj = manager.current_sequence[0]
        assert first_obj.id == "birch_01_north_littleroot"
        assert first_obj.action_type == "navigate"
        assert first_obj.target_location == "Route 101"
        
        # Check last objective
        last_obj = manager.current_sequence[-1]
        assert last_obj.id == "birch_08_enter_lab"
        assert last_obj.action_type == "interact"
        assert last_obj.target_location == "Professor Birch's Lab"
    
    def test_load_hackathon_route102_to_petalburg_sequence(self, manager):
        """Test loading the hackathon Route 102 to Petalburg sequence"""
        manager.load_hackathon_route102_to_petalburg_sequence()
        
        assert manager.sequence_name == "hackathon_route102_to_petalburg"
        assert len(manager.current_sequence) > 0  # Sequence has objectives
        assert manager.current_index == 0
        
        # Check first objective
        first_obj = manager.current_sequence[0]
        assert first_obj.id == "hackathon_01_blue_hat_trainer"
        assert "blue hat" in first_obj.description.lower()
        
        # Check last objective
        last_obj = manager.current_sequence[-1]
        assert last_obj.id == "hackathon_05_west_petalburg"
        assert "Petalburg City" in last_obj.target_location
    
    def test_load_tutorial_to_rustboro_city_sequence(self, manager):
        """Test loading the tutorial_to_rustboro_city sequence"""
        manager.load_tutorial_to_rustboro_city_sequence()
        
        assert manager.sequence_name == "tutorial_to_rustboro_city"
        assert len(manager.current_sequence) > 0  # Sequence has objectives
        assert manager.current_index == 0
        
        # Check first objective
        first_obj = manager.current_sequence[0]
        assert first_obj.id == "tutorial_01_exit_truck"
        assert "truck" in first_obj.description.lower()
        
        # Check last objective
        last_obj = manager.current_sequence[-1]
        assert last_obj.id == "go_to_rustboro_city_pokemon_center_and_heal_pokemon"
        assert "rustboro" in last_obj.target_location.lower()
    
    def test_load_tutorial_to_rustboro_city_with_start_index(self, manager):
        """Test loading sequence with start_index parameter"""
        start_index = 5
        manager.load_tutorial_to_rustboro_city_sequence(start_index=start_index)
        
        assert manager.current_index == start_index
        # Check that objectives before start_index are marked as completed
        for i in range(start_index):
            assert manager.current_sequence[i].completed is True
            assert hasattr(manager.current_sequence[i], 'completed_at')
        
        # Check that objectives after start_index are not completed
        assert manager.current_sequence[start_index].completed is False
    
    def test_get_current_objective_empty(self, manager):
        """Test getting current objective when no sequence is loaded"""
        assert manager.get_current_objective() is None
    
    def test_get_current_objective_with_sequence(self, manager):
        """Test getting current objective when sequence is loaded"""
        manager.load_birch_to_rival_sequence()
        
        current = manager.get_current_objective()
        assert current is not None
        assert current.id == "birch_01_north_littleroot"
        assert manager.current_index == 0
    
    def test_get_current_objective_guidance_empty(self, manager):
        """Test getting guidance when no sequence is active"""
        guidance = manager.get_current_objective_guidance()
        assert guidance is None
    
    def test_get_current_objective_guidance_with_sequence(self, manager):
        """Test getting guidance for current objective"""
        manager.load_birch_to_rival_sequence()
        
        guidance = manager.get_current_objective_guidance()
        assert guidance is not None
        assert guidance["id"] == "birch_01_north_littleroot"
        assert guidance["description"] == "Move north from Littleroot Town to Route 101"
        assert guidance["action_type"] == "navigate"
        assert guidance["target_location"] == "Route 101"
        assert "navigation_hint" in guidance
        assert "completion_condition" in guidance
    
    def test_get_current_objective_guidance_no_auto_completion(self, manager):
        """Test that get_current_objective_guidance does NOT auto-complete objectives"""
        manager.load_birch_to_rival_sequence()
        
        # Get guidance multiple times - should return same objective
        guidance1 = manager.get_current_objective_guidance()
        guidance2 = manager.get_current_objective_guidance()
        
        assert guidance1["id"] == guidance2["id"]
        assert manager.current_index == 0  # Should not advance
    
    def test_mark_objective_completed(self, manager):
        """Test marking an objective as completed"""
        manager.load_birch_to_rival_sequence()
        obj = manager.get_current_objective()
        
        assert obj.completed is False
        manager._mark_objective_completed(obj)
        
        assert obj.completed is True
        assert hasattr(obj, 'completed_at')
        assert isinstance(obj.completed_at, datetime)
    
    def test_advance_to_next_objective(self, manager):
        """Test manually advancing to next objective"""
        manager.load_birch_to_rival_sequence()
        
        # Complete first objective
        obj1 = manager.get_current_objective()
        manager._mark_objective_completed(obj1)
        manager.current_index += 1
        
        # Get next objective
        obj2 = manager.get_current_objective()
        assert obj2 is not None
        assert obj2.id == "birch_02_north_route101"
        assert obj2.completed is False
    
    def test_is_sequence_active(self, manager):
        """Test checking if sequence is active"""
        assert manager.is_sequence_active() is False
        
        manager.load_birch_to_rival_sequence()
        assert manager.is_sequence_active() is True
        
        # Advance past all objectives
        manager.current_index = len(manager.current_sequence)
        assert manager.is_sequence_active() is False
    
    def test_get_sequence_status(self, manager):
        """Test getting sequence status"""
        # Empty status
        status = manager.get_sequence_status()
        assert status["sequence_name"] == ""
        assert status["total_objectives"] == 0
        assert status["current_index"] == 0
        assert status["completed_count"] == 0
        assert status["current_objective"] is None
        assert status["is_complete"] is True
        
        # Loaded sequence status
        manager.load_birch_to_rival_sequence()
        status = manager.get_sequence_status()
        assert status["sequence_name"] == "birch_to_rival"
        assert status["total_objectives"] > 0  # Sequence has objectives
        assert status["current_index"] == 0
        assert status["completed_count"] == 0
        assert status["current_objective"] is not None
        assert status["is_complete"] is False
    
    def test_get_sequence_status_with_completions(self, manager):
        """Test sequence status with some objectives completed"""
        manager.load_birch_to_rival_sequence()
        
        # Complete first 3 objectives
        for i in range(3):
            obj = manager.current_sequence[i]
            manager._mark_objective_completed(obj)
        manager.current_index = 3
        
        status = manager.get_sequence_status()
        assert status["completed_count"] == 3
        assert status["current_index"] == 3
        assert status["is_complete"] is False
    
    def test_reset_sequence(self, manager):
        """Test resetting the sequence"""
        manager.load_birch_to_rival_sequence()
        manager.current_index = 5
        
        manager.reset_sequence()
        
        assert manager.current_sequence == []
        assert manager.current_index == 0
        assert manager.sequence_name == ""
    
    def test_add_dynamic_objectives(self, manager):
        """Test adding dynamic objectives to sequence"""
        manager.load_birch_to_rival_sequence()
        initial_count = len(manager.current_sequence)
        
        dynamic_objs = [
            {
                "id": "dynamic_01",
                "description": "Dynamic objective 1",
                "action_type": "navigate",
                "target_location": "New Location",
                "navigation_hint": "Go somewhere",
                "completion_condition": "location_reached"
            },
            {
                "id": "dynamic_02",
                "description": "Dynamic objective 2",
                "action_type": "interact"
            }
        ]
        
        manager.add_dynamic_objectives(dynamic_objs)
        
        assert len(manager.current_sequence) == initial_count + 2
        assert manager.current_sequence[-2].id == "dynamic_01"
        assert manager.current_sequence[-1].id == "dynamic_02"
    
    def test_add_dynamic_objectives_empty_sequence(self, manager):
        """Test adding dynamic objectives to empty sequence"""
        dynamic_objs = [
            {
                "id": "dynamic_01",
                "description": "First dynamic objective",
                "action_type": "navigate"
            }
        ]
        
        manager.add_dynamic_objectives(dynamic_objs)
        
        assert len(manager.current_sequence) == 1
        assert manager.current_sequence[0].id == "dynamic_01"
    
    def test_get_objective_context_empty(self, manager):
        """Test getting objective context when no sequence is active"""
        context = manager.get_objective_context()
        assert context == ""
    
    def test_get_objective_context_first_objective(self, manager):
        """Test getting context for first objective (no previous)"""
        manager.load_birch_to_rival_sequence()
        
        context = manager.get_objective_context()
        assert context == ""  # No previous objective
    
    def test_get_objective_context_with_previous(self, manager):
        """Test getting context with previous objective"""
        manager.load_birch_to_rival_sequence()
        
        # Complete first objective and advance
        obj1 = manager.get_current_objective()
        manager._mark_objective_completed(obj1)
        manager.current_index = 1
        
        context = manager.get_objective_context()
        assert "PREVIOUS" in context
        assert obj1.description in context
        assert "✅" in context  # Completed status
    
    def test_get_objective_context_incomplete_previous(self, manager):
        """Test context with incomplete previous objective"""
        manager.load_birch_to_rival_sequence()
        manager.current_index = 1  # Advance without completing first
        
        context = manager.get_objective_context()
        assert "PREVIOUS" in context
        assert "❌" in context  # Incomplete status
    
    def test_save_completed_objectives(self, manager):
        """Test saving completed objectives to file"""
        manager.load_birch_to_rival_sequence()
        
        # Complete first 2 objectives
        for i in range(2):
            obj = manager.current_sequence[i]
            manager._mark_objective_completed(obj)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = manager.save_completed_objectives(run_dir=tmpdir)
            
            assert os.path.exists(filename)
            
            # Load and verify
            with open(filename, 'r') as f:
                data = json.load(f)
            
            assert "sequences" in data
            assert len(data["sequences"]) == 1
            assert data["sequences"][0]["total_objectives_completed"] == 2
            assert len(data["sequences"][0]["completed_objectives"]) == 2
    
    def test_save_completed_objectives_auto_dir(self, manager):
        """Test saving with auto-generated directory"""
        manager.load_birch_to_rival_sequence()
        manager._mark_objective_completed(manager.current_sequence[0])
        
        filename = manager.save_completed_objectives()
        
        assert os.path.exists(filename)
        assert ".pokeagent_cache" in filename
        assert "run_" in filename
        
        # Cleanup
        os.remove(filename)
        os.rmdir(os.path.dirname(filename))
        if os.path.exists(".pokeagent_cache") and not os.listdir(".pokeagent_cache"):
            os.rmdir(".pokeagent_cache")
    
    def test_save_completed_objectives_append(self, manager):
        """Test that saving appends to existing file"""
        manager.load_birch_to_rival_sequence()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First save
            manager._mark_objective_completed(manager.current_sequence[0])
            filename1 = manager.save_completed_objectives(run_dir=tmpdir)
            
            # Second save (complete another objective)
            manager.current_index = 1
            manager._mark_objective_completed(manager.current_sequence[1])
            filename2 = manager.save_completed_objectives(run_dir=tmpdir)
            
            assert filename1 == filename2  # Same file
            
            # Load and verify both saves are present
            with open(filename1, 'r') as f:
                data = json.load(f)
            
            assert len(data["sequences"]) == 2  # Two separate saves
    
    def test_load_tutorial_sequence_with_run_dir(self, manager):
        """Test loading sequence with run_dir parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            start_index = 3
            manager.load_tutorial_to_rustboro_city_sequence(
                start_index=start_index,
                run_dir=tmpdir
            )
            
            # Check that file was created
            filename = os.path.join(tmpdir, "completed_objectives.json")
            assert os.path.exists(filename)
            
            # Load and verify
            with open(filename, 'r') as f:
                data = json.load(f)
            
            assert len(data["sequences"]) == 1
            assert data["sequences"][0]["start_index"] == start_index
            assert len(data["sequences"][0]["completed_objectives"]) == start_index
    
    def test_deprecated_is_objective_completed(self, manager):
        """Test that _is_objective_completed is deprecated and returns False"""
        import warnings
        
        manager.load_birch_to_rival_sequence()
        obj = manager.current_sequence[0]
        game_state = {}
        
        # Should issue deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = manager._is_objective_completed(obj, game_state)
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        
        # Should always return False (LLM determines completion)
        assert result is False


class TestDirectObjectiveIntegration:
    """Integration tests for DirectObjectiveManager workflow"""
    
    @pytest.fixture
    def manager(self):
        """Create a DirectObjectiveManager instance for testing"""
        return DirectObjectiveManager()
    
    def test_full_workflow(self, manager):
        """Test complete workflow: load, get guidance, complete, advance"""
        # Load sequence
        manager.load_birch_to_rival_sequence()
        assert manager.is_sequence_active() is True
        
        # Get first objective guidance
        guidance1 = manager.get_current_objective_guidance()
        assert guidance1 is not None
        assert guidance1["id"] == "birch_01_north_littleroot"
        
        # Manually complete (simulating LLM calling complete_direct_objective endpoint)
        obj1 = manager.get_current_objective()
        manager._mark_objective_completed(obj1)
        manager.current_index += 1
        
        # Get second objective guidance
        guidance2 = manager.get_current_objective_guidance()
        assert guidance2 is not None
        assert guidance2["id"] == "birch_02_north_route101"
        
        # Check status
        status = manager.get_sequence_status()
        assert status["completed_count"] == 1
        assert status["current_index"] == 1
        assert status["is_complete"] is False
    
    def test_dynamic_objectives_workflow(self, manager):
        """Test workflow with dynamic objectives"""
        # Start with empty sequence
        assert manager.is_sequence_active() is False
        
        # Add dynamic objectives
        dynamic_objs = [
            {
                "id": "dynamic_01",
                "description": "Navigate to location A",
                "action_type": "navigate",
                "target_location": "Location A"
            },
            {
                "id": "dynamic_02",
                "description": "Interact with object B",
                "action_type": "interact",
                "target_location": "Location B"
            }
        ]
        manager.add_dynamic_objectives(dynamic_objs)
        
        assert manager.is_sequence_active() is True
        
        # Get guidance
        guidance = manager.get_current_objective_guidance()
        assert guidance["id"] == "dynamic_01"
        
        # Complete and advance
        obj = manager.get_current_objective()
        manager._mark_objective_completed(obj)
        manager.current_index += 1
        
        # Get next guidance
        guidance2 = manager.get_current_objective_guidance()
        assert guidance2["id"] == "dynamic_02"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


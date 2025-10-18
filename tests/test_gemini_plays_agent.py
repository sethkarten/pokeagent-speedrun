#!/usr/bin/env python3
"""
Tests for GeminiPlaysAgent
===========================

Tests the Gemini Plays Pokemon agent implementation with:
- Hierarchical goal system
- Map memory and exploration
- Specialized agent delegation
- Self-critique mechanism
- Context reset and summarization
- Meta-tools functionality
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.gemini_plays import (
    GeminiPlaysAgent, Goal, MapMemory, AgentContext,
    PathfindingAgent, BattleAgent, PuzzleAgent
)


class TestGeminiPlaysAgent:
    """Test suite for GeminiPlaysAgent"""
    
    @pytest.fixture
    def mock_vlm_client(self):
        """Create a mock VLM client"""
        mock = Mock()
        mock.get_query = Mock(return_value="Default response")
        mock.get_text_query = Mock(return_value="Default response")
        return mock
    
    @pytest.fixture
    def agent(self, mock_vlm_client):
        """Create a GeminiPlaysAgent instance for testing"""
        return GeminiPlaysAgent(
            vlm_client=mock_vlm_client,
            context_reset_interval=50,
            enable_self_critique=True,
            enable_exploration=True,
            enable_meta_tools=True,
            verbose=False
        )
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample game state"""
        return {
            "player_position": {"x": 10, "y": 20},
            "current_map": "Route 101",
            "badges": 0,
            "team": [{"name": "Torchic", "level": 5}],
            "in_battle": False,
            "dialogue": "",
            "menu": None,
            "in_pokemon_center": False,
            "in_pokemart": False
        }
    
    @pytest.fixture
    def sample_screenshot(self):
        """Create a sample screenshot"""
        return Image.new('RGB', (240, 160), color='green')
    
    def test_agent_initialization(self, mock_vlm_client):
        """Test agent initializes correctly"""
        agent = GeminiPlaysAgent(
            vlm_client=mock_vlm_client,
            context_reset_interval=100,
            enable_self_critique=True,
            enable_exploration=True,
            enable_meta_tools=False,
            verbose=True
        )
        
        assert agent.context_reset_interval == 100
        assert agent.enable_self_critique == True
        assert agent.enable_exploration == True
        assert agent.enable_meta_tools == False
        assert agent.verbose == True
        assert agent.step_count == 0
        assert isinstance(agent.map_memory, MapMemory)
        assert isinstance(agent.context, AgentContext)
    
    def test_map_memory(self):
        """Test MapMemory functionality"""
        map_mem = MapMemory()
        
        # Test marking explored
        map_mem.mark_explored(10, 20, 1)
        assert (10, 20) in map_mem.explored_tiles
        assert map_mem.last_visited[(10, 20)] == 1
        
        # Test exploration percentage
        map_mem.mark_explored(11, 20, 2)
        map_mem.mark_explored(10, 21, 3)
        percentage = map_mem.get_exploration_percentage(100)
        assert percentage == 3.0  # 3 out of 100 tiles
        
        # Test unexplored neighbors
        neighbors = map_mem.get_unexplored_neighbors(10, 20)
        assert len(neighbors) == 2  # Only left and up are unexplored
        assert (9, 20) in neighbors
        assert (10, 19) in neighbors
        
        # Test points of interest
        map_mem.points_of_interest[(10, 20)] = "pokemon_center"
        assert map_mem.points_of_interest[(10, 20)] == "pokemon_center"
    
    def test_goal_system(self):
        """Test Goal dataclass and goal management"""
        goal = Goal(
            type="primary",
            description="Defeat the first gym leader",
            conditions=["badges > 0"],
            priority=1,
            created_at=10
        )
        
        assert goal.type == "primary"
        assert goal.description == "Defeat the first gym leader"
        assert goal.conditions == ["badges > 0"]
        assert goal.priority == 1
        assert goal.created_at == 10
        assert goal.completed == False
    
    def test_goal_generation(self, agent, mock_vlm_client, sample_state):
        """Test LLM-based goal generation"""
        # Mock LLM response with proper JSON
        mock_vlm_client.get_text_query.return_value = '''
        Here are the goals:
        {
            "primary": {
                "description": "Reach Oldale Town",
                "completion_check": "location == oldale_town"
            },
            "secondary": {
                "description": "Train Torchic to level 10",
                "completion_check": "torchic_level >= 10"
            },
            "tertiary": {
                "description": "Catch a new Pokemon",
                "completion_check": "team_size > 1"
            }
        }
        '''
        
        agent._generate_new_goals(sample_state)
        
        assert agent.primary_goal is not None
        assert agent.primary_goal.description == "Reach Oldale Town"
        assert agent.secondary_goal is not None
        assert agent.secondary_goal.description == "Train Torchic to level 10"
        assert agent.tertiary_goal is not None
        assert agent.tertiary_goal.description == "Catch a new Pokemon"
    
    def test_fallback_goals(self, agent, sample_state):
        """Test fallback goal generation when LLM fails"""
        agent._set_fallback_goals(sample_state)
        
        assert agent.primary_goal is not None
        assert "gym badge" in agent.primary_goal.description.lower()
        assert agent.secondary_goal is not None
        assert "train" in agent.secondary_goal.description.lower()
        assert agent.tertiary_goal is not None
        assert "explore" in agent.tertiary_goal.description.lower()
    
    def test_map_memory_update(self, agent, sample_state):
        """Test map memory updates during step"""
        agent._update_map_memory(sample_state)
        
        # Current position should be explored
        assert (10, 20) in agent.map_memory.explored_tiles
        
        # Visible area (5x5) should be explored
        assert (8, 18) in agent.map_memory.explored_tiles
        assert (12, 22) in agent.map_memory.explored_tiles
    
    def test_stuck_detection(self, agent, sample_state):
        """Test self-critique stuck detection"""
        # Simulate being stuck at same position
        for _ in range(12):
            agent._check_if_stuck(sample_state)
        
        # Should trigger after 10 steps at same position
        assert agent.stuck_counter == 0  # Reset after critique
    
    def test_self_critique(self, agent, mock_vlm_client, sample_state):
        """Test self-critique mechanism"""
        # Add some recent actions
        agent.recent_actions.extend(["UP", "UP", "UP", "UP", "UP"])
        
        # Mock critique response
        mock_vlm_client.get_text_query.return_value = "The agent is stuck against a wall. Try moving in a different direction."
        
        with patch.object(agent, '_extract_recovery_actions'):
            agent._perform_self_critique(sample_state)
        
        # Should have called VLM for critique
        assert mock_vlm_client.get_text_query.called
    
    def test_action_queue(self, agent, sample_state, sample_screenshot):
        """Test action queue processing"""
        # Pre-fill action queue
        agent.action_queue.extend(["UP", "DOWN", "LEFT"])
        
        # Step should return queued actions
        action1 = agent.step(sample_state, sample_screenshot)
        assert action1 == "UP"
        assert len(agent.action_queue) == 2
        
        action2 = agent.step(sample_state, sample_screenshot)
        assert action2 == "DOWN"
        assert len(agent.action_queue) == 1
    
    def test_battle_delegation(self, agent, sample_state, sample_screenshot):
        """Test delegation to battle agent"""
        # Set battle state
        battle_state = sample_state.copy()
        battle_state["in_battle"] = True
        
        with patch.object(agent.battle_agent, 'execute', return_value={
            "success": True,
            "buttons": ["A", "DOWN", "A"],
            "message": "Using Fire move"
        }):
            action = agent.step(battle_state, sample_screenshot)
        
        # Should have delegated to battle agent
        assert action == "A"
        assert len(agent.action_queue) == 2  # Remaining buttons queued
    
    def test_pathfinding_agent(self, mock_vlm_client):
        """Test PathfindingAgent specialized agent"""
        pathfinding_agent = PathfindingAgent(mock_vlm_client)
        
        # Mock VLM response with coordinates
        mock_vlm_client.get_text_query.return_value = "15,25"
        
        with patch.object(pathfinding_agent.pathfinder, 'find_path', 
                         return_value=["RIGHT", "RIGHT", "DOWN", "DOWN"]):
            result = pathfinding_agent.execute(
                "Navigate to the Pokemon Center",
                {"player_position": {"x": 10, "y": 20}}
            )
        
        assert result["success"] == True
        assert len(result["buttons"]) == 4
        assert result["target"] == (15, 25)
    
    def test_battle_agent(self, mock_vlm_client):
        """Test BattleAgent specialized agent"""
        battle_agent = BattleAgent(mock_vlm_client)
        
        # Mock VLM response
        mock_vlm_client.get_text_query.return_value = "move 2"
        
        result = battle_agent.execute(
            "Win the battle",
            {"in_battle": True}
        )
        
        assert result["success"] == True
        assert result["buttons"] == ["A", "DOWN", "A"]  # Move 2 selection
    
    def test_puzzle_agent(self, mock_vlm_client, sample_screenshot):
        """Test PuzzleAgent specialized agent"""
        puzzle_agent = PuzzleAgent(mock_vlm_client)
        
        # Mock VLM response (both methods since it uses get_query when screenshot provided)
        mock_vlm_client.get_query.return_value = "UP, UP, LEFT, DOWN, A"
        mock_vlm_client.get_text_query.return_value = "UP, UP, LEFT, DOWN, A"
        
        result = puzzle_agent.execute(
            "Solve the boulder puzzle",
            {},
            sample_screenshot
        )
        
        assert result["success"] == True
        assert result["buttons"] == ["UP", "UP", "LEFT", "DOWN", "A"]
    
    def test_exploration_action(self, agent, mock_vlm_client, sample_state):
        """Test exploration directive"""
        # Mark some tiles as explored
        agent.map_memory.mark_explored(10, 20, 1)
        
        # Mock exploration decision
        mock_vlm_client.get_text_query.return_value = "RIGHT"
        
        action = agent._get_exploration_action(sample_state)
        assert action in ["UP", "DOWN", "LEFT", "RIGHT"]
    
    def test_context_reset(self, agent, sample_state):
        """Test context reset and summarization"""
        # Set up some state
        agent.primary_goal = Goal(
            type="primary",
            description="Test goal",
            conditions=[],
            priority=1,
            created_at=0
        )
        agent.map_memory.mark_explored(10, 20, 1)
        agent.recent_actions.extend(["UP", "DOWN"])
        
        agent._reset_context(sample_state)
        
        # Context should be summarized
        assert agent.context.summary != ""
        assert "Test goal" in agent.context.summary
        assert "0.1% of map" in agent.context.summary  # 1 tile out of 1000
        
        # Recent actions should be cleared
        assert len(agent.recent_actions) == 0
    
    def test_meta_tool_define_agent(self, agent, mock_vlm_client):
        """Test define_agent meta-tool"""
        response = "DEFINE_AGENT: boulder_solver Specialized agent for solving boulder puzzles"
        
        result = agent._handle_define_agent(response)
        
        assert result == True
        assert "boulder_solver" in agent.context.custom_agents
        assert "boulder puzzles" in agent.context.custom_agents["boulder_solver"]
    
    def test_meta_tool_notepad(self, agent):
        """Test notepad write/read meta-tools"""
        # Test write
        response = "NOTEPAD_WRITE: Remember to buy Pokeballs when in town"
        result = agent._handle_notepad_write(response)
        
        assert result == True
        assert len(agent.context.notepad) == 1
        assert "Pokeballs" in agent.context.notepad[0]
        
        # Test read
        result = agent._handle_notepad_read()
        assert result == True
    
    def test_meta_tool_execute_script(self, agent):
        """Test execute_script meta-tool (limited)"""
        response = "EXECUTE_SCRIPT: path = ['UP', 'UP', 'DOWN']"
        
        result = agent._handle_execute_script(response)
        
        assert result == True
        # Should add path to action queue
        assert len(agent.action_queue) == 3
        assert list(agent.action_queue) == ["UP", "UP", "DOWN"]
    
    def test_goal_completion_check(self, agent, mock_vlm_client):
        """Test goal completion checking"""
        goal = Goal(
            type="primary",
            description="Get first badge",
            conditions=["badges > 0"],
            priority=1,
            created_at=0
        )
        
        # Set step count to make it check (needs to be > 10 steps after goal created)
        agent.step_count = 11
        
        # Mock LLM evaluation
        mock_vlm_client.get_text_query.return_value = "YES"
        
        completed = agent._check_goal_completion(goal, {"badges": 1})
        assert completed == True
        
        # Test with NO response
        mock_vlm_client.get_text_query.return_value = "NO, not yet completed"
        completed = agent._check_goal_completion(goal, {"badges": 0})
        assert completed == False
    
    def test_action_parsing(self, agent):
        """Test action parsing from LLM response"""
        # Test valid button
        action = agent._parse_action("Press UP to move north")
        assert action == "UP"
        
        # Test that START is properly parsed (not confused with A)
        action = agent._parse_action("I'll press START to open the menu")
        assert action == "START"
        
        # Test B button
        action = agent._parse_action("Press the B button")
        assert action == "B"
        
        # Test tokenized response
        action = agent._parse_action("A, B, START")
        assert action == "A"  # Returns first valid token
        
        # Test fallback for unclear response
        agent.step_count = 0
        action = agent._parse_action("I'm not sure what to do")
        assert action == "UP"  # Default exploration pattern
    
    def test_integration_with_gemini(self, sample_state, sample_screenshot):
        """Test integration with gemini-2.5-flash-lite"""
        import os
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        from utils.vlm import VLM
        
        # Create real VLM client
        vlm_client = VLM(
            backend="gemini",
            model_name="gemini-2.5-flash-lite"
        )
        
        # Create agent with real VLM
        agent = GeminiPlaysAgent(
            vlm_client=vlm_client,
            context_reset_interval=100,
            enable_self_critique=False,  # Disable to speed up test
            enable_exploration=True,
            enable_meta_tools=False,  # Disable for simpler test
            verbose=True
        )
        
        # Run a step
        action = agent.step(sample_state, sample_screenshot)
        
        # Should return a valid action
        assert action in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R", "NONE"]
        
        # Goals should be generated
        assert agent.primary_goal is not None or agent.secondary_goal is not None
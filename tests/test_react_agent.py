#!/usr/bin/env python3
"""
Tests for ReActAgent
=====================

Tests the ReAct (Reasoning and Acting) agent implementation with:
- Thought generation and reasoning
- Action planning based on thoughts
- Observation processing
- History management
- Reflection mechanism
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.react import (
    ReActAgent, ActionType, Thought, Action, 
    Observation, ReActStep
)


class TestReActAgent:
    """Test suite for ReActAgent"""
    
    @pytest.fixture
    def mock_vlm_client(self):
        """Create a mock VLM client"""
        mock = Mock()
        mock.get_query = Mock(return_value="Default response")
        mock.get_text_query = Mock(return_value="Default response")
        return mock
    
    @pytest.fixture
    def agent(self, mock_vlm_client):
        """Create a ReActAgent instance for testing"""
        return ReActAgent(
            vlm_client=mock_vlm_client,
            max_history_length=10,
            enable_reflection=True,
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
            "battle_active": False,
            "dialogue": "",
            "menu": None,
            "timestamp": 100.0
        }
    
    @pytest.fixture
    def sample_screenshot(self):
        """Create a sample screenshot"""
        return Image.new('RGB', (240, 160), color='green')
    
    def test_agent_initialization(self, mock_vlm_client):
        """Test agent initializes correctly"""
        agent = ReActAgent(
            vlm_client=mock_vlm_client,
            max_history_length=20,
            enable_reflection=True,
            verbose=True
        )
        
        assert agent.max_history_length == 20
        assert agent.enable_reflection == True
        assert agent.verbose == True
        assert len(agent.history) == 0
        assert agent.current_step == 0
        assert len(agent.current_plan) == 0
        assert len(agent.memory) == 0
    
    def test_thought_dataclass(self):
        """Test Thought dataclass"""
        thought = Thought(
            content="The player is in a grassy area and should explore",
            confidence=0.8,
            reasoning_type="strategic"
        )
        
        assert thought.content == "The player is in a grassy area and should explore"
        assert thought.confidence == 0.8
        assert thought.reasoning_type == "strategic"
    
    def test_action_dataclass(self):
        """Test Action dataclass"""
        action = Action(
            type=ActionType.PRESS_BUTTON,
            parameters={"button": "A"},
            justification="Interact with NPC"
        )
        
        assert action.type == ActionType.PRESS_BUTTON
        assert action.parameters["button"] == "A"
        assert action.justification == "Interact with NPC"
    
    def test_observation_dataclass(self):
        """Test Observation dataclass"""
        obs = Observation(
            content="Player moved to new location",
            source="game_state",
            timestamp=100.5
        )
        
        assert obs.content == "Player moved to new location"
        assert obs.source == "game_state"
        assert obs.timestamp == 100.5
    
    def test_react_step_dataclass(self):
        """Test ReActStep dataclass"""
        thought = Thought(content="Test thought")
        action = Action(type=ActionType.WAIT, parameters={})
        obs = Observation(content="Test observation", source="test")
        
        step = ReActStep(
            thought=thought,
            action=action,
            observation=obs,
            step_number=1
        )
        
        assert step.thought == thought
        assert step.action == action
        assert step.observation == obs
        assert step.step_number == 1
    
    def test_thought_generation(self, agent, mock_vlm_client, sample_state, sample_screenshot):
        """Test thought generation from game state"""
        # Mock VLM response (both methods since it uses get_query when screenshot provided)
        response = """
        REASONING_TYPE: tactical
        CONFIDENCE: 0.7
        THOUGHT: We should explore the route to find trainers and items
        """
        mock_vlm_client.get_query.return_value = response
        mock_vlm_client.get_text_query.return_value = response
        
        thought = agent.think(sample_state, sample_screenshot)
        
        assert thought.reasoning_type == "tactical"
        assert thought.confidence == 0.7
        assert "explore the route" in thought.content
        
        # Check VLM was called
        assert mock_vlm_client.get_query.called or mock_vlm_client.get_text_query.called
    
    def test_action_generation(self, agent, mock_vlm_client, sample_state):
        """Test action generation based on thought"""
        thought = Thought(
            content="We need to move north",
            confidence=0.8,
            reasoning_type="tactical"
        )
        
        # Mock VLM response (use single braces for JSON since it's not an f-string)
        mock_vlm_client.get_text_query.return_value = """ACTION_TYPE: press_button
PARAMETERS: {"button": "UP"}
JUSTIFICATION: Move north to explore new area"""
        
        action = agent.act(thought, sample_state)
        
        assert action.type == ActionType.PRESS_BUTTON
        assert action.parameters.get("button") == "UP"
        assert "Move north" in action.justification
    
    def test_observation_generation(self, agent, sample_state):
        """Test observation generation after action"""
        obs = agent.observe(sample_state)
        
        assert isinstance(obs, Observation)
        assert obs.source == "game_state"
        assert obs.timestamp == 100.0
    
    def test_full_step(self, agent, mock_vlm_client, sample_state, sample_screenshot):
        """Test complete ReAct step execution"""
        # Mock thought and action responses
        thought_response = """REASONING_TYPE: general
CONFIDENCE: 0.6
THOUGHT: Should interact with the environment"""
        
        action_response = """ACTION_TYPE: press_button
PARAMETERS: {"button": "A"}
JUSTIFICATION: Try to interact"""
        
        # Mock get_query for think (with screenshot)
        mock_vlm_client.get_query.return_value = thought_response
        # Mock get_text_query for act (no screenshot)
        mock_vlm_client.get_text_query.return_value = action_response
        
        button = agent.step(sample_state, sample_screenshot)
        
        assert button == "A"
        assert len(agent.history) == 1
        assert agent.current_step == 1
        
        # Check step was recorded properly
        step = agent.history[0]
        assert step.thought is not None
        assert step.action is not None
        assert step.step_number == 1
    
    def test_action_to_button_conversion(self, agent):
        """Test converting actions to button commands"""
        # Press button action
        action = Action(
            type=ActionType.PRESS_BUTTON,
            parameters={"button": "START"}
        )
        button = agent._action_to_button(action)
        assert button == "START"
        
        # Wait action
        action = Action(type=ActionType.WAIT, parameters={})
        button = agent._action_to_button(action)
        assert button == "NONE"
        
        # Remember action
        action = Action(
            type=ActionType.REMEMBER,
            parameters={"key": "item_location", "value": "Route 101"}
        )
        button = agent._action_to_button(action)
        assert button == "NONE"
        assert agent.memory["item_location"] == "Route 101"
    
    def test_non_button_action_processing(self, agent):
        """Test processing of non-button actions"""
        # Test REMEMBER action
        action = Action(
            type=ActionType.REMEMBER,
            parameters={"key": "npc_name", "value": "Professor Birch"}
        )
        agent._process_non_button_action(action)
        assert agent.memory["npc_name"] == "Professor Birch"
        
        # Test PLAN action
        action = Action(
            type=ActionType.PLAN,
            parameters={"plan": ["Get Pokemon", "Battle rival", "Go to gym"]}
        )
        agent._process_non_button_action(action)
        assert agent.current_plan == ["Get Pokemon", "Battle rival", "Go to gym"]
        
        # Test OBSERVE action (no-op)
        action = Action(type=ActionType.OBSERVE, parameters={})
        agent._process_non_button_action(action)  # Should not error
    
    def test_history_management(self, agent):
        """Test history is maintained within max length"""
        # Add more steps than max_history_length
        for i in range(15):
            step = ReActStep(
                thought=Thought(content=f"Thought {i}"),
                action=Action(type=ActionType.WAIT, parameters={}),
                step_number=i
            )
            agent._add_to_history(step)
        
        # Should only keep last 10 (max_history_length)
        assert len(agent.history) == 10
        assert agent.history[0].step_number == 5
        assert agent.history[-1].step_number == 14
    
    def test_recent_history_summary(self, agent):
        """Test generation of recent history summary"""
        # Add some history
        for i in range(3):
            step = ReActStep(
                thought=Thought(content=f"Thought about action {i}"),
                action=Action(
                    type=ActionType.PRESS_BUTTON,
                    parameters={"button": "A"}
                ),
                observation=Observation(
                    content=f"Result of action {i}",
                    source="game_state"
                ),
                step_number=i + 1
            )
            agent.history.append(step)
        
        summary = agent._get_recent_history_summary()
        
        assert "Step 1:" in summary
        assert "Thought about action 0" in summary
        assert "press_button" in summary
        assert "Result of action" in summary
    
    def test_reflection_mechanism(self, agent, mock_vlm_client):
        """Test periodic reflection on progress"""
        # Set up history and plan
        agent.current_plan = ["Explore", "Battle", "Progress"]
        agent.history.append(
            ReActStep(
                thought=Thought(content="Test thought"),
                action=Action(type=ActionType.WAIT, parameters={}),
                step_number=1
            )
        )
        
        # Mock reflection response
        mock_vlm_client.get_text_query.return_value = "Progress is slow, should try different approach"
        
        agent._reflect_on_progress()
        
        # Check reflection was stored
        assert "last_reflection" in agent.memory
        assert "slow" in agent.memory["last_reflection"]
        assert agent.memory["reflection_step"] == agent.current_step
    
    def test_thought_parsing_edge_cases(self, agent):
        """Test thought parsing with various response formats"""
        # Test with missing fields
        thought = agent._parse_thought("Just a simple thought")
        assert thought.content == "Just a simple thought"
        assert thought.reasoning_type == "general"
        assert thought.confidence == 0.5
        
        # Test with partial fields
        response = """
        CONFIDENCE: 0.9
        This is my thought
        """
        thought = agent._parse_thought(response)
        assert thought.confidence == 0.9
        
        # Test with malformed confidence
        response = """
        CONFIDENCE: not_a_number
        THOUGHT: Valid thought
        """
        thought = agent._parse_thought(response)
        assert thought.confidence == 0.5  # Default
    
    def test_action_parsing_edge_cases(self, agent):
        """Test action parsing with various response formats"""
        # Test with missing fields
        action = agent._parse_action("Just press A")
        assert action.type == ActionType.WAIT  # Default
        
        # Test with invalid action type
        response = """
        ACTION_TYPE: invalid_action
        PARAMETERS: {"button": "A"}
        """
        action = agent._parse_action(response)
        assert action.type == ActionType.WAIT
        
        # Test with malformed JSON parameters
        response = """
        ACTION_TYPE: press_button
        PARAMETERS: not_json
        JUSTIFICATION: Press button
        """
        action = agent._parse_action(response)
        assert action.parameters == {}
    
    def test_change_detection(self, agent, sample_state):
        """Test change detection between states"""
        # Add a previous observation to history
        agent.history.append(
            ReActStep(
                observation=Observation(
                    content="Previous state",
                    source="game_state"
                ),
                step_number=1
            )
        )
        
        changes = agent._detect_changes(sample_state)
        assert "position_changed" in changes
    
    def test_change_summarization(self, agent, sample_state):
        """Test summarizing detected changes"""
        # Test with no changes
        summary = agent._summarize_changes({}, sample_state)
        assert summary == "No significant changes observed"
        
        # Test with position change
        changes = {"position_changed": True}
        summary = agent._summarize_changes(changes, sample_state)
        assert "Player moved to" in summary
        assert "{'x': 10, 'y': 20}" in summary
    
    def test_reflection_trigger(self, agent, mock_vlm_client, sample_state, sample_screenshot):
        """Test that reflection triggers every 10 steps"""
        # Mock responses for think and act
        mock_vlm_client.get_text_query.side_effect = [
            "THOUGHT: Test thought",
            "ACTION_TYPE: wait\nPARAMETERS: {}"
        ] * 20  # Enough for multiple steps
        
        with patch.object(agent, '_reflect_on_progress') as mock_reflect:
            # Run 10 steps
            for i in range(10):
                agent.step(sample_state, sample_screenshot)
            
            # Reflection should have been called at step 10
            assert mock_reflect.called
    
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
        agent = ReActAgent(
            vlm_client=vlm_client,
            max_history_length=10,
            enable_reflection=False,  # Disable for faster test
            verbose=True
        )
        
        # Test thinking
        thought = agent.think(sample_state, sample_screenshot)
        assert thought is not None
        assert thought.content != ""
        
        # Test acting
        action = agent.act(thought, sample_state)
        assert action is not None
        assert action.type in ActionType
        
        # Test full step
        button = agent.step(sample_state, sample_screenshot)
        assert button in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R", "NONE"]
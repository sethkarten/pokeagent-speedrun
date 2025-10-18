#!/usr/bin/env python3
"""
Tests for ClaudePlaysAgent
===========================

Tests the ClaudePlaysPokemon agent implementation with:
- Tool parsing and execution
- Button queue management  
- History summarization
- Navigation tool integration
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.claude_plays import ClaudePlaysAgent, ToolCall


class TestClaudePlaysAgent:
    """Test suite for ClaudePlaysAgent"""
    
    @pytest.fixture
    def mock_vlm_client(self):
        """Create a mock VLM client"""
        mock = Mock()
        mock.get_query = Mock(return_value="Default response")
        mock.get_text_query = Mock(return_value="Default response")
        return mock
    
    @pytest.fixture
    def agent(self, mock_vlm_client):
        """Create a ClaudePlaysAgent instance for testing"""
        return ClaudePlaysAgent(
            vlm_client=mock_vlm_client,
            max_history=10,
            enable_navigation=True,
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
            "menu": None
        }
    
    @pytest.fixture
    def sample_screenshot(self):
        """Create a sample screenshot"""
        # Create a simple 240x160 PIL Image
        return Image.new('RGB', (240, 160), color='green')
    
    def test_agent_initialization(self, mock_vlm_client):
        """Test agent initializes correctly"""
        agent = ClaudePlaysAgent(
            vlm_client=mock_vlm_client,
            max_history=20,
            enable_navigation=True,
            verbose=True
        )
        
        assert agent.max_history == 20
        assert agent.enable_navigation == True
        assert agent.verbose == True
        assert len(agent.messages) == 0
        assert len(agent.button_queue) == 0
        assert agent.step_count == 0
    
    def test_tool_definitions(self, agent):
        """Test tool definitions are correct"""
        tools = agent.get_tools()
        
        # Should have press_buttons and navigate_to (when enabled)
        assert len(tools) == 2
        
        # Check press_buttons tool
        press_buttons = tools[0]
        assert press_buttons["name"] == "press_buttons"
        assert "buttons" in press_buttons["input_schema"]["properties"]
        assert "reason" in press_buttons["input_schema"]["properties"]
        
        # Check navigate_to tool
        navigate = tools[1]
        assert navigate["name"] == "navigate_to"
        assert "x" in navigate["input_schema"]["properties"]
        assert "y" in navigate["input_schema"]["properties"]
    
    def test_tool_parsing(self, agent):
        """Test parsing of tool calls from response"""
        # Test single tool call
        response = '''
        I'll press A to interact with the NPC.
        <use_tool>
        {"tool": "press_buttons", "parameters": {"buttons": ["A"], "reason": "Interact with NPC"}}
        </use_tool>
        '''
        
        tool_calls = agent._parse_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "press_buttons"
        assert tool_calls[0].parameters["buttons"] == ["A"]
        assert tool_calls[0].parameters["reason"] == "Interact with NPC"
        
        # Test multiple tool calls
        response_multi = '''
        Let me navigate and then press A.
        <use_tool>
        {"tool": "navigate_to", "parameters": {"x": 15, "y": 20, "reason": "Go to door"}}
        </use_tool>
        Now let's interact.
        <use_tool>
        {"tool": "press_buttons", "parameters": {"buttons": ["A"], "reason": "Enter door"}}
        </use_tool>
        '''
        
        tool_calls_multi = agent._parse_tool_calls(response_multi)
        assert len(tool_calls_multi) == 2
        assert tool_calls_multi[0].name == "navigate_to"
        assert tool_calls_multi[1].name == "press_buttons"
    
    def test_press_buttons_tool(self, agent, sample_state):
        """Test press_buttons tool execution"""
        tool_call = ToolCall(
            name="press_buttons",
            parameters={
                "buttons": ["UP", "UP", "A", "DOWN"],
                "reason": "Navigate menu"
            }
        )
        
        agent._process_tool_call(tool_call, sample_state)
        
        # Check buttons were added to queue
        assert len(agent.button_queue) == 4
        assert agent.button_queue == ["UP", "UP", "A", "DOWN"]
    
    def test_navigate_to_tool(self, agent, sample_state):
        """Test navigate_to tool execution"""
        tool_call = ToolCall(
            name="navigate_to",
            parameters={
                "x": 15,
                "y": 25,
                "reason": "Move to next area"
            }
        )
        
        # Mock pathfinder
        with patch.object(agent.pathfinder, 'find_path', return_value=["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN"]):
            agent._process_tool_call(tool_call, sample_state)
        
        # Check path was added to queue
        assert len(agent.button_queue) == 10
        assert agent.button_queue[0] == "RIGHT"
        assert agent.button_queue[-1] == "DOWN"
    
    def test_step_with_queued_buttons(self, agent, sample_state, sample_screenshot):
        """Test step returns queued buttons first"""
        # Pre-fill button queue
        agent.button_queue = ["A", "B", "UP"]
        
        # Step should return first queued button
        action = agent.step(sample_state, sample_screenshot)
        assert action == "A"
        assert len(agent.button_queue) == 2
        
        # Next step should return next button
        action = agent.step(sample_state, sample_screenshot)
        assert action == "B"
        assert len(agent.button_queue) == 1
    
    def test_step_queries_vlm(self, agent, mock_vlm_client, sample_state, sample_screenshot):
        """Test step queries VLM when no buttons queued"""
        # Mock VLM response with tool use
        mock_vlm_client.get_query.return_value = '''
        <use_tool>
        {"tool": "press_buttons", "parameters": {"buttons": ["UP"], "reason": "Move north"}}
        </use_tool>
        '''
        
        action = agent.step(sample_state, sample_screenshot)
        
        # Should have called VLM
        assert mock_vlm_client.get_query.called
        
        # Should return the button
        assert action == "UP"
    
    def test_history_summarization(self, agent, mock_vlm_client):
        """Test history gets summarized when too long"""
        # Fill history beyond max
        for i in range(agent.max_history + 5):
            agent.messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"Message {i}"}]
            })
            agent.messages.append({
                "role": "assistant",
                "content": f"Response {i}"
            })
        
        # Mock summarization response
        mock_vlm_client.get_text_query.return_value = "Summary: Player explored Route 101 and caught a Pokemon."
        
        # Trigger summarization
        agent._summarize_history()
        
        # History should be replaced with summary
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert "Summary:" in agent.messages[0]["content"]
    
    def test_invalid_tool_parsing(self, agent):
        """Test handling of invalid tool JSON"""
        response = '''
        <use_tool>
        {invalid json here}
        </use_tool>
        '''
        
        tool_calls = agent._parse_tool_calls(response)
        assert len(tool_calls) == 0
    
    def test_filter_invalid_buttons(self, agent, sample_state):
        """Test that only valid buttons are added to queue"""
        tool_call = ToolCall(
            name="press_buttons",
            parameters={
                "buttons": ["UP", "INVALID", "A", "X", "DOWN"],
                "reason": "Test filtering"
            }
        )
        
        agent._process_tool_call(tool_call, sample_state)
        
        # Only valid buttons should be in queue
        assert agent.button_queue == ["UP", "A", "DOWN"]
    
    def test_navigation_fallback(self, agent, sample_state):
        """Test navigation fallback when pathfinding fails"""
        tool_call = ToolCall(
            name="navigate_to",
            parameters={
                "x": 12,  # 2 units right
                "y": 23,  # 3 units down
                "reason": "Move to target"
            }
        )
        
        # Mock pathfinder to return empty (no path found)
        with patch.object(agent.pathfinder, 'find_path', return_value=[]):
            agent._process_tool_call(tool_call, sample_state)
        
        # Should use simple movement as fallback
        assert len(agent.button_queue) > 0
        # Should have RIGHT and DOWN movements
        assert "RIGHT" in agent.button_queue
        assert "DOWN" in agent.button_queue
    
    def test_message_creation(self, agent, sample_state, sample_screenshot):
        """Test user message creation with state and screenshot"""
        message = agent._create_user_message(sample_state, sample_screenshot)
        
        assert message["role"] == "user"
        assert isinstance(message["content"], list)
        
        # Should have text content
        text_content = [c for c in message["content"] if c["type"] == "text"]
        assert len(text_content) > 0
        assert "Current game state:" in text_content[0]["text"]
        
        # Should have image content
        image_content = [c for c in message["content"] if c["type"] == "image"]
        assert len(image_content) == 1
        assert image_content[0]["source"]["type"] == "base64"
        assert image_content[0]["source"]["media_type"] == "image/png"
    
    def test_tool_text_formatting(self, agent):
        """Test tool descriptions are formatted correctly"""
        tools_text = agent._format_tools_text()
        
        assert "Available tools:" in tools_text
        assert "press_buttons:" in tools_text
        assert "navigate_to:" in tools_text
        assert "<use_tool>" in tools_text
        assert "</use_tool>" in tools_text
    
    def test_context_building(self, agent):
        """Test conversation context building"""
        # Add some messages
        agent.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Test user message"}]
        })
        agent.messages.append({
            "role": "assistant",
            "content": '<use_tool>{"tool": "press_buttons", "parameters": {"buttons": ["A"]}}</use_tool>'
        })
        
        context = agent._build_context()
        
        assert "Recent conversation:" in context
        assert "User: Test user message" in context
        assert '<use_tool>' in context
    
    def test_integration_with_gemini(self, sample_state, sample_screenshot):
        """Test integration with gemini-2.5-flash-lite"""
        # Only run if GEMINI_API_KEY is set
        import os
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        
        from utils.vlm import VLM
        
        # Create real VLM client with gemini
        vlm_client = VLM(
            backend="gemini",
            model_name="gemini-2.5-flash-lite"
        )
        
        # Create agent with real VLM
        agent = ClaudePlaysAgent(
            vlm_client=vlm_client,
            max_history=10,
            enable_navigation=True,
            verbose=True
        )
        
        # Run a step
        action = agent.step(sample_state, sample_screenshot)
        
        # Should return a valid action
        assert action in ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT", "L", "R", "NONE"]
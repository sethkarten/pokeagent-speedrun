#!/usr/bin/env python3
"""
Pytest for Agent Prompts Validation

Tests that validate the actual prompt outputs from agent.deprecated modules:
- agent.deprecated.action: Action decision prompts
- agent.deprecated.memory: Memory context generation
- agent.deprecated.perception: Observation and scene analysis
- agent.deprecated.planning: Strategic planning prompts

This test validates that the agent modules generate proper prompts without "Unknown" values.
"""

import pytest
import json
import sys
import os
import requests
import time
import subprocess
from typing import Dict, Any, List, Set
from unittest.mock import Mock, patch
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent modules (four-module pipeline lives under agent.deprecated)
from agent.deprecated.action import action_step
from agent.deprecated.memory import memory_step, extract_key_state_info
from agent.deprecated.perception import perception_step
from agent.deprecated.planning import planning_step
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary


class TestAgentPrompts:
    """Test class for agent prompts validation"""
    
    @pytest.fixture
    def server_url(self):
        """Server URL for testing"""
        return "http://localhost:8000"
    
    @pytest.fixture
    def mock_vlm(self):
        """Create a mock VLM that captures prompts and returns reasonable responses"""
        mock_vlm = Mock(spec=VLM)
        
        def mock_get_query(frame, prompt, context=""):
            # Capture the prompt for analysis
            if not hasattr(mock_vlm, 'captured_prompts'):
                mock_vlm.captured_prompts = []
            mock_vlm.captured_prompts.append({
                'context': context,
                'prompt': prompt,
                'frame': frame is not None
            })
            
            # Return reasonable responses based on context
            if "PERCEPTION" in context:
                return "I can see the player character on a grassy route with trees and paths."
            elif "ACTION" in context:
                return "UP"
            elif "PLANNING" in context:
                return "Continue exploring the route and look for items or trainers."
            elif "MEMORY" in context:
                return "Updated memory context with current observations."
            else:
                return "Default response"
        
        def mock_get_text_query(prompt, context=""):
            # Capture the prompt for analysis
            if not hasattr(mock_vlm, 'captured_prompts'):
                mock_vlm.captured_prompts = []
            mock_vlm.captured_prompts.append({
                'context': context,
                'prompt': prompt,
                'frame': False
            })
            
            # Return reasonable responses based on context
            if "ACTION" in context:
                return "UP"
            elif "PLANNING" in context:
                return "Continue exploring the route and look for items or trainers."
            elif "MEMORY" in context:
                return "Updated memory context with current observations."
            else:
                return "Default response"
        
        mock_vlm.get_query = mock_get_query
        mock_vlm.get_text_query = mock_get_text_query
        
        return mock_vlm
    
    def find_state_files(self):
        """Find all .state files in the tests/states directory"""
        states_dir = Path(__file__).parent / "states"
        if not states_dir.exists():
            return []
        
        state_files = list(states_dir.glob("*.state"))
        return sorted(state_files)
    
    def start_server_with_state(self, state_file_path: str):
        """Start the server with a specific state file"""
        import subprocess
        
        # Kill any existing server processes
        try:
            subprocess.run(["pkill", "-f", "server.app"], check=False)
            time.sleep(1)
        except:
            pass
        
        # Start new server
        server_process = subprocess.Popen([
            "conda", "run", "-n", "mgba", "python", "-m", "server.app", 
            "--load-state", state_file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        return server_process
    
    def stop_server(self, server_process):
        """Stop the server"""
        if server_process:
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
            except:
                server_process.kill()
    
    def get_state_from_server(self, server_url: str) -> Dict[str, Any]:
        """Get state data from the server"""
        try:
            response = requests.get(f"{server_url}/state", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except Exception:
            return {}
    
    def test_action_module_prompts(self, mock_vlm, server_url):
        """Test that action module generates proper prompts without 'Unknown' values"""
        state_files = self.find_state_files()
        
        for state_file in state_files:
            print(f"\nTesting ACTION module with {state_file}")
            
            # Start server
            server_process = self.start_server_with_state(str(state_file))
            
            try:
                # Get state data
                state_data = self.get_state_from_server(server_url)
                if not state_data:
                    pytest.skip(f"Could not get state data for {state_file}")
                
                # Mock inputs for action_step
                memory_context = "Test memory context"
                current_plan = "Test plan"
                latest_observation = "Test observation"
                frame = None
                recent_actions = ["UP", "A", "RIGHT"]
                
                # Call action_step
                actions = action_step(memory_context, current_plan, latest_observation, frame, state_data, recent_actions, mock_vlm)
                
                # Check captured prompts
                action_prompts = [p for p in mock_vlm.captured_prompts if "ACTION" in p['context']]
                
                assert action_prompts, f"No action prompts captured for {state_file}"
                
                # Analyze the action prompt for issues
                action_prompt = action_prompts[0]['prompt']
                
                # Check for "Unknown" values in the prompt
                assert "Unknown" not in action_prompt, f"Action prompt contains 'Unknown' values in {state_file}"
                
                # Check for required sections
                required_sections = ["COMPREHENSIVE GAME STATE DATA", "ENHANCED ACTION CONTEXT", "ACTION DECISION TASK"]
                for section in required_sections:
                    assert section in action_prompt, f"Action prompt missing section '{section}' in {state_file}"
                
                # Check that actions were returned
                assert actions, f"Action module returned no actions for {state_file}"
                assert isinstance(actions, list), f"Action module returned non-list actions for {state_file}"
                
            finally:
                self.stop_server(server_process)
    
    def test_memory_module_prompts(self, mock_vlm, server_url):
        """Test that memory module generates proper prompts without 'Unknown' values"""
        state_files = self.find_state_files()
        
        for state_file in state_files:
            print(f"\nTesting MEMORY module with {state_file}")
            
            # Start server
            server_process = self.start_server_with_state(str(state_file))
            
            try:
                # Get state data
                state_data = self.get_state_from_server(server_url)
                if not state_data:
                    pytest.skip(f"Could not get state data for {state_file}")
                
                # Test extract_key_state_info function
                key_info = extract_key_state_info(state_data)
                
                # Check for "Unknown" values in key info
                assert "Unknown" not in str(key_info), f"Memory key_info contains 'Unknown' values in {state_file}"
                
                # Check for required fields
                required_fields = ['state_summary', 'player_name', 'money', 'current_map', 'in_battle', 'party_health']
                for field in required_fields:
                    assert field in key_info, f"Memory key_info missing field '{field}' in {state_file}"
                
                # Test memory_step function
                memory_context = "Test memory context"
                current_plan = "Test plan"
                recent_actions = ["UP", "A", "RIGHT"]
                observation_buffer = [
                    {
                        "frame_id": 1,
                        "observation": "Test observation",
                        "state": state_data
                    }
                ]
                
                # Call memory_step
                updated_memory = memory_step(memory_context, current_plan, recent_actions, observation_buffer, mock_vlm)
                
                # Check for "Unknown" values in memory context
                assert "Unknown" not in updated_memory, f"Memory context contains 'Unknown' values in {state_file}"
                
                # Check for required sections
                required_sections = ["COMPREHENSIVE MEMORY CONTEXT", "CURRENT STATE", "CURRENT PLAN", "KEY EVENTS", "RECENT MEMORY"]
                for section in required_sections:
                    assert section in updated_memory, f"Memory context missing section '{section}' in {state_file}"
                
                # Check that memory context is not empty
                assert len(updated_memory.strip()) > 100, f"Memory context seems too short in {state_file}"
                
            finally:
                self.stop_server(server_process)
    
    def test_perception_module_prompts(self, mock_vlm, server_url):
        """Test that perception module generates proper prompts without 'Unknown' values"""
        state_files = self.find_state_files()
        
        for state_file in state_files:
            print(f"\nTesting PERCEPTION module with {state_file}")
            
            # Start server
            server_process = self.start_server_with_state(str(state_file))
            
            try:
                # Get state data
                state_data = self.get_state_from_server(server_url)
                if not state_data:
                    pytest.skip(f"Could not get state data for {state_file}")
                
                # Mock frame
                frame = None
                
                # Call perception_step
                observation, slow_thinking = perception_step(frame, state_data, mock_vlm)
                
                # Check captured prompts
                perception_prompts = [p for p in mock_vlm.captured_prompts if "PERCEPTION" in p['context']]
                
                assert perception_prompts, f"No perception prompts captured for {state_file}"
                
                # Analyze the perception prompt for issues
                perception_prompt = perception_prompts[0]['prompt']
                
                # Check for "Unknown" values in the prompt
                assert "Unknown" not in perception_prompt, f"Perception prompt contains 'Unknown' values in {state_file}"
                
                # Check for required sections
                required_sections = ["COMPREHENSIVE GAME STATE DATA", "VISUAL ANALYSIS TASK"]
                for section in required_sections:
                    assert section in perception_prompt, f"Perception prompt missing section '{section}' in {state_file}"
                
                # Check for analysis instructions
                analysis_keywords = ["CUTSCENE", "MAP", "BATTLE", "DIALOGUE", "MENU"]
                found_keywords = [kw for kw in analysis_keywords if kw in perception_prompt]
                assert len(found_keywords) >= 3, f"Perception prompt missing analysis keywords in {state_file}. Found: {found_keywords}"
                
                # Check that observation was returned
                assert observation, f"Perception module returned no observation for {state_file}"
                assert isinstance(observation, dict), f"Perception module returned non-dict observation for {state_file}"
                
                # Check that slow_thinking is boolean
                assert isinstance(slow_thinking, bool), f"Perception module returned non-boolean slow_thinking for {state_file}"
                
            finally:
                self.stop_server(server_process)
    
    def test_planning_module_prompts(self, mock_vlm, server_url):
        """Test that planning module generates proper prompts without 'Unknown' values"""
        state_files = self.find_state_files()
        
        for state_file in state_files:
            print(f"\nTesting PLANNING module with {state_file}")
            
            # Start server
            server_process = self.start_server_with_state(str(state_file))
            
            try:
                # Get state data
                state_data = self.get_state_from_server(server_url)
                if not state_data:
                    pytest.skip(f"Could not get state data for {state_file}")
                
                # Mock inputs for planning_step
                memory_context = "Test memory context"
                current_plan = None  # Start with no plan
                slow_thinking_needed = True
                
                # Call planning_step
                plan = planning_step(memory_context, current_plan, slow_thinking_needed, state_data, mock_vlm)
                
                # Check captured prompts
                planning_prompts = [p for p in mock_vlm.captured_prompts if "PLANNING" in p['context']]
                
                assert planning_prompts, f"No planning prompts captured for {state_file}"
                
                # Analyze the planning prompt for issues
                planning_prompt = planning_prompts[0]['prompt']
                
                # Check for "Unknown" values in the prompt
                assert "Unknown" not in planning_prompt, f"Planning prompt contains 'Unknown' values in {state_file}"
                
                # Check for required sections
                required_sections = ["COMPREHENSIVE GAME STATE DATA", "STRATEGIC PLANNING TASK"]
                for section in required_sections:
                    assert section in planning_prompt, f"Planning prompt missing section '{section}' in {state_file}"
                
                # Check for planning instructions
                planning_keywords = ["IMMEDIATE GOAL", "SHORT-TERM OBJECTIVES", "LONG-TERM STRATEGY", "EFFICIENCY NOTES"]
                found_keywords = [kw for kw in planning_keywords if kw in planning_prompt]
                assert len(found_keywords) >= 3, f"Planning prompt missing planning keywords in {state_file}. Found: {found_keywords}"
                
                # Check that plan was returned
                assert plan, f"Planning module returned no plan for {state_file}"
                assert isinstance(plan, str), f"Planning module returned non-string plan for {state_file}"
                
            finally:
                self.stop_server(server_process)
    
    def test_all_modules_integration(self, mock_vlm, server_url):
        """Test that all modules work together without 'Unknown' values"""
        state_files = self.find_state_files()
        
        for state_file in state_files:
            print(f"\nTesting ALL MODULES integration with {state_file}")
            
            # Start server
            server_process = self.start_server_with_state(str(state_file))
            
            try:
                # Get state data
                state_data = self.get_state_from_server(server_url)
                if not state_data:
                    pytest.skip(f"Could not get state data for {state_file}")
                
                # Test all modules in sequence
                memory_context = "Test memory context"
                current_plan = None
                latest_observation = "Test observation"
                frame = None
                recent_actions = ["UP", "A", "RIGHT"]
                observation_buffer = [
                    {
                        "frame_id": 1,
                        "observation": "Test observation",
                        "state": state_data
                    }
                ]
                
                # 1. Perception
                observation, slow_thinking = perception_step(frame, state_data, mock_vlm)
                assert observation and isinstance(observation, dict)
                
                # 2. Memory
                updated_memory = memory_step(memory_context, current_plan, recent_actions, observation_buffer, mock_vlm)
                assert "Unknown" not in updated_memory
                
                # 3. Planning
                plan = planning_step(updated_memory, current_plan, slow_thinking, state_data, mock_vlm)
                assert plan and isinstance(plan, str)
                
                # 4. Action
                actions = action_step(updated_memory, plan, observation, frame, state_data, recent_actions, mock_vlm)
                assert actions and isinstance(actions, list)
                
                # Check that no prompts contain "Unknown"
                for prompt_data in mock_vlm.captured_prompts:
                    assert "Unknown" not in prompt_data['prompt'], f"Found 'Unknown' in {prompt_data['context']} prompt for {state_file}"
                
            finally:
                self.stop_server(server_process)


if __name__ == "__main__":
    pytest.main([__file__]) 
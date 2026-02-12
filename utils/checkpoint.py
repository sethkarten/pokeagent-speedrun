#!/usr/bin/env python3
"""
Checkpoint management utilities for saving and loading game states,
agent states, and LLM history.
"""

import os
import json
import traceback
from collections import deque
from datetime import datetime


def save_checkpoint(emulator, llm_logger=None, agent_step_count=0):
    """
    Save a complete checkpoint including game state, milestones, and LLM history.
    
    Args:
        emulator: The game emulator instance
        llm_logger: Optional LLM logger for saving conversation history
        agent_step_count: Current agent step count
    
    Returns:
        bool: True if checkpoint saved successfully, False otherwise
    """
    try:
        from utils.run_data_manager import get_cache_path
        
        # Save emulator state
        print("💾 Saving checkpoint...")
        checkpoint_state = get_cache_path("checkpoint.state")
        emulator.save_state(str(checkpoint_state))
        print(f"   ✅ Saved emulator state to {checkpoint_state}")
        
        # Save milestones
        milestone_data = {
            "milestones": emulator.memory_reader.milestones if emulator.memory_reader else {},
            "step_count": agent_step_count,
            "timestamp": datetime.now().isoformat()
        }
        milestones_file = get_cache_path("checkpoint_milestones.json")
        with open(milestones_file, "w") as f:
            json.dump(milestone_data, f, indent=2)
        print(f"   ✅ Saved milestones to {milestones_file}")
        
        # Save location maps
        try:
            from utils.state_formatter import save_persistent_world_map
            maps_file = get_cache_path("checkpoint_maps.json")
            save_persistent_world_map(str(maps_file))
            print(f"   ✅ Saved location maps to {maps_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to save location maps: {e}")
        
        # Save LLM history if logger available
        if llm_logger:
            # Get conversation history from logger
            history = llm_logger.get_conversation_history()
            
            # Also get metrics if available
            try:
                metrics = llm_logger.get_interaction_metrics()
                checkpoint_data = {
                    "history": history,
                    "metrics": metrics,
                    "step_counter": agent_step_count,
                    "timestamp": datetime.now().isoformat()
                }
            except:
                checkpoint_data = {
                    "history": history,
                    "step_counter": agent_step_count,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Save to cache folder
            checkpoint_file = get_cache_path("checkpoint_llm.txt")
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"   ✅ Saved LLM history to {checkpoint_file}")
        
        print(f"✅ Checkpoint saved at step {agent_step_count}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        traceback.print_exc()
        return False


def load_checkpoint(emulator, llm_logger=None):
    """
    Load a checkpoint including game state, milestones, and LLM history.
    
    Args:
        emulator: The game emulator instance
        llm_logger: Optional LLM logger for loading conversation history
    
    Returns:
        dict: Checkpoint data including step_count, or None if failed
    """
    try:
        from utils.run_data_manager import get_cache_path
        
        checkpoint_data = {}
        
        # Load emulator state
        checkpoint_state = get_cache_path("checkpoint.state")
        if checkpoint_state.exists():
            print("🔄 Loading checkpoint...")
            emulator.load_state(str(checkpoint_state))
            print(f"   ✅ Loaded emulator state from {checkpoint_state}")
        else:
            print(f"   ⚠️ No {checkpoint_state} found")
            return None
        
        # Load milestones
        milestones_file = get_cache_path("checkpoint_milestones.json")
        if milestones_file.exists():
            with open(milestones_file, "r") as f:
                milestone_data = json.load(f)
            if emulator.memory_reader:
                emulator.memory_reader.milestones = milestone_data.get("milestones", {})
            checkpoint_data['step_count'] = milestone_data.get("step_count", 0)
            print(f"   ✅ Loaded milestones from {milestones_file}")
        
        # Load location maps
        try:
            from utils.state_formatter import load_persistent_world_map
            maps_file = get_cache_path("checkpoint_maps.json")
            if maps_file.exists():
                load_persistent_world_map(str(maps_file))
                print(f"   ✅ Loaded location maps from {maps_file}")
            else:
                print(f"   ⚠️ No {maps_file} found")
        except Exception as e:
            print(f"   ⚠️ Failed to load location maps: {e}")
        
        # Load LLM history if logger available
        checkpoint_file = get_cache_path("checkpoint_llm.txt")
        if llm_logger and checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                llm_data = json.load(f)
            
            # Restore conversation history
            if 'history' in llm_data:
                llm_logger.set_conversation_history(llm_data['history'])
            
            checkpoint_data['llm_step_count'] = llm_data.get('step_counter', 0)
            print(f"   ✅ Loaded LLM history from {checkpoint_file}")
        
        print(f"✅ Checkpoint loaded successfully")
        return checkpoint_data
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        traceback.print_exc()
        return None


def save_simple_agent_state(simple_agent, filename="agent_state.json"):
    """
    Save SimpleAgent state to JSON file.
    
    Args:
        simple_agent: The SimpleAgent instance
        filename: Output filename
    
    Returns:
        bool: True if saved successfully
    """
    try:
        # Convert SimpleAgent state to serializable format
        state_data = {
            "step_counter": simple_agent.state.step_counter,
            "stuck_detection": simple_agent.state.stuck_detection,
            "objectives_updated": simple_agent.state.objectives_updated,
            "history": [],
            "objectives": []
        }
        
        # Convert history entries
        for entry in simple_agent.state.history:
            state_data["history"].append({
                "timestamp": entry.timestamp.isoformat() if hasattr(entry.timestamp, 'isoformat') else str(entry.timestamp),
                "screenshot": None,  # Don't save screenshots
                "game_state": entry.game_state,
                "llm_output": entry.llm_output
            })
        
        # Convert objectives
        for obj in simple_agent.state.objectives:
            state_data["objectives"].append({
                "description": obj.description,
                "completed": obj.completed
            })
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        print(f"✅ Saved SimpleAgent state to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save SimpleAgent state: {e}")
        traceback.print_exc()
        return False


def load_simple_agent_state(simple_agent, filename="agent_state.json"):
    """
    Load SimpleAgent state from JSON file.
    
    Args:
        simple_agent: The SimpleAgent instance to load state into
        filename: Input filename
    
    Returns:
        bool: True if loaded successfully
    """
    try:
        from agent.simple import HistoryEntry, Objective
        
        with open(filename, 'r') as f:
            state_data = json.load(f)
        
        # Restore basic counters
        simple_agent.state.step_counter = state_data.get("step_counter", 0)
        simple_agent.state.stuck_detection = state_data.get("stuck_detection", {})
        simple_agent.state.objectives_updated = state_data.get("objectives_updated", False)
        
        # Restore history
        simple_agent.state.history = deque(maxlen=10)
        for entry_data in state_data.get("history", []):
            # Parse timestamp
            timestamp_str = entry_data.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except:
                timestamp = datetime.now()
            
            entry = HistoryEntry(
                timestamp=timestamp,
                screenshot=None,  # Don't restore screenshots
                game_state=entry_data.get("game_state", {}),
                llm_output=entry_data.get("llm_output", {})
            )
            simple_agent.state.history.append(entry)
        
        # Restore objectives
        simple_agent.state.objectives = []
        for obj_data in state_data.get("objectives", []):
            obj = Objective(
                description=obj_data.get("description", ""),
                completed=obj_data.get("completed", False)
            )
            simple_agent.state.objectives.append(obj)
        
        print(f"✅ Loaded SimpleAgent state from {filename}")
        print(f"   - Step counter: {simple_agent.state.step_counter}")
        print(f"   - History entries: {len(simple_agent.state.history)}")
        print(f"   - Objectives: {len(simple_agent.state.objectives)}")
        
        return True
        
    except FileNotFoundError:
        print(f"⚠️ No saved state found at {filename}")
        return False
    except Exception as e:
        print(f"❌ Failed to load SimpleAgent state: {e}")
        traceback.print_exc()
        return False


def load_llm_checkpoint(filename="checkpoint_llm.txt"):
    """
    Load LLM checkpoint data from file.
    
    Args:
        filename: Input filename
    
    Returns:
        dict: Checkpoint data or None if failed
    """
    try:
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return data
        
    except Exception as e:
        print(f"❌ Failed to load LLM checkpoint: {e}")
        return None
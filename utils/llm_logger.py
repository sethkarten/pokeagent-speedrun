#!/usr/bin/env python3
"""
LLM Logger utility for logging all VLM interactions

This module provides a centralized logging system for all LLM interactions,
including input prompts, responses, and metadata. Logs are saved to dated
files in the llm_logs directory.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LLMLogger:
    """Logger for all LLM interactions"""
    
    def __init__(self, log_dir: str = "llm_logs"):
        """Initialize the LLM logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"llm_log_{self.session_id}.jsonl")
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize cumulative metrics
        self.cumulative_metrics = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "total_actions": 0,
            "start_time": time.time(),
            "total_llm_calls": 0
        }
        
        # Model pricing (per 1K tokens) - can be updated based on actual pricing
        self.pricing = {
            "gpt-4o": {"prompt": 0.01, "completion": 0.03},
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
            "o3-mini": {"prompt": 0.0012, "completion": 0.0048},
            "gemini-2.5-flash": {"prompt": 0.000315, "completion": 0.00126},
            "gemini-2.5-pro": {"prompt": 0.00125, "completion": 0.005},
            "default": {"prompt": 0.001, "completion": 0.002}  # Default pricing
        }
        
        # Initialize log file with session info
        self._log_session_start()
        
        logger.info(f"LLM Logger initialized: {self.log_file}")
    
    def _log_session_start(self):
        """Log session start information"""
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "type": "session_start",
            "session_id": self.session_id,
            "log_file": self.log_file
        }
        self._write_log_entry(session_info)
    
    def log_interaction(self, 
                       interaction_type: str,
                       prompt: str,
                       response: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       duration: Optional[float] = None,
                       model_info: Optional[Dict[str, Any]] = None):
        """Log a complete LLM interaction
        
        Args:
            interaction_type: Type of interaction (e.g., "perception", "planning", "action")
            prompt: The input prompt sent to the LLM
            response: The response received from the LLM
            metadata: Additional metadata about the interaction
            duration: Time taken for the interaction in seconds
            model_info: Information about the model used
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "interaction",
            "interaction_type": interaction_type,
            "prompt": prompt,
            "response": response,
            "duration": duration,
            "metadata": metadata or {},
            "model_info": model_info or {}
        }
        
        self._write_log_entry(log_entry)
        
        # Update cumulative metrics
        self.cumulative_metrics["total_llm_calls"] += 1
        
        # Track token usage if available
        if metadata and "token_usage" in metadata:
            token_usage = metadata["token_usage"]
            if token_usage:
                self.cumulative_metrics["total_tokens"] += token_usage.get("total_tokens", 0)
                self.cumulative_metrics["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                self.cumulative_metrics["completion_tokens"] += token_usage.get("completion_tokens", 0)
                
                # Calculate cost based on model
                model_name = model_info.get("model", "") if model_info else ""
                pricing = self.pricing.get("default")
                for key in self.pricing:
                    if key in model_name.lower():
                        pricing = self.pricing[key]
                        break
                
                prompt_cost = (token_usage.get("prompt_tokens", 0) / 1000) * pricing["prompt"]
                completion_cost = (token_usage.get("completion_tokens", 0) / 1000) * pricing["completion"]
                self.cumulative_metrics["total_cost"] += prompt_cost + completion_cost
        
        # Track actions if this is an action interaction
        if "action" in interaction_type.lower():
            # Count actions in response - look for valid button presses
            # Response could be single button like "A" or multiple like "A A B" or with commas
            valid_buttons = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']
            
            # Convert response to uppercase and split by spaces or commas
            response_upper = response.upper()
            tokens = response_upper.replace(',', ' ').split()
            
            # Count each valid button found
            action_count = sum(1 for token in tokens if token in valid_buttons)
            
            # If no actions found but response contains button names, count them
            if action_count == 0:
                # Also check for arrow notations
                action_count += response_upper.count('UP')
                action_count += response_upper.count('DOWN')
                action_count += response_upper.count('LEFT')  
                action_count += response_upper.count('RIGHT')
                action_count += response.count('↑')
                action_count += response.count('↓')
                action_count += response.count('←')
                action_count += response.count('→')
                # Count single letter buttons
                for char in 'ABLR':
                    if char in response_upper:
                        action_count += response_upper.count(char)
            
            if action_count > 0:
                self.cumulative_metrics["total_actions"] += action_count
                logger.debug(f"Counted {action_count} actions in response: {response[:50]}")
        
        # Also log to console for debugging
        if duration is not None:
            logger.info(f"LLM {interaction_type.upper()}: {duration:.2f}s")
            logger.debug(f"Prompt length: {len(prompt)} chars, Response length: {len(response)} chars")
        else:
            logger.info(f"LLM {interaction_type.upper()}")
    
    def log_error(self, 
                  interaction_type: str,
                  prompt: str,
                  error: str,
                  metadata: Optional[Dict[str, Any]] = None):
        """Log an LLM interaction error
        
        Args:
            interaction_type: Type of interaction that failed
            prompt: The input prompt that was sent
            error: The error message
            metadata: Additional metadata about the error
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "interaction_type": interaction_type,
            "prompt": prompt,
            "error": error,
            "metadata": metadata or {}
        }
        
        self._write_log_entry(log_entry)
        logger.error(f"LLM {interaction_type.upper()} ERROR: {error}")
    
    def log_step_start(self, step: int, step_type: str = "agent_step"):
        """Log the start of an agent step
        
        Args:
            step: Step number
            step_type: Type of step (e.g., "agent_step", "perception", "planning")
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "step_start",
            "step": step,
            "step_type": step_type
        }
        
        self._write_log_entry(log_entry)
        logger.info(f"Starting {step_type} {step}")
    
    def log_step_end(self, step: int, step_type: str = "agent_step", 
                    duration: Optional[float] = None,
                    summary: Optional[str] = None):
        """Log the end of an agent step
        
        Args:
            step: Step number
            step_type: Type of step
            duration: Time taken for the step
            summary: Summary of what happened in the step
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "step_end",
            "step": step,
            "step_type": step_type,
            "duration": duration,
            "summary": summary
        }
        
        self._write_log_entry(log_entry)
        if duration:
            logger.info(f"Completed {step_type} {step} in {duration:.2f}s")
        else:
            logger.info(f"Completed {step_type} {step}")
    
    def log_state_snapshot(self, state_data: Dict[str, Any], step: int):
        """Log a snapshot of the game state
        
        Args:
            state_data: The game state data
            step: Current step number
        """
        # Extract key information to avoid logging too much data
        state_summary = {
            "step": step,
            "player_location": state_data.get("player", {}).get("location"),
            "player_position": state_data.get("player", {}).get("position"),
            "game_state": state_data.get("game", {}).get("game_state"),
            "is_in_battle": state_data.get("game", {}).get("is_in_battle"),
            "party_size": len(state_data.get("player", {}).get("party", [])),
            "money": state_data.get("game", {}).get("money"),
            "dialog_text": state_data.get("game", {}).get("dialog_text", "")[:100] + "..." if state_data.get("game", {}).get("dialog_text") else None
        }
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "state_snapshot",
            "step": step,
            "state_summary": state_summary
        }
        
        self._write_log_entry(log_entry)
    
    def log_action(self, action: str, step: int, reasoning: Optional[str] = None):
        """Log an action taken by the agent
        
        Args:
            action: The action taken
            step: Current step number
            reasoning: Reasoning behind the action
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "action",
            "step": step,
            "action": action,
            "reasoning": reasoning
        }
        
        self._write_log_entry(log_entry)
        logger.info(f"Action {step}: {action}")
        if reasoning:
            logger.debug(f"Reasoning: {reasoning}")
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write a log entry to the log file
        
        Args:
            log_entry: The log entry to write
        """
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write log entry: {e}")
    
    def get_cumulative_metrics(self) -> Dict[str, Any]:
        """Get cumulative metrics for the session
        
        Returns:
            Dictionary with cumulative metrics
        """
        # Update runtime
        self.cumulative_metrics["total_run_time"] = time.time() - self.cumulative_metrics["start_time"]
        return self.cumulative_metrics.copy()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session
        
        Returns:
            Dictionary with session summary information
        """
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            interactions = 0
            errors = 0
            total_duration = 0.0
            
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("type") == "interaction":
                        interactions += 1
                        if entry.get("duration"):
                            total_duration += entry["duration"]
                    elif entry.get("type") == "error":
                        errors += 1
                except json.JSONDecodeError:
                    continue
            
            return {
                "session_id": self.session_id,
                "log_file": self.log_file,
                "total_interactions": interactions,
                "total_errors": errors,
                "total_duration": total_duration,
                "average_duration": total_duration / interactions if interactions > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {"error": str(e)}
    
    def save_checkpoint(self, checkpoint_file: str = None, agent_step_count: int = None):
        """Save current LLM interaction history to checkpoint file
        
        Args:
            checkpoint_file: Path to save the checkpoint (defaults to cache folder)
            agent_step_count: Current agent step count for persistence
        """
        try:
            # Use cache folder by default
            if checkpoint_file is None or checkpoint_file == "checkpoint_llm.txt":
                cache_dir = ".pokeagent_cache"
                os.makedirs(cache_dir, exist_ok=True)
                checkpoint_file = os.path.join(cache_dir, "checkpoint_llm.txt")
            # Read all current log entries
            log_entries = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            
            # Update run time in metrics
            self.cumulative_metrics["total_run_time"] = time.time() - self.cumulative_metrics["start_time"]
            
            # Add checkpoint metadata
            checkpoint_data = {
                "checkpoint_timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "original_log_file": self.log_file,
                "total_entries": len(log_entries),
                "agent_step_count": agent_step_count,  # Save current step count
                "cumulative_metrics": self.cumulative_metrics,  # Save metrics
                "log_entries": log_entries
            }
            
            # Add map stitcher data if available via callback
            if hasattr(self, '_map_stitcher_callback') and self._map_stitcher_callback:
                try:
                    self._map_stitcher_callback(checkpoint_data)
                except Exception as e:
                    logger.debug(f"Failed to save map stitcher to checkpoint: {e}")
            
            # Save to checkpoint file
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"LLM checkpoint saved: {checkpoint_file} ({len(log_entries)} entries)")
            
        except Exception as e:
            logger.error(f"Failed to save LLM checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str = None) -> Optional[int]:
        """Load LLM interaction history from checkpoint file
        
        Args:
            checkpoint_file: Path to load the checkpoint from (defaults to cache folder)
            
        Returns:
            Last agent step count from the checkpoint, or None if not found
        """
        try:
            # Use cache folder by default
            if checkpoint_file is None or checkpoint_file == "checkpoint_llm.txt":
                cache_dir = ".pokeagent_cache"
                checkpoint_file = os.path.join(cache_dir, "checkpoint_llm.txt")
            
            if not os.path.exists(checkpoint_file):
                logger.info(f"No checkpoint file found at {checkpoint_file}")
                return None
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            log_entries = checkpoint_data.get("log_entries", [])
            
            # Restore cumulative metrics if available
            if "cumulative_metrics" in checkpoint_data:
                saved_metrics = checkpoint_data["cumulative_metrics"]
                # Restore all metrics including the original start_time
                self.cumulative_metrics.update(saved_metrics)
                
                # If the checkpoint has a start_time, use it to preserve the original session start
                if "start_time" in saved_metrics:
                    logger.info(f"Restored original start time from checkpoint: {saved_metrics['start_time']}")
                else:
                    logger.warning("No start_time found in checkpoint, using current time")
            
            # Restore log entries to current log file
            with open(self.log_file, 'w', encoding='utf-8') as f:
                for entry in log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Try to get step count from checkpoint metadata first
            last_step = checkpoint_data.get("agent_step_count")
            
            # If not in metadata, find the last agent step from log entries
            if last_step is None:
                for entry in reversed(log_entries):
                    if entry.get("type") == "step_start" and "step_number" in entry:
                        last_step = entry["step_number"]
                        break
            
            logger.info(f"LLM checkpoint loaded: {checkpoint_file} ({len(log_entries)} entries, step {last_step})")
            
            # Load map stitcher data if available via callback
            if hasattr(self, '_map_stitcher_load_callback') and self._map_stitcher_load_callback:
                try:
                    self._map_stitcher_load_callback(checkpoint_data)
                except Exception as e:
                    logger.debug(f"Failed to load map stitcher from checkpoint: {e}")
            
            return last_step
            
        except Exception as e:
            logger.error(f"Failed to load LLM checkpoint: {e}")
            return None

# Global logger instance
_llm_logger = None

def get_llm_logger() -> LLMLogger:
    """Get the global LLM logger instance
    
    Returns:
        The global LLM logger instance
    """
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = LLMLogger()
    return _llm_logger

def setup_map_stitcher_checkpoint_integration(memory_reader):
    """Set up map stitcher integration with checkpoint system"""
    logger = get_llm_logger()
    
    def save_callback(checkpoint_data):
        if hasattr(memory_reader, '_map_stitcher') and memory_reader._map_stitcher:
            memory_reader._map_stitcher.save_to_checkpoint(checkpoint_data)
    
    def load_callback(checkpoint_data):
        if hasattr(memory_reader, '_map_stitcher') and memory_reader._map_stitcher:
            memory_reader._map_stitcher.load_from_checkpoint(checkpoint_data)
    
    logger._map_stitcher_callback = save_callback
    logger._map_stitcher_load_callback = load_callback

def log_llm_interaction(interaction_type: str, prompt: str, response: str, 
                       metadata: Optional[Dict[str, Any]] = None,
                       duration: Optional[float] = None,
                       model_info: Optional[Dict[str, Any]] = None):
    """Convenience function to log an LLM interaction
    
    Args:
        interaction_type: Type of interaction
        prompt: Input prompt
        response: LLM response
        metadata: Additional metadata
        duration: Time taken
        model_info: Model information
    """
    logger = get_llm_logger()
    logger.log_interaction(interaction_type, prompt, response, metadata, duration, model_info)

def log_llm_error(interaction_type: str, prompt: str, error: str, 
                 metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to log an LLM error
    
    Args:
        interaction_type: Type of interaction that failed
        prompt: Input prompt
        error: Error message
        metadata: Additional metadata
    """
    logger = get_llm_logger()
    logger.log_error(interaction_type, prompt, error, metadata) 
#!/usr/bin/env python3
"""
Run Data Manager - Centralized data collection

This module manages all run-specific data in a structured format to aid
in the later analysis of data, debugging, and prompt optimization.
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class RunDataManager:
    """Manages structured data collection for a single run"""
    
    def __init__(self, run_id: Optional[str] = None, base_dir: str = "run_data", run_name: Optional[str] = None,
                 first_objective_id: Optional[str] = None, first_objective_desc: Optional[str] = None):
        """Initialize run data manager
        
        Args:
            run_id: Optional run identifier. If None, creates timestamped ID.
            base_dir: Base directory for all runs (default: run_data)
            run_name: Optional name to append to run_id (deprecated - use objectives)
            first_objective_id: ID of the first objective (for consistent naming)
            first_objective_desc: Description of the first objective (for consistent naming)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use objective info for run_id if provided
            if first_objective_id and first_objective_desc:
                # Sanitize objective description (remove special chars, limit length)
                safe_desc = "".join(c if c.isalnum() or c in ('_', '-', ' ') else '_' for c in first_objective_desc)
                safe_desc = safe_desc.replace(' ', '_')[:50]  # Limit length
                run_id = f"{timestamp}_{first_objective_id}_{safe_desc}"
            # Fallback to run_name if provided (deprecated)
            elif run_name:
                import re
                sanitized_name = re.sub(r'[^\w\-_]', '_', run_name)
                run_id = f"run_{timestamp}_{sanitized_name}"
            else:
                # Default format (no objectives specified)
                run_id = f"run_{timestamp}"
        
        self.run_id = run_id
        self.run_dir = self.base_dir / run_id
        
        # Create directory structure
        self._create_directory_structure()
        
        # Track trajectory step counter
        self.trajectory_step = 0
        
        logger.info(f"RunDataManager initialized: {self.run_dir}")
    
    def _create_directory_structure(self):
        """Create the standardized directory structure with 3 components:
        1. prompt_evolution/ - Data for prompt optimization (llm_traces, trajectories)
        2. end_state/ - End-state information (metadata, map_data, frame_cache, videos, logs, game_state)
        3. agent_scratch_space/ - Files written/generated/read by LLM tool calls
        """
        self.run_dir.mkdir(exist_ok=True)
        
        # Component 1: Prompt evolution data
        (self.run_dir / "prompt_evolution" / "llm_traces").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "prompt_evolution" / "trajectories").mkdir(parents=True, exist_ok=True)
        
        # Component 2: End-state information
        (self.run_dir / "end_state" / "game_state").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "end_state" / "map_data").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "end_state" / "frame_cache").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "end_state" / "videos").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "end_state" / "logs").mkdir(parents=True, exist_ok=True)
        
        # Component 3: Agent scratch space (for files written by LLM tool calls)
        (self.run_dir / "agent_scratch_space").mkdir(exist_ok=True)
        
        logger.debug(f"Created directory structure: {self.run_dir}")
    
    def get_prompt_evolution_dir(self) -> Path:
        """Get the prompt evolution directory"""
        return self.run_dir / "prompt_evolution"
    
    def get_end_state_dir(self) -> Path:
        """Get the end state directory"""
        return self.run_dir / "end_state"
    
    def get_scratch_space_dir(self) -> Path:
        """Get the agent scratch space directory"""
        return self.run_dir / "agent_scratch_space"
    
    def save_metadata(self, 
                     command_args: Dict[str, Any],
                     sys_argv: List[str],
                     additional_info: Optional[Dict[str, Any]] = None):
        """Save run metadata including command line information
        
        Args:
            command_args: Parsed command line arguments dictionary
            sys_argv: Original sys.argv for exact reproducibility
            additional_info: Optional additional metadata
        """
        metadata = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "command": " ".join(sys_argv),
            "command_args": command_args,
            "sys_argv": sys_argv,
            "python_version": sys.version,
            "working_directory": os.getcwd(),
        }
        
        # Try to get git commit hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                metadata["git_commit"] = result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get git commit: {e}")
        
        # Add any additional info
        if additional_info:
            metadata.update(additional_info)
        
        # Save metadata in end_state directory
        metadata_file = self.run_dir / "end_state" / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata: {metadata_file}")
    
    def log_trajectory(self,
                      step: int,
                      reasoning: str,
                      action: Dict[str, Any],
                      pre_state: Dict[str, Any],
                      post_state: Dict[str, Any],
                      outcome: Dict[str, Any],
                      llm_prompt: Optional[str] = None,
                      llm_trace_ref: Optional[str] = None):
        """Log a system-level trajectory (reasoning → action → outcome)
        
        Args:
            step: Step number
            reasoning: LLM reasoning/thought process
            action: Action taken (buttons, tool calls, etc.)
            pre_state: Game state before action (simple snapshot with location, coords, context, etc.)
            post_state: Game state after action (simple snapshot with location, coords, context, etc.)
            outcome: Result of the action
            llm_prompt: Full prompt that was sent to the LLM
            llm_trace_ref: Reference to raw LLM log entry
        """
        trajectory_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "action": action,
            "pre_state": pre_state,
            "post_state": post_state,
            "outcome": outcome,
        }
        
        if llm_prompt:
            trajectory_entry["llm_prompt"] = llm_prompt
        
        if llm_trace_ref:
            trajectory_entry["llm_trace_ref"] = llm_trace_ref
        
        # Append to trajectories file in prompt_evolution directory
        trajectories_file = self.run_dir / "prompt_evolution" / "trajectories" / "trajectories.jsonl"
        
        # DEBUG: Log trajectory file path and directory
        logger.debug(f"🔍 [DEBUG] Writing trajectory for step {step} to: {trajectories_file}")
        logger.debug(f"🔍 [DEBUG] Trajectory directory exists: {trajectories_file.parent.exists()}")
        logger.debug(f"🔍 [DEBUG] Run directory: {self.run_dir}")
        
        try:
            # Ensure directory exists
            trajectories_file.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"🔍 [DEBUG] Ensured trajectory directory exists: {trajectories_file.parent}")
            
            with open(trajectories_file, 'a', encoding='utf-8') as f:
                entry_json = json.dumps(trajectory_entry, ensure_ascii=False)
                f.write(entry_json + '\n')
                f.flush()  # Ensure data is written immediately
                logger.debug(f"🔍 [DEBUG] Wrote {len(entry_json)} bytes to trajectory file")
            
            self.trajectory_step += 1
            logger.info(f"🔍 [DEBUG] ✅ Successfully wrote trajectory for step {step} (total: {self.trajectory_step})")
        except Exception as e:
            logger.error(f"🔍 [DEBUG] ❌ Failed to write trajectory entry for step {step}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def create_state_snapshot(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple state snapshot
        
        Args:
            game_state: Full game state dictionary
        
        Returns:
            Simple state snapshot
        """
        snapshot = {}
        
        # Extract player info
        player = game_state.get("player", {})
        snapshot["location"] = player.get("location", "Unknown")
        
        # Extract coordinates
        position = player.get("position", {})
        if isinstance(position, dict):
            snapshot["player_coords"] = [position.get("x"), position.get("y")]
        else:
            snapshot["player_coords"] = None
        
        # Extract map info
        map_info = game_state.get("map", {})
        snapshot["map_id"] = map_info.get("id")
        
        # Extract game context
        game_info = game_state.get("game", {})
        snapshot["context"] = self._determine_context(game_info)
        snapshot["is_in_battle"] = game_info.get("is_in_battle", False)
        snapshot["dialog_active"] = bool(game_info.get("dialog_text"))
        
        return snapshot
    
    def _determine_context(self, game_info: Dict[str, Any]) -> str:
        """Determine the game context from game info"""
        if game_info.get("is_in_battle"):
            return "battle"
        elif game_info.get("dialog_text"):
            return "dialogue"
        elif game_info.get("menu_active"):
            return "menu"
        else:
            return "overworld"
    
    def copy_llm_traces(self, llm_log_file: str):
        """Copy LLM traces from llm_logs directory to run_data prompt_evolution
        
        Args:
            llm_log_file: Path to the LLM log file to copy
        """
        logger.info(f"🔍 [DEBUG] copy_llm_traces called with: {llm_log_file}")
        logger.info(f"🔍 [DEBUG] Run directory: {self.run_dir}")
        
        if not llm_log_file:
            logger.warning("🔍 [DEBUG] No LLM log file path provided")
            return
            
        if not os.path.exists(llm_log_file):
            logger.warning(f"🔍 [DEBUG] LLM log file not found: {llm_log_file}")
            logger.warning(f"🔍 [DEBUG] Current working directory: {os.getcwd()}")
            return
        
        dest_file = self.run_dir / "prompt_evolution" / "llm_traces" / "llm_log.jsonl"
        logger.info(f"🔍 [DEBUG] Destination file: {dest_file}")
        logger.info(f"🔍 [DEBUG] Destination directory exists: {dest_file.parent.exists()}")
        
        try:
            # Ensure directory exists
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"🔍 [DEBUG] Ensured destination directory exists")
            
            # Check source file size
            source_size = os.path.getsize(llm_log_file)
            logger.info(f"🔍 [DEBUG] Source file size: {source_size} bytes")
            
            shutil.copy2(llm_log_file, dest_file)
            
            # Verify copy
            dest_size = os.path.getsize(dest_file) if dest_file.exists() else 0
            logger.info(f"🔍 [DEBUG] ✅ Copied LLM traces: {llm_log_file} -> {dest_file} ({source_size} -> {dest_size} bytes)")
        except Exception as e:
            logger.error(f"🔍 [DEBUG] ❌ Failed to copy LLM traces: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def copy_objectives(self, objectives_file: Optional[str] = None):
        """Copy completed objectives to run_data agent_scratch_space
        
        Args:
            objectives_file: Path to completed_objectives.json. If None, looks in agent_scratch_space
                            (objectives should already be saved there during run).
        """
        if objectives_file is None:
            # Check if objectives already exist in agent_scratch_space
            scratch_space_file = self.run_dir / "agent_scratch_space" / "completed_objectives.json"
            if scratch_space_file.exists():
                logger.info("Objectives already in agent_scratch_space, no copy needed")
                return
            # DEPRECATED: No longer looks in .pokeagent_cache
            logger.warning("No objectives file found in agent_scratch_space")
            return
        
        if os.path.exists(objectives_file):
            dest_file = self.run_dir / "agent_scratch_space" / "completed_objectives.json"
            shutil.copy2(objectives_file, dest_file)
            logger.info(f"Copied objectives: {objectives_file} -> {dest_file}")
        else:
            logger.warning(f"Objectives file not found: {objectives_file}")
    
    def copy_game_state(self, 
                       checkpoint_state: Optional[str] = None,
                       milestones: Optional[str] = None,
                       maps: Optional[str] = None,
                       knowledge_base: Optional[str] = None):
        """Copy game state files to run_data end_state
        
        Args:
            checkpoint_state: Path to checkpoint.state file
            milestones: Path to milestones file
            maps: Path to maps file
            knowledge_base: Path to knowledge_base.json
        """
        from utils.run_data_manager import get_cache_path
        
        game_state_dir = self.run_dir / "end_state" / "game_state"
        
        # Use run-specific cache paths if not explicitly provided
        if checkpoint_state is None:
            checkpoint_state = str(get_cache_path("checkpoint.state"))
        if milestones is None:
            milestones = str(get_cache_path("milestones_progress.json"))
        if maps is None:
            maps = str(get_cache_path("checkpoint_maps.json"))
        if knowledge_base is None:
            knowledge_base = str(get_cache_path("knowledge_base.json"))
        
        files_to_copy = {
            "checkpoint.state": checkpoint_state,
            "milestones.json": milestones,
            "maps.json": maps,
            "knowledge_base.json": knowledge_base
        }
        
        for dest_name, src_path in files_to_copy.items():
            if src_path and os.path.exists(src_path):
                dest_file = game_state_dir / dest_name
                shutil.copy2(src_path, dest_file)
                logger.info(f"Copied {dest_name}")
            else:
                logger.debug(f"Skipping {dest_name} (not found)")
    
    def copy_map_data(self, map_stitcher_file: Optional[str] = None):
        """Copy map stitcher data to run_data end_state
        
        Args:
            map_stitcher_file: Path to map_stitcher_data.json
        """
        from utils.run_data_manager import get_cache_path
        if map_stitcher_file is None:
            map_stitcher_file = str(get_cache_path("map_stitcher_data.json"))
        
        if os.path.exists(map_stitcher_file):
            dest_file = self.run_dir / "end_state" / "map_data" / "map_stitcher_data.json"
            shutil.copy2(map_stitcher_file, dest_file)
            logger.info(f"Copied map data: {map_stitcher_file}")
    
    def copy_submission_log(self, submission_log: Optional[str] = None):
        """Copy submission.log to run_data end_state
        
        Args:
            submission_log: Path to submission.log. If None, looks in run-specific cache
        """
        from utils.run_data_manager import get_cache_path
        if submission_log is None:
            # Check common locations
            possible_paths = [
                str(get_cache_path("submission.log")),
                "submission.log",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    submission_log = path
                    break
        
        if submission_log and os.path.exists(submission_log):
            dest_file = self.run_dir / "end_state" / "logs" / "submission.log"
            shutil.copy2(submission_log, dest_file)
            logger.info(f"Copied submission log: {submission_log}")
    
    def copy_knowledge_base(self, knowledge_base_file: Optional[str] = None):
        """Copy knowledge_base.json to agent_scratch_space
        
        Args:
            knowledge_base_file: Path to knowledge_base.json. If None, looks in run-specific cache
        """
        from utils.run_data_manager import get_cache_path
        if knowledge_base_file is None:
            knowledge_base_file = str(get_cache_path("knowledge_base.json"))
        
        if os.path.exists(knowledge_base_file):
            dest_file = self.run_dir / "agent_scratch_space" / "knowledge_base.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(knowledge_base_file, dest_file)
            logger.info(f"Copied knowledge_base: {knowledge_base_file} -> {dest_file}")
        else:
            logger.debug(f"Knowledge base file not found: {knowledge_base_file}")
    
    def copy_frame_cache(self, frame_cache_file: Optional[str] = None):
        """Copy frame_cache.json to end_state/frame_cache
        
        Args:
            frame_cache_file: Path to frame_cache.json. If None, looks in run-specific cache
        """
        from utils.run_data_manager import get_cache_path
        if frame_cache_file is None:
            frame_cache_file = str(get_cache_path("frame_cache.json"))
        
        if os.path.exists(frame_cache_file):
            dest_file = self.run_dir / "end_state" / "frame_cache" / "frame_cache.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frame_cache_file, dest_file)
            logger.info(f"Copied frame_cache: {frame_cache_file} -> {dest_file}")
        else:
            logger.debug(f"Frame cache file not found: {frame_cache_file}")
    
    def copy_video_recording(self, record_enabled: bool = False):
        """Copy video recording to run_data end_state
        
        Simplified: If recording was enabled, find the most recent .mp4 file and copy it.
        
        Args:
            record_enabled: Whether --record flag was used (if True, search for video file)
        """
        if not record_enabled:
            logger.debug("Video recording was not enabled, skipping video copy")
            return
        
        import glob
        
        logger.info(f"🔍 [VIDEO] Searching for video files (record_enabled=True)")
        logger.info(f"🔍 [VIDEO] Current working directory: {os.getcwd()}")
        logger.info(f"🔍 [VIDEO] Run ID: {self.run_id}")
        
        # Find all .mp4 files matching the pattern in current directory
        # Try multiple search patterns (new format with run_id and old format)
        search_patterns = [
            f"{self.run_id}.mp4",  # New format
            f"./{self.run_id}.mp4",
            "pokegent_recording_*.mp4",  # Old format (fallback)
            "./pokegent_recording_*.mp4",
            os.path.join(os.getcwd(), "pokegent_recording_*.mp4"),
        ]
        
        video_files = []
        for pattern in search_patterns:
            found = glob.glob(pattern)
            if found:
                video_files.extend(found)
                logger.info(f"🔍 [VIDEO] Found {len(found)} files with pattern '{pattern}': {found}")
        
        # Remove duplicates while preserving order
        video_files = list(dict.fromkeys(video_files))
        
        logger.info(f"🔍 [VIDEO] Total unique video files found: {len(video_files)}")
        if video_files:
            logger.info(f"🔍 [VIDEO] Video files: {video_files}")
        
        if video_files:
            # Use the most recent video file (by modification time)
            video_file = max(video_files, key=os.path.getmtime)
            logger.info(f"🔍 [VIDEO] Using most recent video: {video_file}")
            logger.info(f"🔍 [VIDEO] Video file exists: {os.path.exists(video_file)}")
            logger.info(f"🔍 [VIDEO] Video file absolute path: {os.path.abspath(video_file)}")
            
            try:
                # Check source file size
                source_size = os.path.getsize(video_file)
                logger.info(f"🔍 [VIDEO] Source video size: {source_size} bytes")
                
                if source_size == 0:
                    logger.warning(f"⚠️ [VIDEO] Source video file is empty (0 bytes), skipping copy")
                    return
                
                dest_file = self.run_dir / "end_state" / "videos" / os.path.basename(video_file)
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"🔍 [VIDEO] Destination: {dest_file}")
                
                shutil.copy2(video_file, dest_file)
                
                # Verify copy
                if dest_file.exists():
                    dest_size = os.path.getsize(dest_file)
                    logger.info(f"✅ [VIDEO] Copied video: {video_file} -> {dest_file} ({source_size} -> {dest_size} bytes)")
                    if dest_size != source_size:
                        logger.warning(f"⚠️ [VIDEO] Size mismatch! Source: {source_size}, Dest: {dest_size}")
                else:
                    logger.error(f"❌ [VIDEO] Copy failed - destination file does not exist: {dest_file}")
            except Exception as e:
                logger.error(f"❌ [VIDEO] Error copying video: {e}", exc_info=True)
        else:
            logger.warning(f"⚠️ [VIDEO] Video recording was enabled but no pokegent_recording_*.mp4 file found")
            logger.warning(f"🔍 [VIDEO] Searched patterns: {search_patterns}")
            logger.warning(f"🔍 [VIDEO] Current directory contents (first 20): {os.listdir('.')[:20]}")
    
    def save_end_state_snapshot(self):
        """Save current end-state snapshot (can be called periodically)
        
        This saves game state, map data, and submission log without
        overwriting metadata or requiring full finalization.
        """
        try:
            self.copy_game_state()
            self.copy_map_data()
            self.copy_submission_log()
            logger.debug(f"Saved end-state snapshot for run: {self.run_id}")
        except Exception as e:
            logger.warning(f"Failed to save end-state snapshot: {e}")
    
    def finalize_run(self, 
                    end_time: Optional[datetime] = None,
                    final_metrics: Optional[Dict[str, Any]] = None):
        """Finalize the run by updating metadata with end time and metrics
        
        Args:
            end_time: End time of the run
            final_metrics: Final metrics (tokens, cost, steps, etc.)
        """
        metadata_file = self.run_dir / "end_state" / "metadata.json"
        
        if not metadata_file.exists():
            logger.warning("No metadata.json found to finalize")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if end_time is None:
            end_time = datetime.now()
        
        metadata["end_time"] = end_time.isoformat()
        
        # Calculate duration
        if "start_time" in metadata:
            start = datetime.fromisoformat(metadata["start_time"])
            duration = (end_time - start).total_seconds()
            metadata["duration_seconds"] = duration
        
        if final_metrics:
            metadata["final_metrics"] = final_metrics
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Finalized run: {self.run_id}")
    
    def get_run_directory(self) -> Path:
        """Get the run directory path"""
        return self.run_dir
    
    def __str__(self) -> str:
        return f"RunDataManager(run_id={self.run_id}, dir={self.run_dir})"


# Global instance
_run_data_manager: Optional[RunDataManager] = None


def get_run_data_manager() -> Optional[RunDataManager]:
    """Get the global run data manager instance"""
    return _run_data_manager


def initialize_run_data_manager(run_id: Optional[str] = None, run_name: Optional[str] = None,
                               first_objective_id: Optional[str] = None, 
                               first_objective_desc: Optional[str] = None) -> RunDataManager:
    """Initialize the global run data manager
    
    Args:
        run_id: Optional run identifier
        run_name: Optional name to append to run_id (deprecated - use objectives)
        first_objective_id: ID of the first objective (for consistent naming)
        first_objective_desc: Description of the first objective (for consistent naming)
    
    Returns:
        RunDataManager instance
    """
    global _run_data_manager
    _run_data_manager = RunDataManager(
        run_id=run_id, 
        run_name=run_name,
        first_objective_id=first_objective_id,
        first_objective_desc=first_objective_desc
    )
    return _run_data_manager


def get_cache_directory() -> Path:
    """Get the cache directory for the current run
    
    Returns:
        Path to .pokeagent_cache/{run_id}/ or .pokeagent_cache/ if no run_id
    """
    run_manager = get_run_data_manager()
    if run_manager and run_manager.run_id:
        cache_dir = Path(".pokeagent_cache") / run_manager.run_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    else:
        # Fallback: try to get run_id from environment
        run_id = os.environ.get("RUN_DATA_ID")
        if run_id:
            cache_dir = Path(".pokeagent_cache") / run_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
        # Final fallback: use base cache directory (for backward compatibility)
        cache_dir = Path(".pokeagent_cache")
        cache_dir.mkdir(exist_ok=True)
        return cache_dir


def get_cache_path(relative_path: str) -> Path:
    """Get a path within the run-specific cache directory
    
    Args:
        relative_path: Relative path within cache (e.g., "checkpoint.state")
    
    Returns:
        Full path to the file in the run-specific cache
    """
    cache_dir = get_cache_directory()
    return cache_dir / relative_path


def cleanup_old_cache_runs():
    """Clean up old run_ directories in .pokeagent_cache (deprecated structure)
    
    This preserves the files but marks them as deprecated.
    """
    cache_dir = Path(".pokeagent_cache")
    if not cache_dir.exists():
        return
    
    run_dirs = list(cache_dir.glob("run_*"))
    if not run_dirs:
        return
    
    deprecated_dir = cache_dir / "_deprecated_runs"
    deprecated_dir.mkdir(exist_ok=True)
    
    for run_dir in run_dirs:
        if run_dir.is_dir():
            dest = deprecated_dir / run_dir.name
            if not dest.exists():
                shutil.move(str(run_dir), str(dest))
                logger.info(f"Moved deprecated run: {run_dir} -> {dest}")


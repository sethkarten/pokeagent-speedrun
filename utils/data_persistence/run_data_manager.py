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

                # append run_name to run_id if provided
                if run_name:
                    run_id = f"{run_id}_{run_name}"
            # Run name only (e.g. CLI): datetime then name, same pattern as run_data and .pokeagent_cache
            elif run_name:
                import re
                sanitized_name = re.sub(r'[^\w\-_]', '_', run_name)
                run_id = f"{timestamp}_{sanitized_name}"
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
        
        # Component 1: Prompt evolution data (trajectories now in cache, not here)
        (self.run_dir / "prompt_evolution" / "llm_traces").mkdir(parents=True, exist_ok=True)
        
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
        """Save run metadata to cumulative metrics (metadata.json deprecated)
        
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
            "sys": {
                "platform": sys.platform,
                "python_version": sys.version,
            },
        }
        
        # Try to get git remote, branch, and commit to build a GitHub URL
        try:
            commit = None
            branch = None
            remote_url = None
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                remote_url = result.stdout.strip()

            if remote_url:
                if remote_url.startswith("git@github.com:"):
                    remote_url = remote_url.replace("git@github.com:", "https://github.com/")
                if remote_url.startswith("http://"):
                    remote_url = "https://" + remote_url[len("http://") :]
                if remote_url.endswith(".git"):
                    remote_url = remote_url[: -len(".git")]
            if remote_url and branch and commit:
                metadata["github_url"] = f"{remote_url}/tree/{branch}/{commit}"
        except Exception as e:
            logger.debug(f"Could not build github_url: {e}")
        
        # Add any additional info
        if additional_info:
            metadata.update(additional_info)

        # Map mode from additional_info if provided
        if "mode" not in metadata:
            if additional_info and additional_info.get("server_mode"):
                metadata["mode"] = "server"
            elif additional_info and additional_info.get("mode"):
                metadata["mode"] = additional_info.get("mode")

        # Save metadata into cumulative_metrics.json
        try:
            from utils.data_persistence.llm_logger import get_llm_logger

            llm_logger = get_llm_logger()
            if llm_logger:
                llm_logger.set_run_metadata(metadata)
        except Exception as e:
            logger.warning(f"Failed to write run metadata to cumulative metrics: {e}")
    
    def log_trajectory(self,
                      step: int,
                      reasoning: str,
                      action: Dict[str, Any],
                      pre_state: Dict[str, Any],
                      outcome: Optional[Dict[str, Any]] = None,
                      llm_prompt: Optional[str] = None,
                      llm_trace_ref: Optional[str] = None,
                      objective_context: Optional[str] = None,
                      **kwargs):
        """Log a system-level trajectory entry.

        Location and player_coords are promoted from pre_state to
        top-level fields.  ``objective_context`` captures the active
        objective IDs.

        ``post_state`` is deprecated and ignored (accepted via **kwargs
        for backwards compatibility with older callers).

        Trajectory JSONL schema (current):
            Required:
                - step
                - timestamp
                - reasoning
                - action
                - pre_state
                - outcome
                - location
                - player_coords
            Optional:
                - objective_context
                - llm_prompt
                - llm_trace_ref
            Legacy:
                - post_state may exist in older run files but is not written
        """
        trajectory_entry: Dict[str, Any] = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "action": action,
            "pre_state": pre_state,
            "outcome": outcome or {},
            "location": pre_state.get("location"),
            "player_coords": pre_state.get("player_coords"),
        }

        if objective_context:
            trajectory_entry["objective_context"] = objective_context

        if llm_prompt:
            trajectory_entry["llm_prompt"] = llm_prompt

        if llm_trace_ref:
            trajectory_entry["llm_trace_ref"] = llm_trace_ref
        
        # Write to cache as the primary location
        trajectories_file = get_cache_path("trajectory_history.jsonl")

        try:
            trajectories_file.parent.mkdir(parents=True, exist_ok=True)
            with open(trajectories_file, 'a', encoding='utf-8') as f:
                entry_json = json.dumps(trajectory_entry, ensure_ascii=False)
                f.write(entry_json + '\n')
                f.flush()

            self.trajectory_step += 1
            logger.debug(f"Wrote trajectory step {step} to {trajectories_file}")
        except Exception as e:
            logger.error(f"Failed to write trajectory entry for step {step}: {e}")
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

    def copy_objectives_state(self, objectives_state_file: Optional[str] = None):
        """Sync objectives.json from cache to agent_scratch_space.

        Called automatically by DirectObjectiveManager.auto_save and during
        copy_game_state flow.
        """
        if objectives_state_file is None:
            objectives_state_file = str(get_cache_path("objectives.json"))

        if os.path.exists(objectives_state_file):
            dest_file = self.run_dir / "agent_scratch_space" / "objectives.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(objectives_state_file, dest_file)
            logger.debug(f"Synced objectives state: {objectives_state_file} -> {dest_file}")
        else:
            logger.debug(f"objectives.json not found at {objectives_state_file}, skipping sync")

    def copy_game_state(self, 
                       checkpoint_state: Optional[str] = None,
                       milestones: Optional[str] = None,
                       maps: Optional[str] = None,
                       memory: Optional[str] = None,
                       skills: Optional[str] = None,
                       subagents: Optional[str] = None):
        """Copy game state files to run_data end_state
        
        Args:
            checkpoint_state: Path to checkpoint.state file
            milestones: Path to milestones file
            maps: Path to maps file
            memory: Path to memory.json
            skills: Path to skills.json
            subagents: Path to subagents.json
        """
        from utils.data_persistence.run_data_manager import get_cache_path
        
        game_state_dir = self.run_dir / "end_state" / "game_state"
        
        if checkpoint_state is None:
            checkpoint_state = str(get_cache_path("checkpoint.state"))
        if milestones is None:
            milestones = str(get_cache_path("milestones_progress.json"))
        if maps is None:
            maps = str(get_cache_path("checkpoint_maps.json"))
        if memory is None:
            memory_path = get_cache_path("memory.json")
            if not os.path.exists(str(memory_path)):
                memory_path = get_cache_path("knowledge_base.json")
            memory = str(memory_path)
        if skills is None:
            skills = str(get_cache_path("skills.json"))
        if subagents is None:
            subagents = str(get_cache_path("subagents.json"))
        
        files_to_copy = {
            "checkpoint.state": checkpoint_state,
            "milestones.json": milestones,
            "maps.json": maps,
            "memory.json": memory,
            "skills.json": skills,
            "subagents.json": subagents,
        }
        
        for dest_name, src_path in files_to_copy.items():
            if src_path and os.path.exists(src_path):
                dest_file = game_state_dir / dest_name
                shutil.copy2(src_path, dest_file)
                logger.info(f"Copied {dest_name}")
            else:
                logger.debug(f"Skipping {dest_name} (not found)")

        self.copy_objectives_state()
        self._export_bootstrap_artifacts()

    def _export_bootstrap_artifacts(self) -> None:
        """Create ``bootstrap/`` directories with final store state + evolved prompt.

        Writes to both ``end_state/game_state/bootstrap/`` (archival) and
        ``.pokeagent_cache/{run_id}/bootstrap/`` (easy re-use) so that future
        runs can ``--bootstrap-from`` either location.
        """
        from utils.data_persistence.run_data_manager import get_cache_path, get_cache_directory

        destinations = [
            self.run_dir / "end_state" / "game_state" / "bootstrap",
            Path(str(get_cache_directory())) / "bootstrap",
        ]

        store_files = ["memory.json", "skills.json", "subagents.json"]

        for dest_dir in destinations:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                for store_file in store_files:
                    src = Path(str(get_cache_path(store_file)))
                    if src.exists():
                        shutil.copy2(str(src), str(dest_dir / store_file))

                meta_dir = self.run_dir / "prompt_evolution" / "meta_prompts"
                if meta_dir.exists():
                    prompt_files = sorted(meta_dir.glob("steps_*.md"), key=lambda p: p.stat().st_mtime)
                    if prompt_files:
                        shutil.copy2(str(prompt_files[-1]), str(dest_dir / "EVOLVED_ORCHESTRATOR_POLICY.md"))

                logger.info(f"Exported bootstrap artifacts to {dest_dir}")
            except Exception as exc:
                logger.warning(f"Failed to export bootstrap artifacts to {dest_dir}: {exc}")

    def copy_map_data(self, map_stitcher_file: Optional[str] = None):
        """Copy map stitcher data to run_data end_state
        
        Args:
            map_stitcher_file: Path to map_stitcher_data.json
        """
        from utils.data_persistence.run_data_manager import get_cache_path
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
        from utils.data_persistence.run_data_manager import get_cache_path
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
    
    def copy_memory(self, memory_file: Optional[str] = None):
        """Copy memory.json to agent_scratch_space
        
        Args:
            memory_file: Path to memory.json. If None, looks in run-specific cache
        """
        from utils.data_persistence.run_data_manager import get_cache_path
        if memory_file is None:
            memory_path = get_cache_path("memory.json")
            if not os.path.exists(str(memory_path)):
                memory_path = get_cache_path("knowledge_base.json")
            memory_file = str(memory_path)
        
        if os.path.exists(memory_file):
            dest_file = self.run_dir / "agent_scratch_space" / "memory.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(memory_file, dest_file)
            logger.info(f"Copied memory: {memory_file} -> {dest_file}")
        else:
            logger.debug(f"Memory file not found: {memory_file}")

    def copy_skills(self, skills_file: Optional[str] = None):
        """Copy skills.json to agent_scratch_space.

        Args:
            skills_file: Path to skills.json. If None, looks in run-specific cache.
        """
        from utils.data_persistence.run_data_manager import get_cache_path
        if skills_file is None:
            skills_file = str(get_cache_path("skills.json"))

        if os.path.exists(skills_file):
            dest_file = self.run_dir / "agent_scratch_space" / "skills.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skills_file, dest_file)
            logger.info(f"Copied skills: {skills_file} -> {dest_file}")
        else:
            logger.debug(f"Skills file not found: {skills_file}")

    def copy_subagents(self, subagents_file: Optional[str] = None):
        """Copy subagents.json to agent_scratch_space.

        Args:
            subagents_file: Path to subagents.json. If None, looks in run-specific cache.
        """
        from utils.data_persistence.run_data_manager import get_cache_path
        if subagents_file is None:
            subagents_file = str(get_cache_path("subagents.json"))

        if os.path.exists(subagents_file):
            dest_file = self.run_dir / "agent_scratch_space" / "subagents.json"
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(subagents_file, dest_file)
            logger.info(f"Copied subagents: {subagents_file} -> {dest_file}")
        else:
            logger.debug(f"Subagents file not found: {subagents_file}")

    def copy_knowledge_base(self, knowledge_base_file: Optional[str] = None):
        """Deprecated: use copy_memory() instead."""
        import warnings
        warnings.warn("copy_knowledge_base() is deprecated, use copy_memory()", DeprecationWarning, stacklevel=2)
        self.copy_memory(memory_file=knowledge_base_file)
    
    def sync_trajectories_to_run_data(self) -> None:
        """Copy trajectory_history.jsonl from cache to run_data/{run_id}/."""
        src = get_cache_path("trajectory_history.jsonl")
        if not src.exists():
            logger.debug("No trajectory_history.jsonl in cache, skipping sync")
            return
        dest = self.run_dir / "trajectory_history.jsonl"
        try:
            shutil.copy2(str(src), str(dest))
            logger.info(f"Synced trajectories: {src} -> {dest}")
        except Exception as e:
            logger.warning(f"Failed to sync trajectories to run_data: {e}")

    def copy_frame_cache(self, frame_cache_file: Optional[str] = None):
        """Copy frame_cache.json to end_state/frame_cache
        
        Args:
            frame_cache_file: Path to frame_cache.json. If None, looks in run-specific cache
        """
        from utils.data_persistence.run_data_manager import get_cache_path
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
        """Video is saved directly to run_data/end_state/videos/ when recording is enabled.
        No copy needed - this method verifies the file exists (kept for API compatibility).

        Args:
            record_enabled: Whether --record flag was used.
        """
        if not record_enabled:
            logger.debug("Video recording was not enabled, skipping")
            return

        expected = self.run_dir / "end_state" / "videos" / f"{self.run_id}.mp4"
        if expected.exists():
            logger.info(f"Video recording saved directly: {expected}")
        else:
            logger.warning(f"Video recording was enabled but file not found: {expected}")
    
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
        """Finalize the run (metadata.json deprecated)."""
        if end_time is None:
            end_time = datetime.now()
        try:
            from utils.data_persistence.llm_logger import get_llm_logger

            llm_logger = get_llm_logger()
            if llm_logger:
                llm_logger.set_run_metadata({"end_time": end_time.isoformat()})
        except Exception as e:
            logger.warning(f"Failed to write end_time to cumulative metrics metadata: {e}")
        logger.info("Run finalization complete (metadata.json deprecated)")
    
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


def get_cli_workspace_dir() -> Path:
    """Get the CLI agent workspace directory (inside cache for backup/restore).
    For containerized runs, /workspace is mounted from this path."""
    path = get_cache_path("workspace")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_llm_path() -> Path:
    """Get path to checkpoint_llm.txt (cache folder first, then legacy location).
    
    Returns:
        Path to checkpoint_llm.txt
    """
    checkpoint_file = get_cache_path("checkpoint_llm.txt")
    if not checkpoint_file.exists() and Path("checkpoint_llm.txt").exists():
        return Path("checkpoint_llm.txt")
    return checkpoint_file


def cleanup_old_cache_runs() -> None:
    """Deprecated. No-op: CLI and other runs now use datetime_runname (no run_* prefix).
    Kept for compatibility with server shutdown path; does nothing.
    """
    pass


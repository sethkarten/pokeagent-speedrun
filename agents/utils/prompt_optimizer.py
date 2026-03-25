#!/usr/bin/env python3
"""
Prompt Optimizer Module
Implements simplified GEPA-style prompt evolution for reset-free Pokemon Emerald agent.
Optimizes base_prompt.md based on trajectory analysis.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from agents.prompts.paths import POKEAGENT_BASE_PROMPT_PATH, POKEAGENT_SYSTEM_PROMPT_PATH, resolve_repo_path

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimizes agent base prompt based on trajectory analysis."""
    
    def __init__(self, vlm, run_data_manager, base_prompt_path: str = POKEAGENT_BASE_PROMPT_PATH, system_prompt_path: str = POKEAGENT_SYSTEM_PROMPT_PATH):
        """
        Initialize the prompt optimizer.
        
        Args:
            vlm: VLM instance for calling LLM (used to get backend/model info)
            run_data_manager: RunDataManager for accessing trajectories
            base_prompt_path: Path to base prompt file
            system_prompt_path: Path to system prompt file (contains tool definitions)
        """
        # Load system prompt so optimizer knows what tools the agent has access to
        # We'll include this in the optimization prompt (not as system instruction)
        system_prompt_file = resolve_repo_path(system_prompt_path)
        self.system_prompt_content = None
        if system_prompt_file.exists():
            with open(system_prompt_file, 'r') as f:
                self.system_prompt_content = f.read()
            logger.info(f"📋 Loaded system prompt for optimizer: {system_prompt_path} ({len(self.system_prompt_content)} chars)")
        else:
            logger.warning(f"System prompt not found at {system_prompt_path}, optimizer will not know available tools")
        
        # Create a separate VLM instance WITHOUT tools for text-only optimization calls
        # This ensures get_text_query returns a string, not a GenerateContentResponse object
        from utils.agent_infrastructure.vlm_backends import VLM
        self.vlm = VLM(
            backend=vlm.backend_type,
            model_name=vlm.model_name,
            tools=None,  # No tools - text-only mode
            system_instruction=None  # No system instruction - we'll include system prompt in the optimization prompt instead
        )
        self.run_manager = run_data_manager
        _base = Path(base_prompt_path)
        self.base_prompt_path = _base if _base.is_absolute() else resolve_repo_path(base_prompt_path)

        # Load initial base prompt
        if self.base_prompt_path.exists():
            with open(self.base_prompt_path, 'r') as f:
                self.current_base_prompt = f.read()
        else:
            logger.warning(f"Base prompt not found at {base_prompt_path}, using default")
            self.current_base_prompt = self._get_default_prompt()
        
        # Track when optimizations occur
        self.optimization_history = []
        
        logger.info(f"PromptOptimizer initialized with base prompt: {base_prompt_path}")
    
    def _get_default_prompt(self) -> str:
        """Returns a minimal default prompt if base file doesn't exist."""
        return """# Strategic Guidance

## Decision-Making Process
1. Analyze the current situation
2. Plan your next action
3. Execute with appropriate tools

## Core Principles
- Think step-by-step
- Store important information in memory
- Navigate efficiently
- Complete objectives when done
"""
    
    def should_optimize(self, current_step: int, optimization_frequency: int = 10) -> bool:
        """
        Check if optimization should run at this step.
        
        Args:
            current_step: Current agent step number
            optimization_frequency: How often to run optimization
        
        Returns:
            True if optimization should run
        """
        return current_step > 0 and current_step % optimization_frequency == 0
    
    def get_recent_trajectories(self, num_steps: int = 10) -> List[Dict[str, Any]]:
        """
        Load recent trajectory data from run_data.
        
        Args:
            num_steps: Number of recent steps to retrieve
        
        Returns:
            List of trajectory dictionaries
        """
        from utils.data_persistence.run_data_manager import get_cache_path
        trajectory_file = get_cache_path("trajectory_history.jsonl")

        # Fallback to legacy path for older runs
        if not trajectory_file.exists():
            trajectory_file = self.run_manager.run_dir / "prompt_evolution" / "trajectories" / "trajectories.jsonl"

        if not trajectory_file.exists():
            logger.warning("No trajectory file found")
            return []
        
        trajectories = []
        with open(trajectory_file, 'r') as f:
            for line in f:
                if line.strip():
                    trajectories.append(json.loads(line))
        
        # Return last N trajectories
        return trajectories[-num_steps:] if len(trajectories) >= num_steps else trajectories
    
    def optimize_prompt(self, current_step: int, num_trajectory_steps: int = 10) -> str:
        """
        Generate optimized base prompt based on recent trajectories.
        
        Args:
            current_step: Current agent step number
            num_trajectory_steps: Number of recent steps to analyze
        
        Returns:
            Optimized base prompt string
        """
        logger.info(f"🔄 Running prompt optimization at step {current_step}")
        
        # Get recent trajectories
        recent_trajectories = self.get_recent_trajectories(num_trajectory_steps)
        
        if not recent_trajectories:
            logger.warning("No trajectories available for optimization, keeping current prompt")
            return self.current_base_prompt
        
        # Format trajectories for LLM analysis
        trajectory_summary = self._format_trajectories_for_analysis(recent_trajectories)
        logger.info(
            "📋 Trajectory summary for optimization (steps %d–%d, %d chars):\n%s",
            current_step - num_trajectory_steps + 1,
            current_step,
            len(trajectory_summary),
            trajectory_summary,
        )

        # Build optimization prompt with system prompt context
        system_prompt_section = ""
        if self.system_prompt_content:
            system_prompt_section = f"""
## Main Agent's System Prompt (FIXED - Cannot Be Changed):
The main agent receives this as its **system** message every step. You must **not** duplicate it into the base prompt. It only lists **MCP tools** and **hard constraints** (e.g. terminal tool rule, valid buttons, path variance rules). **Strategy, goals, and playstyle** belong in the base prompt you are editing. Use the text below so you know what tools and non-negotiable rules the agent has when improving strategic guidance.

{self.system_prompt_content}

---
"""
        
        # Create optimization prompt
        optimization_prompt = f"""You are a prompt optimization expert. Your task is to improve an AI agent's strategic guidance prompt based on its recent performance in playing Pokemon Emerald.
{system_prompt_section}## Current Base Prompt (Strategic Guidance):
This is the optimizable strategic guidance that gets combined with runtime context (action history, objectives, game state) and sent to the main agent. You can modify this to improve the agent's decision-making.

{self.current_base_prompt}

## Recent Agent Trajectories (Last {len(recent_trajectories)} steps):
{trajectory_summary}

## Your Task:
Analyze the agent's recent performance and create an IMPROVED base prompt that:

1. **Addresses observed failures** - If the agent made mistakes, add specific guidance to prevent them
2. **Reinforces successful patterns** - If certain strategies worked well, emphasize them
3. **Adds learned lessons** - Include insights derived from trajectory analysis

## Analysis Guidelines:

**Look for these patterns:**
- Repeated failures (stuck in loops, wrong tool usage, blocked navigation)
- Successful strategies (good memory usage, efficient pathfinding, smart battle decisions)
- Progress toward objectives (completing tasks, leveling up, advancing story)
- Tool usage patterns (are they using the tools at their disposal effectively?)
- Memory management (are they storing and retrieving information in memory appropriately?)
- Not adapting when stuck → emphasize flexibility and trying new approaches

## Output Format:
Provide the complete improved base prompt as markdown. Make targeted improvements, keeping the elements that are working while adding additional guidance if necessary.

IMPROVED BASE PROMPT:
"""
        
        try:
            # Call LLM to generate optimized prompt
            logger.info("📞 Calling VLM for prompt optimization...")
            response = self.vlm.get_text_query(optimization_prompt, "PromptOptimizer")
            
            optimized_prompt = response.strip()
            
            # Clean up if LLM included extra wrapper text
            if optimized_prompt.startswith("```markdown"):
                optimized_prompt = optimized_prompt[11:]
            if optimized_prompt.startswith("```"):
                optimized_prompt = optimized_prompt[3:]
            if optimized_prompt.endswith("```"):
                optimized_prompt = optimized_prompt[:-3]
            optimized_prompt = optimized_prompt.strip()
            
            # Save the optimized prompt
            self._save_optimized_prompt(optimized_prompt, current_step, current_step - num_trajectory_steps + 1)
            
            # Update current prompt
            self.current_base_prompt = optimized_prompt
            
            # Track optimization
            self.optimization_history.append({
                "step": current_step,
                "timestamp": datetime.now().isoformat(),
                "trajectories_analyzed": len(recent_trajectories)
            })
            
            logger.info(f"✅ Prompt optimization complete at step {current_step}")
            return optimized_prompt
            
        except Exception as e:
            logger.error(f"❌ Prompt optimization failed: {e}", exc_info=True)
            return self.current_base_prompt
    
    def _format_trajectories_for_analysis(self, trajectories: List[Dict[str, Any]]) -> str:
        """Format trajectories into readable text for LLM analysis."""
        formatted = []
        
        for i, traj in enumerate(trajectories, 1):
            step_text = f"\n### Step {traj.get('step', i)}\n"
            step_text += f"**Reasoning:** {traj.get('reasoning', 'N/A')}\n"
            
            # Format action (can be dict or string)
            action = traj.get('action', 'N/A')
            if isinstance(action, dict):
                # Extract tool call names from action dict
                tool_calls = action.get('tool_calls', [])
                if tool_calls:
                    action_names = [tc.get('name', 'unknown') for tc in tool_calls]
                    action_str = ', '.join(action_names)
                    step_text += f"**Action:** {action_str}\n"
                    # Include tool args and results for non-trivial calls
                    for tc in tool_calls:
                        tc_name = tc.get('name', '')
                        tc_args = tc.get('args', {})
                        tc_result = tc.get('result', '')
                        if tc_name in ('run_skill', 'execute_custom_subagent', 'process_subagent', 'process_skill', 'process_memory'):
                            args_str = json.dumps(tc_args, default=str)[:300]
                            step_text += f"  - `{tc_name}` args: {args_str}\n"
                            if tc_result:
                                result_str = str(tc_result)[:300]
                                step_text += f"  - result: {result_str}\n"
                else:
                    step_text += f"**Action:** {action.get('type', 'N/A')}\n"
            else:
                step_text += f"**Action:** {action}\n"
            
            # Include relevant state info
            pre_state = traj.get('pre_state', {})
            post_state = traj.get('post_state', {})
            
            if pre_state:
                step_text += f"**Pre-State:** Location: {pre_state.get('location', 'Unknown')}, "
                step_text += f"Coords: {pre_state.get('player_coords', 'Unknown')}, "
                step_text += f"Context: {pre_state.get('context', 'Unknown')}\n"
            
            if post_state:
                step_text += f"**Post-State:** Location: {post_state.get('location', 'Unknown')}, "
                step_text += f"Coords: {post_state.get('player_coords', 'Unknown')}\n"
            
            # Check for issues (movement attempted but coordinates unchanged)
            if pre_state.get('player_coords') == post_state.get('player_coords'):
                # Extract action names for checking
                action_names = []
                if isinstance(action, dict):
                    tool_calls = action.get('tool_calls', [])
                    action_names = [tc.get('name', '').lower() for tc in tool_calls]
                elif isinstance(action, str):
                    action_names = [action.lower()]
                
                # Check if movement was attempted
                movement_attempted = (
                    'navigate_to' in action_names or
                    any(btn in action_names for btn in ['press_buttons']) or
                    (isinstance(action, str) and any(btn in action.upper() for btn in ['UP', 'DOWN', 'LEFT', 'RIGHT']))
                )
                
                if movement_attempted:
                    step_text += "⚠️ **Issue:** Movement attempted but coordinates unchanged (possibly blocked)\n"
            
            # Check for repeated locations
            if i > 1 and pre_state.get('player_coords') == trajectories[i-2].get('pre_state', {}).get('player_coords'):
                step_text += "⚠️ **Pattern:** Same location as 2 steps ago (possible loop)\n"
            
            formatted.append(step_text)
        
        return "\n".join(formatted)
    
    def _save_optimized_prompt(self, prompt: str, end_step: int, start_step: int):
        """Save optimized prompt to run_data directory."""
        # Create meta-prompts directory
        meta_prompts_dir = self.run_manager.run_dir / "prompt_evolution" / "meta_prompts"
        meta_prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with step range in filename
        filename = f"steps_{start_step}_to_{end_step}.md"
        filepath = meta_prompts_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(prompt)
        
        logger.info(f"💾 Saved optimized prompt: {filepath}")
        
        # Also save metadata
        metadata = {
            "start_step": start_step,
            "end_step": end_step,
            "timestamp": datetime.now().isoformat(),
            "prompt_length": len(prompt),
            "optimization_count": len(self.optimization_history) + 1
        }
        
        metadata_file = meta_prompts_dir / f"steps_{start_step}_to_{end_step}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_current_prompt(self) -> str:
        """Get the current active base prompt."""
        return self.current_base_prompt


def create_prompt_optimizer(vlm, run_data_manager, base_prompt_path: str = POKEAGENT_BASE_PROMPT_PATH, system_prompt_path: str = POKEAGENT_SYSTEM_PROMPT_PATH) -> PromptOptimizer:
    """Factory function to create a PromptOptimizer instance."""
    return PromptOptimizer(vlm, run_data_manager, base_prompt_path, system_prompt_path)


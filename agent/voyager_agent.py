"""
Voyager-Style Agent for Pokemon Emerald
========================================

Implementation based on Voyager (Minecraft) and MineDojo's curriculum learning approach.
Reference: https://arxiv.org/abs/2305.16291 (Voyager: An Open-Ended Embodied Agent with LLMs)

Core architectural principles:
- Skill Library: Agent creates and stores reusable skills as executable code/tools
- Curriculum Learning: Self-proposes increasingly complex tasks
- Iterative Skill Refinement: Skills improve through feedback and self-critique
- Tool Composition: Combine primitive skills into complex behaviors
- Persistent Memory: Skills survive across sessions and checkpoints

Skill Types:
1. Primitive Actions: Basic button sequences (e.g., "walk_north_5_steps")
2. Navigation Skills: Pathfinding routines (e.g., "go_to_pokemon_center")
3. Interaction Skills: NPC/object interactions (e.g., "heal_at_pokecenter")
4. Battle Skills: Combat strategies (e.g., "defeat_trainer_with_type_advantage")
5. Composite Skills: Multi-step procedures combining other skills
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import traceback

from utils.vlm import VLM
from utils.llm_logger import LLMLogger
from utils.state_formatter import format_state_for_llm
from utils.agent_helpers import update_server_metrics

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Represents a learned skill that can be executed as a tool."""
    name: str
    description: str
    skill_type: str  # "primitive", "navigation", "interaction", "battle", "composite"
    code: str  # Python code or action sequence
    parameters: List[str]  # Required parameters
    success_rate: float = 0.0  # Track skill effectiveness
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    version: int = 1  # Allow skill updates
    dependencies: List[str] = field(default_factory=list)  # Other skills this depends on

    def to_dict(self) -> Dict[str, Any]:
        """Serialize skill for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type,
            "code": self.code,
            "parameters": self.parameters,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "version": self.version,
            "dependencies": self.dependencies
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Skill':
        """Deserialize skill from storage."""
        return cls(
            name=data["name"],
            description=data["description"],
            skill_type=data["skill_type"],
            code=data["code"],
            parameters=data["parameters"],
            success_rate=data.get("success_rate", 0.0),
            usage_count=data.get("usage_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            version=data.get("version", 1),
            dependencies=data.get("dependencies", [])
        )


@dataclass
class Task:
    """Represents a self-proposed task in the curriculum."""
    description: str
    difficulty: int  # 1-10 scale
    required_skills: List[str]  # Skills needed to complete
    success: Optional[bool] = None
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)


class SkillLibrary:
    """Manages the collection of learned skills."""

    def __init__(self, save_dir: str = ".pokeagent_cache/skills"):
        self.save_dir = save_dir
        self.skills: Dict[str, Skill] = {}
        os.makedirs(save_dir, exist_ok=True)
        self._load_skills()
        self._initialize_primitive_skills()

    def _initialize_primitive_skills(self):
        """Initialize basic primitive skills that all agents start with."""
        primitives = [
            Skill(
                name="press_button",
                description="Press a single button",
                skill_type="primitive",
                code="return [button]",
                parameters=["button"],
                success_rate=1.0,
                version=1
            ),
            Skill(
                name="walk_direction",
                description="Walk in a direction for N steps",
                skill_type="primitive",
                code="return [direction] * steps",
                parameters=["direction", "steps"],
                success_rate=0.9,
                version=1
            ),
            Skill(
                name="interact",
                description="Interact with object/NPC in front of player",
                skill_type="interaction",
                code="return ['A']",
                parameters=[],
                success_rate=0.95,
                version=1
            ),
            Skill(
                name="open_menu",
                description="Open the start menu",
                skill_type="primitive",
                code="return ['START']",
                parameters=[],
                success_rate=1.0,
                version=1
            ),
            Skill(
                name="close_menu",
                description="Close current menu",
                skill_type="primitive",
                code="return ['B']",
                parameters=[],
                success_rate=1.0,
                version=1
            )
        ]

        for skill in primitives:
            if skill.name not in self.skills:
                self.add_skill(skill)
                logger.info(f"Initialized primitive skill: {skill.name}")

    def add_skill(self, skill: Skill) -> bool:
        """Add or update a skill in the library."""
        if skill.name in self.skills:
            # Update existing skill version
            old_skill = self.skills[skill.name]
            skill.version = old_skill.version + 1
            skill.usage_count = old_skill.usage_count
            logger.info(f"Updated skill '{skill.name}' to version {skill.version}")

        self.skills[skill.name] = skill
        self._save_skill(skill)
        return True

    def get_skill(self, name: str) -> Optional[Skill]:
        """Retrieve a skill by name."""
        return self.skills.get(name)

    def list_skills(self, skill_type: Optional[str] = None) -> List[Skill]:
        """List all skills, optionally filtered by type."""
        if skill_type:
            return [s for s in self.skills.values() if s.skill_type == skill_type]
        return list(self.skills.values())

    def get_skill_summary(self) -> str:
        """Get formatted summary of all available skills."""
        if not self.skills:
            return "No skills learned yet."

        summary_lines = ["Available Skills:"]
        for skill in sorted(self.skills.values(), key=lambda s: s.usage_count, reverse=True):
            summary_lines.append(
                f"  - {skill.name} ({skill.skill_type}): {skill.description} "
                f"[Used: {skill.usage_count}x, Success: {skill.success_rate:.1%}]"
            )
        return "\n".join(summary_lines)

    def _save_skill(self, skill: Skill):
        """Save skill to disk."""
        filepath = os.path.join(self.save_dir, f"{skill.name}.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(skill.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save skill {skill.name}: {e}")

    def _load_skills(self):
        """Load all skills from disk."""
        if not os.path.exists(self.save_dir):
            return

        for filename in os.listdir(self.save_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.save_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        skill = Skill.from_dict(data)
                        self.skills[skill.name] = skill
                        logger.info(f"Loaded skill: {skill.name} (v{skill.version})")
                except Exception as e:
                    logger.error(f"Failed to load skill from {filename}: {e}")


class VoyagerAgent:
    """
    Voyager-style agent with skill learning and curriculum.

    Key features:
    - Learns and stores reusable skills as executable code
    - Self-proposes tasks in a curriculum
    - Iteratively refines skills through feedback
    - Composes complex behaviors from simpler skills
    - Persistent skill library across sessions
    """

    def __init__(
        self,
        vlm: Optional[VLM] = None,
        skill_library: Optional[SkillLibrary] = None,
        enable_skill_creation: bool = True,
        enable_curriculum: bool = True,
        skill_proposal_interval: int = 25,
        verbose: bool = True
    ):
        """
        Initialize the VoyagerAgent.

        Args:
            vlm: Vision-language model client
            skill_library: Skill library (creates new if None)
            enable_skill_creation: Allow agent to create new skills
            enable_curriculum: Enable self-proposed curriculum
            skill_proposal_interval: Steps between skill proposals
            verbose: Detailed logging
        """
        self.vlm = vlm or VLM()
        self.skill_library = skill_library or SkillLibrary()
        self.enable_skill_creation = enable_skill_creation
        self.enable_curriculum = enable_curriculum
        self.skill_proposal_interval = skill_proposal_interval
        self.verbose = verbose

        # Core components
        self.step_count = 0
        self.llm_logger = LLMLogger()

        # Curriculum and task management
        self.current_task: Optional[Task] = None
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []

        # Action queue for multi-step skills
        self.action_queue: deque = deque()

        # Performance tracking
        self.recent_actions: deque = deque(maxlen=20)
        self.last_skill_execution: Optional[Dict[str, Any]] = None

    def step(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step with skill-based reasoning.

        Args:
            game_state: Current game state (includes 'frame')

        Returns:
            Dict with 'action' and 'reasoning'
        """
        self.step_count += 1
        frame = game_state.get('frame')

        # If we have queued actions from a skill, execute them
        if self.action_queue:
            action = self.action_queue.popleft()
            self.recent_actions.append(action)
            return {
                "action": action,
                "reasoning": f"Executing queued skill action ({len(self.action_queue)} remaining)"
            }

        # Periodically propose new skills
        if self.enable_skill_creation and self.step_count % self.skill_proposal_interval == 0:
            self._propose_new_skill(game_state, frame)

        # Periodically update curriculum tasks
        if self.enable_curriculum and self.step_count % 50 == 0:
            self._update_curriculum(game_state)

        # Decide next action using skills
        action, reasoning = self._decide_with_skills(game_state, frame)

        # Update server with agent step and metrics
        update_server_metrics()

        self.recent_actions.append(action)
        return {"action": action, "reasoning": reasoning}

    def _decide_with_skills(self, game_state: Dict[str, Any], frame: Any) -> tuple[str, str]:
        """Make decision using available skills."""
        # Get formatted state
        formatted_state = format_state_for_llm(game_state)

        # Get skill summary
        skill_summary = self.skill_library.get_skill_summary()

        # Build decision prompt with skills as tools
        prompt = f"""You are a Pokemon Emerald agent with a growing skill library.

**Current Game State:**
{formatted_state[:1500]}

**Your Available Skills (Tools):**
{skill_summary}

**Current Task:**
{self.current_task.description if self.current_task else "Explore and progress through the game"}

**Recent Actions:**
{', '.join(list(self.recent_actions)[-5:])}

**Decision Process:**
1. Analyze the current situation and your task
2. Choose which skill(s) to use, OR create a new skill if needed
3. If using existing skill, specify: SKILL: skill_name(param1, param2, ...)
4. If creating new skill, specify: NEW_SKILL: name | description | skill_type | code
5. Otherwise, specify a single button: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT

**Examples:**
- "SKILL: walk_direction(UP, 5)" - Walk north 5 steps
- "SKILL: interact()" - Talk to NPC ahead
- "NEW_SKILL: heal_at_pokecenter | Go to Pokemon Center and heal | composite | ..."
- "A" - Press A button

What should you do next?"""

        # Get VLM response
        if frame is not None:
            response = self.vlm.get_query(frame, prompt, "voyager_decision")
        else:
            response = self.vlm.get_text_query(prompt, "voyager_decision")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="voyager_decision",
            prompt=prompt,
            response=response
        )

        # Parse response
        if "SKILL:" in response:
            return self._execute_skill_from_response(response, game_state)
        elif "NEW_SKILL:" in response:
            return self._create_skill_from_response(response, game_state)
        else:
            # Parse as single button action
            action = self._parse_button(response)
            return action, f"Direct action: {response[:100]}"

    def _execute_skill_from_response(self, response: str, game_state: Dict[str, Any]) -> tuple[str, str]:
        """Execute a skill mentioned in the LLM response."""
        try:
            # Extract skill call: "SKILL: skill_name(param1, param2)"
            skill_line = [line for line in response.split('\n') if 'SKILL:' in line][0]
            skill_call = skill_line.split('SKILL:')[1].strip()

            # Parse skill name and parameters
            if '(' in skill_call:
                skill_name = skill_call[:skill_call.index('(')].strip()
                params_str = skill_call[skill_call.index('(')+1:skill_call.index(')')].strip()
                params = [p.strip().strip('"\'') for p in params_str.split(',') if p.strip()]
            else:
                skill_name = skill_call.strip()
                params = []

            # Get skill from library
            skill = self.skill_library.get_skill(skill_name)
            if not skill:
                logger.warning(f"Skill '{skill_name}' not found in library")
                return "A", f"Skill not found: {skill_name}"

            # Execute skill
            actions = self._execute_skill(skill, params, game_state)

            if actions:
                # Queue actions
                self.action_queue.extend(actions)
                first_action = self.action_queue.popleft()

                # Update skill stats
                skill.usage_count += 1
                skill.last_used = datetime.now()
                self.skill_library.add_skill(skill)

                return first_action, f"Executing skill: {skill_name} ({len(actions)} actions)"
            else:
                return "A", f"Skill {skill_name} returned no actions"

        except Exception as e:
            logger.error(f"Error executing skill: {e}")
            traceback.print_exc()
            return "A", f"Skill execution error: {str(e)}"

    def _execute_skill(self, skill: Skill, params: List[str], game_state: Dict[str, Any]) -> List[str]:
        """Execute a skill and return action sequence."""
        try:
            # Build safe execution environment
            safe_globals = {
                # Parameters
                **{skill.parameters[i]: params[i] if i < len(params) else None
                   for i in range(len(skill.parameters))},
                # Safe built-ins
                "abs": abs, "min": min, "max": max, "len": len, "range": range,
                "int": int, "str": str, "float": float, "bool": bool,
                # Game state access
                "game_state": game_state,
                "player_x": game_state.get("player", {}).get("position", {}).get("x", 0),
                "player_y": game_state.get("player", {}).get("position", {}).get("y", 0),
                # Skill library access for composite skills
                "execute_skill": lambda name, *args: self._execute_skill(
                    self.skill_library.get_skill(name), list(args), game_state
                ) if self.skill_library.get_skill(name) else []
            }

            # Execute skill code
            exec_locals = {}
            exec(skill.code, safe_globals, exec_locals)

            # Get return value (actions)
            actions = exec_locals.get('result') or safe_globals.get('return') or []

            # Validate actions
            valid_buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
            validated_actions = [a for a in actions if a in valid_buttons]

            if self.verbose:
                logger.info(f"Skill '{skill.name}' produced {len(validated_actions)} actions")

            return validated_actions

        except Exception as e:
            logger.error(f"Skill execution error for '{skill.name}': {e}")
            traceback.print_exc()
            return []

    def _create_skill_from_response(self, response: str, game_state: Dict[str, Any]) -> tuple[str, str]:
        """Create a new skill from LLM response."""
        try:
            # Extract skill definition: "NEW_SKILL: name | description | type | code"
            skill_line = [line for line in response.split('\n') if 'NEW_SKILL:' in line][0]
            skill_def = skill_line.split('NEW_SKILL:')[1].strip()

            parts = [p.strip() for p in skill_def.split('|')]
            if len(parts) < 4:
                logger.warning(f"Invalid skill definition: {skill_def}")
                return "A", "Invalid skill definition"

            name, description, skill_type, code = parts[:4]

            # Ask LLM to generate proper skill code
            code = self._generate_skill_code(name, description, skill_type)

            # Parse parameters from code
            parameters = self._extract_parameters(code)

            # Create skill
            skill = Skill(
                name=name,
                description=description,
                skill_type=skill_type,
                code=code,
                parameters=parameters
            )

            # Add to library
            self.skill_library.add_skill(skill)

            if self.verbose:
                logger.info(f"Created new skill: {name} ({skill_type})")

            return "A", f"Created new skill: {name}"

        except Exception as e:
            logger.error(f"Error creating skill: {e}")
            traceback.print_exc()
            return "A", f"Skill creation error: {str(e)}"

    def _generate_skill_code(self, name: str, description: str, skill_type: str) -> str:
        """Use LLM to generate executable skill code."""
        prompt = f"""Generate Python code for a Pokemon Emerald agent skill.

Skill Name: {name}
Description: {description}
Type: {skill_type}

The code should:
1. Return a list of button actions: ["A", "UP", "RIGHT", etc.]
2. Use available parameters and game_state dictionary
3. Be safe and deterministic
4. Handle edge cases

Available in scope:
- Parameters (will be defined based on your code)
- game_state: Dict with player position, map, dialogue, etc.
- player_x, player_y: Current player coordinates
- execute_skill(name, *args): Call other skills

Example skill codes:

```python
# Simple movement skill
def walk_north_n_steps(steps):
    return ["UP"] * int(steps)
result = walk_north_n_steps(steps)
```

```python
# Composite skill using other skills
def go_to_pokecenter():
    actions = []
    # Walk to entrance
    actions.extend(execute_skill("walk_direction", "UP", 5))
    # Enter
    actions.append("A")
    return actions
result = go_to_pokecenter()
```

Generate the code for '{name}':"""

        code_response = self.vlm.get_text_query(prompt, "skill_code_gen")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="voyager_skill_gen",
            prompt=prompt,
            response=code_response
        )

        # Extract code block from response
        if "```python" in code_response:
            code = code_response.split("```python")[1].split("```")[0].strip()
        elif "```" in code_response:
            code = code_response.split("```")[1].split("```")[0].strip()
        else:
            code = code_response.strip()

        return code

    def _extract_parameters(self, code: str) -> List[str]:
        """Extract parameter names from skill code."""
        # Simple extraction: look for function definitions
        params = []
        for line in code.split('\n'):
            if 'def ' in line and '(' in line:
                # Extract parameters from function definition
                param_str = line[line.index('(')+1:line.index(')')].strip()
                if param_str:
                    params = [p.strip() for p in param_str.split(',')]
                break
        return params

    def _propose_new_skill(self, game_state: Dict[str, Any], frame: Any):
        """Propose a new skill based on recent experience."""
        if not self.enable_skill_creation:
            return

        prompt = f"""Based on your recent experience playing Pokemon Emerald, propose a new useful skill.

Current State:
{format_state_for_llm(game_state)[:800]}

Recent Actions:
{', '.join(list(self.recent_actions)[-10:])}

Existing Skills:
{self.skill_library.get_skill_summary()}

What new skill would be useful? Consider:
- Repeated action sequences you perform often
- Complex behaviors that could be automated
- Navigation or interaction patterns
- Battle strategies

Format: NEW_SKILL: name | description | type | brief_purpose

Example: NEW_SKILL: navigate_to_nearest_pokecenter | Find and enter the nearest Pokemon Center | navigation | heal team"""

        if frame is not None:
            response = self.vlm.get_query(frame, prompt, "skill_proposal")
        else:
            response = self.vlm.get_text_query(prompt, "skill_proposal")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="voyager_skill_proposal",
            prompt=prompt,
            response=response
        )

        # If LLM proposed a skill, create it
        if "NEW_SKILL:" in response:
            self._create_skill_from_response(response, game_state)

    def _update_curriculum(self, game_state: Dict[str, Any]):
        """Update the curriculum of self-proposed tasks."""
        if not self.enable_curriculum:
            return

        # Check if current task is complete
        if self.current_task:
            is_complete = self._check_task_completion(self.current_task, game_state)
            if is_complete:
                self.current_task.success = True
                self.completed_tasks.append(self.current_task)
                logger.info(f"Task completed: {self.current_task.description}")
                self.current_task = None

        # Propose new task if needed
        if not self.current_task:
            self.current_task = self._propose_next_task(game_state)

    def _check_task_completion(self, task: Task, game_state: Dict[str, Any]) -> bool:
        """Check if a task has been completed using LLM evaluation."""
        prompt = f"""Has this task been completed in Pokemon Emerald?

Task: {task.description}
Required Skills: {', '.join(task.required_skills)}

Current State:
{format_state_for_llm(game_state)[:600]}

Reply with YES or NO."""

        response = self.vlm.get_text_query(prompt, "task_check")

        return "YES" in response.upper()

    def _propose_next_task(self, game_state: Dict[str, Any]) -> Optional[Task]:
        """Propose the next task in the curriculum using LLM."""
        completed_desc = [t.description for t in self.completed_tasks[-3:]]

        prompt = f"""Propose the next task for your Pokemon Emerald curriculum.

Current State:
{format_state_for_llm(game_state)[:600]}

Recently Completed Tasks:
{chr(10).join('- ' + d for d in completed_desc) if completed_desc else 'None'}

Available Skills:
{self.skill_library.get_skill_summary()}

Propose a challenging but achievable task that builds on your skills.
Format: TASK: description | difficulty (1-10) | required_skills (comma-separated)

Example: TASK: Defeat the first gym leader | 7 | navigate_to_gym,battle_strategy,heal_team"""

        response = self.vlm.get_text_query(prompt, "task_proposal")

        # Log interaction
        self.llm_logger.log_interaction(
            interaction_type="voyager_task_proposal",
            prompt=prompt,
            response=response
        )

        # Parse task
        if "TASK:" in response:
            try:
                task_def = response.split("TASK:")[1].strip()
                parts = [p.strip() for p in task_def.split('|')]
                if len(parts) >= 3:
                    description = parts[0]
                    difficulty = int(parts[1]) if parts[1].isdigit() else 5
                    required_skills = [s.strip() for s in parts[2].split(',')]

                    task = Task(
                        description=description,
                        difficulty=difficulty,
                        required_skills=required_skills
                    )

                    logger.info(f"New task proposed: {description} (difficulty: {difficulty})")
                    return task
            except Exception as e:
                logger.error(f"Error parsing task proposal: {e}")

        return None

    def _parse_button(self, response: str) -> str:
        """Parse a button action from response."""
        valid_buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
        response_upper = response.upper()

        for button in valid_buttons:
            if button in response_upper:
                return button

        return "A"  # Default

    def save_checkpoint(self, checkpoint_path: str = ".pokeagent_cache/voyager_checkpoint.json"):
        """Save agent state including curriculum progress."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "step_count": self.step_count,
            "current_task": {
                "description": self.current_task.description,
                "difficulty": self.current_task.difficulty,
                "required_skills": self.current_task.required_skills,
                "attempts": self.current_task.attempts
            } if self.current_task else None,
            "completed_tasks": [
                {
                    "description": t.description,
                    "difficulty": t.difficulty,
                    "success": t.success
                }
                for t in self.completed_tasks[-10:]  # Keep last 10
            ]
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved Voyager checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str = ".pokeagent_cache/voyager_checkpoint.json"):
        """Load agent state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.info(f"No checkpoint found at {checkpoint_path}")
            return

        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        self.step_count = checkpoint.get("step_count", 0)

        # Restore current task
        if checkpoint.get("current_task"):
            task_data = checkpoint["current_task"]
            self.current_task = Task(
                description=task_data["description"],
                difficulty=task_data["difficulty"],
                required_skills=task_data["required_skills"]
            )
            self.current_task.attempts = task_data.get("attempts", 0)

        # Restore completed tasks
        for task_data in checkpoint.get("completed_tasks", []):
            task = Task(
                description=task_data["description"],
                difficulty=task_data["difficulty"],
                required_skills=[]
            )
            task.success = task_data.get("success", False)
            self.completed_tasks.append(task)

        logger.info(f"Loaded Voyager checkpoint from {checkpoint_path}")


def create_voyager_agent(**kwargs) -> VoyagerAgent:
    """Factory function to create VoyagerAgent."""
    return VoyagerAgent(**kwargs)


def get_voyager_agent(vlm) -> VoyagerAgent:
    """Get or create the global voyager agent instance."""
    global _global_voyager_agent
    if '_global_voyager_agent' not in globals():
        _global_voyager_agent = VoyagerAgent(vlm=vlm)

        # Try to load checkpoint
        checkpoint_path = ".pokeagent_cache/voyager_checkpoint.json"
        if os.path.exists(checkpoint_path):
            _global_voyager_agent.load_checkpoint(checkpoint_path)

    return _global_voyager_agent

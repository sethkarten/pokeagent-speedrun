"""
ReAct Agent for Pokemon Emerald
================================

Implements a ReAct (Reasoning and Acting) agent that follows the pattern:
Thought -> Action -> Observation -> Thought -> ...

This agent explicitly reasons about the game state before taking actions,
making the decision process more interpretable and debuggable.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from utils.vlm import VLM
from utils.llm_logger import LLMLogger
from agent.system_prompt import system_prompt


class ActionType(Enum):
    """Possible action types in the ReAct framework."""
    PRESS_BUTTON = "press_button"
    OBSERVE = "observe"
    REMEMBER = "remember"
    PLAN = "plan"
    WAIT = "wait"


@dataclass
class Thought:
    """Represents a reasoning step."""
    content: str
    confidence: float = 0.0
    reasoning_type: str = "general"  # general, tactical, strategic, diagnostic


@dataclass
class Action:
    """Represents an action to take."""
    type: ActionType
    parameters: Dict[str, Any]
    justification: str = ""


@dataclass
class Observation:
    """Represents an observation from the environment."""
    content: str
    source: str  # game_state, memory, perception
    timestamp: float = 0.0


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    thought: Optional[Thought] = None
    action: Optional[Action] = None
    observation: Optional[Observation] = None
    step_number: int = 0


class ReActAgent:
    """
    ReAct Agent that explicitly reasons before acting.
    
    This agent maintains a history of thoughts, actions, and observations
    to make informed decisions about what to do next in the game.
    """
    
    def __init__(
        self,
        vlm_client: Optional[VLM] = None,
        max_history_length: int = 20,
        enable_reflection: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            vlm_client: Vision-language model client for reasoning
            max_history_length: Maximum number of steps to keep in history
            enable_reflection: Whether to periodically reflect on past actions
            verbose: Whether to print detailed reasoning
        """
        self.vlm_client = vlm_client or VLM()
        self.max_history_length = max_history_length
        self.enable_reflection = enable_reflection
        self.verbose = verbose
        
        self.history: List[ReActStep] = []
        self.current_step = 0
        self.current_plan: List[str] = []
        self.memory: Dict[str, Any] = {}
        
        self.llm_logger = LLMLogger()
        self.system_prompt = system_prompt
        
    def think(self, state: Dict[str, Any], screenshot: Any = None) -> Thought:
        """
        Generate a thought about the current situation.
        
        Args:
            state: Current game state
            screenshot: Current game screenshot
            
        Returns:
            A Thought object with reasoning about the situation
        """
        prompt = self._build_thought_prompt(state, screenshot)
        
        if screenshot:
            response = self.vlm_client.get_query(screenshot, prompt, "react")
        else:
            response = self.vlm_client.get_text_query(prompt, "react")
        
        self.llm_logger.log_interaction(
            interaction_type="react_think",
            prompt=prompt,
            response=response
        )
        
        # Parse the thought from the response
        thought = self._parse_thought(response)
        
        if self.verbose:
            print(f"==> THOUGHT: {thought.content}")
            
        return thought
    
    def act(self, thought: Thought, state: Dict[str, Any]) -> Action:
        """
        Decide on an action based on a thought and current state.
        
        Args:
            thought: The current reasoning
            state: Current game state
            
        Returns:
            An Action object describing what to do
        """
        prompt = self._build_action_prompt(thought, state)
        
        response = self.vlm_client.get_text_query(prompt, "react")
        
        self.llm_logger.log_interaction(
            interaction_type="react_act",
            prompt=prompt,
            response=response
        )
        
        # Parse the action from the response
        action = self._parse_action(response)
        
        if self.verbose:
            print(f">> ACTION: {action.type.value} - {action.parameters}")
            
        return action
    
    def observe(self, state: Dict[str, Any], action_result: Any = None) -> Observation:
        """
        Make an observation about the environment after an action.
        
        Args:
            state: Current game state after action
            action_result: Result of the previous action
            
        Returns:
            An Observation object describing what changed
        """
        # Compare with previous state if available
        changes = self._detect_changes(state)
        
        observation = Observation(
            content=self._summarize_changes(changes, state),
            source="game_state",
            timestamp=state.get("timestamp", 0)
        )
        
        if self.verbose:
            print(f"=A OBSERVATION: {observation.content}")
            
        return observation
    
    def step(self, state: Dict[str, Any], screenshot: Any = None) -> str:
        """
        Execute one complete ReAct step.
        
        Args:
            state: Current game state
            screenshot: Current game screenshot
            
        Returns:
            Button press command for the game
        """
        self.current_step += 1
        
        # Think about the situation
        thought = self.think(state, screenshot)
        
        # Decide on an action
        action = self.act(thought, state)
        
        # Create step record (observation will be added after action execution)
        step = ReActStep(
            thought=thought,
            action=action,
            step_number=self.current_step
        )
        
        # Add to history
        self._add_to_history(step)
        
        # Reflect periodically
        if self.enable_reflection and self.current_step % 10 == 0:
            self._reflect_on_progress()
        
        # Convert action to button press
        return self._action_to_button(action)
    
    def _build_thought_prompt(self, state: Dict[str, Any], screenshot: Any) -> str:
        """Build prompt for generating thoughts."""
        recent_history = self._get_recent_history_summary()
        
        prompt = f"""
You are playing Pokemon Emerald. Analyze the current situation and think about what's happening.

CURRENT STATE:
{json.dumps(state, indent=2)}

RECENT HISTORY:
{recent_history}

Based on the current state and what has happened recently, provide your reasoning about:
1. What is currently happening in the game?
2. What challenges or opportunities do you see?
3. What should be the immediate priority?

Respond with your thought in this format:
REASONING_TYPE: [general/tactical/strategic/diagnostic]
CONFIDENCE: [0.0-1.0]
THOUGHT: [Your detailed reasoning]
"""
        return prompt
    
    def _build_action_prompt(self, thought: Thought, state: Dict[str, Any]) -> str:
        """Build prompt for deciding on actions."""
        prompt = f"""
Based on your reasoning, decide on the next action to take.

YOUR THOUGHT:
{thought.content}

CURRENT STATE:
Player Position: {state.get('player_position', 'unknown')}
Current Map: {state.get('current_map', 'unknown')}
Battle Active: {state.get('battle_active', False)}

Available actions:
- press_button: Press a game button (A, B, UP, DOWN, LEFT, RIGHT, START, SELECT, L, R)
- observe: Take time to observe without acting
- remember: Store important information
- plan: Update your strategic plan
- wait: Wait for something to happen

Respond with your action in this format:
ACTION_TYPE: [press_button/observe/remember/plan/wait]
PARAMETERS: [JSON object with action parameters]
JUSTIFICATION: [Brief explanation of why this action]

Example:
ACTION_TYPE: press_button
PARAMETERS: {{"button": "A"}}
JUSTIFICATION: Interact with the NPC in front of us
"""
        return prompt
    
    def _parse_thought(self, response: str) -> Thought:
        """Parse a thought from LLM response."""
        lines = response.strip().split('\n')
        
        reasoning_type = "general"
        confidence = 0.5
        thought_content = response
        
        for line in lines:
            line = line.strip()  # Strip whitespace from each line
            if line.startswith("REASONING_TYPE:"):
                reasoning_type = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("THOUGHT:"):
                thought_content = line.split(":", 1)[1].strip()
        
        return Thought(
            content=thought_content,
            confidence=confidence,
            reasoning_type=reasoning_type
        )
    
    def _parse_action(self, response: str) -> Action:
        """Parse an action from LLM response."""
        lines = response.strip().split('\n')
        
        action_type = ActionType.WAIT
        parameters = {}
        justification = ""
        
        for line in lines:
            line = line.strip()  # Strip whitespace from each line
            if line.startswith("ACTION_TYPE:"):
                type_str = line.split(":", 1)[1].strip()
                try:
                    action_type = ActionType(type_str)
                except:
                    action_type = ActionType.WAIT
            elif line.startswith("PARAMETERS:"):
                param_str = line.split(":", 1)[1].strip()
                try:
                    parameters = json.loads(param_str)
                except:
                    parameters = {}
            elif line.startswith("JUSTIFICATION:"):
                justification = line.split(":", 1)[1].strip()
        
        return Action(
            type=action_type,
            parameters=parameters,
            justification=justification
        )
    
    def _action_to_button(self, action: Action) -> str:
        """Convert an Action to a button press command."""
        if action.type == ActionType.PRESS_BUTTON:
            return action.parameters.get("button", "NONE")
        elif action.type == ActionType.WAIT:
            return "NONE"
        else:
            # For non-button actions, process them and return no button press
            self._process_non_button_action(action)
            return "NONE"
    
    def _process_non_button_action(self, action: Action):
        """Process actions that don't directly press buttons."""
        if action.type == ActionType.REMEMBER:
            key = action.parameters.get("key", "general")
            value = action.parameters.get("value", "")
            self.memory[key] = value
            
        elif action.type == ActionType.PLAN:
            plan = action.parameters.get("plan", [])
            if isinstance(plan, list):
                self.current_plan = plan
                
        elif action.type == ActionType.OBSERVE:
            # Just observing, no action needed
            pass
    
    def _detect_changes(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect what changed in the game state."""
        changes = {}
        
        if self.history:
            # Get previous state from history
            for step in reversed(self.history):
                if step.observation:
                    # Compare with previous observed state
                    # This is simplified - you'd implement actual comparison
                    changes["position_changed"] = True
                    break
        
        return changes
    
    def _summarize_changes(self, changes: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Summarize what changed in a human-readable way."""
        if not changes:
            return "No significant changes observed"
        
        summary_parts = []
        if changes.get("position_changed"):
            summary_parts.append(f"Player moved to {state.get('player_position', 'unknown')}")
        
        return "; ".join(summary_parts) if summary_parts else "State updated"
    
    def _add_to_history(self, step: ReActStep):
        """Add a step to history, maintaining max length."""
        self.history.append(step)
        
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]
    
    def _get_recent_history_summary(self) -> str:
        """Get a summary of recent history for context."""
        if not self.history:
            return "No previous actions"
        
        recent = self.history[-5:]  # Last 5 steps
        summary = []
        
        for step in recent:
            if step.thought:
                summary.append(f"Step {step.step_number}: Thought: {step.thought.content[:100]}...")
            if step.action:
                summary.append(f"  Action: {step.action.type.value}")
            if step.observation:
                summary.append(f"  Observed: {step.observation.content[:100]}...")
        
        return "\n".join(summary)
    
    def _reflect_on_progress(self):
        """Periodically reflect on progress and adjust strategy."""
        if self.verbose:
            print("= REFLECTING ON PROGRESS...")
        
        reflection_prompt = f"""
Review your recent actions and their outcomes:

RECENT HISTORY:
{self._get_recent_history_summary()}

CURRENT PLAN:
{self.current_plan}

Reflect on:
1. Are you making progress toward your goals?
2. Are there any patterns in failed attempts?
3. Should you adjust your strategy?

Provide a brief reflection and any strategy adjustments.
"""
        
        response = self.vlm_client.get_text_query(reflection_prompt, "react_reflection")
        
        if self.verbose:
            print(f"=> REFLECTION: {response[:200]}...")
        
        # Store reflection in memory
        self.memory["last_reflection"] = response
        self.memory["reflection_step"] = self.current_step


# Convenience function for integration with existing codebase
def create_react_agent(**kwargs) -> ReActAgent:
    """Create a ReAct agent with default settings."""
    return ReActAgent(**kwargs)
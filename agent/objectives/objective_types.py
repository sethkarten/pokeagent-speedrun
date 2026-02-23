"""
Objective Types Module

Defines the core data structures for objectives used throughout the agent system.
This module is imported by both direct_objectives.py and all_obj.py to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DirectObjective:
    """Single direct objective with specific guidance.

    This class represents a single step in a game progression sequence, providing
    the agent with structured information about what to do and how to verify completion.

    Attributes:
        id: Unique identifier for this objective (e.g., "tutorial_001")
        description: Human-readable description of what to accomplish
        action_type: Category of action - "move", "interact", "battle", "wait", "navigate"
        category: Objective category - "story", "battling", or "dynamics"
        target_location: Optional name of location to reach (e.g., "Littleroot Town")
        target_coords: Optional (x, y) coordinates for precise positioning
        navigation_hint: Optional guidance on how to approach/complete the objective
        completion_condition: Optional condition name to verify completion
        priority: Priority level (1 = highest, 2 = medium, 3 = low)
        completed: Whether this objective has been marked complete
        optional: Whether this objective is optional (can be skipped)
        recommended_battling_objectives: List of battling objective IDs recommended before this objective
    """
    id: str
    description: str
    action_type: str  # "move", "interact", "battle", "wait", "navigate"
    category: str = "story"  # "story", "battling", or "dynamics"
    target_location: Optional[str] = None
    target_coords: Optional[tuple] = None
    navigation_hint: Optional[str] = None  # General direction/approach hint
    completion_condition: Optional[str] = None  # How to verify completion
    priority: int = 1  # 1 = highest priority, 2 = medium, 3 = low
    completed: bool = False
    optional: bool = False  # Whether this objective is optional
    recommended_battling_objectives: List[str] = field(default_factory=list)  # Recommended battling objective IDs
    prerequisite_story_objective: Optional[str] = None  # Story objective ID that must be reached before this battling objective shows

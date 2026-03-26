"""
Objective Types Module

Defines the core data structures for objectives used throughout the agent system.
This module is imported by direct objective loaders to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List


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
        completed_at: Timestamp when this objective was completed
        optional: Whether this objective is optional (can be skipped)
        recommended_battling_objectives: List of battling objective IDs recommended before this objective
        prerequisite_story_objective: Story objective ID that must be reached before this battling objective shows
    """
    id: str
    description: str
    action_type: str  # "move", "interact", "battle", "wait", "navigate"
    category: str = "story"  # "story", "battling", or "dynamics"
    target_location: Optional[str] = None
    target_coords: Optional[tuple] = None
    navigation_hint: Optional[str] = None
    completion_condition: Optional[str] = None
    priority: int = 1  # 1 = highest priority, 2 = medium, 3 = low
    completed: bool = False
    completed_at: Optional[datetime] = None
    optional: bool = False
    recommended_battling_objectives: List[str] = field(default_factory=list)
    prerequisite_story_objective: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict for persistence."""
        d: Dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "action_type": self.action_type,
            "category": self.category,
            "priority": self.priority,
            "completed": self.completed,
            "optional": self.optional,
        }
        if self.target_location is not None:
            d["target_location"] = self.target_location
        if self.target_coords is not None:
            d["target_coords"] = list(self.target_coords)
        if self.navigation_hint is not None:
            d["navigation_hint"] = self.navigation_hint
        if self.completion_condition is not None:
            d["completion_condition"] = self.completion_condition
        if self.completed_at is not None:
            d["completed_at"] = self.completed_at.isoformat()
        if self.recommended_battling_objectives:
            d["recommended_battling_objectives"] = self.recommended_battling_objectives
        if self.prerequisite_story_objective is not None:
            d["prerequisite_story_objective"] = self.prerequisite_story_objective
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DirectObjective":
        """Reconstruct a DirectObjective from a serialized dict."""
        completed_at_raw = data.get("completed_at")
        completed_at: Optional[datetime] = None
        if completed_at_raw is not None:
            if isinstance(completed_at_raw, datetime):
                completed_at = completed_at_raw
            else:
                completed_at = datetime.fromisoformat(str(completed_at_raw))

        coords_raw = data.get("target_coords")
        target_coords: Optional[tuple] = None
        if coords_raw is not None:
            target_coords = tuple(coords_raw)

        return cls(
            id=data["id"],
            description=data["description"],
            action_type=data.get("action_type", "navigate"),
            category=data.get("category", "story"),
            target_location=data.get("target_location"),
            target_coords=target_coords,
            navigation_hint=data.get("navigation_hint"),
            completion_condition=data.get("completion_condition"),
            priority=data.get("priority", 1),
            completed=data.get("completed", False),
            completed_at=completed_at,
            optional=data.get("optional", False),
            recommended_battling_objectives=data.get("recommended_battling_objectives", []),
            prerequisite_story_objective=data.get("prerequisite_story_objective"),
        )

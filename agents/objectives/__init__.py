"""
Objective logic for Pokemon Emerald agent: types, sequences, and categorized lists.

- objective_types: DirectObjective dataclass
- direct_objectives: DirectObjectiveManager, get_first_objective_info, supported sequence loaders
- all_obj_categorized: STORY_OBJECTIVES (214), BATTLING_OBJECTIVES (55)
"""

from .objective_types import DirectObjective
from .direct_objectives import DirectObjectiveManager, get_first_objective_info

__all__ = [
    "DirectObjective",
    "DirectObjectiveManager",
    "get_first_objective_info",
]

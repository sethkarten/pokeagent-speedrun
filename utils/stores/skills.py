"""
Skill store — persistent skill library for PokeAgent.

Inherits from BaseStore; skills represent learned strategies, tactics,
and procedural knowledge that the agent accumulates over runs.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from utils.stores.base_store import BaseStore

logger = logging.getLogger(__name__)


@dataclass
class SkillEntry:
    """A single entry in the skill library."""
    id: str = ""
    path: str = "general"
    name: str = ""
    description: str = ""
    effectiveness: str = "medium"  # low / medium / high
    source: str = "orchestrator"
    created_at: str = None  # type: ignore[assignment]
    updated_at: str = None  # type: ignore[assignment]
    importance: int = 3
    mutation_history: List[dict] = field(default_factory=list)
    title: str = ""  # alias for name — used by BaseStore tree overview

    def __post_init__(self):
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if not self.title and self.name:
            self.title = self.name
        elif self.title and not self.name:
            self.name = self.title


class SkillStore(BaseStore[SkillEntry]):
    """Persistent skill library."""

    file_name = "skills.json"
    id_prefix = "skill_"
    store_label = "SKILL LIBRARY"
    empty_message = "No skills learned yet."
    entry_class = SkillEntry

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self.load()

    def _deserialize_entry(self, entry_dict: dict) -> SkillEntry:
        entry_dict.setdefault("mutation_history", [])
        entry_dict.setdefault("source", "orchestrator")
        entry_dict.setdefault("effectiveness", "medium")
        entry_dict.setdefault("importance", 3)
        if entry_dict.get("name") and not entry_dict.get("title"):
            entry_dict["title"] = entry_dict["name"]
        elif entry_dict.get("title") and not entry_dict.get("name"):
            entry_dict["name"] = entry_dict["title"]
        return SkillEntry(**entry_dict)


# Global singleton
_skill_store: Optional[SkillStore] = None


def get_skill_store() -> SkillStore:
    """Get or create the global SkillStore instance."""
    global _skill_store
    if _skill_store is None:
        _skill_store = SkillStore()
    return _skill_store

"""
Subagent store — persistent subagent registry for PokeAgent.

Inherits from BaseStore; subagent entries describe executable configs
for custom (evolved or orchestrator-created) subagents, plus read-only
records for built-in subagents so they appear in the prompt registry.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from utils.stores.base_store import BaseStore, _INTERNAL_FIELDS

logger = logging.getLogger(__name__)

MAX_INSTRUCTIONS_LEN = 12_000
MAX_DIRECTIVE_LEN = 12_000

_SUBAGENT_DISPLAY_EXCLUDE = _INTERNAL_FIELDS | frozenset({
    "system_instructions",
    "directive",
})


@dataclass
class SubagentEntry:
    """A single entry in the subagent registry."""

    id: str = ""
    path: str = "custom"
    name: str = ""
    description: str = ""
    handler_type: str = "looping"  # "one_step" | "looping"
    max_turns: int = 25
    available_tools: List[str] = field(default_factory=list)
    system_instructions: str = ""
    directive: str = ""
    return_condition: str = ""
    importance: int = 3
    source: str = "orchestrator"  # "built-in" | "evolved" | "orchestrator"
    is_builtin: bool = False
    created_at: str = None  # type: ignore[assignment]
    updated_at: str = None  # type: ignore[assignment]
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


BUILTIN_SUBAGENT_CONFIGS: List[dict] = [
    {
        "path": "built-in/one_step",
        "name": "Reflect",
        "description": "Review trajectory, state, and screenshot to diagnose issues.",
        "handler_type": "one_step",
        "max_turns": 1,
        "available_tools": [],
        "is_builtin": True,
        "source": "built-in",
        "importance": 4,
    },
    {
        "path": "built-in/one_step",
        "name": "Verify",
        "description": "Check whether the current objective is actually complete.",
        "handler_type": "one_step",
        "max_turns": 1,
        "available_tools": [],
        "is_builtin": True,
        "source": "built-in",
        "importance": 4,
    },
    {
        "path": "built-in/one_step",
        "name": "Summarize",
        "description": "Summarize trajectory window into a handoff or situation review.",
        "handler_type": "one_step",
        "max_turns": 1,
        "available_tools": [],
        "is_builtin": True,
        "source": "built-in",
        "importance": 3,
    },
    {
        "path": "built-in/one_step",
        "name": "Gym Puzzle Analysis",
        "description": "Expert guidance on gym puzzle mechanics and solutions.",
        "handler_type": "one_step",
        "max_turns": 1,
        "available_tools": [],
        "is_builtin": True,
        "source": "built-in",
        "importance": 3,
    },
    {
        "path": "built-in/looping",
        "name": "Battler",
        "description": "Handle battle encounters with tool access (press_buttons, memory, skills).",
        "handler_type": "looping",
        "max_turns": 200,
        "available_tools": [
            "press_buttons", "process_memory", "process_skill",
            "add_memory", "search_memory", "get_memory_summary",
            "get_progress_summary", "lookup_pokemon_info",
        ],
        "is_builtin": True,
        "source": "built-in",
        "importance": 5, # NOTE: return_to_orchestrator functionality manually built-in to battler subagent, this can always be changed in the future
    },
    {
        "path": "built-in/looping",
        "name": "Planner",
        "description": "Plan and replan objectives using research tools and nested subagents.",
        "handler_type": "looping",
        "max_turns": 25,
        "available_tools": [
            "get_progress_summary", "get_walkthrough",
            "process_memory", "process_skill",
            "search_memory", "get_memory_summary", "add_memory",
            "lookup_pokemon_info",
            "subagent_summarize", "subagent_verify",
            "subagent_reflect", "subagent_gym_puzzle",
            "replan_objectives", # NOTE: return_to_orchestrator functionality manually built-in to planner subagent, this can always be changed in the future
        ],
        "is_builtin": True,
        "source": "built-in",
        "importance": 5,
    },
]


def _validate_char_caps(**fields) -> None:
    """Raise ValueError if system_instructions or directive exceed caps."""
    si = fields.get("system_instructions")
    if si is not None and len(si) > MAX_INSTRUCTIONS_LEN:
        raise ValueError(
            f"system_instructions exceeds {MAX_INSTRUCTIONS_LEN} char cap "
            f"({len(si)} chars)"
        )
    directive = fields.get("directive")
    if directive is not None and len(directive) > MAX_DIRECTIVE_LEN:
        raise ValueError(
            f"directive exceeds {MAX_DIRECTIVE_LEN} char cap "
            f"({len(directive)} chars)"
        )


class SubagentStore(BaseStore[SubagentEntry]):
    """Persistent subagent registry."""

    file_name = "subagents.json"
    id_prefix = "sa_"
    store_label = "SUBAGENT REGISTRY"
    empty_message = "No subagents registered yet."
    entry_class = SubagentEntry

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self.load()

    def load(self) -> None:
        super().load()
        self._seed_builtins()

    def _seed_builtins(self) -> None:
        """Populate built-in entries if none exist yet."""
        has_builtins = any(
            getattr(e, "is_builtin", False)
            for e in self.entries.values()
        )
        if has_builtins:
            return

        for cfg in BUILTIN_SUBAGENT_CONFIGS:
            self.add(**cfg)
        logger.info(
            "Seeded %d built-in subagent entries", len(BUILTIN_SUBAGENT_CONFIGS)
        )

    def add(self, **fields) -> str:
        _validate_char_caps(**fields)
        return super().add(**fields)

    def update(self, entry_id: str, **fields) -> bool:
        _validate_char_caps(**fields)
        return super().update(entry_id, **fields)

    def remove(self, entry_id: str) -> bool:
        entry = self.entries.get(entry_id)
        if entry is not None and getattr(entry, "is_builtin", False):
            logger.warning("Cannot delete built-in subagent %s", entry_id)
            return False
        return super().remove(entry_id)

    def to_display_dict(self, entry: SubagentEntry) -> dict:
        from dataclasses import asdict

        d = asdict(entry)
        for field_name in _SUBAGENT_DISPLAY_EXCLUDE:
            d.pop(field_name, None)
        return d

    def _deserialize_entry(self, entry_dict: dict) -> SubagentEntry:
        entry_dict.setdefault("mutation_history", [])
        entry_dict.setdefault("source", "orchestrator")
        entry_dict.setdefault("handler_type", "looping")
        entry_dict.setdefault("max_turns", 25)
        entry_dict.setdefault("available_tools", [])
        entry_dict.setdefault("system_instructions", "")
        entry_dict.setdefault("directive", "")
        entry_dict.setdefault("return_condition", "")
        entry_dict.setdefault("importance", 3)
        entry_dict.setdefault("is_builtin", False)
        if entry_dict.get("name") and not entry_dict.get("title"):
            entry_dict["title"] = entry_dict["name"]
        elif entry_dict.get("title") and not entry_dict.get("name"):
            entry_dict["name"] = entry_dict["title"]
        return SubagentEntry(**entry_dict)


# Global singleton
_subagent_store: Optional[SubagentStore] = None


def get_subagent_store() -> SubagentStore:
    """Get or create the global SubagentStore instance."""
    global _subagent_store
    if _subagent_store is None:
        _subagent_store = SubagentStore()
    return _subagent_store

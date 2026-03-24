"""
Long-Term Memory for PokeAgent

Provides persistent memory storage for game discoveries, NPC interactions,
item locations, and strategic notes. Replaces the former ``knowledge_base``
module with naming aligned to the AutoEvolve cognitive hierarchy.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single entry in long-term memory (formerly KnowledgeEntry)."""
    id: str
    category: str  # "location", "npc", "item", "pokemon", "strategy", "custom"
    title: str
    content: str
    location: Optional[str] = None
    coordinates: Optional[tuple] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = None
    updated_at: str = None
    importance: int = 3  # 1-5, where 5 is most important
    source: str = "orchestrator"  # "orchestrator" | "auto_evolve"
    last_modified_step: Optional[int] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


# Backward-compat aliases
KnowledgeEntry = MemoryEntry


class Memory:
    """Persistent long-term memory storage system (formerly KnowledgeBase)."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            from utils.data_persistence.run_data_manager import get_cache_directory
            cache_dir = str(get_cache_directory())

        self.cache_dir = cache_dir
        self.memory_file = os.path.join(cache_dir, "memory.json")

        os.makedirs(cache_dir, exist_ok=True)

        self.entries: Dict[str, MemoryEntry] = {}
        self.next_id = 1

        self._migrate_if_needed()
        self.load()

    def _migrate_if_needed(self) -> None:
        """Auto-migrate from knowledge_base.json -> memory.json if needed."""
        legacy_file = os.path.join(self.cache_dir, "knowledge_base.json")
        if not os.path.exists(self.memory_file) and os.path.exists(legacy_file):
            try:
                with open(legacy_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Add default source/last_modified_step to legacy entries
                for entry_dict in data.get("entries", {}).values():
                    entry_dict.setdefault("source", "orchestrator")
                    entry_dict.setdefault("last_modified_step", None)
                with open(self.memory_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"Migrated {legacy_file} -> {self.memory_file}")
            except Exception as e:
                logger.warning(f"Failed to migrate knowledge_base.json: {e}")

    def add(
        self,
        category: str,
        title: str,
        content: str,
        location: Optional[str] = None,
        coordinates: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        importance: int = 3,
        source: str = "orchestrator",
        last_modified_step: Optional[int] = None,
    ) -> str:
        entry_id = f"mem_{self.next_id:04d}"
        self.next_id += 1

        entry = MemoryEntry(
            id=entry_id,
            category=category,
            title=title,
            content=content,
            location=location,
            coordinates=coordinates,
            tags=tags or [],
            importance=importance,
            source=source,
            last_modified_step=last_modified_step,
        )

        self.entries[entry_id] = entry
        self.save()

        logger.info(f"Added memory entry: [{category}] {title}")
        return entry_id

    def update(
        self,
        entry_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        location: Optional[str] = None,
        coordinates: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[int] = None
    ) -> bool:
        if entry_id not in self.entries:
            logger.warning(f"Entry {entry_id} not found")
            return False

        entry = self.entries[entry_id]

        if title is not None:
            entry.title = title
        if content is not None:
            entry.content = content
        if location is not None:
            entry.location = location
        if coordinates is not None:
            entry.coordinates = coordinates
        if tags is not None:
            entry.tags = tags
        if importance is not None:
            entry.importance = importance

        entry.updated_at = datetime.now().isoformat()
        self.save()

        logger.info(f"Updated memory entry: {entry_id}")
        return True

    def remove(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            del self.entries[entry_id]
            self.save()
            logger.info(f"Removed memory entry: {entry_id}")
            return True
        return False

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)

    def search(
        self,
        category: Optional[str] = None,
        location: Optional[str] = None,
        tags: Optional[List[str]] = None,
        query: Optional[str] = None,
        min_importance: int = 1
    ) -> List[MemoryEntry]:
        results = []

        for entry in self.entries.values():
            if entry.importance < min_importance:
                continue
            if category and entry.category != category:
                continue
            if location and entry.location != location:
                continue
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            if query:
                query_lower = query.lower()
                if (query_lower not in entry.title.lower() and
                    query_lower not in entry.content.lower()):
                    continue
            results.append(entry)

        results.sort(key=lambda e: (e.importance, e.updated_at), reverse=True)
        return results

    def get_all(self, category: Optional[str] = None) -> List[MemoryEntry]:
        if category:
            return [e for e in self.entries.values() if e.category == category]
        return list(self.entries.values())

    def get_summary(self, max_entries: int = 20, min_importance: int = 3) -> str:
        important = [e for e in self.entries.values() if e.importance >= min_importance]
        important.sort(key=lambda e: (e.importance, e.updated_at), reverse=True)
        important = important[:max_entries]

        if not important:
            return "No memory entries yet."

        by_category: Dict[str, list] = {}
        for entry in important:
            by_category.setdefault(entry.category, []).append(entry)

        lines = ["=== LONG-TERM MEMORY SUMMARY ==="]

        for category in sorted(by_category.keys()):
            lines.append(f"\n[{category.upper()}]")
            for entry in by_category[category]:
                location_str = f" @ {entry.location}" if entry.location else ""
                coords_str = f" ({entry.coordinates[0]}, {entry.coordinates[1]})" if entry.coordinates else ""
                lines.append(f"  • {entry.title}{location_str}{coords_str}")
                if len(entry.content) <= 100:
                    lines.append(f"    {entry.content}")
                else:
                    lines.append(f"    {entry.content[:97]}...")

        lines.append(f"\nTotal: {len(important)} important entries (showing importance {min_importance}+)")
        return "\n".join(lines)

    def save(self) -> None:
        try:
            data = {
                "next_id": self.next_id,
                "entries": {}
            }

            for entry_id, entry in self.entries.items():
                entry_dict = asdict(entry)
                if entry_dict['coordinates']:
                    entry_dict['coordinates'] = list(entry_dict['coordinates'])
                data["entries"][entry_id] = entry_dict

            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.entries)} memory entries")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def load(self) -> None:
        if not os.path.exists(self.memory_file):
            logger.info("No existing memory file found, starting fresh")
            return

        try:
            if os.path.getsize(self.memory_file) == 0:
                logger.info("Memory file is empty, starting fresh")
                return

            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.next_id = data.get("next_id", 1)

            for entry_id, entry_dict in data.get("entries", {}).items():
                if entry_dict.get('coordinates'):
                    entry_dict['coordinates'] = tuple(entry_dict['coordinates'])
                entry_dict.setdefault("source", "orchestrator")
                entry_dict.setdefault("last_modified_step", None)
                entry = MemoryEntry(**entry_dict)
                self.entries[entry_id] = entry

            logger.info(f"Loaded {len(self.entries)} memory entries")
        except json.JSONDecodeError as e:
            logger.warning(f"Memory file contains invalid JSON: {e}. Starting fresh.")
            self.next_id = 1
            self.entries = {}
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")

    def clear(self) -> None:
        self.entries.clear()
        self.next_id = 1
        self.save()
        logger.info("Cleared memory")


# Backward-compat aliases
KnowledgeBase = Memory

# Global instance
_memory_store = None


def get_memory_store() -> Memory:
    """Get or create the global Memory instance (persistent across runs)."""
    global _memory_store
    if _memory_store is None:
        _memory_store = Memory()
    return _memory_store


# Backward-compat alias
get_knowledge_base = get_memory_store

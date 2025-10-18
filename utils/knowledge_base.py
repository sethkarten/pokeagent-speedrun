"""
Knowledge Base for Claude Plays Pokemon

Provides persistent memory storage for game discoveries, NPC interactions,
item locations, and strategic notes. This mimics the original ClaudePlaysPokemon's
knowledge base system for maintaining context across gameplay sessions.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """A single entry in the knowledge base."""
    id: str
    category: str  # "location", "npc", "item", "pokemon", "strategy", "custom"
    title: str
    content: str
    location: Optional[str] = None  # Map name where this knowledge applies
    coordinates: Optional[tuple] = None  # (x, y) coordinates if applicable
    tags: List[str] = None
    created_at: str = None
    updated_at: str = None
    importance: int = 3  # 1-5, where 5 is most important

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


class KnowledgeBase:
    """
    Persistent knowledge storage system for Claude Plays Pokemon.

    Stores and retrieves game knowledge including:
    - Location discoveries and descriptions
    - NPC positions and dialogue
    - Item locations and availability
    - Pokemon encounters and strategies
    - General gameplay notes
    """

    def __init__(self, cache_dir: str = ".pokeagent_cache"):
        """
        Initialize the knowledge base.

        Args:
            cache_dir: Directory for storing knowledge base file
        """
        self.cache_dir = cache_dir
        self.knowledge_file = os.path.join(cache_dir, "knowledge_base.json")

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # In-memory storage
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.next_id = 1

        # Load existing knowledge
        self.load()

    def add(
        self,
        category: str,
        title: str,
        content: str,
        location: Optional[str] = None,
        coordinates: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        importance: int = 3
    ) -> str:
        """
        Add a new knowledge entry.

        Args:
            category: Category of knowledge (location/npc/item/pokemon/strategy/custom)
            title: Brief title for the entry
            content: Detailed content/description
            location: Map name where this applies
            coordinates: (x, y) position if applicable
            tags: List of tags for searching
            importance: 1-5 importance rating

        Returns:
            Entry ID
        """
        entry_id = f"kb_{self.next_id:04d}"
        self.next_id += 1

        entry = KnowledgeEntry(
            id=entry_id,
            category=category,
            title=title,
            content=content,
            location=location,
            coordinates=coordinates,
            tags=tags or [],
            importance=importance
        )

        self.entries[entry_id] = entry
        self.save()

        logger.info(f"Added knowledge entry: [{category}] {title}")
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
        """
        Update an existing knowledge entry.

        Args:
            entry_id: ID of entry to update
            title: New title (if provided)
            content: New content (if provided)
            location: New location (if provided)
            coordinates: New coordinates (if provided)
            tags: New tags (if provided)
            importance: New importance (if provided)

        Returns:
            True if updated, False if entry not found
        """
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

        logger.info(f"Updated knowledge entry: {entry_id}")
        return True

    def remove(self, entry_id: str) -> bool:
        """
        Remove a knowledge entry.

        Args:
            entry_id: ID of entry to remove

        Returns:
            True if removed, False if not found
        """
        if entry_id in self.entries:
            del self.entries[entry_id]
            self.save()
            logger.info(f"Removed knowledge entry: {entry_id}")
            return True
        return False

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a specific entry by ID."""
        return self.entries.get(entry_id)

    def search(
        self,
        category: Optional[str] = None,
        location: Optional[str] = None,
        tags: Optional[List[str]] = None,
        query: Optional[str] = None,
        min_importance: int = 1
    ) -> List[KnowledgeEntry]:
        """
        Search for knowledge entries.

        Args:
            category: Filter by category
            location: Filter by location
            tags: Filter by tags (entry must have at least one matching tag)
            query: Text search in title and content
            min_importance: Minimum importance level

        Returns:
            List of matching entries, sorted by importance (descending)
        """
        results = []

        for entry in self.entries.values():
            # Filter by importance
            if entry.importance < min_importance:
                continue

            # Filter by category
            if category and entry.category != category:
                continue

            # Filter by location
            if location and entry.location != location:
                continue

            # Filter by tags
            if tags and not any(tag in entry.tags for tag in tags):
                continue

            # Filter by query
            if query:
                query_lower = query.lower()
                if (query_lower not in entry.title.lower() and
                    query_lower not in entry.content.lower()):
                    continue

            results.append(entry)

        # Sort by importance (descending) then by updated_at (descending)
        results.sort(key=lambda e: (e.importance, e.updated_at), reverse=True)
        return results

    def get_all(self, category: Optional[str] = None) -> List[KnowledgeEntry]:
        """Get all entries, optionally filtered by category."""
        if category:
            return [e for e in self.entries.values() if e.category == category]
        return list(self.entries.values())

    def get_summary(self, max_entries: int = 20, min_importance: int = 3) -> str:
        """
        Get a formatted summary of important knowledge.

        Args:
            max_entries: Maximum number of entries to include
            min_importance: Minimum importance level to include

        Returns:
            Formatted text summary
        """
        # Get important entries
        important = [e for e in self.entries.values() if e.importance >= min_importance]
        important.sort(key=lambda e: (e.importance, e.updated_at), reverse=True)
        important = important[:max_entries]

        if not important:
            return "No knowledge entries yet."

        # Group by category
        by_category = {}
        for entry in important:
            if entry.category not in by_category:
                by_category[entry.category] = []
            by_category[entry.category].append(entry)

        # Format summary
        lines = ["=== KNOWLEDGE BASE SUMMARY ==="]

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
        """Save knowledge base to disk."""
        try:
            data = {
                "next_id": self.next_id,
                "entries": {}
            }

            for entry_id, entry in self.entries.items():
                entry_dict = asdict(entry)
                # Convert tuple to list for JSON serialization
                if entry_dict['coordinates']:
                    entry_dict['coordinates'] = list(entry_dict['coordinates'])
                data["entries"][entry_id] = entry_dict

            with open(self.knowledge_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.entries)} knowledge entries")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def load(self) -> None:
        """Load knowledge base from disk."""
        if not os.path.exists(self.knowledge_file):
            logger.info("No existing knowledge base found, starting fresh")
            return

        try:
            with open(self.knowledge_file, 'r') as f:
                data = json.load(f)

            self.next_id = data.get("next_id", 1)

            for entry_id, entry_dict in data.get("entries", {}).items():
                # Convert list back to tuple for coordinates
                if entry_dict.get('coordinates'):
                    entry_dict['coordinates'] = tuple(entry_dict['coordinates'])

                entry = KnowledgeEntry(**entry_dict)
                self.entries[entry_id] = entry

            logger.info(f"Loaded {len(self.entries)} knowledge entries")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")

    def clear(self) -> None:
        """Clear all knowledge entries (for testing)."""
        self.entries.clear()
        self.next_id = 1
        self.save()
        logger.info("Cleared knowledge base")


# Global instance
_knowledge_base = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base

"""
BaseStore — generic persistent store with tree overview, CRUD, and caching.

Concrete stores (Memory, SkillStore) inherit from this and only define
their entry dataclass plus any store-specific logic (e.g. text search).
"""

import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)


class StoreEntry(Protocol):
    """Minimum fields every store entry must expose."""
    id: str
    path: str
    title: str
    importance: int
    mutation_history: list
    created_at: str
    updated_at: str


T = TypeVar("T")


# Fields excluded from orchestrator-facing read responses
_INTERNAL_FIELDS = frozenset({
    "mutation_history",
    "created_at",
    "updated_at",
    "last_modified_step",
    "tags",
})


class BaseStore(Generic[T]):
    """Persistent JSON store with tree overview and CRUD.

    Subclasses must set:
      - ``file_name``   e.g. ``"memory.json"``
      - ``id_prefix``   e.g. ``"mem_"``
      - ``store_label`` e.g. ``"LONG-TERM MEMORY"`` (for overview header)
      - ``empty_message`` e.g. ``"No memory entries yet."``
      - ``entry_class``  the dataclass type used for entries

    And implement ``_deserialize_entry(entry_dict) -> T`` to reconstruct
    an entry from its JSON dict (handles field migration, type coercion).
    """

    file_name: str = ""
    id_prefix: str = ""
    store_label: str = ""
    empty_message: str = "No entries yet."
    entry_class: type = None  # type: ignore[assignment]

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            from utils.data_persistence.run_data_manager import get_cache_directory
            cache_dir = str(get_cache_directory())

        self.cache_dir = cache_dir
        self.store_file = os.path.join(cache_dir, self.file_name)
        os.makedirs(cache_dir, exist_ok=True)

        self.entries: Dict[str, T] = {}
        self.next_id: int = 1
        self._cached_tree: Optional[str] = None
        self._recent_access_ids: set = set()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, **fields) -> str:
        """Create a new entry. Returns the entry ID.

        If ``id`` is provided in *fields* and is not already taken, it is
        used as-is (allows human-readable IDs like ``"move_to_coords"``).
        Otherwise a numeric ID is auto-generated.
        """
        custom_id = fields.get("id")
        if custom_id and custom_id not in self.entries:
            entry_id = custom_id
        else:
            entry_id = f"{self.id_prefix}{self.next_id:04d}"
            self.next_id += 1

        fields["id"] = entry_id
        fields.setdefault("created_at", datetime.now().isoformat())
        fields.setdefault("updated_at", fields["created_at"])
        fields.setdefault("mutation_history", [])

        entry = self.entry_class(**fields)
        self.entries[entry_id] = entry
        self._invalidate_cache()
        self.save()

        title = getattr(entry, "title", "") or getattr(entry, "name", "")
        logger.info(f"Added {self.id_prefix}entry: {entry_id} ({title})")
        return entry_id

    def update(self, entry_id: str, **fields) -> bool:
        """Update fields on an existing entry. Returns False if not found."""
        if entry_id not in self.entries:
            logger.warning(f"Entry {entry_id} not found for update")
            return False

        entry = self.entries[entry_id]
        changed: Dict[str, Any] = {}
        for key, value in fields.items():
            if value is not None and hasattr(entry, key) and key != "mutation_history":
                old = getattr(entry, key)
                if old != value:
                    changed[key] = {"old": old, "new": value}
                setattr(entry, key, value)

        now = datetime.now().isoformat()
        if changed and hasattr(entry, "mutation_history"):
            entry.mutation_history.append({"timestamp": now, "fields": changed})  # type: ignore[attr-defined]

        entry.updated_at = now  # type: ignore[attr-defined]
        self._invalidate_cache()
        self.save()
        logger.info(f"Updated entry: {entry_id}")
        return True

    def remove(self, entry_id: str) -> bool:
        """Delete an entry. Returns False if not found."""
        if entry_id not in self.entries:
            return False
        del self.entries[entry_id]
        self._invalidate_cache()
        self.save()
        logger.info(f"Removed entry: {entry_id}")
        return True

    def get(self, entry_id: str) -> Optional[T]:
        """Look up by ID first, then fall back to name/title match."""
        entry = self.entries.get(entry_id)
        if entry is not None:
            self._recent_access_ids.add(entry_id)
            return entry
        # Fall back: search by name or title (case-insensitive)
        lookup = entry_id.lower().strip()
        for e in self.entries.values():
            name = getattr(e, "name", "") or ""
            title = getattr(e, "title", "") or ""
            if name.lower().strip() == lookup or title.lower().strip() == lookup:
                self._recent_access_ids.add(e.id)
                return e
        return None

    def get_multiple(self, ids: List[str], max_count: int = 3) -> List[T]:
        """Return up to *max_count* entries by ID (order preserved)."""
        results: List[T] = []
        for eid in ids[:max_count]:
            entry = self.entries.get(eid)
            if entry is not None:
                self._recent_access_ids.add(eid)
                results.append(entry)
        return results

    def get_all(self) -> List[T]:
        return list(self.entries.values())

    def clear(self) -> None:
        self.entries.clear()
        self.next_id = 1
        self._invalidate_cache()
        self.save()
        logger.info(f"Cleared {self.store_label} store")

    # ------------------------------------------------------------------
    # Tree overview
    # ------------------------------------------------------------------

    def get_tree_overview(self, max_display: int = 200) -> str:
        """Render a hierarchical tree of entries grouped by path.

        Subtrees containing recently-accessed entries are shown first.

        Returns a string like::

            === LONG-TERM MEMORY OVERVIEW ===
            pokemon:
              - [mem_0001] My Starter Mudkip
              gym_leaders:
                - [mem_0042] Roxanne's Team Composition
            events:
              - [mem_0003] Received Pokedex
        """
        if self._cached_tree is not None:
            return self._cached_tree

        if not self.entries:
            self._cached_tree = self.empty_message
            return self._cached_tree

        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: (-(getattr(e, "importance", 3)), getattr(e, "updated_at", "") or ""),
            reverse=False,
        )

        display_entries = sorted_entries[:max_display]
        overflow = len(sorted_entries) - max_display

        tree: Dict[str, Any] = {}
        for entry in display_entries:
            path = getattr(entry, "path", "") or "uncategorized"
            segments = [s for s in path.strip("/").split("/") if s]
            if not segments:
                segments = ["uncategorized"]
            if len(segments) > 5:
                segments = segments[:5]

            node = tree
            for seg in segments:
                node = node.setdefault(seg, {})
            leaves = node.setdefault("__entries__", [])
            leaves.append(self._format_tree_leaf(entry))

        recent_roots = self._compute_recent_roots(display_entries)

        lines = [f"=== {self.store_label} OVERVIEW ==="]
        self._render_tree(tree, lines, indent=0, recent_roots=recent_roots)

        if overflow > 0:
            lines.append(f"\n(+{overflow} more entries — use process tool to inspect)")

        self._recent_access_ids.clear()
        self._cached_tree = "\n".join(lines)
        return self._cached_tree

    def _compute_recent_roots(self, entries: list) -> set:
        """Return root path segments that contain recently-accessed entries."""
        if not self._recent_access_ids:
            return set()
        roots: set = set()
        for entry in entries:
            if entry.id in self._recent_access_ids:
                path = getattr(entry, "path", "") or "uncategorized"
                segments = [s for s in path.strip("/").split("/") if s]
                if segments:
                    roots.add(segments[0])
                else:
                    roots.add("uncategorized")
        return roots

    def _format_tree_leaf(self, entry: T) -> str:
        """Format a single entry for the tree overview. Override in subclasses."""
        title = getattr(entry, "title", "") or getattr(entry, "name", "")
        return f"[{entry.id}] {title}"

    def _render_tree(self, node: dict, lines: list, indent: int,
                      recent_roots: Optional[set] = None) -> None:
        prefix = "  " * indent
        keys = [k for k in node.keys() if k != "__entries__"]
        if indent == 0 and recent_roots:
            keys.sort(key=lambda k: (0 if k in recent_roots else 1, k))
        else:
            keys.sort()
        # Render leaf entries first, then subtrees
        if "__entries__" in node:
            for leaf in node["__entries__"]:
                lines.append(f"{prefix}- {leaf}")
        for key in keys:
            lines.append(f"{prefix}{key}:")
            self._render_tree(node[key], lines, indent + 1)

    def _invalidate_cache(self) -> None:
        self._cached_tree = None

    # ------------------------------------------------------------------
    # Display filter
    # ------------------------------------------------------------------

    def to_display_dict(self, entry: T) -> dict:
        """Return only orchestrator-visible fields for a read response."""
        d = asdict(entry)  # type: ignore[arg-type]
        for field_name in _INTERNAL_FIELDS:
            d.pop(field_name, None)
        if d.get("coordinates"):
            d["coordinates"] = list(d["coordinates"])
        return d

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _serialize_entry(self, entry: T) -> dict:
        """Convert an entry to a JSON-safe dict."""
        d = asdict(entry)  # type: ignore[arg-type]
        if d.get("coordinates"):
            d["coordinates"] = list(d["coordinates"])
        return d

    def save(self) -> None:
        try:
            data = {
                "next_id": self.next_id,
                "entries": {eid: self._serialize_entry(e) for eid, e in self.entries.items()},
            }
            with open(self.store_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {len(self.entries)} {self.store_label} entries")
        except Exception as e:
            logger.error(f"Failed to save {self.store_label}: {e}")

    def load(self) -> None:
        if not os.path.exists(self.store_file):
            logger.info(f"No existing {self.file_name} found, starting fresh")
            return

        try:
            if os.path.getsize(self.store_file) == 0:
                logger.info(f"{self.file_name} is empty, starting fresh")
                return

            with open(self.store_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.next_id = data.get("next_id", 1)
            for entry_id, entry_dict in data.get("entries", {}).items():
                entry = self._deserialize_entry(entry_dict)
                self.entries[entry_id] = entry

            logger.info(f"Loaded {len(self.entries)} {self.store_label} entries")
        except json.JSONDecodeError as e:
            logger.warning(f"{self.file_name} contains invalid JSON: {e}. Starting fresh.")
            self.next_id = 1
            self.entries = {}
        except Exception as e:
            logger.error(f"Failed to load {self.store_label}: {e}")

    def _deserialize_entry(self, entry_dict: dict) -> T:
        """Reconstruct an entry from a JSON dict.

        Override in subclasses for field migration / type coercion.
        """
        return self.entry_class(**entry_dict)

    # ------------------------------------------------------------------
    # Sync to run_data
    # ------------------------------------------------------------------

    def sync_to_run_data(self, dest_dir: str) -> None:
        """Copy the store file to *dest_dir* (e.g. run_data agent_scratch_space)."""
        if not os.path.exists(self.store_file):
            logger.debug(f"{self.file_name} not found, skipping sync")
            return
        dest = os.path.join(dest_dir, self.file_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(self.store_file, dest)
        logger.info(f"Synced {self.file_name} -> {dest}")

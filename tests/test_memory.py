"""Tests for utils.memory: MemoryEntry serde, Memory store, and migration."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys_path = Path(__file__).parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from utils.memory import Memory, MemoryEntry, KnowledgeBase, KnowledgeEntry, get_memory_store


# ---------------------------------------------------------------------------
# MemoryEntry round-trip
# ---------------------------------------------------------------------------


class TestMemoryEntrySerde:
    def test_defaults(self):
        entry = MemoryEntry(id="mem_0001", category="location", title="Littleroot Town", content="Starting town")
        assert entry.source == "orchestrator"
        assert entry.last_modified_step is None
        assert entry.importance == 3
        assert isinstance(entry.created_at, str)

    def test_backward_compat_alias(self):
        assert KnowledgeEntry is MemoryEntry

    def test_coordinates_stored_as_tuple(self):
        entry = MemoryEntry(id="mem_0002", category="item", title="Potion", content="Found potion", coordinates=(5, 8))
        assert isinstance(entry.coordinates, tuple)


# ---------------------------------------------------------------------------
# Memory store basics
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def test_add_and_search(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        entry_id = store.add(category="location", title="Route 101", content="Grassy route north of Littleroot")
        assert entry_id == "mem_0001"

        results = store.search(category="location")
        assert len(results) == 1
        assert results[0].title == "Route 101"

    def test_save_and_reload(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        store.add(category="pokemon", title="Mudkip", content="Starter", importance=5)
        store.add(category="npc", title="Prof. Birch", content="Gives starter", importance=4)

        store2 = Memory(cache_dir=str(tmp_path))
        assert len(store2.entries) == 2
        assert store2.next_id == 3

    def test_summary_empty(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        assert store.get_summary() == "No memory entries yet."

    def test_summary_with_entries(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        store.add(category="strategy", title="Grind before gym", content="Level up before Roxanne", importance=4)
        summary = store.get_summary()
        assert "LONG-TERM MEMORY SUMMARY" in summary
        assert "Grind before gym" in summary

    def test_update_entry(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        entry_id = store.add(category="item", title="Potion", content="At Route 101")
        store.update(entry_id, content="At Route 101 near the tree")
        assert store.entries[entry_id].content == "At Route 101 near the tree"

    def test_remove_entry(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        entry_id = store.add(category="item", title="Potion", content="At Route 101")
        assert store.remove(entry_id) is True
        assert len(store.entries) == 0

    def test_clear(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        store.add(category="item", title="Potion", content="Found it")
        store.clear()
        assert len(store.entries) == 0
        assert store.next_id == 1

    def test_backward_compat_class_alias(self, tmp_path):
        assert KnowledgeBase is Memory

    def test_source_and_step_fields(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        entry_id = store.add(
            category="strategy",
            title="Battle tip",
            content="Use Water Gun",
            source="auto_evolve",
            last_modified_step=42,
        )
        entry = store.get(entry_id)
        assert entry.source == "auto_evolve"
        assert entry.last_modified_step == 42

    def test_coordinates_round_trip(self, tmp_path):
        store = Memory(cache_dir=str(tmp_path))
        store.add(category="location", title="Gym", content="Rustboro gym", coordinates=(10, 20))

        store2 = Memory(cache_dir=str(tmp_path))
        entry = list(store2.entries.values())[0]
        assert isinstance(entry.coordinates, tuple)
        assert entry.coordinates == (10, 20)


# ---------------------------------------------------------------------------
# Migration from knowledge_base.json -> memory.json
# ---------------------------------------------------------------------------


class TestMigration:
    def _write_legacy_kb(self, cache_dir: Path):
        """Write a knowledge_base.json in the legacy format."""
        data = {
            "next_id": 3,
            "entries": {
                "kb_0001": {
                    "id": "kb_0001",
                    "category": "location",
                    "title": "Route 101",
                    "content": "Grassy area",
                    "location": "Route 101",
                    "coordinates": [4, 8],
                    "tags": ["route"],
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-01T00:00:00",
                    "importance": 4,
                },
                "kb_0002": {
                    "id": "kb_0002",
                    "category": "pokemon",
                    "title": "Mudkip",
                    "content": "Starter pokemon",
                    "location": None,
                    "coordinates": None,
                    "tags": [],
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-01T00:00:00",
                    "importance": 5,
                },
            },
        }
        with open(cache_dir / "knowledge_base.json", "w") as f:
            json.dump(data, f)

    def test_auto_migrates_knowledge_base_to_memory(self, tmp_path):
        self._write_legacy_kb(tmp_path)
        assert not (tmp_path / "memory.json").exists()

        store = Memory(cache_dir=str(tmp_path))

        assert (tmp_path / "memory.json").exists()
        assert len(store.entries) == 2
        assert store.entries["kb_0001"].source == "orchestrator"
        assert store.entries["kb_0001"].last_modified_step is None
        assert isinstance(store.entries["kb_0001"].coordinates, tuple)
        assert store.entries["kb_0001"].coordinates == (4, 8)

    def test_no_migration_when_memory_already_exists(self, tmp_path):
        self._write_legacy_kb(tmp_path)
        with open(tmp_path / "memory.json", "w") as f:
            json.dump({"next_id": 1, "entries": {}}, f)

        store = Memory(cache_dir=str(tmp_path))
        assert len(store.entries) == 0

    def test_migration_preserves_next_id(self, tmp_path):
        self._write_legacy_kb(tmp_path)
        store = Memory(cache_dir=str(tmp_path))
        assert store.next_id == 3


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGetMemoryStore:
    def test_returns_same_instance(self, tmp_path):
        import utils.memory as mem_module

        old_store = mem_module._memory_store
        try:
            mem_module._memory_store = None
            with patch("utils.memory.Memory", side_effect=lambda **kw: object()) as mock_cls:
                s1 = get_memory_store()
                s2 = get_memory_store()
                assert s1 is s2
                mock_cls.assert_called_once()
        finally:
            mem_module._memory_store = old_store

"""Tests for utils.stores.base_store — generic CRUD, persistence, tree overview, display filter."""

import json
import os
import tempfile

import pytest
from dataclasses import dataclass, field
from typing import List, Optional

from utils.stores.base_store import BaseStore


# ---------------------------------------------------------------------------
# Concrete test store
# ---------------------------------------------------------------------------

@dataclass
class DummyEntry:
    id: str = ""
    path: str = "general"
    title: str = ""
    content: str = ""
    importance: int = 3
    created_at: str = None
    updated_at: str = None
    mutation_history: List[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


class DummyStore(BaseStore[DummyEntry]):
    file_name = "dummy.json"
    id_prefix = "dum_"
    store_label = "DUMMY"
    empty_message = "No dummies yet."
    entry_class = DummyEntry

    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)
        self.load()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return DummyStore(cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_add_returns_id(self, store):
        eid = store.add(path="a", title="Hello", content="World")
        assert eid == "dum_0001"
        assert len(store.entries) == 1

    def test_add_increments_ids(self, store):
        store.add(path="a", title="One")
        store.add(path="b", title="Two")
        assert "dum_0001" in store.entries
        assert "dum_0002" in store.entries

    def test_get_returns_entry(self, store):
        eid = store.add(path="x", title="Found", content="body")
        entry = store.get(eid)
        assert entry.title == "Found"
        assert entry.content == "body"

    def test_get_missing_returns_none(self, store):
        assert store.get("dum_9999") is None

    def test_get_multiple(self, store):
        ids = [store.add(path="p", title=f"T{i}") for i in range(5)]
        result = store.get_multiple(ids, max_count=3)
        assert len(result) == 3
        assert result[0].title == "T0"

    def test_update(self, store):
        eid = store.add(path="old", title="Old")
        ok = store.update(eid, title="New", path="new")
        assert ok is True
        assert store.get(eid).title == "New"
        assert store.get(eid).path == "new"

    def test_update_missing_returns_false(self, store):
        assert store.update("dum_9999", title="nope") is False

    def test_remove(self, store):
        eid = store.add(path="a", title="Bye")
        assert store.remove(eid) is True
        assert store.get(eid) is None
        assert len(store.entries) == 0

    def test_remove_missing_returns_false(self, store):
        assert store.remove("dum_9999") is False

    def test_clear(self, store):
        store.add(path="a", title="One")
        store.add(path="b", title="Two")
        store.clear()
        assert len(store.entries) == 0
        assert store.next_id == 1

    def test_get_all(self, store):
        store.add(path="a", title="A")
        store.add(path="b", title="B")
        assert len(store.get_all()) == 2


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        s1 = DummyStore(cache_dir=str(tmp_path))
        s1.add(path="p", title="Saved", content="data", importance=5)
        s1.add(path="q", title="Also", content="saved")

        s2 = DummyStore(cache_dir=str(tmp_path))
        assert len(s2.entries) == 2
        assert s2.get("dum_0001").title == "Saved"
        assert s2.next_id == 3

    def test_empty_file_starts_fresh(self, tmp_path):
        p = tmp_path / "dummy.json"
        p.write_text("")
        s = DummyStore(cache_dir=str(tmp_path))
        assert len(s.entries) == 0

    def test_corrupt_json_starts_fresh(self, tmp_path):
        p = tmp_path / "dummy.json"
        p.write_text("{broken json")
        s = DummyStore(cache_dir=str(tmp_path))
        assert len(s.entries) == 0

    def test_no_file_starts_fresh(self, tmp_path):
        s = DummyStore(cache_dir=str(tmp_path))
        assert len(s.entries) == 0

    def test_auto_save_after_add(self, tmp_path):
        s = DummyStore(cache_dir=str(tmp_path))
        s.add(path="x", title="T")
        raw = json.loads((tmp_path / "dummy.json").read_text())
        assert "dum_0001" in raw["entries"]

    def test_auto_save_after_update(self, tmp_path):
        s = DummyStore(cache_dir=str(tmp_path))
        eid = s.add(path="x", title="Before")
        s.update(eid, title="After")
        raw = json.loads((tmp_path / "dummy.json").read_text())
        assert raw["entries"]["dum_0001"]["title"] == "After"

    def test_auto_save_after_remove(self, tmp_path):
        s = DummyStore(cache_dir=str(tmp_path))
        eid = s.add(path="x", title="Gone")
        s.remove(eid)
        raw = json.loads((tmp_path / "dummy.json").read_text())
        assert "dum_0001" not in raw["entries"]

    def test_sync_to_run_data(self, tmp_path):
        s = DummyStore(cache_dir=str(tmp_path))
        s.add(path="x", title="Sync")
        dest = str(tmp_path / "run_dest")
        s.sync_to_run_data(dest)
        assert (tmp_path / "run_dest" / "dummy.json").exists()


# ---------------------------------------------------------------------------
# Tree overview
# ---------------------------------------------------------------------------

class TestTreeOverview:
    def test_empty_store(self, store):
        assert store.get_tree_overview() == "No dummies yet."

    def test_single_entry(self, store):
        store.add(path="pokemon", title="Mudkip")
        overview = store.get_tree_overview()
        assert "=== DUMMY OVERVIEW ===" in overview
        assert "pokemon:" in overview
        assert "[dum_0001] Mudkip" in overview

    def test_nested_paths(self, store):
        store.add(path="pokemon/starters", title="Mudkip")
        store.add(path="pokemon/gym_leaders", title="Roxanne")
        store.add(path="events", title="Got Pokedex")
        overview = store.get_tree_overview()
        assert "pokemon:" in overview
        assert "  starters:" in overview
        assert "  gym_leaders:" in overview
        assert "events:" in overview

    def test_max_display_cap(self, store):
        for i in range(10):
            store.add(path="bulk", title=f"Entry {i}")
        overview = store.get_tree_overview(max_display=5)
        assert "(+5 more entries" in overview

    def test_deep_path_flattened_at_4(self, store):
        store.add(path="a/b/c/d/e/f", title="Deep")
        overview = store.get_tree_overview()
        # Depth capped at 4 levels
        assert "a:" in overview
        assert "  b:" in overview

    def test_uncategorized_default(self, store):
        store.add(path="", title="No path")
        overview = store.get_tree_overview()
        assert "uncategorized:" in overview

    def test_unicode_in_titles(self, store):
        store.add(path="special", title="🏃 Running Shoes")
        overview = store.get_tree_overview()
        assert "🏃 Running Shoes" in overview


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

class TestCacheInvalidation:
    def test_add_invalidates_cache(self, store):
        store.add(path="a", title="First")
        overview1 = store.get_tree_overview()
        store.add(path="b", title="Second")
        overview2 = store.get_tree_overview()
        assert overview1 != overview2
        assert "Second" in overview2

    def test_update_invalidates_cache(self, store):
        eid = store.add(path="a", title="Before")
        store.get_tree_overview()
        store.update(eid, title="After")
        overview = store.get_tree_overview()
        assert "After" in overview
        assert "Before" not in overview

    def test_remove_invalidates_cache(self, store):
        eid = store.add(path="a", title="Gone")
        store.get_tree_overview()
        store.remove(eid)
        overview = store.get_tree_overview()
        assert "Gone" not in overview

    def test_clear_invalidates_cache(self, store):
        store.add(path="a", title="X")
        store.get_tree_overview()
        store.clear()
        assert store.get_tree_overview() == "No dummies yet."

    def test_repeated_read_uses_cache(self, store):
        store.add(path="a", title="Cached")
        o1 = store.get_tree_overview()
        o2 = store.get_tree_overview()
        assert o1 is o2  # Same object reference = cached


# ---------------------------------------------------------------------------
# Display filter
# ---------------------------------------------------------------------------

class TestDisplayFilter:
    def test_omits_internal_fields(self, store):
        eid = store.add(path="a", title="Visible", content="body")
        entry = store.get(eid)
        dd = store.to_display_dict(entry)
        assert "mutation_history" not in dd
        assert "created_at" not in dd
        assert "updated_at" not in dd
        assert dd["title"] == "Visible"
        assert dd["content"] == "body"
        assert dd["path"] == "a"

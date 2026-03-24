"""Tests for utils.stores.skills: SkillEntry defaults, SkillStore, persistence."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys_path = Path(__file__).parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from utils.stores.skills import SkillStore, SkillEntry, get_skill_store


# ---------------------------------------------------------------------------
# SkillEntry
# ---------------------------------------------------------------------------


class TestSkillEntry:
    def test_defaults(self):
        entry = SkillEntry(id="skill_0001", path="battle", name="Type Chart")
        assert entry.effectiveness == "medium"
        assert entry.source == "orchestrator"
        assert entry.importance == 3
        assert entry.mutation_history == []
        assert isinstance(entry.created_at, str)

    def test_title_synced_from_name(self):
        entry = SkillEntry(id="skill_0001", name="Fast Dialogue")
        assert entry.title == "Fast Dialogue"

    def test_name_synced_from_title(self):
        entry = SkillEntry(id="skill_0001", title="Warp Detection")
        assert entry.name == "Warp Detection"


# ---------------------------------------------------------------------------
# SkillStore basics
# ---------------------------------------------------------------------------


class TestSkillStore:
    def test_add_and_get(self, tmp_path):
        store = SkillStore(cache_dir=str(tmp_path))
        eid = store.add(path="navigation", name="Dialogue Clearing", description="Spam A to clear dialogue")
        assert eid == "skill_0001"
        entry = store.get(eid)
        assert entry.name == "Dialogue Clearing"
        assert entry.path == "navigation"

    def test_save_and_reload(self, tmp_path):
        store = SkillStore(cache_dir=str(tmp_path))
        store.add(path="battle", name="Type Advantage", description="Use SE moves", effectiveness="high", importance=5)
        store.add(path="navigation", name="Warp Usage", description="Use warps to travel")

        store2 = SkillStore(cache_dir=str(tmp_path))
        assert len(store2.entries) == 2
        assert store2.next_id == 3
        assert store2.get("skill_0001").effectiveness == "high"

    def test_tree_overview(self, tmp_path):
        store = SkillStore(cache_dir=str(tmp_path))
        store.add(path="navigation/dialogue", name="Fast Clear")
        store.add(path="battle", name="Type Chart")
        overview = store.get_tree_overview()
        assert "=== SKILL LIBRARY OVERVIEW ===" in overview
        assert "[skill_0001] Fast Clear" in overview
        assert "[skill_0002] Type Chart" in overview

    def test_empty_tree(self, tmp_path):
        store = SkillStore(cache_dir=str(tmp_path))
        assert store.get_tree_overview() == "No skills learned yet."

    def test_update(self, tmp_path):
        store = SkillStore(cache_dir=str(tmp_path))
        eid = store.add(path="battle", name="Old", effectiveness="low")
        store.update(eid, name="New", effectiveness="high")
        entry = store.get(eid)
        assert entry.name == "New"
        assert entry.effectiveness == "high"

    def test_remove(self, tmp_path):
        store = SkillStore(cache_dir=str(tmp_path))
        eid = store.add(path="x", name="Gone")
        store.remove(eid)
        assert store.get(eid) is None

    def test_display_dict_omits_internal_fields(self, tmp_path):
        store = SkillStore(cache_dir=str(tmp_path))
        eid = store.add(path="test", name="Vis", description="desc")
        dd = store.to_display_dict(store.get(eid))
        assert "mutation_history" not in dd
        assert dd["name"] == "Vis"


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------


class TestGetSkillStore:
    def test_returns_same_instance(self, tmp_path):
        import utils.stores.skills as skill_module

        old_store = skill_module._skill_store
        try:
            skill_module._skill_store = None
            with patch("utils.stores.skills.SkillStore", side_effect=lambda **kw: object()) as mock_cls:
                s1 = get_skill_store()
                s2 = get_skill_store()
                assert s1 is s2
                mock_cls.assert_called_once()
        finally:
            skill_module._skill_store = old_store

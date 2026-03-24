"""Tests for process_memory and process_skill endpoint dispatch functions in server.game_tools."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys_path = Path(__file__).parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from utils.stores.memory import Memory
from utils.stores.skills import SkillStore

# Required by process_memory_direct / process_skill_direct (mirrors agent tool contract)
_PROC_REASON = "Test: valid non-empty reasoning for store operation"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def memory_store(tmp_path):
    store = Memory(cache_dir=str(tmp_path))
    store.add(path="pokemon", title="Mudkip", content="Water starter", importance=5)
    store.add(path="events", title="Got Pokedex", content="From Birch", importance=3)
    return store


@pytest.fixture()
def skill_store(tmp_path):
    store = SkillStore(cache_dir=str(tmp_path))
    store.add(path="navigation", name="Dialogue Clear", title="Dialogue Clear", description="Spam A", effectiveness="high", importance=4)
    return store


@pytest.fixture()
def game_tools_memory(memory_store, monkeypatch):
    """Import game_tools with patched memory_store singleton."""
    import server.game_tools as gt
    monkeypatch.setattr(gt, "memory_store", memory_store)
    return gt


@pytest.fixture()
def game_tools_skill(skill_store, monkeypatch):
    """Import game_tools with patched skill_store singleton."""
    import server.game_tools as gt
    monkeypatch.setattr(gt, "skill_store", skill_store)
    return gt


# ---------------------------------------------------------------------------
# process_memory — per action
# ---------------------------------------------------------------------------


class TestProcessMemoryRequiresReasoning:
    def test_missing_reasoning_fails(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("read", [{"id": "mem_0001"}], "")
        assert result["success"] is False
        assert "reasoning" in result["error"].lower()

    def test_none_reasoning_fails(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("read", [{"id": "mem_0001"}], None)
        assert result["success"] is False

    def test_whitespace_only_reasoning_fails(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("read", [{"id": "mem_0001"}], "   \t\n")
        assert result["success"] is False


class TestProcessSkillRequiresReasoning:
    def test_missing_reasoning_fails(self, game_tools_skill):
        result = game_tools_skill.process_skill_direct("read", [{"id": "skill_0001"}], "")
        assert result["success"] is False
        assert "reasoning" in result["error"].lower()


class TestProcessMemoryRead:
    def test_read_single(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("read", [{"id": "mem_0001"}], _PROC_REASON)
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["success"] is True
        entry = result["results"][0]["entry"]
        assert entry["title"] == "Mudkip"
        assert "mutation_history" not in entry

    def test_read_missing_id(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("read", [{"id": "mem_9999"}], _PROC_REASON)
        assert result["results"][0]["success"] is False

    def test_read_no_id_field(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("read", [{}], _PROC_REASON)
        assert result["results"][0]["success"] is False
        assert "Missing" in result["results"][0]["error"]


class TestProcessMemoryAdd:
    def test_add_rejects_empty_title_and_content(self, game_tools_memory, memory_store):
        n_before = len(memory_store.entries)
        result = game_tools_memory.process_memory_direct("add", [{}], _PROC_REASON)
        assert result["success"] is True
        assert result["results"][0]["success"] is False
        assert "non-empty" in result["results"][0]["error"].lower()
        assert len(memory_store.entries) == n_before

    def test_add_single(self, game_tools_memory, memory_store):
        result = game_tools_memory.process_memory_direct("add", [
            {"path": "items", "title": "Potion", "content": "Found on Route 101", "importance": 2}
        ], _PROC_REASON)
        assert result["success"] is True
        entry_id = result["results"][0]["entry_id"]
        assert entry_id == "mem_0003"
        assert memory_store.get(entry_id).title == "Potion"

    def test_add_batch(self, game_tools_memory, memory_store):
        result = game_tools_memory.process_memory_direct("add", [
            {"path": "a", "title": "A", "content": "a"},
            {"path": "b", "title": "B", "content": "b"},
        ], _PROC_REASON)
        assert all(r["success"] for r in result["results"])
        assert len(memory_store.entries) == 4

    def test_add_with_coordinates_string(self, game_tools_memory, memory_store):
        result = game_tools_memory.process_memory_direct("add", [
            {"path": "loc", "title": "Gym", "content": "gym", "coordinates": "10, 20"}
        ], _PROC_REASON)
        entry_id = result["results"][0]["entry_id"]
        assert memory_store.get(entry_id).coordinates == (10, 20)


class TestProcessMemoryUpdate:
    def test_update_single(self, game_tools_memory, memory_store):
        result = game_tools_memory.process_memory_direct("update", [
            {"id": "mem_0001", "title": "Marshtomp"}
        ], _PROC_REASON)
        assert result["results"][0]["success"] is True
        assert memory_store.get("mem_0001").title == "Marshtomp"

    def test_update_missing(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("update", [{"id": "mem_9999", "title": "nope"}], _PROC_REASON)
        assert result["results"][0]["success"] is False

    def test_update_no_id(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("update", [{"title": "missing id"}], _PROC_REASON)
        assert result["results"][0]["success"] is False


class TestProcessMemoryDelete:
    def test_delete_single(self, game_tools_memory, memory_store):
        result = game_tools_memory.process_memory_direct("delete", [{"id": "mem_0002"}], _PROC_REASON)
        assert result["results"][0]["success"] is True
        assert memory_store.get("mem_0002") is None

    def test_delete_missing(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("delete", [{"id": "mem_9999"}], _PROC_REASON)
        assert result["results"][0]["success"] is False


class TestProcessMemoryUnknownAction:
    def test_unknown_action(self, game_tools_memory):
        result = game_tools_memory.process_memory_direct("explode", [{}], _PROC_REASON)
        assert result["results"][0]["success"] is False
        assert "Unknown action" in result["results"][0]["error"]


# ---------------------------------------------------------------------------
# process_skill — per action
# ---------------------------------------------------------------------------


class TestProcessSkillRead:
    def test_read_single(self, game_tools_skill):
        result = game_tools_skill.process_skill_direct("read", [{"id": "skill_0001"}], _PROC_REASON)
        assert result["success"] is True
        assert result["results"][0]["entry"]["name"] == "Dialogue Clear"

    def test_read_missing(self, game_tools_skill):
        result = game_tools_skill.process_skill_direct("read", [{"id": "skill_9999"}], _PROC_REASON)
        assert result["results"][0]["success"] is False


class TestProcessSkillAdd:
    def test_add(self, game_tools_skill, skill_store):
        result = game_tools_skill.process_skill_direct("add", [
            {"path": "battle", "name": "Type Chart", "description": "Matchups", "effectiveness": "high"}
        ], _PROC_REASON)
        assert result["results"][0]["success"] is True
        eid = result["results"][0]["entry_id"]
        assert skill_store.get(eid).name == "Type Chart"


class TestProcessSkillUpdate:
    def test_update(self, game_tools_skill, skill_store):
        result = game_tools_skill.process_skill_direct("update", [
            {"id": "skill_0001", "effectiveness": "low"}
        ], _PROC_REASON)
        assert result["results"][0]["success"] is True
        assert skill_store.get("skill_0001").effectiveness == "low"


class TestProcessSkillDelete:
    def test_delete(self, game_tools_skill, skill_store):
        result = game_tools_skill.process_skill_direct("delete", [{"id": "skill_0001"}], _PROC_REASON)
        assert result["results"][0]["success"] is True
        assert skill_store.get("skill_0001") is None


# ---------------------------------------------------------------------------
# Overview endpoints
# ---------------------------------------------------------------------------


class TestOverviewEndpoints:
    def test_memory_overview(self, game_tools_memory):
        result = game_tools_memory.get_memory_overview_direct()
        assert result["success"] is True
        assert "LONG-TERM MEMORY OVERVIEW" in result["overview"]
        assert "[mem_0001] Mudkip" in result["overview"]

    def test_skill_overview(self, game_tools_skill):
        result = game_tools_skill.get_skill_overview_direct()
        assert result["success"] is True
        assert "SKILL LIBRARY OVERVIEW" in result["overview"]
        assert "[skill_0001] Dialogue Clear" in result["overview"]


# ---------------------------------------------------------------------------
# Backward-compat: add_memory_direct, search_memory_direct
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_add_memory_direct_rejects_empty_title_and_content(self, game_tools_memory, memory_store):
        n_before = len(memory_store.entries)
        result = game_tools_memory.add_memory_direct(category="x", title="   ", content="")
        assert result["success"] is False
        assert "empty" in result["error"].lower()
        assert len(memory_store.entries) == n_before

    def test_add_memory_direct_with_category(self, game_tools_memory, memory_store):
        result = game_tools_memory.add_memory_direct(category="strategy", title="Grind", content="Level up")
        assert result["success"] is True
        entry = memory_store.get(result["entry_id"])
        assert entry.path == "strategy"

    def test_add_memory_direct_with_path(self, game_tools_memory, memory_store):
        result = game_tools_memory.add_memory_direct(path="items/potions", title="Potion", content="Heal 20HP")
        assert result["success"] is True
        entry = memory_store.get(result["entry_id"])
        assert entry.path == "items/potions"

    def test_search_memory_direct(self, game_tools_memory):
        result = game_tools_memory.search_memory_direct(query="Mudkip")
        assert result["success"] is True
        assert result["count"] == 1

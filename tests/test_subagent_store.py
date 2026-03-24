"""Tests for utils.stores.subagents — SubagentEntry serde, CRUD, seeding,
tree overview, delete protection, and char-cap validation."""

import json
import pytest
from dataclasses import asdict

from utils.stores.subagents import (
    BUILTIN_SUBAGENT_CONFIGS,
    MAX_DIRECTIVE_LEN,
    MAX_INSTRUCTIONS_LEN,
    SubagentEntry,
    SubagentStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    return SubagentStore(cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# SubagentEntry serde
# ---------------------------------------------------------------------------

class TestSubagentEntry:
    def test_defaults(self):
        e = SubagentEntry()
        assert e.handler_type == "looping"
        assert e.max_turns == 25
        assert e.available_tools == []
        assert e.is_builtin is False
        assert e.source == "orchestrator"

    def test_name_title_sync_from_name(self):
        e = SubagentEntry(name="Foo")
        assert e.title == "Foo"

    def test_name_title_sync_from_title(self):
        e = SubagentEntry(title="Bar")
        assert e.name == "Bar"

    def test_created_at_auto(self):
        e = SubagentEntry()
        assert e.created_at is not None
        assert e.updated_at == e.created_at

    def test_roundtrip_via_asdict(self):
        e = SubagentEntry(
            id="sa_0001", name="Test", path="custom",
            available_tools=["press_buttons"],
        )
        d = asdict(e)
        e2 = SubagentEntry(**d)
        assert e2.id == "sa_0001"
        assert e2.available_tools == ["press_buttons"]


# ---------------------------------------------------------------------------
# Built-in seeding
# ---------------------------------------------------------------------------

class TestBuiltinSeeding:
    def test_seeds_on_empty_store(self, store):
        builtin_count = len(BUILTIN_SUBAGENT_CONFIGS)
        builtin_entries = [
            e for e in store.entries.values()
            if getattr(e, "is_builtin", False)
        ]
        assert len(builtin_entries) == builtin_count

    def test_does_not_double_seed(self, tmp_path):
        s1 = SubagentStore(cache_dir=str(tmp_path))
        count1 = len(s1.entries)
        s2 = SubagentStore(cache_dir=str(tmp_path))
        assert len(s2.entries) == count1

    def test_builtin_entries_are_read_only_records(self, store):
        for e in store.entries.values():
            if e.is_builtin:
                assert e.source == "built-in"
                assert e.importance >= 4


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_add_custom_subagent(self, store):
        eid = store.add(
            path="custom/battle", name="MyBattler",
            description="Custom battler", handler_type="looping",
            available_tools=["press_buttons"], importance=4,
        )
        assert eid.startswith("sa_")
        entry = store.get(eid)
        assert entry.name == "MyBattler"
        assert entry.handler_type == "looping"

    def test_read_custom_subagent(self, store):
        eid = store.add(name="Reader", directive="Read things")
        entry = store.get(eid)
        assert entry.directive == "Read things"

    def test_update_custom_subagent(self, store):
        eid = store.add(name="Updatable", directive="Old")
        ok = store.update(eid, directive="New")
        assert ok is True
        assert store.get(eid).directive == "New"

    def test_delete_custom_subagent(self, store):
        eid = store.add(name="Deletable")
        assert store.remove(eid) is True
        assert store.get(eid) is None


# ---------------------------------------------------------------------------
# Delete protection for built-ins
# ---------------------------------------------------------------------------

class TestDeleteProtection:
    def test_cannot_delete_builtin(self, store):
        builtin_id = next(
            eid for eid, e in store.entries.items() if e.is_builtin
        )
        assert store.remove(builtin_id) is False
        assert store.get(builtin_id) is not None

    def test_can_update_builtin_description(self, store):
        """Built-ins can be updated (e.g. description) but not deleted."""
        builtin_id = next(
            eid for eid, e in store.entries.items() if e.is_builtin
        )
        ok = store.update(builtin_id, description="Updated desc")
        assert ok is True
        assert store.get(builtin_id).description == "Updated desc"


# ---------------------------------------------------------------------------
# Tree overview
# ---------------------------------------------------------------------------

class TestTreeOverview:
    def test_overview_shows_builtins(self, store):
        overview = store.get_tree_overview()
        assert "SUBAGENT REGISTRY" in overview
        assert "built-in" in overview

    def test_overview_shows_custom(self, store):
        store.add(path="custom", name="MyAgent")
        overview = store.get_tree_overview()
        assert "MyAgent" in overview

    def test_empty_custom_store_shows_only_builtins(self, tmp_path):
        s = SubagentStore(cache_dir=str(tmp_path))
        overview = s.get_tree_overview()
        assert "Reflect" in overview
        assert "Battler" in overview


# ---------------------------------------------------------------------------
# Display dict (excludes system_instructions, directive)
# ---------------------------------------------------------------------------

class TestDisplayDict:
    def test_excludes_sensitive_fields(self, store):
        eid = store.add(
            name="Secret", system_instructions="big prompt",
            directive="do something",
        )
        dd = store.to_display_dict(store.get(eid))
        assert "system_instructions" not in dd
        assert "directive" not in dd
        assert dd["name"] == "Secret"

    def test_excludes_internal_fields(self, store):
        eid = store.add(name="Clean")
        dd = store.to_display_dict(store.get(eid))
        assert "mutation_history" not in dd
        assert "created_at" not in dd
        assert "updated_at" not in dd


# ---------------------------------------------------------------------------
# Char-cap validation
# ---------------------------------------------------------------------------

class TestCharCap:
    def test_add_rejects_oversized_instructions(self, store):
        with pytest.raises(ValueError, match="system_instructions"):
            store.add(
                name="TooLong",
                system_instructions="x" * (MAX_INSTRUCTIONS_LEN + 1),
            )

    def test_add_rejects_oversized_directive(self, store):
        with pytest.raises(ValueError, match="directive"):
            store.add(
                name="TooLong",
                directive="y" * (MAX_DIRECTIVE_LEN + 1),
            )

    def test_update_rejects_oversized_instructions(self, store):
        eid = store.add(name="Ok", system_instructions="short")
        with pytest.raises(ValueError, match="system_instructions"):
            store.update(eid, system_instructions="x" * (MAX_INSTRUCTIONS_LEN + 1))

    def test_update_rejects_oversized_directive(self, store):
        eid = store.add(name="Ok", directive="short")
        with pytest.raises(ValueError, match="directive"):
            store.update(eid, directive="y" * (MAX_DIRECTIVE_LEN + 1))

    def test_add_accepts_at_limit(self, store):
        eid = store.add(
            name="Exact",
            system_instructions="x" * MAX_INSTRUCTIONS_LEN,
            directive="y" * MAX_DIRECTIVE_LEN,
        )
        assert store.get(eid) is not None


# ---------------------------------------------------------------------------
# Persistence roundtrip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        s1 = SubagentStore(cache_dir=str(tmp_path))
        s1.add(name="Persisted", directive="remember me", importance=5)
        custom_count = len(s1.entries)

        s2 = SubagentStore(cache_dir=str(tmp_path))
        assert len(s2.entries) == custom_count
        custom = [e for e in s2.entries.values() if not e.is_builtin]
        assert any(e.name == "Persisted" for e in custom)

    def test_deserialization_defaults(self, tmp_path):
        """Entries missing optional fields get defaults from _deserialize_entry."""
        raw = {
            "entries": {
                "sa_0099": {
                    "id": "sa_0099",
                    "name": "Minimal",
                    "is_builtin": True,
                }
            },
            "next_id": 100,
        }
        (tmp_path / "subagents.json").write_text(json.dumps(raw))
        s = SubagentStore(cache_dir=str(tmp_path))
        entry = s.get("sa_0099")
        assert entry is not None
        assert entry.handler_type == "looping"
        assert entry.max_turns == 25
        assert entry.available_tools == []
        assert entry.title == "Minimal"

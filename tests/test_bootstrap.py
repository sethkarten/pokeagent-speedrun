"""Tests for the bootstrap infrastructure — import, export, round-trip, and prompt seeding."""

import json
import os
import shutil

import pytest

from utils.stores.bootstrap import (
    PROMPT_CANONICAL,
    SANITIZED_PROMPT_FILENAME,
    _extract_numeric_id,
    _merge_into_target,
    _normalize_prompt_text,
    _repath_entries,
    _resolve_prompt,
    bootstrap_stores,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store_json(entries: dict, next_id: int = 1) -> dict:
    return {"next_id": next_id, "entries": entries}


def _write_store(path, entries: dict, next_id: int = 1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_make_store_json(entries, next_id), f)


def _sample_memory_entry(eid="mem_0001", path="pokemon/team", title="My Team",
                          source="orchestrator"):
    return {
        "id": eid,
        "path": path,
        "title": title,
        "content": "Mudkip lv 10",
        "importance": 4,
        "source": source,
        "created_at": "2026-04-01T00:00:00",
        "updated_at": "2026-04-01T00:00:00",
        "mutation_history": [{"timestamp": "t", "fields": {"title": {"old": "x", "new": "y"}}}],
        "tags": [],
    }


def _sample_skill_entry(eid="skill_0001", path="navigation", name="pathfinder"):
    return {
        "id": eid,
        "path": path,
        "name": name,
        "title": name,
        "description": "BFS pathfinding",
        "code": "result = 42",
        "effectiveness": "high",
        "source": "evolved",
        "importance": 5,
        "created_at": "2026-04-01T00:00:00",
        "updated_at": "2026-04-01T00:00:00",
        "mutation_history": [],
    }


def _sample_subagent_entry(eid="sa_0001", path="custom", name="battler"):
    return {
        "id": eid,
        "path": path,
        "name": name,
        "title": name,
        "description": "Handles battles",
        "handler_type": "looping",
        "max_turns": 25,
        "available_tools": ["press_buttons"],
        "system_instructions": "Fight well.",
        "directive": "",
        "return_condition": "battle_over",
        "importance": 4,
        "source": "orchestrator",
        "is_builtin": False,
        "created_at": "2026-04-01T00:00:00",
        "updated_at": "2026-04-01T00:00:00",
        "mutation_history": [],
    }


# ===========================================================================
# TestBootstrapImport
# ===========================================================================

class TestBootstrapImport:
    """Tests for bootstrap_stores() import function."""

    def test_happy_path(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)
        _write_store(str(src / "skills.json"),
                     {"skill_0001": _sample_skill_entry()}, next_id=2)
        _write_store(str(src / "subagents.json"),
                     {"sa_0001": _sample_subagent_entry()}, next_id=2)

        target = tmp_path / "cache"
        target.mkdir()
        result = bootstrap_stores(str(src), str(target))

        assert result["memory"] == 1
        assert result["skills"] == 1
        assert result["subagents"] == 1

        for fname in ["memory.json", "skills.json", "subagents.json"]:
            with open(target / fname) as f:
                data = json.load(f)
            entries = data["entries"]
            assert len(entries) == 1
            entry = next(iter(entries.values()))
            assert entry["path"].startswith("bootstrapped/")
            assert entry["source"] == "bootstrapped"

    def test_missing_store_file(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)
        # No skills or subagents

        target = tmp_path / "cache"
        target.mkdir()
        result = bootstrap_stores(str(src), str(target))

        assert result["memory"] == 1
        assert result["skills"] == 0
        assert result["subagents"] == 0
        assert not (target / "skills.json").exists()

    def test_empty_source_dir_exits(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        target = tmp_path / "cache"
        target.mkdir()
        with pytest.raises(SystemExit):
            bootstrap_stores(str(src), str(target))

    def test_nonexistent_source_dir_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            bootstrap_stores(str(tmp_path / "nope"), str(tmp_path / "cache"))

    def test_repath_single_prefix(self):
        data = _make_store_json({"e1": {"path": "navigation/pathfinding", "source": "evolved"}})
        _repath_entries(data)
        assert data["entries"]["e1"]["path"] == "bootstrapped/navigation/pathfinding"
        assert data["entries"]["e1"]["source"] == "bootstrapped"

    def test_repath_double_prefix(self):
        """Multi-generation bootstrap: already-bootstrapped entries get double-prefixed."""
        data = _make_store_json({"e1": {"path": "bootstrapped/nav", "source": "bootstrapped"}})
        _repath_entries(data)
        assert data["entries"]["e1"]["path"] == "bootstrapped/bootstrapped/nav"

    def test_repath_empty_path(self):
        data = _make_store_json({"e1": {"path": "", "source": "orchestrator"}})
        _repath_entries(data)
        assert data["entries"]["e1"]["path"] == "bootstrapped"

    def test_preserves_mutation_history(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        entry = _sample_memory_entry()
        entry["mutation_history"] = [{"timestamp": "t1", "fields": {"x": {"old": 1, "new": 2}}}]
        _write_store(str(src / "memory.json"), {"mem_0001": entry}, next_id=2)

        target = tmp_path / "cache"
        target.mkdir()
        bootstrap_stores(str(src), str(target))

        with open(target / "memory.json") as f:
            data = json.load(f)
        assert len(data["entries"]["mem_0001"]["mutation_history"]) == 1

    def test_next_id_reconciliation(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "skills.json"),
                     {"skill_0042": _sample_skill_entry(eid="skill_0042")}, next_id=43)

        target = tmp_path / "cache"
        target.mkdir()
        # Pre-existing store in target with lower next_id
        _write_store(str(target / "skills.json"), {}, next_id=5)

        bootstrap_stores(str(src), str(target))

        with open(target / "skills.json") as f:
            data = json.load(f)
        assert data["next_id"] >= 43

    def test_merge_with_existing_entries(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)

        target = tmp_path / "cache"
        target.mkdir()
        existing = _sample_memory_entry(eid="mem_0099", path="routes", title="Route 101")
        existing["source"] = "orchestrator"
        _write_store(str(target / "memory.json"), {"mem_0099": existing}, next_id=100)

        bootstrap_stores(str(src), str(target))

        with open(target / "memory.json") as f:
            data = json.load(f)
        assert len(data["entries"]) == 2
        assert "mem_0099" in data["entries"]
        assert "mem_0001" in data["entries"]
        # Existing entry should NOT be re-pathed
        assert data["entries"]["mem_0099"]["source"] == "orchestrator"
        assert data["entries"]["mem_0001"]["source"] == "bootstrapped"

    def test_id_collision_prefix(self, tmp_path):
        """When a bootstrapped entry ID already exists in target, prefix with bs_."""
        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)

        target = tmp_path / "cache"
        target.mkdir()
        existing = _sample_memory_entry(eid="mem_0001", path="other", title="Existing")
        _write_store(str(target / "memory.json"), {"mem_0001": existing}, next_id=2)

        bootstrap_stores(str(src), str(target))

        with open(target / "memory.json") as f:
            data = json.load(f)
        assert len(data["entries"]) == 2
        assert "mem_0001" in data["entries"]
        assert "bs_mem_0001" in data["entries"]
        assert data["entries"]["bs_mem_0001"]["source"] == "bootstrapped"

    def test_prompt_canonical_found(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)
        (src / PROMPT_CANONICAL).write_text("# Evolved policy")

        target = tmp_path / "cache"
        target.mkdir()
        result = bootstrap_stores(str(src), str(target))
        assert result["prompt_path"] == str(target / SANITIZED_PROMPT_FILENAME)
        assert (target / SANITIZED_PROMPT_FILENAME).read_text().strip() == "# Evolved policy"

    def test_prompt_steps_fallback(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)
        (src / "steps_1_to_50.md").write_text("# old")
        (src / "steps_51_to_100.md").write_text("# latest")

        target = tmp_path / "cache"
        target.mkdir()
        result = bootstrap_stores(str(src), str(target))
        assert result["prompt_path"] == str(target / SANITIZED_PROMPT_FILENAME)
        assert (target / SANITIZED_PROMPT_FILENAME).read_text().strip() == "# latest"

    def test_no_prompt(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)

        target = tmp_path / "cache"
        target.mkdir()
        result = bootstrap_stores(str(src), str(target))
        assert result["prompt_path"] is None
        assert not (target / SANITIZED_PROMPT_FILENAME).exists()

    def test_prompt_sanitization_uses_single_vlm_call(self, tmp_path):
        from unittest.mock import MagicMock, patch

        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)
        (src / PROMPT_CANONICAL).write_text(
            "IMPROVED BASE PROMPT:\n\n"
            "# Strategic Guidance\n\n"
            "## Current Strategy\n"
            "- Finish the battle on Route 114.\n"
        )

        target = tmp_path / "cache"
        target.mkdir()

        fake_vlm = MagicMock()
        fake_vlm.get_text_query.return_value = "# Strategic Guidance\n\nGeneral reusable guidance."

        with patch("utils.agent_infrastructure.vlm_backends.VLM", return_value=fake_vlm) as mocked_vlm:
            result = bootstrap_stores(
                str(src),
                str(target),
                prompt_backend="gemini",
                prompt_model_name="gemini-2.5-flash",
            )

        assert result["prompt_path"] == str(target / SANITIZED_PROMPT_FILENAME)
        assert (target / SANITIZED_PROMPT_FILENAME).read_text().strip() == (
            "# Strategic Guidance\n\nGeneral reusable guidance."
        )
        mocked_vlm.assert_called_once()
        fake_vlm.get_text_query.assert_called_once()

    def test_prompt_sanitization_falls_back_to_normalized_original(self, tmp_path):
        from unittest.mock import patch

        src = tmp_path / "source"
        src.mkdir()
        _write_store(str(src / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)
        (src / PROMPT_CANONICAL).write_text(
            "IMPROVED BASE PROMPT:\n\n# Strategic Guidance\n\nKeep this."
        )

        target = tmp_path / "cache"
        target.mkdir()

        with patch("utils.agent_infrastructure.vlm_backends.VLM", side_effect=RuntimeError("boom")):
            result = bootstrap_stores(
                str(src),
                str(target),
                prompt_backend="gemini",
                prompt_model_name="gemini-2.5-flash",
            )

        assert result["prompt_path"] == str(target / SANITIZED_PROMPT_FILENAME)
        assert (target / SANITIZED_PROMPT_FILENAME).read_text().strip() == (
            "# Strategic Guidance\n\nKeep this."
        )


# ===========================================================================
# TestBootstrapExport
# ===========================================================================

class TestBootstrapExport:
    """Tests for _export_bootstrap_artifacts() in RunDataManager."""

    def _make_run_env(self, tmp_path):
        """Set up a mock RunDataManager environment."""
        run_dir = tmp_path / "run_data" / "test_run"
        cache_dir = tmp_path / "cache"

        game_state_dir = run_dir / "end_state" / "game_state"
        game_state_dir.mkdir(parents=True)
        (run_dir / "prompt_evolution" / "meta_prompts").mkdir(parents=True)

        cache_dir.mkdir(parents=True)

        # Write some store files in cache
        for fname, entries in [
            ("memory.json", {"mem_0001": _sample_memory_entry()}),
            ("skills.json", {"skill_0001": _sample_skill_entry()}),
            ("subagents.json", {"sa_0001": _sample_subagent_entry()}),
        ]:
            _write_store(str(cache_dir / fname), entries, next_id=2)

        return run_dir, cache_dir

    def test_creates_bootstrap_directory(self, tmp_path, monkeypatch):
        run_dir, cache_dir = self._make_run_env(tmp_path)

        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_path",
            lambda name: cache_dir / name,
        )
        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_directory",
            lambda: cache_dir,
        )

        from utils.data_persistence.run_data_manager import RunDataManager

        mgr = RunDataManager.__new__(RunDataManager)
        mgr.run_dir = run_dir
        mgr.run_id = "test_run"
        mgr._export_bootstrap_artifacts()

        bs_dir = run_dir / "end_state" / "game_state" / "bootstrap"
        assert bs_dir.exists()
        assert (bs_dir / "memory.json").exists()
        assert (bs_dir / "skills.json").exists()
        assert (bs_dir / "subagents.json").exists()

        cache_bs = cache_dir / "bootstrap"
        assert cache_bs.exists()
        assert (cache_bs / "memory.json").exists()

    def test_copies_latest_evolved_prompt(self, tmp_path, monkeypatch):
        run_dir, cache_dir = self._make_run_env(tmp_path)
        meta_dir = run_dir / "prompt_evolution" / "meta_prompts"
        (meta_dir / "steps_1_to_50.md").write_text("# old")
        import time; time.sleep(0.05)
        (meta_dir / "steps_51_to_100.md").write_text("# latest")

        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_path",
            lambda name: cache_dir / name,
        )
        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_directory",
            lambda: cache_dir,
        )

        from utils.data_persistence.run_data_manager import RunDataManager
        mgr = RunDataManager.__new__(RunDataManager)
        mgr.run_dir = run_dir
        mgr.run_id = "test_run"
        mgr._export_bootstrap_artifacts()

        bs_dir = run_dir / "end_state" / "game_state" / "bootstrap"
        policy = (bs_dir / "EVOLVED_ORCHESTRATOR_POLICY.md").read_text()
        assert "latest" in policy

    def test_no_evolved_prompt_still_exports_stores(self, tmp_path, monkeypatch):
        run_dir, cache_dir = self._make_run_env(tmp_path)
        # No meta_prompts files

        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_path",
            lambda name: cache_dir / name,
        )
        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_directory",
            lambda: cache_dir,
        )

        from utils.data_persistence.run_data_manager import RunDataManager
        mgr = RunDataManager.__new__(RunDataManager)
        mgr.run_dir = run_dir
        mgr.run_id = "test_run"
        mgr._export_bootstrap_artifacts()

        bs_dir = run_dir / "end_state" / "game_state" / "bootstrap"
        assert (bs_dir / "memory.json").exists()
        assert not (bs_dir / "EVOLVED_ORCHESTRATOR_POLICY.md").exists()

    def test_dual_write(self, tmp_path, monkeypatch):
        run_dir, cache_dir = self._make_run_env(tmp_path)

        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_path",
            lambda name: cache_dir / name,
        )
        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_directory",
            lambda: cache_dir,
        )

        from utils.data_persistence.run_data_manager import RunDataManager
        mgr = RunDataManager.__new__(RunDataManager)
        mgr.run_dir = run_dir
        mgr.run_id = "test_run"
        mgr._export_bootstrap_artifacts()

        run_bs = run_dir / "end_state" / "game_state" / "bootstrap" / "skills.json"
        cache_bs = cache_dir / "bootstrap" / "skills.json"
        assert run_bs.exists()
        assert cache_bs.exists()
        assert run_bs.read_text() == cache_bs.read_text()


# ===========================================================================
# TestBootstrapRoundTrip
# ===========================================================================

class TestBootstrapRoundTrip:
    """Export then import — verify entries survive the full pipeline."""

    def test_export_then_import(self, tmp_path, monkeypatch):
        # 1) Set up a "source run" with stores and an evolved prompt
        run_dir = tmp_path / "run_data" / "source_run"
        cache_dir = tmp_path / "source_cache"
        cache_dir.mkdir(parents=True)
        (run_dir / "end_state" / "game_state").mkdir(parents=True)
        (run_dir / "prompt_evolution" / "meta_prompts").mkdir(parents=True)

        _write_store(str(cache_dir / "memory.json"),
                     {"mem_0001": _sample_memory_entry()}, next_id=2)
        _write_store(str(cache_dir / "skills.json"),
                     {"pathfinder": _sample_skill_entry(eid="pathfinder")}, next_id=1)

        meta = run_dir / "prompt_evolution" / "meta_prompts" / "steps_1_to_50.md"
        meta.write_text("# Evolved guidance\nDo the thing.")

        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_path",
            lambda name: cache_dir / name,
        )
        monkeypatch.setattr(
            "utils.data_persistence.run_data_manager.get_cache_directory",
            lambda: cache_dir,
        )

        # 2) Export
        from utils.data_persistence.run_data_manager import RunDataManager
        mgr = RunDataManager.__new__(RunDataManager)
        mgr.run_dir = run_dir
        mgr.run_id = "source_run"
        mgr._export_bootstrap_artifacts()

        # 3) Import into a fresh target cache
        export_dir = run_dir / "end_state" / "game_state" / "bootstrap"
        target_cache = tmp_path / "new_run_cache"
        target_cache.mkdir()

        result = bootstrap_stores(str(export_dir), str(target_cache))

        assert result["memory"] == 1
        assert result["skills"] == 1
        assert result["prompt_path"] is not None

        # 4) Verify entries
        with open(target_cache / "memory.json") as f:
            mem_data = json.load(f)
        mem = mem_data["entries"]["mem_0001"]
        assert mem["path"] == "bootstrapped/pokemon/team"
        assert mem["source"] == "bootstrapped"
        assert len(mem["mutation_history"]) == 1

        with open(target_cache / "skills.json") as f:
            skill_data = json.load(f)
        sk = skill_data["entries"]["pathfinder"]
        assert sk["path"] == "bootstrapped/navigation"
        assert sk["code"] == "result = 42"

        # 5) Verify prompt
        prompt_content = open(result["prompt_path"]).read()
        assert "Evolved guidance" in prompt_content


# ===========================================================================
# TestPromptSeedOverride
# ===========================================================================

class TestPromptSeedOverride:
    """Tests for initial_prompt_override in PromptOptimizer."""

    def test_override_takes_precedence(self, tmp_path):
        from unittest.mock import MagicMock, patch

        seed_file = tmp_path / "seed.md"
        seed_file.write_text("# Seed prompt\nOriginal seed.")

        vlm = MagicMock()
        vlm.backend_type = "gemini"
        vlm.model_name = "gemini-2.5-flash"

        with patch("utils.agent_infrastructure.vlm_backends.VLM"):
            from agents.utils.prompt_optimizer import PromptOptimizer
            opt = PromptOptimizer(
                vlm=vlm,
                run_data_manager=MagicMock(),
                base_prompt_path=str(seed_file),
                initial_prompt_override="# Bootstrapped\nEvolved strategy.",
            )
        assert opt.get_current_prompt() == "# Bootstrapped\nEvolved strategy."

    def test_fallback_to_seed_without_override(self, tmp_path):
        from unittest.mock import MagicMock, patch

        seed_file = tmp_path / "seed.md"
        seed_file.write_text("# Seed prompt")

        vlm = MagicMock()
        vlm.backend_type = "gemini"
        vlm.model_name = "gemini-2.5-flash"

        with patch("utils.agent_infrastructure.vlm_backends.VLM"):
            from agents.utils.prompt_optimizer import PromptOptimizer
            opt = PromptOptimizer(
                vlm=vlm,
                run_data_manager=MagicMock(),
                base_prompt_path=str(seed_file),
            )
        assert "Seed prompt" in opt.get_current_prompt()


# ===========================================================================
# TestPromptNormalization
# ===========================================================================

class TestPromptNormalization:
    def test_strips_optimizer_header(self):
        assert _normalize_prompt_text("IMPROVED BASE PROMPT:\n\n# Strategic Guidance") == (
            "# Strategic Guidance"
        )

    def test_keeps_regular_prompt(self):
        assert _normalize_prompt_text("# Strategic Guidance\nBody") == "# Strategic Guidance\nBody"


# ===========================================================================
# TestResolvePrompt
# ===========================================================================

class TestResolvePrompt:
    """Tests for the _resolve_prompt fallback chain."""

    def test_canonical_first(self, tmp_path):
        (tmp_path / PROMPT_CANONICAL).write_text("canonical")
        (tmp_path / "steps_1_to_50.md").write_text("fallback")
        assert _resolve_prompt(str(tmp_path)).endswith(PROMPT_CANONICAL)

    def test_steps_fallback(self, tmp_path):
        (tmp_path / "steps_1_to_50.md").write_text("v1")
        (tmp_path / "steps_51_to_100.md").write_text("v2")
        path = _resolve_prompt(str(tmp_path))
        assert path.endswith("steps_51_to_100.md")

    def test_none_when_empty(self, tmp_path):
        assert _resolve_prompt(str(tmp_path)) is None


# ===========================================================================
# TestExtractNumericId
# ===========================================================================

class TestExtractNumericId:

    def test_standard_ids(self):
        assert _extract_numeric_id("mem_0042") == 42
        assert _extract_numeric_id("skill_0001") == 1
        assert _extract_numeric_id("sa_0100") == 100

    def test_custom_ids(self):
        assert _extract_numeric_id("pathfinder") is None
        assert _extract_numeric_id("battle_handler") is None

    def test_mixed_ids(self):
        assert _extract_numeric_id("bs_mem_0005") == 5


# ===========================================================================
# TestRecencyTracking (base_store integration)
# ===========================================================================

class TestRecencyTracking:
    """Verify that recently-accessed entries influence tree overview ordering."""

    def test_recent_access_tracked(self):
        from tests.test_base_store import DummyStore
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            s = DummyStore(cache_dir=td)
            eid = s.add(path="a", title="One")
            s.get(eid)
            assert eid in s._recent_access_ids

    def test_recent_roots_promoted(self):
        from tests.test_base_store import DummyStore
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            s = DummyStore(cache_dir=td)
            s.add(path="alpha/sub", title="A1")
            s.add(path="beta/sub", title="B1")
            eid_z = s.add(path="zeta/sub", title="Z1")

            # Access zeta entry
            s._invalidate_cache()
            s.get(eid_z)
            overview = s.get_tree_overview()

            lines = overview.split("\n")
            path_lines = [l.strip().rstrip(":") for l in lines if l.strip() and not l.strip().startswith("-") and not l.strip().startswith("===")]
            # zeta should appear before alpha and beta
            if "zeta" in path_lines and "alpha" in path_lines:
                assert path_lines.index("zeta") < path_lines.index("alpha")

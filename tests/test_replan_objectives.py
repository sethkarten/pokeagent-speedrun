"""Unit tests for DirectObjectiveManager.replan_category(), get_full_sequence_snapshot(),
and the sequence-exhaustion fallback objective."""

import json
import pytest

from agents.objectives.objective_types import DirectObjective
from agents.objectives.direct_objectives import DirectObjectiveManager
from utils.json_utils import coerce_replan_edit_index, convert_protobuf_value, normalize_replan_edits


def _make_manager(
    story_objs=None, battling_objs=None, dynamics_objs=None,
    story_idx=0, battling_idx=0, dynamics_idx=0,
):
    """Helper to build a minimal DirectObjectiveManager in categorized mode."""
    mgr = DirectObjectiveManager()
    mgr.enable_categorized_mode()

    if story_objs:
        mgr.story_sequence = list(story_objs)
    if battling_objs:
        mgr.battling_sequence = list(battling_objs)
    if dynamics_objs:
        mgr.dynamics_sequence = list(dynamics_objs)

    mgr.story_index = story_idx
    mgr.battling_index = battling_idx
    mgr.dynamics_index = dynamics_idx
    return mgr


def _obj(id_: str, desc: str = "test", **kwargs) -> DirectObjective:
    return DirectObjective(
        id=id_,
        description=desc,
        action_type=kwargs.pop("action_type", "navigate"),
        category=kwargs.pop("category", "story"),
        **kwargs,
    )


# ---------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------


class TestReplanCategoryValidation:
    def test_invalid_category(self):
        mgr = _make_manager(story_objs=[_obj("s0")])
        result = mgr.replan_category("invalid", [])
        assert not result["success"]
        assert "Invalid category" in result["error"]

    def test_too_many_edits(self):
        mgr = _make_manager(story_objs=[_obj("s0")])
        edits = [{"index": i, "objective": {"id": f"x{i}", "description": "x", "action_type": "navigate"}} for i in range(6)]
        result = mgr.replan_category("story", edits)
        assert not result["success"]
        assert "Too many edits" in result["error"]

    def test_index_below_current(self):
        mgr = _make_manager(
            story_objs=[_obj("s0"), _obj("s1"), _obj("s2")],
            story_idx=2,
        )
        result = mgr.replan_category("story", [{"index": 1, "objective": {"id": "new", "description": "x", "action_type": "navigate"}}])
        assert not result["success"]
        assert "before the current index" in result["error"]

    def test_non_contiguous_append(self):
        mgr = _make_manager(story_objs=[_obj("s0"), _obj("s1")])
        result = mgr.replan_category("story", [
            {"index": 5, "objective": {"id": "gap", "description": "x", "action_type": "navigate"}},
        ])
        assert not result["success"]
        assert "Non-contiguous" in result["error"]

    def test_delete_nonexistent_index(self):
        mgr = _make_manager(story_objs=[_obj("s0")])
        result = mgr.replan_category("story", [{"index": 10, "objective": None}])
        assert not result["success"]
        assert "does not exist" in result["error"]

    def test_missing_index_field(self):
        mgr = _make_manager(story_objs=[_obj("s0")])
        result = mgr.replan_category("story", [{"objective": {"id": "x", "description": "x"}}])
        assert not result["success"]
        assert "integer 'index'" in result["error"]

    def test_edits_not_a_list(self):
        mgr = _make_manager(story_objs=[_obj("s0")])
        result = mgr.replan_category("story", "not-a-list")
        assert not result["success"]
        assert "must be a list" in result["error"]


# ---------------------------------------------------------------
# Successful operations
# ---------------------------------------------------------------


class TestReplanCategoryHappyPath:
    def test_modify_existing_objective(self):
        mgr = _make_manager(story_objs=[_obj("s0"), _obj("s1")])
        result = mgr.replan_category("story", [
            {"index": 1, "objective": {"id": "s1_modified", "description": "updated", "action_type": "interact"}},
        ])
        assert result["success"]
        assert result["edits_applied"] == [{"action": "modify", "index": 1, "id": "s1_modified"}]
        assert mgr.story_sequence[1].id == "s1_modified"
        assert mgr.story_sequence[1].action_type == "interact"

    def test_append_new_objective(self):
        mgr = _make_manager(story_objs=[_obj("s0")])
        result = mgr.replan_category("story", [
            {"index": 1, "objective": {"id": "s1_new", "description": "new obj", "action_type": "navigate"}},
        ])
        assert result["success"]
        assert result["new_sequence_length"] == 2
        assert result["edits_applied"][0]["action"] == "create"
        assert mgr.story_sequence[1].id == "s1_new"

    def test_append_multiple_contiguous(self):
        mgr = _make_manager(story_objs=[_obj("s0")])
        result = mgr.replan_category("story", [
            {"index": 1, "objective": {"id": "s1", "description": "one", "action_type": "navigate"}},
            {"index": 2, "objective": {"id": "s2", "description": "two", "action_type": "navigate"}},
        ])
        assert result["success"]
        assert result["new_sequence_length"] == 3

    def test_delete_existing_objective(self):
        mgr = _make_manager(story_objs=[_obj("s0"), _obj("s1"), _obj("s2")])
        result = mgr.replan_category("story", [{"index": 1, "objective": None}])
        assert result["success"]
        assert result["new_sequence_length"] == 2
        assert result["edits_applied"][0]["action"] == "delete"
        assert mgr.story_sequence[1].id == "s2"

    def test_delete_with_empty_dict(self):
        mgr = _make_manager(story_objs=[_obj("s0"), _obj("s1")])
        result = mgr.replan_category("story", [{"index": 1, "objective": {}}])
        assert result["success"]
        assert result["new_sequence_length"] == 1

    def test_no_op_empty_edits(self):
        mgr = _make_manager(story_objs=[_obj("s0"), _obj("s1")])
        result = mgr.replan_category("story", [])
        assert result["success"]
        assert result["edits_applied"] == []
        assert result["new_sequence_length"] == 2

    def test_mixed_operations(self):
        """Modify one objective, delete another, and append a new one."""
        mgr = _make_manager(
            story_objs=[_obj("s0"), _obj("s1"), _obj("s2")],
            story_idx=0,
        )
        result = mgr.replan_category("story", [
            {"index": 0, "objective": {"id": "s0_new", "description": "replaced", "action_type": "interact"}},
            {"index": 2, "objective": None},
            {"index": 2, "objective": {"id": "s2_appended", "description": "appended", "action_type": "navigate"}},
        ])
        assert result["success"]
        assert mgr.story_sequence[0].id == "s0_new"

    def test_replan_empty_dynamics_sequence(self):
        mgr = _make_manager()
        result = mgr.replan_category("dynamics", [
            {"index": 0, "objective": {"id": "d0", "description": "first dynamic", "action_type": "navigate"}},
        ])
        assert result["success"]
        assert result["new_sequence_length"] == 1
        assert mgr.dynamics_sequence[0].id == "d0"

    def test_replan_adds_objective_after_create_placeholder(self):
        mgr = _make_manager(
            story_objs=[
                _obj("auto_plan_objectives", action_type="create_new_objectives"),
            ],
            story_idx=0,
        )
        result = mgr.replan_category("story", [
            {
                "index": 1,
                "objective": {
                    "id": "story_advance_after_planning",
                    "description": "Advance story after planning",
                    "action_type": "navigate",
                },
            },
        ])
        assert result["success"] is True
        assert mgr.story_sequence[1].id == "story_advance_after_planning"

    def test_delete_current_objective_adjusts_index(self):
        mgr = _make_manager(
            story_objs=[_obj("s0"), _obj("s1"), _obj("s2")],
            story_idx=1,
        )
        result = mgr.replan_category("story", [{"index": 1, "objective": None}])
        assert result["success"]
        assert mgr.story_index <= len(mgr.story_sequence) - 1

    def test_battling_category(self):
        mgr = _make_manager(battling_objs=[_obj("b0", category="battling")])
        result = mgr.replan_category("battling", [
            {"index": 0, "objective": {"id": "b0_new", "description": "updated battling", "action_type": "battle"}},
        ])
        assert result["success"]
        assert mgr.battling_sequence[0].id == "b0_new"


# ---------------------------------------------------------------
# get_full_sequence_snapshot
# ---------------------------------------------------------------


class TestGetFullSequenceSnapshot:
    def test_snapshot_structure(self):
        mgr = _make_manager(
            story_objs=[_obj("s0"), _obj("s1", completed=True)],
            battling_objs=[_obj("b0", category="battling")],
            story_idx=1,
        )
        mgr.story_sequence[1].completed = True
        snap = mgr.get_full_sequence_snapshot()

        assert snap["story"]["total"] == 2
        assert snap["story"]["current_index"] == 1
        assert snap["story"]["completed"] == 1
        assert snap["story"]["sequence"][0]["id"] == "s0"
        assert snap["story"]["sequence"][1]["completed"] is True

        assert snap["battling"]["total"] == 1
        assert snap["dynamics"]["total"] == 0

    def test_snapshot_empty_sequences(self):
        mgr = _make_manager()
        snap = mgr.get_full_sequence_snapshot()
        for cat in ("story", "battling", "dynamics"):
            assert snap[cat]["total"] == 0
            assert snap[cat]["sequence"] == []


# ---------------------------------------------------------------
# Sequence exhaustion fallback
# ---------------------------------------------------------------


class TestSequenceExhaustion:
    def test_story_exhaustion_appends_fallback(self):
        """When all story objectives are completed and index reaches the end,
        a planning-fallback objective should be appended."""
        mgr = _make_manager(
            story_objs=[_obj("s0"), _obj("s1")],
            story_idx=2,
        )
        for obj in mgr.story_sequence:
            obj.completed = True

        current = mgr._get_current_objective_for_category("story")
        assert current is not None
        assert current.id == "auto_plan_objectives"
        assert current.action_type == "create_new_objectives"
        assert "subagent_plan_objectives" in current.description
        assert len(mgr.story_sequence) == 3

    def test_fallback_not_appended_when_objectives_remain(self):
        """The fallback should NOT appear when there are still uncompleted objectives."""
        mgr = _make_manager(
            story_objs=[_obj("s0"), _obj("s1")],
            story_idx=0,
        )
        current = mgr._get_current_objective_for_category("story")
        assert current.id == "s0"
        assert len(mgr.story_sequence) == 2

    def test_fallback_not_duplicated_on_repeated_calls(self):
        """Calling _get_current_objective_for_category twice after exhaustion
        should not append a second fallback."""
        mgr = _make_manager(story_objs=[_obj("s0")], story_idx=1)
        mgr.story_sequence[0].completed = True

        obj1 = mgr._get_current_objective_for_category("story")
        obj2 = mgr._get_current_objective_for_category("story")
        assert obj1.id == "auto_plan_objectives"
        assert obj2.id == "auto_plan_objectives"
        assert len(mgr.story_sequence) == 2

    def test_empty_story_sequence_returns_none(self):
        """An empty story sequence (no objectives loaded at all) should return None,
        not append the fallback (the fallback only fires when there *were* objectives)."""
        mgr = _make_manager()
        current = mgr._get_current_objective_for_category("story")
        assert current is None


# ---------------------------------------------------------------
# Server-like endpoint simulation tests
# ---------------------------------------------------------------


class TestServerEndpointSimulation:
    """Simulate the shape of MCP endpoint handlers to verify the contract."""

    def test_replan_valid_request(self):
        mgr = _make_manager(story_objs=[_obj("s0"), _obj("s1")])
        result = mgr.replan_category("story", [
            {"index": 1, "objective": {"id": "s1_new", "description": "replaced", "action_type": "interact"}},
        ])
        assert result["success"] is True
        assert result["edits_applied"][0]["action"] == "modify"
        assert result["new_sequence_length"] == 2
        assert result["error"] is None

    def test_replan_validation_error_returns_cleanly(self):
        mgr = _make_manager(story_objs=[_obj("s0")])
        result = mgr.replan_category("story", [
            {"index": 5, "objective": {"id": "gap", "description": "x", "action_type": "navigate"}},
        ])
        assert result["success"] is False
        assert "Non-contiguous" in result["error"]

    def test_full_sequence_is_json_serializable(self):
        mgr = _make_manager(
            story_objs=[_obj("s0"), _obj("s1")],
            battling_objs=[_obj("b0", category="battling")],
        )
        snap = mgr.get_full_sequence_snapshot()
        serialized = json.dumps(snap)
        assert isinstance(json.loads(serialized), dict)

    def test_replan_does_not_affect_completed_objectives(self):
        """Replanning must not change the completed status of earlier objectives."""
        mgr = _make_manager(
            story_objs=[_obj("s0"), _obj("s1"), _obj("s2")],
            story_idx=1,
        )
        mgr.story_sequence[0].completed = True
        result = mgr.replan_category("story", [
            {"index": 1, "objective": {"id": "s1_new", "description": "replaced", "action_type": "navigate"}},
        ])
        assert result["success"] is True
        assert mgr.story_sequence[0].completed is True
        assert mgr.story_sequence[0].id == "s0"


# ---------------------------------------------------------------
# json_utils: protobuf conversion + replan index coercion
# ---------------------------------------------------------------


class TestJsonUtilsReplanProtobuf:
    def test_coerce_string_digit_index(self):
        assert coerce_replan_edit_index("3") == 3
        assert coerce_replan_edit_index("  0 ") == 0

    def test_coerce_float_index(self):
        assert coerce_replan_edit_index(4.0) == 4

    def test_coerce_invalid_string_unchanged(self):
        assert coerce_replan_edit_index("objective") == "objective"

    def test_normalize_replan_edits_coerces_indices(self):
        out = normalize_replan_edits(
            [{"index": "0", "objective": {"id": "a", "description": "d", "action_type": "navigate"}}]
        )
        assert len(out) == 1
        assert out[0]["index"] == 0

    def test_convert_protobuf_sequence_before_mapping(self):
        """Repeated-like protobuf must become a list, not a merged dict."""

        class ProtoList(list):
            __module__ = "proto.marshal.collections.repeated"

        inner_a = {"index": 0, "x": 1}
        inner_b = {"index": 1, "x": 2}
        repeated = ProtoList([inner_a, inner_b])
        assert convert_protobuf_value(repeated) == [inner_a, inner_b]

    def test_replan_category_accepts_string_index_after_normalize(self):
        mgr = _make_manager(story_objs=[_obj("s0"), _obj("s1")])
        edits = normalize_replan_edits(
            [{"index": "1", "objective": {"id": "s1_new", "description": "r", "action_type": "navigate"}}]
        )
        result = mgr.replan_category("story", edits)
        assert result["success"] is True
        assert mgr.story_sequence[1].id == "s1_new"

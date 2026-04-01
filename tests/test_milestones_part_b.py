from pokemon_env.emulator import COMPARISON_MILESTONES, ORDERED_PROGRESS_MILESTONES, EmeraldEmulator


PART_B_SEQUENCE = [
    "ROUTE_116",
    "RUSTURF_TUNNEL",
    "DEVON_CORP_3F_MR_STONE",
    "MR_BRINEYS_HOUSE",
    "DEWFORD_TOWN",
    "ROUTE_106",
    "GRANITE_CAVE_STEVEN",
    "DEWFORD_GYM_ENTERED",
    "KNUCKLE_BADGE",
    "SLATEPORT_CITY",
    "OCEANIC_MUSEUM_AQUA",
    "ROUTE_110",
    "RIVAL_BATTLE_ROUTE_110",
    "MAUVILLE_CITY",
    "MAUVILLE_GYM_ENTERED",
    "DYNAMO_BADGE",
]


class _StubMilestoneTracker:
    def __init__(self, completed=None):
        self.completed = set(completed or [])
        self.marked = []

    def is_completed(self, milestone_id):
        return milestone_id in self.completed

    def mark_completed(self, milestone_id, timestamp=None, agent_step_count=None):
        self.completed.add(milestone_id)
        self.marked.append(milestone_id)
        return True


def _make_emulator(completed=None):
    emu = EmeraldEmulator.__new__(EmeraldEmulator)
    emu.milestone_tracker = _StubMilestoneTracker(completed=completed)
    return emu


def _completed_before(milestone_id):
    idx = ORDERED_PROGRESS_MILESTONES.index(milestone_id)
    return ORDERED_PROGRESS_MILESTONES[:idx]


def _state(location, badges=None):
    return {
        "player": {"location": location},
        "game": {"badges": badges or []},
    }


def test_order_and_comparison_are_canonical():
    assert COMPARISON_MILESTONES == ORDERED_PROGRESS_MILESTONES
    assert len(ORDERED_PROGRESS_MILESTONES) == len(set(ORDERED_PROGRESS_MILESTONES))
    assert "ROXANNE_DEFEATED" not in ORDERED_PROGRESS_MILESTONES
    assert "FIRST_GYM_COMPLETE" not in ORDERED_PROGRESS_MILESTONES

    # Ensure the new part-B sequence appears in-order in canonical milestones.
    part_b_indices = [ORDERED_PROGRESS_MILESTONES.index(mid) for mid in PART_B_SEQUENCE]
    assert part_b_indices == sorted(part_b_indices)


def test_route_116_requires_stone_badge():
    emu = _make_emulator(completed=_completed_before("STONE_BADGE"))
    emu.check_and_update_milestones(_state("ROUTE_116", badges=[]))
    assert "ROUTE_116" not in emu.milestone_tracker.completed

    emu.check_and_update_milestones(_state("ROUTE_116", badges=["Stone"]))
    assert "ROUTE_116" in emu.milestone_tracker.completed


def test_non_linear_dewford_paths_and_slateport_or_logic():
    # Route where Steven delivery is done before beating Brawly.
    emu = _make_emulator(completed=_completed_before("SLATEPORT_CITY") + ["GRANITE_CAVE_STEVEN"])
    emu.check_and_update_milestones(_state("SLATEPORT CITY", badges=["Stone"]))
    assert "SLATEPORT_CITY" in emu.milestone_tracker.completed

    # Route where Brawly is beaten first.
    emu2 = _make_emulator(completed=_completed_before("SLATEPORT_CITY") + ["KNUCKLE_BADGE"])
    emu2.check_and_update_milestones(_state("SLATEPORT CITY", badges=["Stone", "Knuckle"]))
    assert "SLATEPORT_CITY" in emu2.milestone_tracker.completed

    # False-positive prevention: location match without OR prerequisites must not fire.
    emu3 = _make_emulator(completed=_completed_before("GRANITE_CAVE_STEVEN"))
    emu3.check_and_update_milestones(_state("SLATEPORT CITY", badges=["Stone"]))
    assert "SLATEPORT_CITY" not in emu3.milestone_tracker.completed


def test_route_110_requires_museum_event():
    emu = _make_emulator(completed=_completed_before("OCEANIC_MUSEUM_AQUA"))
    emu.check_and_update_milestones(_state("ROUTE 110", badges=["Stone", "Knuckle"]))
    assert "ROUTE_110" not in emu.milestone_tracker.completed

    emu2 = _make_emulator(completed=_completed_before("ROUTE_110") + ["OCEANIC_MUSEUM_AQUA"])
    emu2.check_and_update_milestones(_state("ROUTE_110", badges=["Stone", "Knuckle"]))
    assert "ROUTE_110" in emu2.milestone_tracker.completed


def test_full_part_b_sequence_through_dynamo_badge():
    emu = _make_emulator(completed=_completed_before("ROUTE_116"))

    emu.check_and_update_milestones(_state("ROUTE 116", badges=["Stone"]))
    emu.check_and_update_milestones(_state("RUSTURF TUNNEL", badges=["Stone"]))
    emu.check_and_update_milestones(_state("RUSTBORO CITY DEVON CORP 3F", badges=["Stone"]))
    emu.check_and_update_milestones(_state("ROUTE_104_MR_BRINEYS_HOUSE_ALT", badges=["Stone"]))
    emu.check_and_update_milestones(_state("DEWFORD TOWN", badges=["Stone"]))
    emu.check_and_update_milestones(_state("ROUTE 106", badges=["Stone"]))
    emu.check_and_update_milestones(_state("GRANITE CAVE STEVENS ROOM", badges=["Stone"]))
    emu.check_and_update_milestones(_state("DEWFORD TOWN GYM", badges=["Stone", "Knuckle"]))
    emu.check_and_update_milestones(_state("SLATEPORT CITY", badges=["Stone", "Knuckle"]))
    emu.check_and_update_milestones(_state("SLATEPORT CITY OCEANIC MUSEUM 2F", badges=["Stone", "Knuckle"]))
    emu.check_and_update_milestones(_state("ROUTE_110", badges=["Stone", "Knuckle"]))
    emu.check_and_update_milestones(_state("MAUVILLE CITY", badges=["Stone", "Knuckle"]))
    emu.check_and_update_milestones(_state("MAUVILLE CITY GYM", badges=["Stone", "Knuckle", "Dynamo"]))

    for milestone in PART_B_SEQUENCE:
        assert milestone in emu.milestone_tracker.completed

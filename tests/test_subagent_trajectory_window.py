import json
from pathlib import Path

from agents.subagents.trajectory_window import MAX_TRAJECTORY_WINDOW, load_recent_trajectories


class _RunManagerStub:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir


def _write_trajectories(run_dir: Path, count: int) -> None:
    trajectory_file = run_dir / "prompt_evolution" / "trajectories" / "trajectories.jsonl"
    trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_file.open("w", encoding="utf-8") as handle:
        for step in range(1, count + 1):
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "reasoning": f"reasoning-{step}",
                        "action": {"tool": "press_buttons", "buttons": ["A"]},
                        "pre_state": {"location": "Route 101", "player_coords": [step, step]},
                        "post_state": {"location": "Route 101", "player_coords": [step + 1, step + 1]},
                        "outcome": {"success": True},
                    }
                )
                + "\n"
            )


def test_load_recent_trajectories_returns_last_n_preserving_order(tmp_path):
    _write_trajectories(tmp_path, 8)
    loaded = load_recent_trajectories(_RunManagerStub(tmp_path), last_n_steps=3)
    assert [entry["step"] for entry in loaded] == [6, 7, 8]


def test_load_recent_trajectories_caps_window_at_25(tmp_path):
    _write_trajectories(tmp_path, 30)
    loaded = load_recent_trajectories(_RunManagerStub(tmp_path), last_n_steps=999)
    assert len(loaded) == MAX_TRAJECTORY_WINDOW
    assert loaded[0]["step"] == 6
    assert loaded[-1]["step"] == 30


def test_load_recent_trajectories_handles_missing_or_empty_files(tmp_path):
    assert load_recent_trajectories(_RunManagerStub(tmp_path), last_n_steps=5) == []

    trajectory_file = tmp_path / "prompt_evolution" / "trajectories" / "trajectories.jsonl"
    trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    trajectory_file.write_text("", encoding="utf-8")

    assert load_recent_trajectories(_RunManagerStub(tmp_path), last_n_steps=5) == []

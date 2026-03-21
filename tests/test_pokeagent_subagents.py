import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from agents.PokeAgent import PokeAgent
from agents.subagents.verify import parse_verify_response
from utils.data_persistence.llm_logger import LLMLogger
from utils.data_persistence.run_data_manager import RunDataManager
import utils.data_persistence.llm_logger as llm_logger_module


sys_path = Path(__file__).parent.parent
if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))


PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0ioAAAAASUVORK5CYII="
_POKE_MODULE = importlib.import_module("agents.PokeAgent")


class RecordingVLM:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.calls = []
        RecordingVLM.instances.append(self)

    def get_text_query(self, prompt, interaction_name):
        self.calls.append(("text", interaction_name, prompt))
        if interaction_name == "Subagent_Verify":
            return json.dumps(
                {
                    "objective_category": "story",
                    "objective_description": "Talk to Prof. Birch",
                    "is_complete": True,
                    "confidence": "high",
                    "evidence_for": ["Birch dialogue has already fired."],
                    "evidence_against": [],
                    "recommended_next_action": "Advance to the next story objective.",
                    "reasoning_summary": "Current state and recent steps both show the event is done.",
                }
            )
        return "The agent is looping and should realign."

    def get_query(self, image, prompt, interaction_name):
        assert isinstance(image, Image.Image)
        return self.get_text_query(prompt, interaction_name)


class LoggingVLM(RecordingVLM):
    def get_text_query(self, prompt, interaction_name):
        response = super().get_text_query(prompt, interaction_name)
        llm_logger_module.get_llm_logger().log_interaction(
            interaction_type=interaction_name,
            prompt=prompt,
            response=response,
            metadata={"token_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
            duration=0.01,
            model_info={"model": "test-model"},
            step_number=int(os.environ.get("LLM_STEP_NUMBER", "0")),
        )
        return response

    def get_query(self, image, prompt, interaction_name):
        return self.get_text_query(prompt, interaction_name)


def _build_adapter():
    adapter = MagicMock()

    def call_tool(name, arguments):
        if name == "get_game_state":
            return {
                "success": True,
                "state_text": "Current Location: Route 101\nDialog: Prof. Birch is safe.",
                "player_position": {"x": 5, "y": 8},
                "raw_state": {"player": {"location": "Route 101"}},
                "screenshot_base64": PNG_BASE64,
                "objectives_mode": "categorized",
                "categorized_objectives": {
                    "story": {
                        "id": "story_01",
                        "description": "Talk to Prof. Birch",
                        "completion_condition": "birch_dialog_complete",
                    },
                    "battling_group": [
                        {
                            "id": "battle_01",
                            "description": "Train Mudkip to level 7",
                            "completion_condition": "starter_level_7",
                        }
                    ],
                    "dynamics": {
                        "id": "dyn_01",
                        "description": "Leave Route 101 grass",
                        "completion_condition": "out_of_grass",
                    },
                    "recommended_battling_objectives": ["battle_01"],
                },
                "categorized_status": {
                    "story": {"current_index": 0, "total": 12, "completed": 0},
                    "battling": {"current_index": 0, "total": 5, "completed": 0},
                    "dynamics": {"current_index": 0, "total": 4, "completed": 0},
                },
            }
        if name == "get_progress_summary":
            return {
                "success": True,
                "progress": {
                    "total_milestones_completed": 2,
                    "direct_objectives": {
                        "current_sequence": "categorized_full_game",
                        "objectives_completed_in_current_sequence": 0,
                        "total_in_current_sequence": 21,
                        "is_sequence_complete": False,
                        "current_objective": "Talk to Prof. Birch",
                    },
                },
            }
        if name == "get_knowledge_summary":
            return {"success": True, "summary": "Saved Birch on Route 101."}
        raise AssertionError(f"Unexpected tool call: {name}")

    adapter.call_tool.side_effect = call_tool
    return adapter


def _write_trajectory_window(run_manager: RunDataManager, steps: int = 4):
    trajectory_file = run_manager.get_run_directory() / "prompt_evolution" / "trajectories" / "trajectories.jsonl"
    trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    with trajectory_file.open("w", encoding="utf-8") as handle:
        for step in range(1, steps + 1):
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "reasoning": f"step {step} reasoning",
                        "action": {"tool": "press_buttons", "buttons": ["A"]},
                        "pre_state": {"location": "Route 101", "player_coords": [4, 8]},
                        "post_state": {"location": "Route 101", "player_coords": [5, 8]},
                        "outcome": {"success": True},
                    }
                )
                + "\n"
            )


def _make_agent(tmp_path, vlm_cls=RecordingVLM):
    RecordingVLM.instances.clear()
    run_manager = RunDataManager(run_id="subagent-test", base_dir=str(tmp_path))
    _write_trajectory_window(run_manager)
    adapter = _build_adapter()

    with patch.object(PokeAgent, "_load_system_instructions", return_value="SYS_BODY"), patch.object(
        _POKE_MODULE, "MCPToolAdapter", return_value=adapter
    ), patch.object(_POKE_MODULE, "VLM", vlm_cls), patch.object(
        _POKE_MODULE, "get_run_data_manager", return_value=run_manager
    ):
        agent = PokeAgent(server_url="http://localhost:8000", backend="gemini", model="gemini-2.5-flash")
        agent._local_subagent_vlm = vlm_cls(model_name=agent.model, backend=agent.backend, tools=None)

    agent.step_count = 7
    return agent, run_manager


def test_reflect_uses_toolless_local_subagent_and_trajectory_window(tmp_path):
    agent, run_manager = _make_agent(tmp_path)
    with patch.object(_POKE_MODULE, "get_run_data_manager", return_value=run_manager):
        result = json.loads(agent._execute_reflect({"situation": "I keep pressing A", "last_n_steps": 99}))

    assert result["success"] is True
    assert result["context_analyzed"]["steps_reviewed"] == 4

    local_instances = [instance for instance in RecordingVLM.instances if instance.kwargs.get("tools") is None]
    assert len(local_instances) == 1
    assert local_instances[0].calls[0][1] == "Subagent_Reflect"
    assert "last 25 steps" in local_instances[0].calls[0][2]


def test_parse_verify_response_strips_markdown_json_fence():
    target = {"category": "story", "objective": {"description": "Talk to Mom"}}
    fenced = (
        "```json\n"
        '{"is_complete": true, "confidence": "high", "evidence_for": ["done"], '
        '"evidence_against": [], "recommended_next_action": "next", '
        '"reasoning_summary": "ok", "objective_category": "story", '
        '"objective_description": "Talk to Mom"}\n'
        "```"
    )
    out = parse_verify_response(fenced, target=target)
    assert out["is_complete"] is True
    assert out["confidence"] == "high"
    assert "```" not in out["raw_response"]


def test_verify_returns_structured_verdict_for_categorized_objective(tmp_path):
    agent, run_manager = _make_agent(tmp_path)
    with patch.object(_POKE_MODULE, "get_run_data_manager", return_value=run_manager):
        result = json.loads(agent._execute_verify({"reasoning": "Birch event seems finished", "category": "story"}))

    assert result["success"] is True
    assert result["objective_category"] == "story"
    assert result["is_complete"] is True
    assert result["confidence"] == "high"
    assert result["recommended_next_action"]
    assert result["steps_reviewed"] == 4


def test_local_subagent_runner_can_log_metrics_with_stable_interaction_name(tmp_path):
    metrics_path = tmp_path / "cumulative_metrics.json"
    logger = None
    old_logger = llm_logger_module._llm_logger

    def _get_cache_path(relative_path):
        if "cumulative_metrics" in relative_path:
            return metrics_path
        return tmp_path / relative_path

    try:
        with patch("utils.data_persistence.run_data_manager.get_cache_path", side_effect=_get_cache_path):
            logger = LLMLogger(log_dir=str(tmp_path / "logs"), session_id="subagent-metrics")
        llm_logger_module._llm_logger = logger

        agent, run_manager = _make_agent(tmp_path / "agent_run", vlm_cls=LoggingVLM)
        with patch.object(_POKE_MODULE, "get_run_data_manager", return_value=run_manager):
            result = json.loads(agent._execute_reflect({"situation": "Need a diagnosis"}))

        assert result["success"] is True
        assert logger.cumulative_metrics["total_llm_calls"] >= 1
        assert any(step["step"] == 7 for step in logger.cumulative_metrics["steps"])
    finally:
        llm_logger_module._llm_logger = old_logger

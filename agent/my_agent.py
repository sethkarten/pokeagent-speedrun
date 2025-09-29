
from utils.vlm import VLM
from agent.perception import perception_step
from agent.planning import planning_step
from agent.memory import memory_step
from agent.action import action_step


class MyAgent:
    def __init__(self, args=None):
        """
        Custom four-module agent. Mirrors the signature of the built-in Agent.

        Args:
            args: Command-line arguments with backend, model_name, etc.
                  If omitted, defaults to Gemini flash on the Gemini backend.
        """
        backend = getattr(args, "backend", "gemini")
        model_name = getattr(args, "model_name", "gemini-2.5-flash")

        # Initialise a VLM with these settings
        self.vlm = VLM(model_name=model_name, backend=backend)

        # Initialise memory, current plan, etc.
        self.memory_context = ""
        self.current_plan = None
        self.recent_actions = []
        self.observation_buffer = []

    def step(self, game_state: dict):
        """
        Process a game state and return an action.

        Args:
            game_state: Dictionary built by the client, containing:
                - frame: PIL Image
                - player, game, map, milestones, visual, step_number, status, action_queue_length
        """
        # 1. Extract frame and state data
        frame = game_state["frame"]  # PIL Image
        # FIX: just get the entire game_state
        state_data = game_state

        # 2. Perception: describe the scene and determine if slow thinking is needed
        observation, slow_thinking_needed = perception_step(frame, state_data, self.vlm)

        # Store observation for memory (use step_number as frame_id)
        self.observation_buffer.append({
            "frame_id": state_data["step_number"],
            "observation": observation.get("description", ""),
            "state": state_data,
        })

        # 3. Memory: update memory context
        self.memory_context = memory_step(
            self.memory_context,
            self.current_plan,
            self.recent_actions,
            self.observation_buffer,
            self.vlm
        )
        self.observation_buffer = []  # clear buffer each step

        # 4. Planning: update or create plan
        self.current_plan = planning_step(
            self.memory_context,
            self.current_plan,
            slow_thinking_needed,
            state_data,
            self.vlm
        )

        # 5. Action: decide next button presses
        actions = action_step(
            self.memory_context,
            self.current_plan,
            observation.get("description", ""),
            frame,
            state_data,
            self.recent_actions,
            self.vlm
        )

        # Track actions and return
        self.recent_actions.extend(actions)
        # FIX: client.py expects the agents step() to return a dictionary with the action key
        return {"action": actions}

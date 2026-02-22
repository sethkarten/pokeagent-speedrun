# Agent prompts

Prompt and instruction files used by the agent and server:

- **POKEAGENT.md** – Main system instructions for CLI agents (my_cli_agent, autonomous_cli, vision_only).
- **system_prompt.md** – Lean system prompt (tools + core objective); used by prompt optimizer and autonomous agent when optimization is enabled.
- **base_prompt.md** – Strategic/base prompt (can be optimized by prompt optimizer).
- **SLAM_INSTRUCTIONS.md** – Documentation for SLAM (map building) mode.

All code that loads these files should use paths relative to the **repository root** (e.g. `agent/prompts/POKEAGENT.md`) or resolve via `Path(__file__).resolve().parent.parent / "agent" / "prompts" / "..."`.

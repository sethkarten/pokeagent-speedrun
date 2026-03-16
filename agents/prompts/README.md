# Agent prompts

Prompt and instruction files used by the agent and server:

- `pokeagent-directives/POKEAGENT.md` – Main system instructions for custom benchmark agents.
- `pokeagent-directives/system_prompt.md` – Lean system prompt used by prompt optimizer and autonomous runs when optimization is enabled.
- `pokeagent-directives/prompt-optimization/base_prompt.md` – Strategic/base prompt that prompt optimization mutates.
- `pokeagent-directives/SLAM_INSTRUCTIONS.md` – Documentation for SLAM (map building) mode.
- `cli-agent-directives/pokemon_directive.md` – Directive used by external/containerized CLI agents.

All code that loads these files should use paths relative to the repository root (for example, `agents/prompts/pokeagent-directives/POKEAGENT.md`) or resolve them from the repo root with `Path(__file__).resolve()`.

# Agent prompts

Prompt and instruction files used by the agent and server:

- `pokeagent-directives/POKEAGENT.md` — **Default system instructions** for the in-process PokeAgent when **prompt optimization is off**. Full playbook: tools, behavior, objectives, etc.

- `pokeagent-directives/system_prompt.md` — **Lean fixed system prompt** when **prompt optimization is on**: **tool reference + hard constraints only**. Requires an initialized **`run_data_manager`** (experiment run directory); otherwise `PokeAgent` raises **`RuntimeError`** at construction. Strategy is supplied in the user turn via the optimizable base block.

- `pokeagent-directives/prompt-optimization/base_prompt.md` — **Seed** for the optimizable strategic block. Loaded once into `PromptOptimizer.current_base_prompt`; each optimization step rewrites that in-memory text (and saves snapshots under `run_data/.../prompt_evolution/meta_prompts/`). Subsequent optimizations edit the **previous** optimized base, not a fresh read from disk.

- `pokeagent-directives/SLAM_INSTRUCTIONS.md` — Documentation for SLAM (map building) mode.

- `cli-agent-directives/pokemon_directive.md` — Directive used by external/containerized CLI agents (`run_cli.py`), not wired through `PokeAgent.__init__`.

## Entry points

- **`run.py`** — Does not pass `system_instructions_file`; uses automatic selection from `enable_prompt_optimization` (see `agents/PokeAgent.py`).

- **`python -m agents.PokeAgent`** — Omit `--system-instructions` for the same automatic behavior. Pass `--system-instructions path` to override. Use `--enable-prompt-optimization` and `--optimization-frequency` to match `run.py` flags.

All code that loads these files should use paths relative to the repository root (for example `agents/prompts/pokeagent-directives/POKEAGENT.md`) or resolve them from the repo root with `agents.prompts.paths.resolve_repo_path`.

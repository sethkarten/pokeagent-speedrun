# Agents

This package contains the main benchmark agent and its supporting modules.

## PokeAgent

- **`PokeAgent.py`** — Main benchmark agent. Uses a VLM to analyze game state and decide actions. Supports direct objectives, a knowledge base, and optional prompt optimization. Selected in `run.py` with `--scaffold pokeagent` or `--scaffold autonomous_cli`.

## Supporting modules

- **`subagents/`** — Local subagent prompt/runners (`battler.py`, `planner.py`, `summarize.py`, `reflect.py`, `verify.py`, `gym_puzzle.py`, …) exposed to the orchestrator as `subagent_*` tools.
- **`subagents/utils/`** — Shared plumbing: `registry.py`, `runtime.py`, `context.py`, `trajectory_window.py`, `puzzle_solver.py`.
- **`prompts/`** — Canonical prompt assets and path helpers (e.g. PokeAgent directives, CLI-agent directives).
- **`objectives/`** — Direct objectives: types, categorization, and sequences (e.g. story, battling).
- **`utils/prompt_optimizer.py`** — Optional naive prompt optimization based on recent trajectories (used when `--enable-prompt-optimization` is set).

## Flow

The agent composes a `VLM` instance (from `utils/agent_infrastructure/vlm_backends.py`) and an MCP tool adapter to call the game server. Each orchestrator step: get state → build prompt (objectives, history, base prompt) → call VLM → execute tool calls (e.g. `press_buttons`, `navigate_to`, local `subagent_*` tools) → update history and metrics. Looping subagents (`subagent_battler`, `subagent_plan_objectives`) consume real global steps per inner turn but only return compacted summaries/results to the orchestrator's short-term memory (capped at 20 turns).

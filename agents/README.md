# Agents

This package contains the main benchmark agent and its supporting modules.

## PokeAgent

- **`PokeAgent.py`** — Main benchmark agent. Uses a VLM to analyze game state and decide actions. Supports direct objectives, a knowledge base, and optional prompt optimization. Selected in `run.py` with `--scaffold pokeagent` or `--scaffold autonomous_cli`.

## Supporting modules

- **`subagents/`** — Local subagent modules and helpers: registry/runtime, context loading, trajectory windows, `subagent_reflect`, `subagent_verify`, `subagent_summarize`, `subagent_battler`, and gym-puzzle support (`gym_puzzle.py`, `puzzle_solver.py`).
- **`prompts/`** — Canonical prompt assets and path helpers (e.g. PokeAgent directives, CLI-agent directives).
- **`objectives/`** — Direct objectives: types, categorization, and sequences (e.g. story, battling).
- **`utils/prompt_optimizer.py`** — Optional naive prompt optimization based on recent trajectories (used when `--enable-prompt-optimization` is set).

## Flow

The agent composes a `VLM` instance (from `utils/agent_infrastructure/vlm_backends.py`) and an MCP tool adapter to call the game server. Each orchestrator step: get state → build prompt (objectives, history, base prompt) → call VLM → execute tool calls (e.g. `press_buttons`, `navigate_to`, local `subagent_*` tools) → update history and metrics. Delegated battle turns also consume real global steps, but only the final compacted battle summary is published back into orchestrator-visible memory.

# Agents

This package contains the main benchmark agent and its supporting modules.

## PokeAgent

- **`PokeAgent.py`** — Main benchmark agent. Uses a VLM to analyze game state and decide actions. Supports direct objectives, a knowledge base, and optional prompt optimization. Selected in `run.py` with `--scaffold pokeagent` or `--scaffold autonomous_cli`.

## Supporting modules

- **`subagents/`** — Helper agents with their own context (given as trajectory information): `reflect`, `verify`, `context`, `trajectory_window`, and **`puzzle_solver.py`** (`GYM_PUZZLES` for `gym_puzzle_agent`).
- **`prompts/`** — Canonical prompt assets and path helpers (e.g. PokeAgent directives, CLI-agent directives).
- **`objectives/`** — Direct objectives: types, categorization, and sequences (e.g. story, battling).
- **`utils/prompt_optimizer.py`** — Optional naive prompt optimization based on recent trajectories (used when `--enable-prompt-optimization` is set).

## Flow

The agent composes a `VLM` instance (from `utils/agent_infrastructure/vlm_backends.py`) and an MCP tool adapter to call the game server. Each step: get state → build prompt (objectives, history, base prompt) → call VLM → execute tool calls (e.g. `press_buttons`, `navigate_to`) → update history and metrics.

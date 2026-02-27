# PokéAgent Challenge: RPG Speedrunning Agent in Pokémon Emerald

![PokéAgent Challenge: RPG Speedrunning Agent in Pokémon Emerald](emerald.png)

An AI agent that plays Pokémon Emerald using vision-language models to perceive the game environment, plan actions, and execute gameplay strategies. This is a **starter kit** designed to be easily customizable for different VLMs and agent behaviors.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create Conda Environment (Recommended)](#2-create-conda-environment-recommended)
  - [3. Install mgba System Library (Required for Python bindings)](#3-install-mgba-system-library-required-for-python-bindings)
  - [4. Install Compatible libffi in Conda (Important!)](#4-install-compatible-libffi-in-conda-important)
  - [5. Install Python Dependencies](#5-install-python-dependencies)
  - [6. Set up Game ROM](#6-set-up-game-rom)
- [VLM Backend Setup](#vlm-backend-setup)
  - [OpenAI](#-openai-gpt-4v-o3-mini-etc)
  - [OpenRouter](#-openrouter-access-to-many-models)
  - [Google Gemini](#-google-gemini)
  - [Local HuggingFace Models](#-local-huggingface-models)
  - [Auto Backend Detection](#-auto-backend-detection)
- [Running the Agent](#running-the-agent)
- [Command Line Options](#command-line-options)
- [Customizing Agent Behavior](#customizing-agent-behavior-prompt-editing-guide)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Submission Instructions](#submission-instructions)
- [Citation](#citation)
- [License](#license)

## Overview

This project implements an AI agent capable of playing Pokémon Emerald on a Game Boy Advance emulator. The agent uses a vision-language model (VLM) to analyze game frames, understand the current game state, and make intelligent decisions to progress through the game.

The system uses a **headless server architecture**: the game and emulator run in a server process, while agents and UIs run as clients. Communication is via HTTP REST, with optional WebSocket streaming for the web UI and MCP (Model Context Protocol) for external CLI agents (e.g., Claude Code).

## Architecture

The design follows the structure documented in `System-Design/architecture/`:

- **Server** (`server/app.py`): FastAPI server on port 8000. Runs the mGBA emulator, game loop, state caching (100ms for full state, 5s for map data), and exposes REST endpoints (`/action`, `/state`, `/status`, `/screenshot`, `/save_state`, `/load_state`, `/checkpoint`, etc.). WebSocket `/ws/frames` streams frames to the web UI. MCP tool endpoints under `/mcp/*` allow external agents to interact with the game.
- **Clients**:
  - **run.py**: Starts the server as a subprocess, then runs an in-repo agent client (pygame display optional). The client polls `/state`, runs the selected agent scaffold (VLM + logic), and submits actions via `POST /action`.
  - **run_cli.py**: For external CLI agents (e.g., Claude Code). Spawns the game server and an MCP server (`server/cli/pokemon_mcp_server.py`) that translates MCP tool calls into HTTP requests to the game server. The external agent talks to the MCP server over stdio.
- **VLM layer** (`utils/vlm_backends.py`): `VLM` facade over multiple backends (OpenAI, Anthropic, OpenRouter, Google Gemini, local HuggingFace, etc.). All backends implement `VLMBackend`; the facade handles tool-format conversion per provider.
- **Persistence**: Runtime cache in `.pokeagent_cache/{run_id}/` (checkpoint state, LLM history, `cumulative_metrics.json`, milestones, maps, knowledge base). Backups in `backups/{run_id}/`. Analysis data in `run_data/{run_id}/` (prompt_evolution, end_state, agent_logs). See `System-Design/architecture/data_persistence/persistence.md`.
- **Metrics**: `LLMLogger` in `utils/llm_logger.py` records LLM interactions and aggregates tokens, cost, and actions into `cumulative_metrics.json`. Step-, milestone-, and objective-level granularity. See `System-Design/architecture/metrics/tracking.md`.
- **Game infrastructure**: `pokemon_env/emulator.py` (EmeraldEmulator), `pokemon_env/memory_reader.py` (PokemonEmeraldReader), Porymap-based map data. See `System-Design/architecture/pokemon_infrastructure/emerald_data.md`.

For deeper detail and known deviations (e.g., monolithic server, polling-based state), see the markdown files under `System-Design/architecture/`.

## Features

- **Multiple VLM backends**: OpenAI, OpenRouter, Google Gemini, Anthropic, local HuggingFace (via `utils/vlm_backends.py`)
- **Vision-based perception**: VLMs analyze game frames and state
- **Agent scaffolds**: Four-module (perception/planning/memory/action), simple, ReAct, ClaudePlays, GeminiPlays, MyCLIAgent, AutonomousCLI, vision-only
- **MCP support**: External CLI agents (e.g., Claude Code) interact via Model Context Protocol
- **Checkpoints & backups**: Save/resume runs; backups in `backups/`; analysis data in `run_data/`
- **Metrics & logging**: Per-step and cumulative tokens, cost, actions; LLM logs and session logs
- **Map system**: Porymap integration, NPC display, movement preview, portal tracking
- **Web interface**: Real-time stream at `http://localhost:8000/stream`
- **Video recording**: Optional MP4 recording of gameplay
- **Customizable prompts**: Edit `agent/system_prompt.py` and module prompts to change behavior

## Directory Structure

```
pokeagent-speedrun/
├── README.md
├── requirements.txt
├── run.py                    # Multiprocess entry: starts server + in-repo agent client
├── run_cli.py                # Entry for external CLI agents (MCP); spawns server + MCP proxy
├── server/
│   ├── app.py                # FastAPI game server (emulator, /state, /action, /mcp/*, etc.)
│   ├── client.py             # In-repo client used by run.py (agent loop, optional pygame)
│   ├── frame_server.py       # Frame streaming
│   ├── stream.html           # Web UI for streaming
│   └── cli/
│       └── pokemon_mcp_server.py   # MCP proxy: stdio ↔ HTTP to game server
├── agent/
│   ├── __init__.py           # Agent factory (scaffold selection)
│   ├── system_prompt.py      # Main system prompt
│   ├── my_cli_agent.py       # MyCLIAgent (CLI scaffolding)
│   ├── my_cli_agent_autonomous.py  # AutonomousCLIAgent (objectives, knowledge base)
│   ├── react.py              # ReAct agent
│   ├── claude_plays.py       # ClaudePlaysPokemon
│   ├── gemini_plays.py       # GeminiPlaysPokemon
│   ├── vision_only_agent.py  # Vision-only agent
│   ├── objectives/           # Direct objectives, types, categorization
│   └── deprecated/           # Legacy four-module (perception, planning, memory, action, simple)
├── utils/
│   ├── vlm_backends.py       # VLM facade and backends (OpenAI, Gemini, etc.)
│   ├── state_formatter.py   # Game state formatting for LLM
│   ├── llm_logger.py        # LLM interaction logging and cumulative metrics
│   ├── run_data_manager.py  # Cache paths, run_data layout
│   ├── backup_manager.py    # Checkpoint backups
│   ├── cli_agent_backends.py # CLI agent backends (e.g., Claude Code)
│   ├── anticheat.py, ocr_dialogue.py, map_formatter.py, map_stitcher.py, ...
│   └── ...
├── pokemon_env/
│   ├── emulator.py           # EmeraldEmulator (mGBA, input, frame advance)
│   ├── memory_reader.py      # PokemonEmeraldReader (DO NOT MODIFY for submissions)
│   ├── emerald_utils.py, enums.py, types.py, utils.py
│   └── ...
├── tests/
│   ├── run_tests.py, states/, ground_truth/, test_*.py
│   └── ...
├── System-Design/            # Architecture documentation (living)
│   ├── README.md
│   └── architecture/
│       ├── client_server/communication.md
│       ├── autonomous_agent/vlm_agents.md
│       ├── cli_agents/scaffolding.md
│       ├── data_persistence/persistence.md
│       ├── metrics/tracking.md
│       └── pokemon_infrastructure/emerald_data.md
├── Emerald-GBAdvance/        # rom.gba (not included), *.state
├── .pokeagent_cache/        # Runtime cache per run (checkpoints, metrics, maps)
├── backups/                 # Backup archives
├── run_data/                # Per-run analysis data
└── llm_logs/                # LLM interaction logs (auto-generated)
```

## Requirements

- Python 3.10–3.11
- Pokémon Emerald ROM (not included; obtain legally)
- One of the supported VLM backends (see VLM Backend Setup)
- mGBA system library for Python bindings

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sethkarten/pokeagent-speedrun
cd pokeagent-speedrun
```

### 2. Environment (uv or Conda)

**Option A – uv (recommended in repo):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

**Option B – Conda:** Create and use a conda env (e.g. `pokeagent`) and install dependencies from `requirements.txt`. If you use conda, install a compatible `libffi` in the env (e.g. `conda install libffi`) so mGBA Python bindings work.

### 3. mGBA System Library

Required for Python bindings. Example (Ubuntu 20.04):

```bash
wget https://github.com/mgba-emu/mgba/releases/download/0.10.5/mGBA-0.10.5-ubuntu64-focal.tar.xz
tar -xf mGBA-0.10.5-ubuntu64-focal.tar.xz
sudo dpkg -i mGBA-0.10.5-ubuntu64-focal/libmgba.deb
```

macOS (x86_64): `brew install mgba`

### 4. Python Dependencies

With uv: `uv sync` (or `uv sync --dev` for dev deps). With pip: `pip install -r requirements.txt`.

### 5. Game ROM

Place your Pokémon Emerald ROM in `Emerald-GBAdvance/rom.gba`. US English SHA-1: `f3ae088181bf583e55daf962a92bb46f4f1d07b7`.

## VLM Backend Setup

Set the appropriate API key and run with the chosen backend.

| Backend   | Env var                | Example run |
|----------|-------------------------|-------------|
| OpenAI   | `OPENAI_API_KEY`        | `python run.py --backend openai --model-name gpt-4o` |
| OpenRouter | `OPENROUTER_API_KEY`  | `python run.py --backend openrouter --model-name anthropic/claude-3.5-sonnet` |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `python run.py --backend gemini --model-name gemini-2.5-flash` |
| Local HuggingFace | (optional) | `python run.py --backend local --model-name Qwen/Qwen2-VL-2B-Instruct` |

Auto-detection: `--backend auto` picks a backend based on available keys.

## Running the Agent

**run.py** (in-repo agent): Starts the game server, then runs the selected agent client (with optional pygame display).

```bash
# Default (my_cli_agent scaffold, Gemini)
python run.py

# OpenAI
python run.py --backend openai --model-name gpt-4o

# Auto agent, headless, record
python run.py --agent-auto --headless --record

# Load state / checkpoint
python run.py --load-state Emerald-GBAdvance/start.state
python run.py --load-checkpoint
```

**run_cli.py** (external CLI agents via MCP): Starts the game server and MCP server; the external agent (e.g., Claude Code) connects via MCP and uses tools to play.

```bash
python run_cli.py --cli-type claude --directive path/to/directive.txt
```

**Debug controls (with display):** M = state overlay, Shift+M = map, S = screenshot, Tab = cycle mode, Space = one agent step, 1/2 = save/load state, arrows/WASD = move, Z/X = A/B.

**Web UI:** `http://localhost:8000/stream` (or `--port`).

## Agent Scaffolds

Choose behavior with `--scaffold` (default: `my_cli_agent`).

| Scaffold          | Description |
|-------------------|-------------|
| `my_cli_agent`    | Default. CLI-style agent with frame buffering and game-state tracking. |
| `autonomous_cli`  | Autonomous agent with objectives and knowledge base. |
| `fourmodule`      | Perception → Planning → Memory → Action (legacy). |
| `simple`          | Direct frame + state → VLM → action; faster, no perception/planning/memory. |
| `react`           | ReAct loop: thought → action → observation. |
| `claudeplays`     | Tool-based (e.g. press_buttons, navigate_to), pathfinding, history summarization. |
| `geminiplays`     | Gemini-native tool-based agent. |
| `vision_only`     | Vision-only agent. |

Examples:

```bash
python run.py --scaffold simple --agent-auto
python run.py --scaffold react --agent-auto
python run.py --scaffold claudeplays --backend openai --model-name gpt-4o --agent-auto
```

Deprecated: `--simple` still maps to `--scaffold simple` but triggers a deprecation warning.

## Command Line Options

```text
run.py:
  --rom PATH, --load-state PATH, --load-checkpoint
  --backend (openai|gemini|local|openrouter|anthropic|auto), --model-name TEXT
  --port INT (default 8000)
  --headless, --agent-auto, --manual
  --record, --scaffold (fourmodule|simple|react|claudeplays|geminiplays|my_cli_agent|autonomous_cli|vision_only)
  --no-ocr (disable OCR dialogue detection)
  --simple (deprecated; use --scaffold simple)
```

## Customizing Agent Behavior (Prompt Editing Guide)

- **System prompt**: `agent/system_prompt.py` — overall role and behavior.
- **Perception**: `agent/deprecated/perception.py` — how the agent interprets the screen (fourmodule).
- **Planning**: `agent/deprecated/planning.py` — high-level strategy (fourmodule).
- **Memory**: `agent/deprecated/memory.py` — what to remember (fourmodule).
- **Action**: `agent/deprecated/action.py` — button decisions (fourmodule).

Edit the prompts in those files and restart the agent. Use `--debug-state` for detailed state in logs. For Nuzlocke-style behavior, change the system prompt and action/memory logic accordingly.

## Advanced Configuration

- **Environment**: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`; optional `PYTHONPATH` for development.
- **Persistence**: Checkpoints and run data are under `.pokeagent_cache/{run_id}/` and `run_data/{run_id}/`; see `System-Design/architecture/data_persistence/persistence.md`. Backups of the .pokeagent_cache/{run_id}/ are saved upon objective or milestone completion for our custom scaffold for vlm agents and proprietary CLI agent scaffold agents, respectively.
- **Metrics**: `cumulative_metrics.json` and LLM logs; see `System-Design/architecture/metrics/tracking.md`.

## Troubleshooting

- **Module not found**: Ensure deps are installed (`uv sync` or `pip install -r requirements.txt`) and `PYTHONPATH` includes the repo root if needed.
- **Out of memory (local models)**: Use a smaller model or a cloud backend (e.g. `--backend gemini --model-name gemini-2.5-flash`).
- **Web UI**: Ensure the server is running and the port (default 8000) is free; open `http://localhost:8000/stream`.
- **API rate limits**: Consider OpenRouter or local models.

## Fair Use and Modification Guidelines

**Allowed:** Changing agent behavior (prompts, planning, memory), adding or changing VLM backends in `utils/vlm_backends.py`, improving logging, tests, docs, performance, UI, and utilities.

**Not allowed (for competitive submissions):** Modifying `pokemon_env/memory_reader.py` or memory-reading logic, changing how game state is extracted, altering emulator core or anti-cheat, or manipulating game memory outside normal button input.

## Submission Instructions

Ready to compete in the PokéAgent Challenge? Follow these submission guidelines to participate in Track 2.

### 🎯 Submission Overview

- **Objective**: Achieve maximum game completion in Pokémon Emerald under time constraints
- **Method**: Agents must interact exclusively through the custom Pokémon Emerald emulator API
- **Flexibility**: Use any method, as long as the final action comes from a neural network
- **Anti-cheat**: All submissions undergo verification to ensure fair competition

### 📋 Submission Requirements

Your submission must include **all three** of the following components:

#### 1. **Code Archive** 
- ZIP or TAR.GZ file containing your complete agent implementation
- Include all dependencies and a clear README with setup instructions
- Ensure your code is reproducible and well-documented

#### 2. **Action & State Logs**
- Detailed logs automatically created by this starter kit during your agent's run
- These logs are generated when you run `python run.py` and include:
  - All agent actions and decisions with timestamps
  - Game state information at each step with cryptographic hashes
  - Performance metrics and decision timing analysis
  - Anti-cheat verification data for submission validation
  - LLM interaction logs for debugging and transparency

#### 3. **Video Evidence**
- YouTube link to a screen recording showing your complete speedrun
- Must show the entire run from start to finish
- Video should clearly demonstrate your agent's performance and final game state

### 🏆 Evaluation Criteria

Your submission will be evaluated on:

1. **Milestone Completion**: Percentage of game milestones accomplished (primary metric)
2. **Completion Time**: Time taken to complete achieved milestones (secondary metric)  
3. **Reproducibility**: Clear documentation and reproducible results

### 📝 How to Submit

Submit your complete package through the official Google Form:

**🔗 [Submit Here: https://forms.gle/nFciH9DrT4RKC1vt9](https://forms.gle/nFciH9DrT4RKC1vt9)**

### 💡 Tips for Success

- **Test thoroughly**: Ensure your agent runs reliably for extended periods
- **Document everything**: Clear setup instructions help with reproducibility
- **Optimize for milestones**: Focus on completing key game objectives rather than perfect play
- **Monitor logs**: Use the generated logs to debug and improve your agent's performance
- **Record quality video**: Clear, uninterrupted footage helps with verification

The submission process emphasizes both performance (how much of the game you complete and how quickly) and transparency (providing logs and video evidence for verification).

## Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{karten2025pokeagent,
  title        = {The PokeAgent Challenge: Competitive and Long-Context Learning at Scale},
  author       = {Karten, Seth and Grigsby, Jake and Milani, Stephanie and Vodrahalli, Kiran
                  and Zhang, Amy and Fang, Fei and Zhu, Yuke and Jin, Chi},
  booktitle    = {NeurIPS Competition Track},
  year         = {2025},
  month        = apr,
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Make sure to comply with the terms of service of any VLM APIs you use.

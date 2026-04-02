# Utils

Shared utilities grouped by subsystem: mapping, data persistence, agent infrastructure, and metric tracking.

## mapping/

Map and pathfinding: `ascii_map_loader`, `map_formatter`, `map_stitcher`, `map_stitcher_singleton`, `pathfinding`, `pokeemerald_parser`, `porymap_json_builder`, `porymap_state`. Used to build the current map view and movement preview for the agent. `state_formatter.py` at the top level re-exports from `utils.mapping.porymap_state` and formats game state for the LLM.

## data_persistence/

- **`backup_manager.py`** — Create and restore backups of `.pokeagent_cache/{run_id}/` (e.g. on objective completion). Restored zips copy **persistent** cache files (e.g. `checkpoint.state`, `objectives.json`, `memory.json`, `skills.json`, `subagents.json`, `trajectory_history.jsonl`, metrics, map stitcher data). They do **not** repopulate the orchestrator’s **short-term** in-process state: `PokeAgent` still starts with an empty rolling `conversation_history`; only **long-term** stores on disk carry over. Past steps remain in `trajectory_history.jsonl` when that file is included in the backup, and tools like `process_trajectory_history` can read them on demand.
- **`run_data_manager.py`** — Run directory layout, cache paths, checkpoint/LLM paths; creates `run_data/{run_id}/` (prompt_evolution, end_state, agent_scratch_space, agent_logs (for cli agents)) and finalizes at shutdown.
- **`llm_logger.py`** — Logs LLM interactions and maintains `cumulative_metrics.json` (tokens, cost, actions, steps, milestones). For CLI agents, metrics are synced via `POST /sync_llm_metrics`.

## agent_infrastructure/

- **`vlm_backends.py`** — `VLM` facade over provider backends (OpenAI, Anthropic, OpenRouter, Gemini, Vertex, etc.). All backends implement `VLMBackend`; the facade handles tool-format conversion per provider.
- **`cli_agent_backends.py`** — Abstract `CliAgentBackend` and concrete backends (Claude Code, Gemini CLI, Codex) for `run_cli.py` (containerized CLI agents).

## metric_tracking/

Session readers (Claude, Gemini, Codex) that parse JSONL/session files and derive per-call metrics; used by `run_cli` to populate and sync `cumulative_metrics.json`. `server_metrics` provides `update_server_metrics()` for the in-repo client.

## Top-level

- **`knowledge_base.py`** — Shared by agents and server (e.g. `game_tools.py`) for add/search persistent knowledge.
- **`state_formatter.py`** — Facade over mapping helpers; formats game state for LLMs.
- **`anticheat.py`**, **`error_handler.py`**, **`json_utils.py`**, **`ocr_dialogue.py`** — Miscellaneous helpers.

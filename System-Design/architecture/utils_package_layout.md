# Utils Package Layout

This document summarizes the refactored `utils/` package structure. The layout organizes code by subsystem ownership: mapping, data persistence, agent infrastructure, and metric tracking.

## utils/mapping/

Map and pathfinding logic:

- **ascii_map_loader**, **map_formatter**, **map_stitcher**, **map_stitcher_singleton**, **pathfinding**, **pokeemerald_parser**, **porymap_json_builder**: moved from top-level `utils/`.
- **porymap_state**: extracted from `state_formatter` (ROM-to-porymap mapping, `_format_porymap_info`, `ROM_TO_PORYMAP_MAP`). `utils/state_formatter.py` remains as a facade re-exporting from `utils.mapping.porymap_state`.

## utils/data_persistence/

Runtime cache, run data, backups, and LLM logging:

- **backup_manager**: cache backups and restore.
- **run_data_manager**: cache paths, run_data layout, checkpoint/LLM paths.
- **llm_logger**: LLM interaction logging and cumulative metrics.

## utils/agent_infrastructure/

Shared agent runtime glue:

- **cli_agent_backends**: abstract `CliAgentBackend` and concrete backends (Claude Code, Gemini CLI, Codex). Backend selection via `run_cli.py --backend {claude,gemini,codex}`.
- **vlm_backends**: `VLM` facade and provider backends (OpenAI, Anthropic, OpenRouter, Gemini, etc.).

## utils/metric_tracking/

Session readers and server metrics:

- Session readers: claude_jsonl_reader, gemini_session_reader, codex_session_reader, gemini_telemetry_reader.
- **server_metrics**: `update_server_metrics()` (moved from former `utils/agent_helpers.py`). agent_helpers removed.

## Top-level utils/ (shared)

- **state_formatter.py**: Facade; re-exports mapping helpers and formats game state for LLMs.
- **knowledge_base.py**: Shared by `agents/simple/claude_plays.py` and `server/game_tools.py`; not moved to agents/custom.
- **coordinate_overlay.py**, **anticheat.py**, **error_handler.py**, **json_utils.py**, **ocr_dialogue.py**: remain at top-level.

## agents/custom/utils/

Custom-agent-only helpers:

- **prompt_optimizer.py**: Used by PokeAgent for prompt optimization; moved from `utils/`.

## Removed or relocated

- **get_local_ip**: Inlined into `server/app.py`; `utils/get_local_ip.py` removed.
- **Deleted modules**: helpers, map_trimmer, map_visualizer, pathfinding_ascii, checkpoint, recording.

## Tests

- **tests/test_utils_package_imports.py**: Smoke test for the new package layout and import paths.

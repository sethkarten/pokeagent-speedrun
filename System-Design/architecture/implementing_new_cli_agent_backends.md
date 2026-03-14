# Implementing New CLI Agent Backends

Guidance for implementing backends for external CLI agents (e.g. Claude Code, Gemini CLI, Codex).

---

**IMPORTANT — Claude Code coupling**

`run_cli.py` currently uses Claude Code–specific conventions. Multi-backend support requires refactoring: session persistence fallback (`claude_memory/projects/-workspace`), `agent_memory_subdir`, stream events (`stream-json` types), JSONL polling paths. See `external_mcp_agents.md` for the full list.

---

## Overview

CLI agent backends let the Pokemon Emerald harness talk to external coding agents that run as separate processes and use MCP to interact with the game.

## Key points (from Claude Code)

- **Stream format**: Claude Code uses `--output-format stream-json`; event types `system`, `assistant`, `user`, `result`. Cost/tokens arrive only on `result` at session end; use it to write the metrics snapshot.
- **Checkpointing**: CLI agents don’t call `/checkpoint`; the orchestrator must call `/checkpoint` and `/save_agent_history` periodically (e.g. 60s).
- **Thinking for UI**: POST tool reasoning as `[tool_name] reasoning` for `tool_use` blocks; skip raw `text` to avoid clutter.
- **Processes**: Use `os.setsid()` and `os.killpg()` for clean shutdown of the process group.
- **Session**: Claude’s `--continue` uses `~/.claude/`; mount from host `.pokeagent_cache/{run_id}/claude_memory/` for persistence.
- **Containers**: Run in Docker with firewall (allow API + MCP SSE; block game port). MCP via SSE URL; pass `--containerized`.

## Backend interface

Implement `CliAgentBackend` (in `utils/cli_agent_backends.py`):

- **`name`** (property): e.g. `"claude"`, `"gemini"`.
- **`build_launch_cmd(directive_path, server_url, working_dir, ...)`**: Returns `(cmd, env, bootstrap, temp_mcp_config_path)`.
- **`handle_stream_event(event, metrics, server_url=None, snapshot_path=None)`**: Parse agent stdout, update `CliSessionMetrics`, POST thinking to `/thinking`, write snapshot on `result`.

MCP config: local mode uses `command` + `args` + `env` (e.g. `python -m server.cli.pokemon_mcp_server`); containerized mode uses `url` (e.g. `http://host.docker.internal:8449/sse`).

## Directive and paths

Put the system prompt for your agent under:

`agents/prompts/cli-agent-directives/your_backend_directive.md`

## Implementation steps

1. **Backend class**: Subclass `CliAgentBackend`, implement `name`, `build_launch_cmd`, `handle_stream_event`.
2. **Register**: In `get_backend(cli_type)` in `utils/cli_agent_backends.py`, return your backend for `cli_type == "your_backend"`.
3. **Directive**: Add `agents/prompts/cli-agent-directives/your_backend_directive.md`.
4. **Test**: `python run_cli.py --backend your_backend --directive agents/prompts/cli-agent-directives/your_backend_directive.md --termination-condition gym_badge_count --termination-threshold 1`.

## Metrics

`CliSessionMetrics` tracks `tool_use_count` (incremental) and cost/tokens/turns (when agent emits them, e.g. on `result`). Snapshot: `.pokeagent_cache/{run_id}/cli_metrics_snapshot.json`. Optional: external monitors (e.g. claude-monitor) reading from mounted `~/.claude/`; keep integration minimal.

## Common issues

- MCP not starting → check `PYTHONPATH`. Tool calls failing → check MCP config JSON. Hangs on exit → use process group kill. No metrics → implement `handle_stream_event` for your format. Container network → allow MCP SSE port. Session not persisting → mount backend memory dir from host.

## Best practices

Minimal tool set (e.g. `get_game_state`, `press_buttons`, `navigate_to`); let agents use native knowledge/reflection. Use milestone-based progress for backups. Clean process and signal handling. Prefer containerized runs. Document backend-specific quirks. See `external_mcp_agents.md` for more.

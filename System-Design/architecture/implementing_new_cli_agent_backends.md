# Implementing New CLI Agent Backends

This document provides guidance on implementing backends for external CLI agents (like Claude Code, Gemini CLI, or Codex).

---

**IMPORTANT — Claude Code Coupling Regression**

`run_cli.py` has regressed to Claude Code-specific conventions. Refactoring is needed for multi-backend support. Key areas to abstract:

*   **Session persistence fallback**: `_load_last_session_id()` fallback uses `claude_memory/projects/-workspace` — backend-specific.
*   **`agent_memory_subdir`**: Paths assume `claude_memory`; JSONL polling uses this.
*   **Stream events**: `handle_stream_event` expects Claude Code `stream-json` types (`system`, `assistant`, `user`, `result`).
*   **JSONL polling**: `_poll_claude_jsonl_and_append_steps` hardcodes `claude_memory/projects/-workspace/*.jsonl` structure.

See `external_mcp_agents.md` for full list.

---

## Overview

CLI agent backends allow the Pokemon Emerald environment to interface with external coding agents that operate via command-line interfaces. These agents run as separate processes and communicate through the MCP (Model Context Protocol) to interact with the game.

## Key Learnings from Claude Code Implementation

### Stream-JSON Output Format
- Claude Code emits events in JSON format via `--output-format stream-json`
- Event types: `system`, `assistant`, `user`, `result`
- `system` event: Contains session_id, model, and available tools/MCP servers
- `assistant` event: Contains `message.content` with blocks of type `text` or `tool_use`
- `user` event: Contains `tool_result` blocks
- `result` event: Contains final cost/token metrics (only emitted at session end)

### Result Event Timing
- The `result` event with cost/tokens/turns only arrives when the session completes
- For long-running sessions, `cli_metrics_snapshot.json` may be mostly empty until the session ends
- `tool_use_count` can be tracked incrementally, but cost/tokens require waiting for `result`
- Solution: Write snapshot on `result` event to capture final metrics

### Checkpointing for CLI Agents
- CLI agents don't automatically call `/checkpoint` endpoint
- The orchestrator (`run_cli.py`) must periodically call `/checkpoint` and `/save_agent_history`
- Recommended: 60-second interval in the main monitoring loop

### Tool Use vs Text for UI
- `text` blocks: Raw thinking/reasoning, can be verbose
- `tool_use` blocks: Structured actions with `reasoning` parameter
- For UI consistency with VLM agents: POST thinking in `[tool_name] reasoning` format for `tool_use` blocks
- Skip POSTing raw `text` blocks to avoid UI clutter

### Process Management
- Use `os.setsid()` to create process group for clean shutdown
- Terminate entire process group with `os.killpg()` to ensure MCP subprocesses are also killed
- Handle graceful shutdown with timeout before force kill

### Session Continuity
- Claude Code's `--continue` flag resumes from previous session memory
- Session memory is stored in `~/.claude/` directory
- Mount this directory from host cache for persistence across container restarts
- Session number tracked in iteration count, passed to `build_launch_cmd`

### Containerization
- Run CLI agents in isolated Docker containers to prevent unwarranted behaviors
- Use Anthropic's devcontainer pattern (Dockerfile + devcontainer.json + firewall script)
- Firewall rules: Allow only Anthropic API + MCP SSE server, block game server port
- MCP server runs on host as SSE server, agent connects via URL instead of subprocess
- Mount `~/.claude/` from host `.pokeagent_cache/{run_id}/claude_memory/` for memory persistence
- Pass `--containerized` flag to enable Docker mode

## Architecture

### Backend Interface

All backends must inherit from `CliAgentBackend` and implement:

```python
class CliAgentBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'claude', 'gemini')"""
        pass
    
    @abstractmethod
    def build_launch_cmd(
        self,
        directive_path: str,
        server_url: str,
        working_dir: str,
        *,
        dangerously_skip_permissions: bool = True,
        project_root: str | None = None,
        containerized: bool = False,
        session_number: int = 1,
        thinking_effort: str | None = None,
        mcp_sse_port: int | None = None,
        run_id: str | None = None,
        claude_memory_dir: str | None = None,
    ) -> tuple[list[str], dict[str, str], str, str | None]:
        """Build the command to launch the CLI agent.
        
        Returns:
            - cmd: List of command arguments (e.g., ['claude', '--print', ...])
            - env: Environment variables dict
            - bootstrap: Initial prompt/directive content
            - temp_mcp_config_path: Path to temporary MCP config file (or None)
        """
        pass
    
    @abstractmethod
    def handle_stream_event(
        self,
        event: dict,
        metrics: CliSessionMetrics | None,
        server_url: str | None = None,
        snapshot_path: Optional[Path] = None,
    ) -> None:
        """Parse and handle stream events from the agent's stdout.
        
        Update metrics, POST thinking to UI, log events, etc.
        """
        pass
```

### MCP Configuration

The backend must create an MCP config that tells the agent how to connect to the Pokemon server:

**Local mode (stdio)**:
```json
{
  "mcpServers": {
    "pokemon-emerald": {
      "command": "python",
      "args": ["-m", "server.cli.pokemon_mcp_server"],
      "env": {
        "POKEMON_SERVER_URL": "http://localhost:8000",
        "PYTHONPATH": "/path/to/project"
      }
    }
  }
}
```

**Containerized mode (SSE)**:
```json
{
  "mcpServers": {
    "pokemon-emerald": {
      "url": "http://host.docker.internal:8449/sse"
    }
  }
}
```

### Stream Event Parsing

Implement `handle_stream_event` to:
1. Parse agent-specific output format (JSON, text, etc.)
2. Update `CliSessionMetrics` (tool_use_count, cost, tokens, etc.)
3. POST thinking/reasoning to server UI via `/thinking` endpoint
4. Log events for debugging

Example (Claude Code):
```python
def handle_stream_event(self, event, metrics, server_url=None, snapshot_path=None):
    etype = event.get("type", "")
    
    if etype == "system":
        # Extract session metadata
        if metrics:
            metrics.session_id = event.get("session_id", "")
            metrics.model = event.get("model", "")
    
    elif etype == "assistant":
        # Parse message content blocks
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                if metrics:
                    metrics.tool_use_count += 1
                # POST reasoning to UI
                reasoning = block.get("input", {}).get("reasoning", "")
                tool_name = block.get("name", "?")
                self._post_thinking(server_url, f"[{tool_name}] {reasoning}")
    
    elif etype == "result":
        # Update final metrics from result event
        if metrics:
            metrics.total_cost_usd = event.get("cost_usd", 0.0)
            metrics.input_tokens = event.get("input_tokens", 0)
            metrics.output_tokens = event.get("output_tokens", 0)
            metrics.num_turns = event.get("num_turns", 0)
        # Write snapshot with complete metrics
        if snapshot_path:
            with open(snapshot_path, "w") as f:
                json.dump(asdict(metrics), f, indent=2)
```

## Metric Tracking and Monitoring

### Built-in Metrics
- The `CliSessionMetrics` dataclass tracks:
  - `tool_use_count`: Incremented on each tool use (captured immediately)
  - `total_cost_usd`, `input_tokens`, `output_tokens`, `num_turns`: Only available when agent emits these (e.g., on `result` event)
- Metrics snapshot written to `.pokeagent_cache/{run_id}/cli_metrics_snapshot.json`

### External Monitoring (Optional)
For real-time usage monitoring of Claude CLI agents, consider:
- **[claude-monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor)**: Open-source monitor that reads Claude's internal usage files from `~/.claude/`
- Since `~/.claude/` is mounted from `.pokeagent_cache/{run_id}/claude_memory/`, the monitor can track live usage
- **Note**: Keep integration minimal to avoid module bloat. Use as external tool, not as a direct dependency.

### Recommendations
- Rely on agent's native usage reporting when available (e.g., `result` events)
- Use external monitors only for debugging or detailed analysis
- Avoid adding heavy dependencies just for metrics tracking

## Step-by-Step Implementation

### 1. Create Backend Class

```python
# In utils/cli_agent_backends.py
class YourBackend(CliAgentBackend):
    @property
    def name(self) -> str:
        return "your_backend"
    
    def build_launch_cmd(self, ...):
        # Build command, environment, and MCP config
        # Return (cmd, env, bootstrap, temp_config_path)
        pass
    
    def handle_stream_event(self, event, metrics, server_url=None, snapshot_path=None):
        # Parse your agent's output format
        # Update metrics and POST to server
        pass
```

### 2. Register Backend

```python
# In utils/cli_agent_backends.py
def get_backend(cli_type: str) -> CliAgentBackend:
    if cli_type == "claude":
        return ClaudeCodeBackend()
    elif cli_type == "your_backend":
        return YourBackend()
    else:
        raise ValueError(f"Unknown CLI type: {cli_type}")
```

### 3. Create Directive

Create a system prompt file for your agent:
```markdown
# agent/prompts/cli_directives/your_backend_directive.md

You are playing Pokemon Emerald...
```

### 4. Test

```bash
python run_cli.py \
  --cli-type your_backend \
  --directive agent/prompts/cli_directives/your_backend_directive.md \
  --termination-condition gym_badge_count \
  --termination-threshold 1
```

## Testing Your Backend

1. Run with `--cli-type your_backend_name`
2. Monitor the agent logs in `run_data/{run_id}/agent_logs/`
3. Check session metrics in `.pokeagent_cache/{run_id}/cli_metrics_snapshot.json`
4. Verify MCP tools are accessible via the stream output
5. Test containerized mode with `--containerized` flag (if implementing Docker support)

## Common Issues

- **MCP server not starting**: Check PYTHONPATH is correctly set
- **Tool calls not working**: Verify MCP config JSON syntax
- **Process hangs on exit**: Ensure proper process group handling
- **No metrics captured**: Implement `handle_stream_event` to parse your agent's output format
- **Container network issues**: Verify firewall rules allow MCP SSE server port
- **Session memory not persisting**: Check that `~/.claude/` or equivalent is mounted from host

## Best Practices

1. **Minimal Tool Set**: Expose only core game tools (`get_game_state`, `press_buttons`, `navigate_to`) via MCP
2. **Let Agents Think**: Don't expose knowledge/reflection tools; let CLI agents use their native capabilities
3. **Milestone-Based Progress**: Use milestone completion for backups, not objective completion
4. **Clean Process Management**: Use process groups and handle signals properly
5. **Containerize When Possible**: Isolate agents to prevent unintended actions
6. **Document Everything**: CLI agents have diverse behaviors; document quirks and patterns

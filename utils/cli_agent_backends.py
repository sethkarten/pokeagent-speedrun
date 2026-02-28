"""
CLI agent backends for external coding agents (Claude Code, Codex, etc.).

Each backend knows how to build the launch command, parse the agent's
stream output (e.g. stream-json), extract session metrics, and optionally
stream agent thinking to the game server UI.
"""

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CliSessionMetrics:
    """Metrics collected from a single CLI agent session (e.g. stream-json result event)."""
    session_id: str = ""
    model: str = ""
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    num_turns: int = 0
    duration_ms: int = 0
    duration_api_ms: int = 0
    is_error: bool = False
    error: str = ""
    tool_use_count: int = 0


@dataclass
class CliSession:
    """Resources for a single CLI agent subprocess session."""
    process: "subprocess.Popen"
    stop_event: "threading.Event"
    stream_thread: "threading.Thread"
    temp_mcp_config_path: str | None = None


class CliAgentBackend(ABC):
    """Abstract base for CLI agent backends (Claude Code, Codex, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. 'claude', 'codex')."""
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
        **kwargs,
    ) -> tuple[list[str], dict[str, str], str, str | None]:
        """
        Build command, env, bootstrap prompt, and optional temp MCP config path.

        working_dir: CWD for the CLI agent process (e.g. run_data/.../agent_scratch_space).
        project_root: Project root for PYTHONPATH when spawning MCP server (imports).
        **kwargs: Backend-specific options (e.g. containerized, resume_session_id).

        Returns:
            (cmd, env, bootstrap_str, temp_mcp_config_path)
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
        """Process one stream event (e.g. system, assistant, user, result). Update metrics and optionally POST thinking to server."""
        pass

    def run_stream_reader(
        self,
        stdout_pipe,
        stop_event: "threading.Event",
        log_file: io.TextIOWrapper | None,
        metrics: CliSessionMetrics | None,
        server_url: str | None = None,
        snapshot_path: Optional[Path] = None,
    ) -> None:
        """Read stdout line-by-line (JSONL), tee to log_file, parse and handle events."""
        logger.info("[cli-debug] stream reader started: %s", self.name)
        try:
            buffered = io.BufferedReader(stdout_pipe)
            for raw_line in buffered:
                if stop_event.is_set():
                    break
                line = raw_line.decode("utf-8", errors="replace")
                if log_file:
                    log_file.write(line)
                    log_file.flush()
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    if hasattr(sys, "stdout"):
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    continue
                self.handle_stream_event(event, metrics, server_url, snapshot_path)
        except (OSError, ValueError) as e:
            logger.info("[cli-debug] stream reader error: %s", e)
        logger.info("[cli-debug] stream reader exiting: %s", self.name)


class ClaudeCodeBackend(CliAgentBackend):
    """Backend for Anthropic Claude Code CLI (--print --output-format stream-json)."""

    def __init__(self) -> None:
        super().__init__()
        self._last_event_time: float | None = None  # for thinking duration delta

    @property
    def name(self) -> str:
        return "claude"

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
        resume_session_id: str | None = None,
        thinking_effort: str | None = None,
        mcp_sse_port: int | None = None,
        run_id: str | None = None,
        claude_memory_dir: str | None = None,
    ) -> tuple[list[str], dict[str, str], str, str | None]:
        env = os.environ.copy()
        env["POKEMON_MCP_SERVER_URL"] = server_url
        env["POKEMON_SERVER_URL"] = server_url

        # MCP server must resolve project modules; use project root, not working_dir
        pythonpath = project_root if project_root else os.getcwd()

        # Read directive for bootstrap prompt
        directive_content = ""
        if directive_path and os.path.exists(directive_path):
            with open(directive_path, "r") as f:
                directive_content = f.read()
        bootstrap = (
            f"{directive_content}\n\n"
            "Runtime context:\n"
            f"- Pokemon server URL: {server_url}\n"
            "- You are running in a long-lived interactive session.\n"
            "- Act autonomously and continuously, using MCP tools directly as needed.\n"
            "- Poll game state on your own via MCP tools; do not wait for additional operator prompts.\n"
            "- Continue until externally terminated by the orchestrator when completion condition is met.\n"
        ) if directive_content else "Start the Pokemon Emerald agent session."
        
        # Base claude command
        # --verbose required by Claude Code when using --output-format stream-json
        # Pass bootstrap prompt as command argument (Claude doesn't read from stdin in this mode)
        claude_cmd = ["claude", "--print", bootstrap, "--output-format", "stream-json", "--verbose"]
        if dangerously_skip_permissions:
            claude_cmd.append("--dangerously-skip-permissions")
        
        # Resume previous session by ID (avoids interleaving when multiple instances run)
        if resume_session_id:
            claude_cmd.extend(["--resume", resume_session_id])
        
        # Add thinking effort budget (if specified)
        if thinking_effort:
            effort_map = {"low": "low", "medium": "medium", "high": "high"}
            if thinking_effort in effort_map:
                claude_cmd.extend(["--thinking-budget", effort_map[thinking_effort]])

        # MCP config: use SSE URL for containerized, command for local
        if containerized and mcp_sse_port:
            # Docker Containerized mode: MCP server is an SSE server on the host
            # CRITICAL: The "type" field is REQUIRED for remote servers (sse/http).
            # Without it, Claude Code silently ignores the MCP config and exits with no output.
            mcp_config = {
                "mcpServers": {
                    "pokemon-emerald": {
                        "type": "sse",
                        "url": f"http://host.docker.internal:{mcp_sse_port}/sse"
                    }
                }
            }
        else:
            # Local mode: MCP server is spawned as subprocess
            mcp_config = {
                "mcpServers": {
                    "pokemon-emerald": {
                        "command": sys.executable,
                        "args": ["-m", "server.cli.pokemon_mcp_server"],
                        "env": {
                            "POKEMON_SERVER_URL": server_url,
                            "PYTHONPATH": pythonpath,
                        },
                    }
                }
            }
        
        # Return docker run command if containerized, else return bare claude command
        if containerized:
            # Build docker run command
            if not run_id or not claude_memory_dir:
                raise ValueError("containerized mode requires run_id and claude_memory_dir")
            
            # Resolve paths
            project_abs = Path(project_root if project_root else os.getcwd()).resolve()
            claude_memory_path = Path(claude_memory_dir).resolve()
            working_dir_abs = Path(working_dir).resolve()
            
            # Write MCP config to workspace so it's accessible inside the container
            mcp_config_path = working_dir_abs / ".mcp_config.json"
            with open(mcp_config_path, "w") as f:
                json.dump(mcp_config, f, indent=2)
            
            # Write bootstrap prompt to workspace file (accessible at /workspace inside container)
            # Use @file syntax to avoid passing multi-line content as shell argument
            bootstrap_file = working_dir_abs / ".agent_directive.txt"
            with open(bootstrap_file, "w") as f:
                f.write(bootstrap)

            # Build claude command - use @/workspace/.agent_directive.txt to read prompt from file
            # This avoids shell mangling of multi-line bootstrap content passed as CLI argument
            container_claude_cmd = [
                "claude", "--print", "@/workspace/.agent_directive.txt",
                "--output-format", "stream-json", "--verbose",
            ]
            if dangerously_skip_permissions:
                container_claude_cmd.append("--dangerously-skip-permissions")
            if resume_session_id:
                container_claude_cmd.extend(["--resume", resume_session_id])
            if thinking_effort and thinking_effort in ("low", "medium", "high"):
                container_claude_cmd.extend(["--thinking-budget", thinking_effort])
            container_claude_cmd.extend(["--mcp-config", "/workspace/.mcp_config.json"])
            
            docker_cmd = [
                "docker", "run",
                "--rm",
                "--name", f"claude-agent-{run_id}",
                "--cap-add=NET_ADMIN",
                "--security-opt=seccomp=unconfined",
                "--network=bridge",
                "-v", f"{claude_memory_path}:/home/claude-agent/.claude",
                "-v", f"{working_dir_abs}:/workspace",
                "-e", f"MCP_PORT={mcp_sse_port or ''}",
                "-e", f"GAME_SERVER_PORT={server_url.rstrip('/').split(':')[-1]}",
                "-e", f"RUN_DATA_ID={run_id}",
                "claude-agent-devcontainer",
            ] + container_claude_cmd
            
            return docker_cmd, env, None, str(mcp_config_path)
        else:
            # Local mode: write MCP config to /tmp as before
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(mcp_config, temp_file, indent=2)
            temp_file.flush()
            temp_file.close()
            temp_mcp_config_path = temp_file.name
            claude_cmd.extend(["--mcp-config", temp_mcp_config_path])
            return claude_cmd, env, None, temp_mcp_config_path

    def handle_stream_event(
        self,
        event: dict,
        metrics: CliSessionMetrics | None,
        server_url: str | None = None,
        snapshot_path: Optional[Path] = None,
    ) -> None:
        etype = event.get("type", "")

        now = time.time()
        if etype == "system":
            self._last_event_time = now
            if metrics:
                metrics.session_id = event.get("session_id", "")
                metrics.model = event.get("model", "")
            logger.info(
                "[cli] session=%s model=%s tools=%d mcp_servers=%d",
                event.get("session_id", "?"),
                event.get("model", "?"),
                len(event.get("tools", [])),
                len(event.get("mcp_servers", [])),
            )

        elif etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "").strip()
                    if text:
                        duration_sec = (now - self._last_event_time) if self._last_event_time is not None else 0.0
                        self._last_event_time = now
                        preview = (text[:200] + "...") if len(text) > 200 else text
                        logger.info("[cli:text] %s", preview.replace("\n", " "))
                        # Detect expired OAuth token - abort run rather than looping forever
                        if "OAuth token has expired" in text or "authentication_error" in text:
                            logger.error("❌ AUTH ERROR: OAuth token expired in container. Run 'claude' on the host to refresh credentials, then retry.")
                            raise SystemExit(1)
                        # UI uses tool_use format only; text is not posted as thinking
                elif btype == "tool_use":
                    duration_sec = (now - self._last_event_time) if self._last_event_time is not None else 0.0
                    self._last_event_time = now
                    if metrics:
                        metrics.tool_use_count += 1
                    name = block.get("name", "?")
                    inp = block.get("input") or {}
                    if isinstance(inp, str):
                        try:
                            inp = json.loads(inp) if inp else {}
                        except json.JSONDecodeError:
                            inp = {}
                    reasoning = inp.get("reasoning") or inp.get("reason") or ""
                    # Short name for UI: e.g. mcp__pokemon-emerald__navigate_to -> navigate_to
                    short_name = name.split("__")[-1] if "__" in name else name
                    thinking_text = f"[{short_name}] {reasoning}".strip() or f"[{short_name}]"
                    if server_url:
                        self._post_thinking(server_url, thinking_text, duration_sec, interaction_type="ClaudeCodeBackend")
                    inp_preview = json.dumps(inp)
                    if len(inp_preview) > 120:
                        inp_preview = inp_preview[:120] + "..."
                    logger.info("[cli:tool_use] %s %s", name, inp_preview)

        elif etype == "user":
            self._last_event_time = now
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str):
                        preview = (content[:80] + "...") if len(content) > 80 else content
                    elif isinstance(content, list):
                        preview = f"[{len(content)} blocks]"
                    else:
                        preview = str(content)[:80]
                    logger.info(
                        "[cli:tool_result] ...%s -> %s",
                        block.get("tool_use_id", "?")[-8:],
                        preview.replace("\n", " "),
                    )

        elif etype == "result":
            if metrics:
                metrics.total_cost_usd = event.get("total_cost_usd", 0.0)
                metrics.num_turns = event.get("num_turns", 0)
                metrics.duration_ms = event.get("duration_ms", 0)
                metrics.duration_api_ms = event.get("duration_api_ms", 0)
                metrics.is_error = event.get("is_error", False)
                metrics.error = event.get("error", "")
                usage = event.get("usage", {})
                metrics.input_tokens = usage.get("input_tokens", 0)
                metrics.output_tokens = usage.get("output_tokens", 0)
                if snapshot_path:
                    try:
                        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(snapshot_path, "w") as f:
                            json.dump(asdict(metrics), f, indent=2)
                    except Exception as e:
                        logger.debug("Could not write CLI metrics snapshot: %s", e)
            logger.info(
                "[cli:result] cost=$%.4f turns=%d duration=%.1fs error=%s",
                event.get("total_cost_usd", 0),
                event.get("num_turns", 0),
                event.get("duration_ms", 0) / 1000,
                event.get("is_error", False),
            )

    def _post_thinking(
        self,
        server_url: str,
        thinking_text: str,
        duration_sec: float = 0.0,
        interaction_type: str = "ClaudeCodeBackend",
    ) -> None:
        """POST agent thinking to game server for UI streaming (same as VLM agents)."""
        try:
            import requests
            requests.post(
                f"{server_url}/agent_step",
                json={
                    "thinking": thinking_text,
                    "interaction_type": interaction_type,
                    "duration": duration_sec,
                },
                timeout=2,
            )
        except Exception as e:
            logger.debug("Could not POST thinking to server: %s", e)


def get_backend(cli_type: str) -> CliAgentBackend:
    """Return the backend for the given CLI type."""
    if cli_type == "claude":
        return ClaudeCodeBackend()
    if cli_type == "codex":
        raise NotImplementedError(
            "Codex CLI integration is not implemented yet. Use --cli-type claude for now."
        )
    raise ValueError(f"Unknown CLI type: {cli_type}. Supported: claude, codex")


def log_session_to_llm_logger(
    metrics: CliSessionMetrics,
    session_number: int,
    backend_name: str,
) -> None:
    """Record one CLI session into the shared LLMLogger / cumulative_metrics.json."""
    from utils.llm_logger import log_llm_interaction

    duration_sec = metrics.duration_ms / 1000.0 if metrics.duration_ms else None
    token_usage = {
        "prompt_tokens": metrics.input_tokens,
        "completion_tokens": metrics.output_tokens,
        "total_tokens": metrics.input_tokens + metrics.output_tokens,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
    }
    model_info = {"model": metrics.model or "claude-code", "backend": backend_name}
    response_summary = f"[{metrics.num_turns} turns, {metrics.tool_use_count} tool calls]"
    if metrics.is_error and metrics.error:
        response_summary += f" error={metrics.error[:100]}"

    log_llm_interaction(
        interaction_type=f"cli_{backend_name}",
        prompt="[CLI session]",
        response=response_summary,
        metadata={"token_usage": token_usage},
        duration=duration_sec,
        model_info=model_info,
        step_number=session_number,
    )

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
import stat
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

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
    auth_fatal_error: bool = False


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

    @property
    def agent_memory_subdir(self) -> str:
        """Cache subdir for agent memory (e.g. 'claude_memory'). Override for other backends."""
        return "claude_memory"

    @property
    def container_image(self) -> str:
        """Docker image name for containerized runs. Override for other backends."""
        return "claude-agent-devcontainer"

    @property
    def devcontainer_build_context(self) -> str:
        """Path to devcontainer build context (Dockerfile dir), relative to project root.
        Used for docker build -f {context}/Dockerfile {context}. Override for other backends."""
        return ".devcontainer/claude-agent"

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
    ) -> None:
        """Process one stream event (e.g. system, assistant, user, result). Update metrics and optionally POST thinking to server."""
        pass

    def is_auth_fatal_error(self, text: str) -> bool:
        """Return True if text indicates a fatal auth error (e.g. expired OAuth token).
        Override in subclasses for backend-specific detection. Default: False."""
        return False

    def seed_agent_auth(self, agent_memory_dir: Path) -> None:
        """Seed container agent memory with host auth files. Override in subclasses.
        Default: no-op (backends that don't need auth seeding do nothing)."""
        pass

    def run_stream_reader(
        self,
        stdout_pipe,
        stop_event: "threading.Event",
        log_file: io.TextIOWrapper | None,
        metrics: CliSessionMetrics | None,
        server_url: str | None = None,
    ) -> None:
        """Read stdout line-by-line (JSONL), tee to log_file, parse and handle events."""
        logger.debug("stream reader started: %s", self.name)
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
                self.handle_stream_event(event, metrics, server_url)
        except (OSError, ValueError) as e:
            logger.debug("stream reader error: %s", e)
        logger.debug("stream reader exiting: %s", self.name)


class ClaudeCodeBackend(CliAgentBackend):
    """Backend for Anthropic Claude Code CLI (--print --output-format stream-json)."""

    # Container paths (Codex/Gemini backends would use their own, e.g. /home/codex-agent/.codex)
    WORKSPACE_PATH = "/workspace"
    AGENT_MEMORY_PATH = "/home/claude-agent/.claude"

    @property
    def agent_memory_subdir(self) -> str:
        return "claude_memory"

    @property
    def container_image(self) -> str:
        return "claude-agent-devcontainer"

    @property
    def devcontainer_build_context(self) -> str:
        return ".devcontainer/claude-agent"

    DIRECTIVE_FILENAME = ".agent_directive.txt"
    MCP_CONFIG_FILENAME = ".mcp_config.json"

    def __init__(self) -> None:
        super().__init__()
        self._last_event_time: float | None = None  # for thinking duration delta

    @property
    def name(self) -> str:
        return "claude"

    def is_auth_fatal_error(self, text: str) -> bool:
        """Detect Claude Code OAuth token expiration or not logged in. See anthropics/claude-code#18225."""
        return (
            "OAuth token has expired" in text
            or "authentication_error" in text
            or "401" in text
            or "Not logged in" in text
            or "Please run /login" in text
        )

    def seed_agent_auth(self, agent_memory_dir: Path) -> None:
        """Seed container agent memory with host Claude auth files."""
        import shutil

        host_claude_dir = Path.home() / ".claude"
        host_claude_json = Path.home() / ".claude.json"
        if not host_claude_dir.exists() and not host_claude_json.exists():
            print("⚠️  Host ~/.claude/ not found. Run 'claude auth login' first.")
            print("   Container will not be able to authenticate.")
            return

        auth_files = [
            (host_claude_dir / "settings.json", "settings.json"),
            (host_claude_dir / ".credentials.json", ".credentials.json"),
            (host_claude_json, ".claude.json"),
        ]
        seeded_any = False
        for src, dst_name in auth_files:
            if src.exists():
                dst = agent_memory_dir / dst_name
                shutil.copy2(src, dst)
                print(f"   ✓ Seeded {src.name} -> {dst_name}")
                seeded_any = True
        if not seeded_any:
            print("⚠️  No Claude auth files found")
            print("   Run 'claude auth login' first, then retry.")

    def _build_bootstrap_content(self, directive_path: str, server_url: str) -> str:
        """Build bootstrap prompt string from directive file and server URL."""
        directive_content = ""
        if directive_path and os.path.exists(directive_path):
            with open(directive_path, "r") as f:
                directive_content = f.read()
        if directive_content:
            return (
                f"{directive_content}\n\n"
                "Runtime context:\n"
                f"- Pokemon server URL: {server_url}\n"
                "- You are running in a long-lived interactive session.\n"
                "- Act autonomously and continuously, using MCP tools directly as needed.\n"
                "- Poll game state on your own via MCP tools; do not wait for additional operator prompts.\n"
                "- Continue until externally terminated by the orchestrator when completion condition is met.\n"
            )
        return "Start the Pokemon Emerald agent session."

    def _build_mcp_config_sse(self, mcp_sse_port: int) -> dict:
        """Build MCP config dict for SSE (containerized) mode. type=sse is REQUIRED.
        Uses host.docker.internal to reach the MCP server on the host (bridge network)."""
        host = "host.docker.internal"
        return {
            "mcpServers": {
                "pokemon-emerald": {
                    "type": "sse",
                    "url": f"http://{host}:{mcp_sse_port}/sse",
                }
            }
        }

    def _build_mcp_config_stdio(
        self, server_url: str, pythonpath: str
    ) -> dict:
        """Build MCP config dict for stdio (local) mode."""
        return {
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

    def _build_cli_base_args(
        self,
        *,
        dangerously_skip_permissions: bool = True,
        resume_session_id: str | None = None,
        thinking_effort: str | None = None,
    ) -> list[str]:
        """Build base CLI args (output format, verbose, disallowed tools, permissions, resume, thinking)."""
        args = [
            "--output-format", "stream-json",
            "--verbose",
            "--disallowedTools", "AskUserQuestion,EnterPlanMode,ExitPlanMode",
        ]
        if dangerously_skip_permissions:
            args.append("--dangerously-skip-permissions")
        if resume_session_id:
            args.extend(["--resume", resume_session_id])
        if thinking_effort and thinking_effort in ("low", "medium", "high"):
            args.extend(["--thinking-budget", thinking_effort])
        return args

    def _write_workspace_files(
        self,
        working_dir: Path,
        bootstrap: str,
        mcp_config: dict,
    ) -> Path:
        """Write .agent_directive.txt and .mcp_config.json to workspace, set both read-only.

        Idempotent: skips if both files already exist and are read-only (from a previous
        session). Fixes PermissionError when restarting the agent after usage limit or
        other exit, since we no longer try to overwrite read-only files.
        """
        working_dir = Path(working_dir).resolve()
        working_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_file = working_dir / self.DIRECTIVE_FILENAME
        mcp_config_path = working_dir / self.MCP_CONFIG_FILENAME

        def _is_readonly(p: Path) -> bool:
            return p.exists() and not (p.stat().st_mode & stat.S_IWUSR)

        if _is_readonly(bootstrap_file) and _is_readonly(mcp_config_path):
            return mcp_config_path

        bootstrap_file.write_text(bootstrap)
        mcp_json = json.dumps(mcp_config, indent=2)
        with open(mcp_config_path, "w") as f:
            f.write(mcp_json)
            f.flush()
            os.fsync(f.fileno())
        bootstrap_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        mcp_config_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        return mcp_config_path

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
        agent_memory_dir: str | None = None,
    ) -> tuple[list[str], dict[str, str], str, str | None]:
        env = os.environ.copy()
        env["POKEMON_MCP_SERVER_URL"] = server_url
        env["POKEMON_SERVER_URL"] = server_url

        pythonpath = project_root if project_root else os.getcwd()
        bootstrap = self._build_bootstrap_content(directive_path, server_url)
        base_args = self._build_cli_base_args(
            dangerously_skip_permissions=dangerously_skip_permissions,
            resume_session_id=resume_session_id,
            thinking_effort=thinking_effort,
        )

        if containerized:
            if not run_id or not agent_memory_dir:
                raise ValueError("containerized mode requires run_id and agent_memory_dir")
            if not mcp_sse_port:
                raise ValueError("containerized mode requires mcp_sse_port")

            agent_memory_path = Path(agent_memory_dir).resolve()
            working_dir_abs = Path(working_dir).resolve()

            mcp_config = self._build_mcp_config_sse(mcp_sse_port)
            # Write both files into the workspace directory — the container mounts this as /workspace.
            # @file avoids shell mangling; .mcp_config.json is read-only to prevent agent tampering.
            # "type": "sse" is REQUIRED — Claude Code silently exits without output if absent.
            mcp_config_path = self._write_workspace_files(working_dir_abs, bootstrap, mcp_config)

            directive_arg = f"@{self.WORKSPACE_PATH}/{self.DIRECTIVE_FILENAME}"
            mcp_config_arg = f"{self.WORKSPACE_PATH}/{self.MCP_CONFIG_FILENAME}"
            claude_cmd = ["claude", "--print", directive_arg] + base_args + [
                "--mcp-config", mcp_config_arg,
            ]

            game_port = server_url.rstrip("/").split(":")[-1]
            # CLAUDE_CONFIG_DIR is required so OAuth credentials load from the mounted .claude dir.
            # Bridge network with host.docker.internal to reach the MCP SSE server on the host.
            docker_cmd = [
                "docker", "run", "--rm",
                "--name", f"claude-agent-{run_id}",
                "--cap-add=NET_ADMIN",
                "--security-opt=seccomp=unconfined",
                "--network=bridge",
                "--add-host=host.docker.internal:host-gateway",
                "-v", f"{agent_memory_path}:{self.AGENT_MEMORY_PATH}",
                "-v", f"{working_dir_abs}:{self.WORKSPACE_PATH}",
                "-e", f"MCP_PORT={mcp_sse_port}",
                "-e", f"GAME_SERVER_PORT={game_port}",
                "-e", f"RUN_DATA_ID={run_id}",
                "-e", f"CLAUDE_CONFIG_DIR={self.AGENT_MEMORY_PATH}",
                self.container_image,
            ] + claude_cmd

            return docker_cmd, env, bootstrap, None

        else:
            mcp_config = self._build_mcp_config_stdio(server_url, pythonpath)
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(mcp_config, temp_file, indent=2)
            temp_file.flush()
            temp_file.close()
            temp_mcp_config_path = temp_file.name

            claude_cmd = ["claude", "--print", bootstrap] + base_args + [
                "--mcp-config", temp_mcp_config_path,
            ]
            return claude_cmd, env, bootstrap, temp_mcp_config_path

    def _handle_assistant_text_block(self, block: dict, now: float, metrics: CliSessionMetrics | None = None) -> None:
        """Handle assistant text block: OAuth check + logging. Raises SystemExit(1) on auth error."""
        text = block.get("text", "").strip()
        if not text:
            return
        duration_sec = (now - self._last_event_time) if self._last_event_time is not None else 0.0
        self._last_event_time = now
        preview = (text[:200] + "...") if len(text) > 200 else text
        logger.info("[cli:text] %s", preview.replace("\n", " "))
        if self.is_auth_fatal_error(text):
            if metrics:
                metrics.auth_fatal_error = True
            logger.error(
                "❌ AUTH ERROR: OAuth token expired in container. "
                "Run 'claude auth login' on the host to refresh credentials, then retry."
            )
            raise SystemExit(1)

    def _handle_assistant_tool_use_block(
        self,
        block: dict,
        now: float,
        metrics: CliSessionMetrics | None,
        server_url: str | None,
    ) -> None:
        """Handle assistant tool_use block: metrics, _post_thinking, logging."""
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
        short_name = name.split("__")[-1] if "__" in name else name
        thinking_text = f"[{short_name}] {reasoning}".strip() or f"[{short_name}]"
        if server_url:
            self._post_thinking(server_url, thinking_text, duration_sec, interaction_type=self.name)
        inp_preview = json.dumps(inp)
        if len(inp_preview) > 120:
            inp_preview = inp_preview[:120] + "..."
        logger.info("[cli:tool_use] %s %s", name, inp_preview)

    def _handle_user_tool_result_block(self, block: dict) -> None:
        """Handle user tool_result block: logging."""
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

    def handle_stream_event(
        self,
        event: dict,
        metrics: CliSessionMetrics | None,
        server_url: str | None = None,
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
                    self._handle_assistant_text_block(block, now, metrics)
                elif btype == "tool_use":
                    self._handle_assistant_tool_use_block(block, now, metrics, server_url)

        elif etype == "user":
            self._last_event_time = now
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    self._handle_user_tool_result_block(block)

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



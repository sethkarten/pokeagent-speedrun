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
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

    @abstractmethod
    def log_cli_interaction(
        self,
        agent_memory_dir: Path,
        processed_hashes: set,
        last_cli_step: int,
        server_url: str | None = None,
    ) -> tuple[set, int]:
        """Log CLI interaction (steps/metrics) from backend-specific source (file, stream, etc.).

        Returns updated (processed_hashes, last_cli_step).
        """
        pass

    @abstractmethod
    def get_resume_session_id(self, agent_memory_dir: Path) -> str | None:
        """Return the session ID to resume from, or None if not found/supported."""
        pass

    @abstractmethod
    def run_login(self) -> bool:
        """Run interactive login flow. Returns True on success."""
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

    def _post_thinking(
        self,
        server_url: str,
        thinking_text: str,
        duration_sec: float = 0.0,
        interaction_type: str = "cli",
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

    def _write_workspace_files(
        self,
        working_dir: Path,
        bootstrap: str,
        mcp_config: dict,
        directive_filename: str = ".agent_directive.txt",
        mcp_config_filename: str = ".mcp_config.json",
    ) -> Path:
        """Write directive and MCP config to workspace, set both read-only.

        Idempotent: skips if both files already exist and are read-only.
        """
        working_dir = Path(working_dir).resolve()
        working_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_file = working_dir / directive_filename
        mcp_config_path = working_dir / mcp_config_filename

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

    @staticmethod
    def _sync_metrics_to_server(server_url: str) -> None:
        """Push run_cli's in-memory cumulative metrics to the server (single-writer pattern)."""
        try:
            import requests
            from utils.data_persistence.llm_logger import get_llm_logger

            llm_logger = get_llm_logger()
            metrics = llm_logger.get_cumulative_metrics()
            step_count = len(metrics.get("steps", []))
            if step_count == 0:
                return
            resp = requests.post(
                f"{server_url}/sync_llm_metrics",
                json={"cumulative_metrics": metrics},
                timeout=5,
            )
            if resp.status_code != 200:
                logger.warning("Sync metrics failed: %s %s", resp.status_code, resp.text[:200])
            else:
                logger.info("Synced %d step(s) to server", step_count)
        except Exception as e:
            logger.warning("Could not sync metrics to server: %s", e)


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
        return "ClaudeCode"

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

        use_openrouter = getattr(self, "api_gateway", "login") == "openrouter" and os.environ.get("OPENROUTER_API_KEY")
        if use_openrouter:
            print("   Using OPENROUTER_API_KEY; skipping host OAuth credential copy so container uses API key.")
            return

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
            ]

            use_openrouter = getattr(self, "api_gateway", "login") == "openrouter"
            openrouter_key = os.environ.get("OPENROUTER_API_KEY") if use_openrouter else None
            if openrouter_key:
                # OpenRouter: inject required vars for Claude Code (ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN)
                docker_cmd.extend([
                    "-e", "ANTHROPIC_BASE_URL=https://openrouter.ai/api",
                    "-e", f"ANTHROPIC_AUTH_TOKEN={openrouter_key}",
                    "-e", "ANTHROPIC_API_KEY=",
                ])
            else:
                # Direct Anthropic: pass through env vars if user set them
                for env_var in ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"]:
                    if os.environ.get(env_var):
                        docker_cmd.extend(["-e", f"{env_var}={os.environ[env_var]}"])

            docker_cmd.append(self.container_image)
            docker_cmd.extend(claude_cmd)

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

    # ------------------------------------------------------------------
    # Model name normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_model_name(raw: str) -> str:
        """Normalize provider-prefixed model names to canonical pricing-table form.

        OpenRouter: 'anthropic/claude-4.6-sonnet-20260217' -> 'claude-sonnet-4.6'
        Direct:     'claude-sonnet-4-6' -> 'claude-sonnet-4-6' (unchanged)
        """
        if "/" in raw:
            raw = raw.split("/", 1)[1]
        raw = re.sub(r"-\d{8}$", "", raw)
        m = re.match(r"^claude-(\d+(?:\.\d+)?)-(\w+)$", raw)
        if m:
            version, family = m.groups()
            return f"claude-{family}-{version}"
        return raw

    # ------------------------------------------------------------------
    # JSONL polling (metric tracking for CLI agent runs)
    # ------------------------------------------------------------------

    def log_cli_interaction(
        self,
        agent_memory_dir: Path,
        processed_hashes: set,
        last_cli_step: int,
        server_url: str | None = None,
    ) -> tuple[set, int]:
        """Poll Claude Code JSONL files and append new entries as steps to cumulative_metrics.

        Returns updated (processed_hashes, last_cli_step).  Never raises -- all
        errors are logged as warnings so the agent loop is never interrupted.
        """
        from utils.metric_tracking.claude_session_reader import find_jsonl_files, load_new_usage_entries
        from utils.data_persistence.llm_logger import get_llm_logger

        search_path = Path(agent_memory_dir).resolve()
        # Fallback to projects/-workspace/ for Claude
        if not search_path.is_dir():
            # If the base dir exists, look for the projects subdirectory
            # run_cli passes agent_memory_subdir, which is claude_memory
            # Claude Code stores logs in projects/<project_id>/*.jsonl
            # For simplicity in this method, we expect the caller or the reader to handle recursive search
            # But Claude Code defaults: ~/.claude/projects/-workspace/
            # If agent_memory_dir is the root (~/.claude), we search there.
            # Ideally we pass the exact path. run_cli passes get_cache_path(backend.agent_memory_subdir).
            pass

        if not search_path.is_dir():
             logger.warning("JSONL poll: agent_memory_dir not a directory: %s", search_path)
             return processed_hashes, last_cli_step

        # Claude logic: look inside projects/-workspace if it exists
        claude_workspace = search_path / "projects" / "-workspace"
        target_path = claude_workspace if claude_workspace.is_dir() else search_path

        try:
            new_entries, processed_hashes = load_new_usage_entries(target_path, processed_hashes)
        except Exception as exc:
            logger.warning("JSONL poll failed: %s", exc)
            return processed_hashes, last_cli_step

        jsonl_count = len(find_jsonl_files(target_path))
        if not new_entries:
            if jsonl_count > 0:
                logger.debug(
                    "JSONL poll: %d file(s) under %s, 0 new entries (already processed or no usage)",
                    jsonl_count, target_path,
                )
            return processed_hashes, last_cli_step

        logger.info("JSONL poll: appending %d step(s) from %s", len(new_entries), target_path)

        new_entries.sort(key=lambda e: (e["_parsed_timestamp"] or datetime.min.replace(tzinfo=timezone.utc)))

        llm_logger = get_llm_logger()
        # Use last step's timestamp so first entry of this poll gets correct duration (not 0.0)
        steps = llm_logger.cumulative_metrics.get("steps", [])
        prev_ts: float | None = steps[-1]["timestamp"] if steps else None
        for entry in new_entries:
            tokens = entry["_tokens"]
            tool_calls = entry["_tool_calls"]
            parsed_ts = entry.get("_parsed_timestamp")
            ts_float = parsed_ts.timestamp() if parsed_ts is not None else time.time()
            duration = max(0.0, ts_float - prev_ts) if prev_ts is not None else 0.0
            prev_ts = ts_float

            msg = entry.get("message")
            raw_model = msg.get("model") if isinstance(msg, dict) else None
            model_name = self.normalize_model_name(raw_model) if raw_model else "claude-code"
            model_info = {"model": model_name}

            last_cli_step += 1
            try:
                llm_logger.append_cli_step(
                    step_number=last_cli_step,
                    token_usage=tokens,
                    duration=duration,
                    timestamp=ts_float,
                    model_info=model_info,
                    tool_calls=tool_calls if tool_calls else None,
                )
            except Exception as exc:
                logger.warning("Failed to append CLI step %d: %s", last_cli_step, exc)
                last_cli_step -= 1

        logger.debug("JSONL poll: appended %d new step(s), last_step=%d", len(new_entries), last_cli_step)
        if server_url and new_entries:
            self._sync_metrics_to_server(server_url)
        return processed_hashes, last_cli_step

    def get_resume_session_id(self, agent_memory_dir: Path) -> str | None:
        """Find the most recent session ID from Claude memory."""
        # Fallback: derive from most recent project in claude_memory (for older backups)
        # Structure: agent_memory_dir / projects / -workspace / *.jsonl
        projects_dir = agent_memory_dir / "projects" / "-workspace"
        if projects_dir.exists():
            jsonl_files = [f for f in projects_dir.iterdir() if f.suffix == ".jsonl"]
            if jsonl_files:
                most_recent = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                return most_recent.stem
        return None

    def run_login(self) -> bool:
        """Run 'claude auth login' interactively. Returns True on success."""
        print("\n🔐 Running 'claude auth login' (interactive)...")
        try:
            result = subprocess.run(["claude", "auth", "login"], check=False)
            if result.returncode != 0:
                print(f"❌ Claude auth login failed (exit code {result.returncode})")
                return False
            print("✅ Claude auth login succeeded")
            return True
        except FileNotFoundError:
            print("❌ Claude CLI not found. Install it first, then retry with --login.")
            return False
        except Exception as e:
            print(f"❌ Claude auth login error: {e}")
            return False


class GeminiCliBackend(CliAgentBackend):
    """Backend for Google Gemini CLI (headless --output-format stream-json).

    Key differences from Claude Code:
    - Auth via GEMINI_API_KEY env var (no OAuth/login flow)
    - MCP + telemetry + yolo configured via settings.json (not CLI flags)
    - Telemetry outfile provides per-API-request step granularity
    - Sessions stored in ~/.gemini/tmp/<workspace_id>/chats/
    """

    WORKSPACE_PATH = "/workspace"
    AGENT_MEMORY_PATH = "/home/gemini-agent/.gemini"
    TELEMETRY_FILENAME = "telemetry.jsonl"

    @property
    def name(self) -> str:
        return "GeminiCLI"

    @property
    def agent_memory_subdir(self) -> str:
        return "gemini_memory"

    @property
    def container_image(self) -> str:
        return "gemini-agent-devcontainer"

    @property
    def devcontainer_build_context(self) -> str:
        return ".devcontainer/gemini-agent"

    def __init__(self) -> None:
        super().__init__()
        self._last_event_time: float | None = None

    # ------------------------------------------------------------------
    # MCP config overrides (Gemini format: no "type" key, adds "trust")
    # ------------------------------------------------------------------

    def _build_mcp_config_sse(self, mcp_sse_port: int) -> dict:
        host = "host.docker.internal"
        return {
            "mcpServers": {
                "pokemon-emerald": {
                    "url": f"http://{host}:{mcp_sse_port}/sse",
                    "trust": True,
                }
            }
        }

    def _build_mcp_config_stdio(self, server_url: str, pythonpath: str) -> dict:
        return {
            "mcpServers": {
                "pokemon-emerald": {
                    "command": sys.executable,
                    "args": ["-m", "server.cli.pokemon_mcp_server"],
                    "env": {
                        "POKEMON_SERVER_URL": server_url,
                        "PYTHONPATH": pythonpath,
                    },
                    "trust": True,
                }
            }
        }

    # ------------------------------------------------------------------
    # Settings.json generation (MCP + telemetry + yolo)
    # ------------------------------------------------------------------

    def _build_settings(
        self,
        mcp_config: dict,
        telemetry_outfile: str,
        thinking_effort: str | None = None,
    ) -> dict:
        """Build a complete Gemini CLI settings.json."""
        settings = {
            **mcp_config,
            "model": {
                # Limit compression threshold to 0.25 * 1M tokens to avoid timeouts
                "compressionThreshold": 0.25,
            },
            "telemetry": {
                "enabled": False, # session-based step tracking, akin to claude code implementation for metrics
                "target": "local",
                "otlpEndpoint": "",
                "outfile": telemetry_outfile,
            },
        }
        if thinking_effort in ("low", "medium", "high"):
            level = thinking_effort.upper()
            budget = {"low": 512, "medium": 2048, "high": 8192}[thinking_effort]
            settings["modelConfigs"] = {
                "customOverrides": [
                    # Specify new overrides as additional gemini models are released
                    {"model": "gemini-3.1-pro-preview", "generateContentConfig": {"thinkingConfig": {"thinkingLevel": level}}},
                    {"model": "gemini-3.1-pro-preview-customtools", "generateContentConfig": {"thinkingConfig": {"thinkingLevel": level}}},
                    {"model": "gemini-3-flash-preview", "generateContentConfig": {"thinkingConfig": {"thinkingLevel": level}}},
                    {"model": "gemini-3-pro-preview", "generateContentConfig": {"thinkingConfig": {"thinkingLevel": level}}},
                    {"model": "gemini-2.5-flash", "generateContentConfig": {"thinkingConfig": {"thinkingBudget": budget}}},
                    {"model": "gemini-2.5-pro", "generateContentConfig": {"thinkingConfig": {"thinkingBudget": budget}}},
                ],
            }
        return settings

    def _write_gemini_settings(
        self,
        settings_dir: Path,
        mcp_config: dict,
        telemetry_outfile: str,
        thinking_effort: str | None = None,
    ) -> Path:
        """Write settings.json into the given directory. Returns the path."""
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings = self._build_settings(mcp_config, telemetry_outfile, thinking_effort)
        settings_path = settings_dir / "settings.json"

        if settings_path.exists():
            try:
                existing = json.loads(settings_path.read_text())
                existing.update(settings)
                settings = existing
            except (json.JSONDecodeError, OSError):
                pass

        settings_path.write_text(json.dumps(settings, indent=2))
        logger.info("Wrote Gemini settings.json: %s", settings_path)
        return settings_path

    # ------------------------------------------------------------------
    # build_launch_cmd
    # ------------------------------------------------------------------

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

        bootstrap_content = self._build_bootstrap_content(directive_path, server_url)

        if containerized:
            if not run_id or not agent_memory_dir:
                raise ValueError("containerized mode requires run_id and agent_memory_dir")
            if not mcp_sse_port:
                raise ValueError("containerized mode requires mcp_sse_port")

            agent_memory_path = Path(agent_memory_dir).resolve()
            working_dir_abs = Path(working_dir).resolve()

            mcp_config = self._build_mcp_config_sse(mcp_sse_port)
            telemetry_outfile = f"{self.AGENT_MEMORY_PATH}/{self.TELEMETRY_FILENAME}"

            self._write_gemini_settings(agent_memory_path, mcp_config, telemetry_outfile, thinking_effort)

            # Also write the directive into the workspace for reference
            self._write_workspace_files(working_dir_abs, bootstrap_content, mcp_config)

            game_port = server_url.rstrip("/").split(":")[-1]

            docker_cmd = [
                "docker", "run", "--rm",
                "--name", f"gemini-agent-{run_id}",
                "--cap-add=NET_ADMIN",
                "--security-opt=seccomp=unconfined",
                "--network=bridge",
                "--add-host=host.docker.internal:host-gateway",
                "-v", f"{agent_memory_path}:{self.AGENT_MEMORY_PATH}",
                "-v", f"{working_dir_abs}:{self.WORKSPACE_PATH}",
                "-w", self.WORKSPACE_PATH,
                "-e", f"MCP_PORT={mcp_sse_port}",
                "-e", f"GAME_SERVER_PORT={game_port}",
                "-e", f"RUN_DATA_ID={run_id}",
            ]

            for env_var in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
                if os.environ.get(env_var):
                    docker_cmd.extend(["-e", f"{env_var}={os.environ[env_var]}"])

            docker_cmd.append(self.container_image)
            # Pass prompt via stdin to avoid shell mangling of multi-line content.
            # The directive is already written to /workspace/.agent_directive.txt by _write_workspace_files.
            # Single-quote the entire command so init-firewall's "su -c ... $*" does not interpret the pipe.
            _inner = "cat /workspace/.agent_directive.txt | gemini --yolo --output-format stream-json"
            if resume_session_id:
                _inner += " --resume " + shlex.quote(resume_session_id)
            shell_cmd = "'" + _inner.replace("'", "'\"'\"'") + "'"
            docker_cmd.extend(["sh", "-c", shell_cmd])

            return docker_cmd, env, bootstrap_content, None

        else:
            pythonpath = project_root if project_root else os.getcwd()
            mcp_config = self._build_mcp_config_stdio(server_url, pythonpath)

            # Write project-level .gemini/settings.json in the working dir
            gemini_settings_dir = Path(working_dir) / ".gemini"
            telemetry_path = str(gemini_settings_dir / self.TELEMETRY_FILENAME)
            self._write_gemini_settings(gemini_settings_dir, mcp_config, telemetry_path, thinking_effort)

            gemini_cmd = [
                "gemini",
                "--yolo",
                "--output-format", "stream-json",
                "-p", bootstrap_content,
            ]
            if resume_session_id:
                gemini_cmd.extend(["--resume", resume_session_id])
            return gemini_cmd, env, bootstrap_content, None

    # ------------------------------------------------------------------
    # handle_stream_event (real-time stream-json parsing)
    # ------------------------------------------------------------------

    def _handle_gemini_init(
        self,
        event: dict,
        now: float,
        metrics: CliSessionMetrics | None,
    ) -> None:
        """Handle init event: session metadata."""
        self._last_event_time = now
        if metrics:
            metrics.session_id = event.get("session_id", "")
            metrics.model = event.get("model", "")
        logger.info(
            "[gemini] session=%s model=%s",
            event.get("session_id", "?"),
            event.get("model", "?"),
        )

    def _handle_gemini_tool_use(
        self,
        event: dict,
        now: float,
        metrics: CliSessionMetrics | None,
        server_url: str | None,
    ) -> None:
        """Handle tool_use event: metrics, _post_thinking, logging."""
        duration_sec = (now - self._last_event_time) if self._last_event_time is not None else 0.0
        self._last_event_time = now
        if metrics:
            metrics.tool_use_count += 1
        tool_name = event.get("tool_name") or event.get("name", "?")
        args = event.get("parameters") or event.get("arguments") or {}
        args_preview = json.dumps(args)
        if len(args_preview) > 120:
            args_preview = args_preview[:120] + "..."
        logger.info("[gemini:tool_use] %s %s", tool_name, args_preview)
        reasoning = args.get("reasoning") or args.get("reason") or ""
        thinking_text = f"[{tool_name}] {reasoning}".strip() or f"[{tool_name}]"
        if server_url:
            self._post_thinking(server_url, thinking_text, duration_sec, interaction_type=self.name)

    def _handle_gemini_tool_result(self, event: dict, now: float) -> None:
        """Handle tool_result event: logging."""
        self._last_event_time = now
        content = event.get("output") or event.get("content", "")
        preview = (str(content)[:80] + "...") if len(str(content)) > 80 else str(content)
        logger.info("[gemini:tool_result] %s", preview.replace("\n", " "))

    def _handle_gemini_message(self, event: dict, now: float) -> None:
        """Handle message event: logging (delta chunks at DEBUG)."""
        self._last_event_time = now
        role = event.get("role", "")
        text = event.get("content", "")
        if isinstance(text, str) and text:
            preview = (text[:240] + "…") if len(text) > 240 else text
            if event.get("delta") and role == "assistant":
                logger.debug("[gemini:%s] %s", role, preview.replace("\n", " "))
            else:
                logger.info("[gemini:%s] %s", role, preview.replace("\n", " "))

    def _handle_gemini_error(self, event: dict) -> None:
        """Handle error event: logging."""
        error_msg = event.get("message") or event.get("error") or str(event)
        logger.warning("[gemini:error] %s", error_msg)

    def _handle_gemini_result(
        self,
        event: dict,
        metrics: CliSessionMetrics | None,
    ) -> None:
        """Handle result event: final metrics."""
        if metrics:
            stats = event.get("stats") or {}
            metrics.input_tokens = stats.get("input_tokens", 0)
            metrics.output_tokens = stats.get("output_tokens", 0)
            metrics.duration_ms = stats.get("duration_ms", 0)
            metrics.is_error = event.get("is_error", False)
            metrics.error = event.get("error", "")
        logger.info(
            "[gemini:result] error=%s tokens_in=%d tokens_out=%d",
            event.get("is_error", False),
            (event.get("stats") or {}).get("input_tokens", 0),
            (event.get("stats") or {}).get("output_tokens", 0),
        )

    def handle_stream_event(
        self,
        event: dict,
        metrics: CliSessionMetrics | None,
        server_url: str | None = None,
    ) -> None:
        """Dispatch stream events to typed handlers (consistent with Claude backend)."""
        etype = event.get("type", "")
        now = time.time()

        if etype == "init":
            self._handle_gemini_init(event, now, metrics)
        elif etype == "tool_use":
            self._handle_gemini_tool_use(event, now, metrics, server_url)
        elif etype == "tool_result":
            self._handle_gemini_tool_result(event, now)
        elif etype == "message":
            self._handle_gemini_message(event, now)
        elif etype == "error":
            self._handle_gemini_error(event)
        elif etype == "result":
            self._handle_gemini_result(event, metrics)

    # ------------------------------------------------------------------
    # log_cli_interaction (session-based step tracking, like Claude)
    # ------------------------------------------------------------------

    def log_cli_interaction(
        self,
        agent_memory_dir: Path,
        processed_hashes: set,
        last_cli_step: int,
        server_url: str | None = None,
    ) -> tuple[set, int]:
        """Poll Gemini session JSON files for new gemini messages and append steps."""
        from utils.metric_tracking.gemini_session_reader import load_new_usage_entries
        from utils.data_persistence.llm_logger import get_llm_logger

        search_path = Path(agent_memory_dir).resolve()
        if not search_path.is_dir():
            return processed_hashes, last_cli_step

        try:
            new_entries, updated_hashes = load_new_usage_entries(search_path, processed_hashes)
        except Exception as exc:
            logger.warning("Gemini session poll failed: %s", exc)
            return processed_hashes, last_cli_step

        if not new_entries:
            return updated_hashes, last_cli_step

        logger.info("Gemini session: appending %d step(s)", len(new_entries))
        new_entries.sort(key=lambda e: (e.get("_parsed_timestamp") or datetime.min.replace(tzinfo=timezone.utc)))

        llm_logger = get_llm_logger()
        # Use last step's timestamp so first entry of this poll gets correct duration (not 0.0)
        steps = llm_logger.cumulative_metrics.get("steps", [])
        prev_ts: float | None = steps[-1]["timestamp"] if steps else None
        for entry in new_entries:
            tokens = entry.get("_tokens", {})
            tool_calls = entry.get("_tool_calls", [])
            parsed_ts = entry.get("_parsed_timestamp")
            ts_float = parsed_ts.timestamp() if parsed_ts is not None else time.time()
            duration = max(0.0, ts_float - prev_ts) if prev_ts is not None else 0.0
            prev_ts = ts_float

            model_name = entry.get("_model", "gemini-pro")
            model_info = {"model": model_name}

            last_cli_step += 1
            try:
                llm_logger.append_cli_step(
                    step_number=last_cli_step,
                    token_usage=tokens,
                    duration=duration,
                    timestamp=ts_float,
                    model_info=model_info,
                    tool_calls=tool_calls if tool_calls else None,
                )
            except Exception as exc:
                logger.warning("Failed to append Gemini CLI step %d: %s", last_cli_step, exc)
                last_cli_step -= 1

        if server_url and new_entries:
            self._sync_metrics_to_server(server_url)

        return updated_hashes, last_cli_step

    # ------------------------------------------------------------------
    # get_resume_session_id
    # ------------------------------------------------------------------

    def get_resume_session_id(self, agent_memory_dir: Path) -> str | None:
        """Find the most recent Gemini session ID from chat history.

        Gemini stores sessions in ~/.gemini/tmp/<workspace_id>/chats/<session_id>.json.
        We look for the most recently modified .json file in any chats/ subdirectory.
        """
        tmp_dir = agent_memory_dir / "tmp"
        if not tmp_dir.is_dir():
            return None
        chat_files: list[Path] = []
        for chats_dir in tmp_dir.rglob("chats"):
            if chats_dir.is_dir():
                chat_files.extend(f for f in chats_dir.iterdir() if f.suffix == ".json")
        if not chat_files:
            return None
        most_recent = max(chat_files, key=lambda p: p.stat().st_mtime)
        return most_recent.stem

    # ------------------------------------------------------------------
    # run_login (no-op for API key auth)
    # ------------------------------------------------------------------

    def run_login(self) -> bool:
        print("\nℹ️  Gemini uses GEMINI_API_KEY from environment. No interactive login required.")
        return True


class CodexCliBackend(CliAgentBackend):
    """Backend for OpenAI Codex CLI (codex exec --json, non-interactive only).

    Uses exec mode exclusively for JSONL stdout and resume support.
    Config and sessions live under ~/.codex (CODEX_HOME); MCP via config.toml.
    """

    WORKSPACE_PATH = "/workspace"
    AGENT_MEMORY_PATH = "/home/codex-agent/.codex"
    DIRECTIVE_FILENAME = ".agent_directive.txt"
    CONFIG_FILENAME = "config.toml"
    SESSIONS_SUBDIR = "sessions"

    @property
    def name(self) -> str:
        return "CodexCLI"

    @property
    def agent_memory_subdir(self) -> str:
        return "codex_memory"

    @property
    def container_image(self) -> str:
        return "codex-agent-devcontainer"

    @property
    def devcontainer_build_context(self) -> str:
        return ".devcontainer/codex-agent"

    def __init__(self) -> None:
        super().__init__()
        self._last_event_time: float | None = None
        self._pending_reasoning: str = ""  # Buffer for reasoning block; merged into next tool call

    def is_auth_fatal_error(self, text: str) -> bool:
        """Detect Codex auth failures (OpenRouter/API key or ChatGPT login)."""
        return (
            "not logged in" in text.lower()
            or "401" in text
            or "authentication" in text.lower()
            or "expired" in text.lower()
            or "invalid api key" in text.lower()
        )

    def seed_agent_auth(self, agent_memory_dir: Path) -> None:
        """Seed container with host Codex auth (credentials, config) if present."""
        import shutil

        use_openrouter = getattr(self, "api_gateway", "login") == "openrouter"
        if use_openrouter:
            print("   Using OPENROUTER_API_KEY; skipping host credential copy so container uses API key.")
            return

        host_codex = Path.home() / ".codex"
        if not host_codex.exists():
            print("⚠️  Host ~/.codex not found.")
            print("   Use --api-gateway openrouter (with OPENROUTER_API_KEY) or run 'codex login' for ChatGPT auth.")
            return

        # Copy auth-related files; avoid overwriting our MCP config
        for name in ["auth.json", "credentials.json"]:
            src = host_codex / name
            if src.exists():
                dst = agent_memory_dir / name
                shutil.copy2(src, dst)
                print(f"   ✓ Seeded {name}")

    def _ensure_codex_config(
        self,
        agent_memory_dir: Path,
        mcp_sse_url: str | None = None,
        mcp_sse_port: int = 8002,
        *,
        server_url: str | None = None,
        project_root: str | None = None,
    ) -> Path:
        """Ensure config.toml exists with MCP config. Use SSE url when containerized, stdio when local."""
        agent_memory_dir = Path(agent_memory_dir).resolve()
        agent_memory_dir.mkdir(parents=True, exist_ok=True)
        config_path = agent_memory_dir / self.CONFIG_FILENAME

        if mcp_sse_url:
            mcp_section = f'''
[mcp_servers.pokemon-emerald]
url = "{mcp_sse_url}"
'''
        elif server_url and project_root:
            exe = Path(sys.executable).as_posix()
            mcp_section = f'''
[mcp_servers.pokemon-emerald]
command = "{exe}"
args = ["-m", "server.cli.pokemon_mcp_server"]
env = {{ "POKEMON_SERVER_URL" = "{server_url}", "PYTHONPATH" = "{project_root}" }}
'''
        else:
            mcp_url = f"http://localhost:{mcp_sse_port}/mcp"
            mcp_section = f'''
[mcp_servers.pokemon-emerald]
url = "{mcp_url}"
'''

        openrouter_block = ""
        if getattr(self, "api_gateway", "login") == "openrouter" and os.environ.get("OPENROUTER_API_KEY"):
            openrouter_block = '''
# OpenRouter (uses OPENROUTER_API_KEY from environment)
model_provider = "openrouter"
model = "openai/gpt-5.3-codex"

[model_providers.openrouter]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
env_key = "OPENROUTER_API_KEY"
'''

        if config_path.exists():
            try:
                content = config_path.read_text()
                if "[mcp_servers.pokemon-emerald]" not in content and "pokemon-emerald" not in content:
                    content = content.rstrip() + "\n" + mcp_section
                    config_path.write_text(content)
                    logger.info("Appended MCP config to existing config.toml")
                elif mcp_section and "/sse" in content and "pokemon-emerald" in content:
                    # Migrate legacy /sse URL to Streamable HTTP /mcp (Codex expects /mcp)
                    content = content.replace("/sse\"", "/mcp\"")
                    config_path.write_text(content)
                    logger.info("Updated MCP URL from /sse to /mcp in config.toml")
                if openrouter_block and "model_provider" not in content and "[model_providers.openrouter]" not in content:
                    content = openrouter_block.strip() + "\n\n" + content
                    config_path.write_text(content)
                    logger.info("Prepended OpenRouter config to existing config.toml")
            except (OSError, Exception) as e:
                logger.warning("Could not merge config.toml: %s", e)
        else:
            header = openrouter_block.strip() if openrouter_block else "# Set model_provider and model, or use codex login for ChatGPT auth."
            base = f'''# Codex config for pokeagent (auto-generated)
{header}

{mcp_section.strip()}
'''
            config_path.write_text(base)
            logger.info("Wrote config.toml: %s", config_path)

        return config_path

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

        bootstrap = self._build_bootstrap_content(directive_path, server_url)
        working_dir_abs = Path(working_dir).resolve()
        working_dir_abs.mkdir(parents=True, exist_ok=True)

        # Write directive to workspace for codex to read
        directive_file = working_dir_abs / self.DIRECTIVE_FILENAME
        directive_file.write_text(bootstrap)

        mcp_port = mcp_sse_port or 8002
        # Codex expects Streamable HTTP at /mcp (not legacy SSE at /sse).
        # See https://developers.openai.com/codex/mcp/
        mcp_url = f"http://host.docker.internal:{mcp_port}/mcp"

        if containerized:
            if not run_id or not agent_memory_dir:
                raise ValueError("containerized mode requires run_id and agent_memory_dir")
            if not mcp_sse_port:
                raise ValueError("containerized mode requires mcp_sse_port")

            agent_memory_path = Path(agent_memory_dir).resolve()
            self._ensure_codex_config(agent_memory_path, mcp_sse_url=mcp_url, mcp_sse_port=mcp_port)

            game_port = server_url.rstrip("/").split(":")[-1]

            docker_cmd = [
                "docker", "run", "--rm",
                "--name", f"codex-agent-{run_id}",
                "--cap-add=NET_ADMIN",
                "--security-opt=seccomp=unconfined",
                "--network=bridge",
                "--add-host=host.docker.internal:host-gateway",
                "-v", f"{agent_memory_path}:{self.AGENT_MEMORY_PATH}",
                "-v", f"{working_dir_abs}:{self.WORKSPACE_PATH}",
                "-w", self.WORKSPACE_PATH,
                "-e", f"MCP_PORT={mcp_sse_port}",
                "-e", f"GAME_SERVER_PORT={game_port}",
                "-e", f"RUN_DATA_ID={run_id}",
                "-e", f"CODEX_HOME={self.AGENT_MEMORY_PATH}",
            ]

            use_openrouter = getattr(self, "api_gateway", "login") == "openrouter"
            if use_openrouter and os.environ.get("OPENROUTER_API_KEY"):
                docker_cmd.extend(["-e", f"OPENROUTER_API_KEY={os.environ['OPENROUTER_API_KEY']}"])
            else:
                for env_var in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
                    if os.environ.get(env_var):
                        docker_cmd.extend(["-e", f"{env_var}={os.environ[env_var]}"])

            docker_cmd.append(self.container_image)

            # --skip-git-repo-check: required when workspace is not a git repo (e.g. agent_scratch_space)
            # See https://github.com/openai/codex/issues/7522
            skip_git = "--skip-git-repo-check"
            cfg = f" -c model_reasoning_effort={thinking_effort}" if thinking_effort in ("low", "medium", "high") else ""
            if resume_session_id:
                if resume_session_id == "--last":
                    inner = f"codex exec resume --last --json {skip_git}{cfg}"
                else:
                    inner = f"codex exec resume {shlex.quote(resume_session_id)} --json {skip_git}{cfg}"
            else:
                inner = f"cat {self.WORKSPACE_PATH}/{self.DIRECTIVE_FILENAME} | codex exec --json -C {self.WORKSPACE_PATH} --dangerously-bypass-approvals-and-sandbox {skip_git}{cfg} -"

            shell_cmd = "'" + inner.replace("'", "'\"'\"'") + "'"
            docker_cmd.extend(["sh", "-c", shell_cmd])

            return docker_cmd, env, bootstrap, None

        else:
            # Local: use agent_memory_dir for config if provided; use stdio MCP (no SSE server)
            if agent_memory_dir:
                agent_memory_path = Path(agent_memory_dir).resolve()
                agent_memory_path.mkdir(parents=True, exist_ok=True)
                pythonpath = project_root or os.getcwd()
                self._ensure_codex_config(
                    agent_memory_path,
                    server_url=server_url,
                    project_root=pythonpath,
                )
                env["CODEX_HOME"] = str(agent_memory_path)

            codex_cfg = ["-c", f"model_reasoning_effort={thinking_effort}"] if thinking_effort in ("low", "medium", "high") else []
            cfg_str = f" {' '.join(codex_cfg)}" if codex_cfg else ""
            if resume_session_id:
                if resume_session_id == "--last":
                    return (["codex", "exec", "resume", "--last", "--json", "--skip-git-repo-check"] + codex_cfg, env, bootstrap, None)
                return (["codex", "exec", "resume", resume_session_id, "--json", "--skip-git-repo-check"] + codex_cfg, env, bootstrap, None)

            cat_cmd = f"cat {shlex.quote(str(directive_file))} | codex exec --json -C {shlex.quote(working_dir)} --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check{cfg_str} -"
            return (["sh", "-c", cat_cmd], env, bootstrap, None)

    def _handle_thread_started(
        self,
        event: dict,
        now: float,
        metrics: CliSessionMetrics | None,
    ) -> None:
        """Handle thread.started: set session_id from thread_id."""
        self._last_event_time = now
        thread_id = event.get("thread_id", "")
        if metrics and thread_id:
            metrics.session_id = thread_id
        logger.info("[codex] thread.started thread_id=%s", thread_id)

    def _handle_turn_completed(
        self,
        event: dict,
        metrics: CliSessionMetrics | None,
    ) -> None:
        """Handle turn.completed: update metrics from usage."""
        if metrics:
            usage = event.get("usage") or {}
            metrics.input_tokens = int(usage.get("input_tokens", 0) or 0)
            metrics.output_tokens = int(usage.get("output_tokens", 0) or 0)
        logger.info(
            "[codex] turn.completed input=%d output=%d",
            (event.get("usage") or {}).get("input_tokens", 0),
            (event.get("usage") or {}).get("output_tokens", 0),
        )

    def _handle_turn_failed(self, event: dict, metrics: CliSessionMetrics | None) -> None:
        """Handle turn.failed: set error state."""
        if metrics:
            metrics.is_error = True
            metrics.error = (event.get("error") or {}).get("message", "turn failed")
        logger.warning("[codex] turn.failed: %s", (event.get("error") or {}).get("message", ""))

    def _handle_item_mcp_tool_call(
        self,
        event: dict,
        now: float,
        metrics: CliSessionMetrics | None,
        server_url: str | None,
    ) -> None:
        """Handle item.completed with item.type mcp_tool_call: tool_use count and _post_thinking.

        Formats as [tool] {reasoning} to match Claude/Gemini, using reasoning from args or
        buffered reasoning block. Single _post_thinking per tool call (no separate reasoning line).
        """
        duration_sec = (now - self._last_event_time) if self._last_event_time is not None else 0.0
        self._last_event_time = now
        if metrics:
            metrics.tool_use_count += 1
        item = event.get("item") or {}
        tool_name = item.get("tool", "?")
        short_name = tool_name.split("__")[-1] if "__" in tool_name else tool_name
        args = item.get("arguments") or {}
        args_preview = json.dumps(args)
        if len(args_preview) > 120:
            args_preview = args_preview[:120] + "..."
        logger.info("[codex:mcp_tool_call] %s %s", tool_name, args_preview)
        reasoning = (
            args.get("reasoning")
            or args.get("reason")
            or self._pending_reasoning
            or ""
        )
        self._pending_reasoning = ""  # Consume buffer
        thinking_text = f"[{short_name}] {reasoning}".strip() or f"[{short_name}]"
        if server_url:
            self._post_thinking(server_url, thinking_text, duration_sec, interaction_type=self.name)

    def _handle_item_reasoning(
        self,
        event: dict,
        now: float,
        server_url: str | None,
    ) -> None:
        """Handle item.completed with item.type reasoning: buffer for next tool call.

        Reasoning is merged into the next mcp_tool_call as [tool] {reasoning} (no separate post).
        """
        self._last_event_time = now
        item = event.get("item") or {}
        text = (item.get("text") or "").strip()
        if text:
            self._pending_reasoning = (text[:500] + "...") if len(text) > 500 else text

    def _handle_error(self, event: dict) -> None:
        """Handle error event (may be transient reconnect notice)."""
        msg = event.get("message", "")
        if "Reconnecting" in msg:
            logger.debug("[codex] %s", msg)
        else:
            logger.warning("[codex] error: %s", msg)

    def handle_stream_event(
        self,
        event: dict,
        metrics: CliSessionMetrics | None,
        server_url: str | None = None,
    ) -> None:
        """Dispatch Codex exec --json events to typed handlers."""
        etype = event.get("type", "")
        now = time.time()

        if etype == "thread.started":
            self._handle_thread_started(event, now, metrics)
        elif etype == "turn.completed":
            self._handle_turn_completed(event, metrics)
        elif etype == "turn.failed":
            self._handle_turn_failed(event, metrics)
        elif etype == "error":
            self._handle_error(event)
        elif etype in ("item.started", "item.updated", "item.completed"):
            item = event.get("item") or {}
            item_type = item.get("type", "")
            if etype == "item.completed" and item_type == "mcp_tool_call":
                self._handle_item_mcp_tool_call(event, now, metrics, server_url)
            elif etype == "item.completed" and item_type == "reasoning":
                self._handle_item_reasoning(event, now, server_url)

    def log_cli_interaction(
        self,
        agent_memory_dir: Path,
        processed_hashes: set,
        last_cli_step: int,
        server_url: str | None = None,
    ) -> tuple[set, int]:
        """Poll Codex session JSONL files and append new entries as steps."""
        from utils.metric_tracking.codex_session_reader import load_new_usage_entries
        from utils.data_persistence.llm_logger import get_llm_logger

        search_path = Path(agent_memory_dir).resolve()
        if not search_path.is_dir():
            return processed_hashes, last_cli_step

        try:
            new_entries, updated_hashes = load_new_usage_entries(search_path, processed_hashes)
        except Exception as exc:
            logger.warning("Codex session poll failed: %s", exc)
            return processed_hashes, last_cli_step

        if not new_entries:
            return updated_hashes, last_cli_step

        logger.info("Codex session: appending %d step(s)", len(new_entries))
        new_entries.sort(key=lambda e: (e.get("_parsed_timestamp") or datetime.min.replace(tzinfo=timezone.utc)))

        llm_logger = get_llm_logger()
        # Use last step's timestamp so first entry of this poll gets correct duration (not 0.0)
        steps = llm_logger.cumulative_metrics.get("steps", [])
        prev_ts: float | None = steps[-1]["timestamp"] if steps else None
        for entry in new_entries:
            tokens = entry.get("_tokens", {})
            tool_calls = entry.get("_tool_calls", [])
            parsed_ts = entry.get("_parsed_timestamp")
            ts_float = parsed_ts.timestamp() if parsed_ts is not None else time.time()
            duration = max(0.0, ts_float - prev_ts) if prev_ts is not None else 0.0
            prev_ts = ts_float

            model_name = entry.get("_model", "gpt-5-codex")
            model_info = {"model": model_name}

            last_cli_step += 1
            try:
                llm_logger.append_cli_step(
                    step_number=last_cli_step,
                    token_usage=tokens,
                    duration=duration,
                    timestamp=ts_float,
                    model_info=model_info,
                    tool_calls=tool_calls if tool_calls else None,
                )
            except Exception as exc:
                logger.warning("Failed to append Codex CLI step %d: %s", last_cli_step, exc)
                last_cli_step -= 1

        if server_url and new_entries:
            self._sync_metrics_to_server(server_url)

        return updated_hashes, last_cli_step

    def get_resume_session_id(self, agent_memory_dir: Path) -> str | None:
        """Find the most recent Codex session ID from sessions dir.
        Codex stores sessions in nested dirs like sessions/2026/03/10/rollout-*.jsonl."""
        sessions_dir = Path(agent_memory_dir) / self.SESSIONS_SUBDIR
        if not sessions_dir.is_dir():
            return None
        files = list(sessions_dir.rglob("*.jsonl")) + list(sessions_dir.rglob("*.json"))
        if not files:
            return None
        most_recent = max(files, key=lambda p: p.stat().st_mtime)
        return most_recent.stem

    def run_login(self) -> bool:
        """Run codex login if needed; no-op when using OpenRouter API key."""
        if getattr(self, "api_gateway", "login") == "openrouter":
            print("\nℹ️  Using --api-gateway openrouter. No interactive login required.")
            return True
        print("\n🔐 Running 'codex login' (interactive)...")
        try:
            result = subprocess.run(["codex", "login"], check=False)
            return result.returncode == 0
        except FileNotFoundError:
            print("❌ Codex CLI not found. Install: npm install -g @openai/codex")
            return False


class HermesCliBackend(CliAgentBackend):
    """Backend for the Nous Hermes agent, bridged through a local JSONL wrapper."""

    WORKSPACE_PATH = "/workspace"
    PROJECT_ROOT_PATH = "/opt/pokeagent-src"
    AGENT_MEMORY_PATH = "/home/hermes-agent/.hermes"
    DIRECTIVE_FILENAME = ".agent_directive.txt"
    CONFIG_FILENAME = "config.yaml"
    CONFIG_MARKER_BEGIN = "# BEGIN POKEAGENT HERMES MCP"
    CONFIG_MARKER_END = "# END POKEAGENT HERMES MCP"

    @property
    def name(self) -> str:
        return "HermesCLI"

    @property
    def agent_memory_subdir(self) -> str:
        return "hermes_memory"

    @property
    def container_image(self) -> str:
        return "hermes-agent-devcontainer"

    @property
    def devcontainer_build_context(self) -> str:
        return ".devcontainer/hermes-agent"

    def __init__(self) -> None:
        super().__init__()
        self._last_event_time: float | None = None
        self._last_seen_session_totals: dict[str, dict] = {}

    def is_auth_fatal_error(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "invalid api key" in lowered
            or "authentication" in lowered
            or "not configured" in lowered
            or "missing api key" in lowered
        )

    @staticmethod
    def _normalize_tool_name(name: str) -> str:
        for prefix in ("mcp_pokemon_emerald_", "mcp__pokemon-emerald__"):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name.split("__")[-1] if "__" in name else name

    def seed_agent_auth(self, agent_memory_dir: Path) -> None:
        """Seed run-local Hermes home from the user's ~/.hermes directory when present."""
        import shutil

        host_hermes_dir = Path.home() / ".hermes"
        if not host_hermes_dir.exists():
            print("ℹ️  Host ~/.hermes not found; Hermes will rely on environment variables/config generated for this run.")
            return

        skipped_names = {"audio_cache", "hermes-agent", "image_cache", "logs", "sessions", "state.db"}
        for child in host_hermes_dir.iterdir():
            if child.name in skipped_names:
                continue
            target = agent_memory_dir / child.name
            try:
                if child.is_dir():
                    shutil.copytree(child, target, dirs_exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, target)
            except OSError as exc:
                logger.warning("Could not seed Hermes path %s: %s", child, exc)

    def _build_hermes_mcp_block(
        self,
        *,
        mcp_url: str | None = None,
        server_url: str | None = None,
        pythonpath: str | None = None,
    ) -> str:
        if mcp_url:
            body = (
                "mcp_servers:\n"
                "  pokemon-emerald:\n"
                f'    url: "{mcp_url}"\n'
                "    tools:\n"
                "      prompts: false\n"
                "      resources: false\n"
            )
        else:
            executable = Path(sys.executable).resolve().as_posix()
            body = (
                "mcp_servers:\n"
                "  pokemon-emerald:\n"
                f'    command: "{executable}"\n'
                '    args: ["-m", "server.cli.pokemon_mcp_server"]\n'
                "    env:\n"
                f'      POKEMON_SERVER_URL: "{server_url}"\n'
                f'      PYTHONPATH: "{pythonpath}"\n'
                "    tools:\n"
                "      prompts: false\n"
                "      resources: false\n"
            )
        return f"{self.CONFIG_MARKER_BEGIN}\n{body}{self.CONFIG_MARKER_END}\n"

    def _ensure_hermes_config(
        self,
        agent_memory_dir: Path,
        *,
        mcp_url: str | None = None,
        server_url: str | None = None,
        pythonpath: str | None = None,
    ) -> Path:
        agent_memory_dir.mkdir(parents=True, exist_ok=True)
        config_path = agent_memory_dir / self.CONFIG_FILENAME
        existing = ""
        if config_path.exists():
            try:
                existing = config_path.read_text(encoding="utf-8")
            except OSError:
                existing = ""

        if self.CONFIG_MARKER_BEGIN in existing and self.CONFIG_MARKER_END in existing:
            start = existing.index(self.CONFIG_MARKER_BEGIN)
            end = existing.index(self.CONFIG_MARKER_END) + len(self.CONFIG_MARKER_END)
            existing = (existing[:start] + existing[end:]).strip()

        injected = self._build_hermes_mcp_block(
            mcp_url=mcp_url,
            server_url=server_url,
            pythonpath=pythonpath,
        )
        new_content = (existing.rstrip() + "\n\n" + injected) if existing.strip() else injected
        config_path.write_text(new_content, encoding="utf-8")
        return config_path

    def _build_wrapper_cmd(
        self,
        directive_path: Path,
        working_dir: Path,
        server_url: str,
        hermes_home: Path,
        *,
        python_bin: str = "python3",
        resume_session_id: str | None = None,
        project_root: str | Path | None = None,
    ) -> list[str]:
        # Run wrapper as script (not -m) so Hermes's utils.py is found when PYTHONPATH has hermes-agent first
        root = Path(project_root) if project_root else Path(self.PROJECT_ROOT_PATH)
        wrapper_script = str(root / "utils" / "agent_infrastructure" / "hermes_wrapper.py")
        cmd = [
            python_bin,
            wrapper_script,
            "--directive-path",
            str(directive_path),
            "--working-dir",
            str(working_dir),
            "--server-url",
            server_url,
            "--hermes-home",
            str(hermes_home),
        ]
        if resume_session_id:
            cmd.extend(["--resume-session-id", resume_session_id])

        if getattr(self, "api_gateway", "login") == "openrouter" and os.environ.get("OPENROUTER_API_KEY"):
            cmd.extend(
                [
                    "--provider",
                    "openrouter",
                    "--base-url",
                    "https://openrouter.ai/api/v1",
                    "--api-key-env",
                    "OPENROUTER_API_KEY",
                    "--model",
                    os.environ.get("HERMES_MODEL", "google/gemini-3-flash-preview"),
                ]
            )
        else:
            model = os.environ.get("HERMES_MODEL")
            provider = os.environ.get("HERMES_PROVIDER")
            base_url = os.environ.get("HERMES_BASE_URL")
            api_key_env = os.environ.get("HERMES_API_KEY_ENV")
            if model:
                cmd.extend(["--model", model])
            if provider:
                cmd.extend(["--provider", provider])
            if base_url:
                cmd.extend(["--base-url", base_url])
            if api_key_env:
                cmd.extend(["--api-key-env", api_key_env])
        return cmd

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
        del dangerously_skip_permissions, session_number, thinking_effort
        env = os.environ.copy()
        env["POKEMON_MCP_SERVER_URL"] = server_url
        env["POKEMON_SERVER_URL"] = server_url

        working_dir_abs = Path(working_dir).resolve()
        working_dir_abs.mkdir(parents=True, exist_ok=True)
        bootstrap = self._build_bootstrap_content(directive_path, server_url)
        directive_file = working_dir_abs / self.DIRECTIVE_FILENAME
        directive_file.write_text(bootstrap, encoding="utf-8")

        if containerized:
            if not run_id or not agent_memory_dir:
                raise ValueError("containerized mode requires run_id and agent_memory_dir")
            if not mcp_sse_port:
                raise ValueError("containerized mode requires mcp_sse_port")
            if not project_root:
                raise ValueError("containerized mode requires project_root")

            agent_memory_path = Path(agent_memory_dir).resolve()
            project_root_path = Path(project_root).resolve()
            debug_dir_path = project_root_path / ".cursor"
            # Use Streamable HTTP (/mcp) - Hermes MCP client sends POST; /sse expects GET and returns 405
            mcp_url = f"http://host.docker.internal:{mcp_sse_port}/mcp"
            self._ensure_hermes_config(agent_memory_path, mcp_url=mcp_url)

            # Use container paths: working_dir and directive are mounted at WORKSPACE_PATH
            container_directive = f"{self.WORKSPACE_PATH}/{self.DIRECTIVE_FILENAME}"
            container_working_dir = self.WORKSPACE_PATH
            wrapper_cmd = self._build_wrapper_cmd(
                Path(container_directive),
                Path(container_working_dir),
                server_url,
                Path(self.AGENT_MEMORY_PATH),
                python_bin="python3",
                resume_session_id=resume_session_id,
                project_root=self.PROJECT_ROOT_PATH,
            )
            game_port = server_url.rstrip("/").split(":")[-1]

            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                f"hermes-agent-{run_id}",
                "--cap-add=NET_ADMIN",
                "--security-opt=seccomp=unconfined",
                "--network=bridge",
                "--add-host=host.docker.internal:host-gateway",
                "-v",
                f"{agent_memory_path}:{self.AGENT_MEMORY_PATH}",
                "-v",
                f"{working_dir_abs}:{self.WORKSPACE_PATH}",
                "-v",
                f"{project_root_path}:{self.PROJECT_ROOT_PATH}:ro",
                "-v",
                f"{debug_dir_path}:{debug_dir_path}",
                "-w",
                self.WORKSPACE_PATH,
                "-e",
                f"MCP_PORT={mcp_sse_port}",
                "-e",
                f"GAME_SERVER_PORT={game_port}",
                "-e",
                f"RUN_DATA_ID={run_id}",
                "-e",
                f"HERMES_HOME={self.AGENT_MEMORY_PATH}",
                "-e",
                f"PYTHONPATH={self.PROJECT_ROOT_PATH}",
            ]

            passthrough_envs = [
                "OPENROUTER_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "HERMES_MODEL",
                "HERMES_PROVIDER",
                "HERMES_BASE_URL",
                "HERMES_API_KEY_ENV",
                "HERMES_DISABLE_MULTIMODAL",
                "HERMES_API_TIMEOUT",
                "HERMES_VISION_TIMEOUT",
            ]
            for env_var in passthrough_envs:
                if os.environ.get(env_var):
                    docker_cmd.extend(["-e", f"{env_var}={os.environ[env_var]}"])

            docker_cmd.append(self.container_image)
            docker_cmd.extend(wrapper_cmd)
            return docker_cmd, env, bootstrap, None

        agent_memory_path = Path(agent_memory_dir).resolve() if agent_memory_dir else (working_dir_abs / ".hermes")
        pythonpath = project_root or os.getcwd()
        self._ensure_hermes_config(
            agent_memory_path,
            server_url=server_url,
            pythonpath=pythonpath,
        )
        env["HERMES_HOME"] = str(agent_memory_path)
        env["PYTHONPATH"] = (
            f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else pythonpath
        )
        wrapper_cmd = self._build_wrapper_cmd(
            directive_file,
            working_dir_abs,
            server_url,
            agent_memory_path,
            python_bin=sys.executable,
            resume_session_id=resume_session_id,
            project_root=pythonpath,
        )
        return wrapper_cmd, env, bootstrap, None

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
        elif etype == "thinking":
            content = (event.get("content") or "").strip()
            if content and server_url:
                duration = (now - self._last_event_time) if self._last_event_time is not None else 0.0
                self._post_thinking(server_url, content, duration, interaction_type=self.name)
            self._last_event_time = now
        elif etype == "tool_use":
            if metrics:
                metrics.tool_use_count += 1
            args = (
                event.get("arguments")
                or event.get("parameters")
                or event.get("input")
                or {}
            )
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            reasoning = args.get("reasoning") or args.get("reason") or ""
            tool_name = event.get("tool_name") or event.get("name") or "tool"
            short_name = self._normalize_tool_name(tool_name)
            if server_url:
                duration = (now - self._last_event_time) if self._last_event_time is not None else 0.0
                thinking_text = f"[{short_name}] {reasoning}".strip() if reasoning else f"[{short_name}]"
                self._post_thinking(
                    server_url,
                    thinking_text,
                    duration,
                    interaction_type=self.name,
                )
            self._last_event_time = now
        elif etype == "result":
            if metrics:
                metrics.session_id = event.get("session_id", metrics.session_id)
                metrics.model = event.get("model", metrics.model)
                metrics.total_cost_usd = float(event.get("total_cost_usd", 0.0) or 0.0)
                metrics.num_turns = int(event.get("num_turns", 0) or 0)
                metrics.duration_ms = int(event.get("duration_ms", 0) or 0)
                metrics.duration_api_ms = int(event.get("duration_api_ms", 0) or 0)
                metrics.is_error = bool(event.get("is_error", False))
                metrics.error = event.get("error", "") or ""
                usage = event.get("usage") or {}
                metrics.input_tokens = int(usage.get("input_tokens", 0) or 0)
                metrics.output_tokens = int(usage.get("output_tokens", 0) or 0)
        elif etype == "error":
            message = event.get("message", "") or event.get("error", "")
            if metrics and self.is_auth_fatal_error(message):
                metrics.auth_fatal_error = True
                metrics.error = message
                metrics.is_error = True

    def log_cli_interaction(
        self,
        agent_memory_dir: Path,
        processed_hashes: set,
        last_cli_step: int,
        server_url: str | None = None,
    ) -> tuple[set, int]:
        from utils.metric_tracking.hermes_session_reader import load_new_usage_entries
        from utils.data_persistence.llm_logger import get_llm_logger

        search_path = Path(agent_memory_dir).resolve()
        if not search_path.is_dir():
            return processed_hashes, last_cli_step

        try:
            new_entries, updated_hashes, self._last_seen_session_totals = load_new_usage_entries(
                search_path,
                processed_hashes,
                self._last_seen_session_totals,
            )
        except Exception as exc:
            logger.warning("Hermes session poll failed: %s", exc)
            return processed_hashes, last_cli_step

        if not new_entries:
            return updated_hashes, last_cli_step

        new_entries.sort(key=lambda e: (e.get("_parsed_timestamp") or datetime.min.replace(tzinfo=timezone.utc)))
        llm_logger = get_llm_logger()
        steps = llm_logger.cumulative_metrics.get("steps", [])
        prev_ts: float | None = steps[-1]["timestamp"] if steps else None

        for entry in new_entries:
            tokens = entry.get("_tokens", {})
            tool_calls = entry.get("_tool_calls", [])
            parsed_ts = entry.get("_parsed_timestamp")
            ts_float = parsed_ts.timestamp() if parsed_ts is not None else time.time()
            duration = max(0.0, ts_float - prev_ts) if prev_ts is not None else 0.0
            prev_ts = ts_float
            model_info = {"model": entry.get("_model", "hermes-agent")}
            last_cli_step += 1
            try:
                llm_logger.append_cli_step(
                    step_number=last_cli_step,
                    token_usage=tokens,
                    duration=duration,
                    timestamp=ts_float,
                    model_info=model_info,
                    tool_calls=tool_calls if tool_calls else None,
                )
            except Exception as exc:
                logger.warning("Failed to append Hermes CLI step %d: %s", last_cli_step, exc)
                last_cli_step -= 1

        if server_url and new_entries:
            self._sync_metrics_to_server(server_url)
        return updated_hashes, last_cli_step

    def get_resume_session_id(self, agent_memory_dir: Path) -> str | None:
        from utils.metric_tracking.hermes_session_reader import get_latest_session_id

        return get_latest_session_id(agent_memory_dir)

    def run_login(self) -> bool:
        if getattr(self, "api_gateway", "login") == "openrouter":
            print("\nℹ️  Using --api-gateway openrouter. No interactive Hermes login required.")
            return True
        print("\n🔐 Running 'hermes model' (interactive)...")
        try:
            result = subprocess.run(["hermes", "model"], check=False)
            return result.returncode == 0
        except FileNotFoundError:
            print("❌ Hermes CLI not found. Install Hermes or run the backend in containerized mode.")
            return False


def get_backend(cli_type: str) -> CliAgentBackend:
    """Return the backend for the given CLI type."""
    if cli_type == "claude":
        return ClaudeCodeBackend()
    if cli_type == "gemini":
        return GeminiCliBackend()
    if cli_type == "codex":
        return CodexCliBackend()
    if cli_type == "hermes":
        return HermesCliBackend()
    raise ValueError(f"Unknown CLI type: {cli_type}. Supported: claude, gemini, codex, hermes")

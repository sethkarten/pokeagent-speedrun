#!/usr/bin/env python3
"""
Entry point for external CLI agent experiments (Claude Code, GPT Codex, etc.).

This script:
1. Spawns the Pokemon Emerald server with MCP endpoints
2. Launches an external CLI agent (e.g., claude, codex) as a subprocess
3. Monitors termination conditions (e.g., gym badge count) and terminates when met

The CLI agent interacts with the game via MCP tools exposed by the server.
"""

import os
import sys
import time
import argparse
import socket
import subprocess
import signal
import threading
import logging
import shutil
import json
import secrets
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cli_agent_backends import (
    CliSession,
    CliSessionMetrics,
    get_backend,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _terminate_process(
    process: subprocess.Popen,
    graceful_timeout: int,
    label: str,
    use_process_group: bool = False,
) -> None:
    """Terminate process gracefully, then force kill if needed."""
    if process is None or process.poll() is not None:
        return

    print(f"\n{label}...")
    try:
        if use_process_group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=graceful_timeout)
    except subprocess.TimeoutExpired:
        print(f"   Force killing after {graceful_timeout}s...")
        try:
            if use_process_group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return


def _cleanup_cli_session(session: CliSession | None, log_file=None):
    """Clean up resources from a single CLI agent session."""
    if session:
        if session.stop_event:
            session.stop_event.set()
        if session.temp_mcp_config_path:
            try:
                os.remove(session.temp_mcp_config_path)
            except OSError:
                pass
    if log_file and not log_file.closed:
        try:
            log_file.close()
        except OSError:
            pass


def preflight_cli(args) -> bool:
    """Validate CLI availability and show actionable setup errors."""
    if args.cli_type != "claude":
        return True

    claude_path = shutil.which("claude")
    if not claude_path:
        print("❌ Claude Code CLI not found on PATH.")
        print("   Install: curl -fsSL https://claude.ai/install.sh | bash")
        print("   Then verify with: claude --version")
        return False

    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            print("❌ Claude Code CLI is installed but --version failed.")
            print(f"   stdout: {result.stdout.strip()}")
            print(f"   stderr: {result.stderr.strip()}")
            print("   Try: claude (then login) and retry this run.")
            return False
    except Exception as e:
        print(f"❌ Failed to run 'claude --version': {e}")
        print("   Ensure Claude Code CLI is installed and authenticated.")
        return False

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("ℹ️  ANTHROPIC_API_KEY is set. Claude may use API-key auth instead of CLI login.")

    return True


def _run_claude_login() -> bool:
    """Run 'claude auth login' interactively. Returns True on success."""
    print("\n🔐 Running 'claude auth login' (interactive)...")
    try:
        result = subprocess.run(
            ["claude", "auth", "login"],
            check=False,
        )
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


def _build_container_image(backend) -> bool:
    """Build the container image for the given backend. Returns True on success."""
    project_root = Path(__file__).resolve().parent
    ctx = backend.devcontainer_build_context
    ctx_path = project_root / ctx
    dockerfile = ctx_path / "Dockerfile"
    if not ctx_path.is_dir() or not dockerfile.exists():
        print(f"❌ Devcontainer build context not found: {ctx_path}")
        return False
    print(f"🐳 Building container image: {backend.container_image}")
    print(f"   Context: {ctx_path}")
    try:
        # Match container user UID/GID to host so bind-mounted files are readable both ways
        uid, gid = os.getuid(), os.getgid()
        result = subprocess.run(
            [
                "docker", "build",
                "-t", backend.container_image,
                "--build-arg", f"USER_UID={uid}",
                "--build-arg", f"USER_GID={gid}",
                "-f", str(dockerfile),
                str(ctx_path),
            ],
            check=False,
        )
        if result.returncode != 0:
            print(f"❌ Docker build failed (exit code {result.returncode})")
            return False
        print(f"✅ Container image built: {backend.container_image}")
        return True
    except FileNotFoundError:
        print("❌ Docker not found. Install Docker and retry.")
        return False
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False


def start_server(args, run_id=None):
    """Start the server process with appropriate arguments.
    
    Reuses the server spawning logic from run.py.
    
    Args:
        args: Command line arguments
        run_id: Optional run_id to pass to server via environment variable
        
    Returns:
        subprocess.Popen: Server process, or None if failed
    """
    python_exe = sys.executable
    server_cmd = [python_exe, "-m", "server.app", "--port", str(args.port)]
    
    # Set up environment for server
    server_env = os.environ.copy()
    if run_id:
        server_env["RUN_DATA_ID"] = run_id
    server_env.pop("POKEAGENT_CLI_MODE", None)  # server owns total_actions; merge preserves steps from run_cli

    # Pass LLM session_id if available
    llm_session_id = os.environ.get("LLM_SESSION_ID")
    if llm_session_id:
        server_env["LLM_SESSION_ID"] = llm_session_id

    # Single-writer metrics: server is the only writer
    server_env["LLM_METRICS_WRITE_ENABLED"] = "true"

    # Protect state endpoints so the CLI agent cannot load/save state via Bash curl
    server_env["POKEMON_STATE_API_KEY"] = secrets.token_urlsafe(16)
    
    # Pass through server-relevant arguments
    if hasattr(args, 'record') and args.record:
        server_cmd.append("--record")
    
    if hasattr(args, 'load_checkpoint') and args.load_checkpoint:
        from utils.run_data_manager import get_cache_path
        checkpoint_state = get_cache_path("checkpoint.state")
        if checkpoint_state.exists():
            server_cmd.extend(["--load-state", str(checkpoint_state)])
            server_env["LOAD_CHECKPOINT_MODE"] = "true"
            print(f"🔄 Server will load checkpoint: {checkpoint_state}")
        else:
            print(f"⚠️ Checkpoint file not found: {checkpoint_state}")
    elif hasattr(args, 'load_state') and args.load_state:
        server_cmd.extend(["--load-state", args.load_state])
    
    if hasattr(args, 'no_ocr') and args.no_ocr:
        server_cmd.append("--no-ocr")
    
    if hasattr(args, 'direct_objectives') and args.direct_objectives:
        server_cmd.extend(["--direct-objectives", args.direct_objectives])
        if hasattr(args, 'direct_objectives_start') and args.direct_objectives_start > 0:
            server_cmd.extend(["--direct-objectives-start", str(args.direct_objectives_start)])
    
    # Start server as subprocess
    try:
        print(f"📋 Server command: {' '.join(server_cmd)}")
        server_process = subprocess.Popen(
            server_cmd,
            env=server_env,
            universal_newlines=True,
            bufsize=1
        )
        print(f"✅ Server started with PID {server_process.pid}")
        print("⏳ Waiting 5 seconds for server to initialize...")
        time.sleep(5)
        
        return server_process
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None


def start_frame_server(port):
    """Start the lightweight frame server for stream.html visualization."""
    try:
        frame_cmd = ["python", "-m", "server.frame_server", "--port", str(port + 1)]
        frame_process = subprocess.Popen(
            frame_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"🖼️  Frame server started with PID {frame_process.pid}")
        return frame_process
    except Exception as e:
        print(f"⚠️ Could not start frame server: {e}")
        return None


def start_mcp_sse_server(
    server_url: str,
    mcp_port: int,
    project_root: str | None = None,
    log_path: Path | None = None,
) -> "subprocess.Popen | None":
    """Start the MCP SSE server used by containerized agents.

    Args:
        server_url:   Game server URL the MCP proxy forwards to.
        mcp_port:     Port to bind the SSE server on.
        project_root: Added to PYTHONPATH so server modules resolve.
        log_path:     File to capture server stdout/stderr (avoids pipe-buffer deadlock).

    Returns:
        Running Popen handle, or None on failure.
    """
    try:
        mcp_env = os.environ.copy()
        mcp_env["MCP_TRANSPORT"] = "sse"
        mcp_env["MCP_PORT"] = str(mcp_port)
        mcp_env["POKEMON_SERVER_URL"] = server_url
        if project_root:
            mcp_env["PYTHONPATH"] = project_root

        # Route output to a log file (avoids pipe-buffer deadlock when uvicorn is verbose)
        log_fh = open(log_path, "w") if log_path else subprocess.DEVNULL
        mcp_process = subprocess.Popen(
            [sys.executable, "-m", "server.cli.pokemon_mcp_server"],
            env=mcp_env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        print(f"🌐 MCP SSE server started (PID {mcp_process.pid}, port {mcp_port})")

        # Poll until the port is accepting connections (up to 5 s)
        deadline = time.monotonic() + 5.0
        port_up = False
        while time.monotonic() < deadline:
            if mcp_process.poll() is not None:
                break
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    if s.connect_ex(("127.0.0.1", mcp_port)) == 0:
                        port_up = True
                        break
            except OSError:
                pass
            time.sleep(0.2)

        if not port_up:
            hint = f" (see {log_path})" if log_path else ""
            if mcp_process.poll() is not None:
                print(f"   ❌ MCP server exited early (code {mcp_process.poll()}){hint}")
            else:
                print(f"   ⚠️  MCP server port {mcp_port} not responding after 5 s{hint}")
            return None

        print(f"   ✓ MCP server listening on port {mcp_port}")
        return mcp_process
    except Exception as e:
        print(f"❌ Failed to start MCP SSE server: {e}")
        return None


def check_termination_condition(server_url: str, condition_type: str = "gym_badge_count", threshold: int = 1) -> dict:
    """Check if termination condition is met by polling the server.
    
    Args:
        server_url: Base URL of the server (e.g., http://localhost:8000)
        condition_type: Type of termination condition (e.g., "gym_badge_count")
        threshold: Threshold value for the condition (e.g., 1 for first badge)
        
    Returns:
        dict with condition status
    """
    try:
        response = requests.get(
            f"{server_url}/termination_condition",
            params={"condition_type": condition_type, "threshold": threshold},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to check termination condition: {e}")
        return {"condition_met": False, "error": str(e)}


def termination_monitor(
    server_url: str,
    termination_triggered: threading.Event,
    condition_type: str = "gym_badge_count",
    threshold: int = 1,
    poll_interval: int = 10,
):
    """Poll server for termination condition. Sets event when met."""
    logger.info(f"Termination monitor started: {condition_type} >= {threshold}, polling every {poll_interval}s")
    while not termination_triggered.is_set():
        result = check_termination_condition(server_url, condition_type, threshold)
        if result.get("condition_met"):
            termination_triggered.set()
            logger.info(f"Termination condition met: {condition_type}={result.get('current_value', '?')} >= {threshold}")
            if result.get("badge_names"):
                logger.info(f"   Badges: {result['badge_names']}")
            return
        time.sleep(poll_interval)


@dataclass
class Services:
    """Running services (server, frame server, MCP) and URLs."""
    server: subprocess.Popen | None
    frame_server: subprocess.Popen | None
    mcp_process: subprocess.Popen | None
    server_url: str


def _restore_from_backup(backup_path: str) -> bool:
    """Restore cache from backup zip. Returns True if successful."""
    from utils.backup_manager import restore_cache_from_backup
    from utils.run_data_manager import get_cache_directory

    print(f"\n📦 Restoring from backup: {backup_path}")
    success = restore_cache_from_backup(
        backup_file=backup_path,
        create_backup_of_current=False,
    )
    if success:
        print(f"✅ Backup restored to: {get_cache_directory()}")
    else:
        print("❌ Failed to restore backup, continuing with fresh state")
    return success


def _start_services(args, run_manager) -> Services | None:
    """Start server, frame server, and optionally MCP SSE server. Returns Services or None if server failed."""
    server_process = start_server(args, run_manager.run_id)
    if not server_process:
        return None

    frame_server_process = start_frame_server(args.port)
    server_url = f"http://localhost:{args.port}"

    mcp_process = None
    if args.containerized:
        if args.mcp_sse_port is None:
            args.mcp_sse_port = args.port + 2
        print(f"\n🐳 Containerized mode enabled (MCP SSE port {args.mcp_sse_port})")
        project_root_for_mcp = str(Path(__file__).resolve().parent)
        log_dir = Path(run_manager.get_run_directory()) / "agent_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        mcp_log = log_dir / "mcp_server.log"
        mcp_process = start_mcp_sse_server(
            server_url, args.mcp_sse_port, project_root_for_mcp, log_path=mcp_log
        )
        if not mcp_process:
            return None

    return Services(
        server=server_process,
        frame_server=frame_server_process,
        mcp_process=mcp_process,
        server_url=server_url,
    )


def _cleanup_services(
    services: Services | None,
    cli_session: CliSession | None,
    cli_log_file,
    args,
    graceful_timeout: int = 30,
) -> None:
    """Clean up server, frame server, MCP server, and CLI agent."""
    if cli_session is not None and cli_session.process.poll() is None:
        _terminate_process(
            process=cli_session.process,
            graceful_timeout=graceful_timeout,
            label="🤖 Stopping CLI agent",
            use_process_group=True,
        )
    if cli_log_file and not cli_log_file.closed:
        try:
            cli_log_file.close()
        except OSError:
            pass
    if services:
        if services.server:
            print("📡 Stopping server process...")
            services.server.terminate()
            try:
                services.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("   Force killing server...")
                services.server.kill()
        if services.frame_server:
            print("🖼️  Stopping frame server...")
            services.frame_server.terminate()
            try:
                services.frame_server.wait(timeout=2)
            except subprocess.TimeoutExpired:
                services.frame_server.kill()
        if services.mcp_process:
            print("🌐 Stopping MCP SSE server...")
            services.mcp_process.terminate()
            try:
                services.mcp_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                services.mcp_process.kill()
    print("👋 Goodbye!")


def launch_cli_agent(
    backend,
    server_url: str,
    directive_path: str,
    working_dir: str,
    *,
    project_root: str | None = None,
    dangerously_skip_permissions: bool = True,
    log_file=None,
    metrics: CliSessionMetrics | None = None,
    containerized: bool = False,
    session_number: int = 1,
    resume_session_id: str | None = None,
    thinking_effort: str | None = None,
    mcp_sse_port: int | None = None,
    run_id: str | None = None,
    agent_memory_dir: str | None = None,
) -> CliSession:
    """Launch an external CLI agent session as subprocess using the given backend."""
    cmd, env, bootstrap, temp_mcp_config_path = backend.build_launch_cmd(
        directive_path,
        server_url,
        working_dir,
        dangerously_skip_permissions=dangerously_skip_permissions,
        project_root=project_root,
        containerized=containerized,
        session_number=session_number,
        resume_session_id=resume_session_id,
        thinking_effort=thinking_effort,
        mcp_sse_port=mcp_sse_port,
        run_id=run_id,
        agent_memory_dir=agent_memory_dir,
    )
    if directive_path:
        print(f"📜 Loaded directive from: {directive_path}")
    print(f"🤖 Launching {backend.name} CLI agent...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Working directory: {working_dir}")
    print(f"   MCP Server: {server_url}")

    process = subprocess.Popen(
        cmd,
        cwd=working_dir,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        bufsize=0,
    )
    # Bootstrap is now passed as a command argument, not via stdin
    # (Claude Code --print mode doesn't read from stdin)
    process.stdin.close()

    stream_stop_event = threading.Event()
    stream_thread = threading.Thread(
        target=backend.run_stream_reader,
        args=(process.stdout, stream_stop_event, log_file, metrics, server_url),
        daemon=True,
    )
    stream_thread.start()
    print(f"✅ CLI agent started (PID {process.pid})")
    return CliSession(
        process=process,
        stop_event=stream_stop_event,
        stream_thread=stream_thread,
        temp_mcp_config_path=temp_mcp_config_path,
    )


def _sync_metrics_to_server(server_url: str) -> None:
    """Push run_cli's in-memory cumulative metrics to the server (single-writer pattern)."""
    try:
        from utils.llm_logger import get_llm_logger

        llm_logger = get_llm_logger()
        metrics = llm_logger.get_cumulative_metrics()
        step_count = len(metrics.get("steps", []))
        if step_count == 0:
            return  # nothing useful to sync
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


def _poll_claude_jsonl_and_append_steps(
    agent_memory_dir: Path,
    processed_hashes: set,
    last_cli_step: int,
    server_url: str | None = None,
) -> tuple[set, int]:
    """Poll Claude Code JSONL files and append new entries as steps to cumulative_metrics.

    Returns updated (processed_hashes, last_cli_step).  Never raises – all
    errors are logged as warnings so the agent loop is never interrupted.
    """
    from utils.claude_jsonl_reader import find_jsonl_files, load_new_usage_entries
    from utils.llm_logger import get_llm_logger

    # Use absolute path so polling works regardless of cwd
    search_path = Path(agent_memory_dir).resolve()
    if not search_path.is_dir():
        logger.warning("JSONL poll: agent_memory_dir not a directory: %s", search_path)
        return processed_hashes, last_cli_step

    try:
        new_entries, processed_hashes = load_new_usage_entries(search_path, processed_hashes)
    except Exception as exc:
        logger.warning("JSONL poll failed: %s", exc)
        return processed_hashes, last_cli_step

    jsonl_count = len(find_jsonl_files(search_path))
    if not new_entries:
        if jsonl_count > 0:
            logger.debug("JSONL poll: %d file(s) under %s, 0 new entries (already processed or no usage)", jsonl_count, search_path)
        return processed_hashes, last_cli_step

    logger.info("JSONL poll: appending %d step(s) from %s", len(new_entries), search_path)

    # Sort by timestamp so steps are appended in chronological order
    new_entries.sort(key=lambda e: (e["_parsed_timestamp"] or datetime.min.replace(tzinfo=None)))

    llm_logger = get_llm_logger()
    prev_ts: float | None = None
    for entry in new_entries:
        tokens = entry["_tokens"]
        tool_calls = entry["_tool_calls"]
        parsed_ts = entry.get("_parsed_timestamp")
        ts_float = parsed_ts.timestamp() if parsed_ts is not None else time.time()
        duration = max(0.0, ts_float - prev_ts) if prev_ts is not None else 0.0
        prev_ts = ts_float

        msg = entry.get("message")
        model_name = msg.get("model") if isinstance(msg, dict) else None
        model_info = {"model": model_name or "claude-code"}

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
        _sync_metrics_to_server(server_url)
    return processed_hashes, last_cli_step


def _run_agent_loop(
    services: Services,
    args,
    run_manager,
    agent_memory_dir: Path,
) -> tuple[str | None, CliSession | None, object]:
    """Run the agent loop until termination. Returns (termination_reason, last_cli_session, cli_log_file)."""
    from utils.run_data_manager import get_cache_path

    run_id = run_manager.run_id
    server_url = services.server_url
    termination_triggered = threading.Event()
    termination_reason: str | None = None

    print(f"\n🎥 Stream View: http://127.0.0.1:{args.port}/stream")
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"⚠️ Server health check failed: {health_response.status_code}")
    except Exception as e:
        print(f"⚠️ Could not verify server health: {e}")

    print(f"\n Starting termination monitor...")
    print(f"   Condition: {args.termination_condition} >= {args.termination_threshold}")
    print(f"   Poll interval: {args.poll_interval}s")
    monitor_thread = threading.Thread(
        target=termination_monitor,
        args=(
            server_url,
            termination_triggered,
            args.termination_condition,
            args.termination_threshold,
            args.poll_interval,
        ),
        daemon=True,
    )
    monitor_thread.start()

    backend = get_backend(args.cli_type)
    # Use agent_scratch_space for both local and containerized (matches pre-refactor 05602465)
    agent_scratch_space = run_manager.get_scratch_space_dir()
    agent_scratch_space.mkdir(parents=True, exist_ok=True)
    working_dir = str(agent_scratch_space)
    project_root = str(Path(__file__).resolve().parent)
    log_dir = os.path.join(str(run_manager.get_run_directory()), "agent_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info("CLI agent working_dir=%s project_root=%s", working_dir, project_root)

    if args.containerized:
        print(f"   MCP: bridge network, host.docker.internal:{args.mcp_sse_port}")

    cli_session: CliSession | None = None
    cli_log_file = None
    iteration = 0
    last_session_id: str | None = None

    # JSONL step tracking – persists across agent session restarts within one run
    processed_hashes: set[str] = set()
    from utils.llm_logger import get_llm_logger as _get_llm_logger
    _existing_steps = _get_llm_logger().cumulative_metrics.get("steps", [])
    last_cli_step: int = max((s["step"] for s in _existing_steps), default=-1)
    del _existing_steps, _get_llm_logger

    while not termination_triggered.is_set():
        iteration += 1
        logger.info(f"--- Agent session #{iteration} ---")

        session_metrics = CliSessionMetrics()
        log_path = os.path.join(log_dir, f"session_{iteration:03d}.jsonl")
        cli_log_file = open(log_path, "w")
        logger.info("Agent JSONL log: %s", log_path)

        cli_session = launch_cli_agent(
            backend,
            server_url=server_url,
            directive_path=args.directive,
            working_dir=working_dir,
            project_root=project_root,
            dangerously_skip_permissions=args.dangerously_skip_permissions,
            log_file=cli_log_file,
            metrics=session_metrics,
            containerized=args.containerized,
            session_number=iteration,
            resume_session_id=last_session_id,
            thinking_effort=args.agent_thinking_effort,
            mcp_sse_port=args.mcp_sse_port if args.containerized else None,
            run_id=run_id,
            agent_memory_dir=str(agent_memory_dir),
        )

        wait_start = time.monotonic()
        last_heartbeat = 0.0
        last_checkpoint_time = time.monotonic()

        while cli_session.process.poll() is None:
            if services.server and services.server.poll() is not None:
                logger.error("Server died, aborting")
                _terminate_process(cli_session.process, 10, "Stopping agent", use_process_group=True)
                _cleanup_cli_session(cli_session, cli_log_file)
                return ("server_died", cli_session, None)
            now = time.monotonic()

            if now - last_checkpoint_time >= 60.0:
                try:
                    requests.post(f"{server_url}/checkpoint", timeout=5)
                    requests.post(f"{server_url}/save_agent_history", timeout=5)
                    logger.debug("Saved checkpoint and agent history")
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")
                last_checkpoint_time = now

            if now - last_heartbeat >= 15.0:
                elapsed = int(now - wait_start)
                logger.info("Agent running (pid=%s, elapsed=%ds)", cli_session.process.pid, elapsed)
                last_heartbeat = now
                # Poll Claude Code JSONL files and sync new steps to server
                processed_hashes, last_cli_step = _poll_claude_jsonl_and_append_steps(
                    agent_memory_dir, processed_hashes, last_cli_step, server_url=server_url
                )

            time.sleep(1)

        _cleanup_cli_session(cli_session, cli_log_file)
        cli_log_file = None

        if session_metrics.session_id:
            last_session_id = session_metrics.session_id

        logger.info(
            "Session #%d metrics: cost=$%.4f tokens=%d/%d turns=%d tools=%d session_id=%s",
            iteration,
            session_metrics.total_cost_usd,
            session_metrics.input_tokens,
            session_metrics.output_tokens,
            session_metrics.num_turns,
            session_metrics.tool_use_count,
            last_session_id or "none",
        )
        # Final JSONL poll after session exits to capture any buffered entries
        processed_hashes, last_cli_step = _poll_claude_jsonl_and_append_steps(
            agent_memory_dir, processed_hashes, last_cli_step, server_url=server_url
        )

        if termination_triggered.is_set():
            logger.info("Termination condition met, not restarting agent.")
            termination_reason = "termination_condition_met"
            break
        if session_metrics.auth_fatal_error:
            logger.error("Auth fatal error detected, stopping (do not restart).")
            termination_reason = "auth_error"
            break
        if (
            cli_session.process.returncode == 0
            and not session_metrics.session_id
            and session_metrics.num_turns == 0
            and session_metrics.tool_use_count == 0
        ):
            logger.error(
                "Agent exited with code 0 but produced no output (no session, turns, or tools). "
                "Likely MCP connection failure. Check MCP SSE server is running and reachable at host.docker.internal."
            )
            termination_reason = "mcp_connection_failure"
            break
        logger.info(
            "Agent session #%d exited (code=%s), restarting...",
            iteration,
            cli_session.process.returncode,
        )
        time.sleep(3)

    if termination_triggered.is_set():
        print("\n✅ Task complete (termination condition met).")
    return (termination_reason, cli_session, cli_log_file)


def main():
    """Main entry point for CLI agent experiments."""
    parser = argparse.ArgumentParser(
        description="Run external CLI agents (Claude Code, Codex) for Pokemon Emerald experiments"
    )
    parser.add_argument("--cli-type", type=str, default="claude", choices=["claude", "codex"],
                       help="Type of CLI agent to launch (default: claude)")
    parser.add_argument("--login", action="store_true",
                       help="Run 'claude auth login' before starting (when --cli-type claude)")
    parser.add_argument("--directive", type=str,
                       default="agent/prompts/cli_directives/pokemon_directive.md",
                       help="Path to system prompt/directive file for CLI agent")
    parser.add_argument("--port", type=int, default=8000, help="Port for the game server (default: 8000)")
    parser.add_argument("--load-state", type=str, help="Load a saved state file on startup")
    parser.add_argument("--load-checkpoint", action="store_true", help="Load from checkpoint files")
    parser.add_argument("--backup-state", type=str,
                       help="Load from a backup zip file (extracts into run cache and auto-enables --load-checkpoint).")
    parser.add_argument("--termination-condition", type=str, default="gym_badge_count",
                       help="Termination condition type (default: gym_badge_count)")
    parser.add_argument("--termination-threshold", type=int, default=1,
                       help="Threshold for termination condition (default: 1 for first badge)")
    parser.add_argument("--poll-interval", type=int, default=10,
                       help="Seconds between termination condition polls (default: 10)")
    parser.add_argument("--graceful-timeout", type=int, default=30,
                       help="Graceful shutdown timeout in seconds before force kill (default: 30)")
    parser.add_argument("--dangerously-skip-permissions", action=argparse.BooleanOptionalAction, default=True,
                       help="Run Claude in YOLO mode (default: enabled).")
    parser.add_argument("--record", action="store_true", help="Record video of the gameplay")
    parser.add_argument("--no-ocr", action="store_true", default=True, help="Disable OCR dialogue detection")
    parser.add_argument("--direct-objectives", type=str, help="Load a specific direct objective sequence")
    parser.add_argument("--direct-objectives-start", type=int, default=0, help="Start index for direct objectives")
    parser.add_argument("--run-name", type=str, default=None, help="Optional name for the run directory")
    parser.add_argument("--containerized", action="store_true", default=False,
                       help="Run CLI agent in containerized environment")
    parser.add_argument("--build", action="store_true",
                       help="Build the container image before running (when --containerized)")
    parser.add_argument("--mcp-sse-port", type=int, default=None,
                       help="Port for MCP SSE server when containerized (default: game_port + 2)")
    parser.add_argument("--agent-thinking-effort", type=str, choices=["low", "medium", "high"],
                       help="Thinking effort level for CLI agent (low/medium/high)")

    args = parser.parse_args()

    print("=" * 60)
    print("🎮 Pokemon Emerald - External CLI Agent Experiment")
    print("=" * 60)

    from utils.run_data_manager import initialize_run_data_manager

    run_manager = initialize_run_data_manager(run_name=args.run_name or f"cli_{args.cli_type}")
    run_id = run_manager.run_id
    print(f"📁 Run data directory: {run_manager.get_run_directory()}")

    llm_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["LLM_SESSION_ID"] = llm_session_id
    os.environ["RUN_DATA_ID"] = run_id
    os.environ["LLM_METRICS_WRITE_ENABLED"] = "false"  # server is the single writer; run_cli syncs via /sync_llm_metrics
    print(f"📝 Session ID: {llm_session_id}")

    run_manager.save_metadata(
        command_args=vars(args),
        sys_argv=sys.argv,
        additional_info={
            "entry_point": "run_cli.py",
            "cli_type": args.cli_type,
            "termination_condition": args.termination_condition,
            "termination_threshold": args.termination_threshold,
        },
    )

    if not preflight_cli(args):
        return 1

    if args.cli_type == "claude" and args.login:
        if not _run_claude_login():
            return 1

    if args.backup_state:
        if _restore_from_backup(args.backup_state):
            args.load_checkpoint = True

    services = _start_services(args, run_manager)
    if not services:
        print("❌ Failed to start services, exiting...")
        return 1

    from utils.run_data_manager import get_cache_path

    backend = get_backend(args.cli_type)
    agent_memory_dir = get_cache_path(backend.agent_memory_subdir)
    agent_memory_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Agent memory directory: {agent_memory_dir}")

    if args.containerized and args.build:
        if not _build_container_image(backend):
            return 1

    if args.containerized:
        backend.seed_agent_auth(agent_memory_dir)

    termination_reason: str | None = None
    cli_session: CliSession | None = None
    cli_log_file = None

    try:
        termination_reason, cli_session, cli_log_file = _run_agent_loop(
            services, args, run_manager, agent_memory_dir
        )
        if termination_reason == "server_died":
            return 1
        if termination_reason == "auth_error":
            print("\n❌ Auth error: Run 'claude auth login' on the host, then retry with --login or ensure credentials are valid.")
            return 1
        if termination_reason == "mcp_connection_failure":
            print("\n❌ MCP connection failure: Agent could not connect to MCP server.")
            print("   MCP server must be reachable at host.docker.internal on the configured port.")
            print("   Ensure --add-host=host.docker.internal:host-gateway resolves on this host.")

            return 1
        return 0
    except KeyboardInterrupt:
        termination_reason = "user_interrupt"
        print("\n\n🛑 Shutdown requested by user")
        return 0
    finally:
        if termination_reason:
            try:
                from utils.backup_manager import create_cli_agent_termination_backup
                backup_path = create_cli_agent_termination_backup(run_id, termination_reason)
                if backup_path:
                    print(f"📦 Termination backup: {backup_path}")
            except Exception as e:
                logger.warning("Failed to create termination backup: %s", e)
        _cleanup_services(
            services,
            cli_session,
            cli_log_file,
            args,
            graceful_timeout=args.graceful_timeout,
        )


if __name__ == "__main__":
    sys.exit(main())

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
import subprocess
import signal
import threading
import logging
import shutil
import json
import secrets
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cli_agent_backends import (
    CliSession,
    CliSessionMetrics,
    get_backend,
    log_session_to_llm_logger,
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


def start_mcp_sse_server(server_url: str, mcp_port: int, project_root: str | None = None):
    """Start the MCP server in SSE transport mode for containerized agents.
    
    Args:
        server_url: Game server URL (for the MCP proxy to forward to)
        mcp_port: Port for the MCP SSE server
        project_root: Project root for PYTHONPATH
        
    Returns:
        subprocess.Popen: MCP server process
    """
    try:
        python_exe = sys.executable
        mcp_cmd = [python_exe, "-m", "server.cli.pokemon_mcp_server"]
        
        # Set up environment for SSE transport
        mcp_env = os.environ.copy()
        mcp_env["MCP_TRANSPORT"] = "sse"
        mcp_env["MCP_PORT"] = str(mcp_port)
        mcp_env["POKEMON_SERVER_URL"] = server_url
        if project_root:
            mcp_env["PYTHONPATH"] = project_root
        
        mcp_process = subprocess.Popen(
            mcp_cmd,
            env=mcp_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT  # Merge stderr into stdout for easier capture
        )
        print(f"🌐 MCP SSE server started with PID {mcp_process.pid} on port {mcp_port}")
        print(f"   Remote clients can connect via: http://localhost:{mcp_port}/sse")
        time.sleep(3)  # Give the server time to start (uvicorn needs a moment)
        
        # Verify MCP server started and is listening
        import socket as _socket
        poll_result = mcp_process.poll()
        port_listening = False
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.settimeout(1)
                port_listening = (s.connect_ex(('127.0.0.1', mcp_port)) == 0)
        except Exception:
            pass
        
        if not port_listening:
            print(f"   ⚠️  MCP server port {mcp_port} is NOT listening!")
            if poll_result is not None:
                try:
                    stdout_data = mcp_process.stdout.read(4096).decode("utf-8", errors="replace") if mcp_process.stdout else ""
                    print(f"   MCP server exited with code {poll_result}:")
                    print(f"   {stdout_data[:300]}")
                except Exception:
                    pass
        else:
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
    snapshot_path=None,
    containerized: bool = False,
    session_number: int = 1,
    resume_session_id: str | None = None,
    thinking_effort: str | None = None,
    mcp_sse_port: int | None = None,
    run_id: str | None = None,
    claude_memory_dir: str | None = None,
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
        claude_memory_dir=claude_memory_dir,
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

    logger.info("[cli-debug] launch_cli_agent: subprocess spawned pid=%s, stdin fed+closed", process.pid)
    stream_stop_event = threading.Event()
    snapshot_path_arg = Path(snapshot_path) if snapshot_path else None
    stream_thread = threading.Thread(
        target=backend.run_stream_reader,
        args=(process.stdout, stream_stop_event, log_file, metrics, server_url, snapshot_path_arg),
        daemon=True,
    )
    stream_thread.start()
    logger.info("[cli-debug] launch_cli_agent: stream thread started")
    print(f"✅ CLI agent started with PID {process.pid}")
    return CliSession(
        process=process,
        stop_event=stream_stop_event,
        stream_thread=stream_thread,
        temp_mcp_config_path=temp_mcp_config_path,
    )


def main():
    """Main entry point for CLI agent experiments."""
    parser = argparse.ArgumentParser(
        description="Run external CLI agents (Claude Code, Codex) for Pokemon Emerald experiments"
    )
    
    # CLI agent configuration
    parser.add_argument("--cli-type", type=str, default="claude",
                       choices=["claude", "codex"],
                       help="Type of CLI agent to launch (default: claude)")
    parser.add_argument("--directive", type=str, 
                       default="agent/prompts/cli_directives/pokemon_directive.md",
                       help="Path to system prompt/directive file for CLI agent")
    
    # Server configuration
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for the game server (default: 8000)")
    parser.add_argument("--load-state", type=str,
                       help="Load a saved state file on startup")
    parser.add_argument("--load-checkpoint", action="store_true",
                       help="Load from checkpoint files")
    parser.add_argument(
        "--backup-state",
        type=str,
        help="Load from a backup zip file (extracts into run cache and auto-enables --load-checkpoint).",
    )
    
    # Termination condition
    parser.add_argument("--termination-condition", type=str, default="gym_badge_count",
                       help="Termination condition type (default: gym_badge_count)")
    parser.add_argument("--termination-threshold", type=int, default=1,
                       help="Threshold for termination condition (default: 1 for first badge)")
    parser.add_argument("--poll-interval", type=int, default=10,
                       help="Seconds between termination condition polls (default: 10)")
    parser.add_argument("--graceful-timeout", type=int, default=30,
                       help="Graceful shutdown timeout in seconds before force kill (default: 30)")
    parser.add_argument(
        "--dangerously-skip-permissions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Claude in YOLO mode (default: enabled). Use --no-dangerously-skip-permissions to disable.",
    )
    
    # Features
    parser.add_argument("--record", action="store_true",
                       help="Record video of the gameplay")
    parser.add_argument("--no-ocr", action="store_true", default=True,
                       help="Disable OCR dialogue detection")
    parser.add_argument("--direct-objectives", type=str,
                       help="Load a specific direct objective sequence")
    parser.add_argument("--direct-objectives-start", type=int, default=0,
                       help="Start index for direct objectives")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Optional name for the run directory")
    
    # Containerization
    parser.add_argument("--containerized", action="store_true", default=False,
                       help="Run CLI agent in containerized environment (default: disabled for compatibility)")
    parser.add_argument("--mcp-sse-port", type=int, default=None,
                       help="Port for MCP SSE server when containerized (default: game_port + 2)")
    
    # Agent thinking effort
    parser.add_argument("--agent-thinking-effort", type=str, 
                       choices=["low", "medium", "high"],
                       help="Thinking effort level for CLI agent (low/medium/high)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎮 Pokemon Emerald - External CLI Agent Experiment")
    print("=" * 60)
    
    # Initialize run data manager
    from utils.run_data_manager import initialize_run_data_manager
    
    run_manager = initialize_run_data_manager(
        run_name=args.run_name or f"cli_{args.cli_type}"
    )
    run_id = run_manager.run_id
    print(f"📁 Run data directory: {run_manager.get_run_directory()}")
    
    # Generate LLM session_id
    llm_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["LLM_SESSION_ID"] = llm_session_id
    os.environ["RUN_DATA_ID"] = run_id
    os.environ["POKEAGENT_CLI_MODE"] = "1"
    print(f"📝 Session ID: {llm_session_id}")
    
    # Save metadata
    run_manager.save_metadata(
        command_args=vars(args),
        sys_argv=sys.argv,
        additional_info={
            "entry_point": "run_cli.py",
            "cli_type": args.cli_type,
            "termination_condition": args.termination_condition,
            "termination_threshold": args.termination_threshold
        }
    )
    
    if not preflight_cli(args):
        return 1

    server_process = None
    frame_server_process = None
    cli_session = None
    cli_log_file = None
    monitor_thread = None
    termination_triggered = threading.Event()
    
    try:
        # Restore from backup if requested (same behavior as run.py)
        if args.backup_state:
            from utils.backup_manager import restore_cache_from_backup
            from utils.run_data_manager import get_cache_directory

            print(f"\n📦 Restoring from backup: {args.backup_state}")
            success = restore_cache_from_backup(
                backup_file=args.backup_state,
                create_backup_of_current=False,
            )

            if success:
                print(f"✅ Backup restored to: {get_cache_directory()}")
                args.load_checkpoint = True
            else:
                print("❌ Failed to restore backup, continuing with fresh state")

        # Start server
        print("\n📡 Starting server process...")
        server_process = start_server(args, run_id)
        
        if not server_process:
            print("❌ Failed to start server, exiting...")
            return 1
        
        # Start frame server for visualization
        frame_server_process = start_frame_server(args.port)
        
        server_url = f"http://localhost:{args.port}"
        print(f"\n🎥 Stream View: http://127.0.0.1:{args.port}/stream")
        
        # Verify server is healthy
        try:
            health_response = requests.get(f"{server_url}/health", timeout=5)
            if health_response.status_code != 200:
                print(f"⚠️ Server health check failed: {health_response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not verify server health: {e}")
        
        # Resolve MCP SSE port: default to game_port + 2 (game_port + 1 is frame server)
        if args.mcp_sse_port is None:
            args.mcp_sse_port = args.port + 2
        
        # Start MCP SSE server if containerized mode is enabled
        mcp_process = None
        if args.containerized:
            print(f"\n🐳 Containerized mode enabled")
            print(f"   MCP SSE server will run on port {args.mcp_sse_port}")
            project_root_for_mcp = str(Path(__file__).resolve().parent)
            mcp_process = start_mcp_sse_server(server_url, args.mcp_sse_port, project_root_for_mcp)
            if not mcp_process:
                print("❌ Failed to start MCP SSE server, exiting...")
                return 1
        
        # Prepare claude_memory directory for session persistence
        from utils.run_data_manager import get_cache_path
        claude_memory_dir = get_cache_path("claude_memory")
        claude_memory_dir.mkdir(parents=True, exist_ok=True)
        
        # For containerized mode: seed with host's Claude Code subscription auth
        if args.containerized:
            host_claude_dir = Path.home() / ".claude"
            host_claude_json = Path.home() / ".claude.json"  # Config lives in home directory
            if host_claude_dir.exists() or host_claude_json.exists():
                import shutil
                # Copy auth files (not session history) from host to container mount
                # Note: .claude.json config is in home dir, .credentials.json is in .claude/ subdir
                auth_files = [
                    (host_claude_dir / "settings.json", "settings.json"),
                    (host_claude_dir / ".credentials.json", ".credentials.json"),  # Credentials in .claude/ subdir
                    (host_claude_json, ".claude.json"),  # Config from home directory
                ]
                seeded_any = False
                for src, dst_name in auth_files:
                    if src.exists():
                        dst = claude_memory_dir / dst_name
                        if not dst.exists():  # Only seed on first run
                            shutil.copy2(src, dst)
                            print(f"   ✓ Seeded {src.name} -> {dst_name}")
                            seeded_any = True
                if not seeded_any:
                    print("⚠️  No Claude auth files found")
                    print("   Run 'claude auth login' first, then retry.")
            else:
                print("⚠️  Host ~/.claude/ not found. Run 'claude auth login' first.")
                print("   Container will not be able to authenticate.")
        
        print(f"\n💾 Claude memory directory: {claude_memory_dir}")
        
        # Start termination monitor thread (flag-only; does not kill process)
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

        cli_session = None
        cli_log_file = None
        iteration = 0
        last_session_id: str | None = None  # track for --resume across iterations
        backend = get_backend(args.cli_type)

        # CLI agent cwd = run's agent_scratch_space so generated files stay per-run
        agent_scratch_space = run_manager.get_scratch_space_dir()
        agent_scratch_space.mkdir(parents=True, exist_ok=True)
        working_dir = str(agent_scratch_space)
        # Project root for MCP server PYTHONPATH (so it can import server.*, utils.*)
        project_root = str(Path(__file__).resolve().parent)
        logger.info("CLI agent working_dir=%s project_root=%s", working_dir, project_root)

        log_dir = os.path.join(str(run_manager.get_run_directory()), "agent_logs")
        os.makedirs(log_dir, exist_ok=True)

        while not termination_triggered.is_set():
            iteration += 1
            logger.info(f"--- Agent session #{iteration} ---")

            session_metrics = CliSessionMetrics()
            log_path = os.path.join(log_dir, f"session_{iteration:03d}.jsonl")
            cli_log_file = open(log_path, "w")
            logger.info("Agent JSONL log: %s", log_path)

            from utils.run_data_manager import get_cache_path
            snapshot_path = get_cache_path("cli_metrics_snapshot.json")

            cli_session = launch_cli_agent(
                backend,
                server_url=server_url,
                directive_path=args.directive,
                working_dir=working_dir,
                project_root=project_root,
                dangerously_skip_permissions=args.dangerously_skip_permissions,
                log_file=cli_log_file,
                metrics=session_metrics,
                snapshot_path=snapshot_path,
                containerized=args.containerized,
                session_number=iteration,
                resume_session_id=last_session_id,
                thinking_effort=args.agent_thinking_effort,
                mcp_sse_port=args.mcp_sse_port if args.containerized else None,
                run_id=run_id,
                claude_memory_dir=str(claude_memory_dir),
            )

            logger.info("[cli-debug] main: entered wait loop for CLI pid=%s", cli_session.process.pid)
            wait_start = time.monotonic()
            last_debug_log = 0.0
            last_checkpoint_time = time.monotonic()

            while cli_session.process.poll() is None:
                if server_process.poll() is not None:
                    logger.error("Server died, aborting")
                    _terminate_process(
                        cli_session.process, 10, "Stopping agent", use_process_group=True
                    )
                    return 1
                now = time.monotonic()
                
                # Checkpoint every 60 seconds
                if now - last_checkpoint_time >= 60.0:
                    try:
                        # Save checkpoint (game state + milestones)
                        requests.post(f"{server_url}/checkpoint", timeout=5)
                        # Save agent history (LLM logs)
                        requests.post(f"{server_url}/save_agent_history", timeout=5)
                        logger.info("[cli-debug] Saved checkpoint and agent history")
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint: {e}")
                    last_checkpoint_time = now

                if now - last_debug_log >= 15.0:
                    elapsed = int(now - wait_start)
                    logger.info(
                        "[cli-debug] main: still waiting for CLI agent to exit (pid=%s, elapsed=%ds)",
                        cli_session.process.pid,
                        elapsed,
                    )
                    last_debug_log = now
                    
                    # Save intermediate metrics snapshot
                    try:
                        with open(snapshot_path, "w") as f:
                            json.dump(asdict(session_metrics), f, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to save metrics snapshot: {e}")
                        
                time.sleep(1)

            _cleanup_cli_session(cli_session, cli_log_file)
            cli_log_file = None

            # Persist session_id for --resume on next iteration
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
            log_session_to_llm_logger(session_metrics, iteration, backend.name)

            if termination_triggered.is_set():
                logger.info("Termination condition met, not restarting agent.")
                break
            logger.info(
                "Agent session #%d exited (code=%s), restarting...",
                iteration,
                cli_session.process.returncode,
            )
            time.sleep(3)

        if termination_triggered.is_set():
            print("\n✅ Task complete (termination condition met).")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
        return 0
        
    finally:
        if cli_session is not None and cli_session.process.poll() is None:
            _terminate_process(
                process=cli_session.process,
                graceful_timeout=args.graceful_timeout if "args" in locals() else 30,
                label="🤖 Stopping CLI agent",
                use_process_group=True,
            )
        if cli_log_file and not cli_log_file.closed:
            cli_log_file.close()
        # Clean up server
        if server_process:
            print("📡 Stopping server process...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("   Force killing server...")
                server_process.kill()
        
        # Clean up frame server
        if frame_server_process:
            print("🖼️  Stopping frame server...")
            frame_server_process.terminate()
            try:
                frame_server_process.wait(timeout=2)
            except:
                frame_server_process.kill()
        
        # Clean up MCP SSE server (if containerized mode was used)
        if 'mcp_process' in locals() and mcp_process:
            print("🌐 Stopping MCP SSE server...")
            mcp_process.terminate()
            try:
                mcp_process.wait(timeout=2)
            except:
                mcp_process.kill()
        
        print("👋 Goodbye!")


if __name__ == "__main__":
    sys.exit(main())

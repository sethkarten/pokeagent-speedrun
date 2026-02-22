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
import tempfile
import json
from datetime import datetime
from pathlib import Path

import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _build_claude_bootstrap_prompt(directive_content: str, server_url: str) -> str:
    """Build the first prompt sent to the interactive Claude session."""
    return (
        f"{directive_content}\n\n"
        "Runtime context:\n"
        f"- Pokemon server URL: {server_url}\n"
        "- You are running in a long-lived interactive session.\n"
        "- Act autonomously and continuously, using MCP tools directly as needed.\n"
        "- Poll game state on your own via MCP tools; do not wait for additional operator prompts.\n"
        "- Continue until externally terminated by the orchestrator when completion condition is met.\n"
    )


def _stream_pipe_output(
    stdout_pipe,
    stop_event: threading.Event,
) -> None:
    """Stream subprocess stdout pipe to console for observability.

    Uses unbuffered reads (bufsize=0 on the Popen) so each os-level read
    returns as soon as data is available, giving real-time streaming.
    """
    first_data = True
    logger.info("[cli-debug] stream_pipe_output: thread started")
    try:
        while not stop_event.is_set():
            data = stdout_pipe.read(4096)
            if not data:
                logger.info("[cli-debug] stream_pipe_output: EOF from CLI")
                return
            if first_data:
                logger.info("[cli-debug] stream_pipe_output: first data received (%d bytes)", len(data))
                first_data = False
            sys.stdout.write(data.decode(errors="replace"))
            sys.stdout.flush()
    except (OSError, ValueError) as e:
        logger.info("[cli-debug] stream_pipe_output: error: %s", e)


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


def _cleanup_cli_session(master_fd, stop_event, temp_mcp_config_path):
    """Clean up resources from a single CLI agent session."""
    if stop_event:
        stop_event.set()
    if master_fd is not None:
        try:
            os.close(master_fd)
        except OSError:
            pass
    if temp_mcp_config_path:
        try:
            os.remove(temp_mcp_config_path)
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
    cli_type: str,
    server_url: str,
    directive_path: str,
    working_dir: str = None,
    dangerously_skip_permissions: bool = True,
) -> tuple[subprocess.Popen, int, threading.Event, threading.Thread, str | None]:
    """Launch an external CLI agent session as subprocess.
    
    Args:
        cli_type: Type of CLI agent ("claude" or "codex")
        server_url: MCP server URL for the agent to connect to
        directive_path: Path to the system prompt/directive file
        working_dir: Working directory for the CLI agent
        
    Returns:
        Tuple containing process, pty master FD, stop event, stream thread, and temp mcp config path (if any)
    """
    if working_dir is None:
        working_dir = os.getcwd()
    
    # Read directive content
    directive_content = ""
    if directive_path and os.path.exists(directive_path):
        with open(directive_path, 'r') as f:
            directive_content = f.read()
        print(f"📜 Loaded directive from: {directive_path}")
    else:
        print(f"⚠️ Directive file not found: {directive_path}")
    
    temp_mcp_config_path = None

    bootstrap = ""

    # Build command based on CLI type
    if cli_type == "claude":
        # --print: non-interactive batch mode (read prompt, process, print, exit).
        # Without --print Claude enters TUI mode which hangs as a subprocess.
        cmd = ["claude", "--print"]
        if dangerously_skip_permissions:
            cmd.append("--dangerously-skip-permissions")

        env = os.environ.copy()
        env["POKEMON_MCP_SERVER_URL"] = server_url
        env["POKEMON_SERVER_URL"] = server_url

        # Build a run-local MCP config so Claude can invoke the Pokemon MCP server
        # with the correct backend server URL/port.
        mcp_config = {
            "mcpServers": {
                "pokemon-emerald": {
                    "command": sys.executable,
                    "args": ["-m", "server.cli.pokemon_mcp_server"],
                    "env": {
                        "POKEMON_SERVER_URL": server_url,
                        "PYTHONPATH": os.getcwd(),
                    },
                }
            }
        }
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(mcp_config, temp_file, indent=2)
        temp_file.flush()
        temp_file.close()
        temp_mcp_config_path = temp_file.name
        cmd.extend(["--mcp-config", temp_mcp_config_path])

        if directive_content:
            bootstrap = _build_claude_bootstrap_prompt(directive_content, server_url)
            logger.info("[cli-debug] launch_cli_agent: directive+bootstrap length=%s chars", len(bootstrap))

    elif cli_type == "codex":
        raise NotImplementedError(
            "Codex CLI integration is not implemented yet. Use --cli-type claude for now."
        )
    else:
        raise ValueError(f"Unknown CLI type: {cli_type}. Supported: claude, codex")
    
    print(f"🤖 Launching {cli_type} CLI agent...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Working directory: {working_dir}")
    print(f"   MCP Server: {server_url}")
    
    # --print mode is non-interactive; regular pipes suffice (no PTY needed).
    # Prompt is fed via stdin, matching the pattern from ralph.sh.
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

    # Feed directive via stdin then close (like: claude --print < directive.md)
    if bootstrap:
        process.stdin.write(bootstrap.encode())
    process.stdin.close()

    logger.info("[cli-debug] launch_cli_agent: subprocess spawned pid=%s, stdin fed+closed", process.pid)
    stream_stop_event = threading.Event()
    stream_thread = threading.Thread(
        target=_stream_pipe_output,
        args=(process.stdout, stream_stop_event),
        daemon=True,
    )
    stream_thread.start()
    logger.info("[cli-debug] launch_cli_agent: stream thread started")

    print(f"✅ CLI agent started with PID {process.pid}")
    # Return None for master_fd (no PTY); cleanup handles None gracefully.
    return process, None, stream_stop_event, stream_thread, temp_mcp_config_path


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
    cli_process = None
    cli_master_fd = None
    cli_stream_stop_event = None
    cli_stream_thread = None
    temp_mcp_config_path = None
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

        cli_process = None
        iteration = 0
        while not termination_triggered.is_set():
            iteration += 1
            logger.info(f"--- Agent session #{iteration} ---")
            (
                cli_process,
                cli_master_fd,
                cli_stream_stop_event,
                cli_stream_thread,
                temp_mcp_config_path,
            ) = launch_cli_agent(
                cli_type=args.cli_type,
                server_url=server_url,
                directive_path=args.directive,
                working_dir=os.getcwd(),
                dangerously_skip_permissions=args.dangerously_skip_permissions,
            )

            logger.info("[cli-debug] main: entered wait loop for CLI pid=%s", cli_process.pid)
            # Wait for Claude to exit naturally
            wait_start = time.monotonic()
            last_debug_log = 0.0
            while cli_process.poll() is None:
                if server_process.poll() is not None:
                    logger.error("Server died, aborting")
                    _terminate_process(
                        cli_process, 10, "Stopping agent", use_process_group=True
                    )
                    return 1
                now = time.monotonic()
                if now - last_debug_log >= 15.0:
                    elapsed = int(now - wait_start)
                    logger.info(
                        "[cli-debug] main: still waiting for CLI agent to exit (pid=%s, elapsed=%ds)",
                        cli_process.pid,
                        elapsed,
                    )
                    last_debug_log = now
                time.sleep(1)

            _cleanup_cli_session(cli_master_fd, cli_stream_stop_event, temp_mcp_config_path)

            if termination_triggered.is_set():
                logger.info("Termination condition met, not restarting agent.")
                break
            logger.info(
                f"Agent session #{iteration} exited (code={cli_process.returncode}), restarting..."
            )
            time.sleep(3)

        if termination_triggered.is_set():
            print("\n✅ Task complete (termination condition met).")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
        return 0
        
    finally:
        if cli_process is not None and cli_process.poll() is None:
            _terminate_process(
                process=cli_process,
                graceful_timeout=args.graceful_timeout if "args" in locals() else 30,
                label="🤖 Stopping CLI agent",
                use_process_group=True,
            )
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
        
        print("👋 Goodbye!")


if __name__ == "__main__":
    sys.exit(main())

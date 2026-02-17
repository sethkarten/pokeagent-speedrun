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
import pty
import select
import tempfile
import json
import re
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
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


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


def _stream_pty_output(
    master_fd: int,
    stop_event: threading.Event,
    auto_accept_bypass: bool = False,
) -> None:
    """Stream PTY output to stdout for observability."""
    pending = ""
    accepted = False
    while not stop_event.is_set():
        try:
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd not in ready:
                continue
            data = os.read(master_fd, 4096)
            if not data:
                return
            text = data.decode(errors="replace")
            pending = (pending + text)[-8000:]
            sys.stdout.write(text)
            sys.stdout.flush()

            # Claude may block on first-time bypass permissions confirmation.
            # Auto-confirm once so the orchestrator does not stall indefinitely.
            if auto_accept_bypass and not accepted:
                clean_pending = ANSI_ESCAPE_RE.sub("", pending)
                clean_pending = " ".join(clean_pending.split())
                normalized = re.sub(r"[^a-z]", "", clean_pending.lower())
                if "bypasspermissionsmode" in normalized and "entertoconfirm" in normalized:
                    os.write(master_fd, b"2\r")
                    accepted = True
        except OSError:
            return


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
    cli_process: subprocess.Popen,
    condition_type: str = "gym_badge_count",
    threshold: int = 1,
    poll_interval: int = 10,
    graceful_timeout: int = 30,
    termination_triggered: threading.Event | None = None,
):
    """Monitor termination condition and terminate CLI agent when met.
    
    Runs in a separate thread, polls the server for termination condition,
    and sends SIGTERM to CLI agent when condition is met.
    
    Args:
        server_url: Base URL of the server
        cli_process: CLI agent subprocess
        condition_type: Type of termination condition
        threshold: Threshold value for the condition
        poll_interval: Seconds between polls
        graceful_timeout: Seconds to wait after SIGTERM before SIGKILL
    """
    logger.info(f"Termination monitor started: {condition_type} >= {threshold}, polling every {poll_interval}s")
    
    while cli_process.poll() is None:  # While CLI agent is running
        result = check_termination_condition(server_url, condition_type, threshold)
        
        if result.get("condition_met"):
            if termination_triggered is not None:
                termination_triggered.set()
            logger.info(f"🎉 Termination condition met! {condition_type}={result.get('current_value', '?')} >= {threshold}")
            if result.get("badge_names"):
                logger.info(f"   Badges: {result['badge_names']}")
            
            # Graceful shutdown
            print(f"\n🛑 Termination condition met - stopping CLI agent...")
            _terminate_process(
                process=cli_process,
                graceful_timeout=graceful_timeout,
                label="🤖 Stopping CLI agent",
                use_process_group=True,
            )
            
            return
        
        time.sleep(poll_interval)
    
    logger.info("CLI agent exited on its own")


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

    # Build command based on CLI type
    if cli_type == "claude":
        cmd = ["claude"]
        if dangerously_skip_permissions:
            # Use non-interactive permission mode to avoid approval stalls.
            cmd.extend(["--permission-mode", "dontAsk"])

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
            cmd.append(_build_claude_bootstrap_prompt(directive_content, server_url))

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
    
    # Launch interactive CLI agent in PTY so Claude behaves like terminal mode
    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        cmd,
        cwd=working_dir,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        preexec_fn=os.setsid,
        close_fds=True,
    )
    os.close(slave_fd)

    stream_stop_event = threading.Event()
    stream_thread = threading.Thread(
        target=_stream_pty_output,
        args=(master_fd, stream_stop_event, dangerously_skip_permissions),
        daemon=True,
    )
    stream_thread.start()

    print(f"✅ CLI agent started with PID {process.pid}")
    return process, master_fd, stream_stop_event, stream_thread, temp_mcp_config_path


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
                       default="agent/cli_directives/pokemon_directive.md",
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
        
        # Launch CLI agent
        print(f"\n🤖 Launching {args.cli_type} CLI agent...")
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
        
        # Start termination monitor thread
        print(f"\n👁️  Starting termination monitor...")
        print(f"   Condition: {args.termination_condition} >= {args.termination_threshold}")
        print(f"   Poll interval: {args.poll_interval}s")
        
        monitor_thread = threading.Thread(
            target=termination_monitor,
            args=(
                server_url,
                cli_process,
                args.termination_condition,
                args.termination_threshold,
                args.poll_interval,
                args.graceful_timeout,
                termination_triggered,
            ),
            daemon=True
        )
        monitor_thread.start()
        
        # Wait for CLI agent to complete, while ensuring server stays healthy
        print("\n⏳ Waiting for CLI agent to complete or termination condition...")
        while True:
            cli_exit = cli_process.poll()
            if cli_exit is not None:
                break

            if server_process and server_process.poll() is not None:
                print("\n❌ Server exited unexpectedly while CLI agent is still running.")
                _terminate_process(
                    process=cli_process,
                    graceful_timeout=args.graceful_timeout,
                    label="🤖 Stopping CLI agent",
                    use_process_group=True,
                )
                return 1

            time.sleep(1)

        exit_code = cli_exit
        if termination_triggered.is_set():
            print("\n✅ CLI agent stopped due to completion condition.")
            return 0
        if exit_code != 0:
            print("\n⚠️ CLI agent exited with a non-zero code.")
            print("   If this was unexpected, verify Claude login by running `claude` manually first.")
        print(f"\n✅ CLI agent exited with code: {exit_code}")
        return exit_code
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
        return 0
        
    finally:
        # Clean up CLI agent
        if cli_process and cli_process.poll() is None:
            _terminate_process(
                process=cli_process,
                graceful_timeout=args.graceful_timeout if "args" in locals() else 30,
                label="🤖 Stopping CLI agent",
                use_process_group=True,
            )

        if cli_stream_stop_event:
            cli_stream_stop_event.set()

        if cli_master_fd is not None:
            try:
                os.close(cli_master_fd)
            except OSError:
                pass

        if temp_mcp_config_path:
            try:
                os.remove(temp_mcp_config_path)
            except OSError:
                pass
        
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

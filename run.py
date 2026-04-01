#!/usr/bin/env python3
"""
Main entry point for the Pokemon Agent.
This is a streamlined version that focuses on multiprocess mode only.
"""

import os
import sys
import time
import argparse
import subprocess
import signal

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_AGENT_CONFIGS = {
    "pokeagent": {
        "name": "PokeAgent",
        "details": [
            "Custom VLM benchmark agent with tool scaffolding",
            "All MCP tools enabled (built-in subagents + generic primitives)",
            "Supports autonomous objective creation and prompt optimization",
        ],
        "module": "agents.PokeAgent",
        "class": "PokeAgent",
        "use_backend": True,
        "supports_prompt_optimization": True,
    },
    "simple": {
        "name": "PokeAgent",
        "details": [
            "Minimal PokeAgent scaffold (H_min baseline)",
            "No built-in subagent tools; orchestrator owns replan_objectives directly",
            "Empty subagent registry (agent must create its own)",
        ],
        "module": "agents.PokeAgent",
        "class": "PokeAgent",
        "use_backend": True,
        "supports_prompt_optimization": False,
    },
    "simplest": {  # Legacy alias for simple
        "name": "PokeAgent",
        "details": [
            "Legacy alias for 'simple' scaffold",
        ],
        "module": "agents.PokeAgent",
        "class": "PokeAgent",
        "use_backend": True,
        "supports_prompt_optimization": False,
    },
    "autoevolve": {
        "name": "PokeAgent",
        "details": [
            "AutoEvolve scaffold (H_auto): starts from H_min + evolutionary optimization",
            "No built-in subagent tools; empty registry (agent creates its own)",
            "Reset-free prompt evolution from trajectory segments",
        ],
        "module": "agents.PokeAgent",
        "class": "PokeAgent",
        "use_backend": True,
        "supports_prompt_optimization": True,
    },
    "autonomous_cli": {
        "name": "PokeAgent",
        "details": [
            "Legacy scaffold alias for PokeAgent",
            "Custom VLM benchmark agent with tool scaffolding",
            "Supports autonomous objective creation and prompt optimization",
        ],
        "module": "agents.PokeAgent",
        "class": "PokeAgent",
        "use_backend": True,
        "supports_prompt_optimization": True,
    },
    "vision_only": {
        "name": "VisionOnlyAgent",
        "details": [
            "Relies purely on visual input (screenshots)",
            "No map information or pathfinding assistance",
            "Navigates using directional buttons only",
        ],
        "module": "agents.vision_only_agent",
        "class": "VisionOnlyAgent",
        "use_backend": True,
        "supports_walkthrough": True,
        "supports_slam": True,
    },
}

SCAFFOLD_DESCRIPTIONS = {
    "pokeagent": "PokeAgent (VLM benchmark agent with tool scaffolding)",
    "simple": "PokeAgent-Simple (H_min: no built-in subagents, direct replan, empty registry)",
    "simplest": "PokeAgent-Simple (legacy alias for 'simple')",
    "autoevolve": "AutoEvolve (H_auto: H_min + reset-free evolutionary optimization)",
    "autonomous_cli": "PokeAgent (legacy alias)",
    "vision_only": "Vision-Only Agent (no map info, no pathfinding, button sequences)",
}

SUPPORTED_SCAFFOLDS = list(CUSTOM_AGENT_CONFIGS.keys())

SERVER_MANAGED_SCAFFOLDS = list(CUSTOM_AGENT_CONFIGS.keys())


def start_server(args, run_id=None):
    """Start the server process with appropriate arguments
    
    Args:
        args: Command line arguments
        run_id: Optional run_id to pass to server via environment variable
    """
    # Use the same Python executable that's running this script
    python_exe = sys.executable
    server_cmd = [python_exe, "-m", "server.app", "--port", str(args.port)]
    
    # Pass run_id and llm_session_id to server via environment variable if provided
    server_env = os.environ.copy()
    if run_id:
        server_env["RUN_DATA_ID"] = run_id
    
    # Pass LLM session_id if available (for consistent logging across processes)
    llm_session_id = os.environ.get("LLM_SESSION_ID")
    if llm_session_id:
        server_env["LLM_SESSION_ID"] = llm_session_id

    # Single-writer metrics: server is the only writer
    server_env["LLM_METRICS_WRITE_ENABLED"] = "true"

    # simple/simplest/autoevolve scaffolds start with an empty subagent registry
    if getattr(args, "scaffold", "pokeagent") in ("simple", "simplest", "autoevolve"):
        server_env["EXCLUDE_BUILTIN_SUBAGENTS"] = "1"

    # Pass through server-relevant arguments
    if args.game:
        server_cmd.extend(["--game", args.game])
        server_env["GAME_TYPE"] = args.game
        # Also set in current (client) process so agent scaffolds can detect game type
        os.environ["GAME_TYPE"] = args.game

    if args.record:
        server_cmd.append("--record")
    
    if args.load_checkpoint:
        # Auto-load checkpoint.state when --load-checkpoint is used
        from utils.data_persistence.run_data_manager import get_cache_path
        checkpoint_state = get_cache_path("checkpoint.state")
        if checkpoint_state.exists():
            server_cmd.extend(["--load-state", str(checkpoint_state)])
            # Set environment variable to enable LLM checkpoint loading
            server_env["LOAD_CHECKPOINT_MODE"] = "true"
            print(f"🔄 Server will load checkpoint: {checkpoint_state}")
            metrics_file = get_cache_path("cumulative_metrics.json")
            print(f"🔄 LLM metrics will be restored from {metrics_file}")
        else:
            print(f"⚠️ Checkpoint file not found: {checkpoint_state}")
    elif args.load_state:
        server_cmd.extend(["--load-state", args.load_state])
    
    # Don't pass --manual to server - server should always run in server mode
    # The --manual flag only affects client behavior
    
    if args.no_ocr:
        server_cmd.append("--no-ocr")
    
    if args.direct_objectives:
        server_cmd.extend(["--direct-objectives", args.direct_objectives])
        if args.direct_objectives_start > 0:
            server_cmd.extend(["--direct-objectives-start", str(args.direct_objectives_start)])
        if args.direct_objectives_battling_start > 0:
            server_cmd.extend(["--direct-objectives-battling-start", str(args.direct_objectives_battling_start)])
    
    # Server always runs headless - display handled by client
    
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
        print("⏳ Waiting 3 seconds for server to initialize...")
        time.sleep(3)
        
        return server_process
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None


def start_frame_server(port):
    """Start the lightweight frame server for stream.html visualization"""
    try:
        frame_cmd = ["python", "-m", "server.frame_server", "--port", str(port+1)]
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


def start_custom_agent(agent_config, args):
    """Generic helper to start any custom benchmark agent.

    Args:
        agent_config: Dict with keys: 'name', 'description', 'details' (list), 'module', 'class'
        args: Command line arguments

    Returns:
        Exit code from agent.run()
    """
    print(f"\n🖥️  {agent_config['name']}")
    print("=" * 60)
    print("✅ Server is running")
    print(f"🤖 Starting {agent_config['name']}...")
    for detail in agent_config.get('details', []):
        print(f"   {detail}")
    print("")

    # Ensure EXCLUDE_BUILTIN_SUBAGENTS is visible in the agent process too
    if getattr(args, "scaffold", "pokeagent") in ("simple", "simplest", "autoevolve"):
        os.environ["EXCLUDE_BUILTIN_SUBAGENTS"] = "1"

    # Dynamic import
    module = __import__(agent_config['module'], fromlist=[agent_config['class']])
    agent_class = getattr(module, agent_config['class'])
    print(f"📦 {agent_config['class']} imported successfully", flush=True)

    print(f"🔧 Creating agent with model={args.model_name}", flush=True)

    # Build agent kwargs based on config
    agent_kwargs = {
        "server_url": f"http://localhost:{args.port}",
        "model": args.model_name,
        "max_steps": args.max_steps if hasattr(args, 'max_steps') else None
    }

    # Add backend if specified in config
    if agent_config.get('use_backend', False):
        agent_kwargs["backend"] = args.backend

    # Add allow_walkthrough if specified in config
    if agent_config.get('supports_walkthrough', False):
        agent_kwargs["allow_walkthrough"] = args.allow_walkthrough if hasattr(args, 'allow_walkthrough') else False

    # Add allow_slam if specified in config
    if agent_config.get('supports_slam', False):
        agent_kwargs["allow_slam"] = args.allow_slam if hasattr(args, 'allow_slam') else False

    # Add prompt optimization if specified in config
    if agent_config.get('supports_prompt_optimization', False):
        agent_kwargs["enable_prompt_optimization"] = args.enable_prompt_optimization if hasattr(args, 'enable_prompt_optimization') else False
        agent_kwargs["optimization_frequency"] = args.optimization_frequency if hasattr(args, 'optimization_frequency') else 10

    # Pass scaffold name to PokeAgent so it can select tool set and prompt
    if agent_config.get("class") == "PokeAgent":
        agent_kwargs["scaffold"] = args.scaffold

    agent = agent_class(**agent_kwargs)
    print("✅ Agent created", flush=True)

    return agent.run()


def main():
    """Main entry point for the Pokemon Agent"""
    parser = argparse.ArgumentParser(description="Pokemon AI Agent")
    
    # Core arguments
    parser.add_argument("--game", type=str, default="emerald", choices=["red", "emerald"],
                       help="Which game to run: 'red' (Pokemon Red, Game Boy) or 'emerald' (Pokemon Emerald, GBA)")
    parser.add_argument("--rom", type=str, default="Emerald-GBAdvance/rom.gba",
                       help="Path to ROM file")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for web interface")
    
    # State loading
    parser.add_argument("--load-state", type=str, 
                       help="Load a saved state file on startup")
    parser.add_argument("--load-checkpoint", action="store_true", 
                       help="Load from checkpoint files")
    parser.add_argument("--backup-state", type=str,
                       help="Load from a backup zip file (extracts to cache and loads checkpoint). This is the preferred convention for loading a run that doesnt start from the beginning.")
    
    # Agent configuration
    parser.add_argument("--backend", type=str, default="gemini", 
                       help="VLM backend (openai, gemini, openrouter, anthropic, auto)")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash", 
                       help="Model name to use")
    parser.add_argument("--scaffold", type=str, default="pokeagent",
                       choices=SUPPORTED_SCAFFOLDS,
                       help="Agent scaffold: pokeagent (default)/autonomous_cli, or vision_only")
    
    # Operation modes
    parser.add_argument("--headless", action="store_true", 
                       help="Run without pygame display (headless)")
    parser.add_argument("--agent-auto", action="store_true", 
                       help="Agent acts automatically")
    parser.add_argument("--manual", action="store_true", 
                       help="Start in manual mode instead of agent mode")
    
    # Features
    parser.add_argument("--record", action="store_true", 
                       help="Record video of the gameplay")
    parser.add_argument("--no-ocr", action="store_true", default=True,
                       help="Disable OCR dialogue detection")
    parser.add_argument("--direct-objectives", type=str,
                       help="Load a specific direct objective sequence ('categorized_full_game' or 'autonomous_objective_creation')")
    parser.add_argument("--direct-objectives-start", type=int, default=0,
                       help="Start index for story objectives in legacy mode, or story objectives in categorized mode")
    parser.add_argument("--direct-objectives-battling-start", type=int, default=0,
                       help="Start index for battling objectives (only used in categorized mode)")
    parser.add_argument("--clear-memory", action="store_true",
                       help="Clear the memory.json file before starting the run")
    parser.add_argument("--clear-knowledge-base", action="store_true",
                       dest="clear_memory",
                       help="Deprecated alias for --clear-memory")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Optional name to append to run directory (e.g., 'test_run' -> 'run_20251129_191503_test_run')")
    parser.add_argument("--enable-prompt-optimization", action="store_true",
                       help="Enable reflective prompt optimization based on trajectory analysis")
    parser.add_argument("--optimization-frequency", type=int, default=10,
                       help="How often to run prompt optimization (default: every 10 steps)")
    parser.add_argument("--allow-walkthrough", action="store_true",
                       help="Enable get_walkthrough tool for vision_only agent")
    parser.add_argument("--allow-slam", action="store_true",
                       help="Enable SLAM (map building) for vision_only agent")
    args = parser.parse_args()

    # Fix ROM default for Red (parser default is Emerald ROM)
    if args.rom == "Emerald-GBAdvance/rom.gba" and args.game == "red":
        args.rom = "PokemonRed-GBC/pokered.gbc"

    print("=" * 60)
    game_label = "Pokemon Red" if args.game == "red" else "Pokemon Emerald"
    print(f"🎮 {game_label} AI Agent")
    print("=" * 60)
    
    # Get first objective info for consistent run naming
    first_objective_id = None
    first_objective_desc = None
    if args.direct_objectives:
        from agents.objectives import get_first_objective_info
        first_objective_id, first_objective_desc = get_first_objective_info(
            args.direct_objectives, 
            args.direct_objectives_start
        )
        if first_objective_id:
            print(f"🎯 First objective: {first_objective_id} - {first_objective_desc}")
        else:
            print(f"⚠️  Could not retrieve first objective from '{args.direct_objectives}' (index {args.direct_objectives_start})")
            print(f"    Using timestamp-based run_id format instead")
    
    # Initialize run data manager for this run (client creates the run_id)
    from utils.data_persistence.run_data_manager import initialize_run_data_manager
    
    run_manager = initialize_run_data_manager(
        run_name=args.run_name,
        first_objective_id=first_objective_id,
        first_objective_desc=first_objective_desc
    )
    run_id = run_manager.run_id
    print(f"📁 Run data directory: {run_manager.get_run_directory()}")
    
    # Generate LLM session_id for consistent logging across processes
    from datetime import datetime
    llm_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["LLM_SESSION_ID"] = llm_session_id
    print(f"📝 LLM session ID: {llm_session_id}")
    
    # Pass run_id to server via environment variable to avoid conflicts
    os.environ["RUN_DATA_ID"] = run_id
    if args.run_name:
        os.environ["RUN_NAME"] = args.run_name
    
    # Save metadata with command line information
    run_manager.save_metadata(
        command_args=vars(args),
        sys_argv=sys.argv,
        additional_info={
            "entry_point": "run.py",
            "mode": "multiprocess_client"
        }
    )
    
    # Get first objective info for consistent run naming
    first_objective_id = None
    first_objective_desc = None
    if args.direct_objectives:
        from agents.objectives import get_first_objective_info
        first_objective_id, first_objective_desc = get_first_objective_info(
            args.direct_objectives, 
            args.direct_objectives_start
        )
        if first_objective_id:
            print(f"🎯 First objective: {first_objective_id} - {first_objective_desc}")
        else:
            print(f"⚠️  Could not retrieve first objective from '{args.direct_objectives}' (index {args.direct_objectives_start})")
            print(f"    Using timestamp-based run_id format instead")
    
    # Initialize run data manager for this run (client creates the run_id)
    from utils.data_persistence.run_data_manager import initialize_run_data_manager
    
    run_manager = initialize_run_data_manager(
        run_name=args.run_name,
        first_objective_id=first_objective_id,
        first_objective_desc=first_objective_desc
    )
    run_id = run_manager.run_id
    print(f"📁 Run data directory: {run_manager.get_run_directory()}")
    
    # Generate LLM session_id for consistent logging across processes
    from datetime import datetime
    llm_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["LLM_SESSION_ID"] = llm_session_id
    print(f"📝 LLM session ID: {llm_session_id}")
    
    # Pass run_id to server via environment variable to avoid conflicts
    os.environ["RUN_DATA_ID"] = run_id
    if args.run_name:
        os.environ["RUN_NAME"] = args.run_name
    
    # Save metadata with command line information
    run_manager.save_metadata(
        command_args=vars(args),
        sys_argv=sys.argv,
        additional_info={
            "entry_point": "run.py",
            "mode": "multiprocess_client"
        }
    )
    
    server_process = None
    frame_server_process = None
    
    try:
        # Restore from backup if requested
        if args.backup_state:
            from utils.data_persistence.backup_manager import restore_cache_from_backup
            from utils.data_persistence.run_data_manager import get_cache_directory
            
            print(f"\n📦 Restoring from backup: {args.backup_state}")
            
            # Restore backup to run-specific cache directory
            success = restore_cache_from_backup(
                backup_file=args.backup_state,
                create_backup_of_current=False  # Don't backup empty cache
            )
            
            if success:
                print(f"✅ Backup restored to: {get_cache_directory()}")
                # Auto-load checkpoint from restored backup
                args.load_checkpoint = True
            else:
                print(f"❌ Failed to restore backup, continuing with fresh state")
        
        if args.clear_memory:
            from utils.data_persistence.run_data_manager import get_cache_path
            import json
            memory_file = get_cache_path("memory.json")
            legacy_file = get_cache_path("knowledge_base.json")
            cleared = False
            for f in (memory_file, legacy_file):
                if f.exists():
                    empty_data = {"next_id": 1, "entries": {}}
                    with open(f, 'w') as fh:
                        json.dump(empty_data, fh, indent=2)
                    print(f"🧹 Cleared memory: {f}")
                    cleared = True
            if not cleared:
                print(f"ℹ️  Memory file does not exist yet: {memory_file}")
        
        # Auto-start server if requested
        if args.agent_auto or args.manual or args.scaffold in SERVER_MANAGED_SCAFFOLDS:
            print("\n📡 Starting server process...")
            server_process = start_server(args)
            
            if not server_process:
                print("❌ Failed to start server, exiting...")
                return 1

            # Single-writer metrics: client should not write to cache
            os.environ["LLM_METRICS_WRITE_ENABLED"] = "false"
            
            # Also start frame server for web visualization
            frame_server_process = start_frame_server(args.port)
        else:
            print("\n📋 Manual server mode - start server separately with:")
            print("   python -m server.app --port", args.port)
            if args.load_state:
                print(f"   (Add --load-state {args.load_state} to server command)")
            print("\n⏳ Waiting 3 seconds for manual server startup...")
            time.sleep(3)
        
        # Display configuration
        print("\n🤖 Agent Configuration:")
        print(f"   Backend: {args.backend}")
        print(f"   Model: {args.model_name}")
        print(f"   Scaffold: {SCAFFOLD_DESCRIPTIONS.get(args.scaffold, args.scaffold)}")
        if args.no_ocr:
            print("   OCR: Disabled")
        if args.record:
            print("   Recording: Enabled")
        
        print(f"🎥 Stream View: http://127.0.0.1:{args.port}/stream")

        # All supported scaffolds are custom benchmark agents
        if args.scaffold in CUSTOM_AGENT_CONFIGS:
            return start_custom_agent(CUSTOM_AGENT_CONFIGS[args.scaffold], args)
        else:
            print(f"❌ Unsupported scaffold: {args.scaffold}")
            print(f"   Supported: {', '.join(CUSTOM_AGENT_CONFIGS.keys())}")
            return 1
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
        return 0
        
    finally:
        # Clean up server processes
        if server_process:
            print("\n📡 Stopping server process...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("   Force killing server...")
                server_process.kill()
        
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
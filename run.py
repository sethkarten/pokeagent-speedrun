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
from server.client import run_multiprocess_client


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
    
    # Pass through server-relevant arguments
    if args.record:
        server_cmd.append("--record")
    
    if args.load_checkpoint:
        # Auto-load checkpoint.state when --load-checkpoint is used
        from utils.run_data_manager import get_cache_path
        checkpoint_state = get_cache_path("checkpoint.state")
        if checkpoint_state.exists():
            server_cmd.extend(["--load-state", str(checkpoint_state)])
            # Set environment variable to enable LLM checkpoint loading
            server_env["LOAD_CHECKPOINT_MODE"] = "true"
            print(f"🔄 Server will load checkpoint: {checkpoint_state}")
            checkpoint_llm = get_cache_path("checkpoint_llm.txt")
            print(f"🔄 LLM metrics will be restored from {checkpoint_llm}")
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


def start_cli_agent(agent_config, args):
    """Generic helper to start any CLI-based agent

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

    agent = agent_class(**agent_kwargs)
    print("✅ Agent created", flush=True)

    return agent.run()


def main():
    """Main entry point for the Pokemon Agent"""
    parser = argparse.ArgumentParser(description="Pokemon Emerald AI Agent")
    
    # Core arguments
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
                       help="VLM backend (openai, gemini, local, openrouter)")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash", 
                       help="Model name to use")
    parser.add_argument("--scaffold", type=str, default="my_cli_agent",
                       choices=["fourmodule", "simple", "react", "claudeplays", "geminiplays", "cli", "my_cli_agent", "autonomous_cli", "vision_only"],
                       help="Agent scaffold: my_cli_agent (default), simple, react, claudeplays, geminiplays, cli (server-only for external CLI agents), my_cli_agent (custom CLI agent), autonomous_cli (autonomous agent with all tools), or vision_only (vision-only agent without map/pathfinding)")
    parser.add_argument("--simple", action="store_true", 
                       help="DEPRECATED: Use --scaffold simple instead")
    
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
                       help="Load a specific direct objective sequence (e.g., 'tutorial_to_rival', 'categorized_full_game')")
    parser.add_argument("--direct-objectives-start", type=int, default=0,
                       help="Start index for story objectives in legacy mode, or story objectives in categorized mode")
    parser.add_argument("--direct-objectives-battling-start", type=int, default=0,
                       help="Start index for battling objectives (only used in categorized mode)")
    parser.add_argument("--clear-knowledge-base", action="store_true",
                       help="Clear the knowledge_base.json file before starting the run")
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
    
    print("=" * 60)
    print("🎮 Pokemon Emerald AI Agent")
    print("=" * 60)
    
    # Get first objective info for consistent run naming
    first_objective_id = None
    first_objective_desc = None
    if args.direct_objectives:
        from agent.direct_objectives import get_first_objective_info
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
    from utils.run_data_manager import initialize_run_data_manager
    
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
            from utils.backup_manager import restore_cache_from_backup
            from utils.run_data_manager import get_cache_directory
            
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
        
        # Clear knowledge base if requested
        if args.clear_knowledge_base:
            from utils.run_data_manager import get_cache_path
            knowledge_base_file = get_cache_path("knowledge_base.json")
            if knowledge_base_file.exists():
                # Clear the file by writing empty JSON structure
                import json
                empty_data = {
                    "next_id": 1,
                    "entries": {}
                }
                with open(knowledge_base_file, 'w') as f:
                    json.dump(empty_data, f, indent=2)
                print(f"🧹 Cleared knowledge base: {knowledge_base_file}")
            else:
                print(f"ℹ️  Knowledge base file does not exist yet: {knowledge_base_file}")
        
        # Auto-start server if requested
        if args.agent_auto or args.manual or args.scaffold in ["cli", "my_cli_agent", "autonomous_cli", "geminiplays", "vision_only"]:
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
        
        # Handle deprecated --simple flag
        if args.simple:
            print("⚠️ --simple is deprecated. Using --scaffold simple")
            args.scaffold = "simple"
        
        # Display configuration
        print("\n🤖 Agent Configuration:")
        print(f"   Backend: {args.backend}")
        print(f"   Model: {args.model_name}")
        scaffold_descriptions = {
            "fourmodule": "Four-module architecture (Perception→Planning→Memory→Action)",
            "simple": "Simple mode (direct frame→action)",
            "react": "ReAct agent (Thought→Action→Observation loop)",
            "claudeplays": "ClaudePlaysPokemon (tool-based with history summarization)",
            "geminiplays": "GeminiPlaysPokemon (hierarchical goals, meta-tools, self-critique)",
            "cli": "Gemini API with MCP tools (native function calling)",
            "my_cli_agent": "My Custom CLI Agent (customized Gemini API with MCP tools)",
            "autonomous_cli": "Autonomous CLI Agent (creates own objectives, all tools enabled)",
            "vision_only": "Vision-Only Agent (no map info, no pathfinding, button sequences)"
        }
        print(f"   Scaffold: {scaffold_descriptions.get(args.scaffold, args.scaffold)}")
        if args.no_ocr:
            print("   OCR: Disabled")
        if args.record:
            print("   Recording: Enabled")
        
        print(f"🎥 Stream View: http://127.0.0.1:{args.port}/stream")

        # Configuration for CLI-based agents
        cli_agent_configs = {
            "cli": {
                "name": "CLI Scaffold Mode - Gemini API with MCP Tools",
                "details": [
                    "Using Gemini API directly (no gemini-cli dependency)",
                    "MCP tools exposed via HTTP endpoints"
                ],
                "module": "agent.cli_agent",
                "class": "CLIAgent",
                "use_backend": False
            },
            "my_cli_agent": {
                "name": "My Custom CLI Agent Mode - Customized Gemini API with MCP Tools",
                "details": [
                    "Using customized Gemini API implementation",
                    "MCP tools exposed via HTTP endpoints"
                ],
                "module": "agent.my_cli_agent",
                "class": "MyCLIAgent",
                "use_backend": True,
                "supports_prompt_optimization": True
            },
            "autonomous_cli": {
                "name": "Autonomous CLI Agent Mode - Creates Own Objectives + All Tools",
                "details": [
                    "Using autonomous Gemini API implementation",
                    "ALL MCP tools enabled (23 tools total)",
                    "Agent creates its own objectives dynamically"
                ],
                "module": "agent.my_cli_agent_autonomous",
                "class": "AutonomousCLIAgent",
                "use_backend": True,
                "supports_prompt_optimization": True
            },
            "vision_only": {
                "name": "Vision-Only Agent Mode - No Map Info, No Pathfinding",
                "details": [
                    "Relies purely on visual input (screenshots)",
                    "No map information or pathfinding assistance",
                    "Navigates using directional buttons only"
                ],
                "module": "agent.vision_only_agent",
                "class": "VisionOnlyAgent",
                "use_backend": True,
                "supports_walkthrough": True,
                "supports_slam": True
            }
        }

        # Check if this is a CLI-based agent
        if args.scaffold in cli_agent_configs:
            return start_cli_agent(cli_agent_configs[args.scaffold], args)
        elif args.scaffold == "geminiplays":
            print("\n🖥️  GeminiPlaysAgent Mode - Native Tools with MCP Integration")
            print("=" * 60)
            print("✅ Server is running")
            print("🤖 Starting GeminiPlaysAgent...")
            print("   Using native function calling with 15 tools")
            print("   MCP tools: press_buttons, navigate_to, get_game_state")
            print("   Native tools: goals, memory, self-critique, meta")
            print("")

            # Import and run GeminiPlaysAgent
            from agent.gemini_plays import GeminiPlaysAgent
            from agent import Agent
            print("📦 GeminiPlaysAgent imported successfully", flush=True)

            print(f"🔧 Creating agent with scaffold=geminiplays", flush=True)
            agent_wrapper = Agent(args)
            agent = agent_wrapper.agent_impl  # Get the actual GeminiPlaysAgent instance

            if not isinstance(agent, GeminiPlaysAgent):
                print(f"❌ Error: Expected GeminiPlaysAgent but got {type(agent)}")
                return 1

            print("✅ Agent created", flush=True)

            max_steps = args.max_steps if hasattr(args, 'max_steps') else None
            return agent.run(max_steps=max_steps)
        else:
            print("\n🚀 Starting client...")
            print("-" * 60)

            # Run the client
            success = run_multiprocess_client(server_port=args.port, args=args)

            return 0 if success else 1
        
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
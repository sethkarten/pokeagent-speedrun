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


def start_server(args):
    """Start the server process with appropriate arguments"""
    # Use the same Python executable that's running this script
    python_exe = sys.executable
    server_cmd = [python_exe, "-m", "server.app", "--port", str(args.port)]
    
    # Pass through server-relevant arguments
    if args.record:
        server_cmd.append("--record")
    
    if args.load_checkpoint:
        # Auto-load checkpoint.state when --load-checkpoint is used
        checkpoint_state = ".pokeagent_cache/checkpoint.state"
        if os.path.exists(checkpoint_state):
            server_cmd.extend(["--load-state", checkpoint_state])
            # Set environment variable to enable LLM checkpoint loading
            os.environ["LOAD_CHECKPOINT_MODE"] = "true"
            print(f"🔄 Server will load checkpoint: {checkpoint_state}")
            print(f"🔄 LLM metrics will be restored from .pokeagent_cache/checkpoint_llm.txt")
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
    
    # Server always runs headless - display handled by client
    
    # Start server as subprocess
    try:
        print(f"📋 Server command: {' '.join(server_cmd)}")
        server_process = subprocess.Popen(
            server_cmd,
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
    
    # Agent configuration
    parser.add_argument("--backend", type=str, default="gemini", 
                       help="VLM backend (openai, gemini, local, openrouter)")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash", 
                       help="Model name to use")
    parser.add_argument("--scaffold", type=str, default="fourmodule",
                       choices=["fourmodule", "simple", "react", "claudeplays", "geminiplays", "cli", "my_cli_agent"],
                       help="Agent scaffold: fourmodule (default), simple, react, claudeplays, geminiplays, cli (server-only for external CLI agents), or my_cli_agent (custom CLI agent)")
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
    parser.add_argument("--no-ocr", action="store_true", 
                       help="Disable OCR dialogue detection")
    parser.add_argument("--direct-objectives", type=str, 
                       help="Load a specific direct objective sequence (e.g., 'tutorial_to_starter')")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎮 Pokemon Emerald AI Agent")
    print("=" * 60)
    
    server_process = None
    frame_server_process = None
    
    try:
        # Auto-start server if requested
        if args.agent_auto or args.manual or args.scaffold == "cli":
            print("\n📡 Starting server process...")
            server_process = start_server(args)
            
            if not server_process:
                print("❌ Failed to start server, exiting...")
                return 1
            
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
            "my_cli_agent": "My Custom CLI Agent (customized Gemini API with MCP tools)"
        }
        print(f"   Scaffold: {scaffold_descriptions.get(args.scaffold, args.scaffold)}")
        if args.no_ocr:
            print("   OCR: Disabled")
        if args.record:
            print("   Recording: Enabled")
        
        print(f"🎥 Stream View: http://127.0.0.1:{args.port}/stream")

        # Check if this is CLI scaffold mode
        if args.scaffold == "cli":
            print("\n🖥️  CLI Scaffold Mode - Gemini API with MCP Tools")
            print("=" * 60)
            print("✅ Server is running")
            print("🤖 Starting CLI agent...")
            print("   Using Gemini API directly (no gemini-cli dependency)")
            print("   MCP tools exposed via HTTP endpoints")
            print("")

            # Import and run CLI agent (native Gemini API)
            from agent.cli_agent import CLIAgent
            print("📦 CLIAgent imported successfully", flush=True)

            print(f"🔧 Creating agent with model={args.model_name}", flush=True)
            agent = CLIAgent(
                server_url=f"http://localhost:{args.port}",
                model=args.model_name,
                max_steps=args.max_steps if hasattr(args, 'max_steps') else None
            )
            print("✅ Agent created", flush=True)

            return agent.run()
        elif args.scaffold == "my_cli_agent":
            print("\n🖥️  My Custom CLI Agent Mode - Customized Gemini API with MCP Tools")
            print("=" * 60)
            print("✅ Server is running")
            print("🤖 Starting My Custom CLI agent...")
            print("   Using customized Gemini API implementation")
            print("   MCP tools exposed via HTTP endpoints")
            print("")

            # Import and run My Custom CLI agent
            from agent.my_cli_agent import MyCLIAgent
            print("📦 MyCLIAgent imported successfully", flush=True)

            print(f"🔧 Creating agent with model={args.model_name}", flush=True)
            agent = MyCLIAgent(
                server_url=f"http://localhost:{args.port}",
                model=args.model_name,
                backend=args.backend,
                max_steps=args.max_steps if hasattr(args, 'max_steps') else None
            )
            print("✅ Agent created", flush=True)

            return agent.run()
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
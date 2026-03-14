"""
Agent modules for Pokemon Emerald speedrunning agent
"""

from utils.agent_infrastructure.vlm_backends import VLM
from .simple.react import ReActAgent, create_react_agent
from .simple.claude_plays import ClaudePlaysAgent, create_claude_plays_agent


class Agent:
    """
    Unified agent interface that encapsulates all agent logic.
    The client just calls agent.step(game_state) and gets back an action.
    """
    
    def __init__(self, args=None):
        """
        Initialize the agent based on configuration.
        
        Args:
            args: Command line arguments with agent configuration
        """
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"
        port = args.port if args and hasattr(args, 'port') else 8000

        # Handle scaffold selection
        if args and hasattr(args, 'scaffold'):
            scaffold = args.scaffold
        else:
            scaffold = "geminiplays"
        
        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent based on scaffold
        self.scaffold = scaffold
        if scaffold == "react":
            # Create ReAct agent
            vlm_client = VLM(backend=backend, model_name=model_name)
            self.agent_impl = create_react_agent(vlm_client=vlm_client, verbose=True)
            print(f"   Scaffold: ReAct (Thought->Action->Observation)")
            
        elif scaffold == "claudeplays":
            # Create ClaudePlaysPokemon agent
            vlm_client = VLM(backend=backend, model_name=model_name)
            self.agent_impl = create_claude_plays_agent(
                vlm_client=vlm_client, 
                max_history=30,
                enable_navigation=True,  # Enable advanced pathfinding navigation
                verbose=True
            )
            print(f"   Scaffold: ClaudePlaysPokemon (tool-based with pathfinding)")
            
        elif scaffold == "geminiplays":
            # Create GeminiPlaysPokemon agent with native tools
            from .simple.gemini_plays import create_gemini_plays_agent, _create_gemini_plays_tools
            server_url = f"http://localhost:{port}"

            # CRITICAL: Create tools first, then create VLM with tools for function calling
            tools = _create_gemini_plays_tools()
            vlm_client = VLM(backend=backend, model_name=model_name, tools=tools)
            print(f"   VLM created with {len(tools)} tools for function calling")

            self.agent_impl = create_gemini_plays_agent(
                vlm_client=vlm_client,
                server_url=server_url,  # MCP server URL
                context_reset_interval=100,  # Reset context every 100 steps
                enable_self_critique=True,
                enable_exploration=True,
                use_mcp_tools=True,  # Enable all 15 tools (3 MCP + 12 native)
                verbose=True
            )
            print(f"   Scaffold: GeminiPlaysPokemon (15 tools: goals, memory, navigation, self-critique)")
            print(f"   Server: {server_url}")
            
        else:
            raise ValueError(
                f"Unsupported scaffold '{scaffold}'. "
                "Use one of: react, claudeplays, geminiplays."
            )
    
    def step(self, game_state):
        """
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - audio: Dict with audio observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and optionally 'reasoning'
        """
        if self.scaffold in ["react", "claudeplays", "geminiplays"]:
            # Delegate to specific agent implementation
            if self.scaffold == "react":
                # ReAct agent expects state dict and screenshot separately
                state = game_state.get('game_state', {})
                screenshot = game_state.get('screenshot', None)
                button = self.agent_impl.step(state, screenshot)
                return {'action': button, 'reasoning': 'ReAct agent decision'}
                
            elif self.scaffold == "claudeplays":
                # ClaudePlaysPokemon agent expects state dict and screenshot separately
                state = game_state.get('game_state', {})
                screenshot = game_state.get('screenshot', None)
                button = self.agent_impl.step(state, screenshot)
                return {'action': button, 'reasoning': 'ClaudePlaysPokemon agent decision'}
                
            elif self.scaffold == "geminiplays":
                # GeminiPlaysPokemon agent expects state dict and screenshot separately
                # Extract state from the structure sent by server/client.py
                state = {
                    'player': game_state.get('player', {}),
                    'game': game_state.get('game', {}),
                    'map': game_state.get('map', {}),
                    'milestones': game_state.get('milestones', {}),
                    'visual': game_state.get('visual', {}),
                    'player_position': game_state.get('player', {}).get('position', {}),
                    'location': game_state.get('map', {}).get('name', 'Unknown'),
                    'team': game_state.get('player', {}).get('party', []),
                    'badges': game_state.get('player', {}).get('badges', 0),
                    'in_battle': game_state.get('game', {}).get('in_battle', False)
                }
                screenshot = game_state.get('frame', None)  # 'frame' not 'screenshot'
                button = self.agent_impl.step(state, screenshot)
                return {'action': button, 'reasoning': 'GeminiPlaysPokemon agent decision'}
                
        raise NotImplementedError(
            f"Unsupported scaffold '{self.scaffold}'. "
            "Use react, claudeplays, or geminiplays instead."
        )


__all__ = [
    'Agent',
    'ReActAgent',
    'create_react_agent',
    'ClaudePlaysAgent',
    'create_claude_plays_agent'
]
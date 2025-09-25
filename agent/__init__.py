"""
Agent modules for Pokemon Emerald speedrunning agent
"""

from utils.vlm import VLM
from .action import action_step
from .memory import memory_step
from .perception import perception_step
from .planning import planning_step
from .simple import SimpleAgent, get_simple_agent, simple_mode_processing_multiprocess, configure_simple_agent_defaults
from .react import ReActAgent, create_react_agent
from .claude_plays import ClaudePlaysAgent, create_claude_plays_agent


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
        
        # Handle scaffold selection (with backward compatibility for --simple)
        if args and hasattr(args, 'scaffold'):
            scaffold = args.scaffold
        elif args and hasattr(args, 'simple') and args.simple:
            scaffold = "simple"
        else:
            scaffold = "fourmodule"
        
        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent based on scaffold
        self.scaffold = scaffold
        if scaffold == "simple":
            # Use global SimpleAgent instance to enable checkpoint persistence
            self.agent_impl = get_simple_agent(self.vlm)
            print(f"   Scaffold: Simple (direct frame->action)")
            
        elif scaffold == "react":
            # Create ReAct agent
            from utils.vlm import VLMClient
            vlm_client = VLMClient(backend=backend, model_name=model_name)
            self.agent_impl = create_react_agent(vlm_client=vlm_client, verbose=True)
            print(f"   Scaffold: ReAct (Thought->Action->Observation)")
            
        elif scaffold == "claudeplays":
            # Create ClaudePlaysPokemon agent
            from utils.vlm import VLMClient
            vlm_client = VLMClient(backend=backend, model_name=model_name)
            self.agent_impl = create_claude_plays_agent(
                vlm_client=vlm_client, 
                max_history=30,
                enable_navigation=True,  # Enable advanced pathfinding navigation
                verbose=True
            )
            print(f"   Scaffold: ClaudePlaysPokemon (tool-based with pathfinding)")
            
        elif scaffold == "geminiplays":
            # Create GeminiPlaysPokemon agent
            from utils.vlm import VLMClient
            from agent.gemini_plays import create_gemini_plays_agent
            vlm_client = VLMClient(backend=backend, model_name=model_name)
            self.agent_impl = create_gemini_plays_agent(
                vlm_client=vlm_client,
                context_reset_interval=100,  # Reset context every 100 turns as per blog
                enable_self_critique=True,
                enable_exploration=True,
                enable_meta_tools=True,
                verbose=True
            )
            print(f"   Scaffold: GeminiPlaysPokemon (hierarchical goals, meta-tools, self-critique)")
            
        else:  # fourmodule (default)
            # Four-module agent context
            self.agent_impl = None  # Will use internal four-module processing
            self.context = {
                'perception_output': None,
                'planning_output': None,
                'memory': []
            }
            print(f"   Scaffold: Four-module (Perception->Planning->Memory->Action)")
    
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
        if self.scaffold in ["simple", "react", "claudeplays", "geminiplays"]:
            # Delegate to specific agent implementation
            if self.scaffold == "simple":
                return self.agent_impl.step(game_state)
                
            elif self.scaffold == "react":
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
                state = game_state.get('game_state', {})
                screenshot = game_state.get('screenshot', None)
                button = self.agent_impl.step(state, screenshot)
                return {'action': button, 'reasoning': 'GeminiPlaysPokemon agent decision'}
                
        else:
            # Four-module processing (default)
            try:
                # 1. Perception - understand what's happening
                perception_output = perception_step(
                    self.vlm, 
                    game_state, 
                    self.context.get('memory', [])
                )
                self.context['perception_output'] = perception_output
                
                # 2. Planning - decide strategy
                planning_output = planning_step(
                    self.vlm, 
                    perception_output, 
                    self.context.get('memory', [])
                )
                self.context['planning_output'] = planning_output
                
                # 3. Memory - update context
                memory_output = memory_step(
                    perception_output, 
                    planning_output, 
                    self.context.get('memory', [])
                )
                self.context['memory'] = memory_output
                
                # 4. Action - choose button press
                action_output = action_step(
                    self.vlm, 
                    game_state, 
                    planning_output,
                    perception_output
                )
                
                return action_output
                
            except Exception as e:
                print(f"❌ Agent error: {e}")
                return None


__all__ = [
    'Agent',
    'action_step',
    'memory_step', 
    'perception_step',
    'planning_step',
    'SimpleAgent',
    'get_simple_agent',
    'simple_mode_processing_multiprocess',
    'configure_simple_agent_defaults',
    'ReActAgent',
    'create_react_agent',
    'ClaudePlaysAgent',
    'create_claude_plays_agent'
]
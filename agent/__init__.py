"""
Agent modules for Pokemon Emerald speedrunning agent
"""

from utils.vlm import VLM
from .action import action_step
from .memory import memory_step
from .perception import perception_step
from .planning import planning_step
from .simple import SimpleAgent, get_simple_agent, simple_mode_processing_multiprocess, configure_simple_agent_defaults


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
        simple_mode = args.simple if args else False
        
        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent mode
        self.simple_mode = simple_mode
        if simple_mode:
            # Use global SimpleAgent instance to enable checkpoint persistence
            self.simple_agent = get_simple_agent(self.vlm)
            print(f"   Mode: Simple (direct frame->action)")
        else:
            # Four-module agent context
            self.context = {
                'perception_output': None,
                'planning_output': None,
                'memory': []
            }
            print(f"   Mode: Four-module architecture")
    
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
        if self.simple_mode:
            # Simple mode - delegate to SimpleAgent
            return self.simple_agent.step(game_state)
        else:
            # Four-module processing
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
    'configure_simple_agent_defaults'
]
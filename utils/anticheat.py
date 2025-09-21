import hashlib
import json
import time
import logging
import numpy as np
from collections import deque

# Set up logging
logger = logging.getLogger(__name__)

# Milestone tracking for progression verification
MILESTONES = [
    'LITTLEROOT_TOWN', 'ROUTE_101', 'OLDALE_TOWN', 'ROUTE_103', 'ROUTE_102', 
    'PETALBURG_CITY', 'ROUTE_104', 'PETALBURG_WOODS', 'RUSTBORO_CITY', 
    'RUSTBORO_GYM', 'DEVON_CORPORATION_3F', 'ROUTE_116', 'RUSTURF_TUNNEL', 
    'MR_BRINEYS_COTTAGE', 'ROUTE_105', 'DEWFORD_TOWN', 'GRANITE_CAVE_STEVEN_ROOM', 
    'ROUTE_109', 'SLATEPORT_CITY', 'SLATEPORT_MUSEUM_1F', 'ROUTE_110', 
    'MAUVILLE_CITY', 'MAUVILLE_GYM'
]

class AntiCheatTracker:
    """
    Tracks anti-cheat metrics and behavioral patterns for Pokemon Emerald AI agent.
    """
    
    def __init__(self):
        self.latest_milestone = None
        self.previous_position = None
        self.decision_times = deque(maxlen=100)  # Track last 100 decision times
        self.invalid_actions = 0
        self.total_actions = 0
        self.exploration_moves = 0  # Moves that deviate from direct paths
        self.backtrack_moves = 0    # Moves that return to previous positions
        self.position_history = deque(maxlen=20)  # Track recent positions
        self.start_time = None  # Will be set when logging is initialized
        
        # Set up submission logging (avoid duplicate handlers)
        self.submission_logger = logging.getLogger('submission')
        self.submission_logger.setLevel(logging.INFO)
        
        # Only add handler if none exists
        if not self.submission_logger.handlers:
            submission_handler = logging.FileHandler('submission.log', mode='a')
            submission_handler.setLevel(logging.INFO)  # Explicitly set handler level
            submission_formatter = logging.Formatter('%(message)s')
            submission_handler.setFormatter(submission_formatter)
            self.submission_logger.addHandler(submission_handler)
            self.submission_logger.propagate = False
    
    def create_state_hash(self, state_data):
        """
        Create a hash of critical game state elements for integrity verification.
        """
        # Extract key elements that shouldn't change unexpectedly
        hash_elements = {
            'player_name': state_data.get('player', {}).get('name', ''),
            'money': state_data.get('player', {}).get('money') or state_data.get('game', {}).get('money', 0),
            'location': state_data.get('player', {}).get('location', ''),
            'coordinates': state_data.get('player', {}).get('coordinates', {}),
            'in_battle': state_data.get('game', {}).get('in_battle', False),
            'step_number': state_data.get('step_number', 0)
        }
        
        # Add party pokemon data
        party_data = state_data.get('player', {}).get('party') or state_data.get('game', {}).get('party')
        party_hash_data = []
        if party_data:
            pokemon_list = []
            if isinstance(party_data, dict) and party_data.get('pokemon'):
                pokemon_list = party_data.get('pokemon', [])
            elif isinstance(party_data, list):
                pokemon_list = party_data
            
            for pokemon in pokemon_list[:6]:
                if pokemon:
                    party_hash_data.append({
                        'species': pokemon.get('species', ''),
                        'level': pokemon.get('level', 0),
                        'max_hp': pokemon.get('max_hp', 0)
                    })
        
        hash_elements['party'] = party_hash_data
        
        # Create deterministic hash
        hash_string = json.dumps(hash_elements, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()[:8]
    
    def analyze_movement_behavior(self, current_pos, previous_pos, action_taken):
        """
        Analyze movement patterns for behavioral fingerprinting.
        """
        if not current_pos or not previous_pos:
            return
        
        current_x = current_pos.get('x')
        current_y = current_pos.get('y')
        prev_x = previous_pos.get('x')
        prev_y = previous_pos.get('y')
        
        if None in [current_x, current_y, prev_x, prev_y]:
            return
        
        # Calculate movement delta
        dx = current_x - prev_x
        dy = current_y - prev_y
        
        # Check if movement matches action
        action_str = str(action_taken).upper()
        expected_movement = False
        
        if 'UP' in action_str and dy == -1 and dx == 0:
            expected_movement = True
        elif 'DOWN' in action_str and dy == 1 and dx == 0:
            expected_movement = True
        elif 'LEFT' in action_str and dx == -1 and dy == 0:
            expected_movement = True
        elif 'RIGHT' in action_str and dx == 1 and dy == 0:
            expected_movement = True
        elif action_str in ['A', 'B', 'START', 'SELECT'] and dx == 0 and dy == 0:
            expected_movement = True  # Non-movement actions
        
        if not expected_movement and (dx != 0 or dy != 0):
            self.invalid_actions += 1
        
        # Check for backtracking (returning to a recent position)
        current_pos_tuple = (current_x, current_y)
        if current_pos_tuple in list(self.position_history)[-10:]:  # Last 10 positions
            self.backtrack_moves += 1
        
        # Add to position history
        self.position_history.append(current_pos_tuple)
        
        # Simple exploration detection (movement without clear direction)
        if abs(dx) + abs(dy) > 1:  # Diagonal or multi-tile movement (unusual)
            self.exploration_moves += 1
    
    def calculate_behavioral_metrics(self):
        """
        Calculate behavioral fingerprinting metrics.
        """
        avg_decision_time = sum(self.decision_times) / len(self.decision_times) if self.decision_times else 0
        error_rate = self.invalid_actions / max(self.total_actions, 1)
        exploration_ratio = self.exploration_moves / max(self.total_actions, 1)
        backtrack_ratio = self.backtrack_moves / max(self.total_actions, 1)
        
        return {
            'avg_decision_time': round(avg_decision_time, 3),
            'error_rate': round(error_rate, 3),
            'exploration_ratio': round(exploration_ratio, 3),
            'backtrack_ratio': round(backtrack_ratio, 3),
            'decision_variance': round(np.var(list(self.decision_times)), 3) if len(self.decision_times) > 1 else 0
        }
    
    def detect_milestone(self, location_name):
        """
        Detect if the current location corresponds to a milestone.
        Maps actual game location names to milestone identifiers.
        """
        if not location_name or location_name == 'Unknown':
            return None
        
        # Normalize location name (remove spaces, convert to uppercase)
        normalized = location_name.replace(' ', '_').upper()
        
        # Direct mapping for locations that match exactly
        if normalized in MILESTONES:
            return normalized
        
        # Special mappings for locations that might have different names
        location_mappings = {
            'LITTLEROOT': 'LITTLEROOT_TOWN',
            'ROUTE101': 'ROUTE_101',
            'ROUTE_101': 'ROUTE_101',
            'OLDALE': 'OLDALE_TOWN',
            'ROUTE103': 'ROUTE_103',
            'ROUTE_103': 'ROUTE_103',
            'ROUTE102': 'ROUTE_102',
            'ROUTE_102': 'ROUTE_102',
            'PETALBURG': 'PETALBURG_CITY',
            'ROUTE104': 'ROUTE_104',
            'ROUTE_104': 'ROUTE_104',
            'PETALBURG_WOODS': 'PETALBURG_WOODS',
            'RUSTBORO': 'RUSTBORO_CITY',
            'RUSTBORO_GYM': 'RUSTBORO_GYM',
            'DEVON_CORP': 'DEVON_CORPORATION_3F',
            'DEVON_CORPORATION': 'DEVON_CORPORATION_3F',
            'ROUTE116': 'ROUTE_116',
            'ROUTE_116': 'ROUTE_116',
            'RUSTURF': 'RUSTURF_TUNNEL',
            'MR_BRINEY': 'MR_BRINEYS_COTTAGE',
            'BRINEYS_COTTAGE': 'MR_BRINEYS_COTTAGE',
            'ROUTE105': 'ROUTE_105',
            'ROUTE_105': 'ROUTE_105',
            'DEWFORD': 'DEWFORD_TOWN',
            'GRANITE_CAVE': 'GRANITE_CAVE_STEVEN_ROOM',
            'ROUTE109': 'ROUTE_109',
            'ROUTE_109': 'ROUTE_109',
            'SLATEPORT': 'SLATEPORT_CITY',
            'SLATEPORT_MUSEUM': 'SLATEPORT_MUSEUM_1F',
            'ROUTE110': 'ROUTE_110',
            'ROUTE_110': 'ROUTE_110',
            'MAUVILLE': 'MAUVILLE_CITY',
            'MAUVILLE_GYM': 'MAUVILLE_GYM'
        }
        
        # Check for partial matches
        for key, milestone in location_mappings.items():
            if key in normalized:
                return milestone
        
        return None
    
    def update_milestone(self, current_location):
        """Update the latest milestone based on current location"""
        detected_milestone = self.detect_milestone(current_location)
        if detected_milestone:
            # Check if this is a new milestone (further in progression)
            if self.latest_milestone is None:
                self.latest_milestone = detected_milestone
                logger.info(f"[MILESTONE] First milestone detected: {self.latest_milestone}")
            else:
                try:
                    current_index = MILESTONES.index(self.latest_milestone)
                    new_index = MILESTONES.index(detected_milestone)
                    
                    # Only update if we've progressed further
                    if new_index > current_index:
                        self.latest_milestone = detected_milestone
                        logger.info(f"[MILESTONE] New milestone reached: {self.latest_milestone}")
                except ValueError:
                    # If milestone not found in list, just log it
                    logger.warning(f"[MILESTONE] Unknown milestone detected: {detected_milestone}")
        
        return self.latest_milestone
    
    def log_submission_data(self, step, state_data, action_taken, decision_time, state_hash, manual_mode=False, milestone_override=None):
        """Log structured data for anticheat verification"""
        # Extract key information
        player_data = state_data.get('player', {})
        game_data = state_data.get('game', {})
        map_info = state_data.get('map', {})
        
        # Position - use the same logic as state formatter
        position = None
        if 'coordinates' in player_data:
            position = player_data['coordinates']
        elif 'position' in player_data and player_data['position']:
            position = player_data['position']
        
        if position and isinstance(position, dict):
            x_val = position.get('x', '?')
            y_val = position.get('y', '?')
            pos_str = f"({x_val},{y_val})"
        else:
            pos_str = "(?,?)"
        
        # Battle state
        battle_state = "BATTLE" if game_data.get('in_battle', False) else "OVERWORLD"
        
        # Party summary - use similar logic to state formatter
        party_data = player_data.get('party') or game_data.get('party')
        party_summary = []
        if party_data:
            pokemon_list = []
            if isinstance(party_data, dict) and party_data.get('pokemon'):
                pokemon_list = party_data.get('pokemon', [])
            elif isinstance(party_data, list):
                pokemon_list = party_data
            
            for pokemon in pokemon_list[:6]:  # Max 6 pokemon
                if pokemon:
                    # Use same field names as get_comprehensive_state for consistency
                    species = pokemon.get('species_name', pokemon.get('species', 'Unknown'))
                    level = pokemon.get('level', '?')
                    hp = pokemon.get('current_hp', '?')
                    max_hp = pokemon.get('max_hp', '?')
                    status = pokemon.get('status', 'OK')
                    party_summary.append(f"{species}(Lv{level},HP:{hp}/{max_hp},{status})")
        
        party_str = "[" + ",".join(party_summary) + "]" if party_summary else "[]"
        
        # Action taken
        action_str = str(action_taken) if action_taken else "None"
        
        # Current map - correctly extract from player.location
        current_map = player_data.get('location', 'Unknown')
        
        # Use provided milestone or update and get latest milestone
        if milestone_override:
            milestone_str = milestone_override
        else:
            milestone = self.update_milestone(current_map)
            milestone_str = milestone if milestone else "NONE"
        
        # Money
        money = player_data.get('money')
        if money is None:
            money = game_data.get('money', 0)
        
        # Analyze movement behavior
        if position and isinstance(position, dict):
            self.analyze_movement_behavior(position, self.previous_position, action_taken)
            self.previous_position = position.copy()
        
        self.total_actions += 1
        self.decision_times.append(decision_time)
        
        # Get behavioral metrics
        behavioral_metrics = self.calculate_behavioral_metrics()
        
        # Calculate current runtime
        current_runtime = 0.0
        runtime_str = "00:00:00"
        if self.start_time is not None:
            current_runtime = time.time() - self.start_time
            hours = int(current_runtime // 3600)
            minutes = int((current_runtime % 3600) // 60)
            seconds = int(current_runtime % 60)
            runtime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Determine mode string
        mode_str = "MANUAL" if manual_mode else "AGENT"
        
        # Log structured entry with enhanced anti-cheat data including runtime and mode
        log_entry = (f"STEP={step} | POS={pos_str} | MAP={current_map} | MILESTONE={milestone_str} | "
                    f"STATE={battle_state} | MONEY=${money} | PARTY={party_str} | ACTION={action_str} | "
                    f"MODE={mode_str} | DECISION_TIME={decision_time:.3f}s | RUNTIME={runtime_str} | "
                    f"STATE_HASH={state_hash} | "
                    f"AVG_TIME={behavioral_metrics['avg_decision_time']}s | "
                    f"ERROR_RATE={behavioral_metrics['error_rate']} | "
                    f"EXPLORE_RATIO={behavioral_metrics['exploration_ratio']} | "
                    f"BACKTRACK_RATIO={behavioral_metrics['backtrack_ratio']} | "
                    f"TIME_VAR={behavioral_metrics['decision_variance']}")
        
        self.submission_logger.info(log_entry)
        
        # Force flush to ensure immediate write
        for handler in self.submission_logger.handlers:
            handler.flush()
    
    def initialize_submission_log(self, model_name):
        """Initialize submission log with header information"""
        self.start_time = time.time()  # Store start time for total runtime calculation
        
        # Clear the file first by writing directly
        with open('submission.log', 'w') as f:
            f.write("")
        
        self.submission_logger.info("=== POKEMON EMERALD AGENT SUBMISSION LOG ===")
        self.submission_logger.info(f"Model: {model_name} | Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.submission_logger.info("Format: STEP | POS | MAP | MILESTONE | STATE | MONEY | PARTY | ACTION | MODE | DECISION_TIME | RUNTIME | STATE_HASH | AVG_TIME | ERROR_RATE | EXPLORE_RATIO | BACKTRACK_RATIO | TIME_VAR")
        self.submission_logger.info("=" * 120) 
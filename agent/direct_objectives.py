"""
Direct Objectives Module

Provides hardcoded, step-by-step objectives for specific game states to simplify
the agent's reasoning loop and ensure more reliable progression through key sequences.

This module contains predefined objective sequences for critical game phases
where the agent needs to follow a specific, well-defined path.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class DirectObjective:
    """Single direct objective with specific guidance"""
    id: str
    description: str
    action_type: str  # "move", "interact", "battle", "wait", "navigate"
    target_location: Optional[str] = None
    target_coords: Optional[tuple] = None
    navigation_hint: Optional[str] = None  # General direction/approach hint
    completion_condition: Optional[str] = None  # How to verify completion
    priority: int = 1  # 1 = highest priority
    completed: bool = False

class DirectObjectiveManager:
    """Manages hardcoded objective sequences for specific game states"""
    
    def __init__(self):
        self.current_sequence: List[DirectObjective] = []
        self.current_index: int = 0
        self.sequence_name: str = ""
        
    def load_birch_to_rival_sequence(self):
        """Load the hardcoded sequence for transitioning from birch state to rival state"""
        self.sequence_name = "birch_to_rival"
        self.current_sequence = [
            DirectObjective(
                id="birch_01_north_littleroot",
                description="Move north from Littleroot Town to Route 101",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Move north from Littleroot Town",
                completion_condition="location_contains_route_101",
                priority=1
            ),
            DirectObjective(
                id="birch_02_north_route101",
                description="Continue north from Route 101 to Oldale Town",
                action_type="navigate", 
                target_location="Oldale Town",
                navigation_hint="Continue moving north from Route 101",
                completion_condition="location_contains_oldale",
                priority=1
            ),
            DirectObjective(
                id="birch_03_north_oldale",
                description="Move north from Oldale Town to Route 103",
                action_type="navigate",
                target_location="Route 103", 
                navigation_hint="Move north from Oldale Town to Route 103",
                completion_condition="location_contains_route_103",
                priority=1
            ),
            DirectObjective(
                id="birch_04_battle_rival",
                description="Battle the rival on Route 103",
                action_type="battle",
                target_location="Route 103",
                navigation_hint="Approach and battle the rival trainer",
                completion_condition="battle_completed",
                priority=1
            ),
            DirectObjective(
                id="birch_05_south_route103",
                description="Move south from Route 103 back to Oldale Town",
                action_type="navigate",
                target_location="Oldale Town",
                navigation_hint="Move south from Route 103 back to Oldale Town",
                completion_condition="location_contains_oldale",
                priority=1
            ),
            DirectObjective(
                id="birch_06_south_oldale",
                description="Move south from Oldale Town to Route 101",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Move south from Oldale Town to Route 101",
                completion_condition="location_contains_route_101",
                priority=1
            ),
            DirectObjective(
                id="birch_07_south_route101",
                description="Move south from Route 101 back to Littleroot Town",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="Move south from Route 101 back to Littleroot Town",
                completion_condition="location_contains_littleroot",
                priority=1
            ),
            DirectObjective(
                id="birch_08_enter_lab",
                description="Enter Professor Birch's lab to receive Pokédex",
                action_type="interact",
                target_location="Professor Birch's Lab",
                navigation_hint="Move to lab entrance and interact to receive Pokédex",
                completion_condition="pokedex_received",
                priority=1
            )
        ]
        self.current_index = 0
        logger.info(f"Loaded birch_to_rival sequence with {len(self.current_sequence)} objectives")
        
    def load_hackathon_route102_to_petalburg_sequence(self):
        """Load the hardcoded sequence for navigating from Route 102 to Petalburg City"""
        self.sequence_name = "hackathon_route102_to_petalburg"
        self.current_sequence = [
            DirectObjective(
                id="hackathon_01_blue_hat_trainer",
                description="Travel west past the boy trainer in a blue hat",
                action_type="navigate",
                target_location="Route 102",
                navigation_hint="Look for a trainer with a blue hat - you need to get past him, and he will try to battle you.",
                completion_condition="passed_blue_hat_trainer",
                priority=1
            ),
            DirectObjective(
                id="hackathon_02_brown_hat_trainer", 
                description="Travel past the boy in a brown hat through the open walkable tiles",
                action_type="navigate",
                target_location="Route 102",
                navigation_hint="Look for a trainer with a brown hat - you need to get past. He will only battle you if you interact with him.",
                completion_condition="passed_brown_hat_trainer",
                priority=1
            ),
            DirectObjective(
                id="hackathon_03_north_grass_trainer",
                description="Travel past the boy in the grass by the ledges with the gap",
                action_type="navigate", 
                target_location="Route 102",
                navigation_hint="Move north through tall grass area where there's a trainer. You will only battle him if you interact with him.",
                completion_condition="passed_grass_trainer",
                priority=1
            ),
            DirectObjective(
                id="hackathon_04_north_ledge_gaps",
                description="Walk through the ledge with the gap",
                action_type="navigate",
                target_location="Route 102", 
                navigation_hint="Move through ledge area with. Use pathfinding to navigate precisely through the ledge sections by requesting a path beyond the ledges.",
                completion_condition="passed_ledge_gaps",
                priority=1
            ),
            DirectObjective(
                id="hackathon_05_west_petalburg",
                description="Immediately travel south-west to Petalburg City",
                action_type="navigate",
                target_location="Petalburg City",
                navigation_hint="Move south-west directly to Petalburg City entrance",
                completion_condition="reached_petalburg_city",
                priority=1
            )
        ]
        self.current_index = 0
        logger.info(f"Loaded hackathon Route 102 to Petalburg sequence with {len(self.current_sequence)} objectives")
        
    def load_tutorial_to_rustboro_city_sequence(self, start_index: int = 0, run_dir: Optional[str] = None):
        """Load the combined sequence from tutorial to rustboro city (10 objectives total).
        
        This is one continuous sequence from the start of the game through the rustboro city.
        """
        self.sequence_name = "tutorial_to_rustboro_city"
        self.current_sequence = [
            # ========== TUTORIAL TO STARTER (Objectives 1-14) ==========
            DirectObjective(
                id="tutorial_01_exit_truck",
                description="Exit the moving truck and enter Littleroot Town",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="Continue walking right to the door (D) to enter Littleroot Town",
                completion_condition="location_contains_littleroot",
                priority=1
            ),
            DirectObjective(
                id="tutorial_02_go_to_bedroom",
                description="Once you exit the truck, you're mom will immediately greet you and take you into your (Brendan) house. Press A to advance through the dialogue (this will take you to 1F of the house) and navigate to the stairs to go upstairs to the player's (your)bedroom",
                action_type="navigate",
                target_location="Player's Bedroom",
                navigation_hint="Walk north towards and through the stairs (S) at (8, 2) to go up to the bedroom. Once you enter the bedroom, your mom will continue to redirect you back to your bedroom until you interact with the clock.",
                completion_condition="player_bedroom_reached",
                priority=1
            ),
            DirectObjective(
                id="tutorial_03_interact_with_clock",
                description="Interact with the clock on the wall to set the time. Interacting with it will trigger a new screen with a clock that you have to navigate through.",
                action_type="interact",
                target_location="Player's Bedroom",
                navigation_hint="Press A on the clock on the wall to set the time. The clock (K) is 2 tiles left of the stairs (S) and on the wall. The clock is at position (5,1) so you must navigate to position (5,2) once you  and face the clock by pressing UP.",
                completion_condition="clock_set",
                priority=1
            ),
            DirectObjective(
                id="tutorial_04_exit_player_house",
                description="Exit the player's house by walking down the stairs (S) and through the door (D)",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="Go downstairs. This will trigger an event with your mom sending you towards the TV. Once this is over, exit through the door (D) to your south-east. No need to re-enter the stairs. Remember to to walk through doors, not interact with them by pressing A.",
                completion_condition="exited_player_house",
                priority=1
            ),
            DirectObjective(
                id="tutorial_05_enter_rival_house",
                description="Enter the rival's house (next door) by immediately navigating to the right and walking through the door (D).",
                action_type="interact",
                target_location="Rival's House",
                navigation_hint="Your rival's house will be to the right of your house and looks identical. Don't overshoot walking to the right, once you observe the DOOR (D), in your movement preview walk through it by walking UP. If you've overshot the door, you have to walk left to re-align yourself with the door. Once you've identified the door's position on the map, navigate to those coordinates",
                completion_condition="rival_house_entered",
                priority=1
            ),
            DirectObjective(
                id="tutorial_06_go_to_rival_bedroom",
                description="Go upstairs to the rival's bedroom. Their mother will immediately greet you. After your interaction with her, walk north immediately to go up the stairs.",
                action_type="navigate",
                target_location="Rival's Bedroom",
                navigation_hint="Use the stairs to go up to the rival's bedroom. The stairs are a straight line north of the entrance (~7/8 tiles north of the entrance door (D)).",
                completion_condition="rival_bedroom_reached",
                priority=1
            ),
            DirectObjective(
                id="tutorial_07_talk_to_rival",
                description="Once in your rival's room, interact with the pokeball next to her bed. This will trigger a conversation with her, after which she will walk over to her computer. ",
                action_type="interact",
                target_location="Rival's Bedroom",
                navigation_hint="Approach and interact with the pokeball (pressing A once facing the pokeball) to the south-east of the entrance (next to the bed), once the conversation is over, your rival will walk over to her computer.",
                completion_condition="rival_conversation_complete",
                priority=1
            ),
            DirectObjective(
                id="tutorial_08_exit_rival_house",
                description="Exit the rival's house by walking through the stairs (S) at (1,1) you entered your rival's room through and through the door (D) you entered her house through",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="The stairs should be next to the top-left corner of the rival's bedroom (1,1). Once you get down the stairs, walk south to the entrance door (D) and then walk through the warp (DOWN) to exit the house.",
                completion_condition="exited_rival_house",
                priority=1
            ),
            DirectObjective(
                id="tutorial_09_north_to_route101",
                description="Move north from Littleroot Town to Route 101",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Move north from Littleroot Town to reach Route 101. The route is straight north in between your house and the rival's house, so you will have to navigate left once you've left the rival's house and then north past both you and your rival's house. Continue to go north through the passage that leads to route 101.",
                completion_condition="location_contains_route_101",
                priority=1
            ),
            DirectObjective(
                id="tutorial_10_find_and_approach_birch",
                description="Find Professor Birch on Route 101 and interact with the bag to pick your starter. Walk a few steps into route 101 to trigger the event with Professor Birch. Then approach the bag on the ground by the ledge (at position 7, 14) and interact with it to pick your starter (treeko, leftmost option). Once you pick treeko, it will trigger a battle with a zigzagoon. Make sure you see treeko in the visual frame before pressing A.",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Walk into Route 101 to find Professor Birch (the event should trigger automatically). The bag will be on the ground to your left, in between you and professor birch at position (7, 14). Face the bag and press A to interact with it. Make sure you are facing the bag in the visual frame before pressing A.",
                completion_condition="birch_encounter_triggered",
                priority=1
            ),
            DirectObjective(
                id="tutorial_11_select_treeko_and_battle_zigzagoon",
                description="Select treeko as your starter and Battle the zigzagoon.",
                action_type="battle",
                target_location="Route 101",
                navigation_hint="Navigate the pokemon selection screen using left and select treeko as your starter. Make sure you see treeko in the visual frame before pressing A. Battle the zigzagoon by selecting a damaging move and pressing A to attack",
                completion_condition="zigzagoon_battle_complete",
                priority=1
            ),
            DirectObjective(
                id="tutorial_12_interact_with_professor_birch",
                description="After battling the zigzagoon, professor birch will interact with you. Advance through the dialogue by pressing A to continue. After this interaction, he will transport you to the lab",
                action_type="interact",
                target_location="Route 101",
                navigation_hint="N/A",
                completion_condition="professor_birch_interaction_complete",
                priority=1
            ),
            DirectObjective(
                id="tutorial_13_talk_to_professor_birch_in_lab",
                description="Talk to professor birch in the lab to receive the pokedex. Advance through the dialogue by pressing A to continue.",
                action_type="navigate",
                target_location="Professor Birch's Lab",
                navigation_hint="Talk to professor birch.",
                completion_condition="dialogue_complete",
                priority=1
            ),
            DirectObjective(
                id="tutorial_14_exit_professor_birch_lab",
                description="Once dialogue is complete, exit the lab by walking immediately south through the door (D) to littleroot town",
                action_type="interact",
                target_location="Professor Birch's Lab",
                navigation_hint="Walk south through the door (D) to littleroot town. For reference, the lab is to the south of your mom's house.",
                completion_condition="exited_professor_birch_lab",
                priority=1
            ),
            # ========== BIRCH_2 TO RIVAL (Objectives 15-20) ==========
            DirectObjective(
                id="birch_2_01_north_route101",
                description="Travel north to Route 101",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Move north from Littleroot Town to reach Route 101",
                completion_condition="location_contains_route_101",
                priority=1
            ),
            DirectObjective(
                id="birch_2_02_route101_to_oldale",
                description="Navigate route 101 to Oldale Town (to the north)",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Use pathfinding to navigate precisely around obstacles so that you can continue to make progress northwards.",
                completion_condition="location_contains_oldale",
                priority=1
            ),
            DirectObjective(
                id="birch_2_03_oldale_to_route103",
                description="Travel north through Oldale Town to Route 103",
                action_type="navigate",
                target_location="Route 103",
                navigation_hint="Move north through Oldale Town to reach Route 103 using navigate_to() for fast and efficient navigation.",
                completion_condition="location_contains_route_103",
                priority=1
            ),
            DirectObjective(
                id="birch_2_04_route103_rival",
                description="Travel north through Route 103 to find and interact with your rival.",
                action_type="navigate",
                target_location="Route 103",
                navigation_hint="Use navigate_to() to efficiently navigate towards your rival's position on the map.",
                completion_condition="passed_route103_first_grass",
                priority=1
            ),
            DirectObjective(
                id="birch_2_05_battle_rival",
                description="Interact with rival and battle them",
                action_type="battle",
                target_location="Route 103",
                navigation_hint="Approach the rival trainer and interact with them to start the battle. Use your starter Pokemon and damaging moves to win the battle.",
                completion_condition="battle_completed",
                priority=1
            ),
            DirectObjective(
                id="birch_2_06_south_to_professor_birch_lab",
                description="Travel back south to Professor Birch's lab to receive the pokedex",
                action_type="navigate",
                target_location="Professor Birch's Lab",
                navigation_hint="Travel south through oldale town, route 101, and littleroot town to reach Professor Birch's lab",
                completion_condition="received_pokedex",
                priority=1
            ),

            # ========== Rival to Petalburg (Objectives 21-24) ==========
            DirectObjective(
                id="professor_birch_to_route_102",
                description="Travel to route 102",
                action_type="navigate",
                target_location="Route 102",
                navigation_hint="Travel to route 101 -> oldale town -> route 102.",
                completion_condition="location_contains_route_102",
                priority=1
            ),
            DirectObjective(
                id="route_102_to_petalburg",
                description="Travel to petalburg city. You may have to encounter and battle trainers along the way",
                action_type="navigate",
                target_location="Petalburg City",
                navigation_hint="Travel to route 102 -> petalburg city",
                completion_condition="reached_petalburg_city",
                priority=1
            ),
            DirectObjective(
                id="petalburg_city_to_dad_first_meeting",
                description="Travel to dad's first meeting with you",
                action_type="navigate",
                target_location="Petalburg City Gym",
                navigation_hint="Travel to petalburg city -> petalburg city gym",
                completion_condition="reached_dad_first_meeting",
                priority=1
            ),
            DirectObjective(
                id="help_wally_catch_ralts",
                description="Help Wally catch a Ralts",
                action_type="battle",
                target_location="Petalburg City",
                navigation_hint="After receiving pokeballs from your dad, Wally will take you to go and catch a Ralts. You will need to use a pokeball from his bag to catch the Ralts.",
                completion_condition="caught_ralts",
                priority=1
            ),

            # ========== Petalburg to Rustboro City (Objectives 25-27) ==========
            DirectObjective(
                id="petalbug_to_route_104",
                description="Travel to route 104",
                action_type="navigate",
                target_location="Route 104",
                navigation_hint="Travel from petalburg city -> route 104",
                completion_condition="reached_route_104",
                priority=1
            ),
            DirectObjective(
                id="route_104_to_petalburg_woods",
                description="Travel north to petalburg woods",
                action_type="navigate",
                target_location="Petalburg Woods",
                navigation_hint="Travel from route 104 to -> petalburg woods by requesting a path to the leftmost set of warps (S), closest to you. (note, if you request a path and fail to make meaningful progress, try to request a path to a different warp that can get you to the same destination)",
                completion_condition="reached_petalburg_woods",
                priority=1
            ),
            # DirectObjective(
            #     id="petalburg_woods_to_route_104",
            #     description="Travel to north to the final section of route 104 ",
            #     action_type="navigate",
            #     target_location="Route 104",
            #     navigation_hint="Travel north from petalburg woods to -> route 104 by using the navigate_to() tool. Note (you will need to navigate to the north most warp to reach the northern section of route 104",
            #     completion_condition="reached_route_104",
            #     priority=1
            # ),
            DirectObjective(
                id="route_104_north_to_rustboro_city",
                description="Travel to rustboro city",
                action_type="navigate",
                target_location="Rustboro City",
                navigation_hint="Travel from route 104 north -> rustboro city",
                completion_condition="reached_rustboro_city",
                priority=1
            ),

            # ========== Rustboro City to Rustboro Gym (Objectives 28-36) ==========
            DirectObjective(
                id="rustboro_pokemon_center",
                description="Visit the pokemon center in rustboro city",
                action_type="navigate",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Travel from rustboro city -> rustboro city pokemon center",
                completion_condition="reached_rustboro_city_pokemon_center",
                priority=1
            ),
            DirectObjective(
                id="heal_pokemon_at_rustboro_pokemon_center",
                description="Heal your pokemon at the pokemon center in rustboro city",
                action_type="interact",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Interact with the nurse in the rustboro city to heal your pokemon",
                completion_condition="healed_pokemon_at_rustboro_pokemon_center",
                priority=1
            ),
            DirectObjective(
                id="catch_first_pokemon_in_route_104",
                description="Keep walking around the grass until you encounter and catch a pokemon. (once the opposing pokemon has < 50% health, use a pokeball). CRITICAL: The following are valid pokemon that can be caught in route 104: marill, poochyena, taillow, wingull. Any other pokemon can simply be defeated!",
                action_type="navigate",
                target_location="Route 104",
                navigation_hint="Navigate back to route 104, enter the grass patch in the north west corner of the map, and keep walking around the grass until you encounter and catch a pokemon. Once in the bag, navigate the PokeBalls menu carefully via (LEFT, RIGHT) and then carefully select a pokeball to use.",
                completion_condition="caught_first_pokemon",
                priority=1
            ),
            DirectObjective(
                id="catch_second_pokemon_in_route_104",
                description="Keep walking around the grass until you encounter and catch a pokemon. (once the opposing pokemon has < 50% health, use a pokeball). CRITICAL: Dont catch wurmple or pokemon already in your party!",
                action_type="navigate",
                target_location="Route 104",
                navigation_hint="Navigate back to route 104, enter the grass patch in the north west corner of the map, and keep walking around the grass until you encounter and catch a pokemon. Once in the bag, navigate the PokeBalls menu carefully via (LEFT, RIGHT) and then carefully select a pokeball to use.",
                completion_condition="caught_second_pokemon",
                priority=1
            ),
            DirectObjective(
                id="rustboro_pokemon_center_2",
                description="Visit the pokemon center in rustboro city again",
                action_type="navigate",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Travel from route 104 -> rustboro city -> rustboro city pokemon center",
                completion_condition="reached_rustboro_city_pokemon_center",
                priority=1
            ),
            DirectObjective(
                id="heal_pokemon_at_rustboro_pokemon_center_2",
                description="Heal your pokemon at the pokemon center in rustboro city again",
                action_type="interact",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Interact with the nurse in the rustboro city to heal your pokemon",
                completion_condition="healed_pokemon_at_rustboro_pokemon_center",
                priority=1
            ),
            DirectObjective(
                id="enter_rustboro_gym",
                description="Enter the rustboro gym",
                action_type="navigate",
                target_location="Rustboro City Gym",
                navigation_hint="Travel from rustboro city -> rustboro city gym",
                completion_condition="entered_rustboro_gym",
                priority=1
            ),
            DirectObjective(
                id="navigate_to_rustboro_gym_leader",
                description="Navigate to the rustboro gym leader. You will battle pokemon trainers along the way!",
                action_type="navigate",
                target_location="Rustboro City Gym",
                navigation_hint="The gym leader (roxanne) is located in the north of the gym. Use navigate_to() to efficiently navigate towards the gym leader. If navigate_to() is failing, manually via PRESS_BUTTON a few steps away and try to pathfind from a different location.",
                completion_condition="reached_rustboro_gym_leader",
                priority=1
            ),
            DirectObjective(
                id="battle_rustboro_gym_leader",
                description="Battle the rustboro gym leader. Prioritize using supereffective moves. If a pokemon faints make sure to use (LEFT/RIGHT/UP/DOWN) in the pokemon selection screen before pressing A to carefully select the next pokemon. Make sure the pokemon you are selecting is highligted with a red outline (this is how you know the pokemon is selected) before pressing A!",
                action_type="battle",
                target_location="Rustboro City Gym",
                navigation_hint="Battle the rustboro gym leader. If you lose this battle, renavigate back to rustboro gym -> roxanne and try again.",
                completion_condition="roxanne_defeated_and_received_badge",
                priority=1
            ),
        ]
        self.current_index = min(start_index, len(self.current_sequence))
        # Mark all objectives before start_index as completed
        initial_completed = []
        for i in range(start_index):
            if i < len(self.current_sequence):
                obj = self.current_sequence[i]
                obj.completed = True
                obj.completed_at = datetime.now()  # Mark as completed at load time
                initial_completed.append({
                    "id": obj.id,
                    "description": obj.description,
                    "target_location": obj.target_location,
                    "action_type": obj.action_type,
                    "completed_at": obj.completed_at.isoformat(),
                    "completed_at_load": True  # Flag to indicate these were pre-completed
                })
        
        # Save initial completed objectives to run directory if provided and start_index > 0
        if run_dir and start_index > 0 and initial_completed:
            try:
                filename = os.path.join(run_dir, "completed_objectives.json")
                initial_data = {
                    "sequence_name": self.sequence_name,
                    "completed_at": datetime.now().isoformat(),
                    "completed_objectives": initial_completed,
                    "total_objectives_completed": len(initial_completed),
                    "total_objectives": len(self.current_sequence),
                    "start_index": start_index,
                    "note": f"Initial completed objectives (loaded at index {start_index})"
                }
                history = {"sequences": [initial_data], "last_updated": datetime.now().isoformat()}
                with open(filename, 'w') as f:
                    json.dump(history, f, indent=2)
                logger.info(f"💾 Saved {len(initial_completed)} initial completed objectives to {filename}")
            except Exception as e:
                logger.warning(f"Failed to save initial completed objectives: {e}")
        
        logger.info(f"Loaded tutorial_to_rival sequence with {len(self.current_sequence)} objectives, starting at index {self.current_index} ({len(initial_completed)} pre-completed)")
        
    def get_current_objective(self) -> Optional[DirectObjective]:
        """Get the current objective in the sequence"""
        if self.current_index < len(self.current_sequence):
            return self.current_sequence[self.current_index]
        return None
        
    def get_current_objective_guidance(self, game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get guidance for the current objective instead of specific actions"""
        current_obj = self.get_current_objective()
        if not current_obj:
            return None
            
        # Check if current objective is completed
        if self._is_objective_completed(current_obj, game_state):
            self._mark_objective_completed(current_obj)
            self.current_index += 1
            logger.info(f"Completed objective: {current_obj.description}")
            
            # Get next objective
            current_obj = self.get_current_objective()
            if not current_obj:
                logger.info("All objectives in sequence completed!")
                return None
                
        # Return guidance for current objective
        return {
            "id": current_obj.id,
            "description": current_obj.description,
            "action_type": current_obj.action_type,
            "target_location": current_obj.target_location,
            "target_coords": current_obj.target_coords,
            "navigation_hint": current_obj.navigation_hint,
            "completion_condition": current_obj.completion_condition
        }
        
    def _is_objective_completed(self, objective: DirectObjective, game_state: Dict[str, Any]) -> bool:
        """Check if an objective is completed based on game state"""
        try:
            location = game_state.get("player", {}).get("location", "").upper()
            
            if objective.completion_condition == "location_contains_route_101":
                return "ROUTE 101" in location or "ROUTE_101" in location
            elif objective.completion_condition == "location_contains_oldale":
                return "OLDALE" in location
            elif objective.completion_condition == "location_contains_route_103":
                return "ROUTE 103" in location or "ROUTE_103" in location
            elif objective.completion_condition == "location_contains_littleroot":
                # More specific check - should be in Littleroot Town but not in a house
                return "LITTLEROOT" in location and "HOUSE" not in location
            elif objective.completion_condition == "player_house_entered":
                # Check if we're in the player's house (could be labeled as Brendan's house due to emulator bug)
                return "HOUSE" in location and "LITTLEROOT" in location
            elif objective.completion_condition == "battle_completed":
                # Check if we're no longer in battle
                return not game_state.get("game", {}).get("is_in_battle", False)
            elif objective.completion_condition == "pokedex_received":
                # Check if we have the Pokédex milestone
                return game_state.get("milestones", {}).get("RECEIVED_POKEDEX", {}).get("completed", False)
            elif objective.completion_condition == "passed_blue_hat_trainer":
                # Let LLM determine completion based on context
                # This will be handled by LLM completion commands
                return False
            elif objective.completion_condition == "passed_brown_hat_trainer":
                # Let LLM determine completion based on context
                return False
            elif objective.completion_condition == "passed_grass_trainer":
                # Let LLM determine completion based on context
                return False
            elif objective.completion_condition == "passed_ledge_gaps":
                # Let LLM determine completion based on context
                return False
            elif objective.completion_condition == "reached_petalburg_city":
                # Check if we've reached Petalburg City (this one is reliable)
                return "PETALBURG" in location
            elif objective.completion_condition == "passed_route101_ledge_grass":
                # Let LLM determine completion based on context
                return False
            elif objective.completion_condition == "passed_route103_first_grass":
                # Let LLM determine completion based on context
                return False
            elif objective.completion_condition == "passed_route103_second_grass":
                # Let LLM determine completion based on context
                return False
            elif objective.completion_condition == "dynamic_objectives_created":
                # This is marked complete when create_direct_objectives is successfully called
                # The completion is handled in the create_direct_objectives endpoint
                return False  # Always return False here - completion is handled explicitly by the endpoint
                
        except Exception as e:
            logger.warning(f"Error checking objective completion: {e}")
            
        return False
        
    def _mark_objective_completed(self, objective: DirectObjective):
        """Mark an objective as completed"""
        objective.completed = True
        objective.completed_at = datetime.now()
    
    def add_dynamic_objectives(self, objectives_data: List[Dict[str, Any]]):
        """Add dynamically created objectives to the current sequence
        
        Args:
            objectives_data: List of dicts with objective properties (id, description, action_type, etc.)
        """
        for obj_data in objectives_data:
            obj = DirectObjective(
                id=obj_data.get("id", f"dynamic_{len(self.current_sequence) + 1}"),
                description=obj_data["description"],
                action_type=obj_data.get("action_type", "navigate"),
                target_location=obj_data.get("target_location"),
                target_coords=obj_data.get("target_coords"),
                navigation_hint=obj_data.get("navigation_hint"),
                completion_condition=obj_data.get("completion_condition"),
                priority=1
            )
            self.current_sequence.append(obj)
        logger.info(f"Added {len(objectives_data)} dynamic objectives to sequence")
    
    def save_completed_objectives(self, run_dir: Optional[str] = None):
        """Save completed objectives history to file in timestamped run directory
        
        Args:
            run_dir: Optional run directory path. If None, uses timestamped directory.
        """
        if not run_dir:
            # Create timestamped directory similar to video recordings
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(".pokeagent_cache", f"run_{timestamp}")
        
        os.makedirs(run_dir, exist_ok=True)
        
        filename = os.path.join(run_dir, "completed_objectives.json")
        
        completed_data = {
            "sequence_name": self.sequence_name,
            "completed_at": datetime.now().isoformat(),
            "completed_objectives": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "target_location": obj.target_location,
                    "action_type": obj.action_type,
                    "completed_at": obj.completed_at.isoformat() if hasattr(obj, 'completed_at') and obj.completed_at else None
                }
                for obj in self.current_sequence if obj.completed
            ],
            "total_objectives_completed": sum(1 for obj in self.current_sequence if obj.completed),
            "total_objectives": len(self.current_sequence)
        }
        
        # Load existing history and append (preserving initial completed objectives if present)
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                history = json.load(f)
            # Check if this sequence name already exists (from initial load)
            existing_sequences = history.get("sequences", [])
            # If there's already a sequence with the same name from initial load, we'll append to it
            # Otherwise, just append the new completion data
        else:
            history = {"sequences": []}
        
        # Append the new completion data (this could be final completion or dynamic objectives)
        history["sequences"].append(completed_data)
        history["last_updated"] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Saved completed objectives to {filename}")
        return filename
        
    def get_sequence_status(self) -> Dict[str, Any]:
        """Get current status of the objective sequence"""
        return {
            "sequence_name": self.sequence_name,
            "total_objectives": len(self.current_sequence),
            "current_index": self.current_index,
            "completed_count": sum(1 for obj in self.current_sequence if obj.completed),
            "current_objective": self.get_current_objective().description if self.get_current_objective() else None,
            "is_complete": self.current_index >= len(self.current_sequence)
        }
        
    def reset_sequence(self):
        """Reset the current sequence"""
        self.current_sequence = []
        self.current_index = 0
        self.sequence_name = ""
        
    def is_sequence_active(self) -> bool:
        """Check if a sequence is currently active"""
        return len(self.current_sequence) > 0 and self.current_index < len(self.current_sequence)
    
    def get_objective_context(self, game_state: Dict[str, Any]) -> str:
        """Get previous objective context for better agent understanding (NEXT removed to avoid confusion)"""
        if not self.is_sequence_active():
            return ""
        
        context_parts = []
        
        # Previous objective only (removed NEXT to avoid confusing the agent)
        if self.current_index > 0:
            prev_obj = self.current_sequence[self.current_index - 1]
            status = "✅" if prev_obj.completed else "❌"
            context_parts.append(f"⏮️  PREVIOUS: {prev_obj.description} {status}")
        
        # Skip current objective in context - it's displayed separately
        # NEXT objective removed - it was confusing the agent
        
        return "\n".join(context_parts)



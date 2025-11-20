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
        """Load the hardcoded sequence for transitioning from birch state to rival state.
        
        Note: This sequence is primarily used by my_simple.py agent, not the CLI agent.
        The CLI agent uses load_tutorial_to_rustboro_city_sequence() instead.
        """
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
        """Load the hardcoded sequence for navigating from Route 102 to Petalburg City.
        
        Note: This sequence is primarily used by my_simple.py agent, not the CLI agent.
        The CLI agent uses load_tutorial_to_rustboro_city_sequence() instead.
        """
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
            DirectObjective( # 0
                id="tutorial_01_exit_truck",
                description="Exit the moving truck and enter Littleroot Town",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="Continue walking right to the door (D) to enter Littleroot Town",
                completion_condition="location_contains_littleroot",
                priority=1
            ),
            DirectObjective( # 1
                id="tutorial_02_go_to_bedroom",
                description="Once you exit the truck, you're mom will immediately greet you and take you into your (Brendan) house. Press A to advance through the dialogue (this will take you to 1F of the house) and navigate to the stairs to go upstairs to the player's (your)bedroom",
                action_type="navigate",
                target_location="Player's Bedroom",
                navigation_hint="Walk north towards and through the stairs (S) at (8, 2) to go up to the bedroom. Once you enter the bedroom, your mom will continue to redirect you back to your bedroom until you interact with the clock.",
                completion_condition="player_bedroom_reached",
                priority=1
            ),
            DirectObjective( # 2
                id="tutorial_03_interact_with_clock",
                description="Interact with the clock on the wall to set the time. Interacting with it will trigger a new screen with a clock that you have to navigate through.",
                action_type="interact",
                target_location="Player's Bedroom",
                navigation_hint="Press Up and A on the clock on the wall to set the time. The clock (K) is 2 tiles left of the stairs (S) and on the wall. The clock is at position (5,1) so you must navigate to position (5,2) and once you face the clock by pressing UP and A.",
                completion_condition="clock_set",
                priority=1
            ),
            DirectObjective( # 3
                id="tutorial_04_exit_player_house",
                description="Exit the player's house by walking down the stairs (S) and through the door (D)",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="Go downstairs. This will trigger an event with your mom sending you towards the TV. Once this is over, exit through the door (D) to your south-east. No need to re-enter the stairs. Remember to to walk through doors, not interact with them by pressing A.",
                completion_condition="exited_player_house",
                priority=1
            ),
            DirectObjective( # 4
                id="tutorial_05_enter_rival_house",
                description="Enter the rival's house (next door) by immediately navigating to the right and walking through the door (D).",
                action_type="interact",
                target_location="Rival's House",
                navigation_hint="Your rival's house will be to the right of your house and looks identical. Don't overshoot walking to the right, once you observe the DOOR (D), in your movement preview walk through it by walking UP. If you've overshot the door, you have to walk left to re-align yourself with the door. Once you've identified the door's position on the map, navigate to those coordinates",
                completion_condition="rival_house_entered",
                priority=1
            ),
            DirectObjective( # 5
                id="tutorial_06_go_to_rival_bedroom",
                description="Go upstairs to the rival's bedroom. Their mother will immediately greet you. After your interaction with her, walk north immediately to go up the stairs.",
                action_type="navigate",
                target_location="Rival's Bedroom",
                navigation_hint="Use the stairs to go up to the rival's bedroom. The stairs are a straight line north of the entrance (~7/8 tiles north of the entrance door (D)).",
                completion_condition="rival_bedroom_reached",
                priority=1
            ),
            DirectObjective( # 6
                id="tutorial_07_talk_to_rival",
                description="Once in your rival's room, interact with the pokeball next to her bed. This will trigger a conversation with her, after which she will walk over to her computer. ",
                action_type="interact",
                target_location="Rival's Bedroom",
                navigation_hint="Approach and interact with the pokeball (pressing A once facing the pokeball) to the south-east of the entrance (next to the bed), once the conversation is over, your rival will walk over to her computer.",
                completion_condition="rival_conversation_complete",
                priority=1
            ),
            DirectObjective( # 7
                id="tutorial_08_exit_rival_house",
                description="Exit the rival's house by walking through the stairs (S) at (1,1) you entered your rival's room through and through the door (D) you entered her house through",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="The stairs should be next to the top-left corner of the rival's bedroom (1,1). Once you get down the stairs, walk south to the entrance door (D) and then walk through the warp (DOWN) to exit the house.",
                completion_condition="exited_rival_house",
                priority=1
            ),
            DirectObjective( # 8
                id="tutorial_09_north_to_route101",
                description="Move north from Littleroot Town to Route 101",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Move north from Littleroot Town to reach Route 101. The route is straight north in between your house and the rival's house, so you will have to navigate left once you've left the rival's house and then north past both you and your rival's house. Continue to go north through the passage that leads to route 101.",
                completion_condition="location_contains_route_101",
                priority=1
            ),
            DirectObjective( # 9
                id="tutorial_10_find_and_approach_birch",
                description="Find Professor Birch on Route 101 and interact with the bag to pick your starter. Walk a few steps into route 101 to trigger the event with Professor Birch. Then approach the bag on the ground by the ledge (at position 7, 14) and interact with it to pick your starter (treeko, leftmost option). Once you pick treeko, it will trigger a battle with a zigzagoon. Make sure you see treeko in the visual frame before pressing A.",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Walk into Route 101 to find Professor Birch (the event should trigger automatically). The bag will be on the ground to your left, in between you and professor birch at position (7, 14). Face the bag and press A to interact with it. Make sure you are facing the bag in the visual frame before pressing A.",
                completion_condition="birch_encounter_triggered",
                priority=1
            ),
            DirectObjective( # 10
                id="tutorial_11_select_treeko_and_battle_zigzagoon",
                description="Select treeko as your starter and Battle the zigzagoon.",
                action_type="battle",
                target_location="Route 101",
                navigation_hint="Navigate the pokemon selection screen using left and select treeko as your starter. Make sure you see treeko in the visual frame before pressing A. Battle the zigzagoon by selecting a damaging move and pressing A to attack",
                completion_condition="zigzagoon_battle_complete",
                priority=1
            ),
            DirectObjective( # 11
                id="tutorial_12_interact_with_professor_birch",
                description="After battling the zigzagoon, professor birch will interact with you. Advance through the dialogue by pressing A to continue. After this interaction, he will transport you to the lab",
                action_type="interact",
                target_location="Route 101",
                navigation_hint="N/A",
                completion_condition="professor_birch_interaction_complete",
                priority=1
            ),
            DirectObjective(  # 12
                id="tutorial_13_talk_to_professor_birch_in_lab",
                description="Talk to professor birch in the lab to receive the pokedex. Advance through the dialogue by pressing A to continue.",
                action_type="navigate",
                target_location="Professor Birch's Lab",
                navigation_hint="Talk to professor birch.",
                completion_condition="dialogue_complete",
                priority=1
            ),
            DirectObjective( # 13
                id="tutorial_14_exit_professor_birch_lab",
                description="Once dialogue is complete, exit the lab by walking immediately south through the door (D) to littleroot town",
                action_type="interact",
                target_location="Professor Birch's Lab",
                navigation_hint="Walk south through the door (D) to littleroot town. For reference, the lab is to the south of your mom's house.",
                completion_condition="exited_professor_birch_lab",
                priority=1
            ),
            # ========== BIRCH_2 TO RIVAL (Objectives 15-20) ==========
            DirectObjective( # 14
                id="birch_2_01_north_route101",
                description="Travel north to Route 101",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Move north from Littleroot Town to reach Route 101",
                completion_condition="location_contains_route_101",
                priority=1
            ),
            DirectObjective( # 15
                id="birch_2_02_route101_to_oldale",
                description="Navigate route 101 to Oldale Town (to the north)",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Use pathfinding to navigate precisely around obstacles so that you can continue to make progress northwards.",
                completion_condition="location_contains_oldale",
                priority=1
            ),
            DirectObjective( # 16
                id="birch_2_02b_heal_at_pokecenter_oldale",
                description="Find and enter the Pokemon Center in Oldale Town to heal your Pokemon",
                action_type="interact",
                target_location="Oldale Town",
                navigation_hint="The Pokemon Center has a red roof and is marked with a red 'P' symbol. Navigate to the building and walk through the door. Inside, talk to Nurse Joy (the NPC behind the counter) and press A to heal your Pokemon. Wait for the healing animation to complete.",
                completion_condition="pokemon_healed_at_center",
                priority=1
            ),
            DirectObjective( # 17
                id="birch_2_03_oldale_to_route103",
                description="Travel north through Oldale Town to Route 103",
                action_type="navigate",
                target_location="Route 103",
                navigation_hint="Move north through Oldale Town to reach Route 103 using navigate_to() for fast and efficient navigation.",
                completion_condition="location_contains_route_103",
                priority=1
            ),
            DirectObjective( # 18
                id="birch_2_04_route103_rival",
                description="Travel north through Route 103 to find and interact with your rival.",
                action_type="navigate",
                target_location="Route 103",
                navigation_hint="Use navigate_to() to efficiently navigate towards your rival's position on the map.",
                completion_condition="passed_route103_first_grass",
                priority=1
            ),
            DirectObjective( # 19
                id="birch_2_05_battle_rival",
                description="Interact with rival and battle them",
                action_type="battle",
                target_location="Route 103",
                navigation_hint="Approach the rival trainer and interact with them to start the battle. Use your starter Pokemon and damaging moves to win the battle.",
                completion_condition="battle_completed",
                priority=1
            ),
            DirectObjective( # 20
                id="birch_2_06_south_to_professor_birch_lab",
                description="Travel back south to Professor Birch's lab to receive the pokedex",
                action_type="navigate",
                target_location="Professor Birch's Lab",
                navigation_hint="Travel south through oldale town, route 101, and littleroot town to reach Professor Birch's lab",
                completion_condition="received_pokedex",
                priority=1
            ),

            # ========== Rival to Petalburg (Objectives 21-24) ==========
            DirectObjective( # 21
                id="professor_birch_to_route_102",
                description="Travel to route 102",
                action_type="navigate",
                target_location="Route 102",
                navigation_hint="Travel to route 101 -> oldale town -> route 102.",
                completion_condition="location_contains_route_102",
                priority=1
            ),
            DirectObjective( # 22
                id="route_102_to_petalburg",
                description="Travel to petalburg city. You may have to encounter and battle trainers along the way",
                action_type="navigate",
                target_location="Petalburg City",
                navigation_hint="Travel to route 102 -> petalburg city",
                completion_condition="reached_petalburg_city",
                priority=1
            ),
            DirectObjective( # 23
                id="petalburg_city_to_dad_first_meeting",
                description="Travel to dad's first meeting with you",
                action_type="navigate",
                target_location="Petalburg City Gym",
                navigation_hint="Travel to petalburg city -> petalburg city gym",
                completion_condition="reached_dad_first_meeting",
                priority=1
            ),
            DirectObjective( # 24
                id="help_wally_catch_ralts",
                description="Help Wally catch a Ralts",
                action_type="battle",
                target_location="Petalburg City",
                navigation_hint="After receiving pokeballs from your dad, Wally will take you to go and catch a Ralts. You will need to use a pokeball from his bag to catch the Ralts.",
                completion_condition="caught_ralts",
                priority=1
            ),

            DirectObjective( # 25
                id="petalburg_heal_at_pokecenter",
                description="Find and enter the Pokemon Center in Petalburg City to heal your Pokemon",
                action_type="interact",
                target_location="Petalburg City",
                navigation_hint="The Pokemon Center has a red roof and is marked with a red 'P' symbol. Navigate to the building and walk through the door. Inside, talk to Nurse Joy (the NPC behind the counter) and press A to heal your Pokemon. Wait for the healing animation to complete.",
                completion_condition="pokemon_healed_at_petalburg_center",
                priority=1
            ),

            # ========== Petalburg to Rustboro City (Objectives 25-28) ==========
            DirectObjective( # 26
                id="petalbug_to_route_104",
                description="Travel to route 104",
                action_type="navigate",
                target_location="Route 104",
                navigation_hint="Travel from petalburg city -> route 104",
                completion_condition="reached_route_104",
                priority=1
            ),
            DirectObjective( # 27
                id="route_104_to_petalburg_woods",
                description="Travel north to petalburg woods",
                action_type="navigate",
                target_location="Petalburg Woods",
                navigation_hint="Travel from route 104 to -> petalburg woods by requesting a path to the leftmost set of warps (S), closest to you. (note, if you request a path and fail to make meaningful progress, try to request a path to a different warp that can get you to the same destination)",
                completion_condition="reached_petalburg_woods",
                priority=1
            ),
            DirectObjective( # 28
                id="petalburg_woods_to_route_104_north",
                description="Travel north route_104_north",
                action_type="navigate",
                target_location="Route 104 North",
                navigation_hint="Travel from petalburg woods to route 104 north at the very north of the woods. You will need to battle trainers, including team aquanut on the way.",
                completion_condition="reached_route_104_north",
                priority=1
            ),

            DirectObjective( # 29
                id="route_104_north_to_rustboro_city",
                description="Travel to rustboro city",
                action_type="navigate",
                target_location="Rustboro City",
                navigation_hint="Travel from route 104 north -> rustboro city. Follow the path east the get to the bridge north. northish but will need to first go east, then across the bridge north-west-north, then north. Avoid the flower shop",
                completion_condition="reached_rustboro_city",
                priority=1
            ),

            # ========== Rustboro City to Rustboro Gym (Objectives 28-38) ==========
            DirectObjective( # 30
                id="rustboro_pokemon_center",
                description="Visit the pokemon center in rustboro city",
                action_type="navigate",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Travel from rustboro city -> rustboro city pokemon center",
                completion_condition="reached_rustboro_city_pokemon_center",
                priority=1
            ),
            DirectObjective( # 31
                id="heal_pokemon_at_rustboro_pokemon_center",
                description="Heal your pokemon at the pokemon center in rustboro city",
                action_type="interact",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Interact with the nurse in the rustboro city to heal your pokemon",
                completion_condition="healed_pokemon_at_rustboro_pokemon_center",
                priority=1
            ),
            DirectObjective( # 32
                id="catch_first_pokemon_in_route_104",
                description="Keep walking around the grass until you encounter and catch a pokemon. (once the opposing pokemon has < 50% health, use a pokeball). CRITICAL: The following are valid pokemon that can be caught in route 104: marill, poochyena, taillow, wingull. Any other pokemon can simply be defeated!",
                action_type="navigate",
                target_location="Route 104",
                navigation_hint="Navigate back to route 104, enter the grass patch in the north west corner of the map, and keep walking around the grass until you encounter and catch a pokemon. Once in the bag, navigate the PokeBalls menu carefully via (LEFT, RIGHT) and then carefully select a pokeball to use.",
                completion_condition="caught_first_pokemon",
                priority=1
            ),
            DirectObjective( # 33
                id="catch_second_pokemon_in_route_104",
                description="Keep walking around the grass until you encounter and catch a pokemon. (once the opposing pokemon has < 50% health, use a pokeball). CRITICAL: Dont catch wurmple or pokemon already in your party!",
                action_type="navigate",
                target_location="Route 104",
                navigation_hint="Navigate back to route 104, enter the grass patch in the north west corner of the map, and keep walking around the grass until you encounter and catch a pokemon. Once in the bag, navigate the PokeBalls menu carefully via (LEFT, RIGHT) and then carefully select a pokeball to use.",
                completion_condition="caught_second_pokemon",
                priority=1
            ),
            DirectObjective( # 34
                id="rustboro_pokemon_center_2",
                description="Visit the pokemon center in rustboro city again (east to bridge. Then north-west-north across bridge to rustboro.).",
                action_type="navigate",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Travel from route 104 -> rustboro city -> rustboro city pokemon center. You will need to navigate east to the bridge. Then north, west, north across the bridge to get back to rustboro.",
                completion_condition="reached_rustboro_city_pokemon_center",
                priority=1
            ),
            DirectObjective( # 35
                id="heal_pokemon_at_rustboro_pokemon_center_2",
                description="Heal your pokemon at the pokemon center in rustboro city again",
                action_type="interact",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Interact with the nurse (press A while facing nurse joy across the counter) in the rustboro city to heal your pokemon",
                completion_condition="healed_pokemon_at_rustboro_pokemon_center",
                priority=1
            ),
            DirectObjective( # 36
                id="enter_rustboro_gym",
                description="Enter the rustboro gym",
                action_type="navigate",
                target_location="Rustboro City Gym",
                navigation_hint="Travel from rustboro city -> rustboro city gym",
                completion_condition="entered_rustboro_gym",
                priority=1
            ),
            DirectObjective( # 37
                id="navigate_to_rustboro_gym_leader",
                description="Navigate to the rustboro gym leader. You will battle pokemon trainers along the way!",
                action_type="navigate",
                target_location="Rustboro City Gym",
                navigation_hint="The gym leader (roxanne) is located in the north of the gym. Use navigate_to() to efficiently navigate towards the gym leader. If navigate_to() is failing, manually via PRESS_BUTTON a few steps away and try to pathfind from a different location. You will need to navigate around the NPCs. If you get stuck, leave the gym and immediately go back inside.",
                completion_condition="reached_rustboro_gym_leader",
                priority=1
            ),
            DirectObjective(  # 38
                id="battle_rustboro_gym_leader",
                description="Battle the rustboro gym leader by facing her and pressing UP+A. Prioritize using supereffective moves. If a pokemon faints make sure to use (LEFT/RIGHT/UP/DOWN) in the pokemon selection screen before pressing A to carefully select the next pokemon. Make sure the pokemon you are selecting is highligted with a red outline (this is how you know the pokemon is selected) before pressing A!",
                action_type="battle",
                target_location="Rustboro City Gym",
                navigation_hint="Battle the rustboro gym leader. If you lose this battle, renavigate back to rustboro gym -> roxanne and try again.",
                completion_condition="roxanne_defeated_and_received_badge",
                priority=1
            ),
            DirectObjective( # 39
                id="exit_rustboro_gym",
                description="Exit the rustboro gym",
                action_type="navigate",
                target_location="Rustboro City Gym",
                navigation_hint="Exit the rustboro gym by walking south through the gym to the exit. Use navigate_to() to efficiently navigate towards the exit. The gym environment may be somewhat obstructed, so you may need to increase the variance parameter",
                completion_condition="exited_rustboro_gym",
                priority=1
            ),
            DirectObjective( # 40
                id="go_to_rustboro_city_pokemon_center_and_heal_pokemon",
                description="Go to the rustboro city pokemon center and heal your pokemon",
                action_type="navigate",
                target_location="Rustboro City Pokemon Center",
                navigation_hint="Travel to the rustboro city pokemon center and heal your pokemon. Interact with the nurse (press A while facing nurse joy across the counter).",
                completion_condition="reached_rustboro_city_pokemon_center",
                priority=1
            ),
            DirectObjective(  # 41
                id="navigate_to_route_116",
                description="Navigate to Route 116 to track down the Team Aqua grunt who stole the Devon Goods",
                action_type="navigate",
                target_location="Route 116",
                navigation_hint="Exit Rustboro City from the northeast exit. Use navigate_to() to reach Route 116. The Devon researcher will mention seeing the thief heading this direction.",
                completion_condition="reached_route_116",
                priority=1
            ),

            DirectObjective(  # 42
                id="navigate_to_rusturf_tunnel",
                description="Navigate to Rusturf Tunnel entrance at the east end of Route 116",
                action_type="navigate",
                target_location="Rusturf Tunnel",
                navigation_hint="Travel east through Route 116. You may encounter trainers along the way. The Rusturf Tunnel entrance is at the far east end of the route. Use navigate_to() to efficiently reach the tunnel entrance.",
                completion_condition="reached_rusturf_tunnel_entrance",
                priority=1
            ),

            DirectObjective(  # 43
                id="battle_team_aqua_grunt_rusturf_tunnel",
                description="Battle the Team Aqua grunt inside Rusturf Tunnel to retrieve the Devon Goods. Walk all the way to the right and then press A to interact. Keep pressing RIGHT and DOWN and A until you find him. PRESS RIGHT. Use supereffective moves and carefully select pokemon if one faints.",
                action_type="battle",
                target_location="Rusturf Tunnel",
                navigation_hint="Enter Rusturf Tunnel and battle the Team Aqua grunt. He has a level 11 Poochyena. After defeating him, you will automatically retrieve the Devon Goods. If you lose, renavigate back and try again.",
                completion_condition="defeated_team_aqua_grunt_and_retrieved_devon_goods",
                priority=1
            ),

            DirectObjective(  # 44
                id="exit_rusturf_tunnel",
                description="Exit Rusturf Tunnel back to Route 116",
                action_type="navigate",
                target_location="Rusturf Tunnel",
                navigation_hint="Walk back (LEFT+DOWN) through Rusturf Tunnel to the exit. Use navigate_to() to reach the exit efficiently.",
                completion_condition="exited_rusturf_tunnel_to_route_116",
                priority=1
            ),

            DirectObjective(  # 45
                id="navigate_to_rustboro_city_devon_corp",
                description="Return to Rustboro City and navigate to the Devon Corporation building",
                action_type="navigate",
                target_location="Devon Corporation",
                navigation_hint="Travel west through Route 116 back to Rustboro City. The Devon Corporation building is in the northwest corner of Rustboro City. The Devon researcher should be waiting outside. Use navigate_to() to efficiently navigate to Devon Corporation.",
                completion_condition="reached_devon_corporation_entrance",
                priority=1
            ),

            DirectObjective(  # 46
                id="enter_devon_corporation_and_meet_mr_stone",
                description="Enter Devon Corporation and go to the third floor to meet Mr. Stone",
                action_type="navigate",
                target_location="Devon Corporation 3F",
                navigation_hint="Enter the Devon Corporation building. The Devon researcher will escort you inside. Navigate to the third floor where Mr. Stone's office is located. Use navigate_to() or manually navigate up the stairs.",
                completion_condition="reached_mr_stone_office",
                priority=1
            ),

            DirectObjective(  # 47
                id="receive_pokenav_and_letter_from_mr_stone",
                description="Talk to Mr. Stone to receive the PokéNav and Letter for Steven",
                action_type="interact",
                target_location="Devon Corporation 3F",
                navigation_hint="Face Mr. Stone and press A to talk to him. He will thank you for retrieving the Devon Goods and give you a PokéNav and a Letter to deliver to Steven in Dewford Town.",
                completion_condition="received_pokenav_and_letter",
                priority=1
            ),

            DirectObjective(  # 48
                id="exit_devon_corporation",
                description="Exit the Devon Corporation building",
                action_type="navigate",
                target_location="Devon Corporation",
                navigation_hint="Navigate back down to the first floor and exit the Devon Corporation building. A scientist will add the Match Call function to your PokéNav when you leave.",
                completion_condition="exited_devon_corporation",
                priority=1
            ),

            DirectObjective(  # 49
                id="navigate_to_route_104_north_mr_briney",
                description="GO TO THE MAIN MENU TO EXIT THE POKEMON NAVIGATOR. Navigate to Route 104 north to find Mr. Briney at his cottage",
                action_type="navigate",
                target_location="Route 104 North - Mr. Briney's Cottage",
                navigation_hint="Exit Rustboro City from the south exit. Travel through Petalburg Woods to reach Route 104 north. Mr. Briney's cottage is on the southern beach area of Route 104. You may need to navigate through Petalburg Woods first if approaching from Rustboro. Use navigate_to() efficiently.",
                completion_condition="reached_mr_briney_cottage_route_104",
                priority=1
            ),

            DirectObjective(  # 50
                id="talk_to_mr_briney_and_sail_to_dewford",
                description="Talk to Mr. Briney and sail to Dewford Town",
                action_type="interact",
                target_location="Route 104 South - Mr. Briney's Cottage",
                navigation_hint="Enter Mr. Briney's cottage on Route 104 south (accessed from the beach area). Talk to Mr. Briney and he will offer to sail you to Dewford Town. Accept his offer to begin the sea voyage through Routes 105 and 106.",
                completion_condition="sailing_to_dewford_town",
                priority=1
            ),

            DirectObjective(  # 51
                id="arrive_at_dewford_town",
                description="Arrive at Dewford Town via Mr. Briney's boat",
                action_type="cutscene",
                target_location="Dewford Town",
                navigation_hint="Wait for the sailing cutscene to complete. You will receive a call from your dad (Norman) during the voyage. The boat will automatically arrive at Dewford Town.",
                completion_condition="arrived_at_dewford_town",
                priority=1
            ),

            DirectObjective(  # 52
                id="explore_dewford_town_and_get_old_rod",
                description="Explore Dewford Town and obtain the Old Rod from the fisherman",
                action_type="interact",
                target_location="Dewford Town",
                navigation_hint="Talk to the Fisherman standing outside the Dewford Gym (east side of town). Answer his question to receive the Old Rod, which allows you to fish for Pokemon in any body of water.",
                completion_condition="received_old_rod",
                priority=1
            ),

            DirectObjective(  # 53
                id="navigate_to_dewford_gym_entrance",
                description="Navigate to the Dewford Gym entrance",
                action_type="navigate",
                target_location="Dewford Gym",
                navigation_hint="The Dewford Gym is located on the east side of Dewford Town. Use navigate_to() to reach the gym entrance efficiently.",
                completion_condition="reached_dewford_gym_entrance",
                priority=1
            ),

            DirectObjective(  # 54
                id="navigate_to_dewford_gym_leader",
                description="Navigate to the Dewford Gym leader Brawly. The gym is dark initially and lights up as you defeat trainers.",
                action_type="navigate",
                target_location="Dewford Gym",
                navigation_hint="The Dewford Gym specializes in Fighting-type Pokemon. The gym is dimly lit initially. Navigate through the gym, battling trainers to light up sections. Brawly is located at the back of the gym. Use navigate_to() to efficiently navigate towards Brawly. You can avoid some trainers if needed as the gym layout allows direct access to the leader.",
                completion_condition="reached_dewford_gym_leader",
                priority=1
            ),

            DirectObjective(  # 55
                id="battle_dewford_gym_leader_brawly",
                description="Battle Dewford Gym Leader Brawly. Use Flying-type and Psychic-type moves for advantage. His team has Machop (Lv16), Meditite (Lv16), and Makuhita (Lv19). Watch out for Bulk Up and Focus Punch.",
                action_type="battle",
                target_location="Dewford Gym",
                navigation_hint="Battle Brawly using supereffective Flying or Psychic-type moves. His Machop and Makuhita have the Guts ability. His Meditite can use Light Screen, Reflect, and Focus Punch. Avoid using Normal, Rock, Steel, and Dark-type Pokemon. If you lose, heal at the Pokemon Center and return to battle again.",
                completion_condition="defeated_brawly_and_received_knuckle_badge",
                priority=1
            ),

            DirectObjective(  # 56
                id="exit_dewford_gym",
                description="Exit the Dewford Gym",
                action_type="navigate",
                target_location="Dewford Gym",
                navigation_hint="After defeating Brawly and receiving the Knuckle Badge and TM08 (Bulk Up), navigate to the gym exit. The gym should now be fully lit. Use navigate_to() to reach the exit efficiently.",
                completion_condition="exited_dewford_gym",
                priority=1
            ),

            DirectObjective(  # 57
                id="navigate_to_dewford_pokemon_center_and_heal",
                description="Go to Dewford Town Pokemon Center and heal your Pokemon",
                action_type="navigate",
                target_location="Dewford Town Pokemon Center",
                navigation_hint="Navigate to the Pokemon Center in Dewford Town and heal your Pokemon before exploring Granite Cave.",
                completion_condition="reached_dewford_pokemon_center_and_healed",
                priority=1
            ),

            DirectObjective(  # 58
                id="navigate_to_route_106",
                description="Navigate to Route 106 from Dewford Town",
                action_type="navigate",
                target_location="Route 106",
                navigation_hint="Exit Dewford Town from the west exit to reach Route 106. You cannot fully explore the water areas yet without Surf. Battle the trainers on the beach.",
                completion_condition="reached_route_106",
                priority=1
            ),

            DirectObjective(  # 59
                id="navigate_to_granite_cave_entrance",
                description="Navigate to Granite Cave entrance on Route 106",
                action_type="navigate",
                target_location="Granite Cave",
                navigation_hint="Granite Cave is located at the northwest area of Route 106. Navigate through Route 106 beach area to reach the cave entrance. Use navigate_to() to efficiently reach Granite Cave.",
                completion_condition="reached_granite_cave_entrance",
                priority=1
            ),

            DirectObjective(  # 60
                id="navigate_through_granite_cave_1f_to_find_hiker",
                description="Enter Granite Cave and navigate to find the hiker on 1F who gives you HM05 Flash",
                action_type="navigate",
                target_location="Granite Cave 1F",
                navigation_hint="Enter Granite Cave. Near the entrance on the first floor, there should be a hiker who will give you HM05 (Flash). Talk to him to receive Flash, which will help illuminate the dark cave. Use navigate_to() to find the hiker.",
                completion_condition="received_hm05_flash_from_hiker",
                priority=1
            ),

            DirectObjective(  # 61
                id="navigate_to_granite_cave_basement_1f",
                description="Navigate deeper into Granite Cave to basement level B1F",
                action_type="navigate",
                target_location="Granite Cave B1F",
                navigation_hint="From Granite Cave 1F, find the stairs leading down to B1F. The basement level is dark, so you may want to use Flash (though not required). Navigate down the stairs using navigate_to() or manual movement.",
                completion_condition="reached_granite_cave_b1f",
                priority=1
            ),

            DirectObjective(  # 62
                id="find_steven_in_granite_cave",
                description="Navigate through Granite Cave to find Steven Stone and deliver the Letter from Mr. Stone",
                action_type="navigate",
                target_location="Granite Cave B2F",
                navigation_hint="Steven is located in the deepest part of Granite Cave on floor B2F. Navigate from B1F down to B2F. Steven will be in the back area examining the rocks. The cave is dark so Flash helps but is not required. Use navigate_to() to efficiently navigate through the cave floors. You may encounter wild Pokemon and trainers along the way.",
                completion_condition="found_steven_in_granite_cave",
                priority=1
            ),

            DirectObjective(  # 63
                id="deliver_letter_to_steven",
                description="Talk to Steven Stone and deliver the Letter from Mr. Stone",
                action_type="interact",
                target_location="Granite Cave B2F",
                navigation_hint="Face Steven and press A to talk to him. Deliver the Letter from Mr. Stone. Steven will thank you and give you TM47 (Steel Wing) as a reward.",
                completion_condition="delivered_letter_to_steven",
                priority=1
            ),

            DirectObjective(  # 64
                id="exit_granite_cave",
                description="Exit Granite Cave back to Route 106",
                action_type="navigate",
                target_location="Granite Cave",
                navigation_hint="Navigate back through Granite Cave from B2F to B1F to 1F and finally exit to Route 106. Use navigate_to() to efficiently navigate back to the entrance. You can use an Escape Rope if you have one to exit instantly.",
                completion_condition="exited_granite_cave_to_route_106",
                priority=1
            ),

            DirectObjective(  # 65
                id="return_to_dewford_town_from_route_106",
                description="Return to Dewford Town from Route 106",
                action_type="navigate",
                target_location="Dewford Town",
                navigation_hint="Navigate east through Route 106 back to Dewford Town. Use navigate_to() to efficiently return to Dewford Town.",
                completion_condition="returned_to_dewford_town_from_route_106",
                priority=1
            ),

            DirectObjective(  # 66
                id="talk_to_mr_briney_to_sail_to_route_109",
                description="Find Mr. Briney in Dewford Town and sail to Route 109 (Slateport City direction)",
                action_type="interact",
                target_location="Dewford Town",
                navigation_hint="Talk to Mr. Briney at the pier in Dewford Town. Ask him to sail you to Route 109/Slateport City. He will take you through Routes 107 and 108 to Route 109.",
                completion_condition="sailing_to_route_109",
                priority=1
            ),

            # ========== Route 109 & Seashore House (Objectives 67-71) ==========
            DirectObjective(  # 67
                id="arrive_at_route_109",
                description="Arrive at Route 109 via Mr. Briney's boat and disembark",
                action_type="cutscene",
                target_location="Route 109",
                navigation_hint="Wait for the sailing cutscene to complete. The boat will automatically arrive at Route 109 beach. You will receive a PokéNav call from your dad (Norman) during the voyage.",
                completion_condition="arrived_at_route_109",
                priority=1
            ),

            DirectObjective(  # 68
                id="explore_route_109_beach",
                description="Explore Route 109 beach area and collect any visible items",
                action_type="navigate",
                target_location="Route 109",
                navigation_hint="Route 109 is a beach area with trainers and hidden items. Navigate the beach and battle trainers if desired. Use navigate_to() to efficiently explore the area.",
                completion_condition="explored_route_109_beach",
                priority=1
            ),

            DirectObjective(  # 69
                id="enter_seashore_house",
                description="Enter the Seashore House on Route 109 beach",
                action_type="navigate",
                target_location="Seashore House",
                navigation_hint="The Seashore House is a small building on Route 109 beach. Navigate to it and enter through the door.",
                completion_condition="entered_seashore_house",
                priority=1
            ),

            DirectObjective(  # 70
                id="battle_trainers_seashore_house",
                description="Battle the three trainers in Seashore House to receive six Soda Pops as reward",
                action_type="battle",
                target_location="Seashore House",
                navigation_hint="Inside the Seashore House, defeat all three trainers in sequence. They will have Water-type Pokémon. After defeating all three, talk to the house owner to receive six Soda Pops (healing items that restore 60 HP).",
                completion_condition="defeated_all_seashore_house_trainers",
                priority=1
            ),

            DirectObjective(  # 71 - OPTIONAL
                id="optional_return_to_rustboro_for_exp_share",
                description="OPTIONAL: Return to Rustboro City via Mr. Briney to receive Exp. Share from Mr. Stone",
                action_type="navigate",
                target_location="Devon Corporation",
                navigation_hint="This is optional but recommended. Talk to Mr. Briney to sail back to Route 104, then navigate through Petalburg Woods to Rustboro City. Visit Mr. Stone on the third floor of Devon Corporation. He will give you the Exp. Share, which allows non-battling Pokémon to gain experience. After receiving it, return to Route 109 via Mr. Briney.",
                completion_condition="received_exp_share_from_mr_stone",
                priority=2
            ),

            # ========== Slateport City (Objectives 72-77) ==========
            DirectObjective(  # 72
                id="navigate_to_slateport_city",
                description="Navigate north from Route 109 to Slateport City",
                action_type="navigate",
                target_location="Slateport City",
                navigation_hint="From Route 109, head north to reach Slateport City. Use navigate_to() for efficient navigation.",
                completion_condition="reached_slateport_city",
                priority=1
            ),

            DirectObjective(  # 73
                id="heal_at_slateport_pokemon_center",
                description="Find and enter the Pokémon Center in Slateport City to heal your Pokémon",
                action_type="interact",
                target_location="Slateport City Pokémon Center",
                navigation_hint="The Pokémon Center has a red roof and is marked with a red 'P' symbol. Navigate to the building and walk through the door. Inside, talk to Nurse Joy (the NPC behind the counter) and press A to heal your Pokémon.",
                completion_condition="healed_pokemon_at_slateport_center",
                priority=1
            ),

            DirectObjective(  # 74 - OPTIONAL
                id="optional_explore_slateport_market",
                description="OPTIONAL: Explore Slateport Market and Pokémon Fan Club for items",
                action_type="navigate",
                target_location="Slateport City",
                navigation_hint="The Slateport Market is located in the northern part of the city. You can purchase vitamins, obtain a Powder Jar, and visit the Pokémon Fan Club for a Soothe Bell (requires high friendship). The Name Rater is also available to change Pokémon nicknames.",
                completion_condition="explored_slateport_market",
                priority=2
            ),

            DirectObjective(  # 75
                id="navigate_to_oceanic_museum",
                description="Navigate to the Oceanic Museum in Slateport City",
                action_type="navigate",
                target_location="Oceanic Museum",
                navigation_hint="The Oceanic Museum is a large building in Slateport City. Look for a building with a distinctive appearance. Use navigate_to() to reach the museum entrance. Pay $50 to enter.",
                completion_condition="reached_oceanic_museum_entrance",
                priority=1
            ),

            DirectObjective(  # 76
                id="battle_team_aqua_in_oceanic_museum",
                description="Enter the Oceanic Museum, pay $50 entry fee, and battle two Team Aqua Grunts on the second floor to protect the Devon Goods",
                action_type="battle",
                target_location="Oceanic Museum 2F",
                navigation_hint="After paying the $50 entry fee, go to the second floor. You'll encounter two Team Aqua Grunts threatening Captain Stern. Battle both grunts to protect the Devon Goods. After defeating them, you'll meet Team Aqua's leader, Archie, who will leave peacefully. Use supereffective moves and heal between battles if needed.",
                completion_condition="defeated_team_aqua_grunts_in_museum",
                priority=1
            ),

            DirectObjective(  # 77
                id="deliver_devon_goods_to_captain_stern",
                description="Talk to Captain Stern to deliver the Devon Goods and receive TM46 (Thief) as reward",
                action_type="interact",
                target_location="Oceanic Museum 2F",
                navigation_hint="After defeating the Team Aqua Grunts, talk to Captain Stern. Deliver the Devon Goods to him. He will thank you and give you TM46 (Thief) as a reward.",
                completion_condition="delivered_devon_goods_to_stern",
                priority=1
            ),

            # ========== Route 110 (Objectives 78-84) ==========
            DirectObjective(  # 78
                id="navigate_to_route_110",
                description="Exit Slateport City and navigate north to Route 110",
                action_type="navigate",
                target_location="Route 110",
                navigation_hint="Exit Slateport City from the north exit to reach Route 110. Use navigate_to() for efficient navigation.",
                completion_condition="reached_route_110",
                priority=1
            ),

            DirectObjective(  # 79
                id="meet_professor_birch_on_route_110",
                description="Meet Professor Birch on Route 110 who will register you for PokéNav features",
                action_type="interact",
                target_location="Route 110",
                navigation_hint="Professor Birch should be on Route 110 near the entrance from Slateport. Talk to him and he will register you for additional PokéNav features.",
                completion_condition="met_birch_on_route_110",
                priority=1
            ),

            DirectObjective(  # 80 - OPTIONAL
                id="optional_trick_house_first_puzzle",
                description="OPTIONAL: Complete the first Trick House puzzle for Rare Candy reward",
                action_type="navigate",
                target_location="Trick House",
                navigation_hint="The Trick House is a small building on Route 110. Enter and talk to the Trick Master. Find the hidden password (use Itemfinder or search carefully), then navigate through the maze while battling trainers. Complete the puzzle to receive a Rare Candy. This is the first of eight progressively unlocking puzzles.",
                completion_condition="completed_trick_house_first_puzzle",
                priority=2
            ),

            DirectObjective(  # 81
                id="navigate_through_route_110",
                description="Navigate north through Route 110 toward Mauville City",
                action_type="navigate",
                target_location="Route 110",
                navigation_hint="Travel north through Route 110. You may encounter trainers and wild Pokémon. Continue navigating north using navigate_to() until you encounter your rival.",
                completion_condition="navigated_through_route_110",
                priority=1
            ),

            DirectObjective(  # 82
                id="battle_rival_route_110",
                description="Battle your rival May/Brendan on Route 110. Their team includes Wingull (Lv.18), Lombre (Lv.18), and their starter evolution (Lv.20)",
                action_type="battle",
                target_location="Route 110",
                navigation_hint="Your rival will challenge you to a battle. Their team has evolved and grown stronger: Wingull (Water/Flying, Lv.18), Lombre (Water/Grass, Lv.18), and their starter's second evolution (Lv.20) which has type advantage over yours. Use supereffective moves and heal between Pokémon if needed. Reward: $900.",
                completion_condition="defeated_rival_on_route_110",
                priority=1
            ),

            DirectObjective(  # 83
                id="receive_itemfinder_from_rival",
                description="After defeating your rival, they will give you the Itemfinder",
                action_type="interact",
                target_location="Route 110",
                navigation_hint="Talk to your rival after the battle. They will give you the Itemfinder, a device that helps locate hidden items on the ground.",
                completion_condition="received_itemfinder",
                priority=1
            ),

            DirectObjective(  # 84
                id="heal_at_route_110_pokemon_center",
                description="Heal your Pokémon at the Pokémon Center on Route 110 before continuing",
                action_type="interact",
                target_location="Route 110 Pokémon Center",
                navigation_hint="There may be a Pokémon Center on Route 110 or nearby. Navigate to it and heal your Pokémon before continuing to Mauville City.",
                completion_condition="healed_pokemon_at_route_110_center",
                priority=1
            ),

            # ========== Route 118 to Mauville City (Objectives 85-93) ==========
            DirectObjective(  # 85
                id="navigate_to_mauville_city",
                description="Continue north from Route 110 to reach Mauville City",
                action_type="navigate",
                target_location="Mauville City",
                navigation_hint="From Route 110, continue north to reach Mauville City. You may need to navigate through some grass and trainer areas. Use navigate_to() for efficient navigation.",
                completion_condition="reached_mauville_city",
                priority=1
            ),

            DirectObjective(  # 86
                id="obtain_bicycle_from_rydels_cycles",
                description="Visit Rydel's Cycles in Mauville City to receive a free bicycle (choose between Mach Bike or Acro Bike)",
                action_type="interact",
                target_location="Rydel's Cycles",
                navigation_hint="Rydel's Cycles is located in Mauville City. Enter the shop and talk to Rydel. He will give you a free bicycle. You can choose between the Mach Bike (faster, can ride up sandy slopes) or Acro Bike (can perform tricks, cross narrow bridges). Choose based on your preference - you can exchange it later.",
                completion_condition="received_bicycle_from_rydel",
                priority=1
            ),

            DirectObjective(  # 87
                id="heal_at_mauville_pokemon_center",
                description="Heal your Pokémon at the Mauville City Pokémon Center before challenging the gym",
                action_type="interact",
                target_location="Mauville City Pokémon Center",
                navigation_hint="Navigate to the Pokémon Center in Mauville City. Talk to Nurse Joy to heal your Pokémon before the upcoming gym battle and Wally encounter.",
                completion_condition="healed_pokemon_at_mauville_center",
                priority=1
            ),

            DirectObjective(  # 88
                id="battle_wally_outside_mauville_gym",
                description="Battle Wally outside Mauville Gym. He has a Ralts (Lv.16)",
                action_type="battle",
                target_location="Mauville City",
                navigation_hint="Navigate to the Mauville Gym entrance. Wally will be waiting outside and will challenge you to a battle. He has a single Ralts at Level 16 (Psychic-type, weak to Bug, Ghost, and Dark-type moves). This should be an easy battle.",
                completion_condition="defeated_wally_outside_gym",
                priority=1
            ),

            DirectObjective(  # 89
                id="enter_mauville_gym",
                description="Enter the Mauville Gym to challenge Gym Leader Wattson",
                action_type="navigate",
                target_location="Mauville Gym",
                navigation_hint="After defeating Wally, enter the Mauville Gym through the front door. The gym specializes in Electric-type Pokémon.",
                completion_condition="entered_mauville_gym",
                priority=1
            ),

            DirectObjective(  # 90
                id="navigate_to_gym_leader_wattson",
                description="Navigate through the Mauville Gym puzzle to reach Gym Leader Wattson",
                action_type="navigate",
                target_location="Mauville Gym",
                navigation_hint="The Mauville Gym has an electric barrier puzzle. You need to battle trainers in the correct sequence to deactivate barriers blocking your path to Wattson. Navigate through the gym using navigate_to() or manually. Battle trainers as needed to progress. Wattson is in the back of the gym.",
                completion_condition="reached_gym_leader_wattson",
                priority=1
            ),

            DirectObjective(  # 91
                id="battle_gym_leader_wattson",
                description="Battle Gym Leader Wattson. His team: Voltorb (Lv.20), Electrike (Lv.20), Magneton (Lv.22), Manectric (Lv.24). Use Ground-type moves for advantage.",
                action_type="battle",
                target_location="Mauville Gym",
                navigation_hint="Battle Wattson using Ground-type moves for maximum effectiveness (Electric attacks don't affect Ground-types). His Pokémon: Voltorb (Lv.20 - can use Self-Destruct), Electrike (Lv.20), Magneton (Lv.22 - Steel/Electric, weak to Fire and Fighting), Manectric (Lv.24 - his strongest). Avoid using Water or Flying-types. Reward: Dynamo Badge, TM34 (Shock Wave), ability to use Rock Smash outside battle. If you lose, heal at the Pokémon Center and return.",
                completion_condition="defeated_wattson_and_received_dynamo_badge",
                priority=1
            ),

            DirectObjective(  # 92
                id="exit_mauville_gym",
                description="Exit the Mauville Gym after defeating Wattson",
                action_type="navigate",
                target_location="Mauville Gym",
                navigation_hint="After receiving the Dynamo Badge and TM34, navigate back through the gym to the exit. Use navigate_to() to reach the exit efficiently.",
                completion_condition="exited_mauville_gym",
                priority=1
            ),

            DirectObjective(  # 93
                id="heal_at_mauville_pokemon_center_after_gym",
                description="Return to Mauville Pokémon Center to heal your Pokémon after the gym battle",
                action_type="interact",
                target_location="Mauville City Pokémon Center",
                navigation_hint="Navigate back to the Pokémon Center in Mauville City and heal your Pokémon. You're now ready to continue your journey.",
                completion_condition="healed_pokemon_after_mauville_gym",
                priority=1
            ),

            # ========== Route 111 to Fallarbor Town (Objectives 94-105) ==========
            DirectObjective(  # 94
                id="navigate_to_route_111_south",
                description="Navigate north from Mauville City to Route 111 (south section)",
                action_type="navigate",
                target_location="Route 111",
                navigation_hint="Exit Mauville City from the north to reach Route 111. Note that a desert with sandstorm blocks the direct path north, so you'll navigate around it.",
                completion_condition="reached_route_111_south",
                priority=1
            ),

            DirectObjective(  # 95 - OPTIONAL
                id="optional_winstrate_family_challenge",
                description="OPTIONAL: Complete the Winstrate family challenge (4 consecutive battles) to receive Macho Brace",
                action_type="battle",
                target_location="Route 111 - Winstrate House",
                navigation_hint="The Winstrate family home is at the water's edge on Route 111. Battle Victor, Victoria, Vivi, and Vicky in sequence without healing between battles. Reward: Macho Brace item.",
                completion_condition="defeated_winstrate_family",
                priority=2
            ),

            DirectObjective(  # 96
                id="navigate_to_route_112_south",
                description="Navigate west from Route 111 to Route 112 (south section)",
                action_type="navigate",
                target_location="Route 112",
                navigation_hint="From Route 111, head west to Route 112. You'll encounter Team Magma grunts blocking the cable car to Mt. Chimney summit, forcing you to take an alternate route through Fiery Path.",
                completion_condition="reached_route_112_south",
                priority=1
            ),

            DirectObjective(  # 97
                id="navigate_through_fiery_path",
                description="Navigate through Fiery Path cave to reach the northern section of Route 112",
                action_type="navigate",
                target_location="Fiery Path",
                navigation_hint="Enter Fiery Path, a sweltering passage through the mountain base. Navigate through the cave to the northern exit. Boulders block side tunnels (require Strength move later). Battle trainers if encountered.",
                completion_condition="exited_fiery_path_north",
                priority=1
            ),

            DirectObjective(  # 98
                id="navigate_to_route_113",
                description="Navigate through Route 112 north to Route 113",
                action_type="navigate",
                target_location="Route 113",
                navigation_hint="From the northern exit of Fiery Path, continue through Route 112 north to reach Route 113. This route is constantly covered in volcanic ash from Mt. Chimney.",
                completion_condition="reached_route_113",
                priority=1
            ),

            DirectObjective(  # 99
                id="obtain_soot_sack_from_glass_workshop",
                description="Visit the Glass Workshop on Route 113 to receive the Soot Sack",
                action_type="interact",
                target_location="Glass Workshop - Route 113",
                navigation_hint="Enter the Glass Workshop on Route 113. Talk to the owner to receive the Soot Sack, which collects volcanic ash as you walk. Ash can be traded for flutes and decorations.",
                completion_condition="received_soot_sack",
                priority=1
            ),

            DirectObjective(  # 100
                id="navigate_to_fallarbor_town",
                description="Continue west through Route 113 to reach Fallarbor Town",
                action_type="navigate",
                target_location="Fallarbor Town",
                navigation_hint="Navigate west through Route 113 to reach Fallarbor Town. Battle trainers along the route if desired. Collect items like TM32 (Double Team) if visible.",
                completion_condition="reached_fallarbor_town",
                priority=1
            ),

            DirectObjective(  # 101
                id="heal_at_fallarbor_pokemon_center",
                description="Heal your Pokémon at Fallarbor Town Pokémon Center",
                action_type="interact",
                target_location="Fallarbor Town Pokémon Center",
                navigation_hint="Navigate to the Pokémon Center in Fallarbor Town and heal your Pokémon.",
                completion_condition="healed_at_fallarbor_center",
                priority=1
            ),

            DirectObjective(  # 102 - OPTIONAL
                id="optional_meet_lanette_and_move_tutors",
                description="OPTIONAL: Talk to Lanette to upgrade PC system and visit Move Tutors in Fallarbor Town",
                action_type="interact",
                target_location="Fallarbor Town",
                navigation_hint="Visit Lanette in Fallarbor Town who will upgrade your PC system. The Move Tutor girl teaches Metronome (one-time). The Move Maniac teaches forgotten moves for Heart Scales.",
                completion_condition="met_lanette_and_tutors",
                priority=2
            ),

            DirectObjective(  # 103
                id="navigate_to_route_114",
                description="Navigate east from Fallarbor Town to Route 114",
                action_type="navigate",
                target_location="Route 114",
                navigation_hint="Exit Fallarbor Town heading east to reach Route 114. This route is pockmarked by fallen meteorites and leads to Meteor Falls.",
                completion_condition="reached_route_114",
                priority=1
            ),

            DirectObjective(  # 104
                id="navigate_to_meteor_falls",
                description="Navigate through Route 114 to reach Meteor Falls entrance",
                action_type="navigate",
                target_location="Meteor Falls",
                navigation_hint="Travel through Route 114 to reach Meteor Falls. Battle trainers along the way if encountered. Use navigate_to() for efficient navigation.",
                completion_condition="reached_meteor_falls_entrance",
                priority=1
            ),

            DirectObjective(  # 105
                id="witness_team_magma_steal_meteorite",
                description="Enter Meteor Falls and witness Team Magma stealing Professor Cozmo's Meteorite",
                action_type="cutscene",
                target_location="Meteor Falls",
                navigation_hint="Enter Meteor Falls. You'll witness Team Magma stealing Professor Cozmo's Meteorite. Team Aqua leader Archie will arrive to oppose them, but Team Magma escapes. Collect the Full Heal and Moon Stone if desired.",
                completion_condition="witnessed_meteorite_theft",
                priority=1
            ),

            # ========== Mt. Chimney to Lavaridge Town (Objectives 106-120) ==========
            DirectObjective(  # 106
                id="navigate_to_mt_chimney_cable_car",
                description="Return to Route 112 and take the cable car to Mt. Chimney summit",
                action_type="navigate",
                target_location="Mt. Chimney",
                navigation_hint="Navigate back through Route 114 to Route 112. The Team Magma grunts are now gone, so you can access the cable car. Take the cable car to Mt. Chimney summit.",
                completion_condition="reached_mt_chimney_summit",
                priority=1
            ),

            DirectObjective(  # 107
                id="battle_team_magma_on_mt_chimney",
                description="Battle Team Magma grunts and admins on Mt. Chimney to stop their evil plan",
                action_type="battle",
                target_location="Mt. Chimney",
                navigation_hint="On Mt. Chimney, defeat two Team Magma Grunts, then battle Admin Tabitha (Numel, Poochyena, Zubat). Finally, confront Leader Maxie (Mightyena, Zubat, Camerupt). Use Water and Ground-type moves effectively. Recover the Meteorite upon victory.",
                completion_condition="defeated_team_magma_mt_chimney",
                priority=1
            ),

            DirectObjective(  # 108
                id="navigate_down_jagged_pass",
                description="Navigate down Jagged Pass using ledge jumps to reach Lavaridge Town",
                action_type="navigate",
                target_location="Jagged Pass",
                navigation_hint="From Mt. Chimney summit, navigate down Jagged Pass by jumping off ledges. Battle trainers along the way. Collect items like Full Heal, Burn Heal, and Great Ball if desired.",
                completion_condition="descended_jagged_pass",
                priority=1
            ),

            DirectObjective(  # 109
                id="arrive_at_lavaridge_town",
                description="Arrive at Lavaridge Town at the base of Jagged Pass",
                action_type="navigate",
                target_location="Lavaridge Town",
                navigation_hint="Continue down Jagged Pass until you reach Lavaridge Town, a town known for its hot springs and Pokémon Gym.",
                completion_condition="reached_lavaridge_town",
                priority=1
            ),

            DirectObjective(  # 110
                id="heal_at_lavaridge_pokemon_center",
                description="Heal your Pokémon at Lavaridge Town Pokémon Center",
                action_type="interact",
                target_location="Lavaridge Town Pokémon Center",
                navigation_hint="Navigate to the Pokémon Center and heal your team before challenging the gym. You can also collect a hidden Ice Heal from the hot springs and receive a Wynaut egg from an NPC.",
                completion_condition="healed_at_lavaridge_center",
                priority=1
            ),

            DirectObjective(  # 111 - OPTIONAL
                id="optional_obtain_items_in_lavaridge",
                description="OPTIONAL: Collect Charcoal item and visit Herb Shop in Lavaridge Town",
                action_type="interact",
                target_location="Lavaridge Town",
                navigation_hint="Talk to the old man to receive Charcoal. Visit the Herb Shop to purchase herbal medicines. Learn Mimic move from the Move Tutor.",
                completion_condition="obtained_lavaridge_items",
                priority=2
            ),

            DirectObjective(  # 112
                id="enter_lavaridge_gym",
                description="Enter the Lavaridge Gym to challenge Gym Leader Flannery",
                action_type="navigate",
                target_location="Lavaridge Gym",
                navigation_hint="Navigate to the Lavaridge Gym. The gym specializes in Fire-type Pokémon.",
                completion_condition="entered_lavaridge_gym",
                priority=1
            ),

            DirectObjective(  # 113
                id="navigate_to_gym_leader_flannery",
                description="Navigate through the Lavaridge Gym to reach Gym Leader Flannery",
                action_type="navigate",
                target_location="Lavaridge Gym",
                navigation_hint="The gym has a puzzle with holes in the floor. Navigate carefully to reach Flannery. Battle gym trainers if needed.",
                completion_condition="reached_gym_leader_flannery",
                priority=1
            ),

            DirectObjective(  # 114
                id="battle_gym_leader_flannery",
                description="Battle Gym Leader Flannery. Her team: Numel (Lv.26), Slugma (Lv.26), Camerupt (Lv.28), Torkoal (Lv.29). Use Water or Ground-type moves.",
                action_type="battle",
                target_location="Lavaridge Gym",
                navigation_hint="Battle Flannery using Water, Ground, or Rock-type moves for super-effective damage. Her Torkoal is her strongest Pokémon with high defense. Reward: Heat Badge, TM50 (Overheat). If you lose, heal and return.",
                completion_condition="defeated_flannery_and_received_heat_badge",
                priority=1
            ),

            DirectObjective(  # 115
                id="exit_lavaridge_gym",
                description="Exit the Lavaridge Gym after defeating Flannery",
                action_type="navigate",
                target_location="Lavaridge Gym",
                navigation_hint="After receiving the Heat Badge and TM50, navigate back through the gym to the exit.",
                completion_condition="exited_lavaridge_gym",
                priority=1
            ),

            DirectObjective(  # 116
                id="receive_go_goggles_from_rival",
                description="Meet your rival May/Brendan outside the gym who will give you the Go-Goggles",
                action_type="interact",
                target_location="Lavaridge Town",
                navigation_hint="After exiting the gym, your rival will meet you and give you the Go-Goggles. These allow you to navigate through sandstorms on Route 111's desert.",
                completion_condition="received_go_goggles",
                priority=1
            ),

            DirectObjective(  # 117
                id="heal_at_lavaridge_center_after_gym",
                description="Return to Lavaridge Pokémon Center to heal after the gym battle",
                action_type="interact",
                target_location="Lavaridge Town Pokémon Center",
                navigation_hint="Heal your Pokémon at the Pokémon Center before continuing your journey.",
                completion_condition="healed_after_lavaridge_gym",
                priority=1
            ),

            # ========== Route 111 Desert to Petalburg Gym (Objectives 118-130) ==========
            DirectObjective(  # 118
                id="navigate_to_route_111_desert",
                description="Navigate to Route 111's desert area using the Go-Goggles",
                action_type="navigate",
                target_location="Route 111 Desert",
                navigation_hint="From Lavaridge Town, navigate back through Jagged Pass and Mt. Chimney cable car to Route 111. With the Go-Goggles, you can now explore the desert area that was previously blocked by sandstorm.",
                completion_condition="reached_route_111_desert",
                priority=1
            ),

            DirectObjective(  # 119 - OPTIONAL
                id="optional_explore_mirage_tower_for_fossil",
                description="OPTIONAL: Explore Mirage Tower in the desert and obtain either Root Fossil or Claw Fossil",
                action_type="navigate",
                target_location="Mirage Tower",
                navigation_hint="Mirage Tower is a four-floor structure in the desert with unstable floors. Requires Mach Bike and Rock Smash. Navigate to the top floor and choose between Root Fossil (Lileep) or Claw Fossil (Anorith). You can only take one! The tower disappears after taking a fossil.",
                completion_condition="obtained_fossil_from_mirage_tower",
                priority=2
            ),

            DirectObjective(  # 120 - OPTIONAL
                id="optional_collect_desert_items",
                description="OPTIONAL: Collect items in Route 111 desert (Stardust, Protein, Rare Candy, TM37)",
                action_type="navigate",
                target_location="Route 111 Desert",
                navigation_hint="Explore the desert to find Stardust, Protein, Rare Candy, and TM37 (Sandstorm). Battle trainers for experience.",
                completion_condition="collected_desert_items",
                priority=2
            ),

            DirectObjective(  # 121
                id="navigate_to_petalburg_city_for_norman",
                description="Navigate to Petalburg City to challenge your father Norman at the Gym",
                action_type="navigate",
                target_location="Petalburg City",
                navigation_hint="With four badges earned, you can now challenge Norman at Petalburg Gym. Navigate from Route 111 through Route 101/102 to Petalburg City, or use Fly if you've taught it to a Pokémon.",
                completion_condition="reached_petalburg_for_norman_battle",
                priority=1
            ),

            DirectObjective(  # 122
                id="heal_at_petalburg_pokemon_center",
                description="Heal your Pokémon at Petalburg City Pokémon Center before the gym",
                action_type="interact",
                target_location="Petalburg City Pokémon Center",
                navigation_hint="Navigate to the Pokémon Center and heal your team. Norman's gym is challenging, so ensure you're well-prepared.",
                completion_condition="healed_at_petalburg_for_norman",
                priority=1
            ),

            DirectObjective(  # 123
                id="enter_petalburg_gym",
                description="Enter the Petalburg Gym to challenge your father Norman",
                action_type="navigate",
                target_location="Petalburg Gym",
                navigation_hint="Navigate to the Petalburg Gym. The gym has eight trainer rooms, each focusing on different battle strategies. You must defeat at least three trainers to reach Norman.",
                completion_condition="entered_petalburg_gym_for_norman",
                priority=1
            ),

            DirectObjective(  # 124
                id="navigate_to_gym_leader_norman",
                description="Battle at least three gym trainers and navigate to Gym Leader Norman",
                action_type="navigate",
                target_location="Petalburg Gym",
                navigation_hint="The gym has themed rooms: Speed, Accuracy, Confusion, Defense, Recovery, Strength, and One-Hit KO. Choose at least three rooms to battle trainers and unlock the path to Norman.",
                completion_condition="reached_gym_leader_norman",
                priority=1
            ),

            DirectObjective(  # 125
                id="battle_gym_leader_norman",
                description="Battle Gym Leader Norman (your father). His team: Spinda (Lv.27), Vigoroth (Lv.27), Linoone (Lv.29), Slaking (Lv.31). Use Fighting-type moves.",
                action_type="battle",
                target_location="Petalburg Gym",
                navigation_hint="Battle Norman using Fighting-type moves for super-effective damage against Normal-types. Exploit Slaking's Truant ability (it can only attack every other turn). All his Pokémon know Facade. Reward: Balance Badge, TM42 (Facade), enables use of Surf. If you lose, heal and return.",
                completion_condition="defeated_norman_and_received_balance_badge",
                priority=1
            ),

            DirectObjective(  # 126
                id="exit_petalburg_gym",
                description="Exit the Petalburg Gym after defeating Norman",
                action_type="navigate",
                target_location="Petalburg Gym",
                navigation_hint="After receiving the Balance Badge and TM42, navigate back through the gym to the exit.",
                completion_condition="exited_petalburg_gym_after_norman",
                priority=1
            ),

            DirectObjective(  # 127
                id="receive_hm03_surf_from_wallys_father",
                description="Receive HM03 (Surf) from Wally's father outside the gym",
                action_type="interact",
                target_location="Petalburg City",
                navigation_hint="After exiting the gym, Wally's father will meet you and give you HM03 (Surf). This allows you to travel across water and access many new areas.",
                completion_condition="received_hm03_surf",
                priority=1
            ),

            DirectObjective(  # 128
                id="visit_mom_for_amulet_coin",
                description="Visit your mom in Littleroot Town to receive the Amulet Coin",
                action_type="interact",
                target_location="Littleroot Town",
                navigation_hint="Travel south to Littleroot Town and visit your mom in your house. She will give you the Amulet Coin, which doubles prize money from battles when held by a Pokémon.",
                completion_condition="received_amulet_coin_from_mom",
                priority=1
            ),

            DirectObjective(  # 129
                id="heal_after_norman_and_surf",
                description="Heal your Pokémon at a Pokémon Center after obtaining Surf",
                action_type="interact",
                target_location="Pokémon Center",
                navigation_hint="Heal your Pokémon at any nearby Pokémon Center. You're now ready to explore water routes with Surf.",
                completion_condition="healed_after_receiving_surf",
                priority=1
            ),

            # ========== Route 118 to Fortree City (Objectives 130-145) ==========
            DirectObjective(  # 130
                id="navigate_to_route_118",
                description="Navigate to Route 118 east of Mauville City using Surf",
                action_type="navigate",
                target_location="Route 118",
                navigation_hint="From Mauville City, head east to Route 118. You'll need Surf to fully explore this route. Battle trainers and collect items along the way.",
                completion_condition="reached_route_118",
                priority=1
            ),

            DirectObjective(  # 131
                id="obtain_good_rod_from_fisherman",
                description="Talk to the Fisherman on Route 118 to receive the Good Rod",
                action_type="interact",
                target_location="Route 118",
                navigation_hint="Find the Fisherman on Route 118 and talk to him. He will give you the Good Rod, which allows you to catch better Pokémon when fishing.",
                completion_condition="received_good_rod",
                priority=1
            ),

            DirectObjective(  # 132 - OPTIONAL
                id="optional_new_mauville_quest",
                description="OPTIONAL: Complete the New Mauville quest for Wattson to receive TM24 (Thunderbolt)",
                action_type="navigate",
                target_location="New Mauville",
                navigation_hint="Return to Mauville City and talk to Wattson. He'll request assistance shutting down New Mauville's generator and give you the Basement Key. Access New Mauville by surfing east beneath Seaside Cycling Road on Route 110. Navigate using colored floor switches and shut down the generator. Reward: Thunder Stone and TM24 (Thunderbolt).",
                completion_condition="completed_new_mauville_quest",
                priority=2
            ),

            DirectObjective(  # 133
                id="navigate_to_route_119",
                description="Navigate from Route 118 to Route 119 heading north",
                action_type="navigate",
                target_location="Route 119",
                navigation_hint="From Route 118, travel north to reach Route 119, a tropical rainforest route with extreme weather. Use Surf to cross water areas.",
                completion_condition="reached_route_119",
                priority=1
            ),

            DirectObjective(  # 134
                id="navigate_to_weather_institute",
                description="Navigate through Route 119 to reach the Weather Institute",
                action_type="navigate",
                target_location="Weather Institute",
                navigation_hint="The Weather Institute is located at the northern end of Route 119. Navigate through the route, battling trainers if encountered. Team Aqua has infiltrated the facility.",
                completion_condition="reached_weather_institute",
                priority=1
            ),

            DirectObjective(  # 135
                id="battle_team_aqua_at_weather_institute",
                description="Battle Team Aqua Grunts and Admin Shelly at the Weather Institute to save the scientists",
                action_type="battle",
                target_location="Weather Institute",
                navigation_hint="Enter the Weather Institute and battle Team Aqua Grunts on the first floor. Then face Admin Shelly (Carvanha Lv.28, Mightyena Lv.28) on the second floor. Reward: Castform (Lv.25) holding Mystic Water.",
                completion_condition="defeated_team_aqua_weather_institute",
                priority=1
            ),

            DirectObjective(  # 136
                id="battle_rival_route_119_receive_fly",
                description="Battle your rival May/Brendan on Route 119 and receive HM02 (Fly)",
                action_type="battle",
                target_location="Route 119",
                navigation_hint="After exiting the Weather Institute, your rival will challenge you to a battle. Their team: Pelipper/Lombre/Slugma and their starter evolution (Lv.29-31). Use type advantages. Reward: HM02 (Fly) and $1,860.",
                completion_condition="defeated_rival_route_119_received_fly",
                priority=1
            ),

            DirectObjective(  # 137
                id="navigate_to_fortree_city",
                description="Continue north through Route 119 to reach Fortree City",
                action_type="navigate",
                target_location="Fortree City",
                navigation_hint="From Route 119, continue north to reach Fortree City, a settlement built among ancient trees. The gym entrance will be blocked by something invisible initially.",
                completion_condition="reached_fortree_city",
                priority=1
            ),

            DirectObjective(  # 138
                id="heal_at_fortree_pokemon_center",
                description="Heal your Pokémon at Fortree City Pokémon Center",
                action_type="interact",
                target_location="Fortree City Pokémon Center",
                navigation_hint="Navigate to the Pokémon Center in Fortree City and heal your team.",
                completion_condition="healed_at_fortree_center",
                priority=1
            ),

            DirectObjective(  # 139 - OPTIONAL
                id="optional_fortree_city_activities",
                description="OPTIONAL: Complete activities in Fortree City (TM10 Hidden Power quiz, Pokémon trade, Secret Base shop)",
                action_type="interact",
                target_location="Fortree City",
                navigation_hint="Visit the northwest house for TM10 (Hidden Power) quiz. Trade Volbeat for Plusle in another house. Access Secret Base Shop via ladder near Poké Mart. Interact with Pokémon Breeder's Wingull.",
                completion_condition="completed_fortree_activities",
                priority=2
            ),

            DirectObjective(  # 140
                id="navigate_to_route_120",
                description="Navigate east from Fortree City to Route 120",
                action_type="navigate",
                target_location="Route 120",
                navigation_hint="Head east from Fortree City to reach Route 120, a lengthy route with many trainers. You need to progress here to get the Devon Scope.",
                completion_condition="reached_route_120",
                priority=1
            ),

            DirectObjective(  # 141
                id="meet_steven_and_receive_devon_scope",
                description="Meet Steven on the northern bridge of Route 120 who will give you the Devon Scope",
                action_type="interact",
                target_location="Route 120",
                navigation_hint="Navigate through Route 120 until you reach the northern bridge. Steven will be waiting and will give you the Devon Scope, which reveals invisible Kecleon.",
                completion_condition="received_devon_scope_from_steven",
                priority=1
            ),

            DirectObjective(  # 142
                id="return_to_fortree_and_reveal_kecleon",
                description="Return to Fortree City and use the Devon Scope to reveal and catch/defeat the Kecleon blocking the gym",
                action_type="battle",
                target_location="Fortree City",
                navigation_hint="Navigate back to Fortree City. Use the Devon Scope on the invisible obstacle blocking the gym entrance. A Kecleon (Lv.30) will be revealed. You can catch it or defeat it.",
                completion_condition="removed_kecleon_from_gym_entrance",
                priority=1
            ),

            DirectObjective(  # 143
                id="enter_fortree_gym",
                description="Enter the Fortree Gym to challenge Gym Leader Winona",
                action_type="navigate",
                target_location="Fortree Gym",
                navigation_hint="With the Kecleon removed, enter the Fortree Gym. The gym specializes in Flying-type Pokémon.",
                completion_condition="entered_fortree_gym",
                priority=1
            ),

            DirectObjective(  # 144
                id="navigate_to_gym_leader_winona",
                description="Navigate through the Fortree Gym turnstile puzzle to reach Gym Leader Winona",
                action_type="navigate",
                target_location="Fortree Gym",
                navigation_hint="The gym has a turnstile puzzle. Navigate through the rotating gates to reach Winona. Battle gym trainers if needed.",
                completion_condition="reached_gym_leader_winona",
                priority=1
            ),

            DirectObjective(  # 145
                id="battle_gym_leader_winona",
                description="Battle Gym Leader Winona. Her team: Swablu (Lv.29), Tropius (Lv.29), Pelipper (Lv.30), Skarmory (Lv.31), Altaria (Lv.33). Use Electric, Ice, or Rock-type moves.",
                action_type="battle",
                target_location="Fortree Gym",
                navigation_hint="Battle Winona using Electric, Ice, or Rock-type moves for super-effective damage against Flying-types. Her Altaria is Dragon/Flying, so Ice is 4x effective. Reward: Feather Badge, TM40 (Aerial Ace), enables Fly field move, Pokémon obey up to level 70. If you lose, heal and return.",
                completion_condition="defeated_winona_and_received_feather_badge",
                priority=1
            ),

            DirectObjective(  # 146
                id="exit_fortree_gym",
                description="Exit the Fortree Gym after defeating Winona",
                action_type="navigate",
                target_location="Fortree Gym",
                navigation_hint="After receiving the Feather Badge and TM40, navigate back through the gym to the exit.",
                completion_condition="exited_fortree_gym",
                priority=1
            ),

            DirectObjective(  # 147
                id="heal_after_fortree_gym",
                description="Heal your Pokémon at Fortree City Pokémon Center after the gym battle",
                action_type="interact",
                target_location="Fortree City Pokémon Center",
                navigation_hint="Return to the Pokémon Center and heal your team before continuing.",
                completion_condition="healed_after_fortree_gym",
                priority=1
            ),

            # ========== Route 121 to Mt. Pyre (Objectives 148-160) ==========
            DirectObjective(  # 148
                id="navigate_to_route_121",
                description="Navigate from Route 120 to Route 121 heading toward Lilycove City",
                action_type="navigate",
                target_location="Route 121",
                navigation_hint="From Route 120, head east to Route 121. This is a shorter route leading to Lilycove City with the Safari Zone nearby.",
                completion_condition="reached_route_121",
                priority=1
            ),

            DirectObjective(  # 149
                id="navigate_to_lilycove_city",
                description="Navigate through Route 121 to reach Lilycove City",
                action_type="navigate",
                target_location="Lilycove City",
                navigation_hint="Continue through Route 121 to reach Lilycove City. Battle trainers and collect items along the way.",
                completion_condition="reached_lilycove_city_first_time",
                priority=1
            ),

            DirectObjective(  # 150
                id="heal_at_lilycove_pokemon_center",
                description="Heal your Pokémon at Lilycove City Pokémon Center",
                action_type="interact",
                target_location="Lilycove City Pokémon Center",
                navigation_hint="Navigate to the Pokémon Center in Lilycove City and heal your team.",
                completion_condition="healed_at_lilycove_center",
                priority=1
            ),

            DirectObjective(  # 151
                id="navigate_to_route_122",
                description="Navigate south from Lilycove City to Route 122 (ocean route)",
                action_type="navigate",
                target_location="Route 122",
                navigation_hint="From Lilycove City, use Surf to head south to Route 122, a short ocean route.",
                completion_condition="reached_route_122",
                priority=1
            ),

            DirectObjective(  # 152
                id="navigate_to_mt_pyre",
                description="Navigate to Mt. Pyre from Route 122",
                action_type="navigate",
                target_location="Mt. Pyre",
                navigation_hint="From Route 122, navigate to Mt. Pyre, a sacred mountain where you'll catch up with Team Aqua.",
                completion_condition="reached_mt_pyre",
                priority=1
            ),

            DirectObjective(  # 153
                id="climb_mt_pyre_to_summit",
                description="Climb Mt. Pyre through 6 floors to reach the summit",
                action_type="navigate",
                target_location="Mt. Pyre Summit",
                navigation_hint="Enter Mt. Pyre and speak with the old woman on 1F to receive Cleanse Tag. Navigate through 6 floors, battling trainers (Poké Maniacs, Hex Maniacs, etc.). Collect TM30 (Shadow Ball) on 6F. Watch for holes that cause you to fall.",
                completion_condition="reached_mt_pyre_summit",
                priority=1
            ),

            DirectObjective(  # 154
                id="witness_team_aqua_steal_red_orb",
                description="Witness Team Aqua's Archie steal the Red Orb from Mt. Pyre summit",
                action_type="cutscene",
                target_location="Mt. Pyre Summit",
                navigation_hint="At the summit, you'll witness Archie steal the Red Orb from the altar. After he flees, an elderly caretaker will give you the Magma Emblem.",
                completion_condition="received_magma_emblem_from_caretaker",
                priority=1
            ),

            DirectObjective(  # 155
                id="navigate_to_jagged_pass_for_magma_hideout",
                description="Navigate to Jagged Pass to use the Magma Emblem and access Magma Hideout",
                action_type="navigate",
                target_location="Jagged Pass",
                navigation_hint="Use Fly to return to Lavaridge Town, then navigate to Jagged Pass. At the mountain's midway point, use the Magma Emblem to open a hidden entrance to Magma Hideout.",
                completion_condition="opened_magma_hideout_entrance",
                priority=1
            ),

            DirectObjective(  # 156
                id="navigate_through_magma_hideout",
                description="Navigate through Team Magma's hideout (B1F-B4F) battling grunts along the way",
                action_type="navigate",
                target_location="Magma Hideout",
                navigation_hint="Enter the Magma Hideout. Use Strength to move boulders on the entrance floor. Navigate through B1F-B4F, battling Team Magma Grunts. Collect Full Restore, Max Elixir, Nugget, and PP Max along the way.",
                completion_condition="navigated_to_magma_hideout_leaders",
                priority=1
            ),

            DirectObjective(  # 157
                id="battle_tabitha_in_magma_hideout",
                description="Battle Magma Admin Tabitha on B4F. His team: Numel (Lv.26), Mightyena (Lv.28), Zubat (Lv.30), Camerupt (Lv.33)",
                action_type="battle",
                target_location="Magma Hideout B4F",
                navigation_hint="Battle Tabitha using Water and Ground-type moves against his Fire-type Pokémon. Reward: $1,320.",
                completion_condition="defeated_tabitha_in_magma_hideout",
                priority=1
            ),

            DirectObjective(  # 158
                id="battle_maxie_in_magma_hideout",
                description="Battle Magma Leader Maxie on B4F. His team: Mightyena (Lv.37), Crobat (Lv.37), Camerupt (Lv.41)",
                action_type="battle",
                target_location="Magma Hideout B4F",
                navigation_hint="After defeating Tabitha, battle Maxie. Use Water and Ground-type moves effectively. Reward: $3,120.",
                completion_condition="defeated_maxie_in_magma_hideout",
                priority=1
            ),

            DirectObjective(  # 159
                id="witness_groudon_awakening_and_escape",
                description="Witness Maxie awaken Groudon with the Blue Orb, only for it to escape immediately",
                action_type="cutscene",
                target_location="Magma Hideout B4F",
                navigation_hint="After defeating Maxie, watch the cutscene where he awakens Groudon with the Blue Orb. The legendary Pokémon immediately escapes before battle.",
                completion_condition="witnessed_groudon_awakening",
                priority=1
            ),

            DirectObjective(  # 160
                id="exit_magma_hideout",
                description="Exit the Magma Hideout and heal your Pokémon",
                action_type="navigate",
                target_location="Magma Hideout",
                navigation_hint="Navigate back through the hideout to the exit. Fly to a nearby town and heal at a Pokémon Center.",
                completion_condition="exited_magma_hideout_and_healed",
                priority=1
            ),

            # ========== Slateport to Team Aqua Hideout (Objectives 161-175) ==========
            DirectObjective(  # 161
                id="navigate_to_slateport_harbor",
                description="Navigate to Slateport City harbor to witness Captain Stern and Archie's submarine theft",
                action_type="navigate",
                target_location="Slateport City Harbor",
                navigation_hint="Use Fly to reach Slateport City. Navigate to the harbor area northeast of town to witness a TV interview scene with Captain Stern. Archie will announce his plans and escape with the submarine.",
                completion_condition="witnessed_submarine_theft_slateport",
                priority=1
            ),

            DirectObjective(  # 162 - OPTIONAL
                id="optional_explore_lilycove_department_store",
                description="OPTIONAL: Visit Lilycove Department Store for items, TMs, and daily lottery",
                action_type="navigate",
                target_location="Lilycove Department Store",
                navigation_hint="The Lilycove Department Store has 6 floors with Poké Balls, healing items, vitamins, TMs (Fire Blast, Thunder, Blizzard, etc.), Secret Base decorations, and a daily lottery on 1F. On the rooftop, an NPC teaches Substitute move (one-time).",
                completion_condition="explored_lilycove_department_store",
                priority=2
            ),

            DirectObjective(  # 163 - OPTIONAL
                id="optional_battle_rival_at_department_store",
                description="OPTIONAL: Battle your rival May/Brendan outside Lilycove Department Store",
                action_type="battle",
                target_location="Lilycove City",
                navigation_hint="Your rival may challenge you outside the Department Store entrance. Their team varies based on your starter choice. Reward: $2,040.",
                completion_condition="defeated_rival_at_lilycove",
                priority=2
            ),

            DirectObjective(  # 164
                id="navigate_to_lilycove_cove_lily_motel",
                description="Navigate to Cove Lily Motel in Lilycove City and talk to Scott",
                action_type="interact",
                target_location="Cove Lily Motel",
                navigation_hint="Visit the Cove Lily Motel and talk to Scott for potential bonus Battle Points.",
                completion_condition="talked_to_scott_at_motel",
                priority=1
            ),

            DirectObjective(  # 165
                id="navigate_to_team_aqua_hideout",
                description="Navigate to Team Aqua Hideout northeast of Lilycove City",
                action_type="navigate",
                target_location="Team Aqua Hideout",
                navigation_hint="From Lilycove City, use Surf to navigate northeast to find the Team Aqua Hideout entrance. Your goal is to stop Archie and retrieve the submarine.",
                completion_condition="reached_team_aqua_hideout_entrance",
                priority=1
            ),

            DirectObjective(  # 166
                id="navigate_through_team_aqua_hideout",
                description="Navigate through Team Aqua Hideout using warp panels to reach Archie's office",
                action_type="navigate",
                target_location="Team Aqua Hideout",
                navigation_hint="Battle the Grunt with Poochyena on 1F. Navigate B1F using warp puzzle sequence. In Archie's office, collect Master Ball, Nugget, Max Elixir, and Nest Ball (two items are Electrode disguises - be careful!).",
                completion_condition="navigated_to_aqua_hideout_submarine_room",
                priority=1
            ),

            DirectObjective(  # 167
                id="battle_aqua_admin_matt",
                description="Battle Aqua Admin Matt before the submarine room. His team: Mightyena (Lv.34), Golbat (Lv.34)",
                action_type="battle",
                target_location="Team Aqua Hideout",
                navigation_hint="Navigate through warp panels to reach the submarine room. Before entering, battle Admin Matt. Use Fighting and Electric-type moves. Reward: $1,360.",
                completion_condition="defeated_matt_in_aqua_hideout",
                priority=1
            ),

            DirectObjective(  # 168
                id="witness_archie_escape_in_submarine",
                description="Witness Archie escape in the submarine before you can stop him",
                action_type="cutscene",
                target_location="Team Aqua Hideout",
                navigation_hint="After defeating Matt, you'll witness Archie escape in the submarine. You must pursue him to Seafloor Cavern by ocean route.",
                completion_condition="witnessed_archie_submarine_escape",
                priority=1
            ),

            DirectObjective(  # 169
                id="exit_team_aqua_hideout",
                description="Exit Team Aqua Hideout and heal your Pokémon",
                action_type="navigate",
                target_location="Team Aqua Hideout",
                navigation_hint="Navigate back through the hideout to the exit. Fly to a nearby city and heal at a Pokémon Center.",
                completion_condition="exited_aqua_hideout_and_healed",
                priority=1
            ),

            # ========== Route 124 to Mossdeep Gym (Objectives 170-185) ==========
            DirectObjective(  # 170
                id="navigate_to_route_124",
                description="Navigate to Route 124 east from Lilycove City using Surf",
                action_type="navigate",
                target_location="Route 124",
                navigation_hint="From Lilycove City, use Surf to head east to Route 124, a water route with multiple trainers. You can trade colored shards for evolution stones at the Treasure Hunter's House.",
                completion_condition="reached_route_124",
                priority=1
            ),

            DirectObjective(  # 171 - OPTIONAL
                id="optional_visit_treasure_hunter_for_stones",
                description="OPTIONAL: Trade colored shards for evolution stones at Treasure Hunter's House on Route 124",
                action_type="interact",
                target_location="Route 124 - Treasure Hunter's House",
                navigation_hint="Trade Red Shard for Fire Stone, Yellow Shard for Thunder Stone, Blue Shard for Water Stone, or Green Shard for Leaf Stone.",
                completion_condition="visited_treasure_hunter",
                priority=2
            ),

            DirectObjective(  # 172
                id="navigate_to_mossdeep_city",
                description="Navigate from Route 124 to Mossdeep City",
                action_type="navigate",
                target_location="Mossdeep City",
                navigation_hint="From Route 124, head east to Mossdeep City, an island location with mild winds. Home to the Space Center and seventh Gym.",
                completion_condition="reached_mossdeep_city",
                priority=1
            ),

            DirectObjective(  # 173
                id="heal_at_mossdeep_pokemon_center",
                description="Heal your Pokémon at Mossdeep City Pokémon Center",
                action_type="interact",
                target_location="Mossdeep City Pokémon Center",
                navigation_hint="Navigate to the Pokémon Center and heal your team before challenging the gym.",
                completion_condition="healed_at_mossdeep_center",
                priority=1
            ),

            DirectObjective(  # 174 - OPTIONAL
                id="optional_obtain_super_rod_and_items",
                description="OPTIONAL: Obtain Super Rod and collect items in Mossdeep City",
                action_type="interact",
                target_location="Mossdeep City",
                navigation_hint="Get Super Rod from fisherman east of Gym. Collect Net Ball (southeast corner), King's Rock (from boy near Steven's house), Sun Stone (sailor at Space Center). Black Belt teaches DynamicPunch (one-time). Complete Wingull errand for Mental Herb.",
                completion_condition="obtained_mossdeep_items",
                priority=2
            ),

            DirectObjective(  # 175
                id="enter_mossdeep_gym",
                description="Enter the Mossdeep Gym to challenge Gym Leaders Liza & Tate",
                action_type="navigate",
                target_location="Mossdeep Gym",
                navigation_hint="Navigate to the Mossdeep Gym. The gym specializes in Psychic-type Pokémon and features dual Gym Leaders.",
                completion_condition="entered_mossdeep_gym",
                priority=1
            ),

            DirectObjective(  # 176
                id="navigate_to_gym_leaders_liza_and_tate",
                description="Navigate through the Mossdeep Gym puzzle (warp panels and spin tiles) to reach Liza & Tate",
                action_type="navigate",
                target_location="Mossdeep Gym",
                navigation_hint="The gym has interconnected rooms with warp panels and spin tiles. Defeat Psychic and Gentleman trainers along the way to reach the Gym Leaders.",
                completion_condition="reached_gym_leaders_liza_and_tate",
                priority=1
            ),

            DirectObjective(  # 177
                id="battle_gym_leaders_liza_and_tate",
                description="Battle Gym Leaders Liza & Tate in a Double Battle. Their team: Claydol (Lv.41), Xatu (Lv.41), Lunatone (Lv.42), Solrock (Lv.42). Use Dark or Ghost-type moves.",
                action_type="battle",
                target_location="Mossdeep Gym",
                navigation_hint="This is a Double Battle against two leaders simultaneously. Use Dark and Ghost-type moves for super-effective damage against Psychic-types. Water moves can hit multiple targets. Reward: Mind Badge, TM04 (Calm Mind). If you lose, heal and return.",
                completion_condition="defeated_liza_and_tate_received_mind_badge",
                priority=1
            ),

            DirectObjective(  # 178
                id="exit_mossdeep_gym",
                description="Exit the Mossdeep Gym after defeating Liza & Tate",
                action_type="navigate",
                target_location="Mossdeep Gym",
                navigation_hint="After receiving the Mind Badge and TM04, navigate back through the gym to the exit.",
                completion_condition="exited_mossdeep_gym",
                priority=1
            ),

            DirectObjective(  # 179
                id="navigate_to_mossdeep_space_center",
                description="Navigate to the Mossdeep Space Center for the Team Magma event",
                action_type="navigate",
                target_location="Mossdeep Space Center",
                navigation_hint="After the gym victory, navigate to the Space Center. Team Magma is attempting to steal rocket fuel.",
                completion_condition="reached_space_center_for_team_magma",
                priority=1
            ),

            DirectObjective(  # 180
                id="battle_team_magma_at_space_center",
                description="Battle Team Magma Grunts at the Space Center (7 grunts total - 4 on first floor, 3 upstairs)",
                action_type="battle",
                target_location="Mossdeep Space Center",
                navigation_hint="Defeat four grunts on the first floor, then three more upstairs. These are consecutive battles with no breaks, so use healing items as needed.",
                completion_condition="defeated_space_center_grunts",
                priority=1
            ),

            DirectObjective(  # 181
                id="double_battle_maxie_and_tabitha_with_steven",
                description="Team up with Steven in a Double Battle against Maxie and Tabitha",
                action_type="battle",
                target_location="Mossdeep Space Center",
                navigation_hint="Steven will team with you in a Double Battle. Steven uses Metang (Lv.42), Skarmory (Lv.43), and Aggron (Lv.44). Focus on Maxie and Tabitha's Pokémon. Maxie will abandon his plans after defeat.",
                completion_condition="defeated_maxie_and_tabitha_with_steven",
                priority=1
            ),

            DirectObjective(  # 182
                id="receive_hm08_dive_from_steven",
                description="Receive HM08 (Dive) from Steven after defeating Team Magma",
                action_type="interact",
                target_location="Mossdeep Space Center",
                navigation_hint="After the battle, Steven will reward you with HM08 (Dive), which enables exploration of underwater areas.",
                completion_condition="received_hm08_dive",
                priority=1
            ),

            DirectObjective(  # 183
                id="heal_after_space_center_battles",
                description="Heal your Pokémon at Mossdeep City Pokémon Center after the Space Center battles",
                action_type="interact",
                target_location="Mossdeep City Pokémon Center",
                navigation_hint="Return to the Pokémon Center and heal your team. You're now ready to explore underwater areas with Dive.",
                completion_condition="healed_after_space_center",
                priority=1
            ),

            # ========== Route 127 to Seafloor Cavern (Objectives 184-200) ==========
            DirectObjective(  # 184
                id="navigate_to_route_127",
                description="Navigate from Mossdeep City to Route 127 using Surf",
                action_type="navigate",
                target_location="Route 127",
                navigation_hint="From Mossdeep City, use Surf to navigate to Route 127. Battle trainers and collect items (Zinc, Rare Candy, Carbos on surface; underwater items with Dive).",
                completion_condition="reached_route_127",
                priority=1
            ),

            DirectObjective(  # 185
                id="navigate_to_route_128",
                description="Navigate from Route 127 to Route 128",
                action_type="navigate",
                target_location="Route 128",
                navigation_hint="Continue navigating to Route 128, a coastal region with trainers and underwater treasures (Heart Scales, Protein, Pearl).",
                completion_condition="reached_route_128",
                priority=1
            ),

            DirectObjective(  # 186
                id="navigate_to_seafloor_cavern_entrance",
                description="Navigate to the underwater cavern entrance beneath Route 128 using Dive",
                action_type="navigate",
                target_location="Seafloor Cavern Entrance",
                navigation_hint="Use Dive to descend underwater on Route 128. Find the cavern entrance that leads to Seafloor Cavern. This is where Team Aqua is attempting to awaken Kyogre.",
                completion_condition="reached_seafloor_cavern_entrance",
                priority=1
            ),

            DirectObjective(  # 187
                id="navigate_through_seafloor_cavern",
                description="Navigate through Seafloor Cavern using Rock Smash, Strength, and Surf",
                action_type="navigate",
                target_location="Seafloor Cavern",
                navigation_hint="Room 1: Clear boulders with Rock Smash and Strength. Rooms 5-6: Navigate water currents while surfing. Room 8: Solve boulder puzzle to advance. Battle Team Aqua Grunts along the way.",
                completion_condition="navigated_to_seafloor_cavern_depths",
                priority=1
            ),

            DirectObjective(  # 188
                id="battle_aqua_admin_shelly_seafloor",
                description="Battle Aqua Admin Shelly in Room 7. Her team: Sharpedo (Lv.37), Mightyena (Lv.37)",
                action_type="battle",
                target_location="Seafloor Cavern",
                navigation_hint="Battle Shelly using Electric and Grass-type moves against her Water and Dark-type Pokémon.",
                completion_condition="defeated_shelly_in_seafloor_cavern",
                priority=1
            ),

            DirectObjective(  # 189
                id="battle_archie_at_seafloor_cavern",
                description="Battle Team Aqua Leader Archie in the deepest chamber. His team: Mightyena (Lv.41), Crobat (Lv.41), Sharpedo (Lv.43)",
                action_type="battle",
                target_location="Seafloor Cavern",
                navigation_hint="Battle Archie using Electric, Grass, and Fighting-type moves. His Sharpedo is Water/Dark type. Reward: TM26 (Earthquake) from the deepest chamber.",
                completion_condition="defeated_archie_at_seafloor_cavern",
                priority=1
            ),

            DirectObjective(  # 190
                id="witness_kyogre_awakening_and_escape",
                description="Witness Kyogre awaken and flee, causing extreme weather across eastern Hoenn",
                action_type="cutscene",
                target_location="Seafloor Cavern",
                navigation_hint="After defeating Archie, Kyogre awakens and flees, causing extreme weather. Maxie and Team Magma arrive to discuss intervention. Steven appears and directs you toward Sootopolis City.",
                completion_condition="witnessed_kyogre_awakening",
                priority=1
            ),

            DirectObjective(  # 191
                id="exit_seafloor_cavern",
                description="Exit Seafloor Cavern and heal your Pokémon",
                action_type="navigate",
                target_location="Seafloor Cavern",
                navigation_hint="Navigate back through Seafloor Cavern to the exit. Use Dive to surface, then Fly to a nearby city to heal.",
                completion_condition="exited_seafloor_cavern_and_healed",
                priority=1
            ),

            # ========== Sootopolis City to Sky Pillar (Objectives 192-210) ==========
            DirectObjective(  # 192
                id="navigate_to_route_126_and_sootopolis",
                description="Navigate to Route 126 and find Sootopolis City in the volcanic crater",
                action_type="navigate",
                target_location="Sootopolis City",
                navigation_hint="Navigate to Route 126, an underwater route surrounding a volcanic crater. Use Dive to descend, then find the entrance to Sootopolis City. The city is only accessible by sea or air.",
                completion_condition="reached_sootopolis_city_first_visit",
                priority=1
            ),

            DirectObjective(  # 193
                id="witness_groudon_kyogre_battle",
                description="Witness Groudon and Kyogre locked in battle at Sootopolis City",
                action_type="cutscene",
                target_location="Sootopolis City",
                navigation_hint="Upon arriving at Sootopolis, you'll witness the climactic battle between Groudon and Kyogre. The weather emergency must be resolved before challenging the gym.",
                completion_condition="witnessed_groudon_kyogre_battle",
                priority=1
            ),

            DirectObjective(  # 194
                id="navigate_to_cave_of_origin",
                description="Navigate to the Cave of Origin in Sootopolis City",
                action_type="navigate",
                target_location="Cave of Origin",
                navigation_hint="Navigate through Sootopolis City to find the Cave of Origin entrance. Steven should guide you there.",
                completion_condition="reached_cave_of_origin",
                priority=1
            ),

            DirectObjective(  # 195
                id="find_wallace_in_cave_of_origin",
                description="Navigate through Cave of Origin's three levels to find Wallace on B1F",
                action_type="navigate",
                target_location="Cave of Origin B1F",
                navigation_hint="Navigate through the Cave of Origin dungeon to the lowest floor (B1F) where Wallace is waiting. Talk to him to learn about Rayquaza.",
                completion_condition="found_wallace_in_cave_of_origin",
                priority=1
            ),

            DirectObjective(  # 196
                id="learn_about_rayquaza_from_wallace",
                description="Talk to Wallace to learn about Rayquaza at Sky Pillar",
                action_type="interact",
                target_location="Cave of Origin B1F",
                navigation_hint="Wallace will tell you about a third super-ancient Pokémon named Rayquaza located at Sky Pillar that can stop the conflict.",
                completion_condition="learned_about_rayquaza_from_wallace",
                priority=1
            ),

            DirectObjective(  # 197
                id="navigate_to_route_129",
                description="Navigate to Route 129 heading toward Pacifidlog Town",
                action_type="navigate",
                target_location="Route 129",
                navigation_hint="Exit Cave of Origin and Sootopolis City. Navigate west to Route 129, battling trainers along the way.",
                completion_condition="reached_route_129",
                priority=1
            ),

            DirectObjective(  # 198
                id="navigate_through_routes_130_and_131",
                description="Navigate through Routes 130 and 131 toward Sky Pillar",
                action_type="navigate",
                target_location="Route 131",
                navigation_hint="Continue through Route 130 (which occasionally displays Mirage Island) and Route 131. Battle trainers and collect items along the way.",
                completion_condition="reached_route_131",
                priority=1
            ),

            DirectObjective(  # 199 - OPTIONAL
                id="optional_visit_pacifidlog_town",
                description="OPTIONAL: Visit Pacifidlog Town for trades, move tutoring, and friendship evaluations",
                action_type="navigate",
                target_location="Pacifidlog Town",
                navigation_hint="Pacifidlog Town is a floating village on Route 131. Visit for side activities, NPC trades, and move tutors.",
                completion_condition="visited_pacifidlog_town",
                priority=2
            ),

            DirectObjective(  # 200
                id="navigate_to_sky_pillar",
                description="Navigate to Sky Pillar on northern Route 131",
                action_type="navigate",
                target_location="Sky Pillar",
                navigation_hint="From Route 131, navigate north to find Sky Pillar, a 5-floor tower with cracked tiles on 4F.",
                completion_condition="reached_sky_pillar",
                priority=1
            ),

            DirectObjective(  # 201
                id="climb_sky_pillar_to_awaken_rayquaza",
                description="Climb Sky Pillar to the rooftop and awaken Rayquaza",
                action_type="navigate",
                target_location="Sky Pillar Rooftop",
                navigation_hint="Navigate through 5 floors of Sky Pillar. Watch for cracked tiles on 4F that cause falls to 3F. Wild Pokémon include Golbat, Sableye, Claydol, Banette, and Altaria. Climb to the rooftop to find Rayquaza.",
                completion_condition="reached_sky_pillar_rooftop",
                priority=1
            ),

            DirectObjective(  # 202
                id="awaken_rayquaza",
                description="Interact with Rayquaza to awaken it. It will fly away to Sootopolis City.",
                action_type="interact",
                target_location="Sky Pillar Rooftop",
                navigation_hint="Approach Rayquaza on the rooftop and interact with it. The legendary dragon will awaken and fly away to Sootopolis City to stop the battle.",
                completion_condition="awakened_rayquaza",
                priority=1
            ),

            DirectObjective(  # 203
                id="return_to_sootopolis_after_rayquaza",
                description="Return to Sootopolis City to witness Rayquaza stop the Groudon/Kyogre battle",
                action_type="navigate",
                target_location="Sootopolis City",
                navigation_hint="Use Fly or navigate back to Sootopolis City. Witness Rayquaza descend and stop the Groudon/Kyogre battle. The weather normalizes.",
                completion_condition="witnessed_rayquaza_stop_battle",
                priority=1
            ),

            DirectObjective(  # 204
                id="receive_hm07_waterfall_from_wallace",
                description="Receive HM07 (Waterfall) from Wallace near Sootopolis Gym",
                action_type="interact",
                target_location="Sootopolis City",
                navigation_hint="After the legendary Pokémon event, talk to Wallace near the Gym. He will give you HM07 (Waterfall).",
                completion_condition="received_hm07_waterfall",
                priority=1
            ),

            DirectObjective(  # 205
                id="heal_at_sootopolis_pokemon_center",
                description="Heal your Pokémon at Sootopolis City Pokémon Center before the gym",
                action_type="interact",
                target_location="Sootopolis City Pokémon Center",
                navigation_hint="Navigate to the Pokémon Center and heal your team before challenging the final gym.",
                completion_condition="healed_at_sootopolis_center",
                priority=1
            ),

            # ========== Sootopolis Gym (Objectives 206-215) ==========
            DirectObjective(  # 206 - OPTIONAL
                id="optional_collect_sootopolis_items",
                description="OPTIONAL: Collect items in Sootopolis City (TM31, Wailmer Doll, Elixirs, berries)",
                action_type="interact",
                target_location="Sootopolis City",
                navigation_hint="Collect TM31 (Brick Break) from Black Belt in northwest house, Wailmer Doll from girl east of Pokémon Center, Elixir ×2 from competitive brothers (show large Seedot or Lotad). Pokémon Center girl teaches Double-Edge (one-time).",
                completion_condition="collected_sootopolis_items",
                priority=2
            ),

            DirectObjective(  # 207
                id="enter_sootopolis_gym",
                description="Enter the Sootopolis Gym to challenge Gym Leader Juan",
                action_type="navigate",
                target_location="Sootopolis Gym",
                navigation_hint="Navigate to the Sootopolis Gym. The gym specializes in Water-type Pokémon and features an ice puzzle.",
                completion_condition="entered_sootopolis_gym",
                priority=1
            ),

            DirectObjective(  # 208
                id="navigate_ice_puzzle_to_juan",
                description="Navigate through the Sootopolis Gym ice-tile puzzle to reach Gym Leader Juan",
                action_type="navigate",
                target_location="Sootopolis Gym",
                navigation_hint="The gym has three separate ice-tile grids with specific movement patterns. Navigate carefully across the slippery tiles. Battle Beauty, Lass, Lady, and Pokéfan trainers with Water-type Pokémon along the way.",
                completion_condition="reached_gym_leader_juan",
                priority=1
            ),

            DirectObjective(  # 209
                id="battle_gym_leader_juan",
                description="Battle Gym Leader Juan. His team: Luvdisc (Lv.41), Whiscash (Lv.41), Sealeo (Lv.43), Crawdaunt (Lv.43), Kingdra (Lv.46). Use Grass and Electric-type moves.",
                action_type="battle",
                target_location="Sootopolis Gym",
                navigation_hint="Battle Juan using Grass and Electric-type moves. His Kingdra (Water/Dragon, Lv.46) is the final threat with no Electric weakness. Whiscash is Water/Ground (weak only to Grass). Crawdaunt is Water/Dark. Reward: Rain Badge, TM03 (Water Pulse), PokéNav registration, enables Waterfall field move. If you lose, heal and return.",
                completion_condition="defeated_juan_and_received_rain_badge",
                priority=1
            ),

            DirectObjective(  # 210
                id="exit_sootopolis_gym",
                description="Exit the Sootopolis Gym after defeating Juan and obtaining all eight badges",
                action_type="navigate",
                target_location="Sootopolis Gym",
                navigation_hint="After receiving the Rain Badge and TM03, navigate back through the gym to the exit. You now have all eight Gym Badges!",
                completion_condition="exited_sootopolis_gym_with_all_badges",
                priority=1
            ),

            DirectObjective(  # 211
                id="heal_after_sootopolis_gym",
                description="Heal your Pokémon after obtaining all eight badges",
                action_type="interact",
                target_location="Sootopolis City Pokémon Center",
                navigation_hint="Return to the Pokémon Center and heal your team. You're now ready to head to the Pokémon League!",
                completion_condition="healed_after_all_eight_badges",
                priority=1
            ),

            # ========== Victory Road (Objectives 212-225) ==========
            DirectObjective(  # 212 - OPTIONAL
                id="optional_catch_rayquaza_at_sky_pillar",
                description="OPTIONAL: Return to Sky Pillar to catch Rayquaza (Level 70) with Mach Bike",
                action_type="battle",
                target_location="Sky Pillar",
                navigation_hint="With the Mach Bike, navigate floors 1F-5F crossing unstable floor patches. On 4F, intentionally fall through cracked tiles to access previously unreachable areas. Climb to 6F where Rayquaza awaits at level 70. Consider using Master Ball. Ice-type moves are super-effective.",
                completion_condition="caught_rayquaza_at_sky_pillar",
                priority=2
            ),

            DirectObjective(  # 213 - OPTIONAL
                id="optional_catch_legendary_golems",
                description="OPTIONAL: Catch Regirock, Regice, and Registeel using the Sealed Chamber puzzle",
                action_type="navigate",
                target_location="Sealed Chamber",
                navigation_hint="Navigate to Sealed Chamber on Route 134 (underwater, use Dive). Use Dig on north wall in first chamber. Arrange Wailord (first) and Relicanth (last) in party and examine back room wall to unlock three ruins: Desert Ruins/Route 111 (Regirock Lv.40), Island Cave/Route 105 (Regice Lv.40), Ancient Tomb/Route 120 (Registeel Lv.40).",
                completion_condition="completed_regi_quest",
                priority=2
            ),

            DirectObjective(  # 214
                id="navigate_to_ever_grande_city_south",
                description="Navigate to Ever Grande City (south) from Mossdeep City",
                action_type="navigate",
                target_location="Ever Grande City",
                navigation_hint="Use Fly to Mossdeep City or Surf/Waterfall. Sail southward from Mossdeep, then eastward through Route 128's narrow channel to reach Ever Grande City with all eight badges.",
                completion_condition="reached_ever_grande_city_south",
                priority=1
            ),

            DirectObjective(  # 215
                id="heal_and_stock_up_at_ever_grande",
                description="Visit Pokémon Center and Poké Mart at Ever Grande City to prepare for Victory Road and Pokémon League",
                action_type="interact",
                target_location="Ever Grande City Pokémon Center",
                navigation_hint="Heal your Pokémon at the Center. Stock up on healing items (Hyper Potions, Full Restores, Full Heals, Revives) and Poké Balls at the Mart. Talk to Scott before continuing.",
                completion_condition="prepared_at_ever_grande_city",
                priority=1
            ),

            DirectObjective(  # 216
                id="enter_victory_road",
                description="Enter Victory Road, the final challenge before the Pokémon League",
                action_type="navigate",
                target_location="Victory Road",
                navigation_hint="From Ever Grande City south, navigate to Victory Road entrance. You'll need Surf, Strength, Rock Smash, and Waterfall to navigate through.",
                completion_condition="entered_victory_road",
                priority=1
            ),

            DirectObjective(  # 217
                id="navigate_victory_road_1f",
                description="Navigate Victory Road 1F, battling Cooltrainers and encountering Wally",
                action_type="navigate",
                target_location="Victory Road 1F",
                navigation_hint="Battle Cooltrainer Albert on 1F. You'll encounter Wally here who challenges you to a battle.",
                completion_condition="navigated_victory_road_1f",
                priority=1
            ),

            DirectObjective(  # 218
                id="battle_wally_in_victory_road",
                description="Battle your rival Wally. His team: Altaria (Lv.44), Delcatty (Lv.43), Roselia (Lv.44), Magneton (Lv.41), Gardevoir (Lv.45). Use Ice and Dark-type moves.",
                action_type="battle",
                target_location="Victory Road 1F",
                navigation_hint="Wally's team has grown significantly stronger. His Gardevoir (Psychic/Fairy, Lv.45) is his ace. Ice-type moves work well against Altaria. Dark-type moves effective against Gardevoir. Use Electric against Altaria.",
                completion_condition="defeated_wally_in_victory_road",
                priority=1
            ),

            DirectObjective(  # 219
                id="navigate_victory_road_b1f",
                description="Navigate Victory Road B1F using Strength to move boulders",
                action_type="navigate",
                target_location="Victory Road B1F",
                navigation_hint="Use Strength to move boulders and create paths. Battle Cooltrainers Shannon and Samuel. Collect items along the way.",
                completion_condition="navigated_victory_road_b1f",
                priority=1
            ),

            DirectObjective(  # 220
                id="navigate_victory_road_b2f",
                description="Navigate Victory Road B2F with waterfalls and bridges",
                action_type="navigate",
                target_location="Victory Road B2F",
                navigation_hint="Navigate multiple bridges and waterfalls. Face Cooltrainers Julie, Owen, Caroline, and Vito. Use Waterfall to climb waterfalls. Collect Max Elixir, PP Up, Ultra Ball, Full Restore, TM29 (Psychic), Max Repel, Full Heal, Elixir.",
                completion_condition="navigated_victory_road_b2f",
                priority=1
            ),

            DirectObjective(  # 221
                id="exit_victory_road_to_ever_grande_north",
                description="Exit Victory Road to reach Ever Grande City (north)",
                action_type="navigate",
                target_location="Ever Grande City North",
                navigation_hint="Complete the final sections of Victory Road (battle Cooltrainer Edgar) and exit to the northern part of Ever Grande City. Follow the stone walkway to the Pokémon League entrance.",
                completion_condition="reached_ever_grande_city_north",
                priority=1
            ),

            DirectObjective(  # 222
                id="heal_before_elite_four",
                description="Heal your Pokémon at the Pokémon League Pokémon Center before challenging the Elite Four",
                action_type="interact",
                target_location="Pokémon League Pokémon Center",
                navigation_hint="This is your last chance to heal before the Elite Four gauntlet. Visit the Pokémon Center in the lobby. Stock up on healing items at the Poké Mart (Hyper Potions, Full Restores, Full Heals). Consider purchasing stat-boosting battle items from Lilycove Department Store if you haven't already.",
                completion_condition="healed_before_elite_four",
                priority=1
            ),

            # ========== Elite Four and Champion (Objectives 223-232) ==========
            DirectObjective(  # 223
                id="battle_elite_four_sidney",
                description="Battle Elite Four Sidney (Dark-type Master). His team: Mightyena (Lv.46), Shiftry (Lv.48), Cacturne (Lv.46), Crawdaunt (Lv.48), Absol (Lv.49). Use Bug and Fighting-type moves.",
                action_type="battle",
                target_location="Elite Four - Sidney's Chamber",
                navigation_hint="Enter the first chamber and battle Sidney. Use Bug and Fighting-type moves for super-effective damage. Psychic attacks won't work on Dark-types. Reward: $4,900. You cannot leave or heal between Elite Four battles!",
                completion_condition="defeated_elite_four_sidney",
                priority=1
            ),

            DirectObjective(  # 224
                id="battle_elite_four_phoebe",
                description="Battle Elite Four Phoebe (Ghost-type Master). Her team: Dusclops (Lv.48), Banette (Lv.49), Sableye (Lv.50), Banette (Lv.49), Dusclops (Lv.51). Use Ghost and Dark-type moves.",
                action_type="battle",
                target_location="Elite Four - Phoebe's Chamber",
                navigation_hint="Battle Phoebe using Ghost and Dark-type attacks for super-effective damage. Normal and Fighting-type moves won't work. Reward: $5,100.",
                completion_condition="defeated_elite_four_phoebe",
                priority=1
            ),

            DirectObjective(  # 225
                id="battle_elite_four_glacia",
                description="Battle Elite Four Glacia (Ice-type Master). Her team: Sealeo (Lv.50), Glalie (Lv.50), Sealeo (Lv.52), Glalie (Lv.52), Walrein (Lv.53). Use Fire, Fighting, Rock, or Steel-type moves.",
                action_type="battle",
                target_location="Elite Four - Glacia's Chamber",
                navigation_hint="Battle Glacia using Fire, Fighting, Rock, or Steel-type moves. Electric and Grass-type moves are also effective. Reward: $5,300.",
                completion_condition="defeated_elite_four_glacia",
                priority=1
            ),

            DirectObjective(  # 226
                id="battle_elite_four_drake",
                description="Battle Elite Four Drake (Dragon-type Master). His team: Shelgon (Lv.52), Altaria (Lv.54), Kingdra (Lv.53), Flygon (Lv.53), Salamence (Lv.55). Use Dragon and Ice-type moves.",
                action_type="battle",
                target_location="Elite Four - Drake's Chamber",
                navigation_hint="Battle Drake using Dragon-type moves (work against all his Pokémon) or Ice-type moves (super-effective against most). Be careful with Kingdra (Water/Dragon) which resists Ice. Reward: $5,500.",
                completion_condition="defeated_elite_four_drake",
                priority=1
            ),

            DirectObjective(  # 227
                id="save_before_champion_wallace",
                description="Save your game before entering the Champion's chamber",
                action_type="interact",
                target_location="Before Champion Chamber",
                navigation_hint="After defeating all four Elite Four members, you're at the entrance to the Champion's chamber. SAVE YOUR GAME before proceeding. This is a crucial save point!",
                completion_condition="saved_before_champion",
                priority=1
            ),

            DirectObjective(  # 228
                id="battle_champion_wallace",
                description="Battle Champion Wallace (Water-type focus). His team: Wailord (Lv.57), Tentacruel (Lv.55), Ludicolo (Lv.56), Whiscash (Lv.56), Gyarados (Lv.56), Milotic (Lv.58). Use Grass and Electric-type moves.",
                action_type="battle",
                target_location="Champion Chamber",
                navigation_hint="Battle Wallace, the final opponent. Use Grass/Electric vs Wailord. Electric/Ground/Psychic vs Tentacruel. Flying/Poison/Bug vs Ludicolo. Grass (4x) vs Whiscash. Electric/Rock vs Gyarados. Electric/Grass vs Milotic (his ace with high defenses and healing). Reward: $11,600 and Champion title!",
                completion_condition="defeated_champion_wallace",
                priority=1
            ),

            DirectObjective(  # 229
                id="enter_hall_of_fame",
                description="Follow Wallace to the Hall of Fame and record your champions",
                action_type="cutscene",
                target_location="Hall of Fame",
                navigation_hint="After victory, Wallace leads you to the Hall of Fame. Place your Poké Balls in the machine to save your champions' data for posterity. The game automatically saves upon completion. Congratulations, Champion!",
                completion_condition="entered_hall_of_fame",
                priority=1
            ),

            DirectObjective(  # 230
                id="watch_credits_and_complete_game",
                description="Watch the ending credits and complete the main story of Pokémon Emerald",
                action_type="cutscene",
                target_location="Credits",
                navigation_hint="Enjoy the ending credits. The main story of Pokémon Emerald is complete! Post-game content and Battle Frontier will become available after the credits.",
                completion_condition="completed_main_story",
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
        
    def load_part_1_walkthrough_claude_4_5_sequence(self, start_index: int = 0, run_dir: Optional[str] = None):
        """Load the Part 1 walkthrough sequence from game start through Route 104.
        
        This comprehensive sequence covers the entire Part 1 of Pokemon Emerald,
        from character creation through reaching Route 104.
        """
        self.sequence_name = "part_1_walkthrough_claude_4_5"
        self.current_sequence = [
            DirectObjective(
                id="intro_01_character_creation",
                description="Watch Professor Birch's introduction and select your character's gender and name (up to 7 characters)",
                action_type="select",
                target_location="Title Screen",
                navigation_hint="Use directional buttons to select gender and input name when prompted by Professor Birch",
                completion_condition="screen_transitions_to_moving_truck_interior",
                priority=1
            ),
            
            DirectObjective(
                id="home_01_exit_truck",
                description="Exit the moving truck and enter Littleroot Town. Mom will greet you and take you inside your new home",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="Walk right to the door to exit the truck. Mom will automatically greet you outside and lead you into the house",
                completion_condition="player_enters_house_first_floor",
                priority=1
            ),
            
            DirectObjective(
                id="home_02_go_to_bedroom",
                description="Navigate upstairs to your bedroom and set the clock on the wall",
                action_type="interact",
                target_location="Player's Bedroom",
                navigation_hint="Walk north to the stairs at position (8, 2) and go upstairs. Interact with the clock at position (5, 1) by standing at (5, 2) and facing UP. Press A to set the time",
                completion_condition="clock_setting_screen_closes_and_player_control_returns",
                priority=1
            ),
            
            DirectObjective(
                id="home_03_watch_tv",
                description="Return downstairs and watch the TV segment with Mom about Petalburg Gym where your father Norman is the new Gym Leader",
                action_type="dialogue",
                target_location="Player's House 1F",
                navigation_hint="Go back downstairs and talk to Mom. She'll call you over to watch a TV program about your dad Norman at Petalburg Gym",
                completion_condition="tv_segment_dialogue_ends_and_mom_suggests_visiting_birch",
                priority=1
            ),
            
            DirectObjective(
                id="littleroot_01_visit_birch_house",
                description="Visit Professor Birch's house next door, talk to his wife downstairs, then go upstairs to meet May/Brendan. Inspect the Poké Ball on the floor to trigger their appearance",
                action_type="navigate",
                target_location="May/Brendan's Bedroom",
                navigation_hint="Exit your house and enter the adjacent house. Talk to Birch's wife on the first floor, then go upstairs. The room appears empty - inspect the Poké Ball on the floor. May/Brendan will enter and introduce themselves before leaving",
                completion_condition="may_brendan_exits_room_after_dialogue_about_helping_birch",
                priority=1
            ),
            
            DirectObjective(
                id="route101_01_save_birch",
                description="Travel to Route 101 north of Littleroot Town and find Professor Birch being chased by a wild Zigzagoon",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Exit Littleroot Town heading north to Route 101. Continue north until you find Professor Birch being chased. The cutscene will automatically trigger",
                completion_condition="cutscene_starts_showing_birch_being_chased",
                priority=1
            ),
            
            DirectObjective(
                id="route101_02_choose_starter",
                description="Choose your starter Pokémon from Birch's Bag: Treecko (Grass), Torchic (Fire), or Mudkip (Water)",
                action_type="select",
                target_location="Route 101",
                navigation_hint="Birch will ask you to choose a Pokémon from his Bag. Open the Bag and select one starter: Treecko (strong vs Water/Rock/Ground, weak to Fire/Bug/Poison/Flying/Ice), Torchic (strong vs Grass/Bug/Ice/Steel, weak to Water/Ground), or Mudkip (strong vs Fire/Rock/Ground, weak to Grass/Electric)",
                completion_condition="starter_pokemon_selection_confirmed_and_battle_begins",
                priority=1
            ),
            
            DirectObjective(
                id="route101_03_defeat_zigzagoon",
                description="Battle and defeat the wild Level 2 Zigzagoon attacking Professor Birch using your newly chosen starter",
                action_type="battle",
                target_location="Route 101",
                navigation_hint="Use your starter's basic attack move to defeat the Level 2 Zigzagoon. This should be an easy first battle",
                completion_condition="battle_ends_and_birch_thanks_you_invites_to_lab",
                priority=1
            ),
            
            DirectObjective(
                id="lab_01_receive_starter",
                description="Return to Professor Birch's Lab in Littleroot Town where he officially gives you the starter Pokémon and directs you to find May/Brendan on Route 103",
                action_type="dialogue",
                target_location="Professor Birch's Lab",
                navigation_hint="Walk south back to Littleroot Town and enter the Lab (southern building). Birch will thank you and let you keep the starter. He'll encourage you to find May/Brendan on Route 103 for training tips",
                completion_condition="dialogue_ends_and_birch_mentions_route_103",
                priority=1
            ),
            
            DirectObjective(
                id="oldale_01_travel_and_explore",
                description="Travel north through Route 101 to reach Oldale Town. Speak to the Poké Mart worker near the southeast house to receive a free Potion",
                action_type="navigate",
                target_location="Oldale Town",
                navigation_hint="Walk north through tall grass on Route 101 (wild Pokémon appear but can't be caught yet - you have no Poké Balls). In Oldale, talk to the woman near the southeast house. She'll show you the Poké Mart and give you a free Potion as part of a promotion",
                completion_condition="received_potion_item_and_dialogue_ends",
                priority=1
            ),
            
            DirectObjective(
                id="route103_01_rival_battle",
                description="Travel to Route 103 north of Oldale Town, walk through the tall grass, and battle your rival May/Brendan",
                action_type="battle",
                target_location="Route 103",
                navigation_hint="Head north from Oldale to Route 103. Walk west through tall grass until you encounter your rival. They'll challenge you to a battle with a Level 5 starter that has type advantage over yours (Torchic if you chose Treecko, Mudkip if you chose Torchic, Treecko if you chose Mudkip). Use Potions if your HP gets low",
                completion_condition="battle_ends_and_rival_says_theyre_returning_to_lab",
                priority=1
            ),
            
            DirectObjective(
                id="lab_02_receive_pokedex",
                description="Follow your rival back to Professor Birch's Lab to receive the Pokédex from Birch and 5 Poké Balls from May/Brendan",
                action_type="dialogue",
                target_location="Professor Birch's Lab",
                navigation_hint="Walk south through Route 103, through Oldale Town, then south through Route 101 back to Littleroot Town. Enter the Lab. Professor Birch will give you a Pokédex (automatically records all Pokémon you see or catch). May/Brendan will give you 5 Poké Balls to start catching wild Pokémon",
                completion_condition="received_pokeballs_and_lab_dialogue_fully_ends",
                priority=1
            ),
            
            DirectObjective(
                id="littleroot_02_running_shoes",
                description="Exit the Lab and receive Running Shoes from Mom as you attempt to leave Littleroot Town",
                action_type="dialogue",
                target_location="Littleroot Town",
                navigation_hint="Exit the Lab. As you try to leave Littleroot Town, Mom will automatically stop you outside. She'll give you Running Shoes that let you run at double speed by holding the B button while moving",
                completion_condition="received_running_shoes_and_mom_dialogue_ends",
                priority=1
            ),
            
            DirectObjective(
                id="route102_01_travel_west",
                description="Travel back through Route 101 to Oldale Town, then head west on Route 102 toward Petalburg City",
                action_type="navigate",
                target_location="Route 102",
                navigation_hint="Go north through Route 101 to Oldale Town. The western exit (Route 102) should now be unblocked - the sketch artist has left. Head west to enter Route 102",
                completion_condition="player_location_is_route_102",
                priority=1
            ),
            
            DirectObjective(
                id="petalburg_01_reach_city",
                description="Continue west through Route 102 to arrive at Petalburg City",
                action_type="navigate",
                target_location="Petalburg City",
                navigation_hint="Walk west along Route 102. You may encounter wild Pokémon in tall grass and Trainers may challenge you, but you can avoid them. Continue until you reach Petalburg City",
                completion_condition="player_location_is_petalburg_city",
                priority=1
            ),

            DirectObjective(
                id="petalburg_02_meet_norman",
                description="Enter the Petalburg Gym and meet your father Norman, the Gym Leader",
                action_type="dialogue",
                target_location="Petalburg Gym",
                navigation_hint="Walk to the Pokémon Gym (large building in center-north of the city) and enter through the front door. Talk to Norman standing in the lobby. He'll be surprised you've made it this far",
                completion_condition="norman_dialogue_ends_before_wally_enters",
                priority=1
            ),
            
            DirectObjective(
                id="petalburg_03_help_wally",
                description="Meet Wally who enters the Gym asking for help. Accompany him to Route 102 to help him catch his first Pokémon",
                action_type="dialogue",
                target_location="Route 102",
                navigation_hint="After Norman's dialogue, Wally enters and asks for help catching a Pokémon. Norman loans him a Zigzagoon and Poké Ball. The game will automatically take you back to Route 102 where a cutscene plays showing Wally successfully catching a Ralts using Norman's Zigzagoon",
                completion_condition="wally_catches_ralts_and_cutscene_ends",
                priority=1
            ),
            
            DirectObjective(
                id="petalburg_04_receive_objective",
                description="Return to Petalburg Gym where Norman gives you the objective to challenge Gym Leader Roxanne in Rustboro City before battling other Gym Leaders",
                action_type="dialogue",
                target_location="Petalburg Gym",
                navigation_hint="You'll automatically return to the Gym after Wally's capture. Wally thanks you profusely. Norman then advises you to defeat Gym Leader Roxanne in Rustboro City first before challenging other Gyms. He won't accept your challenge until you have earned four Badges",
                completion_condition="norman_finishes_advice_about_roxanne_and_four_badges",
                priority=1
            ),
            
            DirectObjective(
                id="petalburg_05_mysterious_man",
                description="Exit the Gym and head toward the western exit of Petalburg City where a mysterious man in sunglasses will stop you",
                action_type="dialogue",
                target_location="Petalburg City",
                navigation_hint="Exit the Gym and walk toward the western exit of the city (toward Route 104). A man in sunglasses will automatically stop and talk to you. He judges you as a rookie Trainer, mentions he's searching for powerful Trainers, apologizes for taking your time, then leaves for Route 104",
                completion_condition="sunglasses_man_dialogue_ends_and_he_walks_away",
                priority=1
            ),
            
            DirectObjective(
                id="part1_complete",
                description="Part 1 Complete! You are now ready to continue west to Route 104 and begin your journey toward Rustboro City",
                action_type="navigate",
                target_location="Route 104 Entrance",
                navigation_hint="You now have your starter Pokémon, Pokédex, 5 Poké Balls, and Running Shoes. Your next goal is to travel west through Route 104 to eventually reach Rustboro City and challenge Gym Leader Roxanne. Walk to the western exit of Petalburg City",
                completion_condition="player_is_at_route_104_entrance_or_has_entered_route_104",
                priority=1
            )
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
        
        logger.info(f"Loaded part_1_walkthrough_claude_4_5 sequence with {len(self.current_sequence)} objectives, starting at index {self.current_index} ({len(initial_completed)} pre-completed)")
        
    def get_current_objective(self) -> Optional[DirectObjective]:
        """Get the current objective in the sequence"""
        if self.current_index < len(self.current_sequence):
            return self.current_sequence[self.current_index]
        return None
        
    def get_current_objective_guidance(self, game_state: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get guidance for the current objective.
        
        Note: This method does NOT automatically check completion. The LLM must call
        complete_direct_objective() endpoint to mark objectives as complete.
        
        Args:
            game_state: Optional game state (kept for backward compatibility, not used)
        """
        current_obj = self.get_current_objective()
        if not current_obj:
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
        """DEPRECATED: Check if an objective is completed based on game state.
        
        This method is deprecated because the LLM now uses complete_direct_objective()
        endpoint to explicitly mark objectives as complete. Automatic completion checking
        has been removed from get_current_objective_guidance().
        
        This method is kept for backward compatibility but should not be used in new code.
        """
        import warnings
        warnings.warn(
            "_is_objective_completed() is deprecated. Use complete_direct_objective() endpoint instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Return False - let LLM determine completion via complete_direct_objective()
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
    
    def get_objective_context(self, game_state: Dict[str, Any] = None) -> str:
        """Get previous objective context for better agent understanding.
        
        Note: game_state parameter is kept for backward compatibility but not used.
        
        Returns:
            String with previous objective context (NEXT removed to avoid confusion)
        """
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



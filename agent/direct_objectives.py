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
        
    def load_tutorial_to_starter_sequence(self):
        """Load the hardcoded sequence for transitioning from tutorial completion to starter selection"""
        self.sequence_name = "tutorial_to_starter"
        self.current_sequence = [
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
                id="tutorial_02_enter_player_house",
                description="Enter the player's house in Littleroot Town by pressing A to advance through the dialogue with you mother",
                action_type="interact",
                target_location="Player's House",
                navigation_hint="Move to the house with the red roof and enter through the door",
                completion_condition="player_house_entered",
                priority=1
            ),
            DirectObjective(
                id="tutorial_03_go_to_bedroom",
                description="You're mom will immediately greet you and take you to the house. Press A to advance through the dialogue and navigate to the stairs to go upstairs to the player's (your)bedroom",
                action_type="navigate",
                target_location="Player's Bedroom",
                navigation_hint="Walk north towards and through the stairs (S) to go up to the bedroom. Vigoroth will be carrying boxes on your left. Ignore them.",
                completion_condition="player_bedroom_reached",
                priority=1
            ),
            DirectObjective(
                id="tutorial_04_interact_with_tv",
                description="Interact with the clock on the wall to set the time",
                action_type="interact",
                target_location="Player's Bedroom",
                navigation_hint="Press A on the clock on the wall to set the time. The clock is 2 tiles left of the stairs (D) and on the wall.",
                completion_condition="clock_set",
                priority=1
            ),
            DirectObjective(
                id="tutorial_05_exit_player_house",
                description="Exit the player's house by walking down the stairs (S) and through the door (D)",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="Go downstairs. This will trigger an event with your mom sending you towards the TV. Once this is over, exit through the door (D) to your south-east.",
                completion_condition="exited_player_house",
                priority=1
            ),
            DirectObjective(
                id="tutorial_06_enter_rival_house",
                description="Enter the rival's house (next door) by immediately navigating to the right and walking through the door (D)",
                action_type="interact",
                target_location="Rival's House",
                navigation_hint="Your rival's house will be to the right of your house and looks identical.",
                completion_condition="rival_house_entered",
                priority=1
            ),
            DirectObjective(
                id="tutorial_07_go_to_rival_bedroom",
                description="Go upstairs to the rival's bedroom. Their mother will immediately greet you. After your interaction with her, walk north immediately to go up the stairs.",
                action_type="navigate",
                target_location="Rival's Bedroom",
                navigation_hint="Use the stairs to go up to the rival's bedroom. The stairs are a straight line north of the entrance (~7/8 tiles north of the entrance door (D)).",
                completion_condition="rival_bedroom_reached",
                priority=1
            ),
            DirectObjective(
                id="tutorial_08_talk_to_rival",
                description="Once in your rival's room, interact with the pokeball next to her bed. This will trigger a conversation with her, after which she will walk over to her computer. ",
                action_type="interact",
                target_location="Rival's Bedroom",
                navigation_hint="Approach and interact with the pokeball to the south-east of the entrance (next to the bed), once the",
                completion_condition="rival_conversation_complete",
                priority=1
            ),
            DirectObjective(
                id="tutorial_09_exit_rival_house",
                description="Exit the rival's house by walking through the stairs (S) you entered your rival's room through and through the door (D) you entered her house through",
                action_type="navigate",
                target_location="Littleroot Town",
                navigation_hint="The stairs should be a few tiles left and several tiles north. Once you get down the stairs, walk south through the entrance door (D) to exit the house.",
                completion_condition="exited_rival_house",
                priority=1
            ),
            DirectObjective(
                id="tutorial_10_north_to_route101",
                description="Move north from Littleroot Town to Route 101",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Move north from Littleroot Town to reach Route 101. The route is straight north in between your house and the rival's house.",
                completion_condition="location_contains_route_101",
                priority=1
            ),
            DirectObjective(
                id="tutorial_11_find_prof_birch",
                description="Find Professor Birch on Route 101. This should be triggered by walking a few steps into route 101 and triggering an event with a man in a lab coat.",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="Look for Professor Birch - he should be visible on the route",
                completion_condition="prof_birch_found",
                priority=1
            ),
            DirectObjective(
                id="tutorial_12_approach_birch",
                description="Professor Birch will need your help, interact with the bag on the ground by the ledge and pick your starter Pokemon from it. Once you pick your starter, it will trigger a battle with a zigzagoon.",
                action_type="navigate",
                target_location="Route 101",
                navigation_hint="The bag will be on the ground to your left and in between you and professor birch.",
                completion_condition="birch_encounter_triggered",
                priority=1
            ),
            DirectObjective(
                id="tutorial_13_battle_zigzagoon",
                description="Battle the zigzagoon",
                action_type="battle",
                target_location="Route 101",
                navigation_hint="Battle the zigzagoon by selecting a damaging move and pressing A to attack",
                completion_condition="zigzagoon_battle_complete",
                priority=1
            ),
            DirectObjective(
                id="tutorial_14_interact_with_professor_birch",
                description="After battling the zigzagoon, professor birch will interact with you. Advance through the dialogue by pressing A to continue. After this interaction, he will transport you to the lab",
                action_type="interact",
                target_location="Route 101",
                navigation_hint="N/A",
                completion_condition="professor_birch_interaction_complete",
                priority=1
            ),

            DirectObjective(
                id="tutorial_15_talk_to_professor_birch_in_lab",
                description="Talk to professor birch in the lab to receive the pokedex. Advance through the dialogue by pressing A to continue.",
                action_type="navigate",
                target_location="Professor Birch's Lab",
                navigation_hint="Talk to professor birch.",
                completion_condition="dialogue_complete",
                priority=1
            ),
            DirectObjective(
                id="tutorial_16_exit_professor_birch_lab",
                description="Once dialogue is complete, exit the lab by walking immediately south through the door (D) to littleroot town",
                action_type="interact",
                target_location="Professor Birch's Lab",
                navigation_hint="Walk south through the door (D) to littleroot town. For reference, the lab is to the south of your mom's house.",
                completion_condition="exited_professor_birch_lab",
                priority=1
            )
        ]
        self.current_index = 0
        logger.info(f"Loaded tutorial_to_starter sequence with {len(self.current_sequence)} objectives")
        
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
                
        except Exception as e:
            logger.warning(f"Error checking objective completion: {e}")
            
        return False
        
    def _mark_objective_completed(self, objective: DirectObjective):
        """Mark an objective as completed"""
        objective.completed = True
        objective.completed_at = datetime.now()
        
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
        """Get previous, current, and next objective context for better agent understanding"""
        if not self.is_sequence_active():
            return ""
        
        context_parts = []
        
        # Previous objective
        if self.current_index > 0:
            prev_obj = self.current_sequence[self.current_index - 1]
            status = "✅ COMPLETED" if prev_obj.completed else "❌ INCOMPLETE"
            context_parts.append(f"PREVIOUS OBJECTIVE: {prev_obj.description} [{status}]")
        
        # Current objective
        if self.current_index < len(self.current_sequence):
            current_obj = self.current_sequence[self.current_index]
            context_parts.append(f"CURRENT OBJECTIVE: {current_obj.description}")
            if current_obj.navigation_hint:
                context_parts.append(f"  → Hint: {current_obj.navigation_hint}")
        
        # Next objective
        if self.current_index + 1 < len(self.current_sequence):
            next_obj = self.current_sequence[self.current_index + 1]
            context_parts.append(f"NEXT OBJECTIVE: {next_obj.description}")
        
        return "\n".join(context_parts)



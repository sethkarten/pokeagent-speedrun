"""Pokemon Red milestone tracker.

Identical functionality to MilestoneTracker in pokemon_env/emulator.py,
but with a Red-specific milestone order used for split time calculation.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class RedMilestoneTracker:
    """Persistent milestone tracking system for Pokemon Red."""

    def __init__(self, filename: str = None):
        # Setup cache directory
        from utils.run_data_manager import get_cache_directory
        self.cache_dir = str(get_cache_directory())
        os.makedirs(self.cache_dir, exist_ok=True)

        # Use cache folder for runtime milestone file
        if filename is None:
            filename = os.path.join(self.cache_dir, "milestones_progress.json")
        self.filename = filename  # Runtime cache file (always in cache directory)
        self.loaded_state_milestones_file = None  # Track if we loaded from a state-specific file
        self.milestones = {}
        self.latest_milestone = None
        self.latest_split_time = "00:00:00"
        # Don't automatically load from file - only load when explicitly requested

    def load_from_file(self):
        """Load milestone progress from file."""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.milestones = data.get('milestones', {})

                # Determine the latest completed milestone based on timestamps
                latest_timestamp = 0
                latest_milestone_id = None
                for milestone_id, milestone_data in self.milestones.items():
                    if milestone_data.get('completed', False):
                        timestamp = milestone_data.get('timestamp', 0)
                        if timestamp > latest_timestamp:
                            latest_timestamp = timestamp
                            latest_milestone_id = milestone_id

                # Set the latest milestone if we found one
                if latest_milestone_id:
                    self.latest_milestone = latest_milestone_id
                    self.latest_split_time = self.milestones[latest_milestone_id].get('split_formatted', '00:00:00')
                    logger.info(f"Latest milestone from file: {latest_milestone_id}")

                logger.info(f"Loaded {len(self.milestones)} milestone records from {self.filename}")
            else:
                logger.info("No existing milestone file found, starting fresh")
                self.milestones = {}
        except Exception as e:
            logger.warning(f"Error loading milestones from file: {e}")
            self.milestones = {}

    def save_to_file(self):
        """Save milestone progress to file."""
        try:
            data = {
                'milestones': self.milestones,
                'last_updated': time.time(),
                'version': '1.0'
            }
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved milestone progress to {self.filename}")
        except Exception as e:
            logger.warning(f"Error saving milestones to file: {e}")

    def mark_completed(self, milestone_id: str, timestamp: float = None, agent_step_count: int = None):
        """Mark a milestone as completed and log split time.

        Args:
            milestone_id: ID of the milestone being completed
            timestamp: Optional timestamp (defaults to current time)
            agent_step_count: Optional current agent step count for metrics tracking
        """
        if timestamp is None:
            timestamp = time.time()

        if milestone_id not in self.milestones or not self.milestones[milestone_id].get('completed', False):
            # Calculate split time from previous milestone or start
            split_time = self._calculate_split_time(milestone_id, timestamp)

            self.milestones[milestone_id] = {
                'completed': True,
                'timestamp': timestamp,
                'first_completed': timestamp,
                'split_time': split_time,
                'split_formatted': self._format_time(split_time),
                'total_time': self._calculate_total_time(timestamp),
                'total_formatted': self._format_time(self._calculate_total_time(timestamp))
            }

            # Store the latest completed milestone for easy access
            self.latest_milestone = milestone_id
            self.latest_split_time = self._format_time(split_time)

            logger.info(f"Milestone completed: {milestone_id} (Split: {self._format_time(split_time)})")
            self.save_to_file()

            # Log milestone completion to LLM logger for unified metrics
            if agent_step_count is not None:
                try:
                    from utils.llm_logger import log_milestone_completion
                    log_milestone_completion(milestone_id, agent_step_count, timestamp)
                except Exception as e:
                    logger.debug(f"Could not log milestone to LLM logger: {e}")

            return True
        return False

    def is_completed(self, milestone_id: str) -> bool:
        """Check if a milestone is completed."""
        return self.milestones.get(milestone_id, {}).get('completed', False)

    def get_milestone_data(self, milestone_id: str) -> dict:
        """Get milestone data."""
        return self.milestones.get(milestone_id, {'completed': False, 'timestamp': None})

    def reset_milestone(self, milestone_id: str):
        """Reset a milestone (for testing)."""
        if milestone_id in self.milestones:
            del self.milestones[milestone_id]
            self.save_to_file()
            logger.info(f"Reset milestone: {milestone_id}")

    def _calculate_split_time(self, milestone_id: str, timestamp: float) -> float:
        """Calculate split time from previous milestone completion or start."""
        # Pokemon Red milestone order for split calculation
        milestone_order = [
            # Phase 1: Initialization
            "GAME_RUNNING",
            # Phase 2: Pallet Town
            "PALLET_TOWN_START", "OAK_ENCOUNTER",
            # Phase 3: Route 1 → Viridian City
            "VIRIDIAN_CITY",
            # Phase 4: Viridian Forest → Pewter City
            "PEWTER_CITY", "BROCK_DEFEATED",
            # Phase 5: Mt. Moon → Cerulean City
            "MT_MOON_CROSSED", "CERULEAN_CITY", "MISTY_DEFEATED",
            # Phase 6: Routes south → Vermilion City
            "SS_ANNE", "SURGE_DEFEATED",
            # Phase 7: Rock Tunnel → Lavender Town
            "ROCK_TUNNEL", "LAVENDER_TOWN",
            # Phase 8: Celadon City + Rocket Hideout
            "CELADON_CITY", "ERIKA_DEFEATED", "ROCKET_HIDEOUT",
            # Phase 9: Silph Co. + remaining gym leaders
            "SILPH_CO", "KOGA_DEFEATED", "SABRINA_DEFEATED",
            "BLAINE_DEFEATED", "GIOVANNI_DEFEATED",
            # Phase 10: Victory Road + Elite Four
            "VICTORY_ROAD", "ELITE_FOUR_START", "CHAMPION",
        ]

        try:
            # Special case for first milestone — split time is 0
            if milestone_id == "GAME_RUNNING":
                return 0.0

            if milestone_id not in milestone_order:
                # For unlisted milestones, find the most recent completion
                latest_timestamp = 0
                for _, data in self.milestones.items():
                    if data.get('completed', False) and data.get('timestamp', 0) > latest_timestamp:
                        latest_timestamp = data.get('timestamp', 0)
                return timestamp - latest_timestamp if latest_timestamp > 0 else 0.0

            # Find the previous milestone in the order
            current_index = milestone_order.index(milestone_id)

            # Look backwards for the most recent completed milestone
            for i in range(current_index - 1, -1, -1):
                prev_milestone = milestone_order[i]
                if self.is_completed(prev_milestone):
                    prev_timestamp = self.milestones[prev_milestone].get('timestamp', 0)
                    return timestamp - prev_timestamp

            # If no previous milestone found, calculate from start if we have GAME_RUNNING
            if self.is_completed("GAME_RUNNING"):
                start_timestamp = self.milestones["GAME_RUNNING"].get('timestamp', 0)
                return timestamp - start_timestamp

            # Fallback — no split time available
            return 0.0

        except Exception as e:
            logger.warning(f"Error calculating split time for {milestone_id}: {e}")
            return 0.0

    def _format_time(self, seconds: float) -> str:
        """Format time in HH:MM:SS format."""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        except Exception:
            return "00:00:00"

    def _calculate_total_time(self, timestamp: float) -> float:
        """Calculate total time from game start."""
        try:
            if self.is_completed("GAME_RUNNING"):
                start_timestamp = self.milestones["GAME_RUNNING"].get('timestamp', timestamp)
                return timestamp - start_timestamp
            return 0.0
        except Exception:
            return 0.0

    def get_latest_milestone_info(self) -> tuple:
        """Get the latest milestone information for submission logging.

        Returns: (milestone_name, split_time_formatted, total_time_formatted)
        """
        if self.latest_milestone:
            milestone_data = self.milestones.get(self.latest_milestone, {})
            split_formatted = milestone_data.get('split_formatted', '00:00:00')
            total_formatted = milestone_data.get('total_formatted', '00:00:00')
            return (self.latest_milestone, split_formatted, total_formatted)
        return ("NONE", "00:00:00", "00:00:00")

    def get_all_completed_milestones(self) -> list:
        """Get a list of all completed milestones with their times."""
        completed = []
        for milestone_id, data in self.milestones.items():
            if data.get('completed', False):
                completed.append({
                    'id': milestone_id,
                    'timestamp': data.get('timestamp', 0),
                    'split_time': data.get('split_formatted', '00:00:00'),
                    'total_time': data.get('total_formatted', '00:00:00')
                })
        return sorted(completed, key=lambda x: x['timestamp'])

    def reset_all(self):
        """Reset all milestones (for testing)."""
        self.milestones = {}
        self.save_to_file()
        logger.info("Reset all milestones")

    def load_milestones_for_state(self, state_filename: str = None):
        """Load milestones from file, optionally with a specific state filename."""
        if state_filename:
            state_dir = os.path.dirname(state_filename)
            base_name = os.path.splitext(os.path.basename(state_filename))[0]
            milestone_filename = os.path.join(state_dir, f"{base_name}_milestones.json")

            self.loaded_state_milestones_file = milestone_filename
            logger.info(f"Loading milestones from state-specific file: {milestone_filename}")

            try:
                original_filename = self.filename
                self.filename = milestone_filename
                self.load_from_file()
                self.filename = original_filename
                logger.info(f"Loaded {len(self.milestones)} milestones from state {state_filename}")
                logger.info(f"Runtime milestone cache will be saved to: {self.filename}")
            except FileNotFoundError:
                logger.info(f"Milestone file not found: {milestone_filename}, starting fresh")
                self.milestones = {}
                logger.info(f"Runtime milestone cache will be saved to: {self.filename}")
            except Exception as e:
                logger.error(f"Error loading milestone file {milestone_filename}: {e}")
                logger.info(f"Using runtime milestone cache: {self.filename}")
                self.load_from_file()
        else:
            self.loaded_state_milestones_file = None
            self.filename = os.path.join(self.cache_dir, "milestones_progress.json")
            logger.info(f"Loading milestones from default file: {self.filename}")
            self.load_from_file()

    def save_milestones_for_state(self, state_filename: str = None):
        """Save milestones to file, optionally with a specific state filename."""
        if state_filename:
            state_dir = os.path.dirname(state_filename)
            base_name = os.path.splitext(os.path.basename(state_filename))[0]
            milestone_filename = os.path.join(state_dir, f"{base_name}_milestones.json")

            original_filename = self.filename
            self.filename = milestone_filename
            logger.info(f"Saving {len(self.milestones)} milestones to state-specific file: {milestone_filename}")

            try:
                self.save_to_file()
                logger.info(f"Successfully saved milestones to {milestone_filename}")
            except Exception as e:
                logger.error(f"Error saving milestone file {milestone_filename}: {e}")
                self.filename = original_filename
                self.save_to_file()
                return original_filename
            finally:
                self.filename = original_filename

            return milestone_filename
        else:
            logger.info(f"Saving {len(self.milestones)} milestones to default file: {self.filename}")
            self.save_to_file()
            return self.filename

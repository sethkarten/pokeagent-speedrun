import logging
import time
import threading
import queue
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from PIL import Image

import mgba.core
import mgba.log
import mgba.image
from mgba._pylib import ffi, lib

from .memory_reader import PokemonEmeraldReader

logger = logging.getLogger(__name__)

# some acknowledgement to https://github.com/dvruette/pygba

class MilestoneTracker:
    """Persistent milestone tracking system integrated with emulator"""
    
    def __init__(self, filename: str = "milestones_progress.json"):
        self.filename = filename
        self.milestones = {}
        # Don't automatically load from file - only load when explicitly requested
    
    def load_from_file(self):
        """Load milestone progress from file"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.milestones = data.get('milestones', {})
                logger.info(f"Loaded {len(self.milestones)} milestone records from {self.filename}")
            else:
                logger.info(f"No existing milestone file found, starting fresh")
                self.milestones = {}
        except Exception as e:
            logger.warning(f"Error loading milestones from file: {e}")
            self.milestones = {}
    
    def save_to_file(self):
        """Save milestone progress to file"""
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
    
    def mark_completed(self, milestone_id: str, timestamp: float = None):
        """Mark a milestone as completed"""
        if timestamp is None:
            timestamp = time.time()
        
        if milestone_id not in self.milestones or not self.milestones[milestone_id].get('completed', False):
            self.milestones[milestone_id] = {
                'completed': True,
                'timestamp': timestamp,
                'first_completed': timestamp
            }
            logger.info(f"Milestone completed: {milestone_id}")
            self.save_to_file()
            return True
        return False
    
    def is_completed(self, milestone_id: str) -> bool:
        """Check if a milestone is completed"""
        return self.milestones.get(milestone_id, {}).get('completed', False)
    
    def get_milestone_data(self, milestone_id: str) -> dict:
        """Get milestone data"""
        return self.milestones.get(milestone_id, {'completed': False, 'timestamp': None})
    
    def reset_milestone(self, milestone_id: str):
        """Reset a milestone (for testing)"""
        if milestone_id in self.milestones:
            del self.milestones[milestone_id]
            self.save_to_file()
            logger.info(f"Reset milestone: {milestone_id}")
    
    def reset_all(self):
        """Reset all milestones (for testing)"""
        self.milestones = {}
        self.save_to_file()
        logger.info("Reset all milestones")
    
    def load_milestones_for_state(self, state_filename: str = None):
        """Load milestones from file, optionally with a specific state filename"""
        if state_filename:
            # If a state filename is provided, try to load milestones from a corresponding file
            # Get the directory and base name of the state file
            import os
            state_dir = os.path.dirname(state_filename)
            base_name = os.path.splitext(os.path.basename(state_filename))[0]
            milestone_filename = os.path.join(state_dir, f"{base_name}_milestones.json")
            
            # Update the filename to the state-specific file permanently
            self.filename = milestone_filename
            logger.info(f"Loading milestones from state-specific file: {milestone_filename}")
            
            try:
                self.load_from_file()
                logger.info(f"Loaded {len(self.milestones)} milestones for state {state_filename}")
            except FileNotFoundError:
                logger.info(f"Milestone file not found: {milestone_filename}, starting fresh milestones for this state")
                # Start with empty milestones for this state, don't fall back to default
                self.milestones = {}
                # Save the empty milestone file to establish the state-specific milestone file
                self.save_to_file()
                logger.info(f"Created new milestone file: {milestone_filename}")
            except Exception as e:
                logger.error(f"Error loading milestone file {milestone_filename}: {e}")
                # Fall back to default milestone file
                self.filename = "milestones_progress.json"
                self.load_from_file()
        else:
            # No state filename provided, use default milestone file
            self.filename = "milestones_progress.json"
            logger.info(f"Loading milestones from default file: {self.filename}")
            self.load_from_file()
    
    def save_milestones_for_state(self, state_filename: str = None):
        """Save milestones to file, optionally with a specific state filename"""
        if state_filename:
            # If a state filename is provided, save milestones to a corresponding file
            # Get the directory and base name of the state file
            import os
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
                # Fall back to default milestone file
                self.filename = original_filename
                self.save_to_file()
                return original_filename
            finally:
                # Restore original filename
                self.filename = original_filename
            
            return milestone_filename
        else:
            # Save to default milestone file
            logger.info(f"Saving {len(self.milestones)} milestones to default file: {self.filename}")
            self.save_to_file()
            return self.filename


class EmeraldEmulator:
    """emulator wrapper for Pokémon Emerald with headless frame capture and scripted inputs."""

    def __init__(self, rom_path: str, headless: bool = True, sound: bool = False):
        self.rom_path = rom_path
        self.headless = headless
        self.sound = sound

        self.gba = None
        self.core = None
        self.width = 240
        self.height = 160
        self.running = False

        self.frame_queue = queue.Queue(maxsize=10)
        self.current_frame = None
        self.frame_thread = None
        
        # Memory reader for accessing game state
        self.memory_reader = None

        # Memory cache for efficient reading
        self._mem_cache = {}
        
        # Milestone tracker for progress tracking
        self.milestone_tracker = MilestoneTracker()

        # Dialog state tracking for FPS adjustment
        self._cached_dialog_state = False
        self._last_dialog_check_time = 0
        self._dialog_check_interval = 0.05  # Check dialog state every 50ms (more responsive)
        
        # Track currently loaded state file
        self._current_state_file = None

        # Define key mapping for mgba
        self.KEY_MAP = {
            "a": lib.GBA_KEY_A,
            "b": lib.GBA_KEY_B,
            "start": lib.GBA_KEY_START,
            "select": lib.GBA_KEY_SELECT,
            "up": lib.GBA_KEY_UP,
            "down": lib.GBA_KEY_DOWN,
            "left": lib.GBA_KEY_LEFT,
            "right": lib.GBA_KEY_RIGHT,
            "l": lib.GBA_KEY_L,
            "r": lib.GBA_KEY_R
        }

    def initialize(self):
        """Load ROM and set up emulator"""
        try:
            # Prevents relentless spamming to stdout by libmgba.
            mgba.log.silence()
            
            # Create a temporary directory and copy the gba file into it
            # this is necessary to prevent mgba from overwriting the save file (and to prevent crashes)
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_gba = tmp_dir / "rom.gba"
            tmp_gba.write_bytes(Path(self.rom_path).read_bytes())
            
            # Load the core
            self.core = mgba.core.load_path(str(tmp_gba))
            if self.core is None:
                raise ValueError(f"Failed to load GBA file: {self.rom_path}")
            
            # Auto-load save if it exists
            self.core.autoload_save()
            self.core.reset()
            
            # Get dimensions from the core
            self.width, self.height = self.core.desired_video_dimensions()
            logger.info(f"mgba initialized with ROM: {self.rom_path} and dimensions: {self.width}x{self.height}")
            
            # Set up video buffer for frame capture using mgba.image.Image
            self.video_buffer = mgba.image.Image(self.width, self.height)
            self.core.set_video_buffer(self.video_buffer)
            self.core.reset()  # Reset after setting video buffer
            
            # Initialize memory reader
            self.memory_reader = PokemonEmeraldReader(self.core)
            
            # Set up callback for memory reader to invalidate emulator cache on area transitions
            def invalidate_emulator_cache():
                if hasattr(self, '_cached_state'):
                    delattr(self, '_cached_state')
                if hasattr(self, '_cached_state_time'):
                    delattr(self, '_cached_state_time')
                    
            self.memory_reader._emulator_cache_invalidator = invalidate_emulator_cache
            
            # Set up frame callback to invalidate memory cache
            self.core.add_frame_callback(self._invalidate_mem_cache)
            
            logger.info(f"mgba initialized with ROM: {self.rom_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize mgba: {e}")

    def _invalidate_mem_cache(self):
        """Invalidate memory cache when frame changes"""
        self._mem_cache = {}

    def _get_memory_region(self, region_id: int):
        """Get memory region for efficient reading"""
        if region_id not in self._mem_cache:
            mem_core = self.core.memory.u8._core
            size = ffi.new("size_t *")
            ptr = ffi.cast("uint8_t *", mem_core.getMemoryBlock(mem_core, region_id, size))
            self._mem_cache[region_id] = ffi.buffer(ptr, size[0])[:]
        return self._mem_cache[region_id]

    def read_memory(self, address: int, size: int = 1):
        """Read memory at given address"""
        region_id = address >> lib.BASE_OFFSET
        mem_region = self._get_memory_region(region_id)
        mask = len(mem_region) - 1
        address &= mask
        return mem_region[address:address + size]

    def read_u8(self, address: int):
        """Read unsigned 8-bit value"""
        return int.from_bytes(self.read_memory(address, 1), byteorder='little', signed=False)

    def read_u16(self, address: int):
        """Read unsigned 16-bit value"""
        return int.from_bytes(self.read_memory(address, 2), byteorder='little', signed=False)

    def read_u32(self, address: int):
        """Read unsigned 32-bit value"""
        return int.from_bytes(self.read_memory(address, 4), byteorder='little', signed=False)

    def tick(self, frames: int = 1):
        """Advance emulator by given number of frames"""
        if self.core:
            for _ in range(frames):
                self.core.run_frame()

    def get_current_fps(self, base_fps: int = 30) -> int:
        """Get current FPS - quadruples during dialog for faster text progression"""
        # Use cached dialog state for performance
        return base_fps * 4 if self._cached_dialog_state else base_fps

    def _update_dialog_state_cache(self):
        """Update cached dialog state (called periodically for performance)"""
        import time
        current_time = time.time()
        
        # Only check dialog state periodically to avoid performance issues
        if current_time - self._last_dialog_check_time >= self._dialog_check_interval:
            if self.memory_reader:
                new_dialog_state = self.memory_reader.is_in_dialog()
                if new_dialog_state != self._cached_dialog_state:
                    self._cached_dialog_state = new_dialog_state
                    if new_dialog_state:
                        logger.debug("🎯 Dialog detected - switching to 4x FPS")
                    else:
                        logger.debug("✅ Dialog ended - reverting to normal FPS")
            self._last_dialog_check_time = current_time

    def press_key(self, key: str, frames: int = 2):
        """Press a key for specified number of frames"""
        if key not in self.KEY_MAP:
            raise ValueError(f"Invalid key: {key}")
        if frames < 2:
            raise ValueError("Cannot press a key for less than 2 frames.")
        
        key_code = self.KEY_MAP[key]
        self.core.add_keys(key_code)
        self.tick(frames - 1)
        self.core.clear_keys(key_code)
        self.tick(1)

    def press_buttons(self, buttons: List[str], hold_frames: int = 10, release_frames: int = 10):
        """Press a sequence of buttons"""
        if not self.core:
            return "Emulator not initialized"

        for button in buttons:
            if button.lower() not in self.KEY_MAP:
                logger.warning(f"Unknown button: {button}")
                continue
            
            self.press_key(button.lower(), hold_frames)

        self.tick(release_frames)
        return f"Pressed: {'+'.join(buttons)}"

    def run_frame_with_buttons(self, buttons: List[str]):
        """Set buttons and advance one frame."""
        if not self.core:
            return

        # Set all buttons for one frame
        for button in buttons:
            if button.lower() in self.KEY_MAP:
                key_code = self.KEY_MAP[button.lower()]
                self.core.add_keys(key_code)
        
        self.core.run_frame()
        
        # Clear all buttons
        for button in buttons:
            if button.lower() in self.KEY_MAP:
                key_code = self.KEY_MAP[button.lower()]
                self.core.clear_keys(key_code)
        
        # Update dialog state cache for FPS adjustment
        self._update_dialog_state_cache()
        
        # Clear dialogue cache if A button was pressed (dismisses dialogue)
        if buttons and any(button.lower() == 'a' for button in buttons):
            if self.memory_reader:
                self.memory_reader.clear_dialogue_cache_on_button_press()
        
        # Clear state cache after action to ensure fresh data
        if hasattr(self, '_cached_state'):
            delattr(self, '_cached_state')
        if hasattr(self, '_cached_state_time'):
            delattr(self, '_cached_state_time')

    def get_screenshot(self) -> Optional[Image.Image]:
        """Return the current frame as a PIL image"""
        if not self.core or not self.video_buffer:
            return None
        
        try:
            # Use the built-in to_pil() method from mgba.image.Image
            if hasattr(self.video_buffer, 'to_pil'):
                screenshot = self.video_buffer.to_pil()
                if screenshot:
                    screenshot = screenshot.convert("RGB")
                    return screenshot
                else:
                    logger.warning("mgba.image.Image does not have to_pil method")
                    return None
            else:
                logger.warning("mgba.image.Image does not have to_pil method")
                return None
        except Exception as e:
            logger.error(f"Failed to get screenshot: {e}")
            return None

    def save_state(self, path: Optional[str] = None) -> Optional[bytes]:
        """Save current emulator state to file or return as bytes"""
        if not self.core:
            return None
        
        try:
            # Get the raw state data
            raw_data = self.core.save_raw_state()
            
            # Convert CFFI object to bytes if needed
            if hasattr(raw_data, 'buffer'):
                data = bytes(raw_data.buffer)
            elif hasattr(raw_data, '__len__'):
                data = bytes(raw_data)
            else:
                data = raw_data
            
            if path:
                with open(path, 'wb') as f:
                    f.write(data)
                logger.info(f"State saved to {path}")
                
                # Save corresponding milestones for this state
                milestone_filename = self.milestone_tracker.save_milestones_for_state(path)
                logger.info(f"Milestones saved to {milestone_filename}")
            
            return data
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return None

    def load_state(self, path: Optional[str] = None, state_bytes: Optional[bytes] = None):
        """Load emulator state from file or memory"""
        if not self.core:
            return
        
        try:
            if path:
                with open(path, 'rb') as f:
                    state_bytes = f.read()
            if state_bytes:
                # Ensure state_bytes is actually bytes
                if not isinstance(state_bytes, bytes):
                    state_bytes = bytes(state_bytes)
                self.core.load_raw_state(state_bytes)
                logger.info("State loaded.")
                
                # Reset dialog tracking and invalidate map cache when loading new state
                if self.memory_reader:
                    self.memory_reader.reset_dialog_tracking()
                    # Don't clear buffer address on state load to avoid expensive rescans
                    self.memory_reader.invalidate_map_cache(clear_buffer_address=False)
                    
                    # Run a frame to ensure memory is properly loaded
                    self.core.run_frame()
                    
                    # Only find map buffer addresses if we don't have them cached
                    # This avoids expensive memory scanning on every state load
                    if not self.memory_reader._map_buffer_addr:
                        if not self.memory_reader._find_map_buffer_addresses():
                            logger.warning("Could not find map buffer addresses after state load")
                        else:
                            logger.info(f"Map buffer found at 0x{self.memory_reader._map_buffer_addr:08X}")
                    else:
                        logger.debug(f"Using cached map buffer at 0x{self.memory_reader._map_buffer_addr:08X}")
                
                # Set the current state file for both emulator and memory reader
                self._current_state_file = path
                if self.memory_reader:
                    self.memory_reader._current_state_file = path
                
                # Load corresponding milestones for this state
                if path:
                    self.milestone_tracker.load_milestones_for_state(path)
                    logger.info(f"Milestones loaded for state {path}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def start_frame_capture(self, fps: int = 30):
        """Start asynchronous frame capture"""
        self.running = True
        self.frame_thread = threading.Thread(target=self._frame_loop, args=(fps,), daemon=True)
        self.frame_thread.start()

    def _frame_loop(self, fps: int):
        interval = 1.0 / fps
        while self.running:
            start = time.time()
            frame = self.get_screenshot()
            if frame:
                np_frame = np.array(frame)
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(np_frame)
                self.current_frame = np_frame
            elapsed = time.time() - start
            time.sleep(max(0.001, interval - elapsed))

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return last captured frame"""
        return self.current_frame.copy() if self.current_frame is not None else None

    def process_input(self, input_data: Dict[str, Any]) -> str:
        """Handle JSON-style input payload"""
        try:
            input_type = input_data.get('type', 'button')
            if input_type == 'button':
                button = input_data.get('button')
                if button:
                    return self.press_buttons([button])
            elif input_type == 'sequence':
                buttons = input_data.get('buttons', [])
                return self.press_buttons(buttons)
            elif input_type == 'hold':
                button = input_data.get('button')
                duration = int(input_data.get('duration', 1.0) * 60)
                return self.press_buttons([button], hold_frames=duration)
            return "Invalid input type"
        except Exception as e:
            logger.error(f"Input error: {e}")
            return str(e)

    def stop(self):
        """Stop emulator and cleanup"""
        self.running = False
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=1)
        if self.core:
            self.core = None
        logger.info("Emulator stopped.")

    def get_info(self) -> Dict[str, Any]:
        """Return metadata about emulator state"""
        return {
            "rom_path": self.rom_path,
            "dimensions": (self.width, self.height),
            "initialized": self.core is not None,
            "headless": self.headless,
            "sound": self.sound,
        }

    def get_comprehensive_state(self, screenshot=None) -> Dict[str, Any]:
        """Get comprehensive game state including visual and memory data using enhanced memory reader
        
        Args:
            screenshot: Optional PIL Image screenshot to use. If None, will call get_screenshot()
        """
        # Simple caching to avoid redundant calls within a short time window
        import time
        current_time = time.time()
        
        # Cache state for 100ms to avoid excessive memory reads
        if hasattr(self, '_cached_state') and hasattr(self, '_cached_state_time'):
            if current_time - self._cached_state_time < 0.1:  # 100ms cache
                return self._cached_state
        
        # Use provided screenshot or get a new one
        if screenshot is None:
            screenshot = self.get_screenshot()
        
        # Use the enhanced memory reader's comprehensive state method
        if self.memory_reader:
            state = self.memory_reader.get_comprehensive_state(screenshot)
        else:
            # Fallback to basic state
            state = {
                "visual": {
                    "screenshot": None,
                    "resolution": [self.width, self.height]
                },
                "player": {
                    "position": None,
                    "location": None,
                    "name": None
                },
                "game": {
                    "money": None,
                    "party": None,
                    "game_state": None,
                    "is_in_battle": None,
                    "time": None,
                    "badges": None,
                    "items": None,
                    "item_count": None,
                    "pokedex_caught": None,
                    "pokedex_seen": None
                },
                "map": {
                    "tiles": None,
                    "tile_names": None,
                    "metatile_behaviors": None,
                    "metatile_info": None,
                    "traversability": None
                }
            }
        
        # Use screenshot already captured
        if screenshot is not None and hasattr(screenshot, 'save'):
            state["visual"]["screenshot"] = screenshot
        
        # Cache the result
        self._cached_state = state
        self._cached_state_time = current_time
        
        return state

    def _get_tile_passability(self, tile_data) -> bool:
        """Determine if a tile is passable based on collision bits (like GeminiPlaysPokemonLive)"""
        if not tile_data or len(tile_data) < 3:
            return True  # Default to passable if no data
        
        # tile_data is (metatile_id, behavior, collision, elevation)
        collision = tile_data[2] if len(tile_data) > 2 else 0
        
        # Primary rule: collision == 0 means passable, non-zero means blocked
        return collision == 0

    def _get_tile_encounter_possible(self, tile_data) -> bool:
        """Determine if a tile can trigger encounters based on its behavior"""
        if not tile_data or len(tile_data) < 2:
            return False
        
        # Import here to avoid circular imports
        from .enums import MetatileBehavior
        
        behavior = tile_data[1] if len(tile_data) > 1 else None
        if not behavior:
            return False
        
        # Check for encounter tiles
        encounter_behaviors = {
            MetatileBehavior.TALL_GRASS,
            MetatileBehavior.LONG_GRASS,
            MetatileBehavior.UNUSED_05,
            MetatileBehavior.DEEP_SAND,
            MetatileBehavior.CAVE,
            MetatileBehavior.INDOOR_ENCOUNTER,
            MetatileBehavior.POND_WATER,
            MetatileBehavior.INTERIOR_DEEP_WATER,
            MetatileBehavior.DEEP_WATER,
            MetatileBehavior.OCEAN_WATER,
            MetatileBehavior.SEAWEED,
            MetatileBehavior.ASHGRASS,
            MetatileBehavior.FOOTPRINTS,
            MetatileBehavior.SEAWEED_NO_SURFACING
        }
        
        return behavior in encounter_behaviors

    def _get_tile_surfable(self, tile_data) -> bool:
        """Determine if a tile can be surfed on based on its behavior"""
        if not tile_data or len(tile_data) < 2:
            return False
        
        # Import here to avoid circular imports
        from .enums import MetatileBehavior
        
        behavior = tile_data[1] if len(tile_data) > 1 else None
        if not behavior:
            return False
        
        # Check for surfable tiles
        surfable_behaviors = {
            MetatileBehavior.POND_WATER,
            MetatileBehavior.INTERIOR_DEEP_WATER,
            MetatileBehavior.DEEP_WATER,
            MetatileBehavior.SOOTOPOLIS_DEEP_WATER,
            MetatileBehavior.OCEAN_WATER,
            MetatileBehavior.NO_SURFACING,
            MetatileBehavior.SEAWEED,
            MetatileBehavior.SEAWEED_NO_SURFACING
        }
        
        return behavior in surfable_behaviors

    def get_player_position(self) -> Optional[Dict[str, int]]:
        """Get current player position"""
        if self.memory_reader:
            try:
                coords = self.memory_reader.read_coordinates()
                if coords:
                    return {"x": coords[0], "y": coords[1]}
            except Exception as e:
                logger.warning(f"Failed to read player position: {e}")
        return None

    def get_map_location(self) -> Optional[str]:
        """Get current map location name"""
        if self.memory_reader:
            try:
                return self.memory_reader.read_location()
            except Exception as e:
                logger.warning(f"Failed to read map location: {e}")
        return None

    def get_money(self) -> Optional[int]:
        """Get current money amount"""
        if self.memory_reader:
            try:
                return self.memory_reader.read_money()
            except Exception as e:
                logger.warning(f"Failed to read money: {e}")
        return None

    def get_party_pokemon(self) -> Optional[List[Dict[str, Any]]]:
        """Get current party Pokemon"""
        if self.memory_reader:
            try:
                party = self.memory_reader.read_party_pokemon()
                if party:
                    return [
                        {
                            "species": pokemon.species_name,
                            "level": pokemon.level,
                            "current_hp": pokemon.current_hp,
                            "max_hp": pokemon.max_hp,
                            "status": pokemon.status.get_status_name() if pokemon.status else "OK",
                            "types": [t for t in [pokemon.type1.name if pokemon.type1 else None, 
                                                 pokemon.type2.name if pokemon.type2 else None] if t is not None]
                        }
                        for pokemon in party
                    ]
            except Exception as e:
                logger.warning(f"Failed to read party Pokemon: {e}")
        return None

    def get_map_tiles(self, radius: int = 7) -> Optional[List[List[tuple]]]:
        """Get map tiles around player"""
        if self.memory_reader:
            try:
                return self.memory_reader.read_map_around_player(radius=radius)
            except Exception as e:
                logger.warning(f"Failed to read map tiles: {e}")
        return None

    def test_memory_reading(self) -> Dict[str, Any]:
        """Test memory reading capabilities and return diagnostic information"""
        if not self.memory_reader:
            return {"error": "Memory reader not initialized"}
        
        try:
            # Get memory diagnostics
            diagnostics = self.memory_reader.test_memory_access()
            
            # Test some basic reads
            test_results = {
                "player_name": None,
                "money": None,
                "coordinates": None,
                "party_size": None,
                "location": None
            }
            
            try:
                test_results["player_name"] = self.memory_reader.read_player_name()
            except Exception as e:
                test_results["player_name_error"] = str(e)
            
            try:
                test_results["money"] = self.memory_reader.read_money()
            except Exception as e:
                test_results["money_error"] = str(e)
            
            try:
                test_results["coordinates"] = self.memory_reader.read_coordinates()
            except Exception as e:
                test_results["coordinates_error"] = str(e)
            
            try:
                test_results["party_size"] = self.memory_reader.read_party_size()
            except Exception as e:
                test_results["party_size_error"] = str(e)
            
            try:
                test_results["location"] = self.memory_reader.read_location()
            except Exception as e:
                test_results["location_error"] = str(e)
            
            return {
                "diagnostics": diagnostics,
                "test_results": test_results
            }
        except Exception as e:
            return {"error": f"Failed to run memory tests: {e}"}
    
    def check_and_update_milestones(self, game_state: Dict[str, Any]):
        """Check current game state and update milestones"""
        try:
            # Only check milestones that aren't already completed
            milestones_to_check = [
                "GAME_RUNNING", "HAS_PARTY", "LITTLEROOT_TOWN", "STARTER_CHOSEN", 
                "POKEDEX_RECEIVED", "OLDALE_TOWN", "FIRST_WILD_ENCOUNTER", 
                "FIRST_POKEMON_CAUGHT", "PETALBURG_CITY", "RUSTBORO_CITY", 
                "STONE_BADGE", "DEWFORD_TOWN", "KNUCKLE_BADGE", "SLATEPORT_CITY",
                "MAUVILLE_CITY", "DYNAMO_BADGE", "POKEDEX_5_SEEN", "POKEDEX_10_SEEN",
                "EARNED_1000_POKEDOLLARS", "PARTY_OF_TWO"
            ]
            
            for milestone_id in milestones_to_check:
                if not self.milestone_tracker.is_completed(milestone_id):
                    if self._check_milestone_condition(milestone_id, game_state):
                        self.milestone_tracker.mark_completed(milestone_id)
        
        except Exception as e:
            logger.warning(f"Error checking milestones: {e}")
    
    def _check_milestone_condition(self, milestone_id: str, game_state: Dict[str, Any]) -> bool:
        """Check if a specific milestone condition is met based on current game state"""
        try:
            # Test milestones (should always work)
            if milestone_id == "GAME_RUNNING":
                return True  # If we can execute this, game is running
            elif milestone_id == "HAS_PARTY":
                if game_state:
                    party = game_state.get("player", {}).get("party", [])
                    return len(party) > 0
                return False
            
            # Location-based milestones - check current location
            elif milestone_id == "LITTLEROOT_TOWN":
                if game_state:
                    location = game_state.get("player", {}).get("location", "")
                    return "LITTLEROOT" in str(location).upper()
                return False
            elif milestone_id == "OLDALE_TOWN":
                if game_state:
                    # Only count Oldale Town if we've already been to Littleroot Town
                    if not self.milestone_tracker.is_completed("LITTLEROOT_TOWN"):
                        return False
                    location = game_state.get("player", {}).get("location", "")
                    return "OLDALE" in str(location).upper()
                return False
            elif milestone_id == "PETALBURG_CITY":
                if game_state:
                    # Enforce proper game progression through required towns
                    if not self.milestone_tracker.is_completed("LITTLEROOT_TOWN"):
                        return False
                    if not self.milestone_tracker.is_completed("OLDALE_TOWN"):
                        return False
                    location = game_state.get("player", {}).get("location", "")
                    return "PETALBURG" in str(location).upper()
                return False
            elif milestone_id == "RUSTBORO_CITY":
                if game_state:
                    # Only count Rustboro City if we've already been to Petalburg City
                    if not self.milestone_tracker.is_completed("PETALBURG_CITY"):
                        return False
                    location = game_state.get("player", {}).get("location", "")
                    return "RUSTBORO" in str(location).upper()
                return False
            elif milestone_id == "DEWFORD_TOWN":
                if game_state:
                    # Only count Dewford Town if we've already been to Rustboro City
                    if not self.milestone_tracker.is_completed("RUSTBORO_CITY"):
                        return False
                    location = game_state.get("player", {}).get("location", "")
                    return "DEWFORD" in str(location).upper()
                return False
            elif milestone_id == "SLATEPORT_CITY":
                if game_state:
                    # Only count Slateport City if we've already been to Dewford Town
                    if not self.milestone_tracker.is_completed("DEWFORD_TOWN"):
                        return False
                    location = game_state.get("player", {}).get("location", "")
                    return "SLATEPORT" in str(location).upper()
                return False
            elif milestone_id == "MAUVILLE_CITY":
                if game_state:
                    # Only count Mauville City if we've already been to Slateport City
                    if not self.milestone_tracker.is_completed("SLATEPORT_CITY"):
                        return False
                    location = game_state.get("player", {}).get("location", "")
                    return "MAUVILLE" in str(location).upper()
                return False
                
            # Pokemon system milestones - check party/pokedex state  
            elif milestone_id == "STARTER_CHOSEN":
                if game_state:
                    party = game_state.get("player", {}).get("party", [])
                    return len(party) >= 1 and any(p.get("species_name", "").strip() for p in party)
                return False
            elif milestone_id == "POKEDEX_RECEIVED":
                if game_state:
                    pokedex_seen = game_state.get("game", {}).get("pokedex_seen", 0)
                    return (pokedex_seen if isinstance(pokedex_seen, int) else 0) >= 1
                return False
                
            # Badge milestones - check badge count/list
            elif milestone_id == "STONE_BADGE":
                if game_state:
                    badges = game_state.get("game", {}).get("badges", [])
                    if isinstance(badges, list):
                        return len(badges) >= 1 or any("Stone" in str(b) for b in badges)
                    elif isinstance(badges, int):
                        return badges >= 1
                return False
            elif milestone_id == "KNUCKLE_BADGE":
                if game_state:
                    badges = game_state.get("game", {}).get("badges", [])
                    if isinstance(badges, list):
                        return len(badges) >= 2 or any("Knuckle" in str(b) for b in badges)
                    elif isinstance(badges, int):
                        return badges >= 2
                return False
            elif milestone_id == "DYNAMO_BADGE":
                if game_state:
                    badges = game_state.get("game", {}).get("badges", [])
                    if isinstance(badges, list):
                        return len(badges) >= 3 or any("Dynamo" in str(b) for b in badges)
                    elif isinstance(badges, int):
                        return badges >= 3
                return False
                
            # Progress-based milestones
            elif milestone_id == "FIRST_POKEMON_CAUGHT":
                if game_state:
                    pokedex_caught = game_state.get("game", {}).get("pokedex_caught", 0)
                    return (pokedex_caught if isinstance(pokedex_caught, int) else 0) >= 1
                return False
            elif milestone_id == "FIRST_WILD_ENCOUNTER":
                if game_state:
                    pokedex_seen = game_state.get("game", {}).get("pokedex_seen", 0)
                    return (pokedex_seen if isinstance(pokedex_seen, int) else 0) >= 2
                return False
            elif milestone_id == "PARTY_OF_TWO":
                if game_state:
                    party = game_state.get("player", {}).get("party", [])
                    valid_pokemon = [p for p in party if p.get("species_name", "").strip() and p.get("species_name", "").strip() != "NONE"]
                    return len(valid_pokemon) >= 2
                return False
            elif milestone_id == "EARNED_1000_POKEDOLLARS":
                if game_state:
                    money = game_state.get("game", {}).get("money", 0)
                    return (money if isinstance(money, int) else 0) >= 1000
                return False
            elif milestone_id == "POKEDEX_5_SEEN":
                if game_state:
                    pokedex_seen = game_state.get("game", {}).get("pokedex_seen", 0)
                    return (pokedex_seen if isinstance(pokedex_seen, int) else 0) >= 5
                return False
            elif milestone_id == "POKEDEX_10_SEEN":
                if game_state:
                    pokedex_seen = game_state.get("game", {}).get("pokedex_seen", 0)
                    return (pokedex_seen if isinstance(pokedex_seen, int) else 0) >= 10
                return False
                
            return False
            
        except Exception as e:
            logger.warning(f"Error checking milestone condition {milestone_id}: {e}")
            return False
    
    def get_milestones(self) -> Dict[str, Any]:
        """Get current milestone data and progress"""
        try:
            # Get current game state and update milestones
            # Use cached state if available to avoid redundant calls
            game_state = self.get_comprehensive_state()
            # Only update milestones occasionally to avoid performance issues
            import time
            current_time = time.time()
            if not hasattr(self, '_last_milestone_update') or current_time - self._last_milestone_update > 1.0:  # Update at most once per second
                self.check_and_update_milestones(game_state)
                self._last_milestone_update = current_time
            
            # Use loaded milestones from the milestone tracker
            milestones = []
            for i, (milestone_id, milestone_data) in enumerate(self.milestone_tracker.milestones.items()):
                milestones.append({
                    "id": i + 1,
                    "name": milestone_data.get("name", milestone_id),
                    "category": milestone_data.get("category", "unknown"),
                    "completed": milestone_data.get("completed", False),
                    "timestamp": milestone_data.get("timestamp", None)
                })
            
            # Calculate summary stats
            completed_count = sum(1 for m in milestones if m["completed"])
            total_count = len(milestones)
            
            # Handle location data properly
            location_data = game_state.get("player", {}).get("location", "")
            if isinstance(location_data, dict):
                current_location = location_data.get("map_name", "UNKNOWN")
            else:
                current_location = str(location_data) if location_data else "UNKNOWN"
            
            # Handle badges data properly
            badges_data = game_state.get("game", {}).get("badges", 0)
            if isinstance(badges_data, list):
                badge_count = sum(1 for b in badges_data if b)
            else:
                badge_count = badges_data if isinstance(badges_data, int) else 0
            
            return {
                "milestones": milestones,
                "completed": completed_count,
                "total": total_count,
                "progress": completed_count / total_count if total_count > 0 else 0,
                "current_location": current_location,
                "badges": badge_count,
                "pokedex_seen": game_state.get("game", {}).get("pokedex_seen", 0),
                "pokedex_caught": game_state.get("game", {}).get("pokedex_caught", 0),
                "party_size": len(game_state.get("player", {}).get("party", [])),
                "tracking_system": "file_based",
                "milestone_file": self.milestone_tracker.filename
            }
            
        except Exception as e:
            logger.error(f"Error getting milestones: {e}")
            # Fallback to basic milestones if memory reading fails
            basic_milestones = [
                {"id": 1, "name": "GAME_STARTED", "category": "basic", "completed": True, "timestamp": time.time()},
                {"id": 2, "name": "EMULATOR_RUNNING", "category": "basic", "completed": True, "timestamp": time.time()},
            ]
            return {
                "milestones": basic_milestones,
                "completed": 2,
                "total": 2,
                "progress": 1.0,
                "tracking_system": "fallback",
                "error": str(e)
            }

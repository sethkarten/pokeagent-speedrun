"""Minimal Pokemon Red emulator wrapper using PyBoy.

Implements the same interface as EmeraldEmulator so it can be used
as a drop-in replacement by server/app.py and the agent scaffolds.
All game-state memory reading is delegated to RedMemoryReader.
"""

import io
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .red_map_reader import RedMapReader
from .red_memory_reader import RedMemoryReader
from .red_milestone_tracker import RedMilestoneTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Red-specific milestone list (ordered by expected completion)
# ---------------------------------------------------------------------------
RED_MILESTONES_ORDER = [
    "GAME_RUNNING",
    "PALLET_TOWN_START",
    "OAK_ENCOUNTER",
    "VIRIDIAN_CITY",
    "PEWTER_CITY",
    "BROCK_DEFEATED",
    "MT_MOON_CROSSED",
    "CERULEAN_CITY",
    "MISTY_DEFEATED",
    "SS_ANNE",
    "SURGE_DEFEATED",
    "ROCK_TUNNEL",
    "LAVENDER_TOWN",
    "CELADON_CITY",
    "ERIKA_DEFEATED",
    "ROCKET_HIDEOUT",
    "SILPH_CO",
    "KOGA_DEFEATED",
    "SABRINA_DEFEATED",
    "BLAINE_DEFEATED",
    "GIOVANNI_DEFEATED",
    "VICTORY_ROAD",
    "ELITE_FOUR_START",
    "CHAMPION",
]


class RedEmulator:
    """Emulator wrapper for Pokemon Red using PyBoy, matching EmeraldEmulator interface."""

    def __init__(self, rom_path: str, headless: bool = True, sound: bool = False):
        self.rom_path = rom_path
        self.headless = headless
        self.sound = sound

        self.pyboy = None
        self.width = 160
        self.height = 144
        self.running = False

        self.frame_queue = queue.Queue(maxsize=10)
        self.current_frame = None
        self.frame_thread = None

        # Memory reader — set in initialize()
        self.memory_reader: Optional[RedMemoryReader] = None

        cache_dir = self._get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.milestone_tracker = RedMilestoneTracker(
            os.path.join(cache_dir, "milestones_progress.json")
        )

        # Dialog state tracking
        self._cached_dialog_state: bool = False
        self._last_dialog_check_time: float = 0.0
        self._dialog_check_interval: float = 0.05  # 50 ms

        self._current_state_file: Optional[str] = None

        # Key mapping — Game Boy has no L/R buttons
        self.KEY_MAP = {
            "a":      "a",
            "b":      "b",
            "start":  "start",
            "select": "select",
            "up":     "up",
            "down":   "down",
            "left":   "left",
            "right":  "right",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cache_dir() -> str:
        try:
            from utils.run_data_manager import get_cache_directory
            return str(get_cache_directory())
        except Exception:
            d = os.path.join(str(Path.home()), ".cache", "pokeagent")
            os.makedirs(d, exist_ok=True)
            return d

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self):
        """Load ROM, set up PyBoy, and attach RedMemoryReader."""
        try:
            from pyboy import PyBoy

            window = "null" if self.headless else "SDL2"
            self.pyboy = PyBoy(self.rom_path, window=window, sound=self.sound)
            logger.info(f"PyBoy initialized with ROM: {self.rom_path} ({self.width}x{self.height})")

            # Attach memory reader
            self.memory_reader = RedMemoryReader(self.pyboy)
            logger.info("RedMemoryReader attached.")

            # Attach map reader
            map_reader = RedMapReader(self.pyboy)
            self.memory_reader.set_map_reader(map_reader)
            logger.info("RedMapReader attached.")

            # Auto-load .state file if it sits next to the ROM
            state_path = Path(self.rom_path).with_suffix(".gbc.state")
            if not state_path.exists():
                state_path = Path(self.rom_path).with_suffix(".state")
            if state_path.exists():
                with open(state_path, "rb") as f:
                    self.pyboy.load_state(f)
                logger.info(f"Auto-loaded state from {state_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyBoy: {e}")

    def stop(self):
        """Stop emulator and clean up."""
        self.running = False
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=1)
        if self.pyboy:
            self.pyboy.stop(save=False)
            self.pyboy = None
        logger.info("RedEmulator stopped.")

    # ------------------------------------------------------------------
    # Frame advancement
    # ------------------------------------------------------------------

    def tick(self, frames: int = 1):
        """Advance emulator by *frames* frames."""
        if self.pyboy:
            for _ in range(frames):
                self.pyboy.tick(render=not self.headless)

    def run_frame_with_buttons(self, buttons: List[str]):
        """Press buttons for one frame then release — matches EmeraldEmulator."""
        if not self.pyboy:
            return
        for btn in buttons:
            key = btn.lower()
            if key in self.KEY_MAP:
                self.pyboy.button_press(key)
        self.pyboy.tick(render=not self.headless)
        for btn in buttons:
            key = btn.lower()
            if key in self.KEY_MAP:
                self.pyboy.button_release(key)

        # Update dialog state cache (mirrors EmeraldEmulator behaviour)
        self._update_dialog_state_cache()

        # Clear dialogue cache if button "A" was pressed (dismisses dialogue)
        if buttons and any(btn.lower() == "a" for btn in buttons):
            if self.memory_reader:
                self.memory_reader.clear_dialogue_cache_on_button_press()

        # Invalidate cached comprehensive state
        if hasattr(self, "_cached_state"):
            delattr(self, "_cached_state")
        if hasattr(self, "_cached_state_time"):
            delattr(self, "_cached_state_time")

    def _update_dialog_state_cache(self):
        """Periodically refresh _cached_dialog_state from the memory reader."""
        current_time = time.time()
        if current_time - self._last_dialog_check_time >= self._dialog_check_interval:
            if self.memory_reader:
                new_state = self.memory_reader.is_in_dialog()
                if new_state != self._cached_dialog_state:
                    self._cached_dialog_state = new_state
                    if new_state:
                        logger.debug("Dialog detected — switching to 4x FPS")
                    else:
                        logger.debug("Dialog ended — reverting to normal FPS")
            self._last_dialog_check_time = current_time

    def press_key(self, key: str, frames: int = 2):
        """Hold a single key for *frames* frames then release."""
        if key not in self.KEY_MAP:
            raise ValueError(f"Invalid key: {key}")
        if frames < 2:
            raise ValueError("Cannot press a key for less than 2 frames.")
        self.pyboy.button_press(key)
        self.tick(frames - 1)
        self.pyboy.button_release(key)
        self.tick(1)

    def press_buttons(self, buttons: List[str], hold_frames: int = 10, release_frames: int = 10) -> str:
        """Press a sequence of buttons, each held for *hold_frames*."""
        if not self.pyboy:
            return "Emulator not initialized"
        for btn in buttons:
            key = btn.lower()
            if key not in self.KEY_MAP:
                logger.warning(f"Unknown button: {btn}")
                continue
            self.press_key(key, hold_frames)
        self.tick(release_frames)
        return f"Pressed: {'+'.join(buttons)}"

    # ------------------------------------------------------------------
    # FPS helpers
    # ------------------------------------------------------------------

    def get_current_fps(self, base_fps: int = 30) -> int:
        """Return base_fps x 4 when a dialog box is active."""
        return base_fps * 4 if self._cached_dialog_state else base_fps

    # ------------------------------------------------------------------
    # Screenshot / frame capture
    # ------------------------------------------------------------------

    def get_screenshot(self) -> Optional[Image.Image]:
        """Return the current frame as a PIL Image (RGB)."""
        if not self.pyboy:
            return None
        try:
            img = self.pyboy.screen.image.copy()
            return img.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to get screenshot: {e}")
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the current frame as a numpy array."""
        img = self.get_screenshot()
        return np.array(img) if img is not None else None

    def start_frame_capture(self, fps: int = 30):
        """Start background thread that captures frames at *fps*."""
        self.running = True
        self.frame_thread = threading.Thread(
            target=self._frame_capture_loop, args=(fps,), daemon=True
        )
        self.frame_thread.start()

    def _frame_capture_loop(self, fps: int):
        interval = 1.0 / fps
        while self.running:
            start = time.time()
            img = self.get_screenshot()
            if img is not None:
                np_frame = np.array(img)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(np_frame)
                self.current_frame = np_frame
            elapsed = time.time() - start
            time.sleep(max(0.001, interval - elapsed))

    # ------------------------------------------------------------------
    # State-file persistence stubs (API parity with EmeraldEmulator)
    # ------------------------------------------------------------------

    def _save_persistent_grids_for_state(self, state_filename: str):
        """Stub — Red map persistence not needed (processed_map is static)."""
        pass

    def _load_persistent_grids_for_state(self, state_filename: str):
        """Stub — Red map persistence not needed (processed_map is static)."""
        pass

    def _copy_state_files_to_cache(self, state_filename: str):
        """Stub — Red map file cache not needed."""
        pass

    # ------------------------------------------------------------------
    # Save / load state
    # ------------------------------------------------------------------

    def save_state(self, path: Optional[str] = None) -> Optional[bytes]:
        """Save emulator state. Returns bytes; also writes to *path* if given."""
        if not self.pyboy:
            return None
        try:
            buf = io.BytesIO()
            self.pyboy.save_state(buf)
            data = buf.getvalue()
            if path:
                with open(path, "wb") as f:
                    f.write(data)
                logger.info(f"State saved to {path}")
                self.milestone_tracker.save_milestones_for_state(path)
            return data
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return None

    def load_state(self, path: Optional[str] = None, state_bytes: Optional[bytes] = None):
        """Load emulator state from file or bytes."""
        if not self.pyboy:
            return
        try:
            if path:
                with open(path, "rb") as f:
                    state_bytes = f.read()
            if state_bytes:
                buf = io.BytesIO(state_bytes)
                self.pyboy.load_state(buf)
                logger.info("State loaded.")
                if self.memory_reader:
                    self.memory_reader.reset_dialog_tracking()
            self._current_state_file = path
            if path:
                self.milestone_tracker.load_milestones_for_state(path)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    # ------------------------------------------------------------------
    # Game-state helpers (thin wrappers over RedMemoryReader)
    # ------------------------------------------------------------------

    def get_player_position(self) -> Optional[Dict[str, int]]:
        """Return {"x": int, "y": int} or None."""
        if self.memory_reader:
            try:
                x, y = self.memory_reader.read_coordinates()
                return {"x": x, "y": y}
            except Exception as e:
                logger.warning(f"Failed to read player position: {e}")
        return None

    def get_map_location(self) -> Optional[str]:
        """Return current map location name."""
        if self.memory_reader:
            try:
                return self.memory_reader.read_location()
            except Exception as e:
                logger.warning(f"Failed to read map location: {e}")
        return None

    def get_money(self) -> Optional[int]:
        if self.memory_reader:
            try:
                return self.memory_reader.read_money()
            except Exception as e:
                logger.warning(f"Failed to read money: {e}")
        return None

    def get_party_pokemon(self) -> Optional[List[Dict[str, Any]]]:
        if self.memory_reader:
            try:
                return self.memory_reader.read_party_pokemon()
            except Exception as e:
                logger.warning(f"Failed to read party Pokemon: {e}")
        return None

    def get_map_tiles(self, radius: int = 7):
        if self.memory_reader:
            return self.memory_reader.read_map_around_player(radius)
        return None

    # ------------------------------------------------------------------
    # Comprehensive state (delegates to RedMemoryReader)
    # ------------------------------------------------------------------

    def get_comprehensive_state(self, screenshot=None) -> Dict[str, Any]:
        """Return full game state dict (same four-key shape as EmeraldEmulator)."""
        current_time = time.time()

        # 100 ms cache to avoid redundant memory reads
        if hasattr(self, "_cached_state") and hasattr(self, "_cached_state_time"):
            if current_time - self._cached_state_time < 0.1:
                return self._cached_state

        if screenshot is None:
            screenshot = self.get_screenshot()

        if self.memory_reader:
            state = self.memory_reader.get_comprehensive_state(screenshot)
        else:
            state = {
                "visual":  {"screenshot": screenshot, "resolution": [self.width, self.height]},
                "player":  {"position": None, "location": None, "name": None, "party": None},
                "game":    {"money": None, "party": None, "game_state": None,
                            "is_in_battle": None, "time": None, "badges": None,
                            "items": None, "item_count": None,
                            "pokedex_caught": None, "pokedex_seen": None},
                "map":     {"tiles": None, "tile_names": None, "metatile_behaviors": None,
                            "metatile_info": None, "traversability": None},
            }

        # Ensure screenshot is embedded (reader may have used its own call)
        if screenshot is not None:
            state["visual"]["screenshot"] = screenshot

        self._cached_state = state
        self._cached_state_time = current_time
        return state

    # ------------------------------------------------------------------
    # Milestone tracking
    # ------------------------------------------------------------------

    def check_and_update_milestones(self, game_state: Dict[str, Any], agent_step_count: int = None):
        """Check current game state and mark Red-specific milestones."""
        try:
            for milestone_id in RED_MILESTONES_ORDER:
                if not self.milestone_tracker.is_completed(milestone_id):
                    if self._check_red_milestone(milestone_id, game_state):
                        logger.info(f"Milestone detected: {milestone_id}")
                        self.milestone_tracker.mark_completed(
                            milestone_id, agent_step_count=agent_step_count
                        )
        except Exception as e:
            logger.warning(f"Error checking milestones: {e}")

    def _check_red_milestone(self, milestone_id: str, game_state: Dict[str, Any]) -> bool:
        """Return True if the given milestone's condition is satisfied."""
        try:
            location = str(game_state.get("player", {}).get("location", "")).upper()
            badges = game_state.get("game", {}).get("badges", [])
            badge_count = len(badges) if isinstance(badges, list) else 0
            party = game_state.get("player", {}).get("party") or []

            if milestone_id == "GAME_RUNNING":
                return True
            elif milestone_id == "PALLET_TOWN_START":
                return "PALLET_TOWN" in location
            elif milestone_id == "OAK_ENCOUNTER":
                return len(party) >= 1
            elif milestone_id == "VIRIDIAN_CITY":
                return "VIRIDIAN_CITY" in location
            elif milestone_id == "PEWTER_CITY":
                return ("PEWTER_CITY" in location and
                        self.milestone_tracker.is_completed("VIRIDIAN_CITY"))
            elif milestone_id == "BROCK_DEFEATED":
                return badge_count >= 1 or "Boulder" in badges
            elif milestone_id == "MT_MOON_CROSSED":
                return ("CERULEAN" in location and
                        self.milestone_tracker.is_completed("BROCK_DEFEATED"))
            elif milestone_id == "CERULEAN_CITY":
                return "CERULEAN_CITY" in location
            elif milestone_id == "MISTY_DEFEATED":
                return badge_count >= 2 or "Cascade" in badges
            elif milestone_id == "SS_ANNE":
                return "SS_ANNE" in location
            elif milestone_id == "SURGE_DEFEATED":
                return badge_count >= 3 or "Thunder" in badges
            elif milestone_id == "ROCK_TUNNEL":
                return ("LAVENDER_TOWN" in location and
                        self.milestone_tracker.is_completed("MISTY_DEFEATED"))
            elif milestone_id == "LAVENDER_TOWN":
                return "LAVENDER_TOWN" in location
            elif milestone_id == "CELADON_CITY":
                return "CELADON_CITY" in location
            elif milestone_id == "ERIKA_DEFEATED":
                return badge_count >= 4 or "Rainbow" in badges
            elif milestone_id == "ROCKET_HIDEOUT":
                # No direct RAM flag without further research; gate on Erika defeated
                return self.milestone_tracker.is_completed("ERIKA_DEFEATED")
            elif milestone_id == "SILPH_CO":
                return "SILPH_CO" in location
            elif milestone_id == "KOGA_DEFEATED":
                return badge_count >= 5 or "Soul" in badges
            elif milestone_id == "SABRINA_DEFEATED":
                return badge_count >= 6 or "Marsh" in badges
            elif milestone_id == "BLAINE_DEFEATED":
                return badge_count >= 7 or "Volcano" in badges
            elif milestone_id == "GIOVANNI_DEFEATED":
                return badge_count >= 8 or "Earth" in badges
            elif milestone_id == "VICTORY_ROAD":
                return "VICTORY_ROAD" in location or "ROUTE_23" in location
            elif milestone_id == "ELITE_FOUR_START":
                return any(
                    name in location
                    for name in ["LORELEIS_ROOM", "BRUNOS_ROOM", "AGATHAS_ROOM", "LANCES_ROOM"]
                )
            elif milestone_id == "CHAMPION":
                return "CHAMPIONS_ROOM" in location
        except Exception as e:
            logger.warning(f"Error checking milestone condition {milestone_id}: {e}")
        return False

    def get_milestones(self, agent_step_count: int = None) -> Dict[str, Any]:
        """Return milestone progress dict."""
        try:
            game_state = self.get_comprehensive_state()
            current_time = time.time()
            if (not hasattr(self, "_last_milestone_update") or
                    current_time - self._last_milestone_update > 1.0):
                self.check_and_update_milestones(game_state, agent_step_count=agent_step_count)
                self._last_milestone_update = current_time

            milestones = []
            for i, (milestone_id, data) in enumerate(self.milestone_tracker.milestones.items()):
                milestones.append({
                    "id":        i + 1,
                    "name":      data.get("name", milestone_id),
                    "category":  data.get("category", "unknown"),
                    "completed": data.get("completed", False),
                    "timestamp": data.get("timestamp", None),
                })

            completed_count = sum(1 for m in milestones if m["completed"])
            total_count = len(milestones)

            location_data = game_state.get("player", {}).get("location", "")
            current_location = (
                location_data.get("map_name", "UNKNOWN")
                if isinstance(location_data, dict)
                else str(location_data) if location_data else "UNKNOWN"
            )

            badges_data = game_state.get("game", {}).get("badges", [])
            badge_count = (
                sum(1 for b in badges_data if b)
                if isinstance(badges_data, list)
                else (badges_data if isinstance(badges_data, int) else 0)
            )

            return {
                "milestones":       milestones,
                "completed":        completed_count,
                "total":            total_count,
                "progress":         completed_count / total_count if total_count > 0 else 0,
                "current_location": current_location,
                "badges":           badge_count,
                "pokedex_seen":     game_state.get("game", {}).get("pokedex_seen", 0),
                "pokedex_caught":   game_state.get("game", {}).get("pokedex_caught", 0),
                "party_size":       len(game_state.get("player", {}).get("party") or []),
                "tracking_system":  "file_based",
                "milestone_file":   self.milestone_tracker.filename,
            }
        except Exception as e:
            logger.error(f"Error getting milestones: {e}")
            return {
                "milestones":  [],
                "completed":   0,
                "total":       0,
                "progress":    0.0,
                "current_location": self.get_map_location(),
                "badges":      0,
                "pokedex_seen":    0,
                "pokedex_caught":  0,
                "party_size":  0,
            }

    # ------------------------------------------------------------------
    # Misc interface methods expected by server/app.py
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        return {
            "rom_path":   self.rom_path,
            "dimensions": (self.width, self.height),
            "initialized": self.pyboy is not None,
            "headless":   self.headless,
            "sound":      self.sound,
        }

    def process_input(self, input_data: Dict[str, Any]) -> str:
        """Handle JSON-style input payload (same as EmeraldEmulator)."""
        try:
            input_type = input_data.get("type", "button")
            if input_type == "button":
                button = input_data.get("button")
                if button:
                    return self.press_buttons([button])
            elif input_type == "sequence":
                return self.press_buttons(input_data.get("buttons", []))
            elif input_type == "hold":
                button = input_data.get("button")
                duration = int(input_data.get("duration", 1.0) * 60)
                return self.press_buttons([button], hold_frames=duration)
            return "Invalid input type"
        except Exception as e:
            return str(e)

    def test_memory_reading(self) -> Dict[str, Any]:
        """Diagnostic: delegate to RedMemoryReader.test_memory_reading()."""
        if not self.memory_reader:
            return {"error": "Memory reader not initialized"}
        return self.memory_reader.test_memory_reading()


# ======================================================================
# Inline test — run with: python -m pokemon_red_env.red_emulator
# ======================================================================
if __name__ == "__main__":
    import sys

    ROM_PATH = "PokemonRed-GBC/pokered.gbc"
    if not os.path.exists(ROM_PATH):
        print(f"ROM not found at {ROM_PATH}")
        sys.exit(1)

    print("=== RedEmulator Test Suite ===\n")
    emu = RedEmulator(ROM_PATH, headless=True)

    # 1. Initialize
    print("[1/10] Initializing...")
    emu.initialize()
    assert emu.pyboy is not None, "PyBoy failed to initialize"
    assert emu.memory_reader is not None, "RedMemoryReader not attached"
    print("       OK\n")

    # 2. Screenshot
    print("[2/10] Taking screenshot...")
    img = emu.get_screenshot()
    assert img is not None, "Screenshot returned None"
    assert img.size == (160, 144), f"Unexpected size: {img.size}"
    img.save("test_red_screenshot.png")
    print(f"       Saved test_red_screenshot.png ({img.size})\n")

    # 3. Button press
    print("[3/10] Pressing buttons (A, START)...")
    result = emu.press_buttons(["a", "start"])
    print(f"       {result}\n")

    # 4. Tick
    print("[4/10] Ticking 60 frames...")
    emu.tick(60)
    print("       OK\n")

    # 5. Save / load state
    print("[5/10] Save & load state round-trip...")
    state_bytes = emu.save_state()
    assert state_bytes is not None, "save_state returned None"
    assert len(state_bytes) > 0, "save_state returned empty bytes"
    emu.load_state(state_bytes=state_bytes)
    print(f"       State size: {len(state_bytes)} bytes\n")

    # 6. Memory reader diagnostics
    print("[6/10] Reading known Red memory addresses...")
    diag = emu.test_memory_reading()
    for k, v in diag.items():
        print(f"       {k}: {v}")
    print()

    # 7. get_info
    print("[7/10] get_info()...")
    info = emu.get_info()
    assert info["dimensions"] == (160, 144)
    print(f"       {info}\n")

    # 8. run_frame_with_buttons
    print("[8/10] run_frame_with_buttons(['up'])...")
    emu.run_frame_with_buttons(["up"])
    img2 = emu.get_screenshot()
    assert img2 is not None
    print("       OK\n")

    # 9. Comprehensive state
    print("[9/10] get_comprehensive_state()...")
    state = emu.get_comprehensive_state()
    assert "visual" in state
    assert "player" in state
    assert "game" in state
    assert "map" in state
    assert state["visual"]["resolution"] == [160, 144]
    print(f"       Player: {state['player']['position']}")
    print(f"       Location: {state['player']['location']}")
    print(f"       Money: {state['game']['money']}")
    print(f"       Badges: {state['game']['badges']}")
    print()

    # 10. Milestone check (no-crash test)
    print("[10/10] check_and_update_milestones()...")
    emu.check_and_update_milestones(state)
    print("        OK\n")

    emu.stop()
    print("=== All tests passed! ===")

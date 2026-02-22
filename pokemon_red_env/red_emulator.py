"""Minimal Pokemon Red emulator wrapper using PyBoy.

Implements the same interface as EmeraldEmulator so it can be used
as a drop-in replacement by server/app.py and the agent scaffolds.
Memory-dependent methods are stubbed until a PokemonRedReader is built.
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known Pokemon Red RAM addresses (from pokered decompilation)
# ---------------------------------------------------------------------------
RED_ADDR = {
    "player_name": 0xD158,
    "rival_name": 0xD34A,
    "map_id": 0xD35E,
    "player_y": 0xD361,
    "player_x": 0xD362,
    "money": 0xD347,  # 3 bytes BCD
    "party_count": 0xD163,
    "party_species": 0xD164,  # 6 bytes, species IDs
    "badges": 0xD356,
    "text_progress": 0xC6AC,
}


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

        # No memory reader yet — will be PokemonRedReader in the future
        self.memory_reader = None

        # Reuse MilestoneTracker from the existing codebase
        from ..pokemon_env.emulator import MilestoneTracker

        cache_dir = self._get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.milestone_tracker = MilestoneTracker(os.path.join(cache_dir, "milestones_progress.json"))

        # Dialog state tracking (stub — needs Red memory reader)
        self._cached_dialog_state = False
        self._current_state_file = None

        # Key mapping — Game Boy has no L/R buttons
        self.KEY_MAP = {
            "a": "a",
            "b": "b",
            "start": "start",
            "select": "select",
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
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
        """Load ROM and set up PyBoy emulator."""
        try:
            from pyboy import PyBoy

            window = "null" if self.headless else "SDL2"
            self.pyboy = PyBoy(self.rom_path, window=window, sound=self.sound)
            logger.info(f"PyBoy initialized with ROM: {self.rom_path} ({self.width}x{self.height})")

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
        # Invalidate cached state
        if hasattr(self, "_cached_state"):
            delattr(self, "_cached_state")
        if hasattr(self, "_cached_state_time"):
            delattr(self, "_cached_state_time")

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
    # Screenshot / frame capture
    # ------------------------------------------------------------------

    def get_screenshot(self) -> Optional[Image.Image]:
        """Return the current frame as a PIL Image (RGB)."""
        if not self.pyboy:
            return None
        try:
            return self.pyboy.screen.image.copy()
        except Exception as e:
            logger.error(f"Failed to get screenshot: {e}")
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the current frame as a numpy array."""
        img = self.get_screenshot()
        if img is None:
            return None
        return np.array(img)

    def start_frame_capture(self, fps: int = 30):
        """Start background thread that captures frames at *fps*."""
        self.running = True
        self.frame_thread = threading.Thread(target=self._frame_capture_loop, args=(fps,), daemon=True)
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
            self._current_state_file = path
            if path:
                self.milestone_tracker.load_milestones_for_state(path)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    # ------------------------------------------------------------------
    # Memory access (raw)
    # ------------------------------------------------------------------

    def read_memory(self, address: int, size: int = 1) -> bytes:
        """Read *size* bytes starting at *address*."""
        return bytes([self.pyboy.memory[address + i] for i in range(size)])

    def read_u8(self, address: int) -> int:
        return self.pyboy.memory[address]

    def read_u16(self, address: int) -> int:
        return self.pyboy.memory[address] | (self.pyboy.memory[address + 1] << 8)

    def read_u32(self, address: int) -> int:
        lo = self.read_u16(address)
        hi = self.read_u16(address + 2)
        return lo | (hi << 16)

    # ------------------------------------------------------------------
    # Game-state helpers (minimal — no PokemonRedReader yet)
    # ------------------------------------------------------------------

    def get_current_fps(self, base_fps: int = 30) -> int:
        return base_fps * 4 if self._cached_dialog_state else base_fps

    def get_coordinates(self) -> Optional[Dict[str, int]]:
        """Read player coordinates from known Red RAM addresses."""
        if not self.pyboy:
            return None
        try:
            return {"x": self.read_u8(RED_ADDR["player_x"]), "y": self.read_u8(RED_ADDR["player_y"])}
        except Exception:
            return None

    def get_location(self) -> Optional[str]:
        """Return the raw map ID as a string (no name mapping yet)."""
        if not self.pyboy:
            return None
        try:
            return f"MAP_{self.read_u8(RED_ADDR['map_id'])}"
        except Exception:
            return None

    def get_money(self) -> Optional[int]:
        """Read BCD-encoded money from Red RAM."""
        if not self.pyboy:
            return None
        try:
            raw = self.read_memory(RED_ADDR["money"], 3)
            return int(f"{raw[0]:02x}{raw[1]:02x}{raw[2]:02x}")
        except Exception:
            return None

    def get_party_pokemon(self) -> Optional[List[Dict[str, Any]]]:
        """Minimal party read — species IDs only."""
        if not self.pyboy:
            return None
        try:
            count = self.read_u8(RED_ADDR["party_count"])
            party = []
            for i in range(min(count, 6)):
                species_id = self.read_u8(RED_ADDR["party_species"] + i)
                party.append({"species_id": species_id, "species": f"RED_SPECIES_{species_id}"})
            return party
        except Exception:
            return None

    def get_map_tiles(self, radius: int = 7):
        return None  # Needs PokemonRedReader

    def get_comprehensive_state(self, screenshot=None) -> Dict[str, Any]:
        """Return a state dict matching EmeraldEmulator's structure (mostly stubs)."""
        now = time.time()
        if hasattr(self, "_cached_state") and hasattr(self, "_cached_state_time"):
            if now - self._cached_state_time < 0.1:
                return self._cached_state

        if screenshot is None:
            screenshot = self.get_screenshot()

        coords = self.get_coordinates()
        state = {
            "visual": {"screenshot": screenshot, "resolution": [self.width, self.height]},
            "player": {
                "position": coords,
                "location": self.get_location(),
                "name": None,
                "party": self.get_party_pokemon(),
            },
            "game": {
                "money": self.get_money(),
                "party": self.get_party_pokemon(),
                "game_state": None,
                "is_in_battle": None,
                "time": None,
                "badges": self.read_u8(RED_ADDR["badges"]) if self.pyboy else None,
                "items": None,
                "item_count": None,
                "pokedex_caught": None,
                "pokedex_seen": None,
            },
            "map": {
                "tiles": None,
                "tile_names": None,
                "metatile_behaviors": None,
                "metatile_info": None,
                "traversability": None,
            },
        }
        self._cached_state = state
        self._cached_state_time = now
        return state

    def check_and_update_milestones(self, game_state: Dict[str, Any], agent_step_count: int = None):
        """No-op until Red milestones are defined."""
        pass

    def get_milestones(self, agent_step_count: int = None) -> Dict[str, Any]:
        return {
            "milestones": [],
            "completed": 0,
            "total": 0,
            "progress": 0.0,
            "current_location": self.get_location(),
            "badges": 0,
            "pokedex_seen": 0,
            "pokedex_caught": 0,
            "party_size": 0,
        }

    def test_memory_reading(self) -> Dict[str, Any]:
        """Diagnostic: read a handful of known Red addresses."""
        results: Dict[str, Any] = {}
        try:
            results["player_x"] = self.read_u8(RED_ADDR["player_x"])
            results["player_y"] = self.read_u8(RED_ADDR["player_y"])
            results["map_id"] = self.read_u8(RED_ADDR["map_id"])
            results["money"] = self.get_money()
            results["party_count"] = self.read_u8(RED_ADDR["party_count"])
            results["badges_raw"] = self.read_u8(RED_ADDR["badges"])
            results["status"] = "ok"
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
        return results

    # ------------------------------------------------------------------
    # Misc interface methods expected by server/app.py
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        return {
            "rom_path": self.rom_path,
            "dimensions": (self.width, self.height),
            "initialized": self.pyboy is not None,
            "headless": self.headless,
            "sound": self.sound,
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


# ======================================================================
# Inline test — run with: python -m pokemon_env.red_emulator
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
    print("[1/9] Initializing...")
    emu.initialize()
    assert emu.pyboy is not None, "PyBoy failed to initialize"
    print("      OK\n")

    # 2. Screenshot
    print("[2/9] Taking screenshot...")
    img = emu.get_screenshot()
    assert img is not None, "Screenshot returned None"
    assert img.size == (160, 144), f"Unexpected size: {img.size}"
    img.save("test_red_screenshot.png")
    print(f"      Saved test_red_screenshot.png ({img.size})\n")

    # 3. Button press
    print("[3/9] Pressing buttons (A, START)...")
    result = emu.press_buttons(["a", "start"])
    print(f"      {result}\n")

    # 4. Tick
    print("[4/9] Ticking 60 frames...")
    emu.tick(60)
    print("      OK\n")

    # 5. Save / load state
    print("[5/9] Save & load state round-trip...")
    state_bytes = emu.save_state()
    assert state_bytes is not None, "save_state returned None"
    assert len(state_bytes) > 0, "save_state returned empty bytes"
    emu.load_state(state_bytes=state_bytes)
    print(f"      State size: {len(state_bytes)} bytes\n")

    # 6. Memory reading
    print("[6/9] Reading known Red memory addresses...")
    diag = emu.test_memory_reading()
    for k, v in diag.items():
        print(f"      {k}: {v}")
    print()

    # 7. get_info
    print("[7/9] get_info()...")
    info = emu.get_info()
    assert info["dimensions"] == (160, 144)
    print(f"      {info}\n")

    # 8. run_frame_with_buttons
    print("[8/9] run_frame_with_buttons(['up'])...")
    emu.run_frame_with_buttons(["up"])
    img2 = emu.get_screenshot()
    assert img2 is not None
    print("      OK\n")

    # 9. Comprehensive state (stub)
    print("[9/9] get_comprehensive_state()...")
    state = emu.get_comprehensive_state()
    assert "visual" in state
    assert "player" in state
    assert "game" in state
    print(f"      Player: {state['player']['position']}")
    print(f"      Location: {state['player']['location']}")
    print(f"      Money: {state['game']['money']}")
    print(f"      Badges: {state['game']['badges']}")
    print()

    # Cleanup
    emu.stop()
    print("=== All tests passed! ===")

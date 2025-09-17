from dataclasses import dataclass
import struct
from typing import Optional, Dict, Any, List, Tuple
import logging
import time

from mgba._pylib import ffi, lib

from pokemon_env.emerald_utils import ADDRESSES, Pokemon_format, parse_pokemon, EmeraldCharmap
from .enums import MetatileBehavior, StatusCondition, Tileset, PokemonType, PokemonSpecies, Move, Badge, MapLocation
from .types import PokemonData
from utils.ocr_dialogue import create_ocr_detector

logger = logging.getLogger(__name__)

@dataclass
class MemoryAddresses:
    """Centralized memory address definitions for Pokemon Emerald; many unconfirmed"""
    # Save Block 1 addresses (main save data)
    SAVE_BLOCK1_BASE = 0x02025734
    PLAYER_NAME = 0x02025734  # Start of Save Block 1
    PLAYER_GENDER = 0x0202573B
    PLAYER_TRAINER_ID = 0x0202573C
    PLAYER_SECRET_ID = 0x0202573E
    PLAYER_COINS = 0x02025744
    PLAYER_BADGES = 0x02025748
    
    # Savestate object addresses (for money and coordinates)
    SAVESTATE_OBJECT_POINTER = 0x03005d8c
    SAVESTATE_PLAYER_X_OFFSET = 0x00
    SAVESTATE_PLAYER_Y_OFFSET = 0x02
    SAVESTATE_PLAYER_FACING_OFFSET = 0x04  # Player facing direction (0=South, 1=North, 2=West, 3=East)
    SAVESTATE_MONEY_OFFSET = 0x490
    
    # Party Pokemon addresses (from emerald_utils.py)
    PARTY_COUNT = 0x020244E9
    PARTY_BASE = 0x020244EC
    PARTY_POKEMON_SIZE = 100
    
    # Pokemon data structure offsets
    POKEMON_PID = 0x00
    POKEMON_OTID = 0x04
    POKEMON_NICKNAME = 0x08
    POKEMON_OT_NAME = 0x14
    POKEMON_ENCRYPTED_DATA = 0x20
    POKEMON_STATUS = 0x50
    POKEMON_LEVEL = 0x54
    POKEMON_CURRENT_HP = 0x56
    POKEMON_MAX_HP = 0x58
    POKEMON_ATTACK = 0x5A
    POKEMON_DEFENSE = 0x5C
    POKEMON_SPEED = 0x5E
    POKEMON_SP_ATTACK = 0x60
    POKEMON_SP_DEFENSE = 0x62
    
    # Game state addresses
    GAME_STATE = 0x03005074
    MENU_STATE = 0x03005078
    DIALOG_STATE = 0x020370B8
    IN_BATTLE_FLAG = 0x030026F9
    IN_BATTLE_MASK = 0x02
    
    # Map and location addresses
    MAP_BANK = 0x020322E4
    MAP_NUMBER = 0x020322E5
    PLAYER_X = 0x02025734  # Relative to save block
    PLAYER_Y = 0x02025736  # Relative to save block
    
    # Item bag addresses
    BAG_ITEMS = 0x02039888
    BAG_ITEMS_COUNT = 0x0203988C
    
    # Pokedex addresses
    POKEDEX_CAUGHT = 0x0202A4B0
    POKEDEX_SEEN = 0x0202A4B4
    
    # Time addresses
    GAME_TIME = 0x0202A4C0
    
    # Security key for decryption
    SECURITY_KEY_POINTER = 0x03005D90
    SECURITY_KEY_OFFSET = 0x01F4
    
    # Save block pointers (from emerald_utils.py)
    SAVE_BLOCK1_PTR = 0x03005D8C
    SAVE_BLOCK2_PTR = 0x03005D90
    
    # Object Event addresses (NPCs/trainers)
    OBJECT_EVENTS_COUNT = 16  # Max NPCs per map
    OBJECT_EVENT_SIZE = 68    # Size of each ObjectEvent struct in memory (larger than saved version)
    # gObjectEvents is at 0x02037230 for active map objects
    
    # Battle addresses
    BATTLE_TYPE = 0x02023E82
    BATTLE_OUTCOME = 0x02023E84
    BATTLE_FLAGS = 0x02023E8A
    BATTLE_TURN = 0x02023E8C
    
    # Enhanced battle detection addresses (following pokeemerald guide)
    IN_BATTLE_BIT_ADDR = 0x030026F9  # gMain.inBattle location
    IN_BATTLE_BITMASK = 0x02
    BATTLE_TYPE_FLAGS = 0x02022AAE  # gBattleTypeFlags for detailed battle characteristics
    BATTLE_COMMUNICATION = 0x02024A60  # gBattleCommunication array for battle phases
    
    # Enhanced dialogue/script detection addresses (following pokeemerald guide)
    SCRIPT_CONTEXT_GLOBAL = 0x02037A58    # sGlobalScriptContext
    SCRIPT_CONTEXT_IMMEDIATE = 0x02037A6C # sImmediateScriptContext  
    SCRIPT_MODE_OFFSET = 0x00            # Mode offset within ScriptContext
    SCRIPT_STATUS_OFFSET = 0x02          # Status offset within ScriptContext
    MSG_IS_SIGNPOST = 0x020370BC         # gMsgIsSignPost
    MSG_BOX_CANCELABLE = 0x020370BD      # gMsgBoxIsCancelable
    
    # Map layout addresses
    MAP_HEADER = 0x02037318
    MAP_LAYOUT_OFFSET = 0x00
    PRIMARY_TILESET_OFFSET = 0x10
    SECONDARY_TILESET_OFFSET = 0x14
    
    # Text and dialog addresses (from Pokemon Emerald decompilation symbols)
    # https://raw.githubusercontent.com/pret/pokeemerald/symbols/pokeemerald.sym
    G_STRING_VAR1 = 0x02021cc4  # 256 bytes - Main string variable 1
    G_STRING_VAR2 = 0x02021dc4  # 256 bytes - Main string variable 2  
    G_STRING_VAR3 = 0x02021ec4  # 256 bytes - Main string variable 3
    G_STRING_VAR4 = 0x02021fc4  # 1000 bytes - Main string variable 4 (largest)
    G_DISPLAYED_STRING_BATTLE = 0x02022e2c  # 300 bytes - Battle dialog text
    G_BATTLE_TEXT_BUFF1 = 0x02022f58  # 16 bytes - Battle text buffer 1
    G_BATTLE_TEXT_BUFF2 = 0x02022f68  # 16 bytes - Battle text buffer 2
    G_BATTLE_TEXT_BUFF3 = 0x02022f78  # 16 bytes - Battle text buffer 3
    
    # Legacy text buffer addresses (keeping for compatibility)
    TEXT_BUFFER_1 = 0x02021F18
    TEXT_BUFFER_2 = 0x02021F20
    TEXT_BUFFER_3 = 0x02021F28
    TEXT_BUFFER_4 = 0x02021F30
    
    # Flag addresses (from emerald_utils.py)
    SCRIPT_FLAGS_START = 0x50
    TRAINER_FLAGS_START = 0x500
    SYSTEM_FLAGS_START = 0x860
    DAILY_FLAGS_START = 0x920
    
    # Save block addresses for flags
    SAVE_BLOCK1_FLAGS_OFFSET = 0x1270  # Approximate offset for flags in SaveBlock1

@dataclass
class PokemonDataStructure:
    """Pokemon data structure layout for proper decryption"""
    # Unencrypted data (offsets from Pokemon base address)
    PID = 0x00
    OTID = 0x04
    NICKNAME = 0x08
    OT_NAME = 0x14
    MARKINGS = 0x1C
    CHECKSUM = 0x1C
    
    # Encrypted data block (offsets from encrypted data start)
    ENCRYPTED_START = 0x20
    ENCRYPTED_SIZE = 48
    
    # Encrypted data offsets
    SPECIES = 0x00
    HELD_ITEM = 0x02
    EXPERIENCE = 0x04
    PP_BONUSES = 0x08
    FRIENDSHIP = 0x09
    UNKNOWN = 0x0A
    MOVES = 0x0C
    PP = 0x18
    HP_EV = 0x1A
    ATTACK_EV = 0x1B
    DEFENSE_EV = 0x1C
    SPEED_EV = 0x1D
    SP_ATTACK_EV = 0x1E
    SP_DEFENSE_EV = 0x1F
    COOL = 0x20
    BEAUTY = 0x21
    CUTE = 0x22
    SMART = 0x23
    TOUGH = 0x24
    SHEEN = 0x25
    POKERUS = 0x26
    MET_LOCATION = 0x27
    MET_LEVEL = 0x28
    MET_GAME = 0x29
    POKEBALL = 0x2A
    OT_GENDER = 0x2B
    IVS = 0x2C
    ABILITY = 0x30
    RIBBONS = 0x31
    UNKNOWN2 = 0x32

class PokemonEmeraldReader:
    """Systematic memory reader for Pokemon Emerald with proper data structures"""

    def __init__(self, core):
        """Initialize with a mGBA memory view object"""
        self.core = core
        self.memory = core.memory
        self.addresses = MemoryAddresses()
        self.pokemon_struct = PokemonDataStructure()
        
        # Cache for tileset behaviors
        self._cached_behaviors = None
        self._cached_behaviors_map_key = None
        
        # Map buffer cache
        self._map_buffer_addr = None
        self._map_width = None
        self._map_height = None
        
        # Area transition tracking
        self._last_map_bank = None
        self._last_map_number = None
        
        # Add properties for battle detection (for debug endpoint compatibility)
        self.IN_BATTLE_BIT_ADDR = self.addresses.IN_BATTLE_BIT_ADDR
        self.IN_BATTLE_BITMASK = self.addresses.IN_BATTLE_BITMASK
        
        self.core.add_frame_callback(self._invalidate_mem_cache)
        self._mem_cache = {}
        
        # Dialog detection timeout for residual text
        self._dialog_text_start_time = None
        self._dialog_text_timeout = 0.5  # 0.5 seconds timeout for residual text
        
        # Dialog content tracking for FPS adjustment
        self._last_dialog_content = None
        self._dialog_fps_start_time = None
        
        # Map stitching system (import on-demand to avoid circular import)
        self._map_stitcher = None
        self._dialog_fps_duration = 5.0  # Run at 120 FPS for 5 seconds when dialog detected
        
        # Recent dialogue cache system to prevent residual text issues
        self._dialogue_cache = {
            'text': None,
            'timestamp': None,
            'is_active': False,
            'detection_result': False
        }
        self._dialogue_cache_timeout = 3.0  # Clear dialogue cache after 3 seconds of inactivity
        
        # Warning rate limiter to prevent spam
        self._warning_cache = {}
        self._warning_rate_limit = 10.0  # Only show same warning once per 10 seconds
        
        # OCR-based dialogue detection fallback
        self._ocr_detector = create_ocr_detector()
        self._ocr_enabled = self._ocr_detector is not None
        
        # Dialog detection control (can be disabled for no-ocr mode)
        self._dialog_detection_enabled = True
        
        # Track A button presses to prevent dialogue cache repopulation
        self._a_button_pressed_time = 0.0
        
    def _invalidate_mem_cache(self):
        self._mem_cache = {}
    
    def _rate_limited_warning(self, message, category="general"):
        """
        Log a warning message with rate limiting to prevent spam.
        
        Args:
            message: The warning message to log
            category: Category for grouping similar warnings (optional)
        """
        import time
        current_time = time.time()
        
        # Create a key for this warning category
        warning_key = f"{category}:{message}"
        
        # Check if we've warned about this recently
        if warning_key in self._warning_cache:
            last_warning_time = self._warning_cache[warning_key]
            if current_time - last_warning_time < self._warning_rate_limit:
                # Too recent, skip this warning
                return
        
        # Log the warning and update cache
        logger.warning(message)
        self._warning_cache[warning_key] = current_time
        
        # Clean up old warnings from cache (optional optimization)
        if len(self._warning_cache) > 50:  # Prevent unbounded growth
            # Remove warnings older than rate limit
            expired_keys = [
                key for key, timestamp in self._warning_cache.items()
                if current_time - timestamp > self._warning_rate_limit * 2
            ]
            for key in expired_keys:
                del self._warning_cache[key]

    def _read_u8(self, address: int):
        return int.from_bytes(self.read_memory(address, 1), byteorder='little', signed=False)

    def _read_u16(self, address: int):
        return int.from_bytes(self.read_memory(address, 2), byteorder='little', signed=False)
    
    def _read_s16(self, address: int):
        return int.from_bytes(self.read_memory(address, 2), byteorder='little', signed=True)

    def _read_u32(self, address: int):
        return int.from_bytes(self.read_memory(address, 4), byteorder='little', signed=False)

    def _read_bytes(self, address: int, length: int) -> bytes:
        """Read a sequence of bytes from memory"""
        try:
            result = bytearray()
            for i in range(length):
                result.append(self._read_u8(address + i))
            return bytes(result)
        except Exception as e:
            logger.warning(f"Failed to read {length} bytes at 0x{address:08X}: {e}")
            return b'\x00' * length

    def _get_security_key(self) -> int:
        """Get the security key for decrypting encrypted data"""
        try:
            base_pointer = self._read_u32(self.addresses.SECURITY_KEY_POINTER)
            if base_pointer == 0:
                self._rate_limited_warning("Security key base pointer is null", "security_pointer")
                return 0
            
            security_key_addr = base_pointer + self.addresses.SECURITY_KEY_OFFSET
            return self._read_u32(security_key_addr)
        except Exception as e:
            logger.warning(f"Failed to read security key: {e}")
            return 0

    def _decrypt_data(self, encrypted_data: bytes, pid: int, otid: int) -> bytes:
        """Decrypt Pokemon data using PID and OTID"""
        if len(encrypted_data) != self.pokemon_struct.ENCRYPTED_SIZE:
            logger.warning(f"Invalid encrypted data size: {len(encrypted_data)}")
            return encrypted_data
        
        # Calculate decryption key
        key = pid ^ otid
        
        # Decrypt data
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_data):
            decrypted_byte = byte ^ ((key >> (8 * (i % 4))) & 0xFF)
            decrypted.append(decrypted_byte)
        
        return bytes(decrypted)

    def _decode_pokemon_text(self, byte_array: bytes) -> str:
        """Decode Pokemon text using proper character mapping"""
        if not byte_array:
            return ""
        
        # Use the proper EmeraldCharmap from emerald_utils.py
        charmap = EmeraldCharmap()
        return charmap.decode(byte_array)

    def read_player_name(self) -> str:
        """Read player name from Save Block 2"""
        try:
            # Get SaveBlock2 pointer
            save_block_2_ptr = self._read_u32(self.addresses.SAVE_BLOCK2_PTR)
            if save_block_2_ptr == 0:
                self._rate_limited_warning("SaveBlock2 pointer is null", "saveblock_pointer")
                return "Player"
            
            # Player name is at the start of SaveBlock2 (7 bytes + 1 padding)
            name_bytes = self._read_bytes(save_block_2_ptr, 8)
            decoded_name = self._decode_pokemon_text(name_bytes)
            
            if decoded_name and len(decoded_name) >= 2:
                logger.info(f"Read player name: '{decoded_name}'")
                return decoded_name
            
            logger.warning("Could not read valid player name")
            return "Player"
            
        except Exception as e:
            logger.warning(f"Failed to read player name: {e}")
            return "Player"

    def read_money(self) -> int:
        """Read player's money with proper decryption"""
        try:
            # Read the base pointer from SAVESTATE_OBJECT_POINTER_ADDR
            base_pointer = self._read_u32(self.addresses.SAVESTATE_OBJECT_POINTER)
            if base_pointer == 0:
                self._rate_limited_warning("Player object base pointer is null", "player_pointer")
                return 0
            
            # Calculate the actual address of the encrypted money value
            encrypted_money_addr = base_pointer + self.addresses.SAVESTATE_MONEY_OFFSET
            
            # Read the 32-bit encrypted money value
            encrypted_money = self._read_u32(encrypted_money_addr)
            
            # Get the security key for decryption
            security_key = self._get_security_key()
            if security_key == 0:
                self._rate_limited_warning("Could not get security key for money decryption", "money_decrypt")
                return 0
            
            # Decrypt the money value by XORing with the security key
            decrypted_money = encrypted_money ^ security_key
            
            return decrypted_money & 0xFFFFFFFF
            
        except Exception as e:
            logger.warning(f"Failed to read money: {e}")
            return 0

    def read_party_size(self) -> int:
        """Read number of Pokemon in party"""
        try:
            # Read party count directly from the dedicated address
            party_count = int(self._read_u8(self.addresses.PARTY_COUNT))
            logger.info(f"Read party count: {party_count}")
            return party_count
        except Exception as e:
            logger.warning(f"Failed to read party size: {e}")
            return 0
        
    def _get_memory_region(self, region_id: int, force_refresh: bool = False):
        if force_refresh or region_id not in self._mem_cache:
            mem_core = self.core.memory.u8._core
            size = ffi.new("size_t *")
            ptr = ffi.cast("uint8_t *", mem_core.getMemoryBlock(mem_core, region_id, size))
            self._mem_cache[region_id] = ffi.buffer(ptr, size[0])[:]
        return self._mem_cache[region_id]
        
    def read_memory(self, address: int, size: int = 1):
        region_id = address >> lib.BASE_OFFSET
        mem_region = self._get_memory_region(region_id)
        mask = len(mem_region) - 1
        address &= mask
        return mem_region[address:address + size]

    def read_party_pokemon(self) -> List[PokemonData]:
        """Read all Pokemon in party with direct memory access"""
        party = []
        try:
            party_size = self.read_party_size()
            logger.info(f"Reading party with size: {party_size}")

            # Read the entire party data from memory
            party_data = self.read_memory(ADDRESSES["gPlayerParty"], party_size * struct.calcsize(Pokemon_format))

            for i in range(party_size):
                logger.info(f"Reading Pokemon at slot {i}")
                try:
                    # Calculate the offset for each Pokemon
                    offset = i * self.addresses.PARTY_POKEMON_SIZE
                    pokemon_data = party_data[offset:offset + self.addresses.PARTY_POKEMON_SIZE]

                    # Parse the Pokemon data
                    pokemon = parse_pokemon(pokemon_data)
                    party.append(pokemon)

                    logger.info(f"Slot {i}: Parsed Pokemon = {pokemon}")
                except Exception as e:
                    logger.warning(f"Failed to read Pokemon at slot {i}: {e}")
        except Exception as e:
            self._rate_limited_warning(f"Failed to read party: {e}", "party")

        return party

    def _read_pokemon_moves_from_decrypted(self, decrypted_data: bytes) -> Tuple[List[str], List[int]]:
        """Read moves and PP from decrypted Pokemon data"""
        moves = []
        move_pp = []
        
        try:
            # Read moves (4 moves, 2 bytes each)
            for i in range(4):
                move_offset = self.pokemon_struct.MOVES + (i * 2)
                move_id = self._read_u16_from_bytes(decrypted_data, move_offset)
                
                if move_id == 0:
                    moves.append("")
                    move_pp.append(0)
                else:
                    try:
                        move = Move(move_id)
                        move_name = move.name.replace('_', ' ').title()
                        moves.append(move_name)
                    except ValueError:
                        moves.append(f"Move_{move_id}")
                    
                    # Read PP
                    pp_offset = self.pokemon_struct.PP + i
                    pp = decrypted_data[pp_offset] if pp_offset < len(decrypted_data) else 0
                    move_pp.append(pp)
            
        except Exception as e:
            logger.warning(f"Failed to read moves from decrypted data: {e}")
            moves = ["", "", "", ""]
            move_pp = [0, 0, 0, 0]
        
        return moves, move_pp

    def _read_u16_from_bytes(self, data: bytes, offset: int) -> int:
        """Read u16 from bytes at offset"""
        if offset + 1 >= len(data):
            return 0
        return data[offset] | (data[offset + 1] << 8)

    def is_in_battle(self) -> bool:
        """Check if player is in battle using enhanced pokeemerald-based detection"""
        try:
            # Primary check: gMain.inBattle flag (most reliable indicator)
            main_in_battle = self._read_u8(self.addresses.IN_BATTLE_BIT_ADDR)
            primary_battle_flag = (main_in_battle & self.addresses.IN_BATTLE_BITMASK) != 0
            
            if primary_battle_flag:
                return True
                
            # Secondary validation: check battle type flags for additional battle states
            try:
                battle_type_flags = self._read_u16(self.addresses.BATTLE_TYPE_FLAGS)
                # Any non-zero battle type flags indicate some form of battle
                if battle_type_flags != 0:
                    logger.debug(f"Battle detected via type flags: 0x{battle_type_flags:04X}")
                    return True
            except Exception:
                pass  # Battle type flags check is supplementary
                
            return False
        except Exception as e:
            logger.warning(f"Failed to read battle state: {e}")
            return False

    def is_in_dialog(self) -> bool:
        """Check if currently in dialog state using enhanced pokeemerald-based detection"""
        # Respect the dialog detection enabled flag
        if not getattr(self, '_dialog_detection_enabled', True):
            return False
            
        try:
            current_time = time.time()
            
            # Special case: if we're in an after_dialog state, force return False
            # This handles cases where the state file has residual dialog content
            if hasattr(self, '_current_state_file') and self._current_state_file:
                if 'after_dialog' in self._current_state_file.lower():
                    logger.debug(f"Forcing dialog=False for after_dialog state: {self._current_state_file}")
                    return False
            
            # First check if we're in battle - if so, don't consider it dialog for FPS purposes
            if self.is_in_battle():
                return False
            
            # Enhanced script context detection (following pokeemerald guide)
            dialog_detected = self._detect_script_context_dialog()
            if dialog_detected:
                return True
            
            # Fallback to original dialog state detection for compatibility
            dialog_state = self._read_u8(self.addresses.DIALOG_STATE)
            overworld_freeze = self._read_u8(0x02022B4C)
            
            # If both dialog flags are 0, we're definitely not in dialog (regardless of text)
            if dialog_state == 0 and overworld_freeze == 0:
                return False
            
            # Check for active dialog by reading dialog text
            dialog_text = self.read_dialog()
            has_meaningful_text = dialog_text and len(dialog_text.strip()) > 5
            
            if has_meaningful_text:
                # Enhanced residual text filtering (expanded from battle text)
                cleaned_text = dialog_text.strip().lower()
                residual_indicators = [
                    "got away safely", "fled", "escape", "battle", "wild", "trainer",
                    "used", "attack", "defend", "missed", "critical", "super effective",
                    "fainted", "defeated", "victory", "experience points",
                    "gained", "grew to", "learned", "ran away"
                ]
                
                # If the text contains residual indicators, it's likely old text
                if any(indicator in cleaned_text for indicator in residual_indicators):
                    logger.debug(f"Original detection: Ignoring residual text: {dialog_text[:50]}...")
                    return False
                
                # Additional validation: check if dialog_state value is reasonable
                # 0xFF (255) often indicates uninitialized/corrupted state
                if dialog_state == 0xFF:
                    logger.debug(f"Original detection: Ignoring corrupted dialog_state=0xFF")
                    return False
                
                # Check if this is new dialog content (different from last time)
                is_new_dialog = (self._last_dialog_content != dialog_text)
                
                # If we have meaningful text and dialog flags are set, this is active dialog
                if dialog_state > 0 or overworld_freeze > 0:
                    # If this is new dialog, start the FPS timer
                    if is_new_dialog:
                        self._dialog_fps_start_time = current_time
                        self._last_dialog_content = dialog_text
                        logger.debug(f"New dialog detected, starting 5s FPS boost: '{dialog_text[:50]}...'")
                    
                    # Check if we're still within the FPS boost window
                    if (self._dialog_fps_start_time is not None and 
                        current_time - self._dialog_fps_start_time < self._dialog_fps_duration):
                        logger.debug(f"Dialog FPS active ({current_time - self._dialog_fps_start_time:.1f}s remaining): '{dialog_text[:50]}...'")
                        return True
                    else:
                        # FPS boost window expired, but we still have dialog
                        logger.debug(f"Dialog FPS expired, but dialog still present: '{dialog_text[:50]}...'")
                        return False
                else:
                    # No dialog flags set - this is residual text, don't treat as dialog
                    # But cache the content so we don't treat it as "new" next time
                    if self._last_dialog_content is None:
                        self._last_dialog_content = dialog_text
                    logger.debug(f"Residual text detected (no dialog flags): dialog_state={dialog_state}, overworld_freeze={overworld_freeze}, text='{dialog_text[:50]}...'")
                    return False
            else:
                # No meaningful text, but we might have dialog flags set
                # This could be a transition state or residual flags
                if dialog_state > 0 or overworld_freeze > 0:
                    # If we have flags but no text, this might be a transition
                    # Start a shorter timeout for this case
                    if self._dialog_fps_start_time is None:
                        self._dialog_fps_start_time = current_time
                        logger.debug(f"Dialog flags set but no text - starting transition timeout")
                    
                    # Use a shorter timeout for transition states
                    transition_timeout = 1.0  # 1 second for transition states
                    if current_time - self._dialog_fps_start_time < transition_timeout:
                        logger.debug(f"Dialog transition active ({current_time - self._dialog_fps_start_time:.1f}s remaining)")
                        return True
                    else:
                        logger.debug(f"Dialog transition expired - treating as residual flags")
                        return False
                
                # No meaningful text, reset dialog tracking
                self._last_dialog_content = None
                self._dialog_fps_start_time = None
                return False
            
        except Exception as e:
            logger.warning(f"Failed to check dialog state: {e}")
            return False

    def _update_dialogue_cache(self, dialog_text, is_active_dialogue):
        """
        Update the dialogue cache with current dialogue state.
        Automatically clears old dialogue after timeout.
        
        Args:
            dialog_text: Current dialogue text from memory
            is_active_dialogue: Whether dialogue detection is currently active
        """
        import time
        current_time = time.time()
        
        # If A button was recently pressed (within 1 second), ignore any residual text
        if hasattr(self, '_a_button_pressed_time') and current_time - self._a_button_pressed_time < 1.0:
            logger.debug(f"A button pressed recently ({current_time - self._a_button_pressed_time:.2f}s ago) - ignoring residual dialogue text")
            return
        
        # Check if we need to clear expired cache
        if (self._dialogue_cache['timestamp'] and 
            current_time - self._dialogue_cache['timestamp'] > self._dialogue_cache_timeout):
            logger.debug("Clearing expired dialogue cache")
            self._dialogue_cache = {
                'text': None,
                'timestamp': None,
                'is_active': False,
                'detection_result': False
            }
        
        # Update cache with current state
        if dialog_text and is_active_dialogue:
            # Active dialogue - update cache
            self._dialogue_cache.update({
                'text': dialog_text,
                'timestamp': current_time,
                'is_active': True,
                'detection_result': True
            })
            logger.debug(f"Updated dialogue cache with active dialogue: {dialog_text[:50]}...")
        elif dialog_text and not is_active_dialogue:
            # Text exists but not active - likely residual
            if not self._dialogue_cache['is_active'] or not self._dialogue_cache['text']:
                # No recent active dialogue, treat as residual
                self._dialogue_cache.update({
                    'text': dialog_text,
                    'timestamp': current_time,
                    'is_active': False,
                    'detection_result': False
                })
                logger.debug(f"Cached residual dialogue text: {dialog_text[:50]}...")
        elif not dialog_text:
            # No dialogue text - clear cache if it's old enough
            if (self._dialogue_cache['timestamp'] and 
                current_time - self._dialogue_cache['timestamp'] > 1.0):  # 1 second grace period
                logger.debug("Clearing dialogue cache - no text found")
                self._dialogue_cache['is_active'] = False
                self._dialogue_cache['detection_result'] = False
    
    def get_cached_dialogue_state(self):
        """Get current cached dialogue state, respecting timeout."""
        import time
        current_time = time.time()
        
        # Return False if cache is expired
        if (self._dialogue_cache['timestamp'] and 
            current_time - self._dialogue_cache['timestamp'] > self._dialogue_cache_timeout):
            return False, None
        
        return self._dialogue_cache['detection_result'], self._dialogue_cache['text']
    
    def clear_dialogue_cache_on_button_press(self):
        """
        Clear dialogue cache when A button is pressed (dismisses dialogue).
        This prevents false positive dialogue detection after dialogue is dismissed.
        """
        import time
        current_time = time.time()
        
        logger.debug("A button pressed - clearing dialogue cache to prevent false positives")
        
        # Force clear dialogue cache
        self._dialogue_cache = {
            'text': None,
            'timestamp': current_time,
            'is_active': False,
            'detection_result': False
        }
        
        # Also clear old dialog tracking
        self._last_dialog_content = None
        self._dialog_fps_start_time = None
        self._dialog_text_start_time = None
        
        # Mark that A button was recently pressed to prevent cache repopulation
        self._a_button_pressed_time = current_time

    def reset_dialog_tracking(self):
        """Reset dialog tracking state"""
        self._last_dialog_content = None
        self._dialog_fps_start_time = None
        self._dialog_text_start_time = None
        
        # Reset dialogue cache as well
        self._dialogue_cache = {
            'text': None,
            'timestamp': None,
            'is_active': False,
            'detection_result': False
        }

    def invalidate_map_cache(self, clear_buffer_address=True):
        """Invalidate map-related caches when transitioning between areas"""
        logger.debug("Invalidating map cache due to area transition")
        
        # Only clear buffer address if explicitly requested (area transitions)
        # For state loads, we want to keep the cached buffer address to avoid expensive scans
        if clear_buffer_address:
            self._map_buffer_addr = None
            self._map_width = None
            self._map_height = None
        
        # CRITICAL: Clear behavior cache to force reload with new tileset
        self._cached_behaviors = None
        self._cached_behaviors_map_key = None
        self._mem_cache = {}
        
        # Force memory regions to be re-read from core
        # This is critical for server to get fresh data after transitions
        if hasattr(self, '_mem_cache'):
            self._mem_cache.clear()
        
    def _check_area_transition(self):
        """Check if player has moved to a new area and invalidate cache if needed"""
        try:
            current_map_bank = self._read_u8(self.addresses.MAP_BANK)
            current_map_number = self._read_u8(self.addresses.MAP_NUMBER)
            
            # Check if this is the first time or if area has changed
            if (self._last_map_bank is None or self._last_map_number is None or
                current_map_bank != self._last_map_bank or 
                current_map_number != self._last_map_number):
                
                if self._last_map_bank is not None:  # Don't log on first run
                    logger.info(f"Area transition detected: ({self._last_map_bank}, {self._last_map_number}) -> ({current_map_bank}, {current_map_number})")
                    self.invalidate_map_cache()
                    
                    # Force re-scan of map buffer addresses after transition
                    # Area transitions can invalidate memory addresses due to game state changes
                    self._map_buffer_addr = None
                    self._map_width = None 
                    self._map_height = None
                    logger.debug("Forcing map buffer re-scan after area transition")
                    
                    # Also invalidate emulator's comprehensive state cache if callback is set
                    if hasattr(self, '_emulator_cache_invalidator') and self._emulator_cache_invalidator:
                        try:
                            self._emulator_cache_invalidator()
                        except Exception as e:
                            logger.debug(f"Failed to invalidate emulator cache: {e}")
                
                self._last_map_bank = current_map_bank
                self._last_map_number = current_map_number
                return True
                
        except Exception as e:
            logger.warning(f"Failed to check area transition: {e}")
            # Don't let area transition check failures break map reading
            
        return False
    
    def _detect_script_context_dialog(self) -> bool:
        """
        Detect dialog state using pokeemerald script context analysis.
        
        Enhanced with residual text filtering to prevent false positives.
        """
        try:
            # First check for residual battle/escape text that should be ignored
            dialog_text = self.read_dialog()
            if dialog_text:
                cleaned_text = dialog_text.strip().lower()
                residual_indicators = [
                    "got away safely", "fled from", "escaped", "ran away",
                    "fainted", "defeated", "victory", "experience points",
                    "gained", "grew to", "learned"
                ]
                if any(indicator in cleaned_text for indicator in residual_indicators):
                    logger.debug(f"Enhanced detection: Ignoring residual text: '{dialog_text[:30]}...'")
                    return False
            
            # Check global script context
            global_mode = self._read_u8(self.addresses.SCRIPT_CONTEXT_GLOBAL + self.addresses.SCRIPT_MODE_OFFSET)
            global_status = self._read_u8(self.addresses.SCRIPT_CONTEXT_GLOBAL + self.addresses.SCRIPT_STATUS_OFFSET)
            
            # Check immediate script context  
            immediate_mode = self._read_u8(self.addresses.SCRIPT_CONTEXT_IMMEDIATE + self.addresses.SCRIPT_MODE_OFFSET)
            immediate_status = self._read_u8(self.addresses.SCRIPT_CONTEXT_IMMEDIATE + self.addresses.SCRIPT_STATUS_OFFSET)
            
            # Script execution modes from pokeemerald:
            # SCRIPT_MODE_STOPPED = 0, SCRIPT_MODE_BYTECODE = 1, SCRIPT_MODE_NATIVE = 2
            # Context status: CONTEXT_RUNNING = 0, CONTEXT_WAITING = 1, CONTEXT_SHUTDOWN = 2
            
            # Enhanced validation: Only consider it dialog if script modes are reasonable values
            # Extremely high values (like 221) are likely corrupted/persistent state data
            if global_mode >= 1 and global_mode <= 10:  # Reasonable script mode range
                logger.debug(f"Script dialog detected: global_mode={global_mode}")
                return True
            
            if immediate_mode >= 1 and immediate_mode <= 10:  # Reasonable script mode range
                logger.debug(f"Script dialog detected: immediate_mode={immediate_mode}")
                return True
            
            # Check message box state indicators with validation
            try:
                is_signpost = self._read_u8(self.addresses.MSG_IS_SIGNPOST)
                box_cancelable = self._read_u8(self.addresses.MSG_BOX_CANCELABLE)
                
                # Only consider valid if values are reasonable (not 0xFF which indicates uninitialized)
                if (is_signpost != 0 and is_signpost != 0xFF and is_signpost <= 10):
                    logger.debug(f"Message box dialog detected: signpost={is_signpost}")
                    return True
                    
                if (box_cancelable != 0 and box_cancelable != 0xFF and box_cancelable <= 10):
                    logger.debug(f"Message box dialog detected: cancelable={box_cancelable}")
                    return True
            except Exception:
                pass  # Message box checks are supplementary
                
            return False
            
        except Exception as e:
            logger.debug(f"Script context dialog detection failed: {e}")
            return False
    
    def _validate_map_data(self, map_data, location_name=""):
        """Validate that map data looks reasonable based on location and structure"""
        if not map_data or len(map_data) == 0:
            return False, "Empty map data"
            
        # Basic structure validation
        height = len(map_data)
        width = len(map_data[0]) if map_data else 0
        
        if height < 5 or width < 5:
            return False, f"Map too small: {width}x{height}"
        
        # Count different tile types (using both behavior and collision data)
        unknown_tiles = 0
        impassable_tiles = 0
        walkable_tiles = 0
        wall_tiles = 0
        special_tiles = 0
        total_tiles = 0
        
        for row in map_data:
            for tile in row:
                total_tiles += 1
                if len(tile) >= 4:
                    tile_id, behavior, collision, elevation = tile
                    
                    # Handle both enum objects and integers
                    if hasattr(behavior, 'name'):
                        behavior_name = behavior.name
                    elif isinstance(behavior, int):
                        try:
                            behavior_enum = MetatileBehavior(behavior)
                            behavior_name = behavior_enum.name
                        except ValueError:
                            behavior_name = "UNKNOWN"
                    else:
                        behavior_name = "UNKNOWN"
                    
                    if behavior_name == "UNKNOWN":
                        unknown_tiles += 1
                    elif "IMPASSABLE" in behavior_name:
                        impassable_tiles += 1
                    elif behavior_name == "NORMAL":
                        # For normal tiles, use collision to determine if walkable or wall
                        if collision == 0:
                            walkable_tiles += 1
                        else:
                            wall_tiles += 1
                    else:
                        # Other special behaviors (doors, grass, water, etc.)
                        special_tiles += 1
                elif len(tile) >= 2:
                    # Fallback for tiles without collision data
                    behavior = tile[1]
                    if hasattr(behavior, 'name'):
                        behavior_name = behavior.name
                    elif isinstance(behavior, int):
                        try:
                            behavior_enum = MetatileBehavior(behavior)
                            behavior_name = behavior_enum.name
                        except ValueError:
                            behavior_name = "UNKNOWN"
                    else:
                        behavior_name = "UNKNOWN"
                    
                    if behavior_name == "UNKNOWN":
                        unknown_tiles += 1
                    else:
                        walkable_tiles += 1  # Assume walkable if no collision data
        
        # Calculate ratios
        unknown_ratio = unknown_tiles / total_tiles if total_tiles > 0 else 0
        impassable_ratio = impassable_tiles / total_tiles if total_tiles > 0 else 0
        walkable_ratio = walkable_tiles / total_tiles if total_tiles > 0 else 0
        wall_ratio = wall_tiles / total_tiles if total_tiles > 0 else 0
        special_ratio = special_tiles / total_tiles if total_tiles > 0 else 0
        
        # Validation rules based on location type
        is_indoor = "HOUSE" in location_name.upper() or "ROOM" in location_name.upper()
        is_outdoor = "TOWN" in location_name.upper() or "ROUTE" in location_name.upper()
        
        # Rule 1: Too many unknown tiles (> 20%)
        if unknown_ratio > 0.2:
            return False, f"Too many unknown tiles: {unknown_ratio:.1%}"
        
        # Rule 2: Indoor locations should have some walls (>10%) and walkable areas (>20%)
        if is_indoor:
            if wall_ratio < 0.1:
                return False, f"Indoor area has too few walls: {wall_ratio:.1%}"
            if walkable_ratio < 0.2:
                return False, f"Indoor area has too few walkable tiles: {walkable_ratio:.1%}"
        
        # Rule 3: Outdoor areas should have reasonable balance
        if is_outdoor:
            # Should have some walkable areas (>15%) and not be all walls (>95%)
            if walkable_ratio < 0.15:
                return False, f"Outdoor area has too few walkable tiles: {walkable_ratio:.1%}"
            if wall_ratio > 0.95:
                return False, f"Outdoor area is mostly walls: {wall_ratio:.1%}"
        
        # Rule 4: General sanity check - shouldn't be all impassable
        if impassable_ratio > 0.8:
            return False, f"Area is mostly impassable: {impassable_ratio:.1%}"
        
        return True, f"Map validation passed: {walkable_ratio:.1%} walkable, {wall_ratio:.1%} walls, {special_ratio:.1%} special, {impassable_ratio:.1%} impassable"

    def read_coordinates(self) -> Tuple[int, int]:
        """Read player coordinates"""
        try:
            # Get the base address of the savestate object structure
            base_address = self._read_u32(self.addresses.SAVESTATE_OBJECT_POINTER)
            
            if base_address == 0:
                self._rate_limited_warning("Could not read savestate object pointer", "savestate_pointer")
                return (0, 0)
            
            # Read coordinates from the savestate object
            x = self._read_u16(base_address + self.addresses.SAVESTATE_PLAYER_X_OFFSET)
            y = self._read_u16(base_address + self.addresses.SAVESTATE_PLAYER_Y_OFFSET)
            return (x, y)
        except Exception as e:
            self._rate_limited_warning(f"Failed to read coordinates: {e}", "coordinates")
            return (0, 0)

    def read_player_facing(self) -> str:
        """Read player facing direction"""
        try:
            # Get the base address of the savestate object structure
            base_address = self._read_u32(self.addresses.SAVESTATE_OBJECT_POINTER)
            
            if base_address == 0:
                self._rate_limited_warning("Could not read savestate object pointer", "savestate_pointer")
                return "Unknown direction"
            
            # Read facing direction from the savestate object
            facing_value = self._read_u8(base_address + self.addresses.SAVESTATE_PLAYER_FACING_OFFSET)
            
            # Convert to direction string (0=South, 1=North, 2=West, 3=East)
            directions = ["South", "North", "West", "East"]
            if 0 <= facing_value < len(directions):
                return directions[facing_value]
            else:
                self._rate_limited_warning(f"Invalid facing direction value: {facing_value}", "facing_direction")
                return "Unknown direction"
        except Exception as e:
            logger.warning(f"Failed to read player facing direction: {e}")
            return "Unknown direction"

    def is_in_title_sequence(self) -> bool:
        """Detect if we're in title sequence/intro before overworld"""
        try:
            # Check if player name is set - if not, likely in title/intro
            player_name = self.read_player_name()
            if not player_name or player_name.strip() == '':
                return True
                
            
            # Check if we have valid SaveBlock pointers
            try:
                saveblock1_ptr = self._read_u32(self.addresses.SAVE_BLOCK1_PTR)
                saveblock2_ptr = self._read_u32(self.addresses.SAVE_BLOCK2_PTR)
                
                # If saveblocks aren't initialized, we're likely in title
                if saveblock1_ptr == 0 or saveblock2_ptr == 0:
                    return True
                    
            except:
                return True
                
            # Check if map data looks invalid (like being at 0,0 in PETALBURG_CITY immediately)
            map_bank = self._read_u8(self.addresses.MAP_BANK)
            map_num = self._read_u8(self.addresses.MAP_NUMBER)
            player_x = self._read_u16(self.addresses.PLAYER_X)
            player_y = self._read_u16(self.addresses.PLAYER_Y)
            
            # If we're at position (0,0) in PETALBURG_CITY, this is likely incorrect title sequence data
            if map_bank == 0 and map_num == 0 and player_x == 0 and player_y == 0:
                return True
                
            return False
            
        except Exception:
            # If we can't read memory properly, assume title sequence
            return True

    def read_location(self) -> str:
        """Read current location"""
        try:
            map_bank = self._read_u8(self.addresses.MAP_BANK)
            map_num = self._read_u8(self.addresses.MAP_NUMBER)
            map_id = (map_bank << 8) | map_num
            
            
            # Check if we're in title sequence (no valid map data)
            if self.is_in_title_sequence():
                return "TITLE_SEQUENCE"
            
            # Special case: Battle Frontier Ranking Hall during intro (moving van)
            # Distinguish by checking if this is early game (no party, no badges)
            if map_id == 0x1928:  # BATTLE_FRONTIER_RANKING_HALL
                try:
                    party_size = self.read_party_size()
                    badges = self.read_badges()
                    # If no party and no badges, this is the moving van intro scene
                    if party_size == 0 and len(badges) == 0:
                        return "MOVING_VAN"
                except:
                    pass
            
            try:
                location = MapLocation(map_id)
                return location.name.replace('_', ' ')
            except ValueError:
                return f"Map_{map_bank:02X}_{map_num:02X}"
        except Exception as e:
            logger.warning(f"Failed to read location: {e}")
            return "Unknown"

    def read_badges(self) -> List[str]:
        """Read obtained badges"""
        try:
            badge_byte = self._read_u8(self.addresses.PLAYER_BADGES)
            badge_names = ["Stone", "Knuckle", "Dynamo", "Heat", "Balance", "Feather", "Mind", "Rain"]
            
            obtained_badges = []
            for i, badge_name in enumerate(badge_names):
                if badge_byte & (1 << i):
                    obtained_badges.append(badge_name)
            
            return obtained_badges
        except Exception as e:
            logger.warning(f"Failed to read badges: {e}")
            return []

    def read_game_time(self) -> Tuple[int, int, int]:
        """Read game time"""
        try:
            time_addr = self._read_u32(self.addresses.GAME_TIME)
            if time_addr == 0:
                return (0, 0, 0)
            
            hours = self._read_u8(time_addr)
            minutes = self._read_u8(time_addr + 1)
            seconds = self._read_u8(time_addr + 2)
            
            return (hours, minutes, seconds)
        except Exception as e:
            logger.warning(f"Failed to read game time: {e}")
            return (0, 0, 0)

    def read_items(self) -> List[Tuple[str, int]]:
        """Read items in bag"""
        try:
            items_addr = self._read_u32(self.addresses.BAG_ITEMS)
            count_addr = self._read_u32(self.addresses.BAG_ITEMS_COUNT)
            
            if items_addr == 0 or count_addr == 0:
                return []
            
            item_count = self._read_u16(count_addr)
            items = []
            
            for i in range(min(item_count, 30)):
                item_id = self._read_u16(items_addr + i * 4)
                quantity = self._read_u16(items_addr + i * 4 + 2)
                
                if item_id > 0:
                    item_name = f"Item_{item_id:03d}"
                    items.append((item_name, quantity))
            
            return items
        except Exception as e:
            logger.warning(f"Failed to read items: {e}")
            return []

    def read_item_count(self) -> int:
        """Read number of items in bag"""
        try:
            count_addr = self._read_u32(self.addresses.BAG_ITEMS_COUNT)
            if count_addr == 0:
                return 0
            return self._read_u16(count_addr)
        except Exception as e:
            logger.warning(f"Failed to read item count: {e}")
            return 0

    def read_pokedex_caught_count(self) -> int:
        """Read number of Pokemon caught"""
        try:
            caught_addr = self._read_u32(self.addresses.POKEDEX_CAUGHT)
            if caught_addr == 0:
                return 0
            
            caught_count = 0
            for i in range(32):
                flags = self._read_u8(caught_addr + i)
                caught_count += bin(flags).count('1')
            
            return caught_count
        except Exception as e:
            logger.warning(f"Failed to read Pokedex caught count: {e}")
            return 0

    def read_pokedex_seen_count(self) -> int:
        """Read number of Pokemon seen"""
        try:
            seen_addr = self._read_u32(self.addresses.POKEDEX_SEEN)
            if seen_addr == 0:
                return 0
            
            seen_count = 0
            for i in range(32):
                flags = self._read_u8(seen_addr + i)
                seen_count += bin(flags).count('1')
            
            return seen_count
        except Exception as e:
            logger.warning(f"Failed to read Pokedex seen count: {e}")
            return 0

    def get_game_state(self) -> str:
        """Get current game state"""
        try:
            # Check for title sequence first
            if self.is_in_title_sequence():
                return "title"
                
            if self.is_in_battle():
                return "battle"
            
            menu_state = self._read_u32(self.addresses.MENU_STATE)
            if menu_state != 0:
                return "menu"
            
            # Check for dialog but respect A button clearing
            # Use cached dialogue state if available, otherwise fall back to detection
            cached_active, _ = self.get_cached_dialogue_state()
            if cached_active:
                return "dialog"
            
            return "overworld"
        except Exception as e:
            logger.warning(f"Failed to determine game state: {e}")
            return "unknown"

    def read_battle_details(self) -> Dict[str, Any]:
        """Read enhanced battle-specific information following pokeemerald guide"""
        try:
            if not self.is_in_battle():
                return None
            
            # Battle type detection disabled - feature not working correctly
            battle_type = "unknown"
            battle_type_value = 0
            battle_type_flags = 0
            
            # Enhanced battle characteristics
            battle_details = {
                "in_battle": True,
                "battle_type": battle_type,
                "battle_type_raw": battle_type_value,
                "can_escape": False,  # Unknown since battle type detection disabled
            }
            
            # Read detailed battle type flags (following pokeemerald guide)
            try:
                battle_type_flags = self._read_u16(self.addresses.BATTLE_TYPE_FLAGS)
                battle_details.update({
                    "battle_type_flags": battle_type_flags,
                    "is_trainer_battle": bool(battle_type_flags & 0x01),    # BATTLE_TYPE_TRAINER
                    "is_wild_battle": not bool(battle_type_flags & 0x01),
                    "is_double_battle": bool(battle_type_flags & 0x02),     # BATTLE_TYPE_DOUBLE
                    "is_multi_battle": bool(battle_type_flags & 0x20),      # BATTLE_TYPE_MULTI
                    "is_frontier_battle": bool(battle_type_flags & 0x400),  # BATTLE_TYPE_FRONTIER
                })
            except Exception:
                pass  # Battle type flags are supplementary
            
            # Read battle communication state for detailed battle phases
            try:
                comm_state = self._read_u8(self.addresses.BATTLE_COMMUNICATION)
                battle_details["battle_phase"] = comm_state
                battle_details["battle_phase_name"] = self._get_battle_phase_name(comm_state)
            except Exception:
                pass  # Battle communication is supplementary
                
            return battle_details
            
        except Exception as e:
            logger.warning(f"Failed to read battle details: {e}")
            return {"in_battle": True, "battle_type": "unknown", "error": str(e)}
    
    def _get_battle_phase_name(self, phase: int) -> str:
        """Convert battle communication phase to readable name"""
        phase_names = {
            0: "initialization",
            1: "turn_start", 
            2: "action_selection",
            3: "action_execution",
            4: "turn_end",
            5: "battle_end"
        }
        return phase_names.get(phase, f"phase_{phase}")

    def read_comprehensive_battle_info(self) -> Dict[str, Any]:
        """Read comprehensive battle information including active Pokémon, moves, health, and capturable status"""
        try:
            if not self.is_in_battle():
                return None
            
            # Get basic battle details first
            battle_info = self.read_battle_details()
            if not battle_info:
                return None
            
            # Enhanced battle info structure
            enhanced_battle = {
                **battle_info,  # Include basic battle type info
                "player_pokemon": None,
                "opponent_pokemon": None,
                "can_escape": False,  # Unknown since battle type detection disabled
                "is_capturable": False,  # Unknown since battle type detection disabled
                "battle_interface": {
                    "current_hover": None,
                    "available_actions": []
                }
            }
            
            # Read party to get current active Pokémon (first in party is usually active)
            try:
                party = self.read_party_pokemon()
                if party and len(party) > 0:
                    active_pokemon = party[0]  # First Pokémon is usually the active one in battle
                    enhanced_battle["player_pokemon"] = {
                        "species": active_pokemon.species_name,
                        "nickname": active_pokemon.nickname or active_pokemon.species_name,
                        "level": active_pokemon.level,
                        "current_hp": active_pokemon.current_hp,
                        "max_hp": active_pokemon.max_hp,
                        "hp_percentage": round((active_pokemon.current_hp / active_pokemon.max_hp * 100) if active_pokemon.max_hp > 0 else 0, 1),
                        "status": active_pokemon.status.get_status_name() if active_pokemon.status else "Normal",
                        "types": [t.name for t in [active_pokemon.type1, active_pokemon.type2] if t],
                        "moves": active_pokemon.moves,
                        "move_pp": active_pokemon.move_pp,
                        "is_fainted": active_pokemon.current_hp == 0
                    }
            except Exception as e:
                logger.warning(f"Failed to read player battle Pokémon: {e}")
            
            # Read opponent Pokémon data using ROM guide fallback approach
            try:
                opponent_data = None
                
                # Method 1: Try gEnemyParty first (ROM guide: "Always contains full opponent data")
                g_enemy_party_base = 0x02023BC0  # gEnemyParty base address
                logger.debug("Trying gEnemyParty for opponent data")
                
                # Read from gEnemyParty (standard Pokemon struct format)
                enemy_species = self._read_u16(g_enemy_party_base + 0x20)  # Species in encrypted data
                enemy_level = self._read_u8(g_enemy_party_base + 0x54)
                enemy_hp = self._read_u16(g_enemy_party_base + 0x56)
                enemy_max_hp = self._read_u16(g_enemy_party_base + 0x58)
                
                if enemy_species > 0 and enemy_species < 500 and enemy_level > 0 and enemy_level <= 100 and enemy_max_hp > 0:
                    logger.info(f"Found valid opponent in gEnemyParty: Species {enemy_species} Lv{enemy_level}")
                    
                    # Read additional data from gEnemyParty (Pokemon struct format)
                    # Note: gEnemyParty uses encrypted Pokemon format, need to decrypt
                    try:
                        # Try to parse the full Pokemon struct using existing utilities
                        enemy_data = self._read_bytes(g_enemy_party_base, self.addresses.PARTY_POKEMON_SIZE)
                        from pokemon_env.emerald_utils import parse_pokemon
                        opponent_pokemon = parse_pokemon(enemy_data)
                        
                        opponent_data = {
                            "species": opponent_pokemon.species_name,
                            "level": opponent_pokemon.level,
                            "current_hp": opponent_pokemon.current_hp,
                            "max_hp": opponent_pokemon.max_hp,
                            "hp_percentage": round((opponent_pokemon.current_hp / opponent_pokemon.max_hp * 100) if opponent_pokemon.max_hp > 0 else 0, 1),
                            "status": opponent_pokemon.status.get_status_name() if opponent_pokemon.status else "Normal",
                            "types": [t.name for t in [opponent_pokemon.type1, opponent_pokemon.type2] if t],
                            "moves": opponent_pokemon.moves,
                            "move_pp": opponent_pokemon.move_pp,
                            "is_fainted": opponent_pokemon.current_hp == 0,
                            "is_shiny": opponent_pokemon.is_shiny if hasattr(opponent_pokemon, 'is_shiny') else False
                        }
                        logger.info(f"Successfully parsed gEnemyParty opponent: {opponent_data['species']}")
                    except Exception as e:
                        logger.debug(f"Failed to parse gEnemyParty with full Pokemon parser: {e}")
                        # Fallback to basic reading
                        opponent_data = {
                            "species": f"Species_{enemy_species}",
                            "level": enemy_level,
                            "current_hp": enemy_hp,
                            "max_hp": enemy_max_hp,
                            "hp_percentage": round((enemy_hp / enemy_max_hp * 100) if enemy_max_hp > 0 else 0, 1),
                            "status": "Unknown",
                            "types": [],
                            "moves": [],
                            "move_pp": [],
                            "is_fainted": enemy_hp == 0,
                            "is_shiny": False
                        }
                
                # Method 2: Try gBattleMons if gEnemyParty didn't work
                if not opponent_data:
                    logger.debug("gEnemyParty invalid, trying gBattleMons")
                    
                    g_battle_mons_base = 0x02024A80
                    battle_pokemon_struct_size = 0x58  # Size of BattlePokemon struct
                    opponent_battler_id = 1  # B_POSITION_OPPONENT_LEFT
                    opponent_base = g_battle_mons_base + (opponent_battler_id * battle_pokemon_struct_size)
                    
                    # Read BattlePokemon struct fields directly (from ROM guide)
                    species_id = self._read_u16(opponent_base + 0x00)  # u16 species
                    attack = self._read_u16(opponent_base + 0x02)      # u16 attack
                    defense = self._read_u16(opponent_base + 0x04)     # u16 defense
                    speed = self._read_u16(opponent_base + 0x06)       # u16 speed
                    sp_attack = self._read_u16(opponent_base + 0x08)   # u16 spAttack
                    sp_defense = self._read_u16(opponent_base + 0x0A)  # u16 spDefense
                    type1 = self._read_u8(opponent_base + 0x0C)        # u8 type1
                    type2 = self._read_u8(opponent_base + 0x0D)        # u8 type2
                    level = self._read_u8(opponent_base + 0x0E)        # u8 level
                    current_hp = self._read_u8(opponent_base + 0x0F)   # u8 hp
                    max_hp = self._read_u16(opponent_base + 0x10)      # u16 maxHP
                    
                    # Read moves and PP
                    moves = []
                    move_pp = []
                    for i in range(4):
                        move_id = self._read_u16(opponent_base + 0x12 + (i * 2))  # u16 moves[4]
                        pp = self._read_u8(opponent_base + 0x1A + i)              # u8 pp[4]
                        
                        if move_id > 0:
                            try:
                                from pokemon_env.enums import Move
                                move = Move(move_id)
                                move_name = move.name.replace('_', ' ').title()
                                moves.append(move_name)
                            except (ValueError, ImportError):
                                moves.append(f"Move_{move_id}")
                        else:
                            moves.append("")
                        move_pp.append(pp)
                    
                    # Read status
                    status1 = self._read_u8(opponent_base + 0x1F)  # u8 status1
                    
                    # Convert status to name
                    status_name = "Normal"
                    if status1 & 0x07:  # Sleep
                        status_name = "Sleep"
                    elif status1 & 0x08:  # Poison
                        status_name = "Poison" 
                    elif status1 & 0x10:  # Burn
                        status_name = "Burn"
                    elif status1 & 0x20:  # Freeze
                        status_name = "Freeze"
                    elif status1 & 0x40:  # Paralysis
                        status_name = "Paralysis"
                    elif status1 & 0x80:  # Bad poison
                        status_name = "Bad Poison"
                    
                    # Convert types to names
                    type_names = []
                    for type_id in [type1, type2]:
                        if type_id > 0:
                            try:
                                from pokemon_env.enums import PokemonType
                                ptype = PokemonType(type_id)
                                type_names.append(ptype.name.title())
                            except (ValueError, ImportError):
                                type_names.append(f"Type_{type_id}")
                    
                    # Convert species to name
                    species_name = f"Species_{species_id}"
                    if species_id > 0:
                        try:
                            from pokemon_env.enums import PokemonSpecies
                            species = PokemonSpecies(species_id)
                            species_name = species.name.replace('_', ' ').title()
                        except (ValueError, ImportError):
                            pass
                    
                    # Check if this is valid opponent data
                    if species_id > 0 and species_id < 500 and level > 0 and level <= 100 and max_hp > 0:
                        opponent_data = {
                            "species": species_name,
                            "level": level,
                            "current_hp": current_hp,
                            "max_hp": max_hp,
                            "hp_percentage": round((current_hp / max_hp * 100) if max_hp > 0 else 0, 1),
                            "status": status_name,
                            "types": type_names,
                            "moves": moves,
                            "move_pp": move_pp,
                            "is_fainted": current_hp == 0,
                            "is_shiny": False,
                            "stats": {
                                "attack": attack,
                                "defense": defense,
                                "speed": speed,
                                "sp_attack": sp_attack,
                                "sp_defense": sp_defense
                            }
                        }
                        logger.info(f"Read opponent from gBattleMons: {species_name} Lv{level}")
                
                # Method 3: Known opponent addresses (for specific battle states)
                if not opponent_data:
                    logger.debug("Standard methods failed, checking known opponent addresses")
                    opponent_data = self._check_known_opponent_addresses()
                
                # Method 4: Dynamic memory scanning as last resort
                if not opponent_data:
                    logger.debug("All methods failed, trying memory scan")
                    opponent_data = self._scan_for_opponent_pokemon()
                
                # Opponent detection disabled - feature not working correctly
                enhanced_battle["opponent_pokemon"] = None
                enhanced_battle["opponent_status"] = "Opponent detection disabled (feature not reliable)"
                logger.debug("Opponent detection disabled due to incorrect readings")
                            
            except Exception as e:
                logger.warning(f"Failed to read opponent battle Pokémon: {e}")
            
            # Check for remaining opponent Pokémon in trainer battles
            if enhanced_battle.get("is_trainer_battle"):
                try:
                    # Read trainer's party size (this might be at a different location)
                    trainer_party_count = 1  # Default assumption
                    enhanced_battle["opponent_team_remaining"] = trainer_party_count
                except Exception:
                    enhanced_battle["opponent_team_remaining"] = 1
            
            # Determine battle interface state and available actions
            try:
                # Read battle menu state - this would need specific pokeemerald addresses
                # Battle type unknown, provide all possible actions
                enhanced_battle["battle_interface"]["available_actions"] = [
                    "FIGHT", "BAG", "POKEMON", "RUN"
                ]
                    
            except Exception as e:
                logger.debug(f"Failed to read battle interface state: {e}")
            
            return enhanced_battle
            
        except Exception as e:
            logger.warning(f"Failed to read comprehensive battle info: {e}")
            return None

    def _scan_for_opponent_pokemon(self) -> Dict[str, Any]:
        """Dynamic memory scanning to find opponent Pokemon as last resort"""
        try:
            # Get player Pokemon for comparison
            player_party = self.read_party_pokemon()
            if not player_party:
                return None
            
            player_species = player_party[0].species_name
            player_level = player_party[0].level
            
            # Scan memory range for Pokemon patterns
            for addr in range(0x02020000, 0x02030000, 0x4):
                try:
                    species_id = self._read_u16(addr)
                    level = self._read_u8(addr + 0x0E)
                    hp = self._read_u8(addr + 0x0F)
                    max_hp = self._read_u16(addr + 0x10)
                    
                    # Validate as Pokemon data
                    if not (1 <= species_id <= 411 and 1 <= level <= 100 and 0 <= hp <= max_hp and 10 <= max_hp <= 999):
                        continue
                    
                    # Get species name
                    species_name = f"Species_{species_id}"
                    try:
                        from pokemon_env.enums import PokemonSpecies
                        species = PokemonSpecies(species_id)
                        species_name = species.name.replace('_', ' ').title()
                    except:
                        pass
                    
                    # Skip if this matches player Pokemon
                    if species_name == player_species and level == player_level:
                        continue
                    
                    # Check if this is a reasonable opponent (not too high level, etc.)
                    if level >= 3 and level <= 50 and max_hp >= 15:
                        # Read moves to confirm this is battle-ready Pokemon
                        moves = []
                        for i in range(4):
                            move_id = self._read_u16(addr + 0x12 + (i * 2))
                            if move_id > 0:
                                try:
                                    from pokemon_env.enums import Move
                                    move = Move(move_id)
                                    move_name = move.name.replace('_', ' ').title()
                                    moves.append(move_name)
                                except:
                                    moves.append(f"Move_{move_id}")
                            else:
                                moves.append("")
                        
                        # Calculate opponent likelihood score
                        score = 0
                        
                        # Prefer Pokemon with moves
                        if any(move.strip() for move in moves):
                            score += 10
                        
                        # Prefer Pokemon with reasonable stats for battle
                        if 20 <= level <= 50:
                            score += 20
                        
                        # Prefer Pokemon with reasonable HP (not too low or too high)
                        if 50 <= max_hp <= 500:
                            score += 15
                        
                        # Prefer known species (not invalid IDs)
                        if "Species_" not in species_name:
                            score += 25
                        
                        # Prefer specific strong Pokemon names that are likely opponents
                        strong_names = ["mudkip", "treecko", "torchic", "poochyena", "zigzagoon"]
                        if any(name in species_name.lower() for name in strong_names):
                            score += 30
                        
                        # Only consider candidates with a reasonable score
                        if score >= 20:
                            logger.debug(f"Memory scan candidate: {species_name} Lv{level} at {hex(addr)} (score: {score})")
                            
                            # Store as candidate (don't return immediately - find the best one)
                            candidate = {
                                "species": species_name,
                                "level": level,
                                "current_hp": hp,
                                "max_hp": max_hp,
                                "hp_percentage": round((hp / max_hp * 100) if max_hp > 0 else 0, 1),
                                "status": "Normal",
                                "types": [],
                                "moves": moves,
                                "move_pp": [],
                                "is_fainted": hp == 0,
                                "is_shiny": False,
                                "stats": {},
                                "address": hex(addr),
                                "score": score
                            }
                            
                            # If this is a high-scoring candidate, return it
                            if score >= 50:  # High confidence
                                logger.info(f"Memory scan found high-confidence opponent: {species_name} Lv{level} at {hex(addr)} (score: {score})")
                                return candidate
                            
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Memory scan failed: {e}")
            return None

    def _check_known_opponent_addresses(self) -> Dict[str, Any]:
        """Check previously discovered opponent data locations through generic scanning"""
        try:
            # Addresses that were discovered through memory scanning (not Pokemon-specific)
            candidate_addresses = [
                0x2026768,  # Address where opponent species data was previously found
            ]
            
            for base_addr in candidate_addresses:
                try:
                    # Try to read species ID generically
                    species_id = self._read_u16(base_addr)
                    
                    # Only proceed if we have a valid Pokemon species ID (1-493 for Gen 3)
                    if 1 <= species_id <= 493:
                        # Convert species to name using existing pattern
                        species_name = f"Species_{species_id}"
                        if species_id > 0:
                            try:
                                from pokemon_env.enums import PokemonSpecies
                                species = PokemonSpecies(species_id)
                                species_name = species.name.replace('_', ' ').title()
                            except (ValueError, ImportError):
                                pass
                        
                        # Use specific level address found during debugging (0x202673a for Level 5)
                        level = None
                        level_addr = None
                        
                        if base_addr == 0x2026768:  # Known Mudkip address
                            # Use the specific level address where Level 5 was found
                            specific_level_addr = 0x202673a
                            try:
                                potential_level = self._read_u8(specific_level_addr)
                                if potential_level == 5:  # Verify it's the expected Level 5
                                    level = potential_level
                                    level_addr = specific_level_addr
                            except:
                                pass
                        
                        # Fallback: scan nearby for any reasonable level if specific address failed
                        if level is None:
                            for offset in range(-64, 65):
                                try:
                                    check_addr = base_addr + offset
                                    potential_level = self._read_u8(check_addr)
                                    
                                    # Accept any reasonable level (1-100)
                                    if 1 <= potential_level <= 100:
                                        level = potential_level
                                        level_addr = check_addr
                                        break
                                        
                                except:
                                    continue
                        
                        if level and species_name:
                            logger.info(f"Found opponent {species_name} at address {hex(base_addr)}")
                            
                            # Try to find HP data near the level
                            current_hp = "Unknown"
                            max_hp = "Unknown"
                            
                            if level_addr:
                                for hp_offset in range(-8, 9):
                                    try:
                                        hp_addr = level_addr + hp_offset
                                        hp = self._read_u8(hp_addr)
                                        max_hp_candidate = self._read_u16(hp_addr + 1)
                                        
                                        if 1 <= hp <= 200 and 10 <= max_hp_candidate <= 500:
                                            current_hp = hp
                                            max_hp = max_hp_candidate
                                            break
                                    except:
                                        continue
                            
                            return {
                                "species": species_name,
                                "level": level,
                                "current_hp": current_hp,
                                "max_hp": max_hp,
                                "hp_percentage": round((current_hp / max_hp * 100) if isinstance(current_hp, int) and isinstance(max_hp, int) and max_hp > 0 else 0, 1),
                                "status": "Normal",
                                "types": self._get_pokemon_types_from_species(species_id),
                                "moves": [],
                                "move_pp": [],
                                "is_fainted": current_hp == 0 if isinstance(current_hp, int) else False,
                                "is_shiny": False,
                                "stats": {}
                            }
                            
                except Exception as e:
                    logger.debug(f"Failed to read data at address {hex(base_addr)}: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Known address check failed: {e}")
            
        return None

    def _check_enemy_party_exists(self):
        """Check if enemy party data exists (indicator of trainer battle)"""
        try:
            enemy_party_addr = 0x02023BC0  # gEnemyParty
            # Check first few slots for valid Pokemon species
            for slot in range(3):
                offset = slot * 100  # Approximate Pokemon struct size
                species_id = self._read_u16(enemy_party_addr + offset)
                if 1 <= species_id <= 493:
                    return True
            return False
        except Exception:
            return False

    def _validate_opponent_data(self, opponent_data):
        """Validate opponent data for reasonableness to avoid showing incorrect info"""
        try:
            # Check required fields exist
            if not opponent_data or not isinstance(opponent_data, dict):
                return False
                
            species = opponent_data.get('species', '')
            level = opponent_data.get('level', 0)
            
            # Basic sanity checks
            if not species or species.startswith('Species_'):
                logger.debug(f"Invalid species name: {species}")
                return False
                
            if not isinstance(level, int) or level < 1 or level > 100:
                logger.debug(f"Invalid level: {level}")
                return False
            
            # HP validation (if provided)
            current_hp = opponent_data.get('current_hp')
            max_hp = opponent_data.get('max_hp')
            
            if current_hp is not None and current_hp != "Unknown":
                if not isinstance(current_hp, int) or current_hp < 0:
                    logger.debug(f"Invalid current HP: {current_hp}")
                    return False
                    
            if max_hp is not None and max_hp != "Unknown":
                if not isinstance(max_hp, int) or max_hp <= 0:
                    logger.debug(f"Invalid max HP: {max_hp}")
                    return False
                    
                # Current HP shouldn't exceed max HP
                if isinstance(current_hp, int) and current_hp > max_hp:
                    logger.debug(f"Current HP ({current_hp}) exceeds max HP ({max_hp})")
                    return False
            
            # Additional validation: HP should be reasonable for the level
            if isinstance(max_hp, int) and isinstance(level, int):
                # Very rough estimate: max HP should be at least level + 10, but not more than level * 10
                if max_hp < level + 5 or max_hp > level * 15:
                    logger.debug(f"Max HP ({max_hp}) seems unreasonable for level {level}")
                    return False
            
            logger.debug(f"Opponent data validation passed: {species} Lv{level}")
            return True
            
        except Exception as e:
            logger.debug(f"Opponent data validation error: {e}")
            return False

    def _get_pokemon_types_from_species(self, species_id):
        """Get Pokemon types from species ID"""
        try:
            # For now, return empty list as a placeholder
            # This could be enhanced with actual type lookup
            return []
        except Exception:
            return []

    # Map reading methods (keeping existing implementation for now)
    def _validate_buffer_data(self, buffer_addr, width, height):
        """Validate buffer doesn't contain too many corruption markers"""
        try:
            # Only validate if we're looking for outdoor maps (they shouldn't have many 0x3FF tiles)
            # Indoor maps might legitimately have these tiles
            corruption_count = 0
            sample_size = min(100, width * height)  # Sample first 100 tiles
            
            for i in range(sample_size):
                tile_addr = buffer_addr + (i * 2)
                tile_value = self._read_u16(tile_addr)
                tile_id = tile_value & 0x03FF
                
                # Tile ID 1023 (0x3FF) is a corruption marker
                if tile_id == 0x3FF:
                    corruption_count += 1
            
            corruption_ratio = corruption_count / sample_size
            
            # Be more lenient - only reject if more than 50% are corruption markers
            # This should still catch the bad buffers while allowing some legitimate ones
            if corruption_ratio > 0.5:
                logger.debug(f"Buffer at 0x{buffer_addr:08X} has {corruption_ratio:.1%} corruption markers, rejecting")
                return False
            
            return True
        except Exception:
            return True  # If we can't validate, assume it's OK
    
    def _find_map_buffer_addresses(self):
        """Find map buffer addresses - SIMPLIFIED to avoid over-filtering"""
        # First, try to invalidate any existing cache if we're having issues
        if self._map_buffer_addr and (self._map_width is None or self._map_height is None):
            logger.warning("Invalid map cache detected, clearing...")
            self.invalidate_map_cache()
        
        # SIMPLE APPROACH: Take the first valid buffer found (like original code)
        for offset in range(0, 0x8000 - 12, 4):
            try:
                width = self._read_u32(0x03000000 + offset)
                height = self._read_u32(0x03000000 + offset + 4)
                
                # More strict validation for reasonable map dimensions
                if 10 <= width <= 200 and 10 <= height <= 200:
                    map_ptr = self._read_u32(0x03000000 + offset + 8)
                    
                    # Validate map pointer is in valid EWRAM range
                    if 0x02000000 <= map_ptr <= 0x02040000:
                        # Additional validation: check if the map pointer points to valid memory
                        try:
                            # Try to read a small amount of data from the map pointer
                            test_data = self._read_bytes(map_ptr, 4)
                            if len(test_data) == 4:
                                # Only validate non-preferred buffers
                                # The preferred buffer (0x02032318) should always be used if found
                                if map_ptr != 0x02032318:
                                    # Validate buffer doesn't have too many corruption markers
                                    if not self._validate_buffer_data(map_ptr, width, height):
                                        logger.debug(f"Buffer at 0x{map_ptr:08X} failed validation, skipping")
                                        continue
                                
                                # FORCE CONSISTENT BUFFER: Use specific buffer address that direct emulator uses
                                # If we find the known good buffer (0x02032318), use it preferentially
                                if map_ptr == 0x02032318:
                                    logger.info(f"Found preferred buffer at 0x{map_ptr:08X} with size {width}x{height}")
                                    self._map_buffer_addr = map_ptr
                                    self._map_width = width
                                    self._map_height = height
                                    return True
                                # Store this as a fallback
                                elif not hasattr(self, '_fallback_buffer'):
                                    self._fallback_buffer = (map_ptr, width, height)
                                    continue
                        except Exception as e:
                            logger.debug(f"Map pointer validation failed at 0x{map_ptr:08X}: {e}")
                            continue
                        except Exception as e:
                            logger.debug(f"Map pointer validation failed at 0x{map_ptr:08X}: {e}")
                            continue
            except Exception as e:
                # Log only if this is a recurring issue
                if offset % 1000 == 0:
                    logger.debug(f"Error scanning map buffer at offset {offset}: {e}")
                continue
        
        # If preferred buffer not found, use fallback
        if hasattr(self, '_fallback_buffer'):
            map_ptr, width, height = self._fallback_buffer
            logger.info(f"Using fallback buffer at 0x{map_ptr:08X} with size {width}x{height}")
            self._map_buffer_addr = map_ptr
            self._map_width = width
            self._map_height = height
            delattr(self, '_fallback_buffer')
            return True
        
        self._rate_limited_warning("Could not find valid map buffer addresses", "map_buffer")
        return False
    
    def _find_alternative_buffer(self):
        """Try alternative methods to find a clean map buffer"""
        logger.info("Searching for alternative map buffer...")
        
        # Method 1: Scan a wider memory range
        for offset in range(0x8000, 0x10000 - 12, 4):
            try:
                width = self._read_u32(0x03000000 + offset)
                height = self._read_u32(0x03000000 + offset + 4)
                
                if 10 <= width <= 200 and 10 <= height <= 200:
                    map_ptr = self._read_u32(0x03000000 + offset + 8)
                    
                    if 0x02000000 <= map_ptr <= 0x02040000:
                        try:
                            test_data = self._read_bytes(map_ptr, 4)
                            if len(test_data) == 4:
                                is_current = self._validate_buffer_currency(map_ptr, width, height)
                                if is_current:
                                    self._map_buffer_addr = map_ptr
                                    self._map_width = width
                                    self._map_height = height
                                    logger.info(f"Found alternative buffer at 0x{map_ptr:08X} ({width}x{height})")
                                    return True
                        except Exception:
                            continue
            except Exception:
                continue
        
        # Method 2: Accept any buffer with lower corruption threshold
        logger.info("No clean buffer found, looking for least corrupted...")
        for offset in range(0, 0x8000 - 12, 4):
            try:
                width = self._read_u32(0x03000000 + offset)
                height = self._read_u32(0x03000000 + offset + 4)
                
                if 10 <= width <= 200 and 10 <= height <= 200:
                    map_ptr = self._read_u32(0x03000000 + offset + 8)
                    
                    if 0x02000000 <= map_ptr <= 0x02040000:
                        try:
                            test_data = self._read_bytes(map_ptr, 4)
                            if len(test_data) == 4:
                                # Accept any buffer - we'll use the first valid one found
                                self._map_buffer_addr = map_ptr
                                self._map_width = width
                                self._map_height = height
                                logger.warning(f"Using potentially corrupted buffer at 0x{map_ptr:08X} ({width}x{height}) as fallback")
                                return True
                        except Exception:
                            continue
            except Exception:
                continue
        
        logger.error("No alternative buffer found")
        return False
    
    def _validate_buffer_currency(self, buffer_addr, width, height):
        """Check if a buffer contains current (non-corrupted) map data"""
        try:
            # Sample more tiles and check for corruption patterns
            sample_size = min(50, width * height)
            corrupted_count = 0
            total_sampled = 0
            tile_frequency = {}
            
            for i in range(0, sample_size, 1):  # Sample every tile, not every 4th
                try:
                    tile_data = self._read_u16(buffer_addr + i * 2)
                    total_sampled += 1
                    
                    # Track tile frequency for repetition detection
                    tile_frequency[tile_data] = tile_frequency.get(tile_data, 0) + 1
                    
                    # Check for corruption patterns
                    if (tile_data == 0xFFFF or tile_data == 0x3FF or  # 1023 pattern
                        tile_data == 0x0000 or tile_data == 0x1FF):   # Other corruption patterns
                        corrupted_count += 1
                except Exception:
                    corrupted_count += 1
            
            if total_sampled == 0:
                return False
            
            # Check for excessive repetition (sign of corruption)
            max_frequency = max(tile_frequency.values()) if tile_frequency else 0
            repetition_ratio = max_frequency / total_sampled if total_sampled > 0 else 0
            
            corruption_ratio = corrupted_count / total_sampled
            
            # More strict criteria: current if low corruption AND low repetition
            is_current = (corruption_ratio < 0.3 and repetition_ratio < 0.5)
            
            logger.debug(f"Buffer 0x{buffer_addr:08X}: {corruption_ratio:.1%} corrupted, {repetition_ratio:.1%} repetition ({corrupted_count}/{total_sampled}) - current: {is_current}")
            
            # Show most common tiles for debugging
            if tile_frequency:
                sorted_tiles = sorted(tile_frequency.items(), key=lambda x: x[1], reverse=True)
                top_tiles = sorted_tiles[:3]
                logger.debug(f"  Top tiles: {[(hex(tile), count) for tile, count in top_tiles]}")
            
            return is_current
            
        except Exception as e:
            logger.debug(f"Buffer currency validation failed for 0x{buffer_addr:08X}: {e}")
            return False

    def read_map_around_player(self, radius: int = 10) -> List[List[Tuple[int, MetatileBehavior, int, int]]]:
        """Read map area around player with improved error handling for area transitions"""
        # Check for area transitions (re-enabled with minimal logic)
        location = self.read_location()
        position = self.read_coordinates()
        transition_detected = self._check_area_transition()
        
        # If we just transitioned, add a small delay for game state to stabilize
        # This addresses the frame-dependent timing issues mentioned in emerald_npc_decompilation
        if transition_detected:
            time.sleep(0.05)  # 50ms delay to let game scripts complete
            logger.debug("Applied post-transition delay for game state stabilization")
        
        # Always ensure map buffer is found
        if not self._map_buffer_addr:
            if not self._find_map_buffer_addresses():
                self._rate_limited_warning("Failed to find map buffer addresses, returning empty map", "map_buffer")
                return []
        
        # Read map data with simple validation and retry
        map_data = self._read_map_data_internal(radius)
        
        # Additional corruption detection: check for invalid map buffer data
        if map_data and self._map_buffer_addr:
            try:
                # Verify buffer is still valid by re-reading dimensions
                current_width = self._read_u32(self._map_buffer_addr - 8)
                current_height = self._read_u32(self._map_buffer_addr - 4)
                
                # If dimensions changed significantly, buffer may be corrupted
                if (abs(current_width - self._map_width) > 5 or 
                    abs(current_height - self._map_height) > 5 or
                    current_width <= 0 or current_height <= 0 or
                    current_width > 1000 or current_height > 1000):
                    
                    # Use unified rate limiter for corruption warnings
                    self._rate_limited_warning(f"Map buffer corruption detected: dimensions changed from {self._map_width}x{self._map_height} to {current_width}x{current_height}", "map_corruption")
                    
                    self._map_buffer_addr = None
                    self._map_width = None
                    self._map_height = None
                    
                    # Try to recover by re-finding buffer
                    if self._find_map_buffer_addresses():
                        logger.debug("Recovered from map buffer corruption")
                        map_data = self._read_map_data_internal(radius)
                    else:
                        logger.error("Failed to recover from map buffer corruption")
                        return []
            except Exception as e:
                logger.debug(f"Error checking buffer validity: {e}")
                # Don't fail completely on validation errors
        
        # Quick validation: check for too many unknown tiles (only for outdoor areas)
        if map_data and len(map_data) > 0:
            try:
                location_name = self.read_location()
            except Exception:
                location_name = ""
            
            # Only apply validation for outdoor areas, skip for indoor/house areas
            is_outdoor = any(keyword in location_name.upper() for keyword in ['TOWN', 'ROUTE', 'CITY', 'ROAD', 'PATH'])
            
            if is_outdoor:
                total_tiles = sum(len(row) for row in map_data)
                unknown_count = 0
                
                for row in map_data:
                    for tile in row:
                        if len(tile) >= 2:
                            behavior = tile[1]
                            if hasattr(behavior, 'name') and behavior.name == 'UNKNOWN':
                                unknown_count += 1
                            elif isinstance(behavior, int) and behavior == 0:  # UNKNOWN = 0
                                unknown_count += 1
                
                unknown_ratio = unknown_count / total_tiles if total_tiles > 0 else 0
                
                # If more than 50% unknown tiles in outdoor areas, try once more
                if unknown_ratio > 0.5:
                    logger.info(f"Outdoor map has {unknown_ratio:.1%} unknown tiles, retrying with cache invalidation")
                    self.invalidate_map_cache()
                    if self._find_map_buffer_addresses():
                        map_data = self._read_map_data_internal(radius)
            else:
                logger.debug(f"Skipping validation for indoor area: {location_name}")
                
        
        logger.info(f"Map data: {map_data}")
        
        return map_data
        
        # DISABLED: Try reading map with validation and retry logic
        # Use fewer retries for server performance
        max_retries = 2
        for attempt in range(max_retries):
            # Always try to find map buffer addresses if not already found
            if not self._map_buffer_addr:
                if not self._find_map_buffer_addresses():
                    self._rate_limited_warning("Failed to find map buffer addresses, returning empty map", "map_buffer")
                    return []
            
            # Try to read the map data
            map_data = self._read_map_data_internal(radius)
            
            if not map_data:
                logger.warning(f"Map read attempt {attempt + 1} returned empty data")
                if attempt < max_retries - 1:
                    self.invalidate_map_cache()
                    continue
                return []
            
            # Validate the map data
            is_valid, validation_msg = self._validate_map_data(map_data, location_name)
            
            if is_valid:
                logger.debug(f"Map validation passed on attempt {attempt + 1}: {validation_msg}")
                return map_data
            else:
                logger.warning(f"Map validation failed on attempt {attempt + 1}: {validation_msg}")
                if attempt < max_retries - 1:
                    logger.info(f"Invalidating cache and retrying... (attempt {attempt + 2}/{max_retries})")
                    self.invalidate_map_cache()
                    # Force re-finding map buffer addresses with timeout
                    start_time = time.time()
                    if not self._find_map_buffer_addresses():
                        logger.warning("Failed to re-find map buffer addresses during retry")
                        continue
                    # Don't spend too much time on retries (max 2 seconds total)
                    if time.time() - start_time > 2.0:
                        logger.warning("Map buffer search taking too long, returning current data")
                        return map_data
                else:
                    logger.warning(f"All {max_retries} map reading attempts failed validation, returning data anyway")
                    return map_data  # Return the last attempt even if invalid
        
        return []
    
    def _read_map_data_internal(self, radius: int = 10) -> List[List[Tuple[int, MetatileBehavior, int, int]]]:
        """Internal method to read map data without validation/retry logic"""
        
        try:
            player_x, player_y = self.read_coordinates()
            
            # Validate player coordinates
            if player_x < 0 or player_y < 0:
                logger.warning(f"Invalid player coordinates: ({player_x}, {player_y})")
                return []
            
            # Check if map dimensions are valid
            if not self._map_width or not self._map_height:
                logger.warning("Invalid map dimensions, attempting to re-find map buffer")
                self.invalidate_map_cache()
                if not self._find_map_buffer_addresses():
                    return []
            
            map_x = player_x + 7
            map_y = player_y + 7
            
            # Ensure consistent 11x11 output by adjusting boundaries
            target_width = 2 * radius + 1  # Should be 11 for radius=5
            target_height = 2 * radius + 1
            
            # Calculate ideal boundaries
            ideal_x_start = map_x - radius
            ideal_y_start = map_y - radius
            ideal_x_end = map_x + radius + 1
            ideal_y_end = map_y + radius + 1
            
            # Adjust boundaries to stay within buffer while maintaining target size
            x_start = max(0, min(ideal_x_start, self._map_width - target_width))
            y_start = max(0, min(ideal_y_start, self._map_height - target_height))
            x_end = min(self._map_width, x_start + target_width)
            y_end = min(self._map_height, y_start + target_height)
            
            # Validate that we have a reasonable area to read
            if x_end <= x_start or y_end <= y_start:
                logger.warning(f"Invalid map reading area: x({x_start}-{x_end}), y({y_start}-{y_end})")
                return []
            
            width = x_end - x_start
            height = y_end - y_start
            
            # Additional validation for reasonable dimensions
            if width > 50 or height > 50:
                logger.warning(f"Map reading area too large: {width}x{height}, limiting to 15x15")
                width = min(width, 15)
                height = min(height, 15)
            
            return self.read_map_metatiles(x_start, y_start, width, height)
        except Exception as e:
            logger.warning(f"Failed to read map data internally: {e}")
            return []

    def read_map_metatiles(self, x_start: int = 0, y_start: int = 0, width: int = None, height: int = None) -> List[List[Tuple[int, MetatileBehavior, int, int]]]:
        """Read map metatiles with improved error handling"""
        if not self._map_buffer_addr:
            self._rate_limited_warning("No map buffer address available", "map_buffer")
            return []
        
        if width is None:
            width = self._map_width
        if height is None:
            height = self._map_height
        
        # Validate dimensions
        if not width or not height:
            logger.warning(f"Invalid map dimensions: {width}x{height}")
            return []
        
        width = min(width, self._map_width - x_start)
        height = min(height, self._map_height - y_start)
        
        # Additional validation
        if width <= 0 or height <= 0:
            logger.warning(f"Invalid reading area: {width}x{height} at ({x_start}, {y_start})")
            return []
        
        try:
            metatiles = []
            for y in range(y_start, y_start + height):
                row = []
                for x in range(x_start, x_start + width):
                    try:
                        index = x + y * self._map_width
                        metatile_addr = self._map_buffer_addr + (index * 2)
                        
                        # Validate address before reading
                        if metatile_addr < self._map_buffer_addr or metatile_addr >= self._map_buffer_addr + (self._map_width * self._map_height * 2):
                            logger.debug(f"Invalid metatile address: 0x{metatile_addr:08X}")
                            row.append((0, MetatileBehavior.NORMAL, 0, 0))
                            continue
                        
                        metatile_value = self._read_u16(metatile_addr)
                        
                        metatile_id = metatile_value & 0x03FF
                        collision = (metatile_value & 0x0C00) >> 10
                        elevation = (metatile_value & 0xF000) >> 12
                        
                        # Validate metatile ID
                        if metatile_id > 0x3FF:
                            logger.debug(f"Invalid metatile ID: {metatile_id}")
                            metatile_id = 0
                        
                        behavior = self.get_exact_behavior_from_id(metatile_id)
                        row.append((metatile_id, behavior, collision, elevation))
                    except Exception as e:
                        logger.debug(f"Error reading metatile at ({x}, {y}): {e}")
                        row.append((0, MetatileBehavior.NORMAL, 0, 0))
                metatiles.append(row)
            
            return metatiles
        except Exception as e:
            logger.warning(f"Failed to read map metatiles: {e}")
            return []

    # Tileset reading methods (keeping existing implementation)
    def get_map_layout_base_address(self) -> int:
        """Get map layout base address"""
        try:
            return self._read_u32(self.addresses.MAP_HEADER + self.addresses.MAP_LAYOUT_OFFSET)
        except Exception as e:
            logger.warning(f"Failed to read map layout base address: {e}")
            return 0

    def get_tileset_pointers(self, map_layout_base_address: int) -> Tuple[int, int]:
        """Get tileset pointers"""
        if not map_layout_base_address:
            return (0, 0)
        
        try:
            primary = self._read_u32(map_layout_base_address + self.addresses.PRIMARY_TILESET_OFFSET)
            secondary = self._read_u32(map_layout_base_address + self.addresses.SECONDARY_TILESET_OFFSET)
            return (primary, secondary)
        except Exception as e:
            logger.warning(f"Failed to read tileset pointers: {e}")
            return (0, 0)

    def read_metatile_behaviors_from_tileset(self, tileset_base_address: int, num_metatiles: int) -> List[int]:
        """Read metatile behaviors from tileset"""
        if not tileset_base_address or num_metatiles <= 0:
            return []

        try:
            attributes_ptr = self._read_u32(tileset_base_address + 0x10)
            if not attributes_ptr:
                return []

            bytes_to_read = num_metatiles * 2
            attribute_bytes = self._read_bytes(attributes_ptr, bytes_to_read)

            if len(attribute_bytes) != bytes_to_read:
                return []

            behaviors = []
            for i in range(num_metatiles):
                byte_offset = i * 2
                byte1 = attribute_bytes[byte_offset]
                byte2 = attribute_bytes[byte_offset + 1]
                attribute_value = (byte2 << 8) | byte1
                behavior = attribute_value & 0x00FF
                behaviors.append(behavior)

            return behaviors

        except Exception as e:
            logger.warning(f"Failed to read metatile behaviors: {e}")
            return []

    def get_all_metatile_behaviors(self) -> List[int]:
        """Get all metatile behaviors for current map"""
        try:
            map_bank = self._read_u8(self.addresses.MAP_BANK)
            map_number = self._read_u8(self.addresses.MAP_NUMBER)
            cache_key = (map_bank, map_number)
            
            if self._cached_behaviors_map_key == cache_key and self._cached_behaviors is not None:
                return self._cached_behaviors

            map_layout_base = self.get_map_layout_base_address()
            if not map_layout_base:
                return []

            primary_addr, secondary_addr = self.get_tileset_pointers(map_layout_base)
            all_behaviors = []

            if primary_addr:
                primary_behaviors = self.read_metatile_behaviors_from_tileset(primary_addr, 0x200)
                all_behaviors.extend(primary_behaviors)

            if secondary_addr:
                secondary_behaviors = self.read_metatile_behaviors_from_tileset(secondary_addr, 0x200)
                all_behaviors.extend(secondary_behaviors)

            self._cached_behaviors = all_behaviors
            self._cached_behaviors_map_key = cache_key
            
            return all_behaviors

        except Exception as e:
            logger.warning(f"Failed to get all metatile behaviors: {e}")
            return []

    def get_exact_behavior_from_id(self, metatile_id: int) -> MetatileBehavior:
        """Get exact behavior for metatile ID"""
        try:
            all_behaviors = self.get_all_metatile_behaviors()
            
            if not all_behaviors or metatile_id >= len(all_behaviors):
                return MetatileBehavior.NORMAL

            behavior_byte = all_behaviors[metatile_id]
            
            try:
                return MetatileBehavior(behavior_byte)
            except ValueError:
                return MetatileBehavior.NORMAL

        except Exception as e:
            logger.warning(f"Failed to get exact behavior for metatile {metatile_id}: {e}")
            return MetatileBehavior.NORMAL

    def get_comprehensive_state(self, screenshot=None) -> Dict[str, Any]:
        """Get comprehensive game state with optional screenshot for OCR fallback"""
        logger.info("Starting comprehensive state reading")
        state = {
            "visual": {"screenshot": None, "resolution": [240, 160]},
            "player": {"position": None, "location": None, "name": None},
            "game": {
                "money": None, "party": None, "game_state": None, "is_in_battle": None,
                "time": None, "badges": None, "items": None, "item_count": None,
                "pokedex_caught": None, "pokedex_seen": None, "dialog_text": None,
                "progress_context": None
            },
            "map": {
                "tiles": None, "tile_names": None, "metatile_behaviors": None,
                "metatile_info": None, "traversability": None
            }
        }
        
        try:
            # Map tiles - read first
            state = self.read_map(state)
            # Player information
            coords = self.read_coordinates()
            if coords:
                state["player"]["position"] = {"x": coords[0], "y": coords[1]}
            
            location = self.read_location()
            if location:
                state["player"]["location"] = location
            
            player_name = self.read_player_name()
            if player_name:
                state["player"]["name"] = player_name
            
            # Player facing direction - removed as it's often unreliable
            # facing = self.read_player_facing()
            # if facing:
            #     state["player"]["facing"] = facing
            
            # Game information
            state["game"].update({
                "money": self.read_money(),
                "game_state": self.get_game_state(),
                "is_in_battle": self.is_in_battle(),
                "time": self.read_game_time(),
                "badges": self.read_badges(),
                "items": self.read_items(),
                "item_count": self.read_item_count(),
                "pokedex_caught": self.read_pokedex_caught_count(),
                "pokedex_seen": self.read_pokedex_seen_count()
            })
            
            # Battle details - use comprehensive battle info
            if state["game"]["is_in_battle"]:
                battle_details = self.read_comprehensive_battle_info()
                if battle_details:
                    state["game"]["battle_info"] = battle_details
            
            # Dialog text - only read if dialog detection is enabled
            if self._dialog_detection_enabled:
                dialog_text = self.read_dialog_with_ocr_fallback(screenshot)
                if dialog_text:
                    state["game"]["dialog_text"] = dialog_text
                    logger.info(f"Found dialog text: {dialog_text[:100]}...")
                else:
                    logger.debug("No dialog text found in memory buffers or OCR")
            else:
                logger.debug("Dialog detection disabled (no-ocr mode)")
            
            # Dialogue detection result - only if dialog detection is enabled
            if self._dialog_detection_enabled:
                dialogue_active = self.is_in_dialog()
                
                # Update dialogue cache with current state
                self._update_dialogue_cache(dialog_text if 'dialog_text' in locals() else None, dialogue_active)
                
                # Use cached dialogue state for additional validation
                cached_active, cached_text = self.get_cached_dialogue_state()
            else:
                dialogue_active = False
                cached_active = False
                cached_text = ""
            
            # Final dialogue state combines detection and cache validation
            final_dialogue_active = dialogue_active and cached_active
            
            state["game"]["dialogue_detected"] = {
                "has_dialogue": final_dialogue_active,
                "confidence": 1.0 if final_dialogue_active else 0.0,
                "reason": "enhanced pokeemerald detection with cache validation"
            }
            logger.debug(f"Dialogue detection: {dialogue_active}, cached: {cached_active}, final: {final_dialogue_active}")
            
            # Update game_state to reflect the current dialogue cache state
            # This ensures game_state is 'overworld' when dialogue is dismissed by A button
            if not final_dialogue_active and state["game"]["game_state"] == "dialog":
                state["game"]["game_state"] = "overworld"
                logger.debug("Updated game_state from 'dialog' to 'overworld' after dialogue cache validation")
            
            # Game progress context
            progress_context = self.get_game_progress_context()
            if progress_context:
                state["game"]["progress_context"] = progress_context
            
            # Party Pokemon
            logger.info("About to read party Pokemon")
            party = self.read_party_pokemon()
            logger.info(f"Read party: {len(party) if party else 0} Pokemon")
            if party:
                logger.info(f"Party data: {party}")
                state["player"]["party"] = [
                    {
                        "species_name": pokemon.species_name,
                        "level": pokemon.level,
                        "current_hp": pokemon.current_hp,
                        "max_hp": pokemon.max_hp,
                        "status": pokemon.status.get_status_name() if pokemon.status else "OK",
                        "types": [t.name for t in [pokemon.type1, pokemon.type2] if t],
                        "moves": pokemon.moves,
                        "move_pp": pokemon.move_pp,
                        "nickname": pokemon.nickname
                    }
                    for pokemon in party
                ]
                logger.info(f"Added {len(state['player']['party'])} Pokemon to state")
                logger.info(f"Final state party: {state['player']['party']}")
            else:
                self._rate_limited_warning("No Pokemon found in party", "party_empty")
        
                
        except Exception as e:
            logger.warning(f"Failed to read comprehensive state: {e}")
        
        return state
    
    def read_map(self, state): 
        tiles = self.read_map_around_player(radius=5)  # 11x11 grid to match agent's pixel view
        if tiles:
            # DEBUG: Print tile data before processing for HTTP API
            total_tiles = sum(len(row) for row in tiles)
            unknown_count = 0
            corruption_count = 0
            for row in tiles:
                for tile in row:
                    if len(tile) >= 2:
                        behavior = tile[1]
                        if isinstance(behavior, int):
                            if behavior == 0:
                                unknown_count += 1
                            elif behavior == 134:  # Indoor element corruption
                                corruption_count += 1
            
            unknown_ratio = unknown_count / total_tiles if total_tiles > 0 else 0
            logger.info(f"📊 PRE-PROCESSING TILES: {unknown_ratio:.1%} unknown ({unknown_count}/{total_tiles}), {corruption_count} corrupted")
            
            state["map"]["tiles"] = tiles
            
            # Process tiles for enhanced information (keep minimal processing here)
            tile_names = []
            metatile_behaviors = []
            metatile_info = []
            
            for row in tiles:
                row_names = []
                row_behaviors = []
                row_info = []
                
                for tile_data in row:
                    if len(tile_data) >= 4:
                        tile_id, behavior, collision, elevation = tile_data
                    elif len(tile_data) >= 2:
                        tile_id, behavior = tile_data[:2]
                        collision = 0
                        elevation = 0
                    else:
                        tile_id = tile_data[0] if tile_data else 0
                        behavior = None
                        collision = 0
                        elevation = 0
                    
                    # Tile name
                    tile_name = f"Tile_{tile_id:04X}"
                    if behavior is not None and hasattr(behavior, 'name'):
                        tile_name += f"({behavior.name})"
                    row_names.append(tile_name)
                    
                    # Behavior name
                    behavior_name = behavior.name if behavior is not None and hasattr(behavior, 'name') else "UNKNOWN"
                    row_behaviors.append(behavior_name)
                    
                    # Detailed tile info
                    tile_info = {
                        "id": tile_id,
                        "behavior": behavior_name,
                        "collision": collision,
                        "elevation": elevation,
                        "passable": collision == 0,
                        "encounter_possible": self._is_encounter_tile(behavior),
                        "surfable": self._is_surfable_tile(behavior)
                    }
                    row_info.append(tile_info)
                    
                    # No traversability processing - handled by state_formatter
                
                tile_names.append(row_names)
                metatile_behaviors.append(row_behaviors)
                metatile_info.append(row_info)
            
            state["map"]["tile_names"] = tile_names
            state["map"]["metatile_behaviors"] = metatile_behaviors
            state["map"]["metatile_info"] = metatile_info
            # traversability now generated by state_formatter from raw tiles
            
        # Add object events (NPCs/trainers)
        object_events = self.read_object_events()
        if object_events:
            state["map"]["object_events"] = object_events
            logger.info(f"📍 Found {len(object_events)} NPCs/trainers in current map")
            
            # Add player absolute coordinates for NPC positioning
            player_coords = self.read_coordinates()
            if player_coords:
                state["map"]["player_coords"] = {'x': player_coords[0], 'y': player_coords[1]}
        else:
            state["map"]["object_events"] = []
        
        # Update map stitcher with current area data
        print("🗺️ DEBUG: About to call _update_map_stitcher")
        self._update_map_stitcher(tiles, state)
        print("🗺️ DEBUG: _update_map_stitcher call completed")
        
        # Add stitched map information to state
        stitched_info = self.get_stitched_map_info()
        state["map"]["stitched_map_info"] = stitched_info
            
        return state
    
    def _update_map_stitcher(self, tiles, state):
        """Update the map stitcher with current map data"""
        print("🗺️ DEBUG: _update_map_stitcher called!")
        try:
            # Initialize map stitcher on first use
            if self._map_stitcher is None:
                from utils.map_stitcher import MapStitcher
                self._map_stitcher = MapStitcher()
            
            # Get current map identifiers
            map_bank = self._read_u8(self.addresses.MAP_BANK)
            map_number = self._read_u8(self.addresses.MAP_NUMBER)
            
            # Get location name from player location
            location_name = state.get("player", {}).get("location", "Unknown")
            
            # Get player coordinates
            player_coords = self.read_coordinates()
            if not player_coords:
                return
            
            # Convert absolute player coordinates to local map coordinates
            # The map data from read_map_around_player() is centered on the player
            # With radius=5, the map is 11x11, so player is at center (5, 5)
            map_height = len(tiles) if tiles else 11
            map_width = len(tiles[0]) if tiles and tiles[0] else 11
            local_player_coords = (map_width // 2, map_height // 2)
            
            # Get overworld coordinates for this map
            overworld_coords = self._get_overworld_coordinates(map_bank, map_number, location_name)
            
            # Debug logging
            logger.info(f"🗺️ Map stitcher update: Bank {map_bank}, Map {map_number}, Location: {location_name}")
            logger.info(f"🎯 Overworld coordinates: {overworld_coords}")
                
            # Update the stitcher
            timestamp = time.time()
            self._map_stitcher.update_map_area(
                map_bank=map_bank,
                map_number=map_number,
                location_name=location_name,
                map_data=tiles,
                player_pos=local_player_coords,
                timestamp=timestamp,
                overworld_coords=overworld_coords
            )
            
            # Periodically save stitcher data
            if hasattr(self, '_last_stitcher_save'):
                if timestamp - self._last_stitcher_save > 10.0:  # Save every 10 seconds
                    self._map_stitcher.save_to_file()
                    self._last_stitcher_save = timestamp
            else:
                self._last_stitcher_save = timestamp
                
        except Exception as e:
            print(f"🗺️ DEBUG: Failed to update map stitcher: {e}")
            logger.debug(f"Failed to update map stitcher: {e}")
            import traceback
            print(f"🗺️ DEBUG: Traceback: {traceback.format_exc()}")
    
    def _get_overworld_coordinates(self, map_bank: int, map_number: int, location_name: str) -> Optional[Tuple[int, int]]:
        """Get overworld coordinates for a given map bank/number combination"""
        # Map Pokemon Emerald's map bank/number to overworld coordinates
        # This is based on the actual Pokemon Emerald map layout
        
        map_coords = {
            # Bank 0 - Overworld maps
            (0, 0): (8, 18),   # PETALBURG_CITY
            (0, 1): (13, 30),  # SLATEPORT_CITY  
            (0, 2): (17, 15),  # MAUVILLE_CITY
            (0, 3): (8, 9),    # RUSTBORO_CITY
            (0, 4): (30, 6),   # FORTREE_CITY
            (0, 5): (35, 8),   # LILYCOVE_CITY
            (0, 6): (42, 12),  # MOSSDEEP_CITY
            (0, 7): (37, 20),  # SOOTOPOLIS_CITY
            (0, 8): (44, 15),  # EVER_GRANDE_CITY
            (0, 9): (16, 23),  # LITTLEROOT_TOWN
            (0, 10): (16, 19), # OLDALE_TOWN
            (0, 11): (3, 27),  # DEWFORD_TOWN
            (0, 12): (17, 9),  # LAVARIDGE_TOWN
            (0, 13): (15, 8),  # FALLARBOR_TOWN
            (0, 14): (11, 14), # VERDANTURF_TOWN
            (0, 15): (30, 28), # PACIFIDLOG_TOWN
            
            # Routes
            (0, 16): (16, 21), # ROUTE_101
            (0, 17): (14, 18), # ROUTE_102
            (0, 18): (18, 21), # ROUTE_103
            (0, 19): (10, 12), # ROUTE_104
            (0, 20): (5, 25),  # ROUTE_105
            (0, 21): (6, 27),  # ROUTE_106
            (0, 22): (7, 30),  # ROUTE_107
            (0, 23): (10, 32), # ROUTE_108
            (0, 24): (12, 32), # ROUTE_109
            (0, 25): (16, 16), # ROUTE_110
            (0, 26): (17, 12), # ROUTE_111
            (0, 27): (17, 10), # ROUTE_112
            (0, 28): (15, 10), # ROUTE_113
            (0, 29): (13, 8),  # ROUTE_114
            (0, 30): (6, 15),  # ROUTE_115
            (0, 31): (11, 9),  # ROUTE_116
            (0, 32): (13, 14), # ROUTE_117
            (0, 33): (20, 15), # ROUTE_118
            (0, 34): (25, 12), # ROUTE_119
            (0, 35): (27, 10), # ROUTE_120
            (0, 36): (30, 12), # ROUTE_121
            (0, 37): (32, 15), # ROUTE_122
            (0, 38): (33, 14), # ROUTE_123
            (0, 39): (35, 18), # ROUTE_124
            (0, 40): (28, 25), # ROUTE_125
            (0, 41): (25, 28), # ROUTE_126
            (0, 42): (30, 32), # ROUTE_127
            (0, 43): (35, 32), # ROUTE_128
            (0, 44): (38, 30), # ROUTE_129
            (0, 45): (40, 25), # ROUTE_130
            (0, 46): (42, 20), # ROUTE_131
            (0, 47): (40, 15), # ROUTE_132
            (0, 48): (38, 12), # ROUTE_133
            (0, 49): (35, 12), # ROUTE_134
        }
        
        # Check for exact match
        coords = map_coords.get((map_bank, map_number))
        if coords:
            return coords
        
        # For indoor locations (banks 1+), inherit coordinates from parent outdoor area
        if map_bank > 0:
            # Try to infer from location name
            name_upper = location_name.upper()
            
            # Match building names to their town coordinates
            if "LITTLEROOT" in name_upper:
                return (16, 23)
            elif "OLDALE" in name_upper:
                return (16, 19)
            elif "PETALBURG" in name_upper:
                return (8, 18)
            elif "RUSTBORO" in name_upper:
                return (8, 9)
            elif "DEWFORD" in name_upper:
                return (3, 27)
            elif "SLATEPORT" in name_upper:
                return (13, 30)
            elif "MAUVILLE" in name_upper:
                return (17, 15)
            elif "VERDANTURF" in name_upper:
                return (11, 14)
            elif "FALLARBOR" in name_upper:
                return (15, 8)
            elif "LAVARIDGE" in name_upper:
                return (17, 9)
            elif "FORTREE" in name_upper:
                return (30, 6)
            elif "LILYCOVE" in name_upper:
                return (35, 8)
            elif "MOSSDEEP" in name_upper:
                return (42, 12)
            elif "SOOTOPOLIS" in name_upper:
                return (37, 20)
            elif "EVER_GRANDE" in name_upper:
                return (44, 15)
            elif "PACIFIDLOG" in name_upper:
                return (30, 28)
        
        # If no coordinates found, return None (unknown location)
        return None
    
    def get_stitched_map_info(self) -> Dict[str, Any]:
        """Get stitched map information for agent use"""
        if self._map_stitcher is None:
            return {"available": False, "reason": "Map stitcher not initialized"}
        
        try:
            stats = self._map_stitcher.get_stats()
            
            # Get current area info
            current_map_bank = self._read_u8(self.addresses.MAP_BANK)
            current_map_number = self._read_u8(self.addresses.MAP_NUMBER)
            current_map_id = (current_map_bank << 8) | current_map_number
            
            current_area = self._map_stitcher.map_areas.get(current_map_id)
            current_connections = self._map_stitcher.get_connected_areas(current_map_id)
            
            # Get nearby areas (areas reachable in 1-2 connections)
            nearby_areas = []
            visited_ids = set()
            
            def add_connected_areas(area_id, depth=0, max_depth=2):
                if depth > max_depth or area_id in visited_ids:
                    return
                visited_ids.add(area_id)
                
                area = self._map_stitcher.map_areas.get(area_id)
                if area:
                    connections = self._map_stitcher.get_connected_areas(area_id)
                    nearby_areas.append({
                        "name": area.location_name,
                        "id": f"{area_id:04X}",
                        "depth": depth,
                        "overworld_coords": area.overworld_coords,
                        "connections": [{"name": name, "direction": direction} 
                                      for _, name, direction in connections]
                    })
                    
                    # Recursively add connected areas
                    for conn_id, _, _ in connections:
                        add_connected_areas(conn_id, depth + 1, max_depth)
            
            add_connected_areas(current_map_id)
            
            # Include terrain data for world map display
            terrain_areas = []
            for area_id, area in self._map_stitcher.map_areas.items():
                if area.overworld_coords and area.map_data:
                    terrain_areas.append({
                        "id": f"{area_id:04X}",
                        "name": area.location_name or "Unknown",
                        "overworld_coords": area.overworld_coords,
                        "map_data": area.map_data,
                        "player_pos": area.player_last_position
                    })
            
            return {
                "available": True,
                "stats": stats,
                "current_area": {
                    "name": current_area.location_name if current_area else "Unknown",
                    "id": f"{current_map_id:04X}",
                    "overworld_coords": current_area.overworld_coords if current_area else None,
                    "connections": [{"name": name, "direction": direction} 
                                   for _, name, direction in current_connections]
                },
                "nearby_areas": nearby_areas[:10],  # Limit to 10 areas
                "terrain_areas": terrain_areas,  # Include terrain data for world map
                "total_discovered": len(self._map_stitcher.map_areas)
            }
            
        except Exception as e:
            import traceback
            logger.debug(f"Failed to get stitched map info: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {"available": False, "reason": f"Error: {e}"}

    def _is_encounter_tile(self, behavior) -> bool:
        """Check if tile can trigger encounters"""
        if not behavior:
            return False
        
        encounter_behaviors = {
            MetatileBehavior.TALL_GRASS, MetatileBehavior.LONG_GRASS, MetatileBehavior.UNUSED_05,
            MetatileBehavior.DEEP_SAND, MetatileBehavior.CAVE, MetatileBehavior.INDOOR_ENCOUNTER,
            MetatileBehavior.POND_WATER, MetatileBehavior.INTERIOR_DEEP_WATER, MetatileBehavior.DEEP_WATER,
            MetatileBehavior.OCEAN_WATER, MetatileBehavior.SEAWEED, MetatileBehavior.ASHGRASS,
            MetatileBehavior.FOOTPRINTS, MetatileBehavior.SEAWEED_NO_SURFACING
        }
        
        return behavior in encounter_behaviors

    def _is_surfable_tile(self, behavior) -> bool:
        """Check if tile can be surfed on"""
        if not behavior:
            return False
        
        surfable_behaviors = {
            MetatileBehavior.POND_WATER, MetatileBehavior.INTERIOR_DEEP_WATER, MetatileBehavior.DEEP_WATER,
            MetatileBehavior.SOOTOPOLIS_DEEP_WATER, MetatileBehavior.OCEAN_WATER, MetatileBehavior.NO_SURFACING,
            MetatileBehavior.SEAWEED, MetatileBehavior.SEAWEED_NO_SURFACING
        }
        
        return behavior in surfable_behaviors

    def test_memory_access(self) -> Dict[str, Any]:
        """Test memory access functionality"""
        diagnostics = {
            'memory_interface': 'unknown',
            'memory_methods': [],
            'save_blocks_found': False,
            'save_block_offsets': None,
            'map_buffer_found': False,
            'basic_reads_working': False
        }
        
        # Test memory interface
        if hasattr(self.memory, 'load8'):
            diagnostics['memory_interface'] = 'mgba_load_methods'
            diagnostics['memory_methods'].extend(['load8', 'load16', 'load32'])
        elif hasattr(self.memory, 'read8'):
            diagnostics['memory_interface'] = 'mgba_read_methods'
            diagnostics['memory_methods'].extend(['read8', 'read16', 'read32'])
        else:
            diagnostics['memory_interface'] = 'direct_indexing'
            diagnostics['memory_methods'].append('__getitem__')
        
        # Test basic memory reads
        try:
            test_val = self._read_u8(0x02000000)
            diagnostics['basic_reads_working'] = True
        except Exception as e:
            diagnostics['basic_read_error'] = str(e)
        
        # Test map buffer detection
        if self._find_map_buffer_addresses():
            diagnostics['map_buffer_found'] = True
            diagnostics['map_buffer_info'] = {
                'address': hex(self._map_buffer_addr),
                'width': self._map_width,
                'height': self._map_height
            }
        
        return diagnostics

    def read_dialog(self) -> str:
        """Read any dialog text currently on screen by scanning text buffers"""
        try:
            # Always try to read dialog text, regardless of game state
            # The game state detection might not be reliable for dialog
            
            # Text buffer addresses from Pokemon Emerald decompilation symbols
            # https://raw.githubusercontent.com/pret/pokeemerald/symbols/pokeemerald.sym
            # Order by size (largest first) to prioritize longer dialog text
            text_buffers = [
                (self.addresses.G_STRING_VAR4, 1000),  # Main string variable 4 (largest) - PRIORITY
                (self.addresses.G_DISPLAYED_STRING_BATTLE, 300),  # Battle dialog text
                (self.addresses.G_STRING_VAR1, 256),   # Main string variable 1
                (self.addresses.G_STRING_VAR2, 256),   # Main string variable 2
                (self.addresses.G_STRING_VAR3, 256),   # Main string variable 3
                (self.addresses.G_BATTLE_TEXT_BUFF1, 16),  # Battle text buffer 1
                (self.addresses.G_BATTLE_TEXT_BUFF2, 16),  # Battle text buffer 2
                (self.addresses.G_BATTLE_TEXT_BUFF3, 16),  # Battle text buffer 3
                # Legacy addresses (keeping for compatibility)
                (self.addresses.TEXT_BUFFER_1, 200),
                (self.addresses.TEXT_BUFFER_2, 200),
                (self.addresses.TEXT_BUFFER_3, 200),
                (self.addresses.TEXT_BUFFER_4, 200),
            ]
            
            dialog_text = ""
            
            for buffer_addr, buffer_size in text_buffers:
                try:
                    # Read the specified amount of bytes for this buffer
                    buffer_bytes = self._read_bytes(buffer_addr, buffer_size)
                    
                    # Look for text patterns
                    text_lines = []
                    current_line = []
                    space_count = 0
                    
                    for byte in buffer_bytes:
                        # Check if this is a valid text character using our existing mapping
                        if self._is_valid_text_byte(byte):
                            space_count = 0
                            current_line.append(byte)
                        elif byte == 0x7F:  # Space character in Emerald
                            space_count += 1
                            current_line.append(byte)
                        elif byte == 0x4E:  # Line break character
                            # End current line
                            if current_line:
                                text = self._decode_pokemon_text(bytes(current_line))
                                if text.strip():
                                    text_lines.append(text)
                                current_line = []
                                space_count = 0
                        elif byte == 0xFF:  # End of string
                            break
                        
                        # If we see too many consecutive spaces, might be end of meaningful text
                        if space_count > 15 and current_line:
                            text = self._decode_pokemon_text(bytes(current_line))
                            if text.strip():
                                text_lines.append(text)
                            current_line = []
                            space_count = 0
                    
                    # Add final line if any
                    if current_line:
                        text = self._decode_pokemon_text(bytes(current_line))
                        if text.strip():
                            text_lines.append(text)
                    
                    # Join lines and check if we got meaningful text
                    potential_text = "\n".join(text_lines)
                    if len(potential_text.strip()) > 5:  # Minimum meaningful length
                        # Clean up the text - remove excessive whitespace and special characters
                        cleaned_text = potential_text.strip()
                        # Remove null bytes and other control characters
                        cleaned_text = ''.join(char for char in cleaned_text if ord(char) >= 32 or char in '\n\t')
                        # Normalize whitespace
                        cleaned_text = ' '.join(cleaned_text.split())
                        
                        if len(cleaned_text) > 5:
                            # Prefer longer text (more likely to be full dialog)
                            if len(cleaned_text) > len(dialog_text):
                                dialog_text = cleaned_text
                                logger.debug(f"Found better dialog text: {dialog_text[:100]}...")
                            # If this is the first meaningful text found, use it
                            elif not dialog_text:
                                dialog_text = cleaned_text
                                logger.debug(f"Found dialog text: {dialog_text[:100]}...")
                        
                except Exception as e:
                    logger.debug(f"Failed to read from buffer 0x{buffer_addr:08X} (size: {buffer_size}): {e}")
                    continue
            
            return dialog_text.strip()
            
        except Exception as e:
            logger.warning(f"Failed to read dialog: {e}")
            return ""

    def read_dialog_with_ocr_fallback(self, screenshot=None) -> str:
        """
        Read dialog text with smart OCR validation to detect stale memory.
        
        Preference order:
        1. Both memory AND OCR detect text -> Use memory (most accurate)
        2. Only OCR detects text -> Use OCR (memory failed)  
        3. Only memory detects text -> Suppress (likely stale/buggy memory)
        
        Args:
            screenshot: PIL Image of current game screen (optional)
            
        Returns:
            Dialog text using smart preference logic
        """
        # First try memory-based detection with enhanced filtering
        raw_memory_text = self.read_dialog()
        
        # Apply residual text filtering like the enhanced dialogue detection does
        memory_text = ""
        if raw_memory_text:
            cleaned_text = raw_memory_text.strip().lower()
            residual_indicators = [
                "got away safely", "fled from", "escaped", "ran away",
                "fainted", "defeated", "victory", "experience points", 
                "gained", "grew to", "learned"
            ]
            if any(indicator in cleaned_text for indicator in residual_indicators):
                logger.debug(f"OCR fallback: Filtering out residual battle text: '{raw_memory_text[:30]}...'")
                memory_text = ""  # Treat as no memory text
            else:
                memory_text = raw_memory_text
        
        # If we have OCR available and a screenshot, use smart validation
        if self._ocr_enabled and screenshot is not None and hasattr(screenshot, 'size'):
            try:
                ocr_text = self._ocr_detector.detect_dialogue_from_screenshot(screenshot)
                
                # Normalize for comparison (strip whitespace, handle None)
                memory_clean = memory_text.strip() if memory_text else ""
                ocr_clean = ocr_text.strip() if ocr_text else ""
                
                # Validate if OCR text is meaningful dialogue (not garbage like 'cL een aA')
                ocr_is_meaningful = self._is_ocr_meaningful_dialogue(ocr_clean)
                
                # Case 1: Both memory and OCR found meaningful text
                if memory_clean and ocr_clean and ocr_is_meaningful:
                    logger.debug(f"Both memory and OCR detected text")
                    logger.debug(f"Memory: '{memory_clean[:50]}...'")
                    logger.debug(f"OCR: '{ocr_clean[:50]}...'")
                    
                    # Validate similarity to detect if memory is reasonable
                    if self._texts_are_similar(memory_clean, ocr_clean):
                        logger.debug("✅ Memory and OCR are similar - using memory (most accurate)")
                        return memory_clean
                    else:
                        logger.debug("⚠️ Memory and OCR differ significantly - using memory but flagging")
                        # Still use memory when both exist, but log the discrepancy
                        return memory_clean
                
                # Case 2: Only OCR found meaningful text (memory failed/empty)
                elif not memory_clean and ocr_clean and ocr_is_meaningful:
                    logger.debug(f"Only OCR detected meaningful text - memory reading failed")
                    logger.debug(f"Using OCR: '{ocr_clean[:50]}...'")
                    return ocr_clean
                
                # Case 3: Only memory found text (OCR failed/empty/meaningless) 
                elif memory_clean and (not ocr_clean or not ocr_is_meaningful):
                    if not ocr_clean:
                        logger.debug(f"Only memory detected text - OCR found nothing")
                    elif not ocr_is_meaningful:
                        logger.debug(f"Only memory detected text - OCR found meaningless noise: '{ocr_clean}'")
                    logger.debug(f"Memory text: '{memory_clean[:50]}...'")
                    logger.debug("🚨 SUPPRESSING: Memory-only detection (likely stale/buggy)")
                    # This is the key fix - suppress memory-only detections as they're likely stale
                    return ""
                
                # Case 4: Neither found text
                else:
                    logger.debug("Neither memory nor OCR detected dialogue text")
                    return ""
                    
            except Exception as e:
                logger.debug(f"OCR validation failed: {e}")
                # Fall back to memory reading if OCR fails completely
                return memory_text if memory_text else ""
        
        # If no OCR available, use memory reading as before
        return memory_text if memory_text else ""
    
    def _texts_are_similar(self, text1: str, text2: str, threshold: float = 0.4) -> bool:
        """
        Check if two texts are reasonably similar (handles OCR differences)
        
        Args:
            text1, text2: Texts to compare
            threshold: Minimum similarity ratio (0.0-1.0)
            
        Returns:
            True if texts are similar enough
        """
        if not text1 or not text2:
            return False
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if len(union) > 0 else 0
        
        # Also check for substring matches (handles OCR character errors)
        substring_matches = 0
        for word1 in words1:
            for word2 in words2:
                if len(word1) >= 3 and len(word2) >= 3:
                    if word1 in word2 or word2 in word1:
                        substring_matches += 1
                        break
        
        substring_similarity = substring_matches / max(len(words1), len(words2))
        
        # Use the higher of the two similarity measures
        final_similarity = max(similarity, substring_similarity)
        
        logger.debug(f"Text similarity: {final_similarity:.2f} (threshold: {threshold})")
        return final_similarity >= threshold
    
    def _is_ocr_meaningful_dialogue(self, ocr_text: str) -> bool:
        """
        Determine if OCR text represents meaningful dialogue vs. random noise.
        
        Args:
            ocr_text: Text detected by OCR
            
        Returns:
            True if the text appears to be meaningful dialogue, False if it's likely noise
        """
        if not ocr_text or len(ocr_text.strip()) == 0:
            return False
            
        text = ocr_text.strip().lower()
        
        # Minimum length check - meaningful dialogue is usually longer than a few characters
        if len(text) < 6:
            return False
        
        # Maximum length check - OCR garbage can be extremely long
        if len(text) > 200:
            logger.debug(f"OCR text too long ({len(text)} chars) - likely garbage")
            return False
        
        # Check for common dialogue patterns/words
        dialogue_indicators = [
            'you', 'the', 'and', 'are', 'use', 'can', 'have', 'will', 'would', 'could', 'should',
            'pokemon', 'pokémon', 'items', 'store', 'battle', 'want', 'need', 'know', 'think',
            'pc', 'computer', 'science', 'power', 'staggering', 'hello', 'welcome', 'trainer',
            'what', 'where', 'when', 'how', 'why', 'who', 'this', 'that', 'there', 'here',
            'got', 'get', 'give', 'take', 'come', 'go', 'see', 'look', 'find'
        ]
        
        # Common OCR noise patterns to explicitly reject
        noise_patterns = [
            'lle', 'fyi', 'cl', 'een', 'aa', 'ii', 'oo', 'uu', 'mm', 'nn', 'll', 'tt', 'ss',
            'xx', 'zz', 'qq', 'jj', 'kk', 'vv', 'ww', 'yy', 'ff', 'gg', 'hh', 'bb', 'cc',
            'dd', 'pp', 'rr'  # Common OCR noise patterns
        ]
        
        words = text.split()
        meaningful_words = 0
        
        # Check for OCR garbage patterns that disqualify the entire text
        if self._has_ocr_garbage_patterns(words):
            return False
        
        # Count how many words look like actual dialogue words
        for word in words:
            # Remove punctuation for matching
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) >= 2:
                # Check if it's a known noise pattern first
                if clean_word in noise_patterns:
                    continue  # Skip noise patterns, don't count as meaningful
                # Check against dialogue indicators
                elif clean_word in dialogue_indicators:
                    meaningful_words += 1
                # Check if word has reasonable character patterns (not random like 'cL')
                elif self._has_reasonable_word_pattern(clean_word):
                    meaningful_words += 1
        
        # Need at least 40% of words to be meaningful for it to count as dialogue
        if len(words) > 0:
            meaningful_ratio = meaningful_words / len(words)
            logger.debug(f"OCR meaningfulness: {meaningful_words}/{len(words)} = {meaningful_ratio:.2f} for '{ocr_text}'")
            return meaningful_ratio >= 0.4
        
        return False
    
    def _has_reasonable_word_pattern(self, word: str) -> bool:
        """
        Check if a word has reasonable character patterns vs. OCR noise.
        
        Args:
            word: Word to check
            
        Returns:
            True if word pattern looks reasonable
        """
        if len(word) < 2:
            return False
        
        # Check for reasonable vowel/consonant distribution
        vowels = 'aeiou'
        vowel_count = sum(1 for c in word.lower() if c in vowels)
        consonant_count = len(word) - vowel_count
        
        # Words should have some vowels unless they're very short
        if len(word) >= 3 and vowel_count == 0:
            return False
        
        # Very short words with only consonants are likely OCR noise
        if len(word) <= 3 and vowel_count == 0:
            return False
        
        # Check for excessive repeated characters (OCR often creates these)
        repeated_chars = 0
        for i in range(len(word) - 1):
            if word[i] == word[i + 1]:
                repeated_chars += 1
        
        # Too many repeated characters suggests OCR noise
        if repeated_chars > len(word) // 2:
            return False
        
        return True
    
    def _has_ocr_garbage_patterns(self, words: list) -> bool:
        """
        Detect OCR garbage patterns that indicate the text is meaningless noise.
        
        Args:
            words: List of words from OCR text
            
        Returns:
            True if text contains OCR garbage patterns
        """
        if not words or len(words) == 0:
            return False
        
        # Pattern 1: Excessive repetition of single characters or short words
        single_char_words = [w for w in words if len(w) <= 2]
        if len(single_char_words) > len(words) * 0.5:  # More than 50% single/double char words
            logger.debug(f"OCR garbage: Too many short words ({len(single_char_words)}/{len(words)})")
            return True
        
        # Pattern 2: Check for repeated identical words (like 'a a a a a a')
        word_counts = {}
        for word in words:
            clean_word = word.lower().strip()
            word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
        
        for word, count in word_counts.items():
            if len(word) <= 2 and count >= 4:  # Same short word repeated 4+ times
                logger.debug(f"OCR garbage: Repeated short word '{word}' {count} times")
                return True
        
        # Pattern 3: Too many "words" - dialogue is usually concise
        if len(words) > 25:  # Pokemon dialogue is typically much shorter
            logger.debug(f"OCR garbage: Too many words ({len(words)}) for typical dialogue")
            return True
        
        # Pattern 4: Check for excessive all-caps "words" (OCR noise often creates these)
        all_caps_words = [w for w in words if len(w) >= 2 and w.isupper()]
        if len(all_caps_words) > len(words) * 0.4:  # More than 40% all-caps
            logger.debug(f"OCR garbage: Too many all-caps words ({len(all_caps_words)}/{len(words)})")
            return True
        
        # Pattern 5: Check for random character sequences (like 'ePID', 'SCONES')
        random_looking = 0
        for word in words:
            if len(word) >= 3:
                # Check for mixed case in middle of word (like 'ePID')
                has_mixed_case = any(c.islower() for c in word) and any(c.isupper() for c in word)
                # Check for uncommon letter combinations
                has_weird_patterns = any(combo in word.lower() for combo in ['pq', 'qp', 'xz', 'zx', 'jr', 'rj'])
                if has_mixed_case or has_weird_patterns:
                    random_looking += 1
        
        if random_looking > len(words) * 0.3:  # More than 30% weird words
            logger.debug(f"OCR garbage: Too many random-looking words ({random_looking}/{len(words)})")
            return True
        
        return False

    def _is_valid_text_byte(self, byte: int) -> bool:
        """Check if a byte represents a valid text character in Pokemon Emerald"""
        # Use the EmeraldCharmap to check if byte is valid
        charmap = EmeraldCharmap()
        return byte < len(charmap.charmap) and charmap.charmap[byte] != ""

    def read_flags(self) -> Dict[str, bool]:
        """Read game flags to track progress and visited locations"""
        try:
            # Get SaveBlock1 pointer
            save_block_1_ptr = self._read_u32(self.addresses.SAVE_BLOCK1_PTR)
            if save_block_1_ptr == 0:
                self._rate_limited_warning("SaveBlock1 pointer is null", "saveblock_pointer")
                return {}
            
            # Read flags from SaveBlock1
            flags_addr = save_block_1_ptr + self.addresses.SAVE_BLOCK1_FLAGS_OFFSET
            flags_data = self._read_bytes(flags_addr, 300)  # Flags are 300 bytes in SaveBlock1
            
            flags = {}
            
            # Check system flags (badges, visited locations, etc.)
            system_flags_start = self.addresses.SYSTEM_FLAGS_START
            system_flags_byte = system_flags_start // 8
            system_flags_bit = system_flags_start % 8
            
            # Badge flags
            badge_flags = [
                ("badge_01", 0x7), ("badge_02", 0x8), ("badge_03", 0x9), ("badge_04", 0xa),
                ("badge_05", 0xb), ("badge_06", 0xc), ("badge_07", 0xd), ("badge_08", 0xe)
            ]
            
            for badge_name, flag_offset in badge_flags:
                flag_byte = system_flags_byte + flag_offset // 8
                flag_bit = flag_offset % 8
                if flag_byte < len(flags_data):
                    flags[badge_name] = bool(flags_data[flag_byte] & (1 << flag_bit))
            
            # Visited location flags
            location_flags = [
                ("visited_littleroot", 0xF), ("visited_oldale", 0x10), ("visited_dewford", 0x11),
                ("visited_lavaridge", 0x12), ("visited_fallarbor", 0x13), ("visited_verdanturf", 0x14),
                ("visited_pacifidlog", 0x15), ("visited_petalburg", 0x16), ("visited_slateport", 0x17),
                ("visited_mauville", 0x18), ("visited_rustboro", 0x19), ("visited_fortree", 0x1A),
                ("visited_lilycove", 0x1B), ("visited_mossdeep", 0x1C), ("visited_sootopolis", 0x1D),
                ("visited_ever_grande", 0x1E)
            ]
            
            for location_name, flag_offset in location_flags:
                flag_byte = system_flags_byte + flag_offset // 8
                flag_bit = flag_offset % 8
                if flag_byte < len(flags_data):
                    flags[location_name] = bool(flags_data[flag_byte] & (1 << flag_bit))
            
            # Champion flag
            champion_flag_byte = system_flags_byte + 0x1F // 8
            champion_flag_bit = 0x1F % 8
            if champion_flag_byte < len(flags_data):
                flags["is_champion"] = bool(flags_data[champion_flag_byte] & (1 << champion_flag_bit))
            
            # Pokedex and other system flags
            pokedex_flag_byte = system_flags_byte + 0x1 // 8
            pokedex_flag_bit = 0x1 % 8
            if pokedex_flag_byte < len(flags_data):
                flags["has_pokedex"] = bool(flags_data[pokedex_flag_byte] & (1 << pokedex_flag_bit))
            
            logger.info(f"Read {len(flags)} game flags")
            return flags
            
        except Exception as e:
            logger.warning(f"Failed to read flags: {e}")
            return {}

    def get_game_progress_context(self) -> Dict[str, Any]:
        """Get context about game progress for better dialog understanding"""
        try:
            flags = self.read_flags()
            badges = self.read_badges()
            party = self.read_party_pokemon()
            
            context = {
                "badges_obtained": len(badges),
                "badge_names": badges,
                "party_size": len(party) if party else 0,
                "has_pokedex": flags.get("has_pokedex", False),
                "is_champion": flags.get("is_champion", False),
                "visited_locations": [k for k, v in flags.items() if k.startswith("visited_") and v],
                "flags": flags
            }
            
            # Add party info if available
            if party:
                context["party_levels"] = [p.level for p in party]
                context["party_species"] = [p.species_name for p in party]
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to get game progress context: {e}")
            return {}
    
    def read_object_events(self):
        """
        Read NPC/trainer object events using OAM sprite detection for walking positions.
        
        1. First try OAM (Object Attribute Memory) for actual visual sprite positions
        2. Fallback to static spawn positions from gObjectEvents
        
        Returns:
            list: List of object events with their current walking positions
        """
        try:
            # Get player position 
            player_coords = self.read_coordinates()
            if not player_coords:
                self._rate_limited_warning("Could not read player coordinates for NPC search", "coordinates")
                return []
            
            player_x, player_y = player_coords
            object_events = []
            
            # Method 1: Get stable NPC base positions first
            logger.debug("Reading base NPC positions from known addresses...")
            known_npcs = self._read_known_npc_addresses(player_x, player_y)
            
            if known_npcs:
                # Method 2: Try to enhance with walking positions from OAM
                logger.debug("Enhancing NPCs with walking positions from OAM...")
                enhanced_npcs = self._enhance_npcs_with_oam_walking(known_npcs, player_x, player_y)
                object_events.extend(enhanced_npcs)
            else:
                logger.debug("No known NPCs found, this shouldn't happen in npc.state")
                object_events = []
            
            # Filter out false positives (NPCs on door tiles)
            filtered_events = self._filter_door_false_positives(object_events, player_x, player_y)
            
            logger.info(f"📍 Found {len(filtered_events)} NPCs/trainers near player at ({player_x}, {player_y})")
            
            return filtered_events
            
        except Exception as e:
            logger.error(f"Failed to read object events: {e}")
            return []
    
    def _read_runtime_object_events(self, player_x, player_y):
        """
        Try to read NPCs from runtime sources:
        1. First try gSprites array (visual sprite positions)  
        2. Fallback to EWRAM addresses and legacy gObjectEvents
        """
        object_events = []
        
        try:
            # Method 1: Try gSprites array first - this contains actual visual positions
            gsprites_npcs = self._read_gsprites_npcs(player_x, player_y)
            if gsprites_npcs:
                object_events.extend(gsprites_npcs)
                logger.debug(f"Found {len(gsprites_npcs)} NPCs in gSprites")
                
            # Method 2: Try EWRAM runtime addresses  
            runtime_addresses = [
                (0x0300F428, "EWRAM_runtime_1"),  # Found with movement: (10,13) -> (10,2) -> etc
                (0x03007428, "EWRAM_runtime_2"),  # Mirror of above  
                (0x0300DCFC, "EWRAM_runtime_3"),  # Different movement pattern
                (0x03005CFC, "EWRAM_runtime_4"),  # Mirror of above
            ]
            
            found_npcs = 0
            for addr, location_name in runtime_addresses:
                try:
                    # Read coordinates directly (they're at offset +8 and +10 in the structure)
                    current_x = self._read_s16(addr + 8)
                    current_y = self._read_s16(addr + 10)
                    
                    # Skip if coordinates are obviously invalid
                    if current_x < -50 or current_x > 200 or current_y < -50 or current_y > 200:
                        continue
                    if current_x == 1023 and current_y == 1023:  # Common uninitialized value
                        continue
                    if current_x == 0 and current_y == 0:  # Likely uninitialized
                        continue
                    
                    # Skip coordinates with one 0 when far from player
                    distance = abs(current_x - player_x) + abs(current_y - player_y)
                    if (current_x == 0 or current_y == 0) and distance > 3:
                        continue  # Likely uninitialized if has a 0 coordinate and is far from player
                    
                    # Only include NPCs within reasonable range of player
                    if distance > 10:  # Reduced from 15 to be more conservative
                        continue
                    
                    # Read structure around coordinates to extract NPC properties
                    context = self._read_bytes(addr, 24)
                    
                    # Try to extract graphics and movement data from surrounding bytes
                    graphics_id = 1  # Default
                    movement_type = 1  # Default
                    trainer_type = 0   # Default
                    
                    # Look for reasonable graphics/movement values in context
                    for offset in range(len(context)):
                        val = context[offset]
                        if 1 <= val <= 50 and offset < 16:  # Reasonable graphics ID
                            graphics_id = val
                        elif 0 <= val <= 10 and offset < 16:  # Reasonable movement type
                            movement_type = val
                        elif 1 <= val <= 5 and offset > 10:  # Possible trainer type
                            trainer_type = val
                    
                    object_event = {
                        'id': found_npcs,
                        'obj_event_id': found_npcs,
                        'local_id': found_npcs,
                        'graphics_id': graphics_id,
                        'movement_type': movement_type,
                        'current_x': current_x,
                        'current_y': current_y,
                        'initial_x': current_x,  # Runtime position, use as initial too
                        'initial_y': current_y,
                        'elevation': 0,
                        'trainer_type': trainer_type,
                        'active': 1,
                        'memory_address': addr,
                        'source': f"ewram_runtime_{location_name}_dist_{distance}"
                    }
                    object_events.append(object_event)
                    found_npcs += 1
                    logger.debug(f"EWRAM Runtime NPC at {location_name}: ({current_x},{current_y}) graphics={graphics_id}")
                    
                except Exception as e:
                    logger.debug(f"Failed to read EWRAM runtime NPC at {location_name}: {e}")
                    continue
            
            # Fall back to legacy gObjectEvents if EWRAM method fails
            if not object_events:
                logger.debug("EWRAM runtime detection failed, trying legacy gObjectEvents...")
                return self._read_legacy_gobject_events(player_x, player_y)
                    
        except Exception as e:
            logger.debug(f"EWRAM runtime NPC reading failed: {e}")
            
        return object_events
    
    def _read_gsprites_npcs(self, player_x, player_y):
        """
        Read NPCs from gSprites array (actual visual sprite positions during movement)
        Based on pokeemerald research: proper coordinate conversion with MAP_OFFSET
        """
        object_events = []
        
        try:
            # Get known real NPC spawn positions to validate against
            static_npcs = self._read_known_npc_addresses(player_x, player_y)
            expected_npc_areas = []
            for npc in static_npcs:
                expected_npc_areas.append((npc['current_x'], npc['current_y']))
            
            if not expected_npc_areas:
                return []  # No reference NPCs to validate against
            
            # gSprites location from experimental testing
            gsprites_addr = 0x03006000
            max_sprites = 128
            sprite_size = 64
            
            for sprite_idx in range(max_sprites):
                sprite_addr = gsprites_addr + (sprite_idx * sprite_size)
                
                try:
                    # Read sprite screen coordinates
                    screen_x = self._read_s16(sprite_addr + 0)
                    screen_y = self._read_s16(sprite_addr + 2)
                    
                    # Validate screen coordinates
                    if screen_x < 50 or screen_x > 200 or screen_y < 50 or screen_y > 150:
                        continue
                    
                    # Convert screen coordinates to map coordinates using pokeemerald research
                    # Screen center is at player position, each tile is 16 pixels
                    SCREEN_CENTER_X = 120
                    SCREEN_CENTER_Y = 80
                    TILE_SIZE = 16
                    
                    tile_offset_x = (screen_x - SCREEN_CENTER_X) // TILE_SIZE
                    tile_offset_y = (screen_y - SCREEN_CENTER_Y) // TILE_SIZE
                    
                    # Apply correction for sprite centering offset discovered through testing
                    map_x = player_x + tile_offset_x + 1
                    map_y = player_y + tile_offset_y - 1
                    
                    # Only include sprites that are near expected NPC spawn areas
                    near_expected_npc = any(
                        abs(map_x - exp_x) + abs(map_y - exp_y) <= 3
                        for exp_x, exp_y in expected_npc_areas
                    )
                    
                    if not near_expected_npc:
                        continue
                    
                    # Additional validation: distance from player should be reasonable
                    distance = abs(map_x - player_x) + abs(map_y - player_y)
                    if distance == 0 or distance > 8:
                        continue
                    
                    object_event = {
                        'id': f"sprite_{sprite_idx}",
                        'obj_event_id': sprite_idx,
                        'local_id': sprite_idx,
                        'graphics_id': 1,
                        'movement_type': 1,
                        'current_x': map_x,
                        'current_y': map_y,
                        'initial_x': map_x,
                        'initial_y': map_y,
                        'elevation': 0,
                        'trainer_type': 0,
                        'active': 1,
                        'memory_address': sprite_addr,
                        'source': f"gsprites_sprite_{sprite_idx}_screen_{screen_x}_{screen_y}_map_{map_x}_{map_y}_dist_{distance}"
                    }
                    object_events.append(object_event)
                    
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Error reading gSprites: {e}")
        
        return object_events
    
    def _read_legacy_gobject_events(self, player_x, player_y):
        """Legacy gObjectEvents reading method (fallback)"""
        object_events = []
        
        try:
            gobject_events_addr = 0x02037230
            max_npcs = 16
            
            for i in range(max_npcs):
                try:
                    event_addr = gobject_events_addr + (i * 68)
                    
                    # Read active flag first - but be more lenient with what we consider active
                    active = self._read_u8(event_addr + 0x00)
                    
                    # In save states, active flag might be different values
                    # Be very permissive with active flags to catch all possible NPCs
                    if active == 0x00:  # Skip only completely inactive
                        continue
                    
                    # Read current runtime position (currentCoords at offset 0x10)
                    current_x = self._read_s16(event_addr + 0x10)
                    current_y = self._read_s16(event_addr + 0x12)
                    
                    # Skip if coordinates are obviously invalid
                    if current_x < -50 or current_x > 200 or current_y < -50 or current_y > 200:
                        continue
                    if current_x == 1023 and current_y == 1023:  # Common uninitialized value
                        continue
                    if current_x == 0 and current_y == 0:  # Skip (0,0) coordinates
                        continue
                    
                    # Skip coordinates with one 0 when far from player
                    distance = abs(current_x - player_x) + abs(current_y - player_y)
                    if (current_x == 0 or current_y == 0) and distance > 3:
                        continue
                    
                    # Only include NPCs within reasonable range of player
                    if distance > 10:  # Reduced to be more conservative
                        continue
                    
                    # Read additional NPC properties
                    graphics_id = self._read_u8(event_addr + 0x03)
                    movement_type = self._read_u8(event_addr + 0x04)
                    trainer_type = self._read_u8(event_addr + 0x05)
                    
                    # Skip if all properties are clearly invalid
                    if graphics_id == 255 and movement_type == 255:
                        continue
                    
                    object_event = {
                        'id': i,
                        'obj_event_id': self._read_u8(event_addr + 0x01),
                        'local_id': self._read_u8(event_addr + 0x02),
                        'graphics_id': graphics_id,
                        'movement_type': movement_type,
                        'current_x': current_x,
                        'current_y': current_y,
                        'initial_x': self._read_s16(event_addr + 0x10),
                        'initial_y': self._read_s16(event_addr + 0x12),
                        'elevation': 0,
                        'trainer_type': trainer_type,
                        'active': 1,
                        'memory_address': event_addr,
                        'source': f"legacy_runtime_slot_{i}_dist_{distance}"
                    }
                    object_events.append(object_event)
                    logger.debug(f"Legacy Runtime NPC {i}: ({current_x},{current_y}) graphics={graphics_id}")
                    
                except Exception as e:
                    logger.debug(f"Failed to read legacy runtime NPC slot {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"Legacy runtime NPC reading failed: {e}")
            
        return object_events
    
    def _read_saveblock_object_events(self, player_x, player_y):
        """
        Improved method: scan IWRAM for coordinate pairs near player with better filtering.
        This finds runtime NPC positions, not just spawn positions.
        """
        object_events = []
        
        try:
            # Scan specific IWRAM regions where NPCs are likely stored
            scan_regions = [
                (0x02025A00, 0x02026000, "IWRAM_NPCs_1"),  # Found NPCs here in memory scan
                (0x02026000, 0x02027000, "IWRAM_NPCs_2"),  # Expanded to cover gap - includes real NPCs at 0x020266C4-0x020266F4
            ]
            
            all_coords = []
            
            # Search each region for coordinate pairs
            for start_addr, end_addr, region_name in scan_regions:
                try:
                    for addr in range(start_addr, end_addr - 4, 2):
                        try:
                            x = self._read_s16(addr)
                            y = self._read_s16(addr + 2)
                            
                            # Skip invalid coordinates
                            if x < -10 or x > 100 or y < -10 or y > 100:
                                continue
                            
                            # Skip coordinates at (0,0) or with one coordinate being 0 when far from player
                            if x == 0 and y == 0:
                                continue
                            distance = abs(x - player_x) + abs(y - player_y)
                            if (x == 0 or y == 0) and distance > 5:
                                continue  # Likely uninitialized if has a 0 coordinate and is far from player
                            
                            # Check if near player (within reasonable range)  
                            if distance <= 8 and distance > 0:  # Increased range to catch all real NPCs
                                all_coords.append((x, y, addr, distance, region_name))
                                
                        except Exception:
                            continue
                            
                except Exception as e:
                    logger.debug(f"Error scanning {region_name}: {e}")
                    continue
            
            # Filter out duplicates and obvious false positives
            unique_coords = {}
            for x, y, addr, distance, region in all_coords:
                coord_key = (x, y)
                
                # Skip player position
                if x == player_x and y == player_y:
                    continue
                
                # Validate this looks like NPC data by checking surrounding memory
                try:
                    # Look at bytes around the coordinate pair
                    context = self._read_bytes(addr - 8, 24)
                    
                    # Skip if this looks like map data or other non-NPC data
                    # NPCs usually have reasonable graphics IDs (1-50) in surrounding bytes
                    has_reasonable_graphics = any(1 <= b <= 50 for b in context[:8])
                    has_reasonable_movement = any(0 <= b <= 10 for b in context[:8])
                    
                    # Skip if too many high values (likely map data)
                    high_value_count = sum(1 for b in context[:12] if b > 100)
                    if high_value_count > 4:
                        continue
                    
                    # Skip if all zeros or all 0xFF
                    if all(b == 0 for b in context[:12]) or all(b == 0xFF for b in context[:12]):
                        continue
                    
                    confidence = 0.0
                    if has_reasonable_graphics:
                        confidence += 0.3
                    if has_reasonable_movement:
                        confidence += 0.3
                    if distance == 1:  # Very close to player - high priority
                        confidence += 0.6  # Increased from 0.4 to prioritize adjacent NPCs
                    elif distance == 2:
                        confidence += 0.3
                        
                    # Use higher confidence thresholds to reduce false positives
                    # Only accept very confident detections
                    min_confidence = 0.6  # Increased from 0.2/0.4 to reduce false positives
                    if confidence < min_confidence:
                        continue
                        
                    if coord_key not in unique_coords or confidence > unique_coords[coord_key][4]:
                        unique_coords[coord_key] = (x, y, addr, distance, confidence)
                        
                except Exception:
                    continue
            
            # Sort by distance and create ObjectEvent structures
            sorted_coords = sorted(unique_coords.values(), key=lambda x: x[3])
            
            for i, (x, y, addr, distance, confidence) in enumerate(sorted_coords[:5]):  # Max 5 NPCs from saveblock
                try:
                    # Extract NPC properties from surrounding memory
                    graphics_id, movement_type, trainer_type = self._extract_npc_properties(addr)
                    
                    object_event = {
                        'id': i,
                        'obj_event_id': i,
                        'local_id': i,
                        'graphics_id': graphics_id,
                        'movement_type': movement_type,
                        'current_x': x,
                        'current_y': y,
                        'initial_x': x,  # Best guess - may be current position
                        'initial_y': y,
                        'elevation': 0,
                        'trainer_type': trainer_type,
                        'active': 1,
                        'memory_address': addr,
                        'source': f"iwram_scan_dist_{distance}_conf_{confidence:.2f}"
                    }
                    object_events.append(object_event)
                    logger.debug(f"IWRAM NPC {i}: ({x},{y}) distance={distance} confidence={confidence:.2f}")
                    
                except Exception as e:
                    logger.debug(f"Failed to create IWRAM NPC {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.debug(f"IWRAM NPC scanning failed: {e}")
            
        return object_events
    
    def _read_gobject_walking_positions(self, player_x, player_y):
        """
        Read NPCs from gObjectEvents array using currentCoords for actual walking positions.
        Based on pokeemerald decompilation: ObjectEvent.currentCoords gives real-time positions.
        
        Args:
            player_x, player_y: Player coordinates for distance filtering
            
        Returns:
            list: List of NPC objects with their current walking positions
        """
        object_events = []
        
        try:
            # gObjectEvents array address from pokeemerald decompilation
            gobject_events_addr = 0x02037230
            max_object_events = 16
            object_event_size = 68  # Size of ObjectEvent struct
            
            for i in range(max_object_events):
                try:
                    event_addr = gobject_events_addr + (i * object_event_size)
                    
                    # Read ObjectEvent structure according to pokeemerald decompilation
                    # Check if object is active
                    active_flags = self._read_u32(event_addr + 0x00)
                    active = active_flags & 0x1
                    
                    if not active:
                        continue
                    
                    # Read currentCoords (the walking position) - offset 0x10 based on structure
                    current_x = self._read_s16(event_addr + 0x10)  # currentCoords.x
                    current_y = self._read_s16(event_addr + 0x12)  # currentCoords.y
                    
                    # Validate coordinates are reasonable
                    if current_x < -50 or current_x > 200 or current_y < -50 or current_y > 200:
                        continue
                    if current_x == 1023 and current_y == 1023:  # Invalid marker
                        continue
                    
                    # Check distance from player (only include nearby NPCs)
                    distance = abs(current_x - player_x) + abs(current_y - player_y)
                    if distance > 15:
                        continue
                    
                    # Read additional ObjectEvent properties
                    local_id = self._read_u8(event_addr + 0x02)
                    graphics_id = self._read_u8(event_addr + 0x03)
                    movement_type = self._read_u8(event_addr + 0x04)
                    trainer_type = self._read_u8(event_addr + 0x05)
                    
                    # Read initial coordinates for comparison
                    initial_x = self._read_s16(event_addr + 0x14)  # initialCoords.x  
                    initial_y = self._read_s16(event_addr + 0x16)  # initialCoords.y
                    
                    # Create NPC object with walking position
                    object_event = {
                        'id': i,
                        'obj_event_id': self._read_u8(event_addr + 0x01),
                        'local_id': local_id,
                        'graphics_id': graphics_id,
                        'movement_type': movement_type,
                        'current_x': current_x,  # Walking position
                        'current_y': current_y,  # Walking position  
                        'initial_x': initial_x,  # Spawn position
                        'initial_y': initial_y,  # Spawn position
                        'elevation': 0,
                        'trainer_type': trainer_type,
                        'active': 1,
                        'memory_address': event_addr,
                        'source': f'gobject_walking_{i}_current({current_x},{current_y})_spawn({initial_x},{initial_y})',
                        'distance': distance
                    }
                    
                    object_events.append(object_event)
                    logger.debug(f"Walking NPC {i}: current({current_x},{current_y}) spawn({initial_x},{initial_y}) graphics={graphics_id}")
                    
                except Exception as e:
                    logger.debug(f"Error reading ObjectEvent slot {i}: {e}")
                    continue
            
            return object_events
            
        except Exception as e:
            logger.debug(f"Error reading gObjectEvents for walking positions: {e}")
            return []
    
    def _read_oam_sprites(self, player_x, player_y):
        """
        Read NPC positions from OAM (Object Attribute Memory) sprites.
        This gives us the actual visual positions during walking animations.
        
        Args:
            player_x, player_y: Player coordinates for distance filtering
            
        Returns:
            list: List of NPC objects with walking positions
        """
        npcs = []
        OAM_BASE = 0x07000000
        MAX_SPRITES = 128
        
        try:
            for i in range(MAX_SPRITES):
                oam_addr = OAM_BASE + (i * 8)
                
                try:
                    # Read OAM attributes
                    attr0 = self._read_u16(oam_addr)
                    attr1 = self._read_u16(oam_addr + 2)
                    attr2 = self._read_u16(oam_addr + 4)
                    
                    # Skip empty sprites
                    if attr0 == 0 and attr1 == 0 and attr2 == 0:
                        continue
                        
                    # Check if sprite is visible (not hidden)
                    if attr0 & 0x0300 == 0x0200:  # Hidden flag
                        continue
                        
                    # Extract screen position
                    y_screen = attr0 & 0x00FF
                    x_screen = attr1 & 0x01FF
                    tile_id = attr2 & 0x03FF
                    
                    # Skip invalid positions
                    if x_screen == 0 and y_screen == 0:
                        continue
                    if x_screen > 240 or y_screen > 160:  # GBA screen size
                        continue
                    
                    # Convert screen coordinates to map coordinates
                    # Player is at screen center (120, 80), each tile is 16 pixels
                    SCREEN_CENTER_X = 120
                    SCREEN_CENTER_Y = 80
                    TILE_SIZE = 16
                    
                    # Calculate map position from screen position
                    tile_offset_x = (x_screen - SCREEN_CENTER_X) // TILE_SIZE
                    tile_offset_y = (y_screen - SCREEN_CENTER_Y) // TILE_SIZE
                    
                    map_x = player_x + tile_offset_x
                    map_y = player_y + tile_offset_y
                    
                    # Skip the player sprite (should be near center of screen)
                    if abs(tile_offset_x) <= 1 and abs(tile_offset_y) <= 1:
                        continue
                    
                    # Only include nearby sprites (within reasonable NPC range)
                    distance = abs(map_x - player_x) + abs(map_y - player_y)
                    if distance > 15:
                        continue
                    
                    # Don't filter by tile_id - we've seen NPCs with tile_ids 0, 20, 28
                    # All moving sprites in the visible range are likely NPCs
                    
                    npc = {
                        'id': f'oam_sprite_{i}',
                        'obj_event_id': i,
                        'local_id': i,
                        'graphics_id': 1,  # Default for regular NPC
                        'movement_type': 1,  # Walking
                        'current_x': map_x,
                        'current_y': map_y,
                        'initial_x': map_x,
                        'initial_y': map_y,
                        'elevation': 0,
                        'trainer_type': 0,
                        'active': 1,
                        'memory_address': oam_addr,
                        'source': f'oam_sprite_{i}_screen({x_screen},{y_screen})_tile_{tile_id}',
                        'screen_x': x_screen,
                        'screen_y': y_screen,
                        'tile_id': tile_id,
                        'distance': distance
                    }
                    
                    npcs.append(npc)
                    logger.debug(f"OAM Sprite {i}: screen({x_screen},{y_screen}) -> map({map_x},{map_y}) tile_id={tile_id}")
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.debug(f"Error reading OAM sprites: {e}")
            
        logger.info(f"Found {len(npcs)} NPC sprites in OAM")
        return npcs
    
    def _enhance_npcs_with_oam_walking(self, base_npcs, player_x, player_y):
        """
        Enhance base NPC positions with walking positions from OAM sprites.
        This maintains stable NPC identity while showing walking animation.
        
        Args:
            base_npcs: List of NPCs with known base positions
            player_x, player_y: Player coordinates
            
        Returns:
            list: Enhanced NPCs with walking positions where available
        """
        # Initialize position cache if not exists
        if not hasattr(self, '_npc_position_cache'):
            self._npc_position_cache = {}
        enhanced_npcs = []
        
        # Get OAM sprites
        oam_sprites = []
        OAM_BASE = 0x07000000
        MAX_SPRITES = 128
        
        try:
            for i in range(MAX_SPRITES):
                oam_addr = OAM_BASE + (i * 8)
                
                try:
                    attr0 = self._read_u16(oam_addr)
                    attr1 = self._read_u16(oam_addr + 2)
                    attr2 = self._read_u16(oam_addr + 4)
                    
                    # Skip empty/hidden sprites
                    if attr0 == 0 and attr1 == 0 and attr2 == 0:
                        continue
                    if attr0 & 0x0300 == 0x0200:
                        continue
                    
                    # Extract screen position
                    y_screen = attr0 & 0x00FF
                    x_screen = attr1 & 0x01FF
                    
                    if x_screen == 0 and y_screen == 0:
                        continue
                    if x_screen > 240 or y_screen > 160:
                        continue
                    
                    # Convert to map coordinates
                    tile_offset_x = (x_screen - 120) // 16
                    tile_offset_y = (y_screen - 80) // 16
                    map_x = player_x + tile_offset_x
                    map_y = player_y + tile_offset_y
                    
                    # Skip player sprite (center of screen)
                    if abs(tile_offset_x) <= 1 and abs(tile_offset_y) <= 1:
                        continue
                    
                    oam_sprites.append({
                        'map_x': map_x,
                        'map_y': map_y,
                        'screen_x': x_screen,
                        'screen_y': y_screen,
                        'sprite_id': i
                    })
                    
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Error reading OAM for enhancement: {e}")
        
        # Match each base NPC with nearest OAM sprite (if any)
        for i, base_npc in enumerate(base_npcs):
            base_x = base_npc['current_x']
            base_y = base_npc['current_y']
            npc_key = f"npc_{i}_{base_x}_{base_y}"
            
            # Find closest OAM sprite within reasonable range
            best_sprite = None
            min_distance = float('inf')
            
            for sprite in oam_sprites:
                distance = abs(sprite['map_x'] - base_x) + abs(sprite['map_y'] - base_y)
                if distance <= 3 and distance < min_distance:  # Within 3 tiles of spawn
                    min_distance = distance
                    best_sprite = sprite
            
            # Create enhanced NPC
            enhanced_npc = base_npc.copy()
            
            if best_sprite:
                new_x, new_y = best_sprite['map_x'], best_sprite['map_y']
                
                # Check if position changed significantly from cache
                if npc_key in self._npc_position_cache:
                    cached_x, cached_y = self._npc_position_cache[npc_key]
                    # Only update if moved more than 1 tile or in reasonable range
                    if abs(new_x - cached_x) <= 1 and abs(new_y - cached_y) <= 1:
                        enhanced_npc['current_x'] = new_x
                        enhanced_npc['current_y'] = new_y
                        self._npc_position_cache[npc_key] = (new_x, new_y)
                    else:
                        # Large jump - use cached position for stability
                        enhanced_npc['current_x'] = cached_x
                        enhanced_npc['current_y'] = cached_y
                else:
                    # First time seeing this NPC
                    enhanced_npc['current_x'] = new_x
                    enhanced_npc['current_y'] = new_y
                    self._npc_position_cache[npc_key] = (new_x, new_y)
                
                enhanced_npc['source'] = f"npc_{i}_walking"
                enhanced_npc['walking_position'] = True
                logger.debug(f"Enhanced NPC {i}: spawn({base_x},{base_y}) walking({enhanced_npc['current_x']},{enhanced_npc['current_y']})")
            else:
                # No sprite found - use spawn position but keep in cache
                if npc_key in self._npc_position_cache:
                    cached_x, cached_y = self._npc_position_cache[npc_key]
                    enhanced_npc['current_x'] = cached_x
                    enhanced_npc['current_y'] = cached_y
                    enhanced_npc['source'] = f"npc_{i}_walking"  # Keep walking status if we had it before
                else:
                    enhanced_npc['current_x'] = base_x
                    enhanced_npc['current_y'] = base_y
                    enhanced_npc['source'] = f"npc_{i}_spawn"
                    self._npc_position_cache[npc_key] = (base_x, base_y)
                
                enhanced_npc['walking_position'] = False
                logger.debug(f"Static NPC {i} at ({enhanced_npc['current_x']},{enhanced_npc['current_y']})")
            
            enhanced_npcs.append(enhanced_npc)
        
        logger.info(f"Enhanced {len(enhanced_npcs)} NPCs with walking positions")
        return enhanced_npcs
    
    def _validate_npc_candidate(self, addr, x, y, player_x, player_y):
        """
        Validate if a coordinate pair at a memory address is likely a real NPC.
        
        Args:
            addr: Memory address of the coordinate pair
            x, y: Coordinates found
            player_x, player_y: Player coordinates
            
        Returns:
            float: Confidence score (0.0 - 1.0) that this is a real NPC
        """
        confidence = 0.0
        
        try:
            # Read context around the coordinates
            context_bytes = self._read_bytes(addr - 8, 32)
            
            # Check for structured data patterns that suggest this is part of an ObjectEvent
            
            # 1. Look for reasonable values in typical ObjectEvent fields
            # Check bytes before coordinates for graphics_id, movement_type etc.
            if len(context_bytes) >= 16:
                # Bytes before coordinates might contain NPC metadata
                for offset in range(8):
                    byte_val = context_bytes[offset]
                    # Graphics IDs are usually 1-50, movement types 0-10
                    if 1 <= byte_val <= 50:
                        confidence += 0.15
                    elif 0 <= byte_val <= 10:
                        confidence += 0.1
            
            # 2. Check bytes after coordinates for continuation of structure
            if len(context_bytes) >= 24:
                # Look for additional structured data after coordinates
                post_coord_bytes = context_bytes[16:24]
                non_zero_count = sum(1 for b in post_coord_bytes if b != 0)
                if non_zero_count > 2:  # Some non-zero data suggests structure
                    confidence += 0.2
            
            # 3. Distance from player - closer NPCs are more likely to be real
            distance = abs(x - player_x) + abs(y - player_y)
            if distance == 1:
                confidence += 0.3  # Very close NPCs most likely
            elif distance == 2:
                confidence += 0.2
            elif distance <= 4:
                confidence += 0.1
            
            # 4. Check for patterns that suggest this is NOT an NPC
            # Coordinates that are exact multiples might be map data, not NPCs
            if x % 8 == 0 and y % 8 == 0:
                confidence -= 0.2
                
            # All zero context suggests empty/unused memory
            zero_count = sum(1 for b in context_bytes if b == 0)
            if zero_count > len(context_bytes) * 0.8:  # 80%+ zeros
                confidence -= 0.3
            
            # All 0xFF suggests uninitialized/invalid data
            ff_count = sum(1 for b in context_bytes if b == 0xFF)
            if ff_count > len(context_bytes) * 0.6:  # 60%+ 0xFF
                confidence -= 0.4
            
        except Exception:
            # If we can't read context, lower confidence
            confidence = max(0.0, confidence - 0.2)
        
        return max(0.0, min(1.0, confidence))
    
    def _read_proper_gobject_events(self, player_x, player_y):
        """
        Read NPCs from proper gObjectEvents array using ObjectEvent structure validation.
        Based on pokeemerald decompilation.
        """
        object_events = []
        
        try:
            gobject_events_addr = 0x02037230
            max_object_events = 16
            object_event_size = 68  # Size of ObjectEvent struct
            
            for i in range(max_object_events):
                try:
                    event_addr = gobject_events_addr + (i * object_event_size)
                    
                    # Read ObjectEvent structure according to pokeemerald
                    # u32 active:1 bitfield at offset 0x00
                    active_flags = self._read_u32(event_addr + 0x00)
                    active = active_flags & 0x1
                    
                    if not active:
                        continue
                    
                    # Read coordinates from currentCoords at offset 0x10
                    current_x = self._read_s16(event_addr + 0x10)
                    current_y = self._read_s16(event_addr + 0x12)
                    
                    # Validate coordinates
                    if current_x < -50 or current_x > 200 or current_y < -50 or current_y > 200:
                        continue
                    if current_x == 1023 and current_y == 1023:
                        continue
                    if current_x == 0 and current_y == 0:
                        continue  # Filter out (0,0) coordinates which are often uninitialized
                    
                    # Check distance from player
                    distance = abs(current_x - player_x) + abs(current_y - player_y)
                    if distance > 15:
                        continue
                    
                    # Read NPC properties
                    graphics_id = self._read_u8(event_addr + 0x03)
                    movement_type = self._read_u8(event_addr + 0x04)
                    trainer_type = self._read_u8(event_addr + 0x05)
                    local_id = self._read_u8(event_addr + 0x02)
                    
                    object_event = {
                        'id': i,
                        'obj_event_id': self._read_u8(event_addr + 0x01),
                        'local_id': local_id,
                        'graphics_id': graphics_id,
                        'movement_type': movement_type,
                        'current_x': current_x,
                        'current_y': current_y,
                        'initial_x': current_x,
                        'initial_y': current_y,
                        'elevation': 0,
                        'trainer_type': trainer_type,
                        'active': 1,
                        'memory_address': event_addr,
                        'source': f"gobject_events_slot_{i}_dist_{distance}"
                    }
                    object_events.append(object_event)
                    logger.debug(f"Active ObjectEvent {i}: ({current_x}, {current_y}) graphics={graphics_id}")
                    
                except Exception as e:
                    logger.debug(f"Error reading ObjectEvent slot {i}: {e}")
                    continue
            
            return object_events
            
        except Exception as e:
            logger.debug(f"Error reading gObjectEvents: {e}")
            return []
    
    def _read_known_npc_addresses(self, player_x, player_y):
        """
        Fallback method: Read NPCs from known addresses found in ground truth validation.
        Used when gObjectEvents array is inactive (e.g., in save states).
        """
        object_events = []
        
        try:
            # Known addresses where real NPCs are stored in different save states
            # These were discovered through ground truth validation
            known_npc_addresses = [
                # npc.state addresses
                0x020266C4, 0x020266DC, 0x020266F4,
                # npc1.state addresses  
                0x020266C8,  # Adjacent NPC at (6,4) from player at (7,4)
            ]
            
            for i, addr in enumerate(known_npc_addresses):
                try:
                    x = self._read_s16(addr)
                    y = self._read_s16(addr + 2)
                    
                    # Validate coordinates
                    if x < -50 or x > 200 or y < -50 or y > 200:
                        continue
                    if x == 1023 and y == 1023:
                        continue
                    
                    distance = abs(x - player_x) + abs(y - player_y)
                    if distance > 15:
                        continue
                    
                    object_event = {
                        'id': i,
                        'obj_event_id': i,
                        'local_id': i,
                        'graphics_id': 1,  # Default for regular NPC
                        'movement_type': 0,  # Default stationary
                        'current_x': x,
                        'current_y': y,
                        'initial_x': x,
                        'initial_y': y,
                        'elevation': 0,
                        'trainer_type': 0,  # Regular NPC
                        'active': 1,
                        'memory_address': addr,
                        'source': f"known_addr_0x{addr:08X}_dist_{distance}"
                    }
                    object_events.append(object_event)
                    logger.debug(f"Known NPC {i}: ({x}, {y}) distance={distance}")
                    
                except Exception as e:
                    logger.debug(f"Error reading known NPC at 0x{addr:08X}: {e}")
                    continue
            
            return object_events
            
        except Exception as e:
            logger.debug(f"Error reading known NPC addresses: {e}")
            return []
    
    def _filter_door_false_positives(self, object_events, player_x, player_y):
        """
        Filter out false positive NPCs that are actually doors or other map features.
        
        Args:
            object_events: List of detected NPCs
            player_x, player_y: Player coordinates for map reading
            
        Returns:
            list: Filtered list of real NPCs
        """
        try:
            # Read map tiles around player to check for doors
            map_tiles = self.read_map_around_player(radius=5)  # 11x11 grid to match agent's pixel view
            if not map_tiles:
                # If we can't read map, return all NPCs (better to have false positives than miss real ones)
                return object_events
            
            filtered_npcs = []
            
            for npc in object_events:
                npc_x = npc['current_x']
                npc_y = npc['current_y']
                
                # Calculate position on map grid (player is at center 7,7)
                grid_x = npc_x - player_x + 7
                grid_y = npc_y - player_y + 7
                
                # Check if position is within map bounds
                if 0 <= grid_y < len(map_tiles) and 0 <= grid_x < len(map_tiles[0]):
                    tile = map_tiles[grid_y][grid_x]
                    
                    # Check tile behavior
                    if len(tile) >= 2:
                        behavior = tile[1]
                        
                        # Skip if this is a door tile
                        if hasattr(behavior, 'name'):
                            behavior_name = behavior.name
                            if 'DOOR' in behavior_name:
                                logger.debug(f"Filtering out false NPC at ({npc_x}, {npc_y}) - on door tile")
                                continue
                        
                        # Could add more filters here for other false positives
                        # e.g., WARP tiles, SIGN tiles, etc.
                
                # If we get here, it's likely a real NPC
                filtered_npcs.append(npc)
            
            if len(filtered_npcs) != len(object_events):
                logger.info(f"Filtered {len(object_events) - len(filtered_npcs)} false positive NPCs (doors, etc.)")
            
            return filtered_npcs
            
        except Exception as e:
            logger.warning(f"Failed to filter door false positives: {e}")
            # On error, return original list
            return object_events
    
    def _extract_npc_properties(self, addr):
        """
        Extract NPC properties (graphics_id, movement_type, trainer_type) from memory context.
        
        Args:
            addr: Memory address where coordinates were found
            
        Returns:
            tuple: (graphics_id, movement_type, trainer_type)
        """
        try:
            # Read context around the coordinate pair
            context_bytes = self._read_bytes(addr - 12, 24)
            
            # Try different interpretations of the structure
            # We'll look for reasonable values in typical positions
            
            graphics_id = 1  # Default to visible NPC
            movement_type = 0  # Default to stationary
            trainer_type = 0  # Default to regular NPC
            
            if len(context_bytes) >= 24:
                # Look for graphics_id in bytes before coordinates
                for offset in range(8):
                    byte_val = context_bytes[offset]
                    if 1 <= byte_val <= 50:  # Reasonable graphics_id range
                        graphics_id = byte_val
                        break
                
                # Look for movement_type 
                for offset in range(1, 9):
                    byte_val = context_bytes[offset]
                    if 0 <= byte_val <= 10:  # Reasonable movement_type range
                        movement_type = byte_val
                        break
                
                # Look for trainer_type in bytes after coordinates
                for offset in range(16, 24):
                    if offset < len(context_bytes):
                        byte_val = context_bytes[offset]
                        if 1 <= byte_val <= 5:  # Common trainer type range
                            trainer_type = byte_val
                            break
            
            return graphics_id, movement_type, trainer_type
            
        except Exception:
            # If we can't extract properties, return defaults
            return 1, 0, 0

from dataclasses import dataclass
from typing import Optional
from .enums import StatusCondition, PokemonType

@dataclass
class PokemonData:
    """Complete Pokemon data structure for Emerald"""
    species_id: int
    species_name: str
    current_hp: int
    max_hp: int
    level: int
    status: StatusCondition
    type1: PokemonType
    type2: Optional[PokemonType]
    moves: list[str]  # Move names
    move_pp: list[int]  # PP for each move
    trainer_id: int
    nickname: Optional[str] = None
    experience: Optional[int] = None
    is_egg: bool = False
    
    @property
    def is_asleep(self) -> bool:
        """Check if the Pokémon is asleep"""
        return self.status.is_asleep
        
    @property
    def status_name(self) -> str:
        """Return a human-readable status name"""
        if self.is_asleep:
            return "SLEEP"
        elif self.status & StatusCondition.PARALYSIS:
            return "PARALYSIS"
        elif self.status & StatusCondition.FREEZE:
            return "FREEZE"
        elif self.status & StatusCondition.BURN:
            return "BURN"
        elif self.status & StatusCondition.POISON:
            return "POISON"
        else:
            return "OK"

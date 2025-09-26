"""
Pokemon Emerald Memory Reader Package

This package provides tools for reading memory from Pokemon Emerald games
running in mGBA or other GBA emulators.
"""

from .memory_reader import PokemonEmeraldReader
from .enums import (
    MetatileBehavior, 
    PokemonType, 
    PokemonSpecies, 
    Move, 
    Badge, 
    MapLocation, 
    Tileset, 
    StatusCondition
)
from .types import PokemonData

__version__ = "3.0.0-preview"
__author__ = "Seth Karten"

__all__ = [
    "PokemonEmeraldReader",
    "MetatileBehavior",
    "PokemonType", 
    "PokemonSpecies",
    "Move",
    "Badge",
    "MapLocation",
    "Tileset",
    "StatusCondition",
    "PokemonData"
]

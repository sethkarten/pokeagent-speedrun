"""Porymap mapping and formatting helpers extracted from state_formatter."""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _get_pokeemerald_root() -> Optional[Path]:
    """Get the pokeemerald root directory path. Delegates to centralized resolver."""
    from pokemon_env.porymap_paths import get_porymap_root
    return get_porymap_root()


# ROM location name to Porymap map name mapping
ROM_TO_PORYMAP_MAP = {
    # Intro/Cutscene locations
    "MOVING_VAN": "InsideOfTruck",  # Intro cutscene (moving van)
    
    # Towns
    "LITTLEROOT TOWN": "LittlerootTown",
    "OLDALE TOWN": "OldaleTown",
    "DEWFORD TOWN": "DewfordTown",
    "LAVARIDGE TOWN": "LavaridgeTown",
    "FALLARBOR TOWN": "FallarborTown",
    "VERDANTURF TOWN": "VerdanturfTown",
    "PACIFIDLOG TOWN": "PacifidlogTown",
    
    # Cities
    "PETALBURG CITY": "PetalburgCity",
    "SLATEPORT CITY": "SlateportCity",
    "MAUVILLE CITY": "MauvilleCity",
    "RUSTBORO CITY": "RustboroCity",
    "FORTREE CITY": "FortreeCity",
    "LILYCOVE CITY": "LilycoveCity",
    "MOSSDEEP CITY": "MossdeepCity",
    "SOOTOPOLIS CITY": "SootopolisCity",
    "EVER GRANDE CITY": "EverGrandeCity",
    
    # Routes
    "ROUTE 101": "Route101",
    "ROUTE 102": "Route102",
    "ROUTE 103": "Route103",
    "ROUTE 104": "Route104",
    "ROUTE 105": "Route105",
    "ROUTE 106": "Route106",
    "ROUTE 107": "Route107",
    "ROUTE 108": "Route108",
    "ROUTE 109": "Route109",
    "ROUTE 110": "Route110",
    "ROUTE 111": "Route111",
    "ROUTE 112": "Route112",
    "ROUTE 113": "Route113",
    "ROUTE 114": "Route114",
    "ROUTE 115": "Route115",
    "ROUTE 116": "Route116",
    "ROUTE 117": "Route117",
    "ROUTE 118": "Route118",
    "ROUTE 119": "Route119",
    "ROUTE 120": "Route120",
    "ROUTE 121": "Route121",
    "ROUTE 122": "Route122",
    "ROUTE 123": "Route123",
    "ROUTE 124": "Route124",
    "ROUTE 125": "Route125",
    "ROUTE 126": "Route126",
    "ROUTE 127": "Route127",
    "ROUTE 128": "Route128",
    "ROUTE 129": "Route129",
    "ROUTE 130": "Route130",
    "ROUTE 131": "Route131",
    "ROUTE 132": "Route132",
    "ROUTE 133": "Route133",
    "ROUTE 134": "Route134",
    
    # Buildings (common patterns)
    "PETALBURG WOODS": "PetalburgWoods",
    "RUSTURF TUNNEL": "RusturfTunnel",
    "RUSTURF TUNNEL ALT": "RusturfTunnel",  # Alternative map ID 0x1804

    # Route 110 Trick House (Group 29 = 0x1D)
    "ROUTE 110 TRICK HOUSE ENTRANCE ALT": "Route110_TrickHouseEntrance",
    "ROUTE 110 TRICK HOUSE END ALT": "Route110_TrickHouseEnd",
    "ROUTE 110 TRICK HOUSE CORRIDOR ALT": "Route110_TrickHouseCorridor",
    "ROUTE 110 TRICK HOUSE PUZZLE1 ALT": "Route110_TrickHousePuzzle1",
    "ROUTE 110 TRICK HOUSE PUZZLE2 ALT": "Route110_TrickHousePuzzle2",
    "ROUTE 110 TRICK HOUSE PUZZLE3 ALT": "Route110_TrickHousePuzzle3",
    "ROUTE 110 TRICK HOUSE PUZZLE4 ALT": "Route110_TrickHousePuzzle4",
    "ROUTE 110 TRICK HOUSE PUZZLE5 ALT": "Route110_TrickHousePuzzle5",
    "ROUTE 110 TRICK HOUSE PUZZLE6 ALT": "Route110_TrickHousePuzzle6",
    "ROUTE 110 TRICK HOUSE PUZZLE7 ALT": "Route110_TrickHousePuzzle7",
    "ROUTE 110 TRICK HOUSE PUZZLE8 ALT": "Route110_TrickHousePuzzle8",
    "ROUTE 110 SEASIDE CYCLING ROAD SOUTH ENTRANCE ALT": "Route110_SeasideCyclingRoadSouthEntrance",
    "ROUTE 110 SEASIDE CYCLING ROAD NORTH ENTRANCE ALT": "Route110_SeasideCyclingRoadNorthEntrance",

    # Professor Birch's Lab
    "LITTLEROOT TOWN PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",
    "PROFESSOR BIRCHS LAB": "LittlerootTown_ProfessorBirchsLab",

    # Raw map IDs (fallback when memory reader can't resolve location name)
    "Map_18_0B": "PetalburgWoods",  # Group 0x18 (Dungeons), Map 0x0B
    "Map_18_04": "RusturfTunnel",  # Group 0x18 (Indoor Route 104), Map 0x04
    "MAP_18_04": "RusturfTunnel",  # Alternate capitalization

        # The folloing mappings were generated via cursor and verified
    "ABANDONED SHIP CAPTAINS OFFICE": "AbandonedShip_CaptainsOffice",  # 100.0% match, verified
    "ABANDONED SHIP CORRIDORS 1F": "AbandonedShip_Corridors_1F",  # 100.0% match, verified
    "ABANDONED SHIP CORRIDORS B1F": "AbandonedShip_Corridors_B1F",  # 100.0% match, verified
    "ABANDONED SHIP DECK": "AbandonedShip_Deck",  # 100.0% match, verified
    "ABANDONED SHIP HIDDEN FLOOR CORRIDORS": "AbandonedShip_HiddenFloorCorridors",  # 100.0% match, verified
    "ABANDONED SHIP HIDDEN FLOOR ROOMS": "AbandonedShip_HiddenFloorRooms",  # 100.0% match, verified
    "ABANDONED SHIP ROOM B1F": "AbandonedShip_Room_B1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS 1F": "AbandonedShip_Rooms_1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS B1F": "AbandonedShip_Rooms_B1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS2 1F": "AbandonedShip_Rooms2_1F",  # 100.0% match, verified
    "ABANDONED SHIP ROOMS2 B1F": "AbandonedShip_Rooms2_B1F",  # 100.0% match, verified
    "ABANDONED SHIP UNDERWATER1": "AbandonedShip_Underwater1",  # 100.0% match, verified
    "ABANDONED SHIP UNDERWATER2": "AbandonedShip_Underwater2",  # 100.0% match, verified
    "ALTERING CAVE": "AlteringCave",  # 100.0% match, verified
    "ANCIENT TOMB": "AncientTomb",  # 100.0% match, verified
    "AQUA HIDEOUT 1F": "AquaHideout_1F",  # 100.0% match, verified
    "AQUA HIDEOUT B1F": "AquaHideout_B1F",  # 100.0% match, verified
    "AQUA HIDEOUT B2F": "AquaHideout_B2F",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP1": "AquaHideout_UnusedRubyMap1",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP2": "AquaHideout_UnusedRubyMap2",  # 100.0% match, verified
    "AQUA HIDEOUT UNUSED RUBY MAP3": "AquaHideout_UnusedRubyMap3",  # 100.0% match, verified
    "ARTISAN CAVE 1F": "ArtisanCave_1F",  # 100.0% match, verified
    "ARTISAN CAVE B1F": "ArtisanCave_B1F",  # 100.0% match, verified
    "CAVE OF ORIGIN 1F": "CaveOfOrigin_1F",  # 100.0% match, verified
    "CAVE OF ORIGIN B1F": "CaveOfOrigin_B1F",  # 100.0% match, verified
    "CAVE OF ORIGIN ENTRANCE": "CaveOfOrigin_Entrance",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP1": "CaveOfOrigin_UnusedRubySapphireMap1",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP2": "CaveOfOrigin_UnusedRubySapphireMap2",  # 100.0% match, verified
    "CAVE OF ORIGIN UNUSED RUBY SAPPHIRE MAP3": "CaveOfOrigin_UnusedRubySapphireMap3",  # 100.0% match, verified
    "DESERT RUINS": "DesertRuins",  # 100.0% match, verified
    "DESERT UNDERPASS": "DesertUnderpass",  # 100.0% match, verified
    "DEWFORD TOWN GYM": "DewfordTown_Gym",  # 100.0% match, verified
    "DEWFORD TOWN HALL": "DewfordTown_Hall",  # 100.0% match, verified
    "DEWFORD TOWN HOUSE1": "DewfordTown_House1",  # 100.0% match, verified
    "DEWFORD TOWN HOUSE2": "DewfordTown_House2",  # 100.0% match, verified
    "DEWFORD TOWN POKEMON CENTER 1F": "DewfordTown_PokemonCenter_1F",  # 100.0% match, verified
    "DEWFORD TOWN POKEMON CENTER 2F": "DewfordTown_PokemonCenter_2F",  # 100.0% match, verified
    "EVER GRANDE CITY CHAMPIONS ROOM": "EverGrandeCity_ChampionsRoom",  # 100.0% match, verified
    "EVER GRANDE CITY DRAKES ROOM": "EverGrandeCity_DrakesRoom",  # 100.0% match, verified
    "EVER GRANDE CITY GLACIAS ROOM": "EverGrandeCity_GlaciasRoom",  # 100.0% match, verified
    "EVER GRANDE CITY HALL OF FAME": "EverGrandeCity_HallOfFame",  # 100.0% match, verified
    "EVER GRANDE CITY HALL1": "EverGrandeCity_Hall1",  # 100.0% match, verified
    "EVER GRANDE CITY HALL2": "EverGrandeCity_Hall2",  # 100.0% match, verified
    "EVER GRANDE CITY HALL3": "EverGrandeCity_Hall3",  # 100.0% match, verified
    "EVER GRANDE CITY HALL4": "EverGrandeCity_Hall4",  # 100.0% match, verified
    "EVER GRANDE CITY HALL5": "EverGrandeCity_Hall5",  # 100.0% match, verified
    "EVER GRANDE CITY PHOEBES ROOM": "EverGrandeCity_PhoebesRoom",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON CENTER 1F": "EverGrandeCity_PokemonCenter_1F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON CENTER 2F": "EverGrandeCity_PokemonCenter_2F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON LEAGUE 1F": "EverGrandeCity_PokemonLeague_1F",  # 100.0% match, verified
    "EVER GRANDE CITY POKEMON LEAGUE 2F": "EverGrandeCity_PokemonLeague_2F",  # 100.0% match, verified
    "EVER GRANDE CITY SIDNEYS ROOM": "EverGrandeCity_SidneysRoom",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT BATTLE ROOM": "FallarborTown_BattleTentBattleRoom",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT CORRIDOR": "FallarborTown_BattleTentCorridor",  # 100.0% match, verified
    "FALLARBOR TOWN BATTLE TENT LOBBY": "FallarborTown_BattleTentLobby",  # 100.0% match, verified
    "FALLARBOR TOWN COZMOS HOUSE": "FallarborTown_CozmosHouse",  # 100.0% match, verified
    "FALLARBOR TOWN MART": "FallarborTown_Mart",  # 100.0% match, verified
    "FALLARBOR TOWN MOVE RELEARNERS HOUSE": "FallarborTown_MoveRelearnersHouse",  # 100.0% match, verified
    "FALLARBOR TOWN POKEMON CENTER 1F": "FallarborTown_PokemonCenter_1F",  # 100.0% match, verified
    "FALLARBOR TOWN POKEMON CENTER 2F": "FallarborTown_PokemonCenter_2F",  # 100.0% match, verified
    "FIERY PATH": "FieryPath",  # 100.0% match, verified
    "FIERY PATH INTERIOR": "FieryPath",  # Same location, different name variant
    "FORTREE CITY DECORATION SHOP": "FortreeCity_DecorationShop",  # 100.0% match, verified
    "FORTREE CITY GYM": "FortreeCity_Gym",  # 100.0% match, verified
    "FORTREE CITY HOUSE1": "FortreeCity_House1",  # 100.0% match, verified
    "FORTREE CITY HOUSE2": "FortreeCity_House2",  # 100.0% match, verified
    "FORTREE CITY HOUSE3": "FortreeCity_House3",  # 100.0% match, verified
    "FORTREE CITY HOUSE4": "FortreeCity_House4",  # 100.0% match, verified
    "FORTREE CITY HOUSE5": "FortreeCity_House5",  # 100.0% match, verified
    "FORTREE CITY MART": "FortreeCity_Mart",  # 100.0% match, verified
    "FORTREE CITY POKEMON CENTER 1F": "FortreeCity_PokemonCenter_1F",  # 100.0% match, verified
    "FORTREE CITY POKEMON CENTER 2F": "FortreeCity_PokemonCenter_2F",  # 100.0% match, verified
    "GRANITE CAVE 1F": "GraniteCave_1F",  # 100.0% match, verified
    "GRANITE CAVE 1F ALT": "GraniteCave_1F",  # Map ID 0x1807 - same layout as 0x1907
    "GRANITE CAVE B1F": "GraniteCave_B1F",  # 100.0% match, verified
    "GRANITE CAVE B1F ALT": "GraniteCave_B1F",  # Map ID 0x1808 - same layout as 0x1908
    "GRANITE CAVE B2F": "GraniteCave_B2F",  # 100.0% match, verified
    "GRANITE CAVE B2F ALT": "GraniteCave_B2F",  # Map ID 0x1809 - same layout as 0x1909
    "GRANITE CAVE STEVENS ROOM": "GraniteCave_StevensRoom",  # 100.0% match, verified
    "ISLAND CAVE": "IslandCave",  # 100.0% match, verified
    "JAGGED PASS": "JaggedPass",  # 100.0% match, verified
    "LAVARIDGE TOWN GYM 1F": "LavaridgeTown_Gym_1F",  # 100.0% match, verified
    "LAVARIDGE TOWN GYM B1F": "LavaridgeTown_Gym_B1F",  # 100.0% match, verified
    "LAVARIDGE TOWN HERB SHOP": "LavaridgeTown_HerbShop",  # 100.0% match, verified
    "LAVARIDGE TOWN HOUSE": "LavaridgeTown_House",  # 100.0% match, verified
    "LAVARIDGE TOWN MART": "LavaridgeTown_Mart",  # 100.0% match, verified
    "LAVARIDGE TOWN POKEMON CENTER 1F": "LavaridgeTown_PokemonCenter_1F",  # 100.0% match, verified
    "LAVARIDGE TOWN POKEMON CENTER 2F": "LavaridgeTown_PokemonCenter_2F",  # 100.0% match, verified
    "LILYCOVE CITY CONTEST HALL": "LilycoveCity_ContestHall",  # 100.0% match, verified
    "LILYCOVE CITY CONTEST LOBBY": "LilycoveCity_ContestLobby",  # 100.0% match, verified
    "LILYCOVE CITY COVE LILY MOTEL 1F": "LilycoveCity_CoveLilyMotel_1F",  # 100.0% match, verified
    "LILYCOVE CITY COVE LILY MOTEL 2F": "LilycoveCity_CoveLilyMotel_2F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 1F": "LilycoveCity_DepartmentStore_1F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 2F": "LilycoveCity_DepartmentStore_2F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 3F": "LilycoveCity_DepartmentStore_3F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 4F": "LilycoveCity_DepartmentStore_4F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE 5F": "LilycoveCity_DepartmentStore_5F",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE ELEVATOR": "LilycoveCity_DepartmentStoreElevator",  # 100.0% match, verified
    "LILYCOVE CITY DEPARTMENT STORE ROOFTOP": "LilycoveCity_DepartmentStoreRooftop",  # 100.0% match, verified
    "LILYCOVE CITY HARBOR": "LilycoveCity_Harbor",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE1": "LilycoveCity_House1",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE2": "LilycoveCity_House2",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE3": "LilycoveCity_House3",  # 100.0% match, verified
    "LILYCOVE CITY HOUSE4": "LilycoveCity_House4",  # 100.0% match, verified
    "LILYCOVE CITY LILYCOVE MUSEUM 1F": "LilycoveCity_LilycoveMuseum_1F",  # 100.0% match, verified
    "LILYCOVE CITY LILYCOVE MUSEUM 2F": "LilycoveCity_LilycoveMuseum_2F",  # 100.0% match, verified
    "LILYCOVE CITY MOVE DELETERS HOUSE": "LilycoveCity_MoveDeletersHouse",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON CENTER 1F": "LilycoveCity_PokemonCenter_1F",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON CENTER 2F": "LilycoveCity_PokemonCenter_2F",  # 100.0% match, verified
    "LILYCOVE CITY POKEMON TRAINER FAN CLUB": "LilycoveCity_PokemonTrainerFanClub",  # 100.0% match, verified
    "LILYCOVE CITY UNUSED MART": "LilycoveCity_UnusedMart",  # 100.0% match, verified
    "LITTLEROOT TOWN BRENDANS HOUSE 1F": "LittlerootTown_BrendansHouse_1F",  # 100.0% match, verified
    "LITTLEROOT TOWN BRENDANS HOUSE 2F": "LittlerootTown_BrendansHouse_2F",  # 100.0% match, verified
    "LITTLEROOT TOWN MAYS HOUSE 1F": "LittlerootTown_MaysHouse_1F",  # 100.0% match, verified
    "LITTLEROOT TOWN MAYS HOUSE 2F": "LittlerootTown_MaysHouse_2F",  # 100.0% match, verified
    "MAGMA HIDEOUT 1F": "MagmaHideout_1F",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 1R": "MagmaHideout_2F_1R",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 2R": "MagmaHideout_2F_2R",  # 100.0% match, verified
    "MAGMA HIDEOUT 2F 3R": "MagmaHideout_2F_3R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 1R": "MagmaHideout_3F_1R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 2R": "MagmaHideout_3F_2R",  # 100.0% match, verified
    "MAGMA HIDEOUT 3F 3R": "MagmaHideout_3F_3R",  # 100.0% match, verified
    "MAGMA HIDEOUT 4F": "MagmaHideout_4F",  # 100.0% match, verified
    "MAP RUSTURF TUNNEL": "RusturfTunnel",  # 89.7% match, verified
    "MARINE CAVE END": "MarineCave_End",  # 100.0% match, verified
    "MARINE CAVE ENTRANCE": "MarineCave_Entrance",  # 100.0% match, verified
    "MAUVILLE CITY BIKE SHOP": "MauvilleCity_BikeShop",  # 100.0% match, verified
    "MAUVILLE CITY GAME CORNER": "MauvilleCity_GameCorner",  # 100.0% match, verified
    "MAUVILLE CITY GYM": "MauvilleCity_Gym",  # 100.0% match, verified
    "MAUVILLE CITY HOUSE1": "MauvilleCity_House1",  # 100.0% match, verified
    "MAUVILLE CITY HOUSE2": "MauvilleCity_House2",  # 100.0% match, verified
    "MAUVILLE CITY MART": "MauvilleCity_Mart",  # 100.0% match, verified
    "MAUVILLE CITY POKEMON CENTER 1F": "MauvilleCity_PokemonCenter_1F",  # 100.0% match, verified
    "MAUVILLE CITY POKEMON CENTER 2F": "MauvilleCity_PokemonCenter_2F",  # 100.0% match, verified
    "METEOR FALLS 1F 1R": "MeteorFalls_1F_1R",  # 100.0% match, verified
    "METEOR FALLS 1F 2R": "MeteorFalls_1F_2R",  # 100.0% match, verified
    "METEOR FALLS B1F 1R": "MeteorFalls_B1F_1R",  # 100.0% match, verified
    "METEOR FALLS B1F 2R": "MeteorFalls_B1F_2R",  # 100.0% match, verified
    "METEOR FALLS STEVENS CAVE": "MeteorFalls_StevensCave",  # 100.0% match, verified
    "MIRAGE TOWER 1F": "MirageTower_1F",  # 100.0% match, verified
    "MIRAGE TOWER 2F": "MirageTower_2F",  # 100.0% match, verified
    "MIRAGE TOWER 3F": "MirageTower_3F",  # 100.0% match, verified
    "MIRAGE TOWER 4F": "MirageTower_4F",  # 100.0% match, verified
    "MOSSDEEP CITY GAME CORNER 1F": "MossdeepCity_GameCorner_1F",  # 100.0% match, verified
    "MOSSDEEP CITY GAME CORNER B1F": "MossdeepCity_GameCorner_B1F",  # 100.0% match, verified
    "MOSSDEEP CITY GYM": "MossdeepCity_Gym",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE1": "MossdeepCity_House1",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE2": "MossdeepCity_House2",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE3": "MossdeepCity_House3",  # 100.0% match, verified
    "MOSSDEEP CITY HOUSE4": "MossdeepCity_House4",  # 100.0% match, verified
    "MOSSDEEP CITY MART": "MossdeepCity_Mart",  # 100.0% match, verified
    "MOSSDEEP CITY POKEMON CENTER 1F": "MossdeepCity_PokemonCenter_1F",  # 100.0% match, verified
    "MOSSDEEP CITY POKEMON CENTER 2F": "MossdeepCity_PokemonCenter_2F",  # 100.0% match, verified
    "MOSSDEEP CITY SPACE CENTER 1F": "MossdeepCity_SpaceCenter_1F",  # 100.0% match, verified
    "MOSSDEEP CITY SPACE CENTER 2F": "MossdeepCity_SpaceCenter_2F",  # 100.0% match, verified
    "MOSSDEEP CITY STEVENS HOUSE": "MossdeepCity_StevensHouse",  # 100.0% match, verified
    "MT CHIMNEY": "MtChimney",  # 100.0% match, verified
    "MT CHIMNEY CABLE CAR STATION": "MtChimney_CableCarStation",  # 100.0% match, verified
    "MT PYRE 1F": "MtPyre_1F",  # 100.0% match, verified
    "MT PYRE 2F": "MtPyre_2F",  # 100.0% match, verified
    "MT PYRE 3F": "MtPyre_3F",  # 100.0% match, verified
    "MT PYRE 4F": "MtPyre_4F",  # 100.0% match, verified
    "MT PYRE 5F": "MtPyre_5F",  # 100.0% match, verified
    "MT PYRE 6F": "MtPyre_6F",  # 100.0% match, verified
    "MT PYRE EXTERIOR": "MtPyre_Exterior",  # 100.0% match, verified
    "MT PYRE SUMMIT": "MtPyre_Summit",  # 100.0% match, verified
    "NEW MAUVILLE ENTRANCE": "NewMauville_Entrance",  # 100.0% match, verified
    "NEW MAUVILLE INSIDE": "NewMauville_Inside",  # 100.0% match, verified
    "OLDALE TOWN HOUSE1": "OldaleTown_House1",  # 100.0% match, verified
    "OLDALE TOWN HOUSE2": "OldaleTown_House2",  # 100.0% match, verified
    "OLDALE TOWN MART": "OldaleTown_Mart",  # 100.0% match, verified
    "OLDALE TOWN POKEMON CENTER 1F": "OldaleTown_PokemonCenter_1F",  # 100.0% match, verified
    "OLDALE TOWN POKEMON CENTER 2F": "OldaleTown_PokemonCenter_2F",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE1": "PacifidlogTown_House1",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE2": "PacifidlogTown_House2",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE3": "PacifidlogTown_House3",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE4": "PacifidlogTown_House4",  # 100.0% match, verified
    "PACIFIDLOG TOWN HOUSE5": "PacifidlogTown_House5",  # 100.0% match, verified
    "PACIFIDLOG TOWN POKEMON CENTER 1F": "PacifidlogTown_PokemonCenter_1F",  # 100.0% match, verified
    "PACIFIDLOG TOWN POKEMON CENTER 2F": "PacifidlogTown_PokemonCenter_2F",  # 100.0% match, verified
    "PETALBURG CITY GYM": "PetalburgCity_Gym",  # 100.0% match, verified
    "PETALBURG CITY HOUSE1": "PetalburgCity_House1",  # 100.0% match, verified
    "PETALBURG CITY HOUSE2": "PetalburgCity_House2",  # 100.0% match, verified
    "PETALBURG CITY MART": "PetalburgCity_Mart",  # 100.0% match, verified
    "PETALBURG CITY POKEMON CENTER 1F": "PetalburgCity_PokemonCenter_1F",  # 100.0% match, verified
    "PETALBURG CITY POKEMON CENTER 2F": "PetalburgCity_PokemonCenter_2F",  # 100.0% match, verified
    "PETALBURG CITY WALLYS HOUSE": "PetalburgCity_WallysHouse",  # 100.0% match, verified
    "ROUTE 104 MR BRINEYS HOUSE": "Route104_MrBrineysHouse",  # 100.0% match, verified
    "ROUTE 104 MR BRINEYS HOUSE ALT": "Route104_MrBrineysHouse",  # 91.9% match, verified
    "ROUTE 104 PRETTY PETAL FLOWER SHOP": "Route104_PrettyPetalFlowerShop",  # 100.0% match, verified
    "ROUTE 109 SEASHORE HOUSE": "Route109_SeashoreHouse",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE CORRIDOR": "Route110_TrickHouseCorridor",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE END": "Route110_TrickHouseEnd",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE ENTRANCE": "Route110_TrickHouseEntrance",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE1": "Route110_TrickHousePuzzle1",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE2": "Route110_TrickHousePuzzle2",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE3": "Route110_TrickHousePuzzle3",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE4": "Route110_TrickHousePuzzle4",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE5": "Route110_TrickHousePuzzle5",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE6": "Route110_TrickHousePuzzle6",  # 100.0% match, verified
    "ROUTE 110 TRICK HOUSE PUZZLE7": "Route110_TrickHousePuzzle7",  # 100.0% match, verified
    "ROUTE 111 OLD LADYS REST STOP": "Route111_OldLadysRestStop",  # 100.0% match, verified
    "ROUTE 111 WINSTRATE FAMILYS HOUSE": "Route111_WinstrateFamilysHouse",  # 100.0% match, verified
    "ROUTE 112 CABLE CAR STATION": "Route112_CableCarStation",  # 100.0% match, verified
    "ROUTE 113 GLASS WORKSHOP": "Route113_GlassWorkshop",  # 100.0% match, verified
    "ROUTE_113_GLASS_WORKSHOP": "Route113_GlassWorkshop",  # Enum name variant
    "ROUTE 114 FOSSIL MANIACS HOUSE": "Route114_FossilManiacsHouse",  # 100.0% match, verified
    "ROUTE 114 FOSSIL MANIACS TUNNEL": "Route114_FossilManiacsTunnel",  # 100.0% match, verified
    "ROUTE 114 LANETTES HOUSE": "Route114_LanettesHouse",  # 100.0% match, verified
    "ROUTE 116 TUNNELERS REST HOUSE": "Route116_TunnelersRestHouse",  # 100.0% match, verified
    "ROUTE 117 POKEMON DAY CARE": "Route117_PokemonDayCare",  # 100.0% match, verified
    "ROUTE 119 HOUSE": "Route119_House",  # 100.0% match, verified
    "ROUTE 119 WEATHER INSTITUTE 1F": "Route119_WeatherInstitute_1F",  # 100.0% match, verified
    "ROUTE 119 WEATHER INSTITUTE 2F": "Route119_WeatherInstitute_2F",  # 100.0% match, verified
    "ROUTE 121 SAFARI ZONE ENTRANCE": "Route121_SafariZoneEntrance",  # 100.0% match, verified
    "ROUTE 123 BERRY MASTERS HOUSE": "Route123_BerryMastersHouse",  # 100.0% match, verified
    "ROUTE 124 DIVING TREASURE HUNTERS HOUSE": "Route124_DivingTreasureHuntersHouse",  # 100.0% match, verified
    "RUSTBORO CITY CUTTERS HOUSE": "RustboroCity_CuttersHouse",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 1F": "RustboroCity_DevonCorp_1F",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 2F": "RustboroCity_DevonCorp_2F",  # 100.0% match, verified
    "RUSTBORO CITY DEVON CORP 3F": "RustboroCity_DevonCorp_3F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT1 1F": "RustboroCity_Flat1_1F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT1 2F": "RustboroCity_Flat1_2F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 1F": "RustboroCity_Flat2_1F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 2F": "RustboroCity_Flat2_2F",  # 100.0% match, verified
    "RUSTBORO CITY FLAT2 3F": "RustboroCity_Flat2_3F",  # 100.0% match, verified
    "RUSTBORO CITY GYM": "RustboroCity_Gym",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE1": "RustboroCity_House1",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE2": "RustboroCity_House2",  # 100.0% match, verified
    "RUSTBORO CITY HOUSE3": "RustboroCity_House3",  # 100.0% match, verified
    "RUSTBORO CITY MART": "RustboroCity_Mart",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON CENTER 1F": "RustboroCity_PokemonCenter_1F",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON CENTER 2F": "RustboroCity_PokemonCenter_2F",  # 100.0% match, verified
    "RUSTBORO CITY POKEMON SCHOOL": "RustboroCity_PokemonSchool",  # 100.0% match, verified
    "SCORCHED SLAB": "ScorchedSlab",  # 100.0% match, verified
    "SEAFLOOR CAVERN ENTRANCE": "SeafloorCavern_Entrance",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM1": "SeafloorCavern_Room1",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM2": "SeafloorCavern_Room2",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM3": "SeafloorCavern_Room3",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM4": "SeafloorCavern_Room4",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM5": "SeafloorCavern_Room5",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM6": "SeafloorCavern_Room6",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM7": "SeafloorCavern_Room7",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM8": "SeafloorCavern_Room8",  # 100.0% match, verified
    "SEAFLOOR CAVERN ROOM9": "SeafloorCavern_Room9",  # 100.0% match, verified
    "SEALED CHAMBER INNER ROOM": "SealedChamber_InnerRoom",  # 100.0% match, verified
    "SEALED CHAMBER OUTER ROOM": "SealedChamber_OuterRoom",  # 100.0% match, verified
    "SHOAL CAVE HIGH TIDE ENTRANCE ROOM": "ShoalCave_HighTideEntranceRoom",  # 100.0% match, verified
    "SHOAL CAVE HIGH TIDE INNER ROOM": "ShoalCave_HighTideInnerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE ENTRANCE ROOM": "ShoalCave_LowTideEntranceRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE ICE ROOM": "ShoalCave_LowTideIceRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE INNER ROOM": "ShoalCave_LowTideInnerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE LOWER ROOM": "ShoalCave_LowTideLowerRoom",  # 100.0% match, verified
    "SHOAL CAVE LOW TIDE STAIRS ROOM": "ShoalCave_LowTideStairsRoom",  # 100.0% match, verified
    "SKY PILLAR 1F": "SkyPillar_1F",  # 100.0% match, verified
    "SKY PILLAR 2F": "SkyPillar_2F",  # 100.0% match, verified
    "SKY PILLAR 3F": "SkyPillar_3F",  # 100.0% match, verified
    "SKY PILLAR 4F": "SkyPillar_4F",  # 100.0% match, verified
    "SKY PILLAR 5F": "SkyPillar_5F",  # 100.0% match, verified
    "SKY PILLAR ENTRANCE": "SkyPillar_Entrance",  # 100.0% match, verified
    "SKY PILLAR OUTSIDE": "SkyPillar_Outside",  # 100.0% match, verified
    "SKY PILLAR TOP": "SkyPillar_Top",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT BATTLE ROOM": "SlateportCity_BattleTentBattleRoom",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT CORRIDOR": "SlateportCity_BattleTentCorridor",  # 100.0% match, verified
    "SLATEPORT CITY BATTLE TENT LOBBY": "SlateportCity_BattleTentLobby",  # 100.0% match, verified
    "SLATEPORT CITY HARBOR": "SlateportCity_Harbor",  # 100.0% match, verified
    "SLATEPORT CITY HOUSE": "SlateportCity_House",  # 100.0% match, verified
    "SLATEPORT CITY MART": "SlateportCity_Mart",  # 100.0% match, verified
    "SLATEPORT CITY NAME RATERS HOUSE": "SlateportCity_NameRatersHouse",  # 100.0% match, verified
    "SLATEPORT CITY OCEANIC MUSEUM 1F": "SlateportCity_OceanicMuseum_1F",  # 100.0% match, verified
    "SLATEPORT CITY OCEANIC MUSEUM 2F": "SlateportCity_OceanicMuseum_2F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON CENTER 1F": "SlateportCity_PokemonCenter_1F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON CENTER 2F": "SlateportCity_PokemonCenter_2F",  # 100.0% match, verified
    "SLATEPORT CITY POKEMON FAN CLUB": "SlateportCity_PokemonFanClub",  # 100.0% match, verified
    "SLATEPORT CITY STERNS SHIPYARD 1F": "SlateportCity_SternsShipyard_1F",  # 100.0% match, verified
    "SLATEPORT CITY STERNS SHIPYARD 2F": "SlateportCity_SternsShipyard_2F",  # 100.0% match, verified
    "SOOTOPOLIS CITY GYM 1F": "SootopolisCity_Gym_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY GYM B1F": "SootopolisCity_Gym_B1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE1": "SootopolisCity_House1",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE2": "SootopolisCity_House2",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE3": "SootopolisCity_House3",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE4": "SootopolisCity_House4",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE5": "SootopolisCity_House5",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE6": "SootopolisCity_House6",  # 100.0% match, verified
    "SOOTOPOLIS CITY HOUSE7": "SootopolisCity_House7",  # 100.0% match, verified
    "SOOTOPOLIS CITY LOTAD AND SEEDOT HOUSE": "SootopolisCity_LotadAndSeedotHouse",  # 100.0% match, verified
    "SOOTOPOLIS CITY MART": "SootopolisCity_Mart",  # 100.0% match, verified
    "SOOTOPOLIS CITY MYSTERY EVENTS HOUSE 1F": "SootopolisCity_MysteryEventsHouse_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY MYSTERY EVENTS HOUSE B1F": "SootopolisCity_MysteryEventsHouse_B1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY POKEMON CENTER 1F": "SootopolisCity_PokemonCenter_1F",  # 100.0% match, verified
    "SOOTOPOLIS CITY POKEMON CENTER 2F": "SootopolisCity_PokemonCenter_2F",  # 100.0% match, verified
    "TERRA CAVE END": "TerraCave_End",  # 100.0% match, verified
    "TERRA CAVE ENTRANCE": "TerraCave_Entrance",  # 100.0% match, verified
    "UNDERWATER MARINE CAVE": "Underwater_MarineCave",  # 100.0% match, verified
    "UNDERWATER ROUTE 105": "Underwater_Route105",  # 100.0% match, verified
    "UNDERWATER ROUTE 124": "Underwater_Route124",  # 100.0% match, verified
    "UNDERWATER ROUTE 125": "Underwater_Route125",  # 100.0% match, verified
    "UNDERWATER ROUTE 126": "Underwater_Route126",  # 100.0% match, verified
    "UNDERWATER ROUTE 127": "Underwater_Route127",  # 100.0% match, verified
    "UNDERWATER ROUTE 128": "Underwater_Route128",  # 100.0% match, verified
    "UNDERWATER ROUTE 129": "Underwater_Route129",  # 100.0% match, verified
    "UNDERWATER ROUTE134": "Underwater_Route134",  # 100.0% match, verified
    "UNDERWATER SEAFLOOR CAVERN": "Underwater_SeafloorCavern",  # 100.0% match, verified
    "UNDERWATER SEALED CHAMBER": "Underwater_SealedChamber",  # 100.0% match, verified
    "UNDERWATER SOOTOPOLIS CITY": "Underwater_SootopolisCity",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT BATTLE ROOM": "VerdanturfTown_BattleTentBattleRoom",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT CORRIDOR": "VerdanturfTown_BattleTentCorridor",  # 100.0% match, verified
    "VERDANTURF TOWN BATTLE TENT LOBBY": "VerdanturfTown_BattleTentLobby",  # 100.0% match, verified
    "VERDANTURF TOWN FRIENDSHIP RATERS HOUSE": "VerdanturfTown_FriendshipRatersHouse",  # 100.0% match, verified
    "VERDANTURF TOWN HOUSE": "VerdanturfTown_House",  # 100.0% match, verified
    "VERDANTURF TOWN MART": "VerdanturfTown_Mart",  # 100.0% match, verified
    "VERDANTURF TOWN POKEMON CENTER 1F": "VerdanturfTown_PokemonCenter_1F",  # 100.0% match, verified
    "VERDANTURF TOWN POKEMON CENTER 2F": "VerdanturfTown_PokemonCenter_2F",  # 100.0% match, verified
    "VERDANTURF TOWN WANDAS HOUSE": "VerdanturfTown_WandasHouse",  # 100.0% match, verified
    "VICTORY ROAD 1F": "VictoryRoad_1F",  # 100.0% match, verified
    "VICTORY ROAD B1F": "VictoryRoad_B1F",  # 100.0% match, verified
    "VICTORY ROAD B2F": "VictoryRoad_B2F",  # 100.0% match, verified
}

def _get_porymap_map_name(location_name: Optional[str]) -> Optional[str]:
    """Convert ROM location name to porymap map name."""
    if not location_name:
        return None
    return ROM_TO_PORYMAP_MAP.get(location_name)


def _format_porymap_info(location_name: Optional[str], player_coords: Optional[Tuple[int, int]] = None, badge_count: int = 0, memory_reader: Any = None) -> List[str]:
    """
    Format porymap ground truth data (JSON and ASCII map) for the agent.
    
    Args:
        location_name: Current location name from ROM
        player_coords: Player's (x, y) coordinates
        badge_count: Number of badges player has (for game-state-aware map selection)
        memory_reader: Optional PokemonEmeraldReader for live metatile reads on dynamic maps
    
    Returns list of formatted strings to add to context.
    """
    context_parts = []
    
    if not location_name or location_name == 'TITLE_SEQUENCE' or location_name == 'Unknown':
        return context_parts
    
    try:
        # Import here to avoid circular dependencies and allow graceful failure
        from utils.mapping.porymap_json_builder import build_json_map_for_llm
        from utils.mapping.pokeemerald_parser import PokeemeraldMapLoader
        
        # Get pokeemerald root
        pokeemerald_root = _get_pokeemerald_root()
        if not pokeemerald_root:
            logger.warning(f"Porymap: Could not find pokeemerald root for location '{location_name}'")
            return context_parts
        
        # Convert ROM location name to porymap map name using mapping
        porymap_map_name = ROM_TO_PORYMAP_MAP.get(location_name)

        # Debug log for Glass Workshop issue
        if "GLASS" in location_name.upper() or "WORKSHOP" in location_name.upper():
            logger.warning(f"DEBUG: Glass Workshop mapping - ROM location: '{location_name}' -> porymap: '{porymap_map_name}'")

        # If not in direct mapping, try fuzzy matching
        if not porymap_map_name:
            map_loader = PokeemeraldMapLoader(pokeemerald_root)
            
            def normalize_for_matching(name: str) -> str:
                """Normalize location name for fuzzy matching."""
                # Normalize: lowercase, remove spaces/underscores, remove common suffixes
                normalized = str(name).lower().replace(" ", "").replace("_", "").replace("town", "").replace("city", "").replace("route", "")
                # Also try removing "professor", "birchs", "lab" for building matching
                if "professor" in normalized or "birch" in normalized or "lab" in normalized:
                    normalized = normalized.replace("professor", "").replace("birchs", "").replace("birch", "").replace("lab", "")
                return normalized
            
            rom_normalized = normalize_for_matching(location_name)
            maps_dir = pokeemerald_root / "data" / "maps"

            # Debug log for Glass Workshop fuzzy matching
            if "GLASS" in location_name.upper() or "WORKSHOP" in location_name.upper():
                logger.warning(f"DEBUG: Fuzzy matching Glass Workshop - normalized: '{rom_normalized}'")

            if maps_dir.exists():
                # Try direct directory name match
                best_match = None
                best_match_score = 0

                for map_dir in maps_dir.iterdir():
                    if not map_dir.is_dir() or map_dir.name == "map_groups.json":
                        continue
                    
                    map_name = map_dir.name
                    map_normalized = normalize_for_matching(map_name)
                    
                    # Exact match
                    if rom_normalized == map_normalized:
                        porymap_map_name = map_name
                        logger.info(f"Porymap: Matched '{location_name}' to '{porymap_map_name}' via fuzzy match")
                        break
                    
                    # Partial match scoring (for cases like "LITTLEROOT TOWN PROFESSOR BIRCHS LAB" -> "LittlerootTown_ProfessorBirchsLab")
                    if rom_normalized in map_normalized or map_normalized in rom_normalized:
                        match_length = min(len(rom_normalized), len(map_normalized))
                        if match_length > best_match_score and match_length > 5:  # Require at least 5 chars match
                            best_match = map_name
                            best_match_score = match_length
                
                # Use best partial match if no exact match found
                if not porymap_map_name and best_match:
                    porymap_map_name = best_match
                    logger.info(f"Porymap: Matched '{location_name}' to '{porymap_map_name}' via partial match (score: {best_match_score})")
        
        if not porymap_map_name:
            logger.warning(f"Porymap: Could not map ROM location '{location_name}' to porymap map name")
            return context_parts
        
        logger.info(f"Porymap: Building map for '{porymap_map_name}' (ROM location: '{location_name}', badges: {badge_count})")
        
        # Build JSON map (with grid included for pathfinding, even though we don't show it in prompt)
        # Pass badge_count for game-state-aware map selection (e.g., Petalburg Gym lobby)
        try:
            json_map = build_json_map_for_llm(porymap_map_name, pokeemerald_root, badge_count=badge_count)
        except ValueError as e:
            logger.error(f"Porymap: Failed to build map for '{porymap_map_name}' due to corrupted tileset data: {e}")
            logger.error("This likely indicates missing or corrupted tileset files in pokemon_env/porymap.")
            logger.error("Pathfinding will not be available for this location.")
            return context_parts
        
        # Ensure grid is built (even if we don't include it in the text output)
        if not json_map.get('grid'):
            # Rebuild with grid if needed
            from utils.mapping.porymap_json_builder import build_json_map
            try:
                json_map_with_grid = build_json_map(porymap_map_name, pokeemerald_root, include_grid=True, include_ascii=True)
                if json_map_with_grid and json_map_with_grid.get('grid'):
                    json_map['grid'] = json_map_with_grid['grid']
            except ValueError as e:
                logger.error(f"Porymap: Failed to rebuild grid for '{porymap_map_name}': {e}")

        if not json_map:
            logger.warning(f"Porymap: Failed to build JSON map for '{porymap_map_name}'")
            return context_parts

        # For dynamic maps (e.g. Mauville Gym), replace static grid with live
        # emulator metatiles so barrier changes are reflected in the agent's map.
        if memory_reader is not None:
            from utils.mapping.dynamic_map_overlay import apply_live_overlay_to_json_map
            apply_live_overlay_to_json_map(json_map, memory_reader, location_name)

        # Filter grid based on player elevation to handle multi-level maps
        # For caves/dungeons with multiple connected levels, be more permissive
        # For buildings/bridges with truly separate floors, be strict
        if player_coords and json_map.get('raw_tiles') and json_map.get('grid'):
            try:
                px, py = player_coords[0], player_coords[1]
                raw_tiles = json_map['raw_tiles']

                # Get player's current elevation from the tile they're standing on
                if 0 <= py < len(raw_tiles) and 0 <= px < len(raw_tiles[py]):
                    player_tile = raw_tiles[py][px]
                    if len(player_tile) >= 4:
                        player_elevation = player_tile[3]  # elevation is 4th element

                        # Check if this is a cave/dungeon (has elevation variety but connected paths)
                        # vs a multi-floor building (strict separation)
                        elevations_in_map = set()
                        for row in raw_tiles:
                            for tile in row:
                                if len(tile) >= 4:
                                    elevations_in_map.add(tile[3])

                        # NO tolerance - only allow exact elevation matches
                        # Elevation changes ONLY through stairs/doors/ledges (handled separately)
                        elevation_tolerance = 0  # Must be exact same elevation

                        # First pass: Find all stair and warp positions (S, D, arrow tiles, and ladders)
                        grid = json_map['grid']
                        warp_positions = set()
                        arrow_positions = set()  # Non-warp stairs (directional arrows - ledges)
                        ladder_positions = set()  # Ladder tiles that connect elevations
                        for y in range(len(grid)):
                            for x in range(len(grid[y])):
                                if grid[y][x] in ['S', 'D']:  # Stairs and Doors are both warps
                                    warp_positions.add((x, y))
                                elif grid[y][x] in ['←', '→', '↑', '↓']:  # Arrow tiles (ledges)
                                    arrow_positions.add((x, y))
                                elif grid[y][x] == '&':  # Ladder/bridge tiles
                                    ladder_positions.add((x, y))

                        # Combine all types of elevation connectors
                        # Ladders (&) connect different elevations vertically
                        all_stair_positions = warp_positions | arrow_positions | ladder_positions

                        # Build elevation connectivity graph from ladders AND adjacent walkable tiles
                        # Ladders (&) connect elevations, but also regular tiles at adjacent different elevations (slopes)
                        connected_elevations = set([player_elevation])  # Start with player's elevation

                        # Iteratively find all connected elevations (BFS)
                        prev_size = 0
                        max_iterations = 10  # Prevent infinite loops
                        iteration = 0
                        while len(connected_elevations) != prev_size and iteration < max_iterations:
                            prev_size = len(connected_elevations)
                            iteration += 1

                            # Detect direct walkable connections (slopes between elevations)
                            # Check for adjacent walkable tiles at different elevations
                            for y in range(len(grid)):
                                for x in range(len(grid[y])):
                                    if y < len(raw_tiles) and x < len(raw_tiles[y]):
                                        tile = raw_tiles[y][x]
                                        if len(tile) >= 4 and grid[y][x] in ['.', '~']:
                                            tile_elev = tile[3]
                                            if tile_elev in connected_elevations:
                                                # Check adjacent tiles in all 4 directions
                                                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                                                    nx, ny = x + dx, y + dy
                                                    if 0 <= ny < len(raw_tiles) and 0 <= nx < len(raw_tiles[ny]):
                                                        neighbor_tile = raw_tiles[ny][nx]
                                                        if len(neighbor_tile) >= 4:
                                                            neighbor_elev = neighbor_tile[3]
                                                            if ny < len(grid) and nx < len(grid[ny]):
                                                                neighbor_char = grid[ny][nx]
                                                                # If adjacent tile is walkable and at different elevation, connect them
                                                                if neighbor_char in ['.', '~', 'S', 'D'] and neighbor_elev != tile_elev:
                                                                    connected_elevations.add(neighbor_elev)

                            # Also check ladder tiles for connections
                            for lx, ly in ladder_positions:
                                if ly < len(raw_tiles) and lx < len(raw_tiles[ly]):
                                    ladder_tile = raw_tiles[ly][lx]
                                    if len(ladder_tile) >= 4:
                                        ladder_elev = ladder_tile[3]
                                        if ladder_elev in connected_elevations:
                                            # Check tiles in all 4 directions (ladders can connect up/down/left/right)
                                            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                                                nx, ny = lx + dx, ly + dy
                                                if 0 <= ny < len(raw_tiles) and 0 <= nx < len(raw_tiles[ny]):
                                                    neighbor_tile = raw_tiles[ny][nx]
                                                    if len(neighbor_tile) >= 4:
                                                        neighbor_elev = neighbor_tile[3]
                                                        # Add elevation from walkable tiles OR other ladders
                                                        if ny < len(grid) and nx < len(grid[ny]):
                                                            neighbor_char = grid[ny][nx]
                                                            if neighbor_char in ['.', '~', 'S', 'D', '&']:
                                                                connected_elevations.add(neighbor_elev)

                        # Filter the grid based on elevation
                        filtered_grid = []

                        for y in range(len(grid)):
                            filtered_row = []
                            for x in range(len(grid[y])):
                                original_char = grid[y][x]

                                # Always preserve special markers (warps, doors, NPCs, items, PC, TV/notebook)
                                if original_char in ['D', 'S', 'T', 'K', 'N', 'I', 'P', 'V', '←', '→', '↑', '↓']:
                                    filtered_row.append(original_char)
                                elif y < len(raw_tiles) and x < len(raw_tiles[y]):
                                    tile = raw_tiles[y][x]
                                    if len(tile) >= 4:
                                        tile_elevation = tile[3]
                                        elevation_diff = abs(tile_elevation - player_elevation)

                                        # Check if tile is adjacent to stairs (immediate neighbors only)
                                        # This includes both warps (S/D) and arrow tiles (←/→/↑/↓)
                                        is_adjacent_to_stairs = False
                                        for stair_x, stair_y in all_stair_positions:
                                            if abs(x - stair_x) + abs(y - stair_y) == 1:
                                                is_adjacent_to_stairs = True
                                                break

                                        # Adjacent to stairs: keep WALKABLE tiles accessible regardless of elevation
                                        # But still block walls/cliffs!
                                        if is_adjacent_to_stairs and original_char in ['.', '~', '←', '→', '↑', '↓']:
                                            filtered_row.append(original_char)  # Walkable tiles near stairs stay walkable
                                        # Handle bridge tiles (&) based on whether there's a path underneath
                                        elif original_char == '&':
                                            # Check if bridge has adjacent walkable tiles (. & & & . pattern)
                                            # This indicates a ground path underneath the bridge
                                            # Need to search through consecutive & tiles to find ground at both ends
                                            has_ground_path = False
                                            if 0 <= y < len(grid):
                                                # Search left through consecutive bridge tiles to find ground
                                                left_walkable = False
                                                search_x = x - 1
                                                while search_x >= 0 and search_x < len(grid[y]):
                                                    search_char = grid[y][search_x]
                                                    if search_char == '&':
                                                        # Continue searching left through bridge
                                                        search_x -= 1
                                                    elif search_char in ['.', '~']:
                                                        # Found walkable ground - check elevation
                                                        if search_x < len(raw_tiles[y]):
                                                            search_tile = raw_tiles[y][search_x]
                                                            if len(search_tile) >= 4:
                                                                search_elev = search_tile[3]
                                                                if abs(search_elev - player_elevation) <= elevation_tolerance:
                                                                    left_walkable = True
                                                        break
                                                    else:
                                                        # Hit a non-ground, non-bridge tile
                                                        break

                                                # Search right through consecutive bridge tiles to find ground
                                                right_walkable = False
                                                search_x = x + 1
                                                while search_x < len(grid[y]):
                                                    search_char = grid[y][search_x]
                                                    if search_char == '&':
                                                        # Continue searching right through bridge
                                                        search_x += 1
                                                    elif search_char in ['.', '~']:
                                                        # Found walkable ground - check elevation
                                                        if search_x < len(raw_tiles[y]):
                                                            search_tile = raw_tiles[y][search_x]
                                                            if len(search_tile) >= 4:
                                                                search_elev = search_tile[3]
                                                                if abs(search_elev - player_elevation) <= elevation_tolerance:
                                                                    right_walkable = True
                                                        break
                                                    else:
                                                        # Hit a non-ground, non-bridge tile
                                                        break

                                                # Ground path exists if BOTH left and right ends are walkable at player elevation
                                                has_ground_path = left_walkable and right_walkable

                                            # If there's a ground path underneath, show as walkable
                                            if has_ground_path and tile_elevation > player_elevation:
                                                filtered_row.append('.')  # Can walk under bridge
                                            else:
                                                filtered_row.append('&')  # Keep bridge visible for pathfinding
                                        # Block tiles beyond elevation tolerance
                                        elif elevation_diff > elevation_tolerance:
                                            # Check if tile's elevation is connected via ladders
                                            if tile_elevation in connected_elevations:
                                                filtered_row.append(original_char)  # Allow tiles at connected elevations
                                            # Special case: If player is in water, allow ground/grass tiles to show
                                            # (player can surf to shore, but shore players can't access water - handled by pathfinding)
                                            elif 0 <= py < len(grid) and 0 <= px < len(grid[py]) and grid[py][px] == 'W' and original_char in ['.', '~']:
                                                filtered_row.append(original_char)  # Allow ground tiles from water
                                            else:
                                                filtered_row.append('#')  # Block tiles at very different elevations
                                        else:
                                            filtered_row.append(original_char)  # Keep original tile
                                    else:
                                        filtered_row.append(original_char)
                                else:
                                    filtered_row.append(original_char)
                            filtered_grid.append(filtered_row)

                        # Update the grid and ASCII map
                        json_map['grid'] = filtered_grid

                        # Regenerate ASCII from filtered grid unless this map uses override ASCII
                        # (override ASCII must be kept verbatim so P/V/K/S/I/D match the override)
                        if not json_map.get('ascii_from_override'):
                            ascii_lines = [''.join(row) for row in filtered_grid]
                            json_map['ascii'] = '\n'.join(ascii_lines)

                        # Count how many tiles were blocked by elevation filtering
                        blocked_count = sum(1 for row in filtered_grid for cell in row if cell == '#')
                        original_blocked_count = sum(1 for row in grid for cell in row if cell == '#')
                        newly_blocked = blocked_count - original_blocked_count

                        logger.info(f"Elevation filtering: player at ({px}, {py}) elevation {player_elevation} - map has elevations {sorted(elevations_in_map)} - tolerance: {elevation_tolerance} - blocked {newly_blocked} additional tiles")
            except Exception as e:
                logger.warning(f"Failed to filter map by elevation: {e}")
        
        # Format for LLM
        context_parts.append("\n=== PORYMAP GROUND TRUTH MAP ===")
        context_parts.append(f"Location: {json_map.get('name', porymap_map_name)}")
        context_parts.append(f"Dimensions: {json_map['dimensions']['width']}x{json_map['dimensions']['height']}")
        
        # Add ASCII map with player position marked
        if json_map.get('ascii'):
            ascii_map = json_map['ascii']

            # Insert player position 'P' if provided
            if player_coords and json_map.get('grid'):
                px, py = player_coords[0], player_coords[1]
                grid = json_map['grid']

                # Check if player position is within map bounds
                if 0 <= py < len(grid) and 0 <= px < len(grid[0]) if grid else False:
                    # Split ASCII map into lines
                    ascii_lines = ascii_map.split('\n')

                    # Find the line corresponding to player's Y coordinate
                    if py < len(ascii_lines):
                        line = list(ascii_lines[py])
                        # Replace character at player's X position with 'P'
                        if px < len(line):
                            original_char = line[px]
                            line[px] = 'P'
                            ascii_lines[py] = ''.join(line)
                            ascii_map = '\n'.join(ascii_lines)
            
            context_parts.append("\nASCII Map:")
            context_parts.append(ascii_map)
            context_parts.append("(Legend: 'P' = Player, '.' = walkable, '#' = blocked, 'I' = item, '~' = tall grass, 'X' = out of bounds, 'T' = TV, 'K' = Clock, 'S' = Stairs/Warp, 'D' = Door)")
        
        # NOTE: Warps, Objects/NPCs, and Connections lists are DEPRECATED
        # This data is already included in the Map Data (JSON) section below.
        # Removed to reduce redundancy and potential agent confusion.
        
        # Add compact JSON map data (simplified format to save tokens)
        context_parts.append("\nMap Data (JSON):")
        
        # Include full object details (retain additional fields for precision)
        objects_for_json = json_map.get('objects', [])
        
        # Simplified warps
        simplified_warps = []
        for warp in json_map.get('warps', []):
            simplified_warps.append({
                "x": warp.get('x', 0),
                "y": warp.get('y', 0),
                "elevation": warp.get('elevation', 0),
                "dest_map": warp.get('dest_map', '?'),
                "dest_warp_id": warp.get('dest_warp_id', 0)
            })
        
        # Simplified connections
        simplified_connections = []
        for conn in json_map.get('connections', []):
            simplified_connections.append({
                "direction": conn.get('direction', '?'),
                "offset": conn.get('offset', 0),
                "map": conn.get('map', '?')
            })
        
        # BG events (PC, clock, TV, notebook, etc.)
        bg_events_for_json = json_map.get('bg_events', [])
        
        # Build compact JSON map (matching example format)
        compact_json_map = {
            "name": json_map.get('name'),
            "id": json_map.get('id'),
            "dimensions": json_map.get('dimensions'),
            "warps": simplified_warps,
            "objects": objects_for_json,
            "bg_events": bg_events_for_json,
            "connections": simplified_connections
        }
        context_parts.append(json.dumps(compact_json_map, indent=2))
        
        logger.info(f"Porymap: Successfully added map data for '{porymap_map_name}' ({json_map['dimensions']['width']}x{json_map['dimensions']['height']})")
        
        # Store porymap data in context_parts for later extraction (hidden from LLM but accessible for pathfinding)
        # We'll store it as a special marker that can be extracted from state
        return context_parts, json_map  # Return both formatted text and raw data
        
    except ImportError as e:
        # Log import errors as warnings
        logger.warning(f"Porymap modules not available: {e}")
    except Exception as e:
        # Log errors but don't break state formatting
        logger.warning(f"Error adding porymap info for location '{location_name}': {e}", exc_info=True)
    
    # Return formatted text only if json_map not available
    return context_parts


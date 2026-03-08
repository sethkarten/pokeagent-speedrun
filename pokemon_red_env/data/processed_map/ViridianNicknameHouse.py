tile_type = "house"
map_connection = "0"

tile_map = [
    [38, 41, 38, 41, 0, 0, 45, 46, 0, 0, 36, 36, 0, 0, 38, 41],
    [14, 15, 14, 15, 0, 0, 61, 62, 0, 0, 52, 52, 0, 0, 48, 49],
    [14, 15, 14, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 48, 49],
    [30, 31, 30, 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 30, 31],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 3, 38, 39, 39, 41, 2, 3, 1, 1, 1, 1],
    [1, 1, 1, 1, 18, 19, 54, 47, 47, 57, 18, 19, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 3, 54, 47, 47, 57, 2, 3, 1, 1, 1, 1],
    [1, 1, 1, 1, 18, 19, 60, 58, 58, 59, 18, 19, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [10, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 11],
    [8, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 9],
    [26, 27, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 26, 27],
    [24, 25, 1, 1, 20, 20, 20, 20, 1, 1, 1, 1, 1, 1, 24, 25],
]

coll_map = [
    ["X", "X", "X", "TalkToTownMapText", "X", "X", "X", "X"],
    ["X", "X", "O", "O", "O", "O", "O", "X"],
    ["O", "O", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "X", "X", "O", "O", "O"],
    ["O", "O", "O", "X", "X", "O", "O", "O"],
    ["O", "O", "O", "O", "O", "O", "O", "O"],
    ["X", "O", "O", "O", "O", "O", "O", "X"],
    ["X", "O", "WarpPoint", "WarpPoint", "O", "O", "O", "X"],
]

npc_data = [
    {"x": 5, "y": 3, "sprite": "SPRITE_BALDING_GUY", "movement": "STAY", "direction": "NONE", "text_id": "TEXT_VIRIDIANNICKNAMEHOUSE_BALDING_GUY"},
    {"x": 1, "y": 4, "sprite": "SPRITE_LITTLE_GIRL", "movement": "WALK", "direction": "UP_DOWN", "text_id": "TEXT_VIRIDIANNICKNAMEHOUSE_LITTLE_GIRL"},
    {"x": 5, "y": 5, "sprite": "SPRITE_BIRD", "movement": "WALK", "direction": "LEFT_RIGHT", "text_id": "TEXT_VIRIDIANNICKNAMEHOUSE_SPEAROW"},
    {"x": 4, "y": 0, "sprite": "SPRITE_CLIPBOARD", "movement": "STAY", "direction": "NONE", "text_id": "TEXT_VIRIDIANNICKNAMEHOUSE_SPEARY_SIGN"},
]

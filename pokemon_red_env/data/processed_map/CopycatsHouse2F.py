tile_type = "reds_house"
map_connection = "0"

tile_map = [
    [64, 65, 0, 0, 0, 0, 0, 0, 0, 0, 36, 37, 0, 0, 36, 37],
    [32, 33, 38, 39, 39, 41, 0, 0, 0, 0, 52, 53, 0, 0, 52, 53],
    [66, 67, 44, 42, 42, 43, 1, 1, 1, 1, 1, 1, 1, 1, 10, 11],
    [50, 51, 60, 58, 58, 59, 1, 1, 1, 1, 1, 1, 1, 1, 26, 27],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 6, 7, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 22, 23, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 14, 15, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 30, 31, 1, 1, 1, 1, 1, 1, 1, 1],
    [45, 46, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 68, 69, 1, 1],
    [61, 62, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 9, 1, 1],
    [61, 62, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 70, 71, 1, 1],
    [63, 47, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 24, 25, 1, 1],
]

coll_map = [
    ["X", "X", "X", "X", "X", "X", "X", "X"],
    ["SIGN_COPYCATSHOUSE2F_PC", "TalkToHiddenItems", "X", "O", "O", "O", "O", "WarpPoint"],
    ["O", "O", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "X", "O", "O", "O", "O"],
    ["O", "O", "O", "SIGN_COPYCATSHOUSE2F_SNES", "O", "O", "O", "O"],
    ["X", "O", "O", "O", "O", "O", "X", "O"],
    ["X", "O", "O", "O", "O", "O", "X", "O"],
]

npc_data = [
    {"x": 4, "y": 3, "sprite": "SPRITE_BRUNETTE_GIRL", "movement": "WALK", "direction": "ANY_DIR", "text_id": "TEXT_COPYCATSHOUSE2F_COPYCAT"},
    {"x": 4, "y": 6, "sprite": "SPRITE_BIRD", "movement": "WALK", "direction": "LEFT_RIGHT", "text_id": "TEXT_COPYCATSHOUSE2F_DODUO"},
    {"x": 5, "y": 1, "sprite": "SPRITE_MONSTER", "movement": "STAY", "direction": "DOWN", "text_id": "TEXT_COPYCATSHOUSE2F_MONSTER"},
    {"x": 2, "y": 0, "sprite": "SPRITE_BIRD", "movement": "STAY", "direction": "DOWN", "text_id": "TEXT_COPYCATSHOUSE2F_BIRD"},
    {"x": 1, "y": 6, "sprite": "SPRITE_FAIRY", "movement": "STAY", "direction": "RIGHT", "text_id": "TEXT_COPYCATSHOUSE2F_FAIRY"},
]

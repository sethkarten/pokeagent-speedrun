tile_type = "club"
map_connection = "0"

tile_map = [
    [6, 1, 2, 3, 6, 6, 1, 2, 3, 6, 6, 6, 6, 6, 6, 6],
    [6, 17, 18, 19, 6, 6, 17, 18, 19, 6, 7, 8, 6, 6, 6, 6],
    [11, 12, 14, 31, 15, 11, 12, 14, 15, 31, 7, 8, 15, 31, 15, 31],
    [27, 28, 9, 15, 30, 27, 28, 9, 31, 15, 7, 8, 31, 15, 31, 15],
    [15, 11, 12, 14, 15, 11, 12, 14, 15, 31, 7, 8, 15, 31, 15, 31],
    [30, 27, 28, 9, 30, 27, 28, 9, 31, 15, 7, 8, 31, 15, 31, 15],
    [15, 31, 15, 31, 15, 31, 15, 31, 15, 31, 7, 8, 54, 16, 16, 5],
    [31, 15, 31, 15, 31, 15, 31, 15, 31, 15, 23, 24, 23, 24, 23, 24],
    [15, 11, 12, 14, 15, 31, 15, 31, 15, 31, 15, 31, 15, 31, 15, 31],
    [30, 27, 28, 9, 31, 15, 31, 15, 31, 15, 31, 15, 31, 15, 31, 15],
    [15, 11, 12, 14, 15, 31, 15, 31, 15, 31, 15, 31, 15, 31, 15, 31],
    [30, 27, 28, 9, 31, 15, 31, 15, 31, 15, 31, 15, 31, 15, 31, 15],
    [15, 31, 15, 31, 15, 31, 15, 31, 15, 31, 15, 31, 29, 13, 15, 31],
    [31, 15, 31, 15, 31, 15, 31, 15, 31, 15, 31, 15, 21, 22, 31, 15],
    [15, 31, 15, 31, 10, 10, 10, 10, 15, 31, 15, 31, 15, 31, 29, 13],
    [31, 15, 31, 15, 26, 26, 26, 26, 31, 15, 31, 15, 31, 15, 21, 22],
]

coll_map = [
    ["X", "TalkToPrintNewBikeText", "X", "X", "X", "C", "X", "X"],
    ["X", "X", "TalkToPrintNewBikeText", "X", "O", "C", "O", "O"],
    ["X", "TalkToPrintNewBikeText", "X", "TalkToPrintNewBikeText", "O", "C", "O", "O"],
    ["O", "O", "O", "O", "O", "C", "C", "C"],
    ["TalkToPrintNewBikeText", "X", "O", "O", "O", "O", "O", "O"],
    ["X", "TalkToPrintNewBikeText", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "O", "O", "X", "O"],
    ["O", "O", "WarpPoint", "WarpPoint", "O", "O", "O", "X"],
]

npc_data = [
    {"x": 6, "y": 2, "sprite": "SPRITE_BIKE_SHOP_CLERK", "movement": "STAY", "direction": "NONE", "text_id": "TEXT_BIKESHOP_CLERK"},
    {"x": 5, "y": 6, "sprite": "SPRITE_MIDDLE_AGED_WOMAN", "movement": "WALK", "direction": "UP_DOWN", "text_id": "TEXT_BIKESHOP_MIDDLE_AGED_WOMAN"},
    {"x": 1, "y": 3, "sprite": "SPRITE_YOUNGSTER", "movement": "STAY", "direction": "UP", "text_id": "TEXT_BIKESHOP_YOUNGSTER"},
]

# map_preprocess.py

import json
import os
import re
import glob
from collections import defaultdict

env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
game_code_dir = os.path.join(env_dir, "data")

def parse_collision_tile_ids_asm(collision_asm_path):
    """
    Parses the file `data/tilesets/collision_tile_ids.asm` to extract
    each label (e.g., Overworld_Coll::) and its corresponding list of 
    collision tiles defined as `coll_tiles $xx, $yy, ...`.

    Returns an example like:
        {
        "overworld": {0x00, 0x10, 0x1b, ...},
        "underground": {0x0b, 0x0c, 0x13, ...},
        "gym": { ... },
        "reds_house": { ... },
        ...
        }
    """

    collision_dict = {}
    
    def label_to_tile_type(label):
        label = label.replace("_Coll", "").lower()
        if label in ["dojo", "gym"]:
            return "gym"
        elif label in ["mart", "pokecenter"]:
            return "pokecenter"
        elif label in ["forestgate", "museum", "gate"]:
            return "gate"
        elif "reds" in label:
            return "reds_house"
        return label

    coll_hex_pattern = re.compile(r"\$([0-9A-Fa-f]+)")
    
    with open(collision_asm_path, "r", encoding="utf-8") as f:
        current_labels = []
        for line in f:
            line = line.strip()
            if not line:
                continue

            label_matches = re.findall(r"([A-Za-z0-9_]+)_Coll::", line)
            if label_matches:
                current_labels = label_matches
                continue
            
            if line.startswith("coll_tiles"):
                tile_ids = coll_hex_pattern.findall(line)
                tile_ids_int = [int(x, 16) for x in tile_ids]
                
                for lbl in current_labels:
                    ttype = label_to_tile_type(lbl)
                    if ttype not in collision_dict:
                        collision_dict[ttype] = set()
                    collision_dict[ttype].update(tile_ids_int)
            else:
                current_labels = []
    
    return collision_dict


def parse_hidden_objects(asm_text):
    """Parse hidden_events.asm into {MAP_LABEL: {(x,y): description, ...}, ...}."""
    result = {}

    # Find each hidden_events_for block (macro expands to a label)
    block_pattern = re.compile(
        r'hidden_events_for\s+(\w+)\s*\n(.*?)db\s+-1',
        re.DOTALL
    )

    # Match both hidden_event and hidden_text_predef lines
    # Captures: x, y, function_name (the 3rd argument)
    event_pattern = re.compile(
        r'hidden_(?:event|text_predef)\s+(\d+)\s*,\s*(\d+)\s*,\s*(\w+)'
    )

    for block_match in block_pattern.finditer(asm_text):
        map_label = block_match.group(1)  # e.g. "REDS_HOUSE_2F"
        block_text = block_match.group(2)

        hidden_dict = {}
        for event_match in event_pattern.finditer(block_text):
            x = int(event_match.group(1))
            y = int(event_match.group(2))
            func_name = event_match.group(3)  # e.g. "OpenRedsPC", "PrintRedSNESText"
            hidden_dict[(x, y)] = func_name

        result[map_label] = hidden_dict

    return result

def classify_hidden_object(func_name: str) -> str:
    """Map hidden event function name to grid symbol (matching map_formatter.py)."""
    upper = func_name.upper()
    if "PC" in upper:
        return "PC"
    if any(k in upper for k in ("SNES", "GAMEBOY", "SLOTMACHINE")):
        return "T"
    if any(k in upper for k in ("BOOKCASE", "NOTEBOOK", "MAGAZINE", "BLACKBOARD", "BIKE")):
        return "B"
    if any(k in upper for k in ("POSTER", "EMAIL")):
        return "^"
    if "TRASH" in upper:
        return "U"
    if any(k in upper for k in ("STATUE", "QUIZ", "BINOCULARS", "DOJO")):
        return "?"
    if "BENCH" in upper:
        return "="
    return "#"


def parse_ledge_tiles_asm(ledge_tiles_asm_path):
    """
    Parses the file `ledge_tiles.asm` in the following format:
    db SPRITE_FACING_DOWN,  $2C, $37, D_DOWN
    db SPRITE_FACING_LEFT,  $2C, $27, D_LEFT
    ...
    db -1 ; end

    Returns a list in the form:
    [
        (stand_tile, ledge_tile, 'D'),
        (stand_tile, ledge_tile, 'L'),
        (stand_tile, ledge_tile, 'R'),
        ...
    ]
    """

    rules = []
    
    pattern = re.compile(
        r"db\s+SPRITE_FACING_(DOWN|LEFT|RIGHT)\s*,\s*\$([0-9A-Fa-f]+)\s*,\s*\$([0-9A-Fa-f]+)\s*,\s*(PAD_DOWN|PAD_LEFT|PAD_RIGHT)"
    )

    dir_map = {
        "PAD_DOWN": "↓",
        "PAD_LEFT": "←",
        "PAD_RIGHT": "→",
    }

    with open(ledge_tiles_asm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = pattern.search(line)
            if match:
                stand_hex = match.group(2)
                ledge_hex = match.group(3)
                input_req = match.group(4)

                stand_tile = int(stand_hex, 16)
                ledge_tile = int(ledge_hex, 16)
                direction_char = dir_map.get(input_req, "?")

                rules.append((stand_tile, ledge_tile, direction_char))

    return rules

def parse_pair_collision_tile_ids_asm(path):
    """
    Reads the file `pair_collision_tile_ids.asm` and collects
    tile types (e.g., 'CAVERN', 'FOREST') along with lists of (tile1, tile2) pairs.

    Example result:
    {
        "cavern": set([
        (0x20, 0x05),
        (0x41, 0x05),
        ...
        ]),
        "forest": set([
        (0x30, 0x2E),
        ...
        ])
    }

    Each pair (tile1, tile2) indicates a one-way movement:
    it is not possible to move from tile1 to tile2.
    """

    def normalize_ttype(tt: str):
        return tt.lower().replace(",", "")

    pair_dict = {}
    pattern = re.compile(
        r"db\s+([A-Za-z0-9_]+)\s*,\s*\$([0-9A-Fa-f]+)\s*,\s*\$([0-9A-Fa-f]+)"
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(";")[0]
            if not line:
                continue

            if line.endswith("::"):
                continue

            m = pattern.search(line)
            if m:
                raw_ttype = m.group(1)
                tile1_hex = m.group(2)
                tile2_hex = m.group(3)
                tile1 = int(tile1_hex, 16)
                tile2 = int(tile2_hex, 16)

                ttype = normalize_ttype(raw_ttype)
                if ttype not in pair_dict:
                    pair_dict[ttype] = set()
                pair_dict[ttype].add((tile1, tile2))

    return pair_dict

def load_map_constants_constants_asm(constants_asm_path):
    label_to_size = {}
    pattern = re.compile(r"map_const\s+([A-Z0-9_]+)\s*,\s*(\d+)\s*,\s*(\d+)")
    with open(constants_asm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = pattern.search(line)
            if match:
                label = match.group(1).strip()
                w = int(match.group(2))
                h = int(match.group(3))
                if label == 'UNDERGROUND_PATH_NORTH_SOUTH':
                    h -= 1
                label_to_size[label] = (w, h)
    return label_to_size

def parse_blk(path):
    with open(path, "rb") as f:
        return list(f.read())

def parse_blocks_from_bst(path):
    with open(path, "rb") as f:
        data = f.read()
    blocks = {}
    for block_id in range(len(data) // 16):
        tile_grid = []
        for row in range(4):
            offset = (block_id * 4 + row) * 4
            row_data = list(data[offset : offset + 4])
            tile_grid.append(row_data)
        blocks[block_id] = tile_grid
    return blocks

def build_tile_id_map(blk, blocks, map_width_blocks, map_height_blocks):
    width_tiles = map_width_blocks * 4
    height_tiles = map_height_blocks * 4
    tile_id_map = [[0 for _ in range(width_tiles)] for _ in range(height_tiles)]
    for by in range(map_height_blocks):
        for bx in range(map_width_blocks):
            block_index = by * map_width_blocks + bx
            block_id = blk[block_index]
            tile_grid = blocks[block_id]
            for ty in range(4):
                for tx in range(4):
                    tile_x = bx * 4 + tx
                    tile_y = by * 4 + ty
                    tile_id_map[tile_y][tile_x] = tile_grid[ty][tx]
    return tile_id_map

def parse_map_objects_asm(root_dir, map_name):
    objects_asm_path = os.path.join(root_dir, "data", "maps", "objects", f"{map_name}.asm")
    if not os.path.exists(objects_asm_path):
        return [], [], []

    warp_data = []
    signs_data = []
    npc_data = []

    warp_pattern = re.compile(
        r"warp_event\s+(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*,\s*(\d+)"
    )
    bg_pattern = re.compile(r"bg_event\s+(\d+)\s*,\s*(\d+)\s*,\s*TEXT_(\w+)")
    obj_pattern = re.compile(
        r'object_event\s+(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)'
    )

    with open(objects_asm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            wmatch = warp_pattern.search(line)
            if wmatch:
                warp_data.append({
                    "x": int(wmatch.group(1)),
                    "y": int(wmatch.group(2)),
                    "dest_map": wmatch.group(3),
                    "dest_warp_id": int(wmatch.group(4)),
                })
                continue
            smatch = bg_pattern.search(line)
            if smatch:
                signs_data.append({
                    "x": int(smatch.group(1)),
                    "y": int(smatch.group(2)),
                    "text_id": smatch.group(3),
                })
                continue
            omatch = obj_pattern.search(line)
            if omatch:
                npc_data.append({
                    "x": int(omatch.group(1)),
                    "y": int(omatch.group(2)),
                    "sprite": omatch.group(3),
                    "movement": omatch.group(4),
                    "direction": omatch.group(5),
                    "text_id": omatch.group(6),
                })
                continue

    return warp_data, signs_data, npc_data

def main():
    root_dir = os.path.join(game_code_dir, "pokered")

    with open(os.path.join(root_dir, "data", "events", "hidden_events.asm"), encoding="utf-8") as f:
        asm_text = f.read()

    hidden_objects_dict = parse_hidden_objects(asm_text)

    map_constants_asm_path = os.path.join(root_dir, "constants", "map_constants.asm")
    label_to_size = load_map_constants_constants_asm(map_constants_asm_path)

    collision_asm_path = os.path.join(root_dir, "data", "tilesets", "collision_tile_ids.asm")
    collision_dict = parse_collision_tile_ids_asm(collision_asm_path)

    ledge_tiles_asm_path = os.path.join(root_dir, "data", "tilesets", "ledge_tiles.asm")
    ledge_rules = parse_ledge_tiles_asm(ledge_tiles_asm_path)

    pair_collisions_asm_path = os.path.join(root_dir, "data", "tilesets", "pair_collision_tile_ids.asm")
    pair_collisions_dict = parse_pair_collision_tile_ids_asm(pair_collisions_asm_path)

    text_dict = {
        'plateau': {48: 'IndigoPlateauStatues'},
        'house': {61: 'TownMapText'},
        'lobby': {22: 'ElevatorText'},
    }

    grass_dict = {
        'overworld': {82},
        'plateau': {69},
        'forest': {32},
    }

    water_dict = {
        'overworld': {20},
        'forest': {20},
        'gym': {20},
        'ship': {20},
        'ship_port': {20},
        'cavern': {20},
        'facility': {20},
        'plateau': {20},
    }

    cut_dict = {
        'overworld': {61},
        'gym': {80},
    }

    counter_dict = {
        'pokecenter': {24, 25, 30},
        'mart': {24, 25, 30},
        'gym': {58},
        'dojo': {58},
        'museum': {23, 50},
        'gate': {23, 50},
        'forestgate': {23, 50},
        'cemetery': {18},
        'facility': {18},
        'lobby': {21, 54},
        'facility': {18},
        'club': {7, 23},
    }

    header_files = glob.glob(os.path.join(root_dir, "data", "maps", "headers", "*.asm"))

    output_dir = os.path.join(game_code_dir, "processed_map")
    os.makedirs(output_dir, exist_ok=True)

    for header_file in header_files:
        map_name = os.path.splitext(os.path.basename(header_file))[0]

        map_label = None
        tile_type = None
        with open(header_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("map_header"):
                    m = re.search(r"map_header\s+([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)", line)
                    if m:
                        map_label = m.group(2).strip()
                        tile_type = m.group(3).strip()
                        connect_direction = m.group(4).strip()
                    break

        hidden_dict = hidden_objects_dict.get(map_label)

        if not map_label or not tile_type:
            print(f"[{map_name}] No 'map_label' or 'tile_type'. Skip")
            continue

        tile_type = tile_type.lower()
        if tile_type == "dojo":
            tile_type = "gym"
        elif tile_type == "mart":
            tile_type = "pokecenter"
        elif tile_type in ["forest_gate", "museum", "gate"]:
            tile_type = "gate"
        elif "reds" in tile_type:
            tile_type = "reds_house"

        map_label_upper = map_label.upper()
        if map_label_upper not in label_to_size:
            print(f"[{map_name}] No {map_label} in map_constants.asm. Skip")
            continue

        mw, mh = label_to_size[map_label_upper]

        if map_name == 'UndergroundPathRoute7Copy':
            blk_path = os.path.join(root_dir, "maps", "UndergroundPathRoute7.blk")
        else:
            blk_path = os.path.join(root_dir, "maps", f"{map_name}.blk")

        bst_path = os.path.join(root_dir, "gfx", "blocksets", f"{tile_type}.bst")
        if not (os.path.exists(blk_path) and os.path.exists(bst_path)):
            print(f"[{map_name}] .blk/.bst file not exist. Skip")
            continue

        blk = parse_blk(blk_path)
        blocks = parse_blocks_from_bst(bst_path)
        if len(blk) < mw * mh:
            print(f"[{map_name}] blk file is smaller than expected (width*height={mw*mh})")

        tile_id_map = build_tile_id_map(blk, blocks, mw, mh)

        warp_data, signs_data, npc_data = parse_map_objects_asm(root_dir, map_name)

        # Build lookup sets for warps and signs
        warp_positions = {(w["x"], w["y"]) for w in warp_data}
        sign_positions = {(s["x"], s["y"]): s["text_id"] for s in signs_data}

        coll_set = collision_dict.get(tile_type, set())
        cut_set = cut_dict.get(tile_type, set())
        counter_set = counter_dict.get(tile_type, set())
        text_set = text_dict.get(tile_type)
        grass_set = grass_dict.get(tile_type, set())
        water_set = water_dict.get(tile_type, set())

        coll_map_h = mh * 2
        coll_map_w = mw * 2
        tile_map_h = mh * 4
        tile_map_w = mw * 4

        # Build hidden_objects list with classified symbols
        hidden_obj_list = []
        if hidden_dict:
            for (hx, hy), func_name in hidden_dict.items():
                symbol = classify_hidden_object(func_name)
                hidden_obj_list.append({
                    "x": hx, "y": hy,
                    "script": func_name,
                    "symbol": symbol,
                })

        # Build hidden position → symbol lookup
        hidden_symbol_lookup = {(h["x"], h["y"]): h["symbol"] for h in hidden_obj_list}

        grid = []
        for cy in range(coll_map_h):
            row = []
            tile_y = 2 * cy + 1
            for cx in range(coll_map_w):
                tile_x = 2 * cx
                if tile_y < tile_map_h and tile_x < tile_map_w:
                    tid = tile_id_map[tile_y][tile_x]
                    is_collision = (tid in coll_set)
                    is_text = (tid in text_set.keys()) if text_set else False
                    is_grass = (tid in grass_set)
                    is_water = (tid in water_set)
                    is_cut = (tid in cut_set)
                    is_counter = (tid in counter_set)
                    is_hidden = (cx, cy) in hidden_symbol_lookup
                else:
                    tid = 0
                    is_collision = False
                    is_cut = False
                    is_text = False
                    is_grass = False
                    is_water = False
                    is_counter = False
                    is_hidden = False

                # Classification priority — symbols match map_formatter.py
                if is_text:
                    cell_char = '#'
                elif is_hidden:
                    cell_char = hidden_symbol_lookup[(cx, cy)]
                elif is_grass:
                    cell_char = '~'
                elif is_water:
                    cell_char = 'W'
                elif is_collision:
                    cell_char = '.'
                else:
                    cell_char = '#'

                if is_cut:
                    cell_char = '#'
                elif is_counter:
                    cell_char = 'C'

                if (cx, cy) in warp_positions:
                    cell_char = 'D'
                elif (cx, cy) in sign_positions:
                    cell_char = '?'

                row.append(cell_char)
            grid.append(row)

        def in_bounds(cx_, cy_):
            return (0 <= cx_ < coll_map_w) and (0 <= cy_ < coll_map_h)

        # Ledge detection — only on blocked tiles
        for cy in range(coll_map_h):
            for cx in range(coll_map_w):
                if grid[cy][cx] != '#':
                    continue

                tile_x = 2 * cx
                tile_y = 2 * cy + 1
                if tile_x >= tile_map_w or tile_y >= tile_map_h:
                    continue

                tile_id = tile_id_map[tile_y][tile_x]
                for (stand_tile, ledge_tile, dir_char) in ledge_rules:
                    if tile_id == ledge_tile:
                        if dir_char == '↓':
                            nx, ny = cx, cy - 1
                        elif dir_char == '←':
                            nx, ny = cx + 1, cy
                        elif dir_char == '→':
                            nx, ny = cx - 1, cy
                        else:
                            continue

                        if in_bounds(nx, ny):
                            n_tile_x = 2 * nx
                            n_tile_y = 2 * ny + 1
                            if n_tile_x < tile_map_w and n_tile_y < tile_map_h:
                                neighbor_tile_id = tile_id_map[n_tile_y][n_tile_x]
                                if neighbor_tile_id == stand_tile:
                                    grid[cy][cx] = dir_char
                                    break

        # Pair collisions — mark as blocked wall
        pair_collisions = pair_collisions_dict.get(tile_type, set())
        if pair_collisions:
            for cy in range(coll_map_h):
                for cx in range(coll_map_w):
                    tile_x = 2 * cx
                    tile_y = 2 * cy + 1
                    if tile_x >= tile_map_w or tile_y >= tile_map_h:
                        continue
                    tile_id1 = tile_id_map[tile_y][tile_x]

                    for (dcx, dcy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx = cx + dcx
                        ny = cy + dcy
                        if not in_bounds(nx, ny):
                            continue
                        n_tile_x = 2 * nx
                        n_tile_y = 2 * ny + 1
                        if n_tile_x >= tile_map_w or n_tile_y >= tile_map_h:
                            continue

                        tile_id2 = tile_id_map[n_tile_y][n_tile_x]
                        if (tile_id1, tile_id2) in pair_collisions:
                            if grid[ny][nx] == '#':
                                grid[ny][nx] = '#'  # already blocked

        # Write JSON output
        json_path = os.path.join(output_dir, f"{map_name}.json")
        map_data = {
            "tile_type": tile_type,
            "map_connection": connect_direction,
            "dimensions": {"width": coll_map_w, "height": coll_map_h},
            "tile_map": tile_id_map,
            "grid": grid,
            "warps": warp_data,
            "signs": signs_data,
            "hidden_objects": hidden_obj_list,
            "npc_data": npc_data,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(map_data, f, indent=2, ensure_ascii=False)

        print(f"[{map_name}] -> Successfully Saved: {json_path}")

if __name__ == "__main__":
    main()

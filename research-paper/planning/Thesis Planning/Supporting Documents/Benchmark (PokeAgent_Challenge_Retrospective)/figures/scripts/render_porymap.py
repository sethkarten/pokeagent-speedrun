#!/usr/bin/env python3
"""
Render Pokemon Emerald porymap map data to PNG images.

Reads tile layouts, metatile definitions, tileset images, and palette files
to reconstruct full map images from porymap-exported data.
"""

import json
import struct
import os
import sys
import numpy as np
from PIL import Image

# Base paths
PORYMAP_DATA = "/Users/milkkarten/Documents/latex/PokeAgent-latex/pokeagent-speedrun/porymap_data/data"
OUTPUT_DIR = "/Users/milkkarten/Documents/latex/PokeAgent-latex/figures/track2_screenshots"

# Number of tiles in primary tileset (Gen 3 standard)
PRIMARY_TILE_COUNT = 512
# Number of metatiles in primary tileset (Gen 3 standard: IDs 0-511)
PRIMARY_METATILE_COUNT = 512

# Tileset name mapping from porymap gTileset names to folder names
TILESET_MAP = {
    "gTileset_General": ("primary", "general"),
    "gTileset_Building": ("primary", "building"),
    "gTileset_SecretBase": ("primary", "secret_base"),
    "gTileset_Lab": ("secondary", "lab"),
    "gTileset_PetalburgGym": ("secondary", "petalburg_gym"),
    "gTileset_Petalburg": ("secondary", "petalburg"),
}


def load_palettes(palette_dir):
    """Load all 16 palettes from JASC-PAL format .pal files."""
    palettes = []
    for i in range(16):
        pal_path = os.path.join(palette_dir, f"{i:02d}.pal")
        if os.path.exists(pal_path):
            with open(pal_path, 'r') as f:
                lines = f.read().strip().split('\n')
                assert lines[0] == "JASC-PAL", f"Invalid palette format in {pal_path}"
                num_colors = int(lines[2])
                colors = []
                for j in range(num_colors):
                    parts = lines[3 + j].split()
                    colors.append((int(parts[0]), int(parts[1]), int(parts[2])))
                palettes.append(colors)
        else:
            palettes.append([(0, 0, 0)] * 16)
    return palettes


def load_tiles_from_png(tiles_path):
    """Load 8x8 tiles from a tiles.png (indexed color mode).
    
    Returns a list of numpy arrays, each shape (8, 8) with palette index values.
    """
    img = Image.open(tiles_path)
    arr = np.array(img)
    height, width = arr.shape
    tiles_per_row = width // 8
    tiles_per_col = height // 8
    
    tiles = []
    for row in range(tiles_per_col):
        for col in range(tiles_per_row):
            tile = arr[row*8:(row+1)*8, col*8:(col+1)*8].copy()
            tiles.append(tile)
    
    return tiles


def load_metatiles(metatiles_path):
    """Load metatile definitions from metatiles.bin.
    
    Each metatile is 16 bytes = 8 tile entries (2 bytes each).
    Each entry: bits 0-9 = tile_id, bit 10 = xflip, bit 11 = yflip, bits 12-15 = palette.
    """
    with open(metatiles_path, 'rb') as f:
        data = f.read()
    
    num_metatiles = len(data) // 16
    metatiles = []
    
    for i in range(num_metatiles):
        entries = struct.unpack_from('<8H', data, i * 16)
        tile_entries = []
        for e in entries:
            tile_id = e & 0x3FF
            xflip = bool((e >> 10) & 1)
            yflip = bool((e >> 11) & 1)
            palette = (e >> 12) & 0xF
            tile_entries.append((tile_id, xflip, yflip, palette))
        metatiles.append(tile_entries)
    
    return metatiles


def render_tile(tile_pixels, palette, xflip=False, yflip=False):
    """Render an 8x8 tile with given palette and flips.
    
    Returns: (8, 8, 4) RGBA numpy array
    """
    result = np.zeros((8, 8, 4), dtype=np.uint8)
    
    pixels = tile_pixels.copy()
    if xflip:
        pixels = np.fliplr(pixels)
    if yflip:
        pixels = np.flipud(pixels)
    
    for y in range(8):
        for x in range(8):
            idx = pixels[y, x]
            if idx == 0:
                # Index 0 is transparent
                result[y, x] = (0, 0, 0, 0)
            else:
                r, g, b = palette[idx]
                result[y, x] = (r, g, b, 255)
    
    return result


def render_metatile(metatile_def, primary_tiles, secondary_tiles, palettes):
    """Render a 16x16 metatile.
    
    The metatile has 8 tile entries in this order:
    Layer 0 (bottom): [0]=top-left, [1]=top-right, [2]=bottom-left, [3]=bottom-right
    Layer 1 (top):    [4]=top-left, [5]=top-right, [6]=bottom-left, [7]=bottom-right
    
    Returns: (16, 16, 4) RGBA numpy array
    """
    result = np.zeros((16, 16, 4), dtype=np.uint8)
    
    # Positions for the 4 tiles in a 2x2 grid
    positions = [(0, 0), (8, 0), (0, 8), (8, 8)]  # (x, y)
    
    # Render both layers
    for layer in range(2):
        for i in range(4):
            tile_id, xflip, yflip, pal_idx = metatile_def[layer * 4 + i]
            
            # Get the tile pixels
            if tile_id < PRIMARY_TILE_COUNT:
                if tile_id < len(primary_tiles):
                    tile_pixels = primary_tiles[tile_id]
                else:
                    tile_pixels = np.zeros((8, 8), dtype=np.uint8)
            else:
                sec_idx = tile_id - PRIMARY_TILE_COUNT
                if sec_idx < len(secondary_tiles):
                    tile_pixels = secondary_tiles[sec_idx]
                else:
                    tile_pixels = np.zeros((8, 8), dtype=np.uint8)
            
            # Get palette
            palette = palettes[pal_idx] if pal_idx < len(palettes) else [(0, 0, 0)] * 16
            
            # Render the tile
            rendered = render_tile(tile_pixels, palette, xflip, yflip)
            
            x_off, y_off = positions[i]
            
            # Alpha composite: draw on top where alpha > 0
            mask = rendered[:, :, 3] > 0
            result[y_off:y_off+8, x_off:x_off+8][mask] = rendered[mask]
    
    return result


def render_map(map_name, layout_name, output_filename, primary_tileset_name, secondary_tileset_name):
    """Render a complete map to a PNG file."""
    
    print(f"Rendering map: {map_name}")
    
    # Find layout info from layouts.json
    layouts_json_path = os.path.join(PORYMAP_DATA, "layouts", "layouts.json")
    with open(layouts_json_path, 'r') as f:
        layouts_data = json.load(f)
    
    layout_info = None
    for layout in layouts_data['layouts']:
        if layout['id'] == layout_name:
            layout_info = layout
            break
    
    if layout_info is None:
        print(f"ERROR: Could not find layout for {map_name}")
        return
    
    map_width = layout_info['width']
    map_height = layout_info['height']
    primary_ts_name = layout_info['primary_tileset']
    secondary_ts_name = layout_info['secondary_tileset']
    
    print(f"  Layout: {map_width}x{map_height}")
    print(f"  Primary tileset: {primary_ts_name}")
    print(f"  Secondary tileset: {secondary_ts_name}")
    
    # Resolve tileset paths
    pri_type, pri_folder = TILESET_MAP[primary_ts_name]
    sec_type, sec_folder = TILESET_MAP[secondary_ts_name]
    
    pri_base = os.path.join(PORYMAP_DATA, "tilesets", pri_type, pri_folder)
    sec_base = os.path.join(PORYMAP_DATA, "tilesets", sec_type, sec_folder)
    
    print(f"  Primary path: {pri_base}")
    print(f"  Secondary path: {sec_base}")
    
    # Load palettes: primary provides 0-5, secondary provides 6-15
    pri_palettes = load_palettes(os.path.join(pri_base, "palettes"))
    sec_palettes = load_palettes(os.path.join(sec_base, "palettes"))
    
    merged_palettes = []
    for i in range(16):
        if i < 6:
            merged_palettes.append(pri_palettes[i])
        else:
            merged_palettes.append(sec_palettes[i])
    
    # Load tiles
    pri_tiles = load_tiles_from_png(os.path.join(pri_base, "tiles.png"))
    sec_tiles = load_tiles_from_png(os.path.join(sec_base, "tiles.png"))
    
    print(f"  Primary tiles: {len(pri_tiles)}")
    print(f"  Secondary tiles: {len(sec_tiles)}")
    
    # Load metatiles
    pri_metatiles = load_metatiles(os.path.join(pri_base, "metatiles.bin"))
    sec_metatiles = load_metatiles(os.path.join(sec_base, "metatiles.bin"))
    
    print(f"  Primary metatiles: {len(pri_metatiles)}")
    print(f"  Secondary metatiles: {len(sec_metatiles)}")
    
    # Load map data
    map_bin_path = os.path.join(PORYMAP_DATA, "layouts", map_name, "map.bin")
    with open(map_bin_path, 'rb') as f:
        map_data = f.read()
    
    num_entries = len(map_data) // 2
    map_entries = struct.unpack(f'<{num_entries}H', map_data)
    
    print(f"  Map entries: {num_entries} (expected {map_width * map_height})")
    assert num_entries == map_width * map_height, \
        f"Map size mismatch: {num_entries} != {map_width}*{map_height}={map_width*map_height}"
    
    # Pre-render all needed metatiles
    metatile_cache = {}
    
    # Create the output image
    img_width = map_width * 16
    img_height = map_height * 16
    print(f"  Output image: {img_width}x{img_height}")
    
    output = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    # Fill with opaque black background
    output[:, :, 3] = 255
    
    for y in range(map_height):
        for x in range(map_width):
            entry = map_entries[y * map_width + x]
            metatile_id = entry & 0x3FF
            
            if metatile_id not in metatile_cache:
                # Look up the metatile definition
                if metatile_id < PRIMARY_METATILE_COUNT:
                    if metatile_id < len(pri_metatiles):
                        mt_def = pri_metatiles[metatile_id]
                    else:
                        metatile_cache[metatile_id] = np.zeros((16, 16, 4), dtype=np.uint8)
                        metatile_cache[metatile_id][:, :, 3] = 255
                        continue
                else:
                    sec_idx = metatile_id - PRIMARY_METATILE_COUNT
                    if sec_idx < len(sec_metatiles):
                        mt_def = sec_metatiles[sec_idx]
                    else:
                        metatile_cache[metatile_id] = np.zeros((16, 16, 4), dtype=np.uint8)
                        metatile_cache[metatile_id][:, :, 3] = 255
                        continue
                
                metatile_cache[metatile_id] = render_metatile(
                    mt_def, pri_tiles, sec_tiles, merged_palettes
                )
            
            rendered_mt = metatile_cache[metatile_id]
            px = x * 16
            py = y * 16
            
            # Alpha composite onto output
            mask = rendered_mt[:, :, 3] > 0
            output[py:py+16, px:px+16][mask] = rendered_mt[mask]
    
    # Convert to RGB (drop alpha, use black background)
    rgb_output = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    alpha = output[:, :, 3:4].astype(float) / 255.0
    rgb_output = (output[:, :, :3].astype(float) * alpha).astype(np.uint8)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    img = Image.fromarray(rgb_output)
    img.save(output_path)
    print(f"  Saved: {output_path}")
    print()


def main():
    # Map 1: Birch's Lab
    render_map(
        map_name="LittlerootTown_ProfessorBirchsLab",
        layout_name="LAYOUT_LITTLEROOT_TOWN_PROFESSOR_BIRCHS_LAB",
        output_filename="birch_lab.png",
        primary_tileset_name="gTileset_Building",
        secondary_tileset_name="gTileset_Lab",
    )
    
    # Map 2: Petalburg Gym
    render_map(
        map_name="PetalburgCity_Gym",
        layout_name="LAYOUT_PETALBURG_CITY_GYM",
        output_filename="petalburg_gym_porymap.png",
        primary_tileset_name="gTileset_Building",
        secondary_tileset_name="gTileset_PetalburgGym",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download Gen 9 OU Pokemon sprites and update the Pokemon Pool in the SVG."""

import urllib.request
import base64
from pathlib import Path

# Gen 9 OU popular Pokemon (using ones with available sprites)
GEN9_OU_POKEMON = [
    {"name": "dragapult", "id": 887},
    {"name": "kingambit", "id": 983},
    {"name": "gholdengo", "id": 1000},
    {"name": "iron-valiant", "id": 1006},
    {"name": "great-tusk", "id": 984},
    {"name": "landorus", "id": 645},
    {"name": "heatran", "id": 485},
    {"name": "corviknight", "id": 823},
    {"name": "toxapex", "id": 748},
    {"name": "clefable", "id": 36},
]

def download_sprites(sprites_dir):
    """Download sprites for Gen 9 OU Pokemon."""
    sprites_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

    for pokemon in GEN9_OU_POKEMON:
        name = pokemon["name"]
        pid = pokemon["id"]
        url = f"{base_url}/{pid}.png"
        output_path = sprites_dir / f"{name}.png"

        if not output_path.exists():
            print(f"Downloading {name} (#{pid})...")
            try:
                urllib.request.urlretrieve(url, output_path)
            except Exception as e:
                print(f"  -> Error: {e}")
        else:
            print(f"Already have {name}")

def get_base64_image(filepath):
    """Convert image to base64 data URI."""
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def generate_pool_symbols(sprites_dir):
    """Generate SVG symbol definitions for pool Pokemon."""
    symbols = []
    for pokemon in GEN9_OU_POKEMON:
        name = pokemon["name"]
        sprite_path = sprites_dir / f"{name}.png"
        if sprite_path.exists():
            data_uri = get_base64_image(sprite_path)
            symbol = f'''    <!-- {name.replace("-", " ").title()} sprite (pool) -->
    <symbol id="pool-{name}" viewBox="0 0 96 96">
      <image href="{data_uri}" x="0" y="0" width="96" height="96"/>
    </symbol>'''
            symbols.append(symbol)
    return "\n".join(symbols)

def generate_pool_grid():
    """Generate SVG for the Pokemon pool grid."""
    # 5 Pokemon on top row, 4 on bottom row + "..."
    top_row = GEN9_OU_POKEMON[:5]
    bottom_row = GEN9_OU_POKEMON[5:9]

    grid_svg = []
    grid_svg.append('      <!-- Pokemon sprites showing variety -->')
    grid_svg.append('      <g transform="translate(250, 20)">')

    # Top row - 5 Pokemon
    for i, pokemon in enumerate(top_row):
        x = i * 32
        grid_svg.append(f'        <use href="#pool-{pokemon["name"]}" x="{x}" y="0" width="30" height="30"/>')

    # Bottom row - 4 Pokemon + "..."
    for i, pokemon in enumerate(bottom_row):
        x = 16 + i * 32
        grid_svg.append(f'        <use href="#pool-{pokemon["name"]}" x="{x}" y="32" width="30" height="30"/>')

    grid_svg.append('        <text x="150" y="55" class="small-text text-dark">...</text>')
    grid_svg.append('      </g>')

    return "\n".join(grid_svg)

def update_svg(script_dir):
    """Update the SVG with pool Pokemon sprites."""
    svg_path = script_dir / "battle_loop_diagram.svg"
    sprites_dir = script_dir / "assets" / "sprites"

    # Download sprites
    download_sprites(sprites_dir)

    # Generate symbols and grid
    pool_symbols = generate_pool_symbols(sprites_dir)
    pool_grid = generate_pool_grid()

    # Read SVG
    svg_content = svg_path.read_text()

    # Add pool symbols before </defs>
    svg_content = svg_content.replace('  </defs>', f'{pool_symbols}\n  </defs>')

    # Replace the old pokeball-outline grid in Pokemon Pool
    old_grid = '''      <!-- Mini pokeball icons to show variety -->
      <g transform="translate(250, 25)">
        <use href="#pokeball-outline" x="0" y="0" width="28" height="28"/>
        <use href="#pokeball-outline" x="32" y="0" width="28" height="28"/>
        <use href="#pokeball-outline" x="64" y="0" width="28" height="28"/>
        <use href="#pokeball-outline" x="96" y="0" width="28" height="28"/>
        <use href="#pokeball-outline" x="128" y="0" width="28" height="28"/>
        <use href="#pokeball-outline" x="16" y="32" width="28" height="28"/>
        <use href="#pokeball-outline" x="48" y="32" width="28" height="28"/>
        <use href="#pokeball-outline" x="80" y="32" width="28" height="28"/>
        <use href="#pokeball-outline" x="112" y="32" width="28" height="28"/>
        <text x="150" y="55" class="small-text text-dark">...</text>
      </g>'''

    svg_content = svg_content.replace(old_grid, pool_grid)

    # Write updated SVG
    svg_path.write_text(svg_content)
    print("\nSVG updated with Gen 9 OU Pokemon pool sprites!")

def main():
    script_dir = Path(__file__).parent.parent
    update_svg(script_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Embed Pokemon sprites as base64 into the SVG."""

import base64
from pathlib import Path

# Team assignments
TEAM1 = ["pikachu", "charizard", "mewtwo", "gengar", "gyarados", "snorlax"]
TEAM2 = ["blastoise", "dragonite", "alakazam", "machamp", "lapras", "venusaur"]

def get_base64_image(filepath):
    """Convert image to base64 data URI."""
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def generate_sprite_symbols():
    """Generate SVG symbol definitions for each Pokemon sprite."""
    script_dir = Path(__file__).parent.parent
    sprites_dir = script_dir / "assets" / "sprites"

    symbols = []
    for pokemon in TEAM1 + TEAM2:
        sprite_path = sprites_dir / f"{pokemon}.png"
        if sprite_path.exists():
            data_uri = get_base64_image(sprite_path)
            symbol = f'''    <!-- {pokemon.capitalize()} sprite -->
    <symbol id="sprite-{pokemon}" viewBox="0 0 96 96">
      <image href="{data_uri}" x="0" y="0" width="96" height="96"/>
    </symbol>'''
            symbols.append(symbol)

    return "\n".join(symbols)

def generate_team_row(team, y_offset, team_label, color_class):
    """Generate SVG for a team row with Pokemon sprites."""
    sprites = []
    sprites.append(f'      <g transform="translate(25, {y_offset})">')
    sprites.append(f'        <text x="0" y="22" class="label-text {color_class}">{team_label}:</text>')

    x_positions = [40, 82, 124, 166, 208, 250]
    for i, pokemon in enumerate(team):
        sprites.append(f'        <use href="#sprite-{pokemon}" x="{x_positions[i]}" y="-5" width="40" height="40"/>')

    sprites.append('      </g>')
    return "\n".join(sprites)

def main():
    print("Generated sprite symbols:")
    print("-" * 50)
    symbols = generate_sprite_symbols()
    print(symbols[:500] + "...")  # Preview

    print("\n" + "=" * 50)
    print("Team 1 row SVG:")
    print("-" * 50)
    team1_svg = generate_team_row(TEAM1, 85, "T₁", "agent1")
    print(team1_svg)

    print("\n" + "=" * 50)
    print("Team 2 row SVG:")
    print("-" * 50)
    team2_svg = generate_team_row(TEAM2, 135, "T₂", "agent2")
    print(team2_svg)

    # Save the symbols to a file for easy copying
    output_path = Path(__file__).parent.parent / "sprite_symbols.txt"
    with open(output_path, "w") as f:
        f.write("<!-- Pokemon sprite symbols (add to <defs> section) -->\n")
        f.write(symbols)
        f.write("\n\n<!-- Team 1 row (replace in Species Revealed) -->\n")
        f.write(team1_svg)
        f.write("\n\n<!-- Team 2 row (replace in Species Revealed) -->\n")
        f.write(team2_svg)

    print(f"\n\nSaved to {output_path}")

if __name__ == "__main__":
    main()

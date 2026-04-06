#!/usr/bin/env python3
"""Update the SVG with Pokemon sprites."""

from pathlib import Path

def main():
    script_dir = Path(__file__).parent.parent
    svg_path = script_dir / "battle_loop_diagram.svg"
    symbols_path = script_dir / "sprite_symbols.txt"

    # Read files
    svg_content = svg_path.read_text()
    symbols_content = symbols_path.read_text()

    # Extract sprite symbols (lines with <symbol> definitions)
    lines = symbols_content.split('\n')
    sprite_symbols = '\n'.join(lines[1:49])  # Skip first comment line

    # Extract team rows
    team1_row = '\n'.join(lines[51:60])
    team2_row = '\n'.join(lines[62:71])

    # 1. Add sprite symbols before </defs>
    svg_content = svg_content.replace('  </defs>', f'{sprite_symbols}\n  </defs>')

    # 2. Replace Team 1 pokeball outlines
    old_team1 = '''      <!-- Agent 1 pokeballs revealed (larger) -->
      <g transform="translate(25, 85)">
        <text x="0" y="18" class="label-text agent1">T₁:</text>
        <use href="#pokeball-outline" x="40" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="82" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="124" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="166" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="208" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="250" y="-5" width="36" height="36"/>
      </g>'''

    new_team1 = f'''      <!-- Agent 1 Pokemon revealed -->
{team1_row}'''

    svg_content = svg_content.replace(old_team1, new_team1)

    # 3. Replace Team 2 pokeball outlines
    old_team2 = '''      <!-- Agent 2 pokeballs revealed (larger) -->
      <g transform="translate(25, 135)">
        <text x="0" y="18" class="label-text agent2">T₂:</text>
        <use href="#pokeball-outline" x="40" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="82" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="124" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="166" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="208" y="-5" width="36" height="36"/>
        <use href="#pokeball-outline" x="250" y="-5" width="36" height="36"/>
      </g>'''

    new_team2 = f'''      <!-- Agent 2 Pokemon revealed -->
{team2_row}'''

    svg_content = svg_content.replace(old_team2, new_team2)

    # Write updated SVG
    svg_path.write_text(svg_content)
    print("SVG updated with Pokemon sprites!")

if __name__ == "__main__":
    main()

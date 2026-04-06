#!/usr/bin/env python3
"""Add Pokemon battlefield with lead Pokemon to the battle loop section."""

import urllib.request
import base64
from pathlib import Path

def download_battlefield(assets_dir):
    """Download a simple Pokemon battlefield image."""
    # We'll create a simple SVG battlefield instead of downloading
    # This gives us more control over the look
    pass

def get_base64_image(filepath):
    """Convert image to base64 data URI."""
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def create_battlefield_svg():
    """Create an SVG battlefield element."""
    # Simple Pokemon battlefield - green field with trainer areas
    battlefield = '''    <!-- Pokemon Battlefield -->
    <symbol id="battlefield" viewBox="0 0 400 200">
      <!-- Sky/background gradient -->
      <defs>
        <linearGradient id="sky-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" style="stop-color:#87CEEB;stop-opacity:0.3" />
          <stop offset="100%" style="stop-color:#E0F0FF;stop-opacity:0.5" />
        </linearGradient>
        <linearGradient id="field-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" style="stop-color:#7CB342;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#558B2F;stop-opacity:1" />
        </linearGradient>
      </defs>
      <!-- Background -->
      <rect x="0" y="0" width="400" height="200" fill="url(#sky-gradient)"/>
      <!-- Battle field (perspective) -->
      <path d="M 50,180 L 350,180 L 380,120 L 20,120 Z" fill="url(#field-gradient)" stroke="#33691E" stroke-width="2"/>
      <!-- Field lines -->
      <ellipse cx="200" cy="150" rx="80" ry="20" fill="none" stroke="#FFFFFF" stroke-width="2" opacity="0.5"/>
      <!-- Player 1 platform (left/back) -->
      <ellipse cx="100" cy="130" rx="45" ry="12" fill="#5D4037" stroke="#3E2723" stroke-width="2"/>
      <!-- Player 2 platform (right/front) -->
      <ellipse cx="300" cy="165" rx="50" ry="15" fill="#5D4037" stroke="#3E2723" stroke-width="2"/>
    </symbol>'''
    return battlefield

def generate_battle_scene_symbols(sprites_dir):
    """Generate symbols for battle scene Pokemon (back sprites for P1, front for P2)."""
    symbols = []

    # Pikachu (Agent 1's lead) - use back sprite
    pikachu_back_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/back/25.png"
    pikachu_back_path = sprites_dir / "pikachu_back.png"
    if not pikachu_back_path.exists():
        print("Downloading Pikachu back sprite...")
        urllib.request.urlretrieve(pikachu_back_url, pikachu_back_path)

    # Blastoise (Agent 2's lead) - use front sprite (already have it)
    blastoise_path = sprites_dir / "blastoise.png"

    # Create symbols
    if pikachu_back_path.exists():
        data_uri = get_base64_image(pikachu_back_path)
        symbols.append(f'''    <!-- Pikachu back sprite (Agent 1 lead) -->
    <symbol id="battle-pikachu" viewBox="0 0 96 96">
      <image href="{data_uri}" x="0" y="0" width="96" height="96"/>
    </symbol>''')

    if blastoise_path.exists():
        data_uri = get_base64_image(blastoise_path)
        symbols.append(f'''    <!-- Blastoise front sprite (Agent 2 lead) -->
    <symbol id="battle-blastoise" viewBox="0 0 96 96">
      <image href="{data_uri}" x="0" y="0" width="96" height="96"/>
    </symbol>''')

    return "\n".join(symbols)

def create_battle_scene_element():
    """Create the battle scene SVG element to insert into Phase 3."""
    # This will be placed between the two agent columns
    scene = '''      <!-- Battle Scene (between agent columns) -->
      <g transform="translate(255, 120)">
        <!-- Battlefield background -->
        <use href="#battlefield" x="0" y="0" width="290" height="145"/>
        <!-- Agent 1's Pokemon (Pikachu, back view, left side) -->
        <use href="#battle-pikachu" x="35" y="45" width="70" height="70"/>
        <!-- Agent 2's Pokemon (Blastoise, front view, right side) -->
        <use href="#battle-blastoise" x="175" y="60" width="80" height="80"/>
        <!-- VS indicator -->
        <text x="145" y="100" font-family="Impact, Arial Black, sans-serif" font-size="20" fill="#FF5722" text-anchor="middle" opacity="0.8">VS</text>
      </g>'''
    return scene

def update_svg(script_dir):
    """Update the SVG with battlefield and battle Pokemon."""
    svg_path = script_dir / "battle_loop_diagram.svg"
    sprites_dir = script_dir / "assets" / "sprites"
    sprites_dir.mkdir(parents=True, exist_ok=True)

    # Generate components
    battlefield_symbol = create_battlefield_svg()
    battle_sprites = generate_battle_scene_symbols(sprites_dir)
    battle_scene = create_battle_scene_element()

    # Read SVG
    svg_content = svg_path.read_text()

    # Add symbols before </defs>
    new_symbols = f"{battlefield_symbol}\n{battle_sprites}"
    svg_content = svg_content.replace('  </defs>', f'{new_symbols}\n  </defs>')

    # Insert battle scene into Phase 3, after the agent columns but before the engine
    # Find the marker and insert after Agent 2 column
    insert_marker = '''      <!-- Center Engine Box -->'
      <g transform="translate(265, 390)">'''

    # Actually, let's insert it right after the Agent 2 Column closing tag
    old_section = '''      </g>

      <!-- Center Engine Box -->
      <g transform="translate(265, 390)">'''

    new_section = f'''      </g>

{battle_scene}

      <!-- Center Engine Box -->
      <g transform="translate(265, 390)">'''

    # Find the right place - after Agent 2 Column
    # Look for the pattern after Action a₂
    old_pattern = '''          <text x="72" y="62" class="small-text text-dark">move | switch</text>
        </g>
      </g>

      <!-- Center Engine Box -->'''

    new_pattern = f'''          <text x="72" y="62" class="small-text text-dark">move | switch</text>
        </g>
      </g>

{battle_scene}

      <!-- Center Engine Box -->'''

    svg_content = svg_content.replace(old_pattern, new_pattern)

    # Write updated SVG
    svg_path.write_text(svg_content)
    print("SVG updated with Pokemon battlefield!")

def main():
    script_dir = Path(__file__).parent.parent
    update_svg(script_dir)

if __name__ == "__main__":
    main()

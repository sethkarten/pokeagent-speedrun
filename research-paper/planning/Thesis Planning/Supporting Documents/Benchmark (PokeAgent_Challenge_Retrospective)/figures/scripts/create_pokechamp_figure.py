#!/usr/bin/env python3
"""Generate the PokeChamp Architecture figure for NeurIPS publication."""

import re
import os

def extract_sprites_from_svg(svg_path):
    """Extract sprite symbol definitions from an existing SVG file."""
    with open(svg_path, 'r') as f:
        content = f.read()

    # Extract all symbol definitions
    symbol_pattern = r'(<symbol id="[^"]*"[^>]*>.*?</symbol>)'
    symbols = re.findall(symbol_pattern, content, re.DOTALL)

    # Also extract marker definitions
    marker_pattern = r'(<marker id="[^"]*"[^>]*>.*?</marker>)'
    markers = re.findall(marker_pattern, content, re.DOTALL)

    return symbols, markers

def create_pokechamp_svg():
    """Create the PokeChamp architecture SVG."""

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    battle_loop_svg = os.path.join(script_dir, "battle_loop_diagram.svg")

    # Extract existing sprites
    symbols, markers = extract_sprites_from_svg(battle_loop_svg)

    # Filter to get only the sprites we need
    needed_sprites = [
        'pool-dragapult', 'pool-kingambit', 'pool-gholdengo',
        'pool-iron-valiant', 'pool-great-tusk', 'pool-landorus',
        'pool-heatran', 'pool-corviknight', 'pool-toxapex', 'pool-clefable',
        'sprite-charizard', 'sprite-gengar'
    ]

    sprite_defs = []
    for symbol in symbols:
        for name in needed_sprites:
            if f'id="{name}"' in symbol:
                sprite_defs.append(symbol)
                break

    sprites_xml = '\n    '.join(sprite_defs)

    # Create the SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1520 720" width="1520" height="720">
  <defs>
    <!-- Color definitions matching battle_loop_diagram.svg -->
    <style>
      .agent1 {{ fill: #4C72B0; }}
      .agent2 {{ fill: #DD8452; }}
      .engine {{ fill: #8172B3; }}
      .hidden {{ fill: #C44E52; }}
      .revealed {{ fill: #55A868; }}
      .text-dark {{ fill: #2D3748; }}
      .text-light {{ fill: #718096; }}
      .bg-white {{ fill: #F7FAFC; }}
      /* Larger fonts for NeurIPS publication */
      .title-text {{ font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; font-weight: 700; font-size: 32px; }}
      .header-text {{ font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; font-weight: 700; font-size: 26px; }}
      .label-text {{ font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; font-weight: 600; font-size: 22px; }}
      .body-text {{ font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; font-weight: 500; font-size: 18px; }}
      .small-text {{ font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; font-weight: 400; font-size: 17px; }}
      .mono-text {{ font-family: 'Monaco', 'Consolas', monospace; font-size: 16px; }}
    </style>

    <!-- Arrow markers (smaller) -->
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#718096"/>
    </marker>
    <marker id="arrowhead-blue" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#4C72B0"/>
    </marker>
    <marker id="arrowhead-purple" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#8172B3"/>
    </marker>
    <marker id="arrowhead-green" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#55A868"/>
    </marker>
    <marker id="arrowhead-orange" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#DD8452"/>
    </marker>
    <marker id="arrowhead-red" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#C44E52"/>
    </marker>

    <!-- Brain icon -->
    <symbol id="brain" viewBox="0 0 256 256">
      <path d="M252,124a60.14,60.14,0,0,0-32-53.08,52,52,0,0,0-92-32.11A52,52,0,0,0,36,70.92a60,60,0,0,0,0,106.14,52,52,0,0,0,92,32.13,52,52,0,0,0,92-32.13A60.05,60.05,0,0,0,252,124ZM88,204a28,28,0,0,1-26.85-20.07c1,0,1.89.07,2.85.07h8a12,12,0,0,0,0-24H64A36,36,0,0,1,52,90.05a12,12,0,0,0,8-11.32V72a28,28,0,0,1,56,0v60.18a51.61,51.61,0,0,0-7.2-3.85,12,12,0,1,0-9.6,22A28,28,0,0,1,88,204Zm104-44h-8a12,12,0,0,0,0,24h8c1,0,1.9,0,2.85-.07a28,28,0,1,1-38-33.61,12,12,0,1,0-9.6-22,51.61,51.61,0,0,0-7.2,3.85V72a28,28,0,0,1,56,0v6.73a12,12,0,0,0,8,11.32,36,36,0,0,1-12,70Zm16-44a12,12,0,0,1-12,12,40,40,0,0,1-40-40V84a12,12,0,0,1,24,0v4a16,16,0,0,0,16,16A12,12,0,0,1,208,116ZM100,88a40,40,0,0,1-40,40,12,12,0,0,1,0-24A16,16,0,0,0,76,88V84a12,12,0,0,1,24,0Z"/>
    </symbol>

    <!-- Database icon -->
    <symbol id="database" viewBox="0 0 256 256">
      <path d="M128,24C74.17,24,28,48.34,28,80v96c0,31.66,46.17,56,100,56s100-24.34,100-56V80C228,48.34,181.83,24,128,24Zm0,24c44.18,0,76,17.05,76,32s-31.82,32-76,32S52,94.95,52,80,83.82,48,128,48ZM52,176V152.22c17.07,13.27,43.14,21.82,76,21.82s58.93-8.55,76-21.82V176c0,15-31.82,32-76,32S52,191,52,176Zm152-48c0,15-31.82,32-76,32s-76-17-76-32V112.22c17.07,13.27,43.14,21.82,76,21.82s58.93-8.55,76-21.82Z"/>
    </symbol>

    <!-- Loop/Refresh icon -->
    <symbol id="loop" viewBox="0 0 256 256">
      <path d="M224,48V96a8,8,0,0,1-8,8H168a8,8,0,0,1,0-16h28.69L168.4,59.71a80,80,0,0,0-113.13,0A8,8,0,0,1,44,48.4a96,96,0,0,1,135.77,0L208,76.69V48a8,8,0,0,1,16,0ZM187.73,196.29a80,80,0,0,1-113.13,0L46.31,168H75a8,8,0,0,0,0-16H32a8,8,0,0,0-8,8v48a8,8,0,0,0,16,0V179.31l28.4,28.4a96,96,0,0,0,135.77,0A8,8,0,0,0,192.86,196.4Z"/>
    </symbol>

    <!-- Trainer icon (simple person silhouette) -->
    <symbol id="trainer" viewBox="0 0 256 256">
      <path d="M230.92,212c-15.23-26.33-38.7-45.21-66.09-54.16a72,72,0,1,0-73.66,0C63.78,166.78,40.31,185.66,25.08,212a8,8,0,1,0,13.85,8c18.84-32.56,52.14-52,89.07-52s70.23,19.44,89.07,52a8,8,0,1,0,13.85-8ZM72,96a56,56,0,1,1,56,56A56.06,56.06,0,0,1,72,96Z"/>
    </symbol>

    <!-- Team Galactic Grunt sprite (male - from Pokemon Showdown) -->
    <symbol id="galactic-grunt" viewBox="0 0 80 80">
      <image href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAMAAAC5zwKfAAAAMFBMVEX/////////1b29vcWDg4v/vVLepHveUlq9e2Jq1c1KpKx7SlpKantKSloxMTEAAACC2bLVAAAAAXRSTlMAQObYZgAAAjNJREFUeNrt1Wd6KyEMheHBOsY4CM7+d3uFxr38MKPb86VOex9KYi/ffde9ME71a6QaRDq31oO8UJHHr+PRKI0S1byROnrcDNKclVRXTQwAdaDq6N2ca50GLV2/+g2XZ0R39AL2K7fb1Vo+F6nmULRY7UvPoHG7Qk4MsWbVrKmoPS93Hg+HA2e2JZfdmrErSLqXDoc6JfIE5sz1GLIOUGuaAZcTWHKhaRQZYDGQauCkWFLJ+aQJpIyjlGY9YcuNxMBcJGvNuVbOcalZZ2tonP03cc84uOWia5tCO2EQbtd8gJeZinCJAQWg7vcqPQJMEKEk1SINASJMbMnAfRDIlJoMUPeKEBBJRFIysSFGFEEbokSBkNaaedJ6kNgsmMcBtn4ma90CtgEu3f8mnfT3lAgQZho5uJrrVnDpHW6i1JJLcXEaRDfRSDOxswHmXOomUBxczbwreeOmwOfsuencFrAZ2DlrvAYlGkTvDAPJ1nsUaPXWAsHmXcRYsDMQZBDYb7wQsd95JkaBq9gaY5bQGK5gYxToYrMYBF5EBixhP+mXIYYM8DJEcCPYL2Mi/G1lCyhYwcvxCPOiA+DdMSAbQMAIPp2Y90QA8GnI06A8jYdASpz1kBKAezBZ4CSYvGcwbQKx3IXNIJ9OzoKUV886iElQ3Hsl9klQuMSBlNegbwsnQby+MA/y/dAnPAB8441J82PQxP76FSh5/BTEm92kYA589xRJjD4Ee5f3i8/R8mH0lj+iH+0PKs027VtxAAAAAElFTkSuQmCC" x="0" y="0" width="80" height="80"/>
    </symbol>

    <!-- Team Galactic Grunt sprite (female - from Pokemon Showdown) -->
    <symbol id="galactic-grunt-f" viewBox="0 0 80 80">
      <image href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAABQCAMAAAC5zwKfAAAAMFBMVEX///+9vcUAAABKSlr///9KpKyDg4sxMTFKanv/1b17SlrepHu9e2Jq1c3eUlr/vVKx01MsAAAAAXRSTlMAQObYZgAAAldJREFUeNrt1AeSrDAMhGF3S0gmzM79b/uEmZzBfnn/YivufNXE9N133/322Nob2RgcxuGazJF7zcRx5CXXlbiVHIZhHCeevO4Ut+6bpklA3nrdbqPIaSIAmclud+FtFYNMAgGMKe/ykQuvY9ocAxTzXd4tR/G6KtHEzLsT2OWuDkykWQ4ljsNffiiS/Jj07j3Ir6+vj0XPfgJzXkzeeprzwM/BnGWKMvEM/MrUz8GcJ7Dv85S1gOT9Gasqh+FT0AfNfSnmkffvFXXQ+WX9SCTp8b/9Egm5ERnQ0A99Hws/jdM4FW+i3ILCceinfhiLt0YcpsWD8YIkYVM/ruMilg9keBqHKg+miAGwkUwrE6WIBSUI0AJh8AIYUPTVoOhcEAtohRLM3NYvzxwOHk4HK75l+5yDi2ggrXBVX56832ctoM8OWbgacJqKCA+wQdRlIjxKLaLlvId5xNQkkgC8nRjRiygNwRBFxL0VGBTMXYSNvAiQiO08VYm8hecnMGKjgaLNwMVDO1BgZZ+igF4rEpEZysAWEwUlcW8DEiWT5AZYtRegWUieki9yLSi27GsFEqUZtAhgagGeBppVg3YJNhloZrPX/oytxS2mFia1OmOqqtmiSBMPGvH0PJpXegW0A6hmqu61A8/3wTQCtpPUEk/v4LwWkVeBvPqOGSpE3njlvlRMJG69ZeRmkbj3QpRFXE2Szx7jhbSVIl+9Z2brz5r6zDvdG64CTfX5L0QBBWuemFswYs07cvfoKGwdaBoZn+/TlSAtYnoBMq2K0fN7PMfULhEh06/sB3LFFI1llcKVAAAAAElFTkSuQmCC" x="0" y="0" width="80" height="80"/>
    </symbol>

    <!-- Monitor/Screen icon for Game State -->
    <symbol id="monitor" viewBox="0 0 256 256">
      <path d="M208,40H48A24,24,0,0,0,24,64V176a24,24,0,0,0,24,24H208a24,24,0,0,0,24-24V64A24,24,0,0,0,208,40Zm8,136a8,8,0,0,1-8,8H48a8,8,0,0,1-8-8V64a8,8,0,0,1,8-8H208a8,8,0,0,1,8,8Zm-48,48a8,8,0,0,1-8,8H96a8,8,0,0,1,0-16h64A8,8,0,0,1,168,224Z"/>
    </symbol>

    <!-- Pokeball icon -->
    <symbol id="pokeball" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="45" fill="#f0f0f0" stroke="#333" stroke-width="4"/>
      <path d="M 5 50 A 45 45 0 0 1 95 50" fill="#e53e3e"/>
      <rect x="5" y="47" width="90" height="6" fill="#333"/>
      <circle cx="50" cy="50" r="12" fill="#f0f0f0" stroke="#333" stroke-width="4"/>
      <circle cx="50" cy="50" r="6" fill="#333"/>
    </symbol>

    <!-- Pokemon sprites from battle_loop_diagram.svg -->
    {sprites_xml}
  </defs>

  <!-- Background -->
  <rect x="0" y="0" width="1520" height="880" fill="#FAFBFC"/>

  <!-- Title -->
  <text x="760" y="35" class="title-text text-dark" text-anchor="middle">PokeChamp Architecture</text>

  <!-- Main Battle Loop Container -->
  <g id="battle-loop-container">
    <rect x="30" y="70" width="1460" height="620" rx="24" fill="none" stroke="#8172B3" stroke-width="4" stroke-dasharray="12,6"/>
    <!-- White background to cover the dashed line behind the label -->
    <rect x="665" y="54" width="190" height="32" fill="#FAFBFC"/>
    <g transform="translate(760, 70)">
      <rect x="-90" y="-16" width="180" height="32" rx="6" fill="#8172B3"/>
      <text x="0" y="5" class="label-text" fill="white" text-anchor="middle">BATTLE LOOP</text>
    </g>
  </g>

  <!-- ==================== EXTERNAL INPUT: Historical Usage Stats (top, outside conceptually) ==================== -->
  <g id="historical-stats" transform="translate(50, 110)">
    <rect x="0" y="0" width="340" height="200" rx="16" fill="#FFFFFF" stroke="#4C72B0" stroke-width="4"/>

    <!-- Header -->
    <rect x="0" y="0" width="340" height="45" rx="16" fill="#4C72B0"/>
    <rect x="0" y="24" width="340" height="21" fill="#4C72B0"/>
    <use href="#database" x="12" y="8" width="30" height="30" fill="white"/>
    <text x="50" y="30" class="label-text" fill="white">Usage Statistics</text>

    <!-- Pokemon name header -->
    <rect x="15" y="55" width="310" height="30" rx="6" fill="#EDF2F7"/>
    <text x="25" y="76" class="body-text text-dark">Kingambit</text>
    <use href="#pool-kingambit" x="270" y="50" width="40" height="40"/>

    <!-- Stats on separate lines -->
    <text x="20" y="108" class="small-text text-light">Abilities: Defiant 78.3%</text>
    <text x="20" y="128" class="small-text text-light">Spreads: Adamant 252Atk/252HP 45%</text>
    <text x="20" y="148" class="small-text text-light">Moves: Kowtow Cleave 94%</text>
    <text x="20" y="168" class="small-text text-light">             Sucker Punch 89%</text>
  </g>

  <!-- ==================== GAME STATE (left side of main flow) ==================== -->
  <g id="game-state" transform="translate(50, 330)">
    <rect x="0" y="0" width="350" height="300" rx="16" fill="#FFFFFF" stroke="#55A868" stroke-width="4"/>

    <!-- Header -->
    <rect x="0" y="0" width="350" height="45" rx="16" fill="#55A868"/>
    <rect x="0" y="22" width="350" height="23" fill="#55A868"/>
    <use href="#monitor" x="12" y="8" width="30" height="30" fill="white"/>
    <text x="190" y="32" class="header-text" fill="white" text-anchor="middle">Game State</text>

    <!-- Pokemon Showdown Battle Window (Light Theme) -->
    <g transform="translate(6, 50)">
      <!-- Window chrome (light theme) -->
      <rect x="0" y="0" width="338" height="244" rx="8" fill="#f0f4f8" stroke="#cbd5e0" stroke-width="2"/>

      <!-- Battle arena background (gradient sky) -->
      <defs>
        <linearGradient id="sky-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" style="stop-color:#87CEEB"/>
          <stop offset="60%" style="stop-color:#98D8AA"/>
          <stop offset="100%" style="stop-color:#5D8A48"/>
        </linearGradient>
      </defs>
      <rect x="42" y="5" width="254" height="185" rx="4" fill="url(#sky-gradient)"/>

      <!-- Battle platform shadows -->
      <ellipse cx="225" cy="75" rx="45" ry="14" fill="rgba(0,0,0,0.15)"/>
      <ellipse cx="115" cy="140" rx="45" ry="14" fill="rgba(0,0,0,0.15)"/>

      <!-- Opponent Pokemon (Kingambit) - top right, smaller (far away) -->
      <g transform="translate(190, 15)">
        <use href="#pool-kingambit" x="0" y="0" width="65" height="65"/>
      </g>

      <!-- Player Pokemon (Heatran) - bottom left, larger (close) -->
      <g transform="translate(75, 85)">
        <use href="#pool-heatran" x="0" y="0" width="80" height="80"/>
      </g>

      <!-- Opponent info box (top left of arena - light theme) -->
      <g transform="translate(48, 10)">
        <rect x="0" y="0" width="110" height="38" rx="4" fill="rgba(255,255,255,0.95)" stroke="#e2e8f0" stroke-width="1"/>
        <text x="6" y="14" font-family="Arial" font-size="11" font-weight="bold" fill="#2d3748">Kingambit</text>
        <text x="6" y="24" font-family="Arial" font-size="8" fill="#718096">Lv100</text>
        <rect x="6" y="28" width="98" height="5" rx="2" fill="#e2e8f0"/>
        <rect x="6" y="28" width="80" height="5" rx="2" fill="#48bb78"/>
      </g>

      <!-- Player info box (bottom right of arena - light theme) -->
      <g transform="translate(180, 135)">
        <rect x="0" y="0" width="110" height="38" rx="4" fill="rgba(255,255,255,0.95)" stroke="#e2e8f0" stroke-width="1"/>
        <text x="6" y="14" font-family="Arial" font-size="11" font-weight="bold" fill="#2d3748">Heatran</text>
        <text x="6" y="24" font-family="Arial" font-size="8" fill="#718096">Lv100</text>
        <rect x="6" y="28" width="98" height="5" rx="2" fill="#e2e8f0"/>
        <rect x="6" y="28" width="60" height="5" rx="2" fill="#ecc94b"/>
      </g>

      <!-- Team preview sidebar - Opponent (right side) -->
      <g transform="translate(300, 5)">
        <rect x="0" y="0" width="34" height="34" rx="3" fill="#e8f4f8" stroke="#cbd5e0" stroke-width="1"/>
        <use href="#galactic-grunt-f" x="3" y="3" width="28" height="28"/>
        <rect x="2" y="38" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-dragapult" x="2" y="38" width="28" height="28"/>
        <rect x="2" y="68" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-gholdengo" x="2" y="68" width="28" height="28"/>
        <rect x="2" y="98" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-iron-valiant" x="2" y="98" width="28" height="28"/>
        <rect x="2" y="128" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-great-tusk" x="2" y="128" width="28" height="28"/>
        <rect x="2" y="158" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-landorus" x="2" y="158" width="28" height="28"/>
      </g>

      <!-- Team preview sidebar - Player (left side) -->
      <g transform="translate(4, 5)">
        <rect x="0" y="0" width="34" height="34" rx="3" fill="#e8f4f8" stroke="#cbd5e0" stroke-width="1"/>
        <use href="#galactic-grunt" x="3" y="3" width="28" height="28"/>
        <rect x="2" y="38" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-corviknight" x="2" y="38" width="28" height="28"/>
        <rect x="2" y="68" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-toxapex" x="2" y="68" width="28" height="28"/>
        <rect x="2" y="98" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-clefable" x="2" y="98" width="28" height="28"/>
        <rect x="2" y="128" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-landorus" x="2" y="128" width="28" height="28"/>
        <rect x="2" y="158" width="28" height="28" rx="2" fill="#f7fafc"/>
        <use href="#pool-great-tusk" x="2" y="158" width="28" height="28"/>
      </g>

      <!-- Move selection area (light theme) -->
      <g transform="translate(4, 195)">
        <rect x="0" y="0" width="330" height="44" rx="4" fill="#edf2f7" stroke="#e2e8f0" stroke-width="1"/>
        <text x="10" y="15" font-family="Arial" font-size="10" font-weight="bold" fill="#4a5568">What will Heatran do?</text>
        <rect x="10" y="20" width="75" height="18" rx="3" fill="#e53e3e"/>
        <text x="47" y="33" font-family="Arial" font-size="9" fill="white" text-anchor="middle">Magma Storm</text>
        <rect x="90" y="20" width="75" height="18" rx="3" fill="#805ad5"/>
        <text x="127" y="33" font-family="Arial" font-size="9" fill="white" text-anchor="middle">Earth Power</text>
        <rect x="170" y="20" width="75" height="18" rx="3" fill="#b7791f"/>
        <text x="207" y="33" font-family="Arial" font-size="9" fill="white" text-anchor="middle">Stealth Rock</text>
        <rect x="250" y="20" width="75" height="18" rx="3" fill="#744210"/>
        <text x="287" y="33" font-family="Arial" font-size="9" fill="white" text-anchor="middle">Protect</text>
      </g>
    </g>
  </g>

  <!-- Arrow from Game State to Text Conversion -->
  <path d="M 400,480 L 470,350" stroke="#55A868" stroke-width="3" fill="none" marker-end="url(#arrowhead-green)"/>

  <!-- Arrow from Usage Stats to Text Conversion -->
  <path d="M 390,210 L 470,210" stroke="#4C72B0" stroke-width="3" fill="none" marker-end="url(#arrowhead-blue)"/>

  <!-- ==================== CENTRAL: Text Conversion / LLM Box ==================== -->
  <g id="text-conversion" transform="translate(480, 100)">
    <rect x="0" y="0" width="460" height="500" rx="20" fill="#FFFFFF" stroke="#8172B3" stroke-width="4"/>

    <!-- Header -->
    <rect x="0" y="0" width="460" height="50" rx="20" fill="#8172B3"/>
    <rect x="0" y="26" width="460" height="24" fill="#8172B3"/>
    <use href="#brain" x="12" y="8" width="34" height="34" fill="white"/>
    <text x="250" y="35" class="header-text" fill="white" text-anchor="middle">Scaffolding + LLM</text>

    <!-- Step 1: Game State to Text -->
    <g transform="translate(15, 60)">
      <rect x="0" y="0" width="430" height="75" rx="10" fill="#F7FAFC" stroke="#E2E8F0" stroke-width="2"/>
      <text x="12" y="22" class="body-text text-dark" font-weight="600">1. Convert game state to text</text>
      <text x="20" y="44" class="mono-text text-light">"Opp: Kingambit vs You: Heatran"</text>
      <text x="20" y="62" class="mono-text text-light">"HP: 82% vs 62%, Turn 5"</text>
    </g>

    <!-- Arrow down -->
    <path d="M 230,140 L 230,155" stroke="#718096" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>

    <!-- Step 2: Add context -->
    <g transform="translate(15, 160)">
      <rect x="0" y="0" width="430" height="75" rx="10" fill="#F7FAFC" stroke="#E2E8F0" stroke-width="2"/>
      <text x="12" y="22" class="body-text text-dark" font-weight="600">2. Enrich with usage statistics</text>
      <text x="20" y="44" class="mono-text text-light">"Kingambit: Defiant 78%, Sucker Punch 89%"</text>
      <text x="20" y="62" class="mono-text text-light">"Likely: Kowtow Cleave or Sucker Punch"</text>
    </g>

    <!-- Arrow down -->
    <path d="M 230,240 L 230,255" stroke="#718096" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>

    <!-- Step 3: LLM Decision (two options) -->
    <g transform="translate(15, 260)">
      <rect x="0" y="0" width="430" height="220" rx="10" fill="#EBF4FF" stroke="#4C72B0" stroke-width="3"/>
      <use href="#brain" x="10" y="8" width="28" height="28" fill="#4C72B0"/>
      <text x="44" y="28" class="body-text agent1" font-weight="600">3. LLM Decision</text>

      <!-- Option A: Direct reasoning -->
      <g transform="translate(12, 45)">
        <rect x="0" y="0" width="198" height="160" rx="8" fill="#F7FAFC" stroke="#E2E8F0" stroke-width="2"/>
        <text x="10" y="22" class="small-text text-dark" font-weight="600">Option A: Direct</text>
        <text x="10" y="46" class="small-text text-light">Chain-of-thought</text>
        <text x="10" y="66" class="small-text text-light">reasoning to select</text>
        <text x="10" y="86" class="small-text text-light">optimal action</text>
      </g>

      <!-- Option B: Minimax scaffolding -->
      <g transform="translate(218, 45)">
        <rect x="0" y="0" width="198" height="160" rx="8" fill="#FFF8E7" stroke="#DD8452" stroke-width="2" stroke-dasharray="5,3"/>
        <text x="10" y="22" class="small-text" fill="#DD8452" font-weight="600">Option B: Minimax</text>
        <text x="10" y="46" class="small-text text-light">Lookahead with LLM</text>
        <text x="10" y="66" class="small-text text-light">rubric evaluation &amp;</text>
        <text x="10" y="86" class="small-text text-light">value cutoff pruning</text>
      </g>
    </g>
  </g>

  <!-- ==================== OUTPUT: Action ==================== -->
  <g id="action-output" transform="translate(980, 150)">
    <rect x="0" y="0" width="280" height="140" rx="16" fill="#FFFFFF" stroke="#DD8452" stroke-width="4"/>

    <!-- Header with grunt icon -->
    <rect x="0" y="0" width="280" height="45" rx="16" fill="#DD8452"/>
    <rect x="0" y="24" width="280" height="21" fill="#DD8452"/>
    <use href="#galactic-grunt" x="8" y="5" width="35" height="35"/>
    <text x="150" y="30" class="label-text" fill="white" text-anchor="middle">Action Output</text>

    <!-- Action content -->
    <text x="15" y="70" class="body-text text-dark" font-weight="600">Move: Magma Storm</text>
    <text x="15" y="95" class="small-text text-light">Traps Kingambit, deals chip</text>
    <text x="15" y="115" class="small-text text-light">damage, avoids Sucker Punch</text>
  </g>

  <!-- Arrow from Text Conversion to Action -->
  <path d="M 940,350 L 980,270" stroke="#DD8452" stroke-width="3" fill="none" marker-end="url(#arrowhead-orange)"/>

  <!-- ==================== LOOP FEEDBACK ==================== -->

  <!-- Opponent Action box -->
  <g id="opponent-action" transform="translate(1280, 150)">
    <rect x="0" y="0" width="180" height="140" rx="12" fill="#FFFFFF" stroke="#C44E52" stroke-width="3"/>
    <rect x="0" y="0" width="180" height="45" rx="12" fill="#C44E52"/>
    <rect x="0" y="24" width="180" height="21" fill="#C44E52"/>
    <use href="#galactic-grunt-f" x="8" y="5" width="35" height="35"/>
    <text x="105" y="30" class="label-text" fill="white" text-anchor="middle">Opponent</text>
    <text x="15" y="75" class="small-text text-light">Simultaneous</text>
    <text x="15" y="95" class="small-text text-light">action selection</text>
  </g>

  <!-- Arrow from Action Output down to Battle State Update -->
  <path d="M 1120,290 L 1120,510" stroke="#DD8452" stroke-width="3" fill="none" marker-end="url(#arrowhead-orange)"/>

  <!-- Arrow from Opponent Action down to Battle State Update -->
  <path d="M 1370,290 L 1370,510" stroke="#C44E52" stroke-width="3" fill="none" marker-end="url(#arrowhead-red)"/>

  <!-- Battle State Update box (bottom center-right) -->
  <g id="battle-update" transform="translate(1020, 520)">
    <rect x="0" y="0" width="440" height="65" rx="12" fill="#FFFFFF" stroke="#8172B3" stroke-width="3"/>
    <use href="#pokeball" x="15" y="15" width="35" height="35"/>
    <text x="220" y="40" class="body-text engine" text-anchor="middle" font-weight="600">Battle State Update</text>
  </g>

  <!-- Arrow from Battle State Update back to Game State (loop feedback) - goes down, left along bottom, then up into Game State -->
  <path d="M 1240,585 L 1240,660 L 225,660 L 225,630" stroke="#8172B3" stroke-width="3" fill="none" marker-end="url(#arrowhead-purple)"/>

</svg>
'''

    return svg_content

def main():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(script_dir, "pokechamp_architecture.svg")

    print("Creating PokeChamp architecture figure...")
    svg_content = create_pokechamp_svg()

    with open(output_path, 'w') as f:
        f.write(svg_content)

    print(f"Created: {output_path}")

    # Convert to PDF and PNG
    try:
        import cairosvg
        from PIL import Image

        pdf_path = os.path.join(script_dir, "pokechamp_architecture.pdf")
        png_path = os.path.join(script_dir, "pokechamp_architecture.png")

        print("Converting to PDF...")
        cairosvg.svg2pdf(url=output_path, write_to=pdf_path)
        print(f"Created: {pdf_path}")

        print("Converting to PNG (2x scale)...")
        cairosvg.svg2png(url=output_path, write_to=png_path, scale=2.0)
        print(f"Created: {png_path}")

        img = Image.open(png_path)
        print(f"PNG dimensions: {img.size[0]} x {img.size[1]} px")

    except ImportError:
        print("Note: Install cairosvg and pillow to auto-convert to PDF/PNG")
        print("  pip install cairosvg pillow")

if __name__ == "__main__":
    main()
